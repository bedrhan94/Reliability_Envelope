"""Model registry.

Each model is a sklearn-compatible estimator factory tagged with a *family*
(``linear`` / ``gbdt`` / ``icl``). Heavy backends (xgboost, catboost, tabpfn,
tabicl) are imported lazily inside their builders so an absent package surfaces
as ``status="skipped"`` in the results table instead of crashing the run.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass

from sklearn.base import BaseEstimator, ClassifierMixin

MODEL_FAMILIES = ("linear", "gbdt", "icl")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    builder: Callable[[int], object]  # seed -> unfitted estimator
    required_package: str | None = None
    # ICL backends are memory-bound; cap the train context handed to them.
    is_context_bound: bool = False

    def available(self) -> bool:
        if self.required_package is None:
            return True
        return importlib.util.find_spec(self.required_package) is not None


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _build_logreg(seed: int):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=2000, random_state=seed)


def _build_hist_gbdt(seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(random_state=seed)


def _gbdt_device_kwargs(lib: str) -> dict:
    """GPU kwargs for a GBDT library when ``TICE_GBDT_DEVICE=cuda`` is set, else {} (CPU).

    Added to run the heavy tuned/ensemble strong-baseline arm on the GPU without changing
    the default CPU behaviour, so existing results and tests are untouched. Only XGBoost
    and CatBoost are moved -- LightGBM's pip wheel is CPU-only. GridSearchCV uses n_jobs=1,
    so CV fits hit the single GPU sequentially with no contention.
    """
    import os

    if os.environ.get("TICE_GBDT_DEVICE", "").lower() != "cuda":
        return {}
    if lib == "xgboost":
        return {"device": "cuda"}
    if lib == "catboost":
        return {"task_type": "GPU", "devices": "0"}
    return {}


def _build_xgboost(seed: int):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        tree_method="hist",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
        **_gbdt_device_kwargs("xgboost"),
    )


def _build_catboost(seed: int):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=200,
        depth=6,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        **_gbdt_device_kwargs("catboost"),
    )


def _tuned(estimator, grid: dict, seed: int):
    """Wrap an estimator in a small log-loss-scored CV search.

    Used by the ``*_tuned`` baselines: the published GBDT baselines run at fixed
    hyper-parameters, which a reviewer will read as an unfairly weak comparison
    against tuning-free ICL models. Scoring on ``neg_log_loss`` (not accuracy)
    tunes for exactly the calibration term the reliability utility penalises, so
    these are the *strongest* honest GBDT baselines. ``GridSearchCV`` is
    sklearn-compatible (``predict_proba`` / ``classes_`` delegate to the refit
    best estimator), so the pipeline needs no change.
    """
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    return GridSearchCV(
        estimator, grid, scoring="neg_log_loss", cv=cv, refit=True, n_jobs=1,
        error_score="raise",
    )


def _build_xgboost_tuned(seed: int):
    from xgboost import XGBClassifier

    base = XGBClassifier(
        tree_method="hist", eval_metric="logloss", random_state=seed, n_jobs=1,
        **_gbdt_device_kwargs("xgboost"),
    )
    grid = {"n_estimators": [200, 600], "max_depth": [3, 6], "learning_rate": [0.05, 0.2]}
    return _tuned(base, grid, seed)


def _build_catboost_tuned(seed: int):
    from catboost import CatBoostClassifier

    base = CatBoostClassifier(
        random_seed=seed, verbose=False, allow_writing_files=False,
        **_gbdt_device_kwargs("catboost"),
    )
    grid = {"iterations": [200, 600], "depth": [4, 6], "learning_rate": [0.03, 0.1]}
    return _tuned(base, grid, seed)


def _build_lightgbm(seed: int):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=200, max_depth=6, random_state=seed, n_jobs=1, verbose=-1
    )


def _build_lightgbm_tuned(seed: int):
    from lightgbm import LGBMClassifier

    base = LGBMClassifier(random_state=seed, n_jobs=1, verbose=-1)
    grid = {"n_estimators": [200, 600], "max_depth": [3, 6], "learning_rate": [0.05, 0.2]}
    return _tuned(base, grid, seed)


class _SeedEnsemble(BaseEstimator, ClassifierMixin):
    """Average the predicted probabilities of one estimator refit at several seeds.

    The tabular analogue of the deep ensembles that Ovadia et al. (2019) found to be
    the most shift-robust family they tested -- the one strong baseline our model set
    was missing. Averaging probabilities (not votes) is what makes it a calibration
    intervention rather than only an accuracy one.

    Inherits ``BaseEstimator``/``ClassifierMixin`` rather than duck-typing the API:
    recent scikit-learn resolves estimator tags through the MRO, and a plain class
    fails the pipeline's preflight check with a ``__sklearn_tags__`` error. Parameters
    are stored under their ``__init__`` names so ``get_params`` works.
    """

    def __init__(self, factory=None, seeds=()):
        self.factory = factory
        self.seeds = seeds

    def fit(self, X, y):
        import numpy as np

        self.members_ = []
        for s in self.seeds:
            member = self.factory(s)
            member.fit(X, y)
            self.members_.append(member)
        self.classes_ = np.asarray(self.members_[0].classes_)
        return self

    def predict_proba(self, X):
        import numpy as np

        # Members share the training labels, so their class order agrees; that is
        # checked rather than assumed, because a silent column mismatch would average
        # different classes together and quietly corrupt every calibration metric.
        probas = []
        for m in self.members_:
            if not np.array_equal(np.asarray(m.classes_), self.classes_):
                raise RuntimeError("ensemble members disagree on class order")
            probas.append(np.asarray(m.predict_proba(X), dtype=float))
        return np.mean(probas, axis=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def _build_gbdt_ensemble(seed: int):
    """Five CatBoost fits at different seeds, probabilities averaged."""
    return _SeedEnsemble(_build_catboost, [seed + i * 1000 for i in range(5)])


def _calibrated(estimator, seed: int):
    """Wrap an estimator in post-hoc Platt scaling fitted inside the training split.

    Used by the ``*_cal`` baselines. The reference-confound ablation shows the ICL
    models' envelope advantage tracks their *clean* calibration, which invites the
    obvious rebuttal: just calibrate the baselines. These answer it directly.

    ``CalibratedClassifierCV`` fits the calibrator on held-out folds of the training
    data, so no test information leaks -- important here, because the shift axes
    perturb train (label_noise) and test (covariate_shift) differently and a
    test-fitted calibrator would silently undo the covariate shift. Sigmoid rather
    than isotonic: several of these datasets are multiclass with few hundred rows per
    fold, where isotonic overfits.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    return CalibratedClassifierCV(estimator, method="sigmoid", cv=cv)


def _build_logreg_cal(seed: int):
    return _calibrated(_build_logreg(seed), seed)


def _build_hist_gbdt_cal(seed: int):
    return _calibrated(_build_hist_gbdt(seed), seed)


def _build_xgboost_cal(seed: int):
    return _calibrated(_build_xgboost(seed), seed)


def _build_catboost_cal(seed: int):
    return _calibrated(_build_catboost(seed), seed)


def _torch_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _supported_kwargs(cls: type, **kwargs: object) -> dict:
    """Keep only kwargs the estimator's ``__init__`` actually accepts.

    TabPFN and TabICL constructors differ across versions (``random_state`` vs
    ``seed``, presence of ``device`` / ``n_jobs``); filtering avoids hard
    failures from an unexpected keyword on a particular installed version.
    """
    import inspect

    try:
        params = inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _build_tabpfn(seed: int):
    import os

    # TabPFN v8 (PriorLabs) gates weights behind a license/API key. Never open a
    # browser in our headless pipeline: with TABPFN_NO_BROWSER set it raises a
    # clean error (caught by the runner -> "skipped") unless TABPFN_TOKEN is
    # provided, in which case it authenticates non-interactively.
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    from tabpfn import TabPFNClassifier

    kwargs = _supported_kwargs(
        TabPFNClassifier, random_state=seed, seed=seed, device=_torch_device()
    )
    return TabPFNClassifier(**kwargs)


def _build_tabpfn_client(seed: int):
    """TabPFN via the PriorLabs cloud API (inference runs server-side).

    Reads the access token from ``TABPFN_TOKEN`` (or ``TABPFN_CLIENT_TOKEN``) so
    the secret never lives in source. Avoids the gated HuggingFace weight
    download that the local ``tabpfn`` package needs.
    """
    import os

    import tabpfn_client

    token = os.environ.get("TABPFN_TOKEN") or os.environ.get("TABPFN_CLIENT_TOKEN")
    if token:
        tabpfn_client.set_access_token(token.strip())
    from tabpfn_client import TabPFNClassifier

    kwargs = _supported_kwargs(TabPFNClassifier, random_state=seed, seed=seed)
    return TabPFNClassifier(**kwargs)


def _build_tabicl(seed: int):
    from tabicl import TabICLClassifier

    kwargs = _supported_kwargs(
        TabICLClassifier, random_state=seed, seed=seed, device=_torch_device()
    )
    return TabICLClassifier(**kwargs)


_REGISTRY: dict[str, ModelSpec] = {
    "logreg": ModelSpec("logreg", "linear", _build_logreg),
    "hist_gbdt": ModelSpec("hist_gbdt", "gbdt", _build_hist_gbdt),
    "xgboost": ModelSpec("xgboost", "gbdt", _build_xgboost, required_package="xgboost"),
    "catboost": ModelSpec(
        "catboost", "gbdt", _build_catboost, required_package="catboost"
    ),
    "xgboost_tuned": ModelSpec(
        "xgboost_tuned", "gbdt", _build_xgboost_tuned, required_package="xgboost"
    ),
    "catboost_tuned": ModelSpec(
        "catboost_tuned", "gbdt", _build_catboost_tuned, required_package="catboost"
    ),
    "lightgbm": ModelSpec(
        "lightgbm", "gbdt", _build_lightgbm, required_package="lightgbm"
    ),
    "lightgbm_tuned": ModelSpec(
        "lightgbm_tuned", "gbdt", _build_lightgbm_tuned, required_package="lightgbm"
    ),
    "gbdt_ensemble": ModelSpec(
        "gbdt_ensemble", "gbdt", _build_gbdt_ensemble, required_package="catboost"
    ),
    "logreg_cal": ModelSpec("logreg_cal", "linear", _build_logreg_cal),
    "hist_gbdt_cal": ModelSpec("hist_gbdt_cal", "gbdt", _build_hist_gbdt_cal),
    "xgboost_cal": ModelSpec(
        "xgboost_cal", "gbdt", _build_xgboost_cal, required_package="xgboost"
    ),
    "catboost_cal": ModelSpec(
        "catboost_cal", "gbdt", _build_catboost_cal, required_package="catboost"
    ),
    "tabpfn": ModelSpec(
        "tabpfn", "icl", _build_tabpfn, required_package="tabpfn", is_context_bound=True
    ),
    "tabpfn_client": ModelSpec(
        "tabpfn_client",
        "icl",
        _build_tabpfn_client,
        required_package="tabpfn_client",
        is_context_bound=True,
    ),
    "tabicl": ModelSpec(
        "tabicl", "icl", _build_tabicl, required_package="tabicl", is_context_bound=True
    ),
}


def available_models() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def get_model_spec(name: str) -> ModelSpec:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def is_model_available(name: str) -> bool:
    return get_model_spec(name).available()
