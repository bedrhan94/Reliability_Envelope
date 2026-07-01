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


def _build_xgboost(seed: int):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        tree_method="hist",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
    )


def _build_catboost(seed: int):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=200,
        depth=6,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


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
