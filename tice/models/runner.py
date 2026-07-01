"""Model runner.

Fits one model on a (possibly shifted) split and evaluates it. Any failure --
a missing optional package, an estimator that refuses a degenerate fold, an
undefined metric -- is captured into ``status`` / ``error_message`` so the
results table always gets a row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.pipeline import Pipeline

from tice.datasets.registry import Split
from tice.metrics.classification import compute_metrics
from tice.metrics.utility import reliability_utility
from tice.models.registry import get_model_spec
from tice.preprocess import build_preprocessor
from tice.seed import make_rng


@dataclass(frozen=True)
class RunResult:
    model: str
    family: str
    status: str  # "ok" | "skipped" | "error"
    error_message: str = ""
    accuracy: float = float("nan")
    auc: float = float("nan")
    nll: float = float("nan")
    nll_norm: float = float("nan")
    ece: float = float("nan")
    utility: float = float("nan")
    n_train: int = 0
    metric_notes: str = ""

    @property
    def status_ok(self) -> bool:
        return self.status == "ok"

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "family": self.family,
            "status": self.status,
            "error_message": self.error_message,
            "accuracy": self.accuracy,
            "auc": self.auc,
            "nll": self.nll,
            "nll_norm": self.nll_norm,
            "ece": self.ece,
            "utility": self.utility,
            "n_train": self.n_train,
            "metric_notes": self.metric_notes,
        }


def _align_proba(
    proba: np.ndarray, model_classes: np.ndarray, target_classes: np.ndarray
) -> np.ndarray:
    """Re-order/pad proba columns to the clean class set (missing -> ~0)."""
    out = np.full((proba.shape[0], target_classes.size), 1e-12, dtype=float)
    index = {c: i for i, c in enumerate(target_classes.tolist())}
    for j, c in enumerate(model_classes.tolist()):
        if c in index:
            out[:, index[c]] = proba[:, j]
    return out


def _cap_context(
    split: Split, max_rows: int, seed: int
) -> tuple[Split, int]:
    """Stratified down-sample of the train context for memory-bound ICL models."""
    n = len(split.X_train)
    if n <= max_rows:
        return split, n
    rng = make_rng(seed, "context_cap")
    y = split.y_train.to_numpy()
    keep: list[int] = []
    frac = max_rows / n
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        k = max(1, int(round(frac * cls_idx.size)))
        keep.extend(rng.choice(cls_idx, size=min(k, cls_idx.size), replace=False).tolist())
    keep = sorted(keep)
    capped = Split(
        dataset_id=split.dataset_id,
        X_train=split.X_train.iloc[keep].reset_index(drop=True),
        y_train=split.y_train.iloc[keep].reset_index(drop=True),
        X_test=split.X_test,
        y_test=split.y_test,
        categorical_columns=split.categorical_columns,
        classes=split.classes,
    )
    return capped, len(keep)


def run_model(
    model_name: str,
    split: Split,
    *,
    seed: int,
    ece_bins: int = 15,
    max_context_rows: int = 1000,
) -> RunResult:
    """Train and evaluate ``model_name`` on ``split``; never raises."""
    spec = get_model_spec(model_name)
    target_classes = np.asarray(split.classes)

    if not spec.available():
        return RunResult(
            model=model_name,
            family=spec.family,
            status="skipped",
            error_message=f"optional package '{spec.required_package}' not installed",
        )

    work = split
    n_train = len(split.X_train)
    if spec.is_context_bound:
        work, n_train = _cap_context(split, max_context_rows, seed)

    try:
        estimator = spec.builder(seed)
        pre = build_preprocessor(work.numeric_columns, work.categorical_columns)
        pipe = Pipeline([("pre", pre), ("clf", estimator)])
        pipe.fit(work.X_train, work.y_train.to_numpy())
        proba = np.asarray(pipe.predict_proba(work.X_test))
        model_classes = np.asarray(pipe.classes_)
        proba = _align_proba(proba, model_classes, target_classes)
    except Exception as exc:
        return RunResult(
            model=model_name,
            family=spec.family,
            status="error",
            error_message=f"{type(exc).__name__}: {exc}",
            n_train=n_train,
        )

    metrics = compute_metrics(
        work.y_test.to_numpy(), proba, target_classes, n_bins=ece_bins
    )
    utility = reliability_utility(
        metrics.auc, metrics.accuracy, metrics.nll_norm, metrics.ece
    )

    return RunResult(
        model=model_name,
        family=spec.family,
        status="ok",
        accuracy=metrics.accuracy,
        auc=metrics.auc,
        nll=metrics.nll,
        nll_norm=metrics.nll_norm,
        ece=metrics.ece,
        utility=utility,
        n_train=n_train,
        metric_notes=metrics.notes,
    )
