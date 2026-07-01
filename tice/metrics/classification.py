"""Classification metrics robust to degenerate (class-dropping) folds.

Under aggressive shift a class can vanish from ``y_true`` or from the model's
training data. ``log_loss`` / ``roc_auc_score`` need the full class set, so every
metric is computed against the *clean* class list passed in as ``classes`` and
each metric is wrapped: a metric that cannot be defined becomes ``nan`` and is
recorded in ``notes`` rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

_EPS = 1e-12


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    auc: float
    nll: float
    nll_norm: float
    ece: float
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "auc": self.auc,
            "nll": self.nll,
            "nll_norm": self.nll_norm,
            "ece": self.ece,
            "metric_notes": self.notes,
        }


def _normalize_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.clip(np.asarray(proba, dtype=float), _EPS, 1.0)
    row_sums = proba.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return proba / row_sums


def expected_calibration_error(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Top-label expected calibration error with equal-width confidence bins."""
    proba = _normalize_proba(proba)
    conf = proba.max(axis=1)
    pred = classes[proba.argmax(axis=1)]
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # bin index in [0, n_bins-1]
    idx = np.clip(np.digitize(conf, bins[1:-1], right=False), 0, n_bins - 1)
    n = conf.size
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        bin_conf = conf[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def _safe_auc(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    present = np.unique(y_true)
    if present.size < 2:
        raise ValueError("AUC undefined: fewer than two classes present in y_true")
    if classes.size == 2:
        pos_col = int(np.where(classes == classes[1])[0][0])
        return float(roc_auc_score((y_true == classes[1]).astype(int), proba[:, pos_col]))
    return float(
        roc_auc_score(
            y_true,
            proba,
            multi_class="ovr",
            average="macro",
            labels=classes,
        )
    )


def compute_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    *,
    n_bins: int = 15,
) -> ClassificationMetrics:
    """Compute all metrics; undefined ones become ``nan`` and are noted."""
    y_true = np.asarray(y_true)
    classes = np.asarray(classes)
    proba = _normalize_proba(proba)
    k = classes.size
    pred = classes[proba.argmax(axis=1)]
    notes: list[str] = []

    try:
        accuracy = float(accuracy_score(y_true, pred))
    except Exception as exc:  # pragma: no cover - accuracy is near-always defined
        accuracy = float("nan")
        notes.append(f"accuracy:{exc}")

    try:
        auc = _safe_auc(y_true, proba, classes)
    except Exception as exc:
        auc = float("nan")
        notes.append(f"auc:{exc}")

    try:
        nll = float(log_loss(y_true, proba, labels=classes))
        nll_norm = float(nll / np.log(k)) if k > 1 else float("nan")
    except Exception as exc:
        nll = float("nan")
        nll_norm = float("nan")
        notes.append(f"nll:{exc}")

    try:
        ece = expected_calibration_error(y_true, proba, classes, n_bins=n_bins)
    except Exception as exc:  # pragma: no cover
        ece = float("nan")
        notes.append(f"ece:{exc}")

    return ClassificationMetrics(
        accuracy=accuracy,
        auc=auc,
        nll=nll,
        nll_norm=nll_norm,
        ece=ece,
        notes="; ".join(notes),
    )
