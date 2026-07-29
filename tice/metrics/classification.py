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
    # Additional calibration diagnostics. ECE alone is sensitive to bin count, sample
    # size and class balance, and the paper's central argument is about calibration,
    # so the claim is carried by a proper scoring rule and several ECE variants too.
    brier: float = float("nan")
    ece_adaptive: float = float("nan")
    ece_classwise: float = float("nan")
    calib_slope: float = float("nan")
    calib_intercept: float = float("nan")

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "auc": self.auc,
            "nll": self.nll,
            "nll_norm": self.nll_norm,
            "ece": self.ece,
            "brier": self.brier,
            "ece_adaptive": self.ece_adaptive,
            "ece_classwise": self.ece_classwise,
            "calib_slope": self.calib_slope,
            "calib_intercept": self.calib_intercept,
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


def adaptive_calibration_error(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Top-label ECE with equal-*mass* bins instead of equal-width ones.

    Equal-width binning leaves most bins nearly empty when confidences pile up near
    1, which is the usual case here, so the reported ECE is dominated by a handful
    of points. Equal-mass bins put the same number of samples in each bin and are
    the standard robustness check on that sensitivity.
    """
    proba = _normalize_proba(proba)
    conf = proba.max(axis=1)
    correct = (classes[proba.argmax(axis=1)] == y_true).astype(float)
    n = conf.size
    if n == 0:
        return float("nan")
    order = np.argsort(conf)
    ace = 0.0
    for chunk in np.array_split(order, min(n_bins, n)):
        if chunk.size == 0:
            continue
        ace += (chunk.size / n) * abs(correct[chunk].mean() - conf[chunk].mean())
    return float(ace)


def classwise_calibration_error(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Mean over classes of the one-vs-rest calibration error.

    Top-label ECE ignores the calibration of the non-predicted classes entirely,
    which for a multiclass problem is most of the predictive distribution.
    """
    proba = _normalize_proba(proba)
    y_true = np.asarray(y_true)
    n = y_true.size
    if n == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    errors = []
    for j, cls in enumerate(classes):
        p = proba[:, j]
        target = (y_true == cls).astype(float)
        idx = np.clip(np.digitize(p, bins[1:-1], right=False), 0, n_bins - 1)
        err = 0.0
        for b in range(n_bins):
            mask = idx == b
            if mask.any():
                err += (mask.sum() / n) * abs(target[mask].mean() - p[mask].mean())
        errors.append(err)
    return float(np.mean(errors)) if errors else float("nan")


def brier_score(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error against the one-hot target.

    A strictly proper scoring rule, so unlike ECE it cannot be gamed by a model that
    reports uninformative but well-calibrated probabilities.
    """
    proba = _normalize_proba(proba)
    onehot = (np.asarray(y_true)[:, None] == np.asarray(classes)[None, :]).astype(float)
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def calibration_slope_intercept(
    y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray
) -> tuple[float, float]:
    """Logistic recalibration slope and intercept on the top-label logit.

    Fits ``correct ~ sigmoid(a + b * logit(confidence))``. A perfectly calibrated
    model gives slope 1 and intercept 0; slope < 1 indicates over-confidence, which
    is the direction boosted trees are known to fail in. Returns ``(nan, nan)`` when
    the outcome is single-valued, where the fit is undefined.
    """
    proba = _normalize_proba(proba)
    conf = np.clip(proba.max(axis=1), _EPS, 1 - _EPS)
    correct = (classes[proba.argmax(axis=1)] == np.asarray(y_true)).astype(int)
    if np.unique(correct).size < 2:
        return float("nan"), float("nan")
    from sklearn.linear_model import LogisticRegression

    logit = np.log(conf / (1 - conf)).reshape(-1, 1)
    fit = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logit, correct)
    return float(fit.coef_[0][0]), float(fit.intercept_[0])


def reliability_bins(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    *,
    n_bins: int = 15,
) -> list[dict]:
    """Per-bin (count, mean confidence, empirical accuracy) for reliability diagrams.

    Stored rather than the raw probabilities: it is what the diagram needs, and it is
    three numbers per bin instead of one row per test point.
    """
    proba = _normalize_proba(proba)
    conf = proba.max(axis=1)
    correct = (classes[proba.argmax(axis=1)] == np.asarray(y_true)).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins[1:-1], right=False), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = idx == b
        out.append({
            "bin": b,
            "lo": float(bins[b]),
            "hi": float(bins[b + 1]),
            "n": int(mask.sum()),
            "mean_conf": float(conf[mask].mean()) if mask.any() else float("nan"),
            "accuracy": float(correct[mask].mean()) if mask.any() else float("nan"),
        })
    return out


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

    extra: dict[str, float] = {}
    for name, fn in (
        ("brier", lambda: brier_score(y_true, proba, classes)),
        ("ece_adaptive", lambda: adaptive_calibration_error(y_true, proba, classes,
                                                            n_bins=n_bins)),
        ("ece_classwise", lambda: classwise_calibration_error(y_true, proba, classes,
                                                              n_bins=n_bins)),
    ):
        try:
            extra[name] = fn()
        except Exception as exc:
            extra[name] = float("nan")
            notes.append(f"{name}:{exc}")
    try:
        extra["calib_slope"], extra["calib_intercept"] = calibration_slope_intercept(
            y_true, proba, classes
        )
    except Exception as exc:
        extra["calib_slope"] = extra["calib_intercept"] = float("nan")
        notes.append(f"calib_slope:{exc}")

    return ClassificationMetrics(
        accuracy=accuracy,
        auc=auc,
        nll=nll,
        nll_norm=nll_norm,
        ece=ece,
        **extra,
        notes="; ".join(notes),
    )
