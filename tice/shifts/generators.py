"""Controlled shift generators (item 2 of the spec).

Six shift axes, each deterministic and seed-fixed. Every generator:

* copies the clean split and never mutates it in place;
* treats ``lambda == 0.0`` as an exact no-op (returns an unchanged copy) -- the
  envelope math depends on this invariant;
* derives all randomness from the central seed via :func:`tice.seed.make_rng`.

Real implementations: ``label_noise``, ``class_imbalance``,
``missingness_shift``, ``covariate_shift``, ``context_budget``.
Conditionally real: ``rare_category_shift`` -- only applies to datasets that
expose categorical columns; otherwise it reports itself as not applicable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tice.datasets.registry import Split
from tice.seed import make_rng

RARE_TOKEN = "__RARE__"
COVARIATE_SHIFT_STRENGTH = 3.0  # std multiplier at lambda=1.0


def _copy_split(split: Split) -> Split:
    return Split(
        dataset_id=split.dataset_id,
        X_train=split.X_train.copy(deep=True),
        y_train=split.y_train.copy(deep=True),
        X_test=split.X_test.copy(deep=True),
        y_test=split.y_test.copy(deep=True),
        categorical_columns=split.categorical_columns,
        classes=split.classes,
    )


# --------------------------------------------------------------------------- #
# Individual axes
# --------------------------------------------------------------------------- #
def _label_noise(split: Split, lam: float, base_seed: int) -> Split:
    out = _copy_split(split)
    classes = np.asarray(split.classes)
    if classes.size < 2:
        return out
    rng = make_rng(base_seed, split.dataset_id, "label_noise", lam)
    y = out.y_train.to_numpy().copy()
    n = y.size
    k = int(round(lam * n))
    if k <= 0:
        return out
    idx = rng.choice(n, size=k, replace=False)
    for i in idx:
        choices = classes[classes != y[i]]
        y[i] = choices[rng.integers(choices.size)]
    out.y_train.iloc[:] = y
    return out


def _class_imbalance(split: Split, lam: float, base_seed: int) -> Split:
    """Amplify imbalance by downsampling every non-majority train class by lam."""
    out = _copy_split(split)
    rng = make_rng(base_seed, split.dataset_id, "class_imbalance", lam)
    y = out.y_train.to_numpy()
    counts = pd.Series(y).value_counts()
    if counts.size < 2:
        return out
    majority = counts.idxmax()

    keep_mask = np.ones(y.size, dtype=bool)
    for cls in counts.index:
        if cls == majority:
            continue
        cls_idx = np.flatnonzero(y == cls)
        n_drop = int(round(lam * cls_idx.size))
        # keep at least one sample of the class
        n_drop = min(n_drop, cls_idx.size - 1)
        if n_drop <= 0:
            continue
        drop_idx = rng.choice(cls_idx, size=n_drop, replace=False)
        keep_mask[drop_idx] = False

    out.X_train = out.X_train.loc[keep_mask].reset_index(drop=True)
    out.y_train = out.y_train.loc[keep_mask].reset_index(drop=True)
    return out


def _missingness_shift(split: Split, lam: float, base_seed: int) -> Split:
    """Inject MCAR missingness into the TEST features at rate lam (train stays clean)."""
    out = _copy_split(split)
    if lam <= 0:
        return out
    rng = make_rng(base_seed, split.dataset_id, "missingness_shift", lam)
    X = out.X_test
    mask = rng.random(X.shape) < lam
    for j, col in enumerate(X.columns):
        col_mask = mask[:, j]
        if col_mask.any():
            X.loc[col_mask, col] = np.nan
    out.X_test = X
    return out


def _rare_category_shift(split: Split, lam: float, base_seed: int) -> Split:
    """Replace a fraction lam of TEST categorical values with an unseen token."""
    out = _copy_split(split)
    if lam <= 0 or not split.categorical_columns:
        return out
    rng = make_rng(base_seed, split.dataset_id, "rare_category_shift", lam)
    X = out.X_test
    n = len(X)
    for col in split.categorical_columns:
        k = int(round(lam * n))
        if k <= 0:
            continue
        idx = rng.choice(n, size=k, replace=False)
        X.loc[X.index[idx], col] = RARE_TOKEN
    out.X_test = X
    return out


def _covariate_shift(split: Split, lam: float, base_seed: int) -> Split:
    """Mean-shift TEST numeric features by ``lam * strength * std`` (labels unchanged)."""
    out = _copy_split(split)
    if lam <= 0 or not split.numeric_columns:
        return out
    delta = lam * COVARIATE_SHIFT_STRENGTH
    for col in split.numeric_columns:
        std = float(out.X_train[col].std(ddof=0))
        if not np.isfinite(std) or std == 0.0:
            continue
        out.X_test[col] = out.X_test[col] + delta * std
    return out


def _context_budget(split: Split, lam: float, base_seed: int) -> Split:
    """Shrink the train context to a (1 - lam) stratified fraction."""
    out = _copy_split(split)
    if lam <= 0:
        return out
    rng = make_rng(base_seed, split.dataset_id, "context_budget", lam)
    y = out.y_train.to_numpy()
    keep_frac = 1.0 - lam
    keep_idx: list[int] = []
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        n_keep = max(1, int(round(keep_frac * cls_idx.size)))
        chosen = rng.choice(cls_idx, size=n_keep, replace=False)
        keep_idx.extend(chosen.tolist())
    keep_idx = sorted(keep_idx)
    out.X_train = out.X_train.iloc[keep_idx].reset_index(drop=True)
    out.y_train = out.y_train.iloc[keep_idx].reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Shift:
    """A named shift axis with its perturbation policy and implementation status."""

    axis: str
    split_policy: str  # "train_only" | "test_only" | "train_test"
    implementation: str  # "real" | "conditional"
    fn: Callable[[Split, float, int], Split]
    requires_categorical: bool = False

    def applicable(self, split: Split) -> bool:
        if self.requires_categorical and not split.categorical_columns:
            return False
        return True

    def apply(self, split: Split, lam: float, base_seed: int) -> Split:
        if lam == 0.0:  # hard no-op invariant
            return _copy_split(split)
        return self.fn(split, lam, base_seed)


_SHIFTS: dict[str, Shift] = {
    "label_noise": Shift("label_noise", "train_only", "real", _label_noise),
    "class_imbalance": Shift("class_imbalance", "train_only", "real", _class_imbalance),
    "missingness_shift": Shift(
        "missingness_shift", "test_only", "real", _missingness_shift
    ),
    "rare_category_shift": Shift(
        "rare_category_shift",
        "test_only",
        "conditional",
        _rare_category_shift,
        requires_categorical=True,
    ),
    "covariate_shift": Shift("covariate_shift", "test_only", "real", _covariate_shift),
    "context_budget": Shift("context_budget", "train_only", "real", _context_budget),
}

SHIFT_AXES: tuple[str, ...] = tuple(_SHIFTS)


def get_shift(axis: str) -> Shift:
    if axis not in _SHIFTS:
        raise KeyError(f"Unknown shift axis '{axis}'. Available: {sorted(_SHIFTS)}")
    return _SHIFTS[axis]


def is_applicable(axis: str, split: Split) -> bool:
    return get_shift(axis).applicable(split)


def apply_shift(axis: str, split: Split, lam: float, base_seed: int) -> Split:
    return get_shift(axis).apply(split, lam, base_seed)
