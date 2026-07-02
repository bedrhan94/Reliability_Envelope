"""Dataset registry.

All datasets are produced locally (sklearn toy sets + a deterministic synthetic
mixed-type set), so nothing is ever downloaded. Each dataset carries explicit
numeric / categorical column metadata so shift axes that depend on column type
(``rare_category_shift``) know what they can touch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.model_selection import train_test_split

from tice.seed import derive_seed


@dataclass(frozen=True)
class Dataset:
    """A fully in-memory classification dataset."""

    dataset_id: str
    X: pd.DataFrame
    y: pd.Series
    categorical_columns: tuple[str, ...] = ()
    description: str = ""

    @property
    def numeric_columns(self) -> tuple[str, ...]:
        return tuple(c for c in self.X.columns if c not in set(self.categorical_columns))

    @property
    def classes(self) -> np.ndarray:
        return np.unique(self.y.to_numpy())

    @property
    def n_classes(self) -> int:
        return int(self.classes.size)


@dataclass
class Split:
    """A train/test split plus the column metadata it was carved from.

    Mutable on purpose: shift generators build a deep copy and rewrite its
    frames in place. The canonical clean split is never handed out for mutation
    -- generators copy it first via ``_copy_split``.
    """

    dataset_id: str
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    categorical_columns: tuple[str, ...] = ()
    # Class set of the *clean* dataset. Metrics align to this so shifted folds
    # that drop a class still produce comparable NLL/AUC.
    classes: tuple = field(default_factory=tuple)

    @property
    def numeric_columns(self) -> tuple[str, ...]:
        cat = set(self.categorical_columns)
        return tuple(c for c in self.X_train.columns if c not in cat)


def _from_sklearn(loader: Callable, dataset_id: str) -> Dataset:
    bunch = loader()
    feature_names = [str(n) for n in bunch.feature_names]
    X = pd.DataFrame(bunch.data, columns=feature_names)
    y = pd.Series(bunch.target, name="target")
    return Dataset(dataset_id=dataset_id, X=X, y=y, categorical_columns=())


def _make_synthetic_mixed(dataset_id: str = "synthetic_mixed") -> Dataset:
    """Deterministic mixed numeric/categorical dataset (no download).

    Built so it has *genuine* categorical features with varied cardinality,
    giving ``rare_category_shift`` something real to act on.
    """
    rng = np.random.default_rng(derive_seed(0, dataset_id))
    n = 600

    # Three categorical features of low / medium / high cardinality.
    cat_low = rng.choice(["a", "b"], size=n, p=[0.6, 0.4])
    cat_mid = rng.choice([f"c{i}" for i in range(5)], size=n)
    cat_high = rng.choice([f"k{i}" for i in range(15)], size=n)

    # Numeric features.
    num0 = rng.normal(0.0, 1.0, size=n)
    num1 = rng.normal(2.0, 1.5, size=n)
    num2 = rng.uniform(-1.0, 1.0, size=n)

    # Label depends on a mix of numeric and categorical signal (3 classes).
    logit = (
        1.1 * num0
        - 0.8 * num1
        + 0.5 * (cat_low == "a").astype(float)
        + 0.7 * np.isin(cat_mid, ["c0", "c1"]).astype(float)
    )
    quantiles = np.quantile(logit, [1 / 3, 2 / 3])
    y_vals = np.digitize(logit, quantiles)

    X = pd.DataFrame(
        {
            "num0": num0,
            "num1": num1,
            "num2": num2,
            "cat_low": cat_low,
            "cat_mid": cat_mid,
            "cat_high": cat_high,
        }
    )
    y = pd.Series(y_vals, name="target")
    return Dataset(
        dataset_id=dataset_id,
        X=X,
        y=y,
        categorical_columns=("cat_low", "cat_mid", "cat_high"),
        description="Synthetic mixed-type set with real categorical features.",
    )


# name -> zero-arg factory. Kept small and CPU-friendly on purpose.
_REGISTRY: dict[str, Callable[[], Dataset]] = {
    "breast_cancer": lambda: _from_sklearn(load_breast_cancer, "breast_cancer"),
    "wine": lambda: _from_sklearn(load_wine, "wine"),
    "iris": lambda: _from_sklearn(load_iris, "iris"),
    "digits": lambda: _from_sklearn(load_digits, "digits"),
    "synthetic_mixed": _make_synthetic_mixed,
}


def register_dataset(dataset_id: str, factory: Callable[[], Dataset]) -> None:
    """Register a zero-arg dataset factory under ``dataset_id`` (used by the
    external OpenML loader; the built-in toy sets are registered statically).
    Overwrites an existing entry with the same id."""
    _REGISTRY[dataset_id] = factory


def available_datasets() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def load_dataset(dataset_id: str) -> Dataset:
    if dataset_id not in _REGISTRY:
        raise KeyError(
            f"Unknown dataset '{dataset_id}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[dataset_id]()


def make_clean_split(
    dataset: Dataset,
    *,
    base_seed: int,
    test_size: float = 0.3,
) -> Split:
    """Carve the canonical, deterministic clean train/test split.

    This is the immutable baseline; shift generators copy from it and never
    mutate it in place.
    """
    seed = derive_seed(base_seed, dataset.dataset_id, "clean_split")
    stratify = dataset.y if dataset.n_classes > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    return Split(
        dataset_id=dataset.dataset_id,
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        categorical_columns=dataset.categorical_columns,
        classes=tuple(dataset.classes.tolist()),
    )
