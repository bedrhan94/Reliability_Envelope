"""Dataset profiler (M1).

Extracts a fixed set of meta-features per dataset, including a basic feature
shift proxy (a domain classifier separating train from test rows).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from tice.datasets.registry import Dataset, Split, make_clean_split
from tice.preprocess import build_preprocessor
from tice.seed import derive_seed


@dataclass(frozen=True)
class DatasetProfile:
    dataset_id: str
    n_rows: int
    n_features: int
    n_classes: int
    numeric_feature_ratio: float
    categorical_feature_ratio: float
    mean_cardinality: float
    max_cardinality: float
    class_imbalance_ratio: float
    missing_rate: float
    duplicate_rate: float
    train_test_duplicate_rate: float
    feature_shift_proxy: float

    def as_dict(self) -> dict:
        return asdict(self)


def _cardinality_stats(
    X: pd.DataFrame, categorical_columns: tuple[str, ...]
) -> tuple[float, float]:
    if not categorical_columns:
        return 0.0, 0.0
    cards = [int(X[c].nunique(dropna=True)) for c in categorical_columns]
    return float(np.mean(cards)), float(np.max(cards))


def _class_imbalance_ratio(y: pd.Series) -> float:
    counts = y.value_counts()
    if counts.empty or counts.min() == 0:
        return float("inf")
    return float(counts.max() / counts.min())


def _row_duplicate_rate(X: pd.DataFrame) -> float:
    if len(X) == 0:
        return 0.0
    return float(X.duplicated().mean())


def _train_test_duplicate_rate(X_train: pd.DataFrame, X_test: pd.DataFrame) -> float:
    """Fraction of test rows whose feature vector also appears in train.

    A direct train/test contamination proxy, which is the whole point of the
    "contamination-aware" framing.
    """
    if len(X_test) == 0:
        return 0.0
    train_keys = set(map(tuple, X_train.to_numpy().tolist()))
    hits = sum(1 for row in X_test.to_numpy().tolist() if tuple(row) in train_keys)
    return float(hits / len(X_test))


def _feature_shift_proxy(split: Split, base_seed: int) -> float:
    """Domain-classifier AUC separating train rows (0) from test rows (1).

    ~0.5 means train and test are indistinguishable (no covariate shift);
    values approaching 1.0 indicate strong distribution shift. For the clean
    split this should sit near 0.5.
    """
    X = pd.concat([split.X_train, split.X_test], axis=0, ignore_index=True)
    domain = np.concatenate(
        [np.zeros(len(split.X_train)), np.ones(len(split.X_test))]
    ).astype(int)

    n = len(domain)
    if n < 20 or len(np.unique(domain)) < 2:
        return float("nan")

    pre = build_preprocessor(split.numeric_columns, split.categorical_columns)
    seed = derive_seed(base_seed, split.dataset_id, "feature_shift_proxy")
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    try:
        scores = cross_val_score(pipe, X, domain, cv=3, scoring="roc_auc")
    except Exception:
        return float("nan")
    return float(np.mean(scores))


def profile_dataset(
    dataset: Dataset,
    *,
    base_seed: int,
    test_size: float = 0.3,
    split: Split | None = None,
) -> DatasetProfile:
    """Compute the full meta-feature profile for ``dataset``."""
    if split is None:
        split = make_clean_split(dataset, base_seed=base_seed, test_size=test_size)

    n_features = dataset.X.shape[1]
    n_cat = len(dataset.categorical_columns)
    n_num = n_features - n_cat
    mean_card, max_card = _cardinality_stats(dataset.X, dataset.categorical_columns)

    return DatasetProfile(
        dataset_id=dataset.dataset_id,
        n_rows=int(dataset.X.shape[0]),
        n_features=int(n_features),
        n_classes=int(dataset.n_classes),
        numeric_feature_ratio=float(n_num / n_features) if n_features else 0.0,
        categorical_feature_ratio=float(n_cat / n_features) if n_features else 0.0,
        mean_cardinality=mean_card,
        max_cardinality=max_card,
        class_imbalance_ratio=_class_imbalance_ratio(dataset.y),
        missing_rate=float(dataset.X.isna().to_numpy().mean()),
        duplicate_rate=_row_duplicate_rate(dataset.X),
        train_test_duplicate_rate=_train_test_duplicate_rate(
            split.X_train, split.X_test
        ),
        feature_shift_proxy=_feature_shift_proxy(split, base_seed),
    )


def profile_datasets(
    dataset_ids: list[str],
    *,
    base_seed: int,
    test_size: float = 0.3,
) -> pd.DataFrame:
    """Profile several datasets and return a tidy DataFrame (one row each)."""
    from tice.datasets.registry import load_dataset

    rows = [
        profile_dataset(
            load_dataset(ds), base_seed=base_seed, test_size=test_size
        ).as_dict()
        for ds in dataset_ids
    ]
    return pd.DataFrame(rows)
