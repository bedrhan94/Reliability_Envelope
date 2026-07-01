"""Shared, leakage-safe preprocessing.

A single ``ColumnTransformer`` factory is reused by the model runner and the
profiler's domain classifier so numeric imputation and one-hot encoding behave
identically everywhere. ``handle_unknown="ignore"`` matters for
``rare_category_shift``: novel test categories map to all-zero one-hot rows
instead of crashing.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    *,
    scale: bool = True,
) -> ColumnTransformer:
    """Build a preprocessor: impute+scale numerics, impute+one-hot categoricals."""
    numeric_steps: list = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_columns:
        transformers.append(("num", numeric_pipe, list(numeric_columns)))
    if categorical_columns:
        transformers.append(("cat", categorical_pipe, list(categorical_columns)))

    return ColumnTransformer(transformers=transformers, remainder="drop")
