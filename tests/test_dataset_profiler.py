"""Tests for the dataset profiler (M1 meta-features)."""

from __future__ import annotations

import math

import pytest

from tice.datasets.profiler import profile_dataset
from tice.datasets.registry import load_dataset, make_clean_split

SEED = 7


@pytest.mark.parametrize("dataset_id", ["breast_cancer", "wine", "iris", "synthetic_mixed"])
def test_profile_basic_shape(dataset_id: str) -> None:
    dataset = load_dataset(dataset_id)
    profile = profile_dataset(dataset, base_seed=SEED)

    assert profile.dataset_id == dataset_id
    assert profile.n_rows == dataset.X.shape[0]
    assert profile.n_features == dataset.X.shape[1]
    assert profile.n_classes == dataset.n_classes
    # numeric + categorical ratios partition the feature space
    assert math.isclose(
        profile.numeric_feature_ratio + profile.categorical_feature_ratio, 1.0
    )
    assert profile.class_imbalance_ratio >= 1.0
    assert 0.0 <= profile.missing_rate <= 1.0
    assert 0.0 <= profile.duplicate_rate <= 1.0
    assert 0.0 <= profile.train_test_duplicate_rate <= 1.0


def test_numeric_dataset_has_no_categoricals() -> None:
    profile = profile_dataset(load_dataset("breast_cancer"), base_seed=SEED)
    assert profile.categorical_feature_ratio == 0.0
    assert profile.mean_cardinality == 0.0
    assert profile.max_cardinality == 0.0


def test_mixed_dataset_has_categoricals() -> None:
    profile = profile_dataset(load_dataset("synthetic_mixed"), base_seed=SEED)
    assert profile.categorical_feature_ratio > 0.0
    assert profile.mean_cardinality > 0.0
    assert profile.max_cardinality >= profile.mean_cardinality


def test_feature_shift_proxy_on_clean_split_is_near_chance() -> None:
    dataset = load_dataset("wine")
    split = make_clean_split(dataset, base_seed=SEED)
    profile = profile_dataset(dataset, base_seed=SEED, split=split)
    # A clean random split should be hard to tell apart (AUC ~ 0.5).
    assert math.isnan(profile.feature_shift_proxy) or 0.2 <= profile.feature_shift_proxy <= 0.8
