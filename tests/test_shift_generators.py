"""Tests for the controlled shift generators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from tice.datasets.registry import load_dataset, make_clean_split
from tice.shifts.generators import RARE_TOKEN, SHIFT_AXES, apply_shift, is_applicable

SEED = 11


def _clean(dataset_id: str = "synthetic_mixed"):
    return make_clean_split(load_dataset(dataset_id), base_seed=SEED)


def _assert_split_equal(a, b) -> None:
    assert_frame_equal(a.X_train, b.X_train)
    assert_frame_equal(a.X_test, b.X_test)
    assert_series_equal(a.y_train, b.y_train)
    assert_series_equal(a.y_test, b.y_test)


@pytest.mark.parametrize("axis", SHIFT_AXES)
def test_lambda_zero_is_exact_noop(axis: str) -> None:
    clean = _clean()
    shifted = apply_shift(axis, clean, 0.0, SEED)
    _assert_split_equal(shifted, clean)


@pytest.mark.parametrize("axis", SHIFT_AXES)
def test_determinism(axis: str) -> None:
    clean = _clean()
    a = apply_shift(axis, clean, 0.30, SEED)
    b = apply_shift(axis, clean, 0.30, SEED)
    _assert_split_equal(a, b)


@pytest.mark.parametrize("axis", SHIFT_AXES)
def test_clean_split_not_mutated(axis: str) -> None:
    clean = _clean()
    snapshot = clean.X_train.copy(deep=True)
    apply_shift(axis, clean, 0.40, SEED)
    assert_frame_equal(clean.X_train, snapshot)


def test_label_noise_flips_only_train_labels() -> None:
    clean = _clean()
    shifted = apply_shift("label_noise", clean, 0.40, SEED)
    # train labels changed for some rows
    n_changed = int((shifted.y_train.to_numpy() != clean.y_train.to_numpy()).sum())
    assert n_changed > 0
    # features and test untouched
    assert_frame_equal(shifted.X_train, clean.X_train)
    assert_series_equal(shifted.y_test, clean.y_test)
    # all flipped labels remain valid classes
    assert set(shifted.y_train.unique()).issubset(set(clean.classes))


def test_class_imbalance_shrinks_minority_only() -> None:
    clean = _clean()
    shifted = apply_shift("class_imbalance", clean, 0.40, SEED)
    assert len(shifted.X_train) < len(clean.X_train)
    # majority class count is preserved
    clean_counts = pd.Series(clean.y_train).value_counts()
    shifted_counts = pd.Series(shifted.y_train).value_counts()
    majority = clean_counts.idxmax()
    assert shifted_counts[majority] == clean_counts[majority]
    assert_frame_equal(shifted.X_test, clean.X_test)


def test_missingness_only_in_test() -> None:
    clean = _clean()
    shifted = apply_shift("missingness_shift", clean, 0.30, SEED)
    assert shifted.X_test.isna().to_numpy().any()
    assert not shifted.X_train.isna().to_numpy().any()


def test_covariate_shift_moves_test_numeric_mean() -> None:
    clean = _clean()
    shifted = apply_shift("covariate_shift", clean, 0.40, SEED)
    col = clean.numeric_columns[0]
    assert shifted.X_test[col].mean() > clean.X_test[col].mean()
    # labels unchanged
    assert_series_equal(shifted.y_test, clean.y_test)
    assert_frame_equal(shifted.X_train, clean.X_train)


def test_context_budget_shrinks_train_keeps_classes() -> None:
    clean = _clean()
    shifted = apply_shift("context_budget", clean, 0.40, SEED)
    assert len(shifted.X_train) < len(clean.X_train)
    assert set(shifted.y_train.unique()) == set(clean.y_train.unique())
    assert_frame_equal(shifted.X_test, clean.X_test)


def test_rare_category_applicability_and_injection() -> None:
    # not applicable on a purely numeric dataset
    numeric = make_clean_split(load_dataset("breast_cancer"), base_seed=SEED)
    assert is_applicable("rare_category_shift", numeric) is False

    # applicable on the mixed dataset, injects the unseen token into test
    mixed = _clean("synthetic_mixed")
    assert is_applicable("rare_category_shift", mixed) is True
    shifted = apply_shift("rare_category_shift", mixed, 0.30, SEED)
    cat_col = mixed.categorical_columns[0]
    assert (shifted.X_test[cat_col] == RARE_TOKEN).any()
    # the token is genuinely unseen in train
    assert not (shifted.X_train[cat_col] == RARE_TOKEN).any()


def test_label_noise_amount_scales_with_lambda() -> None:
    clean = _clean()
    n = len(clean.y_train)
    changed = []
    for lam in (0.10, 0.40):
        shifted = apply_shift("label_noise", clean, lam, SEED)
        changed.append(int((shifted.y_train.to_numpy() != clean.y_train.to_numpy()).sum()))
    assert changed[0] < changed[1]
    assert np.isclose(changed[1] / n, 0.40, atol=0.05)
