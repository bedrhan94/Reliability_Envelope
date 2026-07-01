"""Tests for the contamination / independence diagnostics."""

from __future__ import annotations

import numpy as np

from tice.contamination import (
    contamination_report,
    dataset_exposure,
    exact_duplicate_rate,
    independence_risk,
    nearest_train_distance,
    proximity_profile,
)
from tice.datasets.registry import load_dataset, make_clean_split


def _split(name: str):
    return make_clean_split(load_dataset(name), base_seed=42, test_size=0.3)


def test_exposure_tags_public_vs_synthetic():
    assert dataset_exposure("iris") == "public_named"
    assert dataset_exposure("breast_cancer") == "public_named"
    assert dataset_exposure("synthetic_mixed") == "synthetic"
    assert dataset_exposure("anything_else") == "synthetic"


def test_no_exact_train_test_duplicates_on_clean_split():
    # sklearn toy splits share no verbatim rows -> the shipped check is inert.
    assert exact_duplicate_rate(_split("iris")) == 0.0
    assert exact_duplicate_rate(_split("breast_cancer")) == 0.0


def test_nearest_distance_shape_and_sign():
    split = _split("wine")
    dist = nearest_train_distance(split)
    assert dist.shape == (len(split.X_test),)
    assert np.all(dist >= 0.0)


def test_proximity_profile_monotone_in_eps():
    prof = proximity_profile(_split("iris"), (0.0, 0.1, 0.5))
    assert prof[0.0] <= prof[0.1] <= prof[0.5]
    assert prof[0.5] <= 1.0


def test_independence_risk_thresholds():
    assert independence_risk(0.0) == "low"
    assert independence_risk(0.2) == "medium"
    assert independence_risk(0.9) == "high"
    assert independence_risk(float("nan")) == "unknown"


def test_iris_near_duplicate_structure_flags_high_even_though_exact_is_low():
    # The point of the whole module: exact-dup says 'low', near-dup says 'high'.
    rep = contamination_report(_split("iris"))
    assert rep.exact_dup_rate == 0.0
    assert rep.near_dup_rate >= 0.5
    assert rep.independence_risk == "high"


def test_report_dict_roundtrip_keys():
    rep = contamination_report(_split("synthetic_mixed"))
    d = rep.as_dict()
    assert d["exposure"] == "synthetic"
    assert set(d) == {
        "dataset_id",
        "exposure",
        "exact_dup_rate",
        "near_dup_rate",
        "near_dup_eps_sd",
        "independence_risk",
    }
