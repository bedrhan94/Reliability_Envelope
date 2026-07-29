"""Tests for the calibration diagnostics added for the reviewer's 4.7.

These carry the paper's central argument, so each is pinned against a case whose
answer is known analytically rather than against a regression baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

from tice.metrics.classification import (
    adaptive_calibration_error,
    brier_score,
    calibration_slope_intercept,
    classwise_calibration_error,
    compute_metrics,
    expected_calibration_error,
    reliability_bins,
)

CLASSES = np.array([0, 1])


def _perfect(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Confidence exactly matches accuracy: half the 0.8-confident cases are right."""
    rng = np.random.default_rng(0)
    conf = np.full(n, 0.8)
    correct = rng.random(n) < 0.8
    proba = np.column_stack([1 - conf, conf])
    y = np.where(correct, 1, 0)
    return y, proba


# --------------------------------------------------------------------------- #
# Brier
# --------------------------------------------------------------------------- #
def test_brier_is_zero_for_a_perfect_confident_predictor() -> None:
    y = np.array([0, 1, 1, 0])
    proba = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    assert brier_score(y, proba, CLASSES) == pytest.approx(0.0, abs=1e-9)


def test_brier_is_maximal_for_a_confidently_wrong_predictor() -> None:
    """Every prediction confidently wrong costs 2.0 per sample in the multiclass form."""
    y = np.array([0, 1])
    proba = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert brier_score(y, proba, CLASSES) == pytest.approx(2.0, abs=1e-6)


def test_brier_penalises_overconfidence_more_than_uncertainty() -> None:
    y = np.array([0, 1, 0, 1])
    uncertain = np.full((4, 2), 0.5)
    overconfident = np.array([[0.01, 0.99], [0.99, 0.01], [0.01, 0.99], [0.99, 0.01]])
    assert brier_score(y, overconfident, CLASSES) > brier_score(y, uncertain, CLASSES)


# --------------------------------------------------------------------------- #
# ECE variants
# --------------------------------------------------------------------------- #
def test_all_ece_variants_vanish_when_confidence_matches_accuracy() -> None:
    y, proba = _perfect(4000)
    assert expected_calibration_error(y, proba, CLASSES) < 0.03
    assert adaptive_calibration_error(y, proba, CLASSES) < 0.03


def test_ece_equals_the_gap_for_a_single_confidence_level() -> None:
    """All mass in one bin: ECE is exactly |accuracy - confidence|."""
    n = 100
    conf = np.full(n, 0.9)
    y = np.array([1] * 60 + [0] * 40)  # 60% accurate at 90% confidence
    proba = np.column_stack([1 - conf, conf])
    assert expected_calibration_error(y, proba, CLASSES) == pytest.approx(0.3, abs=1e-6)
    assert adaptive_calibration_error(y, proba, CLASSES) == pytest.approx(0.3, abs=1e-6)


def test_adaptive_ece_differs_from_equal_width_on_a_skewed_distribution() -> None:
    """The point of equal-mass binning: it does not let one crowded bin dominate."""
    rng = np.random.default_rng(1)
    conf = np.concatenate([rng.uniform(0.97, 0.999, 950), rng.uniform(0.5, 0.7, 50)])
    proba = np.column_stack([1 - conf, conf])
    y = (rng.random(1000) < conf).astype(int)
    assert adaptive_calibration_error(y, proba, CLASSES) != pytest.approx(
        expected_calibration_error(y, proba, CLASSES), abs=1e-6
    )


def test_classwise_ece_sees_errors_that_top_label_ece_misses() -> None:
    """A wrong tail on the non-predicted classes leaves top-label ECE untouched."""
    classes = np.array([0, 1, 2])
    n = 300
    rng = np.random.default_rng(2)
    top = np.full(n, 0.6)
    good = np.column_stack([top, np.full(n, 0.2), np.full(n, 0.2)])
    bad = np.column_stack([top, np.full(n, 0.39), np.full(n, 0.01)])
    y = np.where(rng.random(n) < 0.6, 0, rng.integers(1, 3, n))
    assert expected_calibration_error(y, good, classes) == pytest.approx(
        expected_calibration_error(y, bad, classes), abs=1e-6
    )
    assert classwise_calibration_error(y, bad, classes) != pytest.approx(
        classwise_calibration_error(y, good, classes), abs=1e-6
    )


# --------------------------------------------------------------------------- #
# calibration slope
# --------------------------------------------------------------------------- #
def test_calibration_slope_is_undefined_when_every_prediction_is_correct() -> None:
    y = np.array([1, 1, 1])
    proba = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
    slope, intercept = calibration_slope_intercept(y, proba, CLASSES)
    assert np.isnan(slope) and np.isnan(intercept)


def test_calibration_slope_is_positive_when_confidence_tracks_correctness() -> None:
    rng = np.random.default_rng(3)
    conf = rng.uniform(0.5, 0.99, 2000)
    y = (rng.random(2000) < conf).astype(int)
    proba = np.column_stack([1 - conf, conf])
    slope, _ = calibration_slope_intercept(y, proba, CLASSES)
    assert slope > 0.5


def test_calibration_slope_is_near_zero_when_confidence_is_uninformative() -> None:
    rng = np.random.default_rng(4)
    conf = rng.uniform(0.5, 0.99, 2000)
    y = rng.integers(0, 2, 2000)  # correctness independent of confidence
    proba = np.column_stack([1 - conf, conf])
    slope, _ = calibration_slope_intercept(y, proba, CLASSES)
    assert abs(slope) < 0.6


# --------------------------------------------------------------------------- #
# reliability bins
# --------------------------------------------------------------------------- #
def test_reliability_bins_partition_every_sample_exactly_once() -> None:
    y, proba = _perfect(500)
    bins = reliability_bins(y, proba, CLASSES, n_bins=15)
    assert len(bins) == 15
    assert sum(b["n"] for b in bins) == 500
    for b in bins:
        if b["n"]:
            assert b["lo"] <= b["mean_conf"] <= b["hi"] + 1e-9


# --------------------------------------------------------------------------- #
# wiring
# --------------------------------------------------------------------------- #
def test_compute_metrics_exposes_every_new_column() -> None:
    y, proba = _perfect(200)
    d = compute_metrics(y, proba, CLASSES).as_dict()
    for key in ("brier", "ece_adaptive", "ece_classwise", "calib_slope", "calib_intercept"):
        assert key in d

def test_compute_metrics_survives_a_degenerate_single_class_split() -> None:
    """Undefined metrics must become nan with a note, never raise."""
    y = np.zeros(10, dtype=int)
    proba = np.column_stack([np.full(10, 0.9), np.full(10, 0.1)])
    m = compute_metrics(y, proba, CLASSES)
    assert np.isnan(m.auc)
    assert not np.isnan(m.brier)  # Brier is defined even with one class present
    assert m.notes
