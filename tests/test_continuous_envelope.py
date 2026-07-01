"""Tests for the de-quantised (continuous) envelope radius."""

from __future__ import annotations

import math

from tice.envelope.reliability import (
    continuous_envelope_radius,
    envelope_radius,
)

_TAU = dict(tau_utility=0.03, tau_ece=0.10, tau_nll=0.75)
_LAMS = [0.0, 0.1, 0.2]


def _rho(utility, ece, nll, ref=0.5):
    return continuous_envelope_radius(_LAMS, utility, ece, nll, ref, **_TAU)


def test_never_fails_returns_max_lambda():
    rho = _rho([0.5, 0.5, 0.5], [0.02, 0.02, 0.02], [0.1, 0.1, 0.1])
    assert rho == 0.2


def test_utility_crossing_is_interpolated_between_grid_points():
    # thr = ref - tau_u = 0.47; utility 0.5 -> 0.3 between lambda 0.1 and 0.2.
    rho = _rho([0.5, 0.5, 0.3], [0.02, 0.02, 0.02], [0.1, 0.1, 0.1])
    assert math.isclose(rho, 0.115, abs_tol=1e-9)


def test_ece_crossing_is_interpolated():
    # ece 0.05 -> 0.15 crosses tau_ece=0.10 halfway between 0.1 and 0.2.
    rho = _rho([0.5, 0.5, 0.5], [0.05, 0.05, 0.15], [0.1, 0.1, 0.1])
    assert math.isclose(rho, 0.15, abs_tol=1e-9)


def test_earliest_criterion_wins():
    # utility crosses ~0.115, ece ~0.15 -> the earlier (utility) governs rho.
    rho = _rho([0.5, 0.5, 0.3], [0.05, 0.05, 0.15], [0.1, 0.1, 0.1])
    assert math.isclose(rho, 0.115, abs_tol=1e-9)


def test_failure_at_zero_gives_zero():
    rho = _rho([0.5, 0.5, 0.5], [0.2, 0.2, 0.2], [0.1, 0.1, 0.1])  # ece>tau at lambda 0
    assert rho == 0.0


def test_nan_metric_counts_as_failure():
    rho = _rho([0.5, float("nan"), 0.5], [0.02, 0.02, 0.02], [0.1, 0.1, 0.1])
    assert rho == 0.1  # fails at the NaN grid point (lambda=0.1)


def test_continuous_brackets_the_quantised_radius():
    # continuous rho must sit in [grid_rho, first_fail_lambda].
    utility, ece, nll = [0.5, 0.5, 0.3], [0.02, 0.02, 0.02], [0.1, 0.1, 0.1]
    failed = [
        (u < 0.5 - _TAU["tau_utility"]) or (e > _TAU["tau_ece"]) or (n > _TAU["tau_nll"])
        for u, e, n in zip(utility, ece, nll, strict=True)
    ]
    grid = envelope_radius(_LAMS, failed)
    cont = _rho(utility, ece, nll)
    assert grid <= cont <= 0.2
