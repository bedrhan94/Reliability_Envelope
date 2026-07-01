"""Tests for utility, failure indicator, envelope radius, and AURE."""

from __future__ import annotations

import math

import pandas as pd

from tice.config import Thresholds
from tice.envelope.reliability import compute_aure, compute_envelopes, envelope_radius
from tice.metrics.utility import evaluate_failure, reliability_utility

LAMBDAS = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40]


# --------------------------------------------------------------------------- #
# envelope_radius
# --------------------------------------------------------------------------- #
def test_rho_fail_at_zero_is_zero() -> None:
    failed = [True, True, True, True, True, True]
    assert envelope_radius(LAMBDAS, failed) == 0.0


def test_rho_never_fails_is_max() -> None:
    failed = [False] * len(LAMBDAS)
    assert envelope_radius(LAMBDAS, failed) == max(LAMBDAS)


def test_rho_fail_in_middle_caps_before_first_failure() -> None:
    # passes through 0.10, fails at 0.20, then "recovers" at 0.30 -- still capped
    failed = [False, False, False, True, False, False]
    assert envelope_radius(LAMBDAS, failed) == 0.10


def test_rho_is_order_independent() -> None:
    shuffled = list(
        zip(
            [0.40, 0.00, 0.20, 0.05, 0.30, 0.10],
            [False, False, True, False, False, False],
            strict=True,
        )
    )
    lambdas = [x[0] for x in shuffled]
    failed = [x[1] for x in shuffled]
    assert envelope_radius(lambdas, failed) == 0.10


# --------------------------------------------------------------------------- #
# utility + failure
# --------------------------------------------------------------------------- #
def test_reliability_utility_formula() -> None:
    u = reliability_utility(auc=0.9, accuracy=0.8, nll_norm=0.5, ece=0.1)
    expected = 0.35 * 0.9 + 0.15 * 0.8 - 0.30 * 0.5 - 0.20 * 0.1
    assert math.isclose(u, expected)


def test_reliability_utility_nan_propagates() -> None:
    assert math.isnan(reliability_utility(float("nan"), 0.8, 0.5, 0.1))


def test_failure_utility_gap() -> None:
    th = Thresholds(tau_utility=0.03, tau_ece=0.10, tau_nll=0.75)
    res = evaluate_failure(
        utility=0.50, ece=0.05, nll_norm=0.3, reference_utility=0.60, thresholds=th
    )
    assert res.failed and "utility_gap" in res.reasons


def test_failure_ece_and_nll_thresholds() -> None:
    th = Thresholds()
    ece_fail = evaluate_failure(
        utility=0.9, ece=0.20, nll_norm=0.3, reference_utility=0.9, thresholds=th
    )
    nll_fail = evaluate_failure(
        utility=0.9, ece=0.05, nll_norm=0.80, reference_utility=0.9, thresholds=th
    )
    assert ece_fail.failed and "ece" in ece_fail.reasons
    assert nll_fail.failed and "nll" in nll_fail.reasons


def test_failure_pass_within_thresholds() -> None:
    th = Thresholds()
    res = evaluate_failure(
        utility=0.59, ece=0.05, nll_norm=0.3, reference_utility=0.60, thresholds=th
    )
    assert not res.failed


def test_failure_on_non_ok_status() -> None:
    th = Thresholds()
    res = evaluate_failure(
        utility=float("nan"),
        ece=float("nan"),
        nll_norm=float("nan"),
        reference_utility=0.6,
        thresholds=th,
        status_ok=False,
    )
    assert res.failed and "undefined" in res.reasons


# --------------------------------------------------------------------------- #
# compute_envelopes + compute_aure
# --------------------------------------------------------------------------- #
def _make_results(model: str, axis: str, fail_from_lambda: float | None) -> pd.DataFrame:
    rows = []
    for lam in LAMBDAS:
        failed = fail_from_lambda is not None and lam >= fail_from_lambda
        rows.append(
            {
                "model": model,
                "dataset_id": "ds",
                "shift_axis": axis,
                "shift_lambda": lam,
                "status": "ok",
                "failed": failed,
            }
        )
    return pd.DataFrame(rows)


def test_compute_envelopes_and_aure() -> None:
    df = pd.concat(
        [
            _make_results("m1", "label_noise", fail_from_lambda=0.20),
            _make_results("m1", "covariate_shift", fail_from_lambda=None),
            _make_results("m2", "label_noise", fail_from_lambda=0.00),
        ],
        ignore_index=True,
    )
    env = compute_envelopes(df)
    rho_lookup = {
        (r.model, r.shift_axis): r.rho for r in env.itertuples()
    }
    assert rho_lookup[("m1", "label_noise")] == 0.10
    assert rho_lookup[("m1", "covariate_shift")] == 0.40
    assert rho_lookup[("m2", "label_noise")] == 0.00

    aure = compute_aure(env)
    aure_lookup = {r.model: r.aure for r in aure.itertuples()}
    assert math.isclose(aure_lookup["m1"], (0.10 + 0.40) / 2)
    assert math.isclose(aure_lookup["m2"], 0.00)


def test_compute_envelopes_drops_all_skipped_groups() -> None:
    rows = [
        {
            "model": "m1",
            "dataset_id": "ds",
            "shift_axis": "rare_category_shift",
            "shift_lambda": lam,
            "status": "skipped",
            "failed": False,
        }
        for lam in LAMBDAS
    ]
    env = compute_envelopes(pd.DataFrame(rows))
    assert env.empty


def test_compute_envelopes_drops_all_error_groups() -> None:
    # A model that errored at every condition (e.g. unauthenticated) is not
    # measurable and must be excluded, not scored rho=0.
    rows = [
        {
            "model": "broken",
            "dataset_id": "ds",
            "shift_axis": "label_noise",
            "shift_lambda": lam,
            "status": "error",
            "failed": True,
        }
        for lam in LAMBDAS
    ]
    env = compute_envelopes(pd.DataFrame(rows))
    assert env.empty


def test_compute_envelopes_keeps_group_with_late_error_and_caps_rho() -> None:
    # ok through 0.10, then errors at 0.20 onward -> group kept, rho capped at 0.10.
    rows = []
    for lam in LAMBDAS:
        late_error = lam >= 0.20
        rows.append(
            {
                "model": "m1",
                "dataset_id": "ds",
                "shift_axis": "covariate_shift",
                "shift_lambda": lam,
                "status": "error" if late_error else "ok",
                "failed": late_error,
            }
        )
    env = compute_envelopes(pd.DataFrame(rows))
    assert len(env) == 1
    assert env.iloc[0]["rho"] == 0.10
