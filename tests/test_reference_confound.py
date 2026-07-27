"""Tests for the reference-confound ablation and the tau/lambda=0 sweep.

The ablation exists to show that the published failure rule can decide a cell at
``lambda=0`` -- before any shift -- so these tests pin the two properties the
argument rests on: the self-referenced rule can *never* fail at ``lambda=0``, and
common support is the intersection over models (identical cells for everyone).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from ablate_reference_confound import (  # noqa: E402
    TAU_ECE,
    TAU_U,
    _cell_keys,
    add_self_ref_failure,
    run,
)

LAMBDAS = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40]


def _frame(spec: dict[str, dict[str, list[float]]], *, seeds: int = 1) -> pd.DataFrame:
    """Build a shift_results-shaped frame: spec[model] = {metric: per-lambda values}."""
    rows = []
    for base_seed in range(seeds):
        for model, series in spec.items():
            for i, lam in enumerate(LAMBDAS):
                rows.append(
                    {
                        "dataset_id": "ds1",
                        "model": model,
                        "shift_axis": "label_noise",
                        "shift_lambda": lam,
                        "base_seed": base_seed,
                        "status": "ok",
                        "utility": series["utility"][i],
                        "ece": series.get("ece", [0.01] * len(LAMBDAS))[i],
                        "nll_norm": series.get("nll_norm", [0.1] * len(LAMBDAS))[i],
                        "reference_utility": series.get(
                            "reference_utility", [0.40] * len(LAMBDAS)
                        )[i],
                        "failed": series["failed"][i],
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# self-referenced rule
# --------------------------------------------------------------------------- #
def test_self_ref_never_fails_at_lambda_zero() -> None:
    """The whole point of the ablation: no cell is decided before the shift starts."""
    df = _frame(
        {
            # a model that is already far below the shared reference when clean
            "weak": {
                "utility": [0.20] * 6,
                "ece": [0.30] * 6,
                "nll_norm": [1.5] * 6,
                "failed": [True] * 6,
            }
        }
    )
    out = add_self_ref_failure(df, _cell_keys(df))
    assert not out.loc[out.shift_lambda == 0.0, "failed_self"].any()
    # ...and a flat model never fails at any lambda under the self-referenced rule
    assert not out["failed_self"].any()


def test_self_ref_trips_on_degradation_from_own_clean_state() -> None:
    df = _frame(
        {
            "m": {
                # drops just past tau_utility below its own clean utility at lambda=0.20
                "utility": [0.50, 0.49, 0.48, 0.50 - TAU_U - 0.01, 0.30, 0.20],
                "failed": [False] * 6,
            }
        }
    )
    out = add_self_ref_failure(df, _cell_keys(df))
    failed_at = out.loc[out.failed_self, "shift_lambda"].tolist()
    assert failed_at == [0.20, 0.30, 0.40]


def test_self_ref_ece_trigger_is_rebased_not_absolute() -> None:
    """A model clean at ECE 0.09 must not fail merely for crossing the absolute 0.10."""
    ece = [0.09, 0.095, 0.12, 0.15, 0.18, 0.09 + TAU_ECE + 0.01]
    df = _frame({"m": {"utility": [0.50] * 6, "ece": ece, "failed": [False] * 6}})
    out = add_self_ref_failure(df, _cell_keys(df))
    # only the last lambda exceeds clean_ece + TAU_ECE, despite four points > 0.10
    assert out.loc[out.failed_self, "shift_lambda"].tolist() == [0.40]


def test_bad_status_fails_under_self_ref() -> None:
    df = _frame({"m": {"utility": [0.5] * 6, "failed": [False] * 6}})
    df.loc[df.shift_lambda == 0.30, "status"] = "error"
    out = add_self_ref_failure(df, _cell_keys(df))
    assert out.loc[out.shift_lambda == 0.30, "failed_self"].all()


# --------------------------------------------------------------------------- #
# cell keys / common support
# --------------------------------------------------------------------------- #
def test_cell_keys_include_base_seed_only_when_run_has_several() -> None:
    single = _frame({"m": {"utility": [0.5] * 6, "failed": [False] * 6}}, seeds=1)
    multi = _frame({"m": {"utility": [0.5] * 6, "failed": [False] * 6}}, seeds=3)
    assert _cell_keys(single) == ["dataset_id", "shift_axis"]
    assert _cell_keys(multi) == ["dataset_id", "shift_axis", "base_seed"]


def test_common_support_is_the_intersection_over_models(tmp_path: Path) -> None:
    """A cell where any model fails at lambda=0 is excluded for *every* model."""
    passing = {"utility": [0.5] * 6, "failed": [False] * 6}
    prefailed = {"utility": [0.5] * 6, "failed": [True] + [False] * 5}
    df = _frame({"good": passing, "bad": prefailed}, seeds=2)
    summary = run_to_frame(df, tmp_path)
    # the single cell (ds1 x label_noise) is pre-failed by "bad" in both seeds
    assert int(summary.n_common.iloc[0]) == 0
    assert summary.aure_common.isna().all()

    df2 = _frame({"good": passing, "also_good": passing}, seeds=2)
    summary2 = run_to_frame(df2, tmp_path)
    assert int(summary2.n_common.iloc[0]) == 2  # both seeds survive


def run_to_frame(df: pd.DataFrame, tmp_path: Path) -> pd.DataFrame:
    csv = tmp_path / "shift_results.csv"
    df.to_csv(csv, index=False)
    return run(csv, tmp_path / "out")


def test_run_writes_both_tables_and_grid_matches_stored_failed(tmp_path: Path) -> None:
    """rho_grid must reproduce the stored `failed` column, unchanged."""
    df = _frame(
        {
            "m": {
                "utility": [0.5] * 6,
                "failed": [False, False, False, True, True, True],
            }
        }
    )
    out_dir = tmp_path / "out"
    summary = run_to_frame(df, tmp_path)
    assert (out_dir / "ablation_summary.csv").exists()
    assert (out_dir / "ablation_per_cell.csv").exists()
    per_cell = pd.read_csv(out_dir / "ablation_per_cell.csv")
    assert per_cell.rho_grid.tolist() == [0.10]
    assert summary.lambda0_fail_rate.iloc[0] == 0.0


# --------------------------------------------------------------------------- #
# tau / lambda=0 sweep
# --------------------------------------------------------------------------- #
def test_tau_sweep_reports_lambda0_failures_and_margin() -> None:
    tau_mod = pytest.importorskip("tau_lambda0_sensitivity")
    flat = {"utility": [0.45] * 6, "ece": [0.02] * 6, "failed": [False] * 6}
    weak = {"utility": [0.30] * 6, "ece": [0.20] * 6, "failed": [True] * 6}
    df = _frame({"tabicl": flat, "tabpfn_client": flat, "catboost": weak})
    s = tau_mod.sweep(df)
    assert len(s) == 336
    # the weak GBDT is pre-failed at lambda=0 across most of the grid
    assert (s.lambda0_fail_max > 0).any()
    # where it is pre-failed, ICL leads purely because of that
    assert (s.margin > 0).any()
