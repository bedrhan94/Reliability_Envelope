"""Tests for the run-audit expected-row count.

``total_expected`` is the completeness evidence quoted in the paper ("2592 rows,
0 skipped, 0 error"), so an under-count would silently claim a complete run. The
multiseed runner multiplies the grid by the number of run seeds; single-seed runs
must be unaffected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from make_run_audit import build_audit  # noqa: E402


def _grid(*, datasets: int = 2, models: int = 3, axes: int = 2, lambdas: int = 6,
          seeds: int | None = None) -> pd.DataFrame:
    rows = []
    seed_values = range(seeds) if seeds is not None else [None]
    for base_seed in seed_values:
        for d in range(datasets):
            for m in range(models):
                for a in range(axes):
                    for lam in range(lambdas):
                        row = {
                            "dataset_id": f"ds{d}", "model": f"m{m}",
                            "shift_axis": f"ax{a}", "shift_lambda": lam / 10,
                            "status": "ok", "error_message": "",
                        }
                        if base_seed is not None:
                            row["base_seed"] = base_seed
                        rows.append(row)
    return pd.DataFrame(rows)


def _run_row(df: pd.DataFrame) -> pd.Series:
    audit = build_audit(df)
    return audit[audit.level == "run"].iloc[0]


def test_expected_matches_grid_without_base_seed() -> None:
    df = _grid()
    row = _run_row(df)
    assert row.total_expected == 2 * 3 * 2 * 6 == len(df)
    assert row.ok_rows == len(df)


def test_expected_scales_with_number_of_run_seeds() -> None:
    df = _grid(seeds=3)
    row = _run_row(df)
    assert row.total_expected == 2 * 3 * 2 * 6 * 3 == len(df)


def test_single_seed_column_does_not_inflate_expected() -> None:
    """A `base_seed` column with one value must behave like no column at all."""
    assert _run_row(_grid(seeds=1)).total_expected == _run_row(_grid()).total_expected


def test_per_condition_seed_column_is_ignored() -> None:
    """The pipeline's own `seed` is a per-condition hash -- it must not multiply."""
    df = _grid()
    df["seed"] = range(len(df))  # every row distinct, as the derived hash would be
    assert _run_row(df).total_expected == 2 * 3 * 2 * 6


def test_problem_rows_are_listed_and_counted() -> None:
    df = _grid(seeds=2)
    df.loc[0, ["status", "error_message"]] = ["error", "boom"]
    df.loc[1, "status"] = "skipped"
    audit = build_audit(df)
    row = audit[audit.level == "run"].iloc[0]
    assert row.error_rows == 1
    assert row.skipped_rows == 1
    assert row.ok_rows == len(df) - 2
    assert row.total_expected == len(df)  # expected counts the grid, not the successes
    problems = audit[audit.level == "problem"]
    assert len(problems) == 2
    assert "boom" in set(problems.reason)
    # the dataset carrying a failure is no longer "fully ok"
    assert row.n_datasets_fully_ok == 1
