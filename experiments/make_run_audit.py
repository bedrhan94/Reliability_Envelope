"""Build a run audit from a shift_results.csv.

Emits `run_audit.csv` with two record levels in one file:
  * level="run"     -- one summary row (total_expected, ok/skipped/error counts,
                       #datasets, #datasets_fully_ok).
  * level="problem" -- one row per skipped/error condition (dataset, model, axis,
                       lambda, status, reason) so nothing is silently dropped.

Usage::

    python experiments/make_run_audit.py --tables results/external/tables_2axis_seed42_partial
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_COLS = [
    "level", "total_expected", "ok_rows", "skipped_rows", "error_rows",
    "n_datasets", "n_datasets_fully_ok",
    "dataset_id", "model", "shift_axis", "shift_lambda", "status", "reason",
]


def build_audit(shift_results: pd.DataFrame) -> pd.DataFrame:
    df = shift_results
    n_ds = df["dataset_id"].nunique()
    n_md = df["model"].nunique()
    n_ax = df["shift_axis"].nunique()
    n_lm = df["shift_lambda"].nunique()
    expected = n_ds * n_md * n_ax * n_lm

    ok_by_ds = df.groupby("dataset_id")["status"].apply(lambda s: (s == "ok").all())
    n_ds_ok = int(ok_by_ds.sum())

    rows: list[dict] = [{
        "level": "run", "total_expected": expected,
        "ok_rows": int((df["status"] == "ok").sum()),
        "skipped_rows": int((df["status"] == "skipped").sum()),
        "error_rows": int((df["status"] == "error").sum()),
        "n_datasets": n_ds, "n_datasets_fully_ok": n_ds_ok,
    }]
    probs = df[df["status"] != "ok"]
    for _, r in probs.iterrows():
        reason = str(r.get("error_message", "") or "")
        rows.append({
            "level": "problem", "dataset_id": r["dataset_id"], "model": r["model"],
            "shift_axis": r["shift_axis"], "shift_lambda": r["shift_lambda"],
            "status": r["status"], "reason": reason[:200],
        })
    return pd.DataFrame(rows).reindex(columns=_COLS)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build run_audit.csv from shift_results.csv.")
    p.add_argument("--tables", type=Path, required=True)
    args = p.parse_args(argv)

    sr_path = Path(args.tables) / "shift_results.csv"
    if not sr_path.exists():
        print(f"[audit] no shift_results at {sr_path}", file=sys.stderr)
        return 1
    df = pd.read_csv(sr_path)
    audit = build_audit(df)
    out = Path(args.tables) / "run_audit.csv"
    audit.to_csv(out, index=False)

    run = audit[audit["level"] == "run"].iloc[0]
    n_prob = int((audit["level"] == "problem").sum())
    print(f"[audit] expected={int(run.total_expected)} ok={int(run.ok_rows)} "
          f"skipped={int(run.skipped_rows)} error={int(run.error_rows)} | "
          f"datasets={int(run.n_datasets)} fully_ok={int(run.n_datasets_fully_ok)} | problems={n_prob}")
    print(f"[audit] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
