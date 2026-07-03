"""Merge a tabpfn_client-only run into the token-less partial run (Sprint 3, Q F).

The partial external run has the 5 token-less models. To complete the 6-model
external comparison without re-running the 5 models, run tabpfn_client ALONE
(with the cloud token) and merge its per-condition rows here. The GBDT reference
(``reference_utility`` / ``best_gbdt``, constant per dataset in clean mode) is
taken from the partial run, so tabpfn_client's failure flags are computed against
the SAME bar as the other models -- then envelopes and AURE are recomputed on the
combined 6-model table.

Plan::

    # 1. run tabpfn_client only (needs the token), to a scratch dir:
    TABPFN_TOKEN=<key> python experiments/run_external_shift.py \
        --config configs/experiments/shift_stress_external_2axis_tabpfn_only.yaml
    # 2. merge into the completed 6-model tables:
    python experiments/merge_tabpfn_client.py \
        --partial results/external/tables_2axis_seed42_partial \
        --tabpfn  results/external/tables_2axis_seed42_tabpfn_only \
        --out     results/external/tables_2axis_seed42

Canonical results are untouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tice.config import Thresholds  # noqa: E402
from tice.envelope.reliability import compute_aure, compute_envelopes  # noqa: E402
from tice.metrics.utility import evaluate_failure  # noqa: E402


def merge(partial: pd.DataFrame, tabpfn: pd.DataFrame, thresholds: Thresholds) -> pd.DataFrame:
    # Per-dataset GBDT reference from the partial run (constant per dataset).
    ref = partial.dropna(subset=["reference_utility"]).groupby("dataset_id")["reference_utility"].first()
    best = partial.groupby("dataset_id")["best_gbdt"].first()

    tp = tabpfn.copy()
    tp["reference_utility"] = tp["dataset_id"].map(ref)
    tp["best_gbdt"] = tp["dataset_id"].map(best)
    failed, reasons = [], []
    for _, r in tp.iterrows():
        res = evaluate_failure(
            utility=r["utility"], ece=r["ece"], nll_norm=r["nll_norm"],
            reference_utility=r["reference_utility"], thresholds=thresholds,
            status_ok=(r["status"] == "ok"),
        )
        failed.append(res.failed)
        reasons.append(res.reason_str)
    tp["failed"] = failed
    tp["failure_reason"] = reasons

    merged = pd.concat([partial, tp[partial.columns]], ignore_index=True)
    return merged.sort_values(["dataset_id", "shift_axis", "model", "shift_lambda"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Merge tabpfn_client rows into the partial external run.")
    p.add_argument("--partial", type=Path, required=True)
    p.add_argument("--tabpfn", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    partial = pd.read_csv(Path(args.partial) / "shift_results.csv")
    tp_path = Path(args.tabpfn) / "shift_results.csv"
    if not tp_path.exists():
        print(f"[merge] no tabpfn shift_results at {tp_path} (run the tabpfn_only config first)", file=sys.stderr)
        return 1
    tabpfn = pd.read_csv(tp_path)
    tabpfn = tabpfn[tabpfn["model"] == "tabpfn_client"]
    if tabpfn.empty:
        print("[merge] tabpfn run has no tabpfn_client rows (token missing? all skipped?)", file=sys.stderr)
        return 1

    merged = merge(partial, tabpfn, Thresholds())
    env = compute_envelopes(merged)
    aure = compute_aure(env)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out / "shift_results.csv", index=False)
    env.to_csv(out / "reliability_envelopes.csv", index=False)
    aure.to_csv(out / "aure_summary.csv", index=False)
    # carry the profiles across for figure joins
    prof = Path(args.partial) / "dataset_profiles.csv"
    if prof.exists():
        pd.read_csv(prof).to_csv(out / "dataset_profiles.csv", index=False)

    print(f"[merge] models now: {sorted(merged.model.unique())}")
    print(f"[merge] rows={len(merged)} | AURE:\n{aure[['model','aure']].sort_values('aure', ascending=False).to_string(index=False)}")
    print(f"[merge] wrote 6-model tables to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
