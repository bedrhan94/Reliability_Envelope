"""Merge the LOCAL-weights TabPFN multiseed run into the 6-model multiseed primary.

The cloud ``tabpfn_client`` hit the PriorLabs daily quota and left 740 of its 1584
multiseed conditions unfinished, so the multi-seed analysis had to fall back to TabICL
as the sole ICL representative. The local ``tabpfn`` package (pinned weights, run on the
GPU) completed all 1584 conditions. This script folds those rows into the multiseed
primary so the multi-seed benchmark has TWO complete ICL models (tabicl + tabpfn).

Like ``merge_tabpfn_client.py`` the local run was executed with ONLY ``tabpfn`` in the
model set, so its ``reference_utility`` / ``best_gbdt`` are NaN (no GBDTs present to form
the bar -- the reference-bar gotcha). We therefore take the per-(dataset, axis, seed)
reference from the primary run and recompute tabpfn's failure flags against the SAME bar
with the canonical ``evaluate_failure``, then recompute envelopes and AURE on the union.

Non-destructive: writes a new directory; the primary table is untouched.

Usage::

    python experiments/merge_tabpfn_local.py \
        --primary results/external/tables_2axis_multiseed_primary \
        --tabpfn  results/external/tables_2axis_multiseed_tabpfn_local \
        --out     results/external/tables_2axis_multiseed_primary_with_tabpfn_local
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

KEY = ["dataset_id", "shift_axis", "base_seed"]


def merge(primary: pd.DataFrame, tabpfn: pd.DataFrame, thresholds: Thresholds) -> pd.DataFrame:
    # Per-(dataset, axis, seed) GBDT reference from the primary run.
    have_ref = primary.dropna(subset=["reference_utility"])
    ref = have_ref.groupby(KEY)["reference_utility"].first()
    best = have_ref.groupby(KEY)["best_gbdt"].first()

    # sanity: the reference must be single-valued within each cell
    spread = have_ref.groupby(KEY)["reference_utility"].nunique()
    if (spread > 1).any():
        bad = spread[spread > 1].head()
        raise SystemExit(f"[merge] reference_utility not unique within cells:\n{bad}")

    tp = tabpfn.copy()
    idx = list(zip(*[tp[k] for k in KEY], strict=True))
    tp["reference_utility"] = [ref.get(k) for k in idx]
    tp["best_gbdt"] = [best.get(k) for k in idx]
    if tp["reference_utility"].isna().any():
        n = int(tp["reference_utility"].isna().sum())
        raise SystemExit(f"[merge] {n} tabpfn rows found no reference in the primary (key mismatch)")

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

    merged = pd.concat([primary, tp[primary.columns]], ignore_index=True)
    return merged.sort_values(
        ["dataset_id", "shift_axis", "base_seed", "model", "shift_lambda"]
    ).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary", type=Path, required=True)
    p.add_argument("--tabpfn", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    primary = pd.read_csv(Path(args.primary) / "shift_results.csv")
    tabpfn = pd.read_csv(Path(args.tabpfn) / "shift_results.csv")
    tabpfn = tabpfn[tabpfn["model"] == "tabpfn"]
    if tabpfn.empty:
        print("[merge] local run has no 'tabpfn' rows", file=sys.stderr)
        return 1

    merged = merge(primary, tabpfn, Thresholds())
    keys = ("model", "dataset_id", "shift_axis", "base_seed")
    env = compute_envelopes(merged, group_keys=keys)
    aure = compute_aure(env)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out / "shift_results.csv", index=False)
    env.to_csv(out / "reliability_envelopes.csv", index=False)
    aure.to_csv(out / "aure_summary.csv", index=False)
    prof = Path(args.primary) / "dataset_profiles.csv"
    if prof.exists():
        pd.read_csv(prof).to_csv(out / "dataset_profiles.csv", index=False)

    tp_ok = int((tabpfn["status"] == "ok").sum())
    print(f"[merge] tabpfn(local) rows merged: {len(tabpfn)} ({tp_ok} ok)")
    print(f"[merge] models now: {sorted(merged.model.unique())}")
    print(f"[merge] rows={len(merged)}")
    print("[merge] AURE (mean rho over all cells incl. seeds):")
    print(aure[["model", "aure"]].sort_values("aure", ascending=False).round(4).to_string(index=False))
    print(f"[merge] wrote tables to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
