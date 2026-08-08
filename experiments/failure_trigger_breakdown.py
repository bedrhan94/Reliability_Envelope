"""Which criterion actually trips at lambda=0, per model?

Section 6.1 argues the zero-shift failures have two structural causes: a *relative*
utility criterion measured against a gradient-boosted reference (which the reference
model itself cannot fail), and an *absolute* calibration bar the boosted trees sit near
out of the box. That argument was stated but never counted -- the failure rule has three
triggers and the paper never showed which one fires.

`failure_reason` is stored per condition, so the breakdown is a re-scoring of existing
outputs with no model inference.

Usage::

    python experiments/failure_trigger_breakdown.py
    python experiments/failure_trigger_breakdown.py --results <path> --lam 0.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PRIMARY = _ROOT / ("results/external/tables_2axis_multiseed_primary_with_tabpfn_local/"
                   "shift_results.csv")
DROP = ("tabpfn_client",)

# the three triggers of the rule in section 3.1, plus the unevaluable case
TRIGGERS = {
    "utility": "utility gap vs reference (RELATIVE)",
    "ece": "ECE above tau_ece (absolute)",
    "nll": "NLL_norm above tau_nll (absolute)",
    "error": "could not be evaluated",
}


def classify(reason: str) -> str:
    """Map a stored reason string onto one of the three triggers."""
    r = str(reason).lower()
    for key in ("utility", "ece", "nll"):
        if key in r:
            return key
    return "error"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=PRIMARY)
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    d = pd.read_csv(args.results)
    d = d[~d.model.isin(DROP)]
    at = d[d.shift_lambda == args.lam]
    failed = at[at.failed.astype(bool)].copy()
    failed["trigger"] = failed.failure_reason.map(classify)

    print(f"lambda = {args.lam}: {len(failed)} failed of {len(at)} cells "
          f"({len(failed) / len(at):.1%})\n")

    tab = (failed.groupby(["model", "trigger"]).size().unstack("trigger", fill_value=0))
    tot = at.groupby("model").size()
    order = [c for c in ("utility", "ece", "nll", "error") if c in tab.columns]
    tab = tab[order]

    head = "  ".join(f"{c:>9s}" for c in order)
    print(f"{'model':14s} {'fail %':>7s}  {head}   (share of that model's failures)")
    for m in tab.index:
        row = tab.loc[m]
        pct = row.sum() / tot[m]
        cells = "  ".join(f"{row[c]:4d} {row[c] / max(row.sum(), 1):4.0%}" for c in order)
        print(f"{m:14s} {pct:6.1%}  {cells}")

    print("\nlegend:")
    for k in order:
        print(f"  {k:8s} {TRIGGERS[k]}")

    # the load-bearing claim: is the relative criterion what separates the families?
    share = tab["utility"].div(tab.sum(axis=1).clip(lower=1))
    print("\nrelative-criterion share of each model's zero-shift failures:")
    for m in share.sort_values(ascending=False).index:
        print(f"  {m:14s} {share[m]:5.0%}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tab.assign(total_cells=tot, fail_rate=tab.sum(axis=1) / tot).to_csv(args.out)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
