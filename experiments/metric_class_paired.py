"""Robustness of the §6.10 metric-class result to the aggregation convention.

§6.10 reports the ICL-vs-GBDT margin as min-ICL minus best(max)-GBDT -- deliberately
conservative. A referee reasonably asks whether that min/max choice manufactures the sign
pattern. This recomputes each summary's margin as **mean-ICL minus mean-GBDT**, paired over
datasets, with a 95% bootstrap CI (resampling datasets), from the saved per-cell tables. If
the three-role sign pattern is identical to the min/max version, the result does not depend
on the aggregation. No models are re-run.

Usage::

    python experiments/metric_class_paired.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ICL = {"tabicl", "tabpfn", "tabpfn_client"}
GBDT = {"catboost", "xgboost", "hist_gbdt", "catboost_tuned", "xgboost_tuned",
        "lightgbm", "lightgbm_tuned", "gbdt_ensemble"}
SUMMARIES = ["aure", "robust_fraction", "abs_utility_area", "abs_ece_radius",
             "self_ref_radius", "utility_drop", "abs_auc_radius", "auc_retention"]
ROLE = {"aure": "admissibility", "robust_fraction": "admissibility",
        "abs_utility_area": "admissibility", "abs_ece_radius": "calibration_tolerance",
        "self_ref_radius": "calibration_tolerance", "utility_drop": "calibration_tolerance",
        "abs_auc_radius": "discrimination", "auc_retention": "discrimination"}
BASES = ["external_primary", "external_strong", "external_uncapped", "pilot"]


def main() -> int:
    rng = np.random.default_rng(0)
    out_rows = []
    for base in BASES:
        pc_path = _ROOT / "results" / "ablations" / "metric_class" / base / "metric_class_per_cell.csv"
        if not pc_path.exists():
            print(f"skip {base}: no per_cell table")
            continue
        pc = pd.read_csv(pc_path)
        icl = [m for m in pc.model.unique() if m in ICL]
        gbdt = [m for m in pc.model.unique() if m in GBDT]
        print(f"\n### {base}   ICL={icl}  GBDT={gbdt}")
        for s in SUMMARIES:
            dm = pc.groupby(["dataset_id", "model"])[s].mean().unstack("model")
            mi = dm[[m for m in icl if m in dm]].mean(axis=1)
            mg = dm[[m for m in gbdt if m in dm]].mean(axis=1)
            diff = (mi - mg).dropna().to_numpy()
            boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            sig = (lo > 0) or (hi < 0)
            out_rows.append({"basis": base, "summary": s, "role": ROLE[s],
                             "mean_margin": float(diff.mean()), "ci95_lo": float(lo),
                             "ci95_hi": float(hi), "sig": sig})
            print(f"  {s:18s}[{ROLE[s]:20s}] {diff.mean():+.4f}  "
                  f"CI[{lo:+.3f},{hi:+.3f}] {'*' if sig else ' '}")
    df = pd.DataFrame(out_rows)
    outdir = _ROOT / "results" / "ablations" / "metric_class"
    df.to_csv(outdir / "paired_mean_margins.csv", index=False)

    # verdict: does the sign pattern match the expected role directions on every basis?
    exp = {"admissibility": 1, "calibration_tolerance": -1, "discrimination": 1}
    df["sign_ok"] = df.apply(lambda r: np.sign(r.mean_margin) == exp[r.role], axis=1)
    print(f"\nsign matches role direction on {int(df.sign_ok.sum())}/{len(df)} (summary×basis) cells")
    print(f"wrote {outdir / 'paired_mean_margins.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
