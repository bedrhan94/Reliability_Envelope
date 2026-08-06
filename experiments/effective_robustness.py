"""Independent check: does the ICL out-of-distribution lead survive *effective robustness*?

The paper's whole framing is that clean-state quality confounds robustness measurement, and
attributes the principle to Taori et al. (2020) [effective robustness] and Miller et al. (2021)
["accuracy on the line"]. Those are *cited*, not *shown*. This script shows the phenomenon
directly in our own data, with a metric we did not design: it is the standard effective-robustness
construction, independent of AURE and of our failure rule entirely.

Method (Taori/Miller): under shift, out-of-distribution (OOD) performance is a near-linear
function of in-distribution (ID) performance in probit space. We fit that ID->OOD line on the
*baseline* models only (GBDTs + linear), then measure where the ICL models fall relative to it:

  effective_robustness(model) = mean over (dataset, axis) of [ probit(OOD) - line(probit(ID)) ]

* residual ~ 0  ->  the model's OOD performance is exactly what its ID performance predicts:
                    its apparent robustness advantage is a clean-performance advantage (the confound).
* residual > 0  ->  genuinely more robust than its clean accuracy predicts (real effective robustness).

The line is fit across the 44 datasets (many points), so the "too few models to fit a trend"
objection that made us decline the per-model Taori residual (metric_proposal.md) does not apply.
We report accuracy (Taori's quantity) and, since the confound in this paper is calibration, we
also report it on utility. No models are re-run.

Usage::

    python experiments/effective_robustness.py \
        --results results/external/tables_2axis_seed42_merged/shift_results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ICL_MODELS = ("tabicl", "tabpfn", "tabpfn_client")
_EPS = 1e-3


def _probit(p: np.ndarray) -> np.ndarray:
    return stats.norm.ppf(np.clip(p, _EPS, 1 - _EPS))


def _id_ood(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per (model, dataset, axis[, seed]): ID = metric at lambda 0, OOD = mean over lambda>0."""
    keys = ["model", "dataset_id", "shift_axis"]
    if "base_seed" in df.columns and df["base_seed"].nunique() > 1:
        keys.append("base_seed")
    ok = df[df["status"] == "ok"]
    rows = []
    for kv, g in ok.groupby(keys, sort=True):
        g0 = g[g.shift_lambda == 0.0]
        gp = g[g.shift_lambda > 0.0]
        if g0.empty or gp.empty:
            continue
        row = dict(zip(keys, kv if isinstance(kv, tuple) else (kv,), strict=True))
        row["id"] = float(g0[metric].mean())
        row["ood"] = float(gp[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _family(model: str, fam_map: dict[str, str]) -> str:
    if fam_map.get(model) == "icl" or model in ICL_MODELS:
        return "icl"
    return fam_map.get(model, "other")


def analyse(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    fam_map = df.drop_duplicates("model").set_index("model")["family"].to_dict()
    io = _id_ood(df, metric)
    io["fam"] = io["model"].map(lambda m: _family(m, fam_map))
    io["z_id"] = _probit(io["id"].to_numpy())
    io["z_ood"] = _probit(io["ood"].to_numpy())

    base = io[io.fam != "icl"]
    if len(base) < 3:
        raise SystemExit("[effrob] too few baseline points to fit the ID->OOD line")
    slope, intercept, r, _, _ = stats.linregress(base["z_id"], base["z_ood"])
    io["z_pred"] = intercept + slope * io["z_id"]
    io["residual"] = io["z_ood"] - io["z_pred"]  # effective robustness in probit units

    out = (
        io.groupby("model")
        .agg(fam=("fam", "first"), id=("id", "mean"), ood=("ood", "mean"),
             eff_robust=("residual", "mean"), n=("residual", "size"))
        .reset_index()
        .sort_values("eff_robust", ascending=False)
    )
    out.attrs["r2"] = r * r
    out.attrs["metric"] = metric
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--metrics", type=str, default="accuracy,utility")
    args = p.parse_args(argv)

    df = pd.read_csv(args.results)
    for metric in args.metrics.split(","):
        if metric not in df.columns:
            print(f"[effrob] no column '{metric}', skipping")
            continue
        out = analyse(df, metric)
        icl = out[out.fam == "icl"]["eff_robust"]
        base = out[out.fam != "icl"]["eff_robust"]
        print(f"\n=== effective robustness on '{metric}'  "
              f"(baseline ID->OOD line R^2={out.attrs['r2']:.3f}) ===")
        print(out.round(4).to_string(index=False))
        print(f"  mean effective robustness (probit): ICL {icl.mean():+.4f}  "
              f"baselines {base.mean():+.4f}  ->  ICL residual above baseline line "
              f"{'≈ 0 (CONFOUND: OOD lead = clean-perf lead)' if abs(icl.mean()) < 0.05 else 'nonzero'}")
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            out.to_csv(args.out / f"effrob_{metric}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
