"""Are lambda values comparable across shift axes, and what if they are not?

Reviewer objection: AURE averages raw radii over six heterogeneous perturbation
mechanisms, but flipping 20% of training labels and shifting test numerics by
0.2*strength*sigma are not the same "severity" in any principled sense, so the
average is hard to interpret.

This quantifies the problem instead of arguing about it, and offers a summary that
does not depend on it:

1. **Effective severity.** For each axis, how much does lambda actually degrade the
   models? If one unit of lambda costs three times as much AUC on one axis as on
   another, the axes are not on a common scale and the raw average inherits that.
2. **Axis-wise results.** AURE and the ICL-vs-GBDT margin per axis, so the aggregate
   never has to carry a claim on its own.
3. **A threshold-free alternative.** Area under the degradation curve, AUDC: the
   trapezoidal mean of test AUC over the lambda grid, divided by AUC at lambda=0.
   It is a retention fraction in roughly [0,1], it uses no failure rule, and it
   therefore has neither the hard-fail behaviour of rho nor its grid quantisation --
   which makes it a useful check on whether the envelope ordering is an artefact of
   the thresholding rather than of the underlying degradation.

AUDC does not solve the incomparability -- lambda still means different things per
axis -- so it is reported per axis as well as pooled.

Usage::

    python experiments/axis_analysis.py \
        --results results/canonical/tables/shift_results.csv --label pilot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tice.envelope.reliability import compute_envelopes  # noqa: E402

ICL = ("tabicl", "tabpfn_client")


def _keys(df: pd.DataFrame) -> tuple[str, ...]:
    k = ("model", "dataset_id", "shift_axis")
    if "base_seed" in df.columns and df["base_seed"].nunique() > 1:
        k += ("base_seed",)
    return k


def effective_severity(df: pd.DataFrame) -> pd.DataFrame:
    """How much AUC does one unit of lambda actually cost, per axis?"""
    ok = df[(df.status == "ok") & df.auc.notna()]
    rows = []
    for axis, g in ok.groupby("shift_axis"):
        piv = g.pivot_table(index="model", columns="shift_lambda", values="auc")
        lams = sorted(c for c in piv.columns if c > 0)
        if 0.0 not in piv.columns or not lams:
            continue
        drop = (piv[0.0] - piv[max(lams)]).mean()
        rows.append({
            "shift_axis": axis,
            "mean_auc_at_0": float(piv[0.0].mean()),
            "mean_auc_at_max": float(piv[max(lams)].mean()),
            "mean_drop": float(drop),
            "drop_per_unit_lambda": float(drop / max(lams)),
            "n_cells": int(len(g)),
        })
    out = pd.DataFrame(rows).sort_values("mean_drop", ascending=False)
    if len(out) > 1:
        lo, hi = out.mean_drop.min(), out.mean_drop.max()
        out.attrs["spread"] = float(hi / lo) if lo > 0 else float("inf")
    return out


def audc(df: pd.DataFrame) -> pd.DataFrame:
    """Trapezoidal mean AUC over the lambda grid, relative to lambda=0."""
    ok = df[(df.status == "ok") & df.auc.notna()]
    keys = [k for k in _keys(df) if k != "model"]
    rows = []
    for (model, *rest), g in ok.groupby(["model", *keys]):
        g = g.sort_values("shift_lambda")
        lam, auc = g.shift_lambda.to_numpy(float), g.auc.to_numpy(float)
        if lam.size < 2 or lam[0] != 0.0 or auc[0] <= 0:
            continue
        area = np.trapezoid(auc, lam) / (lam[-1] - lam[0])
        rec = {"model": model, "audc": float(area / auc[0])}
        rec.update(dict(zip(keys, rest, strict=True)))
        rows.append(rec)
    return pd.DataFrame(rows)


def margin(series: pd.Series, families: pd.Series) -> float:
    icl = [m for m in ICL if m in series.index]
    gbdt = [m for m in series.index if families.get(m) == "gbdt"]
    if not icl or not gbdt:
        return float("nan")
    return float(series[icl].min() - series[gbdt].max())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--label", default="run")
    p.add_argument("--out", type=Path, default=_ROOT / "results" / "ablations" / "axes")
    args = p.parse_args(argv)

    df = pd.read_csv(args.results)
    fam = df.drop_duplicates("model").set_index("model")["family"]
    args.out.mkdir(parents=True, exist_ok=True)

    sev = effective_severity(df)
    print(f"\n=== {args.label}: effective severity per axis ===")
    print(sev.round(4).to_string(index=False))
    if "spread" in sev.attrs:
        print(f"  worst/best axis ratio in mean AUC drop: {sev.attrs['spread']:.1f}x "
              "-- one unit of lambda is not one unit of severity")

    env = compute_envelopes(df, group_keys=_keys(df))
    per_axis = env.pivot_table(index="model", columns="shift_axis", values="rho")
    print(f"\n=== {args.label}: AURE per axis ===")
    print(per_axis.round(4).to_string())
    print("  margin per axis:")
    for axis in per_axis.columns:
        print(f"    {axis:22s} {margin(per_axis[axis].dropna(), fam):+.4f}")
    print(f"    {'POOLED (published AURE)':22s} "
          f"{margin(env.groupby('model').rho.mean(), fam):+.4f}")

    a = audc(df)
    pooled = a.groupby("model").audc.mean().sort_values(ascending=False)
    print(f"\n=== {args.label}: AUDC (threshold-free retention) ===")
    print(pooled.round(4).to_string())
    print(f"  margin: {margin(pooled, fam):+.4f}")
    a_axis = a.pivot_table(index="model", columns="shift_axis", values="audc")
    print("  margin per axis:")
    for axis in a_axis.columns:
        print(f"    {axis:22s} {margin(a_axis[axis].dropna(), fam):+.4f}")

    sev.to_csv(args.out / f"{args.label}_effective_severity.csv", index=False)
    per_axis.to_csv(args.out / f"{args.label}_aure_per_axis.csv")
    a_axis.to_csv(args.out / f"{args.label}_audc_per_axis.csv")
    print(f"\n[axes] wrote 3 tables to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
