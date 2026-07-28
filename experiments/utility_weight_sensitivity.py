"""Is the ICL lead an artefact of how much the utility weights calibration?

The circularity objection, stated plainly: the reliability utility is

    U = 0.35*AUC + 0.15*Acc - 0.30*NLL_norm - 0.20*ECE

so **half its mass is calibration**, the ICL models' clean ECE is ~2.5x better than the
boosted trees', and the paper concludes that the ICL lead is a calibration advantage. A
referee is entitled to ask whether the conclusion was assumed by the scoring choice. The
tau thresholds got a 336-point sweep (``sensitivity.py``); these four weights got nothing.

This sweeps one interpretable axis -- the **calibration share** s, the total weight on
(NLL_norm, ECE) -- holding the within-group ratios of the published weights fixed:

    NLL = 0.6*s        ECE = 0.4*s          (published 0.30 : 0.20)
    AUC = 0.7*(1-s)    Acc = 0.3*(1-s)      (published 0.35 : 0.15)

so s=0.5 reproduces the published weights exactly. For each s the utility, the per-dataset
GBDT reference, the failure flags, the envelopes and AURE are recomputed from a stored
``shift_results.csv`` -- no models are re-run.

Note what does *not* move: ``tau_ece`` and ``tau_nll`` are absolute triggers independent of
the weights, so only the ``utility_gap`` criterion responds to s. That is the point -- it
isolates the weighting choice from the thresholds already swept elsewhere.

Usage::

    python experiments/utility_weight_sensitivity.py \
        --results results/external/tables_2axis_seed42_merged/shift_results.csv \
        --out results/ablations/external_44
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

from tice.config import Thresholds  # noqa: E402
from tice.envelope.reliability import compute_envelopes  # noqa: E402

TH = Thresholds()
ICL = ("tabicl", "tabpfn_client")
SHARES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _group_keys(df: pd.DataFrame) -> tuple[str, ...]:
    keys = ("model", "dataset_id", "shift_axis")
    if "base_seed" in df.columns and df["base_seed"].nunique() > 1:
        keys += ("base_seed",)
    return keys


def rescore_at_share(df: pd.DataFrame, s: float, reference_models: list[str]) -> pd.DataFrame:
    """Recompute utility / reference / failed with calibration share ``s``."""
    w_nll, w_ece = 0.6 * s, 0.4 * s
    w_auc, w_acc = 0.7 * (1 - s), 0.3 * (1 - s)
    d = df.copy()
    d["utility"] = (
        w_auc * d["auc"] + w_acc * d["accuracy"] - w_nll * d["nll_norm"] - w_ece * d["ece"]
    )
    d.loc[d[["auc", "accuracy", "nll_norm", "ece"]].isna().any(axis=1), "utility"] = np.nan

    # The reference is per dataset *and per run seed* -- a multiseed table pooled by
    # dataset alone takes the max over seeds, inflating the bar and changing every
    # failure flag. The s=0.5 guard in `sweep` exists to catch exactly this.
    ref_keys = ["dataset_id"]
    if "base_seed" in d.columns and d["base_seed"].nunique() > 1:
        ref_keys.append("base_seed")
    clean = d[(d.shift_lambda == 0.0) & (d.status == "ok") & d.model.isin(reference_models)]
    ref = clean.groupby(ref_keys).utility.max().rename("_ref")
    d = d.drop(columns=["reference_utility"], errors="ignore").join(ref, on=ref_keys)
    d = d.rename(columns={"_ref": "reference_utility"})

    ok = d["status"].eq("ok") & d["utility"].notna()
    d["failed"] = (~ok) | (
        (d["reference_utility"].notna() & (d["utility"] < d["reference_utility"] - TH.tau_utility))
        | (d["ece"].notna() & (d["ece"] > TH.tau_ece))
        | (d["nll_norm"].notna() & (d["nll_norm"] > TH.tau_nll))
    )
    return d


def sweep(df: pd.DataFrame, reference_models: list[str]) -> pd.DataFrame:
    keys = _group_keys(df)
    fam = df.drop_duplicates("model").set_index("model")["family"]
    gbdt = [m for m in fam.index if fam[m] == "gbdt"]
    icl = [m for m in fam.index if m in ICL]
    # s=0.5 is the published weighting, so it must reproduce the stored `failed` column
    # exactly. If it does not, the rescoring has diverged from the pipeline (reference
    # grouping, weights, or trigger order) and every row of the sweep is suspect.
    check = rescore_at_share(df, 0.5, reference_models)
    mismatch = int((check["failed"].astype(bool) != df["failed"].astype(bool)).sum())
    if mismatch:
        raise SystemExit(
            f"[weight-sweep] rescoring at the published share 0.50 disagrees with the "
            f"stored `failed` column on {mismatch}/{len(df)} rows -- fix before trusting "
            f"the sweep."
        )

    rows = []
    for s in SHARES:
        d = rescore_at_share(df, s, reference_models)
        aure = compute_envelopes(d, group_keys=keys).groupby("model").rho.mean()
        zero = d[d.shift_lambda == 0.0].groupby("model").failed.mean()
        rec = {"calibration_share": s,
               "margin": float(aure[icl].min() - aure[gbdt].max()),
               "lambda0_fail_gbdt_max": float(zero[gbdt].max()),
               "lambda0_fail_icl_max": float(zero[icl].max())}
        for m, a in aure.items():
            rec[f"aure_{m}"] = float(a)
        rec["order"] = "|".join(aure.sort_values(ascending=False).index)
        rows.append(rec)
    return pd.DataFrame(rows)


def verdict(s: pd.DataFrame) -> None:
    pub = s[s.calibration_share == 0.5].iloc[0]
    print(f"published weights (share 0.50): margin {pub.margin:+.4f}")
    print(f"ICL ahead in {int((s.margin > 0).sum())}/{len(s)} settings; "
          f"margin range [{s.margin.min():+.4f}, {s.margin.max():+.4f}]")
    flips = s[np.sign(s.margin) != np.sign(pub.margin)]
    if len(flips):
        lo, hi = flips.calibration_share.min(), flips.calibration_share.max()
        print(f"  sign differs from the published setting at shares {lo:.1f}-{hi:.1f}")
    else:
        print("  the sign of the margin never differs from the published setting")
    zero_free = s[s.lambda0_fail_gbdt_max == 0.0]
    print(f"shares where NO gbdt fails at lambda=0: "
          f"{list(zero_free.calibration_share) if len(zero_free) else 'none'}")
    print(f"distinct model orderings across the sweep: {s.order.nunique()}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference-models", type=str, default="xgboost,catboost")
    args = p.parse_args(argv)

    df = pd.read_csv(args.results)
    refs = [m.strip() for m in args.reference_models.split(",") if m.strip()]
    refs = [m for m in refs if m in set(df.model)] or sorted(
        df[df.family == "gbdt"].model.unique()
    )
    s = sweep(df, refs)
    args.out.mkdir(parents=True, exist_ok=True)
    s.to_csv(args.out / "utility_weight_sensitivity.csv", index=False)

    cols = ["calibration_share", "margin", "lambda0_fail_gbdt_max", "lambda0_fail_icl_max"]
    print(f"reference models: {refs}")
    print(s[cols].round(4).to_string(index=False))
    verdict(s)
    print(f"wrote {args.out / 'utility_weight_sensitivity.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
