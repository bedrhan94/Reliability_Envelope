"""Does a threshold regime exist where nobody fails before the shift starts?

``sensitivity.py`` sweeps the failure thresholds and asks whether the AURE *ranking*
survives. This asks the prior question raised by ``ablate_reference_confound.py``:
the published thresholds make xgboost/hist_gbdt/logreg fail at ``lambda=0`` on
30-45% of cells, which forces ``rho=0`` there before any shift is applied. If that
is an artefact of where the thresholds sit, some other setting should remove it.

For each point of the same 336-point tau grid this records

* the ``lambda=0`` failure rate per model,
* AURE per model and the min-ICL vs best-GBDT margin,
* the same margin restricted to *common support* (cells where every model passes
  at ``lambda=0``), which is the confound-free comparison,

and reports what happens in the **clean-start regime** -- the grid points where no
model fails at ``lambda=0`` at all. If the ICL lead holds there, the tier claim is
safe and the confound is a threshold artefact; if it inverts, the published lead
depends on GBDTs being pre-failed.

Usage::

    python experiments/tau_lambda0_sensitivity.py \
        --results results/external/tables_2axis_stratified_multiseed/shift_results.csv \
        --out results/ablations/external_multiseed
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sensitivity import _GRIDS, _recompute_failed  # noqa: E402

from tice.config import Thresholds  # noqa: E402
from tice.envelope.reliability import envelope_radius  # noqa: E402

ICL = ("tabicl", "tabpfn_client")
GBDT = ("catboost", "xgboost", "hist_gbdt", "catboost_tuned", "xgboost_tuned")


def _cells(df: pd.DataFrame) -> list[str]:
    keys = ["dataset_id", "shift_axis"]
    if "base_seed" in df.columns and df["base_seed"].nunique() > 1:
        keys.append("base_seed")
    return keys


def sweep(df: pd.DataFrame) -> pd.DataFrame:
    cells = _cells(df)
    df = df.copy()
    df["_cell"] = list(map(tuple, df[cells].values))
    n_models = df.model.nunique()
    icl = [m for m in ICL if m in set(df.model)]
    gbdt = [m for m in GBDT if m in set(df.model)]
    groups = list(df.groupby(["model", "_cell"], sort=True))
    zero_mask = df.shift_lambda == 0.0

    rows = []
    for tu, te, tn in product(_GRIDS["tau_utility"], _GRIDS["tau_ece"], _GRIDS["tau_nll"]):
        th = Thresholds(tau_utility=tu, tau_ece=te, tau_nll=tn)
        df["failed"] = _recompute_failed(df, th)

        z = df[zero_mask]
        zero_rate = z.groupby("model").failed.mean()
        passers = z[~z.failed].groupby("_cell").model.nunique()
        common = set(passers[passers == n_models].index)

        rec = {"tau_utility": tu, "tau_ece": te, "tau_nll": tn,
               "lambda0_fail_max": float(zero_rate.max()),
               "lambda0_fail_mean": float(zero_rate.mean()),
               "n_common": len(common)}
        rho_all, rho_common = {}, {}
        for (model, cell), g in groups:
            if g["status"].ne("ok").all():
                continue
            r = envelope_radius(g["shift_lambda"].tolist(),
                                df.loc[g.index, "failed"].astype(bool).tolist())
            rho_all.setdefault(model, []).append(r)
            if cell in common:
                rho_common.setdefault(model, []).append(r)
        aure = {m: sum(v) / len(v) for m, v in rho_all.items() if v}
        aure_c = {m: sum(v) / len(v) for m, v in rho_common.items() if v}
        for m, a in aure.items():
            rec[f"aure_{m}"] = a
        if icl and gbdt:
            rec["margin"] = min(aure[m] for m in icl) - max(aure[m] for m in gbdt)
            if aure_c:
                rec["margin_common"] = (
                    min(aure_c[m] for m in icl) - max(aure_c[m] for m in gbdt)
                )
        rows.append(rec)
    return pd.DataFrame(rows)


def verdict(s: pd.DataFrame) -> None:
    n = len(s)
    clean = s[s.lambda0_fail_max == 0.0]
    print(f"grid points: {n}")
    print(f"points with ZERO lambda=0 failures for every model: {len(clean)} "
          f"({len(clean) / n:.1%})")
    if len(clean):
        print(f"  in that clean-start regime: margin mean {clean.margin.mean():+.4f}, "
              f"ICL ahead in {(clean.margin > 0).mean():.1%} of them")
        best = clean.loc[clean.margin.idxmax()]
        print(f"  most ICL-favourable clean-start point: tau_u={best.tau_utility} "
              f"tau_ece={best.tau_ece} tau_nll={best.tau_nll} margin {best.margin:+.4f}")
    print(f"whole grid: ICL ahead in {(s.margin > 0).mean():.1%} of points "
          f"(mean margin {s.margin.mean():+.4f})")
    if "margin_common" in s:
        c = s.dropna(subset=["margin_common"])
        print(f"common support: ICL ahead in {(c.margin_common > 0).mean():.1%} "
              f"(mean {c.margin_common.mean():+.4f}, median n_common {c.n_common.median():.0f})")
    corr = s[["lambda0_fail_mean", "margin"]].corr().iloc[0, 1]
    print(f"corr(mean lambda=0 fail rate, ICL margin) = {corr:+.3f}  "
          "(positive => the ICL lead grows as more baselines are pre-failed)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    df = pd.read_csv(args.results)
    s = sweep(df)
    args.out.mkdir(parents=True, exist_ok=True)
    s.to_csv(args.out / "tau_lambda0_sensitivity.csv", index=False)
    verdict(s)
    print(f"wrote {args.out / 'tau_lambda0_sensitivity.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
