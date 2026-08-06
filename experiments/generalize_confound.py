"""Generalise the reference confound from AURE to a *class* of reliability summaries.

The paper shows that AURE's published ordering is driven by clean-state admissibility
rather than shift tolerance (``decompose_aure.py``), and that matching the reference
reverses it (``ablate_reference_confound.py``). A Q1 reviewer's fair next question is
whether that is a property of *AURE specifically* or of a *class* of reliability
summaries. ``metric_proposal.md`` asserts the latter but only demonstrates it on AURE.

This script settles it by re-analysis -- no models are re-run. From the same stored
per-condition ``shift_results.csv`` it computes a family of reliability summaries laid
out on a 2x2 of two orthogonal design choices, and shows the ICL advantage is localised
to one cell of that grid:

    reference policy x  what the summary measures
    -----------------   --------------------------------
                        calibration-loaded      discrimination-only
    absolute bar        aure / robust_fraction  abs_auc_radius
                        abs_utility_area
                        abs_ece_radius
    self-referenced     self_ref_radius         auc_retention
                        utility_drop

Prediction (the paper's dual thesis made falsifiable):
  * calibration-loaded + absolute bar  -> ICL leads   (the confound: a clean-calibration
                                                        head start charged as robustness)
  * calibration-loaded + self-referenced -> reverses  (remove the head start, lead goes)
  * discrimination-only (either policy) -> small, sign-stable ICL lead (mechanism E1,
                                                        label-noise ranking stability --
                                                        real, and no confound explains it)

If that pattern holds, the confound is not a property of AURE but of *absolute-bar,
calibration-loaded* reliability summaries as a class, and there is a residual genuine
ICL advantage on discrimination that survives every reference policy.

Usage::

    python experiments/generalize_confound.py \
        --results results/external/tables_2axis_seed42_merged/shift_results.csv \
        --out results/ablations/metric_class/external_primary
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

from tice.envelope.reliability import envelope_radius  # noqa: E402

TAU_U, TAU_ECE, TAU_NLL, TAU_AUC = 0.03, 0.10, 0.75, 0.05
ICL_MODELS = ("tabicl", "tabpfn", "tabpfn_client")

# summary -> (reference_policy, measured_quantity, role). ``role`` is what the summary
# actually isolates once the data spoke: the ICL advantage decomposes into an
# admissibility head start (the confound), a genuine discrimination-robustness
# mechanism, and -- crucially -- *no* calibration-tolerance advantage at all.
# Expected ICL sign per role: admissibility -> lead; calibration_tolerance -> not lead;
# discrimination -> lead. larger = more reliable for every summary.
SUMMARY_TAGS = {
    "aure": ("absolute", "combined", "admissibility"),
    "robust_fraction": ("absolute", "combined", "admissibility"),
    "abs_utility_area": ("absolute", "combined", "admissibility"),
    "abs_ece_radius": ("absolute", "calibration", "calibration_tolerance"),
    "self_ref_radius": ("self_ref", "combined", "calibration_tolerance"),
    "utility_drop": ("self_ref", "combined", "calibration_tolerance"),
    "abs_auc_radius": ("absolute", "discrimination", "discrimination"),
    "auc_retention": ("self_ref", "discrimination", "discrimination"),
}
SUMMARIES = list(SUMMARY_TAGS)
# The discrimination bar (abs_auc_radius) uses an absolute AUC-drop threshold; a referee
# rightly asks whether the discrimination lead is an artefact of one arbitrary choice, so
# we sweep it. The paper's own thesis is that arbitrary absolute thresholds are dangerous.
TAU_AUC_GRID = (0.02, 0.05, 0.10, 0.15)


def _cell_keys(df: pd.DataFrame) -> list[str]:
    keys = ["dataset_id", "shift_axis"]
    if "base_seed" in df.columns and df["base_seed"].nunique() > 1:
        keys.append("base_seed")
    return keys


def _add_self_ref(df: pd.DataFrame, cells: list[str]) -> pd.DataFrame:
    """Re-base the three failure triggers on each model's own clean (lambda=0) state."""
    grp = cells + ["model"]
    clean = (
        df[df.shift_lambda == 0.0]
        .set_index(grp)[["utility", "ece", "nll_norm"]]
        .rename(columns=lambda c: f"clean_{c}")
    )
    out = df.join(clean, on=grp)
    bad = out["status"].ne("ok")
    out["failed_self"] = (
        bad
        | out["utility"].isna()
        | (out["utility"] < out["clean_utility"] - TAU_U)
        | (out["ece"] > out["clean_ece"] + TAU_ECE)
        | (out["nll_norm"] > out["clean_nll_norm"] + TAU_NLL)
    )
    return out


def _add_shared_auc_ref(df: pd.DataFrame, cells: list[str]) -> pd.DataFrame:
    """Per cell, the best clean AUC across models -- a shared, absolute AUC bar."""
    clean = df[df.shift_lambda == 0.0]
    ref = clean.groupby(cells)["auc"].max().rename("ref_auc_clean")
    return df.join(ref, on=cells)


def _summaries_for_cell(g: pd.DataFrame) -> dict | None:
    """All reliability summaries for one (model, dataset, axis[, seed]) envelope."""
    g = g.sort_values("shift_lambda")
    if g["status"].ne("ok").all():
        return None
    lam = g["shift_lambda"].to_numpy(dtype=float)
    failed = g["failed"].astype(bool).to_numpy()
    failed_self = g["failed_self"].astype(bool).to_numpy()
    util = g["utility"].to_numpy(dtype=float)
    auc = g["auc"].to_numpy(dtype=float)
    ece = g["ece"].to_numpy(dtype=float)
    u_ref = float(g["reference_utility"].iloc[0])
    auc_ref = float(g["ref_auc_clean"].iloc[0])

    clean = lam == 0.0
    u0 = util[clean][0] if clean.any() else np.nan
    auc0 = auc[clean][0] if clean.any() else np.nan
    post = lam > 0.0

    # absolute discrimination-only bar: fail when AUC drops below (best clean AUC - tau)
    auc_fail = np.isnan(auc) | (auc < auc_ref - TAU_AUC)
    # absolute calibration-only bar: fail when ECE exceeds the absolute threshold
    ece_fail = np.isnan(ece) | (ece > TAU_ECE)

    bar = u_ref - TAU_U
    abs_area = float(np.nanmean(np.clip(util - bar, 0.0, None))) if len(util) else np.nan
    auc_ret = (
        float(np.nanmean(auc[post] / auc0)) if post.any() and auc0 and auc0 == auc0 else np.nan
    )
    util_drop = -float(np.nanmean(util[post] - u0)) if post.any() and u0 == u0 else np.nan

    return {
        "aure": envelope_radius(lam.tolist(), failed.tolist()),
        "robust_fraction": float((~failed).mean()),
        "abs_utility_area": abs_area,
        "abs_ece_radius": envelope_radius(lam.tolist(), ece_fail.tolist()),
        "abs_auc_radius": envelope_radius(lam.tolist(), auc_fail.tolist()),
        "self_ref_radius": envelope_radius(lam.tolist(), failed_self.tolist()),
        "auc_retention": auc_ret,
        "utility_drop": util_drop,
        "admitted": bool(not failed[lam.argmin()]),
        "rho_grid": envelope_radius(lam.tolist(), failed.tolist()),
    }


def _abs_auc_margin(df: pd.DataFrame, fam: dict[str, str], tau_auc: float) -> float:
    """min-ICL - best-GBDT margin of the absolute-AUC radius at a given tau_auc."""
    rows = []
    for (model, _cell), g in df.groupby(["model", "_cell"], sort=True):
        g = g.sort_values("shift_lambda")
        if g["status"].ne("ok").all():
            continue
        lam = g["shift_lambda"].to_numpy(dtype=float)
        auc = g["auc"].to_numpy(dtype=float)
        auc_ref = float(g["ref_auc_clean"].iloc[0])
        fail = np.isnan(auc) | (auc < auc_ref - tau_auc)
        rows.append({"model": model, "rho": envelope_radius(lam.tolist(), fail.tolist())})
    per = pd.DataFrame(rows).groupby("model")["rho"].mean()
    icl = [m for m in per.index if fam.get(m) == "icl" or m in ICL_MODELS]
    gbdt = [m for m in per.index if fam.get(m) == "gbdt"]
    if not icl or not gbdt:
        return float("nan")
    return float(per[icl].min() - per[gbdt].max())


def _bootstrap_margin_ci(
    per_cell: pd.DataFrame, fam: dict[str, str], metric: str, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% bootstrap CI for the min-ICL - best-GBDT margin, resampling DATASETS.

    Datasets (not cells) are the resampling unit, so the CI reflects between-dataset
    variability -- the honest uncertainty on an aggregate margin over 44 datasets.
    """
    rng = np.random.default_rng(seed)
    icl = [m for m in per_cell.model.unique() if fam.get(m) == "icl" or m in ICL_MODELS]
    gbdt = [m for m in per_cell.model.unique() if fam.get(m) == "gbdt"]
    if not icl or not gbdt:
        return (float("nan"), float("nan"))
    dm = per_cell.groupby(["dataset_id", "model"])[metric].mean().unstack("model")
    datasets = dm.index.to_numpy()
    margins = []
    for _ in range(n_boot):
        sub = dm.loc[rng.choice(datasets, size=len(datasets), replace=True)]
        lo = sub[[m for m in icl if m in sub.columns]].mean().min()
        hi = sub[[m for m in gbdt if m in sub.columns]].mean().max()
        margins.append(lo - hi)
    return float(np.percentile(margins, 2.5)), float(np.percentile(margins, 97.5))


def run(results_csv: Path, out_dir: Path, seed: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    if seed is not None and "base_seed" in df.columns:
        df = df[df.base_seed == seed].reset_index(drop=True)
    cells = _cell_keys(df)
    df = _add_self_ref(df, cells)
    df = _add_shared_auc_ref(df, cells)
    df["_cell"] = list(map(tuple, df[cells].values))
    fam = df.drop_duplicates("model").set_index("model")["family"].to_dict()

    rows = []
    for (model, cell), g in df.groupby(["model", "_cell"], sort=True):
        s = _summaries_for_cell(g)
        if s is None:
            continue
        s.update({"model": model, "cell": cell})
        rows.append(s)
    per_cell = pd.DataFrame(rows)
    per_cell["dataset_id"] = [c[0] for c in per_cell["cell"]]

    agg = per_cell.groupby("model")[SUMMARIES].mean()
    agg["n_cells"] = per_cell.groupby("model").size()
    agg["A_aure"] = per_cell.groupby("model")["admitted"].mean()
    agg["T_aure"] = per_cell[per_cell.admitted].groupby("model")["rho_grid"].mean()
    agg = agg.reset_index().sort_values("aure", ascending=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_cell.drop(columns=["cell"]).to_csv(out_dir / "metric_class_per_cell.csv", index=False)
    agg.to_csv(out_dir / "metric_class_summary.csv", index=False)
    verdict = _margins(agg, fam)
    # 95% bootstrap CI on each margin (resampling datasets) -> honest uncertainty
    ci = {m: _bootstrap_margin_ci(per_cell, fam, m) for m in SUMMARIES}
    verdict["ci95_lo"] = verdict["summary"].map(lambda m: ci[m][0])
    verdict["ci95_hi"] = verdict["summary"].map(lambda m: ci[m][1])
    verdict["sig"] = (verdict["ci95_lo"] > 0) | (verdict["ci95_hi"] < 0)
    verdict.to_csv(out_dir / "metric_class_margins.csv", index=False)

    # tau_AUC robustness of the discrimination-role margin
    tau_sweep = pd.DataFrame(
        [{"tau_auc": t, "abs_auc_margin": _abs_auc_margin(df, fam, t)} for t in TAU_AUC_GRID]
    )
    tau_sweep.to_csv(out_dir / "tau_auc_sweep.csv", index=False)

    print(f"\n=== {results_csv.parent.name}  (n_cells={int(agg.n_cells.max())}) ===")
    print(agg[["model", *SUMMARIES, "A_aure", "T_aure"]].round(4).to_string(index=False))
    print("\n  ICL - best-GBDT margin, 95% bootstrap CI over datasets (sig = CI excludes 0):")
    show = ["summary", "role", "margin", "ci95_lo", "ci95_hi", "sig"]
    print(verdict[show].round(4).to_string(index=False))
    print("\n  tau_AUC robustness of the discrimination margin (abs_auc_radius):")
    print("    " + "   ".join(f"tau={r.tau_auc}: {r.abs_auc_margin:+.4f}"
                              for r in tau_sweep.itertuples()))
    _print_headline(verdict)
    print(f"\n  wrote 4 tables to {out_dir}")
    return verdict


def _margins(agg: pd.DataFrame, fam: dict[str, str]) -> pd.DataFrame:
    idx = agg.set_index("model")
    icl = [m for m in idx.index if fam.get(m) == "icl" or m in ICL_MODELS]
    gbdt = [m for m in idx.index if fam.get(m) == "gbdt"]
    rows = []
    for metric in SUMMARIES:
        if metric not in idx or not icl or not gbdt:
            continue
        lo, hi = idx.loc[icl, metric].min(), idx.loc[gbdt, metric].max()
        ref_pol, measure, role = SUMMARY_TAGS[metric]
        row = {
            "summary": metric,
            "role": role,
            "reference": ref_pol,
            "measures": measure,
            "min_ICL": lo,
            "best_GBDT": hi,
            "margin": lo - hi,
            "ICL_leads": bool(lo - hi > 0),
        }
        if metric == "aure" and {"A_aure", "T_aure"} <= set(idx.columns):
            row["A_margin"] = idx.loc[icl, "A_aure"].min() - idx.loc[gbdt, "A_aure"].max()
            row["T_margin"] = idx.loc[icl, "T_aure"].min() - idx.loc[gbdt, "T_aure"].max()
        rows.append(row)
    return pd.DataFrame(rows)


def _print_headline(v: pd.DataFrame) -> None:
    adm = v[v.role == "admissibility"].ICL_leads
    calt = v[v.role == "calibration_tolerance"].ICL_leads
    disc = v[v.role == "discrimination"].ICL_leads
    print(
        f"\n  HEADLINE -- the ICL advantage decomposes into three roles:"
        f"\n    admissibility head start : ICL leads {int(adm.sum())}/{len(adm)}"
        f"   (the CONFOUND -- clean-state pass rate, reverses under reference-matching)"
        f"\n    calibration tolerance    : ICL leads {int(calt.sum())}/{len(calt)}"
        f"   (NO ICL advantage -- GBDTs tolerate calibration drift at least as well)"
        f"\n    discrimination robustness: ICL leads {int(disc.sum())}/{len(disc)}"
        f"   (the real MECHANISM E1 -- survives every reference policy)"
    )
    if adm.all() and not calt.any() and disc.all():
        print("  => ICL's 'reliability' lead = admissibility head start (confound) + genuine"
              "\n     discrimination robustness; zero calibration-tolerance advantage. [PASS]")
    else:
        print("  => pattern not clean on this basis; inspect margins.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)
    run(args.results, args.out, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
