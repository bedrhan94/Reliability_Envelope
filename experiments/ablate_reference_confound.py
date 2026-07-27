"""Reference-confound ablation: does AURE measure shift tolerance or a clean head start?

The default failure rule compares every model to a *shared, per-dataset constant* --
the best clean GBDT utility -- and to two *absolute* calibration thresholds
(``ECE > tau_ece``, ``NLL_norm > tau_nll``). A model whose clean state already sits
near those thresholds can trip the rule at ``lambda=0``, and since ``rho`` requires an
unbroken pass-run from ``lambda=0``, such a cell scores ``rho=0`` before any shift is
applied. That is a clean-performance deficit, not a robustness measurement.

This script re-analyses an existing ``shift_results.csv`` -- no models are re-run --
under three variants:

``grid``
    The published rule, reproduced as a control.
``common_support``
    The published rule, but averaged only over cells where *every* model passes at
    ``lambda=0``. Same cells for every model, so the comparison is paired.
``self_ref``
    Each model is its own reference: it fails when it degrades by more than tau from
    *its own* clean state, on all three triggers
    (``U(l) < U(0)-tau_u``, ``ECE(l) > ECE(0)+tau_ece``, ``NLL(l) > NLL(0)+tau_nll``).
    Every model passes at ``lambda=0`` by construction, so no cell is decided in
    advance and the radius is pure degradation.

Usage::

    python experiments/ablate_reference_confound.py \
        --results results/external/tables_2axis_stratified_multiseed/shift_results.csv \
        --out results/ablations/external_multiseed
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

from tice.envelope.reliability import envelope_radius  # noqa: E402

TAU_U, TAU_ECE, TAU_NLL = 0.03, 0.10, 0.75


def _cell_keys(df: pd.DataFrame) -> list[str]:
    """Cell = one envelope: dataset x axis (x run seed, when the run has several)."""
    keys = ["dataset_id", "shift_axis"]
    if "base_seed" in df.columns and df["base_seed"].nunique() > 1:
        keys.append("base_seed")
    return keys


def _radius(g: pd.DataFrame, failed_col: str) -> float:
    return envelope_radius(g["shift_lambda"].tolist(), g[failed_col].astype(bool).tolist())


def add_self_ref_failure(df: pd.DataFrame, cells: list[str]) -> pd.DataFrame:
    """Re-base all three failure triggers on each model's own clean (lambda=0) state."""
    grp = cells + ["model"]
    clean = (
        df[df.shift_lambda == 0.0]
        .set_index(grp)[["utility", "ece", "nll_norm"]]
        .rename(columns=lambda c: f"clean_{c}")
    )
    out = df.join(clean, on=grp)
    bad_status = out["status"].ne("ok")
    out["failed_self"] = (
        bad_status
        | out["utility"].isna()
        | (out["utility"] < out["clean_utility"] - TAU_U)
        | (out["ece"] > out["clean_ece"] + TAU_ECE)
        | (out["nll_norm"] > out["clean_nll_norm"] + TAU_NLL)
    )
    return out


def run(results_csv: Path, out_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    cells = _cell_keys(df)
    models = sorted(df.model.unique())
    n_models = len(models)

    # --- cells where every model passes at lambda=0 (common support) -----------
    zero = df[df.shift_lambda == 0.0]
    passers = zero[~zero.failed.astype(bool)].groupby(cells).model.nunique()
    common = set(map(tuple, (passers[passers == n_models].index.to_frame().values)))
    all_cells = set(map(tuple, zero.groupby(cells).size().index.to_frame().values))
    print(f"cells={len(all_cells)}  common-support cells={len(common)} "
          f"({len(common) / max(len(all_cells), 1):.1%})")

    # --- lambda=0 failure rate per model (the headline diagnostic) -------------
    zero_fail = (
        zero.groupby("model").failed.mean().rename("lambda0_fail_rate").round(4)
    )

    df = add_self_ref_failure(df, cells)
    df["_cell"] = list(map(tuple, df[cells].values))

    rows = []
    for (model, cell), g in df.groupby(["model", "_cell"], sort=True):
        if g["status"].ne("ok").all():
            continue
        rows.append(
            {
                "model": model,
                "cell": cell,
                "in_common": cell in common,
                "rho_grid": _radius(g, "failed"),
                "rho_self": _radius(g, "failed_self"),
            }
        )
    per_cell = pd.DataFrame(rows)

    summary = (
        per_cell.groupby("model")
        .agg(
            aure_grid=("rho_grid", "mean"),
            aure_self=("rho_self", "mean"),
            n_cells=("rho_grid", "size"),
        )
        .join(
            per_cell[per_cell.in_common].groupby("model").rho_grid.mean().rename("aure_common")
        )
        .join(zero_fail)
        .reset_index()
        .sort_values("aure_grid", ascending=False)
    )
    summary["n_common"] = len(common)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_cell.to_csv(out_dir / "ablation_per_cell.csv", index=False)
    summary.to_csv(out_dir / "ablation_summary.csv", index=False)

    cols = ["model", "lambda0_fail_rate", "aure_grid", "aure_common", "aure_self", "n_cells"]
    print(summary[cols].round(4).to_string(index=False))
    _print_margins(summary)
    print(f"wrote 2 tables to {out_dir}")
    return summary


def _print_margins(s: pd.DataFrame) -> None:
    icl = ["tabicl", "tabpfn_client"]
    gbdt = [
        "catboost", "xgboost", "hist_gbdt",
        "catboost_tuned", "xgboost_tuned",
        "catboost_cal", "xgboost_cal", "hist_gbdt_cal",
    ]
    idx = s.set_index("model")
    for variant in ("aure_grid", "aure_common", "aure_self"):
        if variant not in idx or idx[variant].isna().all():
            continue
        have_i = [m for m in icl if m in idx.index]
        have_g = [m for m in gbdt if m in idx.index]
        if not have_i or not have_g:
            continue
        lo = idx.loc[have_i, variant].min()
        hi = idx.loc[have_g, variant].max()
        print(f"  {variant:<12} min-ICL {lo:.4f}  best-GBDT {hi:.4f}  margin {lo - hi:+.4f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    run(args.results, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
