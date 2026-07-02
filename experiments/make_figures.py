"""Render the headline reliability figures from the shift-stress outputs.

Reads ``aure_summary.csv`` and writes two PNGs into ``results/figures/``:

* ``aure_overall.png``  -- AURE per model, ranked, coloured by model family.
* ``aure_by_axis.png``  -- per-shift-axis AURE heatmap (models x axes), which is
  where the "where does each model break" story lives.

Usage::

    python experiments/make_figures.py            # reads results/tables
    python experiments/make_figures.py --tables results/tables_backup_20260628
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tice.models.registry import get_model_spec  # noqa: E402

# Family -> colour. ICL foundation models pop; GBDTs share a family hue.
_FAMILY_COLOR = {"icl": "#d1495b", "gbdt": "#4c72b0", "linear": "#8d99ae"}


def _family(model: str) -> str:
    try:
        return get_model_spec(model).family
    except KeyError:
        return "linear"


def _axis_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("aure_")]


def plot_overall(df: pd.DataFrame, out: Path) -> None:
    d = df.sort_values("aure", ascending=True)  # ascending -> best on top in barh
    colors = [_FAMILY_COLOR.get(_family(m), "#8d99ae") for m in d["model"]]

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(d) + 1.5))
    bars = ax.barh(d["model"], d["aure"], color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xlabel("AURE  (mean reliability-envelope radius ρ  —  higher = more shift-robust)")
    ax.set_title("Reliability under distribution shift (AURE per model)")
    ax.set_xlim(0, float(d["aure"].max()) * 1.18)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c) for c in _FAMILY_COLOR.values()
    ]
    ax.legend(handles, [f"{k} models" for k in _FAMILY_COLOR], loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_overall_ci(multiseed: pd.DataFrame, out: Path) -> None:
    """AURE per model with 95% CI error bars from the multi-seed run.

    This is the honest version of ``plot_overall``: bars whose CIs overlap are
    not distinguishable, so third-decimal rankings from a single seed vanish.
    """
    d = multiseed.sort_values("aure_mean", ascending=True)
    mean = d["aure_mean"]
    colors = [_FAMILY_COLOR.get(_family(m), "#8d99ae") for m in d["model"]]
    s_lo = (mean - d["seed_ci95_lo"]).clip(lower=0).to_numpy()
    s_hi = (d["seed_ci95_hi"] - mean).clip(lower=0).to_numpy()
    d_lo = (mean - d["dataset_ci95_lo"]).clip(lower=0).to_numpy()
    d_hi = (d["dataset_ci95_hi"] - mean).clip(lower=0).to_numpy()

    fig, ax = plt.subplots(figsize=(8.5, 0.6 * len(d) + 1.7))
    ax.barh(d["model"], mean, color=colors, alpha=0.85)
    # Wide light whisker = dataset-unit CI (generalisation); tight dark whisker =
    # seed-unit CI (within these datasets). Overlapping wide whiskers => the
    # ordering does not generalise beyond this dataset sample.
    ax.errorbar(mean, list(d["model"]), xerr=[d_lo, d_hi], fmt="none",
                ecolor="#999", elinewidth=1.0, capsize=7, alpha=0.9)
    ax.errorbar(mean, list(d["model"]), xerr=[s_lo, s_hi], fmt="none",
                ecolor="#111", elinewidth=2.2, capsize=3)
    n = int(d["n_seeds"].max())
    nd = int(d["n_datasets"].max())
    ax.set_xlabel("AURE  —  higher = more shift-robust")
    ax.set_title(
        f"{n}-seed AURE: dark = seed CI (within {nd} datasets), light = dataset CI (generalisation)"
    )
    ax.set_xlim(0, float(d["dataset_ci95_hi"].max()) * 1.12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in _FAMILY_COLOR.values()]
    ax.legend(handles, [f"{k} models" for k in _FAMILY_COLOR], loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_by_axis(df: pd.DataFrame, out: Path) -> None:
    axis_cols = _axis_columns(df)
    axes = [c.removeprefix("aure_") for c in axis_cols]
    d = df.sort_values("aure", ascending=False).reset_index(drop=True)
    mat = d[axis_cols].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(1.15 * len(axes) + 2.5, 0.6 * len(d) + 2))
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=0.40)
    ax.set_xticks(range(len(axes)), axes, rotation=30, ha="right")
    ax.set_yticks(range(len(d)), d["model"])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            txt = "–" if np.isnan(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="black")
    ax.set_title("Where each model breaks — AURE per shift axis (green = robust)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="axis AURE (ρ)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_calibration_collapse(
    shift_results: pd.DataFrame,
    out: Path,
    *,
    model: str = "tabpfn_client",
    dataset: str = "breast_cancer",
    axis: str = "covariate_shift",
) -> bool:
    """Diagnostic: under covariate shift, ranking (AUC) survives but calibration
    (NLL, ECE) collapses — which is what actually trips the failure rule.

    Twin y-axes over λ: left = AUC + utility, right = NLL_norm + ECE. Returns
    False (and draws nothing) if the requested cell is absent.
    """
    g = shift_results[
        (shift_results.model == model)
        & (shift_results.dataset_id == dataset)
        & (shift_results.shift_axis == axis)
        & (shift_results.status == "ok")
    ].sort_values("shift_lambda")
    if g.empty:
        return False

    lam = g["shift_lambda"].to_numpy()
    fig, axL = plt.subplots(figsize=(8, 5))
    axL.plot(lam, g["auc"], "-o", color="#2a9d8f", label="AUC (ranking)")
    axL.plot(lam, g["utility"], "-o", color="#264653", label="utility U")
    axL.axhline(0, color="#264653", lw=0.6, ls=":")
    axL.set_xlabel("covariate shift severity λ")
    axL.set_ylabel("AUC  /  utility U")
    axL.set_ylim(min(-2.0, float(g["utility"].min()) - 0.2), 1.1)

    axR = axL.twinx()
    axR.plot(lam, g["nll_norm"], "-s", color="#e76f51", label="NLL_norm")
    axR.plot(lam, g["ece"], "-^", color="#f4a261", label="ECE")
    axR.axhline(0.75, color="#e76f51", lw=0.7, ls="--")  # τ_nll
    axR.axhline(0.10, color="#f4a261", lw=0.7, ls="--")  # τ_ece
    axR.set_ylabel("NLL_norm  /  ECE  (dashed = failure thresholds)")

    # First λ the model is flagged failed, if any.
    failed = g[g["failed"].astype(bool)]
    if not failed.empty:
        lf = float(failed["shift_lambda"].min())
        axL.axvline(lf, color="grey", lw=0.8, ls="-.")
        axL.text(lf, 1.02, f"first fail λ={lf:g}", fontsize=8, color="grey", ha="left")

    lines = axL.get_lines()[:2] + axR.get_lines()[:2]
    axL.legend(lines, [ln.get_label() for ln in lines], loc="center left", frameon=False)
    axL.set_title(
        f"Covariate shift breaks calibration, not discrimination\n"
        f"({model} on {dataset}: AUC flat, NLL/ECE explode)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def plot_label_noise_robustness(shift_results: pd.DataFrame, out: Path) -> bool:
    """Cross-model diagnostic: mean test AUC vs label-noise severity λ.

    Label noise flips *training* labels, so the question is whether a model
    memorises the corrupted labels (ranking collapses) or resists them. ICL
    foundation models stay flat; GBDT/linear models fall off. Returns False if
    the axis is absent.
    """
    ln = shift_results[
        (shift_results.shift_axis == "label_noise") & (shift_results.status == "ok")
    ]
    if ln.empty:
        return False
    piv = ln.groupby(["model", "shift_lambda"]).auc.mean().unstack()

    # ICL first (solid, thick), then the rest — so the contrast reads at a glance.
    models = sorted(piv.index, key=lambda m: (_family(m) != "icl", m))
    markers = {"icl": "o", "gbdt": "s", "linear": "^"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in models:
        fam = _family(m)
        ax.plot(
            piv.columns,
            piv.loc[m],
            marker=markers.get(fam, "o"),
            color=_FAMILY_COLOR.get(fam, "#8d99ae"),
            lw=2.4 if fam == "icl" else 1.4,
            ls="-" if fam == "icl" else "--",
            label=f"{m} ({fam})",
        )
    ax.set_xlabel("label-noise severity λ  (fraction of train labels flipped)")
    ax.set_ylabel("mean test AUC across datasets")
    ax.set_title("Label noise: in-context models resist, GBDTs memorise the flipped labels")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render reliability figures.")
    parser.add_argument("--tables", type=Path, default=_ROOT / "results" / "tables")
    parser.add_argument("--out", type=Path, default=_ROOT / "results" / "figures")
    args = parser.parse_args(argv)

    tables = Path(args.tables)
    summary_path = tables / "aure_summary.csv"
    if not summary_path.exists():
        print(f"[figures] no AURE summary at {summary_path}", file=sys.stderr)
        return 1
    df = pd.read_csv(summary_path)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    overall = out_dir / "aure_overall.png"
    by_axis = out_dir / "aure_by_axis.png"
    plot_overall(df, overall)
    plot_by_axis(df, by_axis)
    print(f"[figures] wrote {overall}")
    print(f"[figures] wrote {by_axis}")

    multiseed_path = tables / "aure_multiseed.csv"
    if multiseed_path.exists():
        overall_ci = out_dir / "aure_overall_ci.png"
        plot_overall_ci(pd.read_csv(multiseed_path), overall_ci)
        print(f"[figures] wrote {overall_ci}")

    shift_path = tables / "shift_results.csv"
    if shift_path.exists():
        shift_results = pd.read_csv(shift_path)
        diag = out_dir / "covariate_calibration_collapse.png"
        if plot_calibration_collapse(shift_results, diag):
            print(f"[figures] wrote {diag}")
        noise = out_dir / "label_noise_robustness.png"
        if plot_label_noise_robustness(shift_results, noise):
            print(f"[figures] wrote {noise}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
