"""Calibration figure for the reviewer's 4.7.

Three panels, each answering a different part of the objection that a single 15-bin
ECE cannot carry the paper's central argument:

  A. Do the calibration measures agree at zero shift? If the "ICL is better
     calibrated out of the box" claim only holds for equal-width top-label ECE, it
     is a binning artefact. Plotted as four measures side by side per model.
  B. Brier score against severity. A strictly proper scoring rule, so it cannot be
     satisfied by uninformative-but-calibrated output the way ECE can.
  C. Calibration slope against severity. Direction rather than magnitude: slope < 1
     is over-confidence, the boosted-tree failure mode; > 1 is under-confidence.

Reads a shift_results.csv carrying the calibration columns and writes
`results/ablations/figures/calibration.png`.

Usage::

    python experiments/make_calibration_figures.py \
        --results results/external/tables_2axis_multiseed_primary/shift_results_complete5.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Springer asks for 600 dpi on combination art; 170 is fine for reading on screen.
# Set TICE_FIG_DPI=600 before running to regenerate at submission resolution.
FIG_DPI = int(os.environ.get("TICE_FIG_DPI", "170"))

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tice.figio import save_figure  # noqa: E402
from tice.models.registry import get_model_spec  # noqa: E402

# Categorical slots 1-3 of the validated palette, reused from the ablation figure so
# the two share one visual language; family colour distinguishes ICL from baselines.
_FAMILY = {"icl": "#2a78d6", "gbdt": "#eb6834", "linear": "#1baf7a"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
MEASURES = [("ece", "ECE\n(equal-width)"), ("ece_adaptive", "ECE\n(equal-mass)"),
            ("ece_classwise", "ECE\n(classwise)"), ("brier", "Brier\nscore")]


def _family(model: str) -> str:
    try:
        return get_model_spec(model).family
    except KeyError:
        return "linear"


def _style(model: str) -> dict:
    fam = _family(model)
    return dict(color=_FAMILY.get(fam, MUTED), lw=2.4 if fam == "icl" else 1.5,
                ls="-" if fam == "icl" else "--", marker="o", markersize=4)


def _tidy(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.8)


def panel_agreement(ax: plt.Axes, clean: pd.DataFrame) -> None:
    models = clean.brier.sort_values().index.tolist()
    width = 0.8 / len(MEASURES)
    for j, (col, label) in enumerate(MEASURES):
        xs = [i + (j - 1.5) * width for i in range(len(models))]
        ax.bar(xs, clean.loc[models, col], width=width * 0.9, label=label,
               color=plt.get_cmap("Blues")(0.35 + 0.18 * j), edgecolor="none")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8.5)
    ax.set_title("A. Do the measures agree at zero shift?", fontsize=11, color=INK,
                 loc="left")
    ax.set_ylabel("calibration error / Brier", fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, labelcolor=INK2)
    _tidy(ax)


def panel_curve(ax: plt.Axes, df: pd.DataFrame, col: str, title: str,
                ylabel: str, ref: float | None = None) -> None:
    piv = df.pivot_table(index="shift_lambda", columns="model", values=col)
    for model in piv.columns:
        ax.plot(piv.index, piv[model], label=model, **_style(model))
    if ref is not None:
        ax.axhline(ref, color="#c3c2b7", lw=1.2, ls=":", zorder=1)
        ax.text(piv.index[-1], ref, " perfect", fontsize=7.5, color=MUTED, va="center")
    ax.set_xlabel("shift severity $\\lambda$", fontsize=9, color=INK2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2)
    _tidy(ax)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path,
                   default=_ROOT / "results" / "ablations" / "figures" / "calibration.png")
    args = p.parse_args(argv)

    df = pd.read_csv(args.results)
    needed = [c for c, _ in MEASURES] + ["calib_slope"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"[calib-fig] {args.results} lacks {missing}; re-run with the current "
              "metrics module")
        return 1
    ok = df[df.status == "ok"]
    clean = ok[ok.shift_lambda == 0].groupby("model")[needed].mean()

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
    panel_agreement(axes[0], clean)
    panel_curve(axes[1], ok, "brier", "B. Brier score vs severity",
                "Brier score (lower is better)")
    panel_curve(axes[2], ok, "calib_slope", "C. Calibration slope vs severity",
                "slope  (<1 over-confident)", ref=1.0)
    fig.suptitle("Calibration beyond a single ECE", fontsize=12.5, color=INK,
                 x=0.006, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, args.out, dpi=FIG_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"[calib-fig] wrote {args.out}")
    print("\nclean-state ordering per measure (best first):")
    for col, label in MEASURES:
        order = " < ".join(clean[col].sort_values().index)
        print(f"  {label.replace(chr(10), ' '):24s} {order}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
