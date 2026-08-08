"""External-validation figures (skeleton; renders whatever external outputs exist).

Reads the external 2-axis run (`results/external/tables_2axis_seed42/`) and the
profile audit, and writes four figures into `results/external/figures/`:

  * external_aure_2axis.png                  -- AURE per model on the external suite
  * external_label_noise_auc_drop.png        -- mean test AUC vs label-noise λ per model
  * external_covariate_calibration_collapse.png -- AUC vs NLL_norm/ECE under covariate shift
  * dataset_profile_vs_failure.png           -- per-dataset reliability vs a dataset property

Each plot is skipped (with a message) if its inputs are absent, so this can be
run before/after the full external run. Nothing here touches canonical results.

Usage::

    python experiments/make_external_figures.py --tables results/external/tables_2axis_seed42
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Springer asks for 600 dpi on combination art; 150 is fine for reading on screen.
# Set TICE_FIG_DPI=600 before running to regenerate at submission resolution.
FIG_DPI = int(os.environ.get("TICE_FIG_DPI", "150"))

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
import pandas as pd  # noqa: E402

from tice.models.registry import get_model_spec  # noqa: E402

_FAMILY_COLOR = {"icl": "#d1495b", "gbdt": "#4c72b0", "linear": "#8d99ae"}


def _family(model: str) -> str:
    try:
        return get_model_spec(model).family
    except KeyError:
        return "linear"


def _style(model: str) -> dict:
    fam = _family(model)
    return dict(color=_FAMILY_COLOR.get(fam, "#8d99ae"),
               lw=2.4 if fam == "icl" else 1.4, ls="-" if fam == "icl" else "--")


def plot_aure_2axis(aure: pd.DataFrame, out: Path) -> bool:
    if aure is None or aure.empty:
        return False
    d = aure.sort_values("aure", ascending=True)
    colors = [_FAMILY_COLOR.get(_family(m), "#8d99ae") for m in d["model"]]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(d) + 1.5))
    bars = ax.barh(d["model"], d["aure"], color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xlabel("AURE (2 axes: label_noise + covariate_shift) — higher = more shift-robust")
    ax.set_title("External validation — AURE per model (2-axis)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    return True


def plot_label_noise_auc(shift: pd.DataFrame, out: Path) -> bool:
    ln = shift[(shift["shift_axis"] == "label_noise") & (shift["status"] == "ok")]
    if ln.empty:
        return False
    piv = ln.groupby(["model", "shift_lambda"]).auc.mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in sorted(piv.index, key=lambda x: (_family(x) != "icl", x)):
        ax.plot(piv.columns, piv.loc[m], marker="o", label=f"{m} ({_family(m)})", **_style(m))
    ax.set_xlabel("label-noise severity λ")
    ax.set_ylabel("mean test AUC across external datasets")
    ax.set_title("External: train-label noise — ICL vs GBDT ranking retention")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    return True


def plot_covariate_collapse(shift: pd.DataFrame, out: Path) -> bool:
    cv = shift[(shift["shift_axis"] == "covariate_shift") & (shift["status"] == "ok")]
    if cv.empty:
        return False
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.8))
    aucs = cv.groupby(["model", "shift_lambda"]).auc.mean().unstack()
    nlls = cv.groupby(["model", "shift_lambda"]).nll_norm.mean().unstack()
    for m in sorted(aucs.index, key=lambda x: (_family(x) != "icl", x)):
        axL.plot(aucs.columns, aucs.loc[m], marker="o", label=f"{m} ({_family(m)})", **_style(m))
        axR.plot(nlls.columns, nlls.loc[m], marker="s", **_style(m))
    axL.set_ylabel("mean AUC (discrimination)")
    axR.set_ylabel("mean NLL_norm (calibration)")
    for ax in (axL, axR):
        ax.set_xlabel("covariate shift severity λ")
        ax.spines[["top", "right"]].set_visible(False)
    axL.legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("External: covariate shift — discrimination holds while calibration degrades")
    fig.tight_layout()
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    return True


def plot_profile_vs_failure(envelopes: pd.DataFrame, profiles: pd.DataFrame, out: Path) -> bool:
    if envelopes is None or envelopes.empty or profiles is None or profiles.empty:
        return False
    per_ds = envelopes.groupby("dataset_id").rho.mean().rename("mean_rho").reset_index()
    m = per_ds.merge(profiles, on="dataset_id", how="inner")
    if m.empty:
        return False
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = m["n_features"].clip(lower=1)
    sc = ax.scatter(x, m["mean_rho"], c=m["class_imbalance_ratio"], cmap="viridis",
                    s=40, norm=matplotlib.colors.LogNorm())
    ax.set_xscale("log")
    ax.set_xlabel("n_features (log)")
    ax.set_ylabel("mean reliability-envelope radius ρ (across models/axes)")
    ax.set_title("External: does a dataset's profile predict reliability?")
    fig.colorbar(sc, ax=ax, label="class imbalance ratio (log)")
    for _, r in m.iterrows():
        ax.annotate(str(r["dataset_id"])[:10], (x[r.name], r["mean_rho"]), fontsize=6, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    return True


def _maybe(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="External-validation figures.")
    p.add_argument("--tables", type=Path, default=_ROOT / "results" / "external" / "tables_2axis_seed42")
    p.add_argument("--profiles", type=Path, default=_ROOT / "results" / "external" / "dataset_profiles_external.csv")
    p.add_argument("--out", type=Path, default=_ROOT / "results" / "external" / "figures")
    p.add_argument("--suffix", type=str, default="", help="appended before .png, e.g. _partial")
    args = p.parse_args(argv)

    tables = Path(args.tables)
    shift = _maybe(tables / "shift_results.csv")
    aure = _maybe(tables / "aure_summary.csv")
    env = _maybe(tables / "reliability_envelopes.csv")
    profiles = _maybe(Path(args.profiles))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("external_aure_2axis.png", lambda o: plot_aure_2axis(aure, o) if aure is not None else False),
        ("external_label_noise_auc_drop.png", lambda o: plot_label_noise_auc(shift, o) if shift is not None else False),
        ("external_covariate_calibration_collapse.png", lambda o: plot_covariate_collapse(shift, o) if shift is not None else False),
        ("dataset_profile_vs_failure.png", lambda o: plot_profile_vs_failure(env, profiles, o) if env is not None else False),
    ]
    for fname, fn in jobs:
        fname = fname.replace(".png", f"{args.suffix}.png")
        try:
            drew = fn(out / fname)
        except Exception as exc:  # noqa: BLE001
            drew = False
            print(f"[external-fig] {fname}: FAILED {type(exc).__name__}: {str(exc)[:100]}")
        print(f"[external-fig] {fname}: {'wrote' if drew else 'skipped (inputs absent)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
