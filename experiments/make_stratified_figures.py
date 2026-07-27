"""Figures for the external stratified multiseed run (Sprint 6).

Reads `results/external/tables_2axis_stratified_multiseed/` and writes four
figures into `results/external/figures_stratified_multiseed/`:

  * stratified_aure_ci.png                        -- AURE per model, seed + dataset 95% CI
  * stratified_label_noise_auc_drop_ci.png        -- mean AUC vs label-noise lambda, seed band
  * stratified_covariate_calibration_collapse_ci.png -- AUC vs NLL/ECE under covariate shift
  * stratified_icl_vs_best_gbdt_margin.png        -- min-ICL minus best-GBDT AURE, per seed

Each plot is skipped (with a message) if its inputs are absent. Nothing here
touches canonical or single-seed external results.

Usage::

    python experiments/make_stratified_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
_ICL = ["tabicl", "tabpfn_client"]
_GBDT = ["catboost", "xgboost", "hist_gbdt"]


def _family(m: str) -> str:
    try:
        return get_model_spec(m).family
    except KeyError:
        return "linear"


def _style(m: str) -> dict:
    fam = _family(m)
    return dict(color=_FAMILY_COLOR.get(fam, "#8d99ae"),
                lw=2.4 if fam == "icl" else 1.4, ls="-" if fam == "icl" else "--")


def plot_aure_ci(ms: pd.DataFrame, out: Path) -> bool:
    if ms is None or ms.empty:
        return False
    d = ms.sort_values("aure_mean", ascending=True)
    mean = d["aure_mean"]
    colors = [_FAMILY_COLOR.get(_family(m), "#8d99ae") for m in d["model"]]
    s = [(mean - d["seed_ci95_lo"]).clip(lower=0).to_numpy(), (d["seed_ci95_hi"] - mean).clip(lower=0).to_numpy()]
    dd = [(mean - d["dataset_ci95_lo"]).clip(lower=0).to_numpy(), (d["dataset_ci95_hi"] - mean).clip(lower=0).to_numpy()]
    fig, ax = plt.subplots(figsize=(8.5, 0.6 * len(d) + 1.6))
    ax.barh(d["model"], mean, color=colors, alpha=0.85)
    ax.errorbar(mean, list(d["model"]), xerr=dd, fmt="none", ecolor="#999", elinewidth=1.0, capsize=7, alpha=0.9)
    ax.errorbar(mean, list(d["model"]), xerr=s, fmt="none", ecolor="#111", elinewidth=2.2, capsize=3)
    ns = int(d["n_seeds"].max())
    nd = int(d["n_datasets"].max())
    ax.set_xlabel("AURE (2-axis) — higher = more shift-robust")
    ax.set_title(f"Stratified external ({nd} datasets, {ns} seeds): AURE — dark=seed CI, light=dataset CI")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def plot_label_noise_ci(sr: pd.DataFrame, out: Path) -> bool:
    ln = sr[(sr["shift_axis"] == "label_noise") & (sr["status"] == "ok")]
    if ln.empty:
        return False
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in sorted(ln["model"].unique(), key=lambda x: (_family(x) != "icl", x)):
        g = ln[ln["model"] == m]
        # per-seed mean AUC over datasets at each lambda -> band across seeds
        per = g.groupby(["base_seed", "shift_lambda"]).auc.mean().unstack()
        lam = per.columns.to_numpy()
        mean = per.mean(axis=0).to_numpy()
        lo, hi = per.min(axis=0).to_numpy(), per.max(axis=0).to_numpy()
        st = _style(m)
        ax.plot(lam, mean, marker="o", label=f"{m} ({_family(m)})", **st)
        ax.fill_between(lam, lo, hi, color=st["color"], alpha=0.12)
    ax.set_xlabel("label-noise severity λ")
    ax.set_ylabel("mean test AUC (band = seed min–max)")
    ax.set_title("Stratified external: label-noise ranking retention (3 seeds)")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def plot_covariate_ci(sr: pd.DataFrame, profiles: pd.DataFrame, out: Path) -> bool:
    num = set(profiles.loc[profiles["n_numeric"] > 0, "dataset_id"]) if profiles is not None else set(sr.dataset_id)
    cv = sr[(sr["shift_axis"] == "covariate_shift") & (sr["status"] == "ok") & (sr["dataset_id"].isin(num))]
    if cv.empty:
        return False
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.8))
    for m in sorted(cv["model"].unique(), key=lambda x: (_family(x) != "icl", x)):
        g = cv[cv["model"] == m]
        a = g.groupby("shift_lambda").auc.mean()
        n = g.groupby("shift_lambda").nll_norm.mean()
        axL.plot(a.index, a.values, marker="o", label=f"{m} ({_family(m)})", **_style(m))
        axR.plot(n.index, n.values, marker="s", **_style(m))
    axL.set_ylabel("mean AUC (discrimination)")
    axR.set_ylabel("mean NLL_norm (calibration)")
    for ax in (axL, axR):
        ax.set_xlabel("covariate shift severity λ")
        ax.spines[["top", "right"]].set_visible(False)
    axL.legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Stratified external: covariate shift — calibration degrades faster than discrimination")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def plot_margin_by_seed(aure_by_seed: pd.DataFrame, out: Path) -> bool:
    if aure_by_seed is None or aure_by_seed.empty:
        return False
    w = aure_by_seed.pivot_table(index="seed", columns="model", values="aure")
    margin = w[_ICL].min(axis=1) - w[_GBDT].max(axis=1)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = ["#2a9d8f" if v > 0 else "#e76f51" for v in margin.values]
    ax.bar([str(s) for s in margin.index], margin.values, color=colors, alpha=0.85)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xlabel("seed")
    ax.set_ylabel("min-ICL AURE  −  best-GBDT AURE")
    ax.set_title(f"ICL vs best-GBDT margin per seed (mean {margin.mean():+.3f})")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def _maybe(p: Path):
    return pd.read_csv(p) if p.exists() else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stratified multiseed figures.")
    p.add_argument("--tables", type=Path, default=_ROOT / "results" / "external" / "tables_2axis_stratified_multiseed")
    p.add_argument("--out", type=Path, default=_ROOT / "results" / "external" / "figures_stratified_multiseed")
    args = p.parse_args(argv)
    T = Path(args.tables)
    sr = _maybe(T / "shift_results.csv")
    ms = _maybe(T / "aure_multiseed.csv")
    abs_ = _maybe(T / "aure_by_seed.csv")
    prof = _maybe(T / "dataset_profiles.csv")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("stratified_aure_ci.png", lambda o: plot_aure_ci(ms, o)),
        ("stratified_label_noise_auc_drop_ci.png", lambda o: plot_label_noise_ci(sr, o) if sr is not None else False),
        ("stratified_covariate_calibration_collapse_ci.png", lambda o: plot_covariate_ci(sr, prof, o) if sr is not None else False),
        ("stratified_icl_vs_best_gbdt_margin.png", lambda o: plot_margin_by_seed(abs_, o)),
    ]
    for fname, fn in jobs:
        try:
            drew = fn(out / fname)
        except Exception as exc:  # noqa: BLE001
            drew = False
            print(f"[strat-fig] {fname}: FAILED {type(exc).__name__}: {str(exc)[:100]}")
        print(f"[strat-fig] {fname}: {'wrote' if drew else 'skipped (inputs absent)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
