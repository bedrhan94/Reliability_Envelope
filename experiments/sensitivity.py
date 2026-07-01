"""Threshold sensitivity of AURE (referee issue: 7 arbitrary constants, no ablation).

The failure rule turns three thresholds -- ``tau_utility`` / ``tau_ece`` /
``tau_nll`` -- into the binary ``failed`` flag that AURE is built on. Those
numbers (0.03 / 0.10 / 0.75) are unjustified in the spec, so the whole result
could be an artefact of where they were set. This script re-derives AURE from an
existing ``shift_results.csv`` (no models are re-run: the stored utility / ece /
nll_norm / reference_utility fully determine ``failed``) as each threshold is
swept, and asks the only questions that matter:

* do model AURE *rankings* move, or just the absolute numbers?
* over a full grid, how often does the ICL-beats-GBDT headline survive, and how
  often does the tabicl-vs-tabpfn ordering flip?

Writes ``aure_sensitivity.csv`` and ``aure_sensitivity.png``; prints a verdict.

Usage::

    python experiments/sensitivity.py            # reads results/tables
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

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tice.config import Thresholds  # noqa: E402
from tice.envelope.reliability import compute_aure, compute_envelopes  # noqa: E402
from tice.models.registry import get_model_spec  # noqa: E402

_DEFAULT = Thresholds()  # 0.03 / 0.10 / 0.75
_FAMILY_COLOR = {"icl": "#d1495b", "gbdt": "#4c72b0", "linear": "#8d99ae"}
_GRIDS = {
    "tau_utility": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10],
    "tau_ece": [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25],
    "tau_nll": [0.30, 0.50, 0.75, 1.00, 1.25, 1.50],
}


def _family(model: str) -> str:
    try:
        return get_model_spec(model).family
    except KeyError:
        return "linear"


def _recompute_failed(df: pd.DataFrame, th: Thresholds) -> pd.Series:
    """Vectorised copy of ``evaluate_failure`` over the whole table."""
    ok = df["status"] == "ok"
    ref = df["reference_utility"]
    util_gap = ok & ref.notna() & (df["utility"] < ref - th.tau_utility)
    ece_fail = ok & df["ece"].notna() & (df["ece"] > th.tau_ece)
    nll_fail = ok & df["nll_norm"].notna() & (df["nll_norm"] > th.tau_nll)
    return (~ok) | util_gap | ece_fail | nll_fail


def _aure_for(df: pd.DataFrame, th: Thresholds) -> pd.Series:
    d = df.copy()
    d["failed"] = _recompute_failed(d, th)
    aure = compute_aure(compute_envelopes(d))
    return aure.set_index("model")["aure"]


def sweep_one_at_a_time(df: pd.DataFrame) -> pd.DataFrame:
    """Vary each threshold over its grid with the other two at default."""
    rows: list[dict] = []
    for name, grid in _GRIDS.items():
        for val in grid:
            kwargs = {"tau_utility": _DEFAULT.tau_utility, "tau_ece": _DEFAULT.tau_ece, "tau_nll": _DEFAULT.tau_nll}
            kwargs[name] = val
            aure = _aure_for(df, Thresholds(**kwargs))
            for model, a in aure.items():
                rows.append({"threshold": name, "value": val, "model": model, "aure": float(a)})
    return pd.DataFrame(rows)


def grid_stability(df: pd.DataFrame) -> dict:
    """Full Cartesian grid: how often do the headline claims survive?"""
    icl = [m for m in df["model"].unique() if _family(m) == "icl"]
    gbdt = [m for m in df["model"].unique() if _family(m) == "gbdt"]
    tops: dict[str, int] = {}
    icl_top2 = icl_dominates = tabicl_gt_tabpfn = n = 0
    for tu, te, tn in product(_GRIDS["tau_utility"], _GRIDS["tau_ece"], _GRIDS["tau_nll"]):
        aure = _aure_for(df, Thresholds(tau_utility=tu, tau_ece=te, tau_nll=tn))
        n += 1
        ranked = aure.sort_values(ascending=False)
        tops[ranked.index[0]] = tops.get(ranked.index[0], 0) + 1
        if set(ranked.index[:2]) == set(icl):
            icl_top2 += 1
        if icl and gbdt and aure[icl].min() > aure[gbdt].max():
            icl_dominates += 1
        if {"tabicl", "tabpfn_client"} <= set(aure.index) and aure["tabicl"] > aure["tabpfn_client"]:
            tabicl_gt_tabpfn += 1
    return {
        "n_grid": n,
        "top_model_counts": tops,
        "pct_icl_top2": 100 * icl_top2 / n,
        "pct_icl_dominates_gbdt": 100 * icl_dominates / n,
        "pct_tabicl_gt_tabpfn": 100 * tabicl_gt_tabpfn / n,
    }


def plot_sensitivity(sweep: pd.DataFrame, out: Path) -> None:
    names = list(_GRIDS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, name in zip(axes, names, strict=True):
        sub = sweep[sweep["threshold"] == name]
        for model, g in sub.groupby("model"):
            g = g.sort_values("value")
            fam = _family(model)
            ax.plot(g["value"], g["aure"], marker="o", color=_FAMILY_COLOR.get(fam, "#8d99ae"),
                    lw=2.2 if fam == "icl" else 1.3, ls="-" if fam == "icl" else "--", label=model)
        ax.axvline(getattr(_DEFAULT, name), color="grey", ls=":", lw=1)
        ax.set_xlabel(f"{name}  (dotted = default {getattr(_DEFAULT, name)})")
        ax.set_ylabel("AURE")
        ax.set_title(f"AURE vs {name}")
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("AURE threshold sensitivity (one-at-a-time; other two at default)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AURE threshold sensitivity analysis.")
    parser.add_argument("--tables", type=Path, default=_ROOT / "results" / "tables")
    parser.add_argument("--out", type=Path, default=_ROOT / "results")
    args = parser.parse_args(argv)

    shift_path = Path(args.tables) / "shift_results.csv"
    if not shift_path.exists():
        print(f"[sensitivity] no shift_results at {shift_path}", file=sys.stderr)
        return 1
    df = pd.read_csv(shift_path)

    sweep = sweep_one_at_a_time(df)
    (Path(args.tables) / "aure_sensitivity.csv").write_text(sweep.to_csv(index=False), encoding="utf-8")
    fig_dir = Path(args.out) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_sensitivity(sweep, fig_dir / "aure_sensitivity.png")

    stab = grid_stability(df)
    pd.set_option("display.width", 160)
    print(f"[sensitivity] full grid = {stab['n_grid']} threshold combinations")
    print(f"  ICL is the top-2 (both):        {stab['pct_icl_top2']:.0f}% of the grid")
    print(f"  every ICL AURE > every GBDT:    {stab['pct_icl_dominates_gbdt']:.0f}% of the grid")
    print(f"  tabicl AURE > tabpfn AURE:      {stab['pct_tabicl_gt_tabpfn']:.0f}% of the grid  <-- ordering stability")
    print(f"  #1 model across grid:           {stab['top_model_counts']}")
    print(f"[sensitivity] wrote {args.tables/'aure_sensitivity.csv'} and {fig_dir/'aure_sensitivity.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
