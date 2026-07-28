"""Compare the grid-quantised AURE with the de-quantised (continuous) AURE.

Referee issue ④: rho is one of six grid values, so a whole-step difference (or
none) can be pure quantisation. This recomputes AURE from an existing
``shift_results.csv`` using :func:`continuous_envelope_radius` (interpolated
failure boundary; no models re-run) and reports how much resolution that buys
and whether any model ranking changes.

Usage::

    python experiments/continuous_aure.py            # reads results/tables
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

from tice.config import Thresholds  # noqa: E402
from tice.envelope.reliability import (  # noqa: E402
    compute_aure,
    compute_envelopes,
    continuous_envelope_radius,
)

_TH = Thresholds()


def continuous_envelopes(df: pd.DataFrame) -> pd.DataFrame:
    """One continuous rho per (model, dataset, axis) -- per run seed when there are several.

    Without ``base_seed`` in the grouping a multiseed table collapses every seed into
    one envelope, so the lambda sequence contains each value once per seed and the
    interpolated crossing is computed over duplicated points. Grouping on it matches
    ``compute_envelopes``' behaviour and keeps the quantised and continuous columns
    comparable.
    """
    keys = ["model", "dataset_id", "shift_axis"]
    if "base_seed" in df.columns and df["base_seed"].nunique() > 1:
        keys.append("base_seed")
    rows: list[dict] = []
    for values, g in df.groupby(keys, sort=True):
        model, ds, axis = values[0], values[1], values[2]
        ok = g[g["status"] == "ok"].sort_values("shift_lambda")
        if ok.empty:
            continue
        ref = float(ok["reference_utility"].iloc[0])
        rho = continuous_envelope_radius(
            ok["shift_lambda"].tolist(),
            ok["utility"].tolist(),
            ok["ece"].tolist(),
            ok["nll_norm"].tolist(),
            ref,
            tau_utility=_TH.tau_utility,
            tau_ece=_TH.tau_ece,
            tau_nll=_TH.tau_nll,
        )
        rows.append({"model": model, "dataset_id": ds, "shift_axis": axis, "rho": rho})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quantised vs continuous AURE.")
    parser.add_argument("--tables", type=Path, default=_ROOT / "results" / "tables")
    args = parser.parse_args(argv)

    shift_path = Path(args.tables) / "shift_results.csv"
    if not shift_path.exists():
        print(f"[continuous] no shift_results at {shift_path}", file=sys.stderr)
        return 1
    df = pd.read_csv(shift_path)

    # Both sides must group on the run seed when the table has several, or each seed's
    # lambda sweep is folded into one envelope and the two columns stop being comparable.
    group_keys = ("model", "dataset_id", "shift_axis")
    if "base_seed" in df.columns and df["base_seed"].nunique() > 1:
        group_keys += ("base_seed",)
    quant_env = compute_envelopes(df, group_keys=group_keys)
    quant = compute_aure(quant_env).set_index("model")["aure"]
    cont_env = continuous_envelopes(df)
    cont = cont_env.groupby("model")["rho"].mean()

    out = pd.DataFrame({"aure_quantised": quant, "aure_continuous": cont})
    out["delta"] = out["aure_continuous"] - out["aure_quantised"]
    out = out.sort_values("aure_continuous", ascending=False)

    pd.set_option("display.width", 160)
    print("=== AURE: grid-quantised vs de-quantised (continuous) ===")
    print(out.round(4).to_string())

    print("\n=== resolution (distinct rho values across all envelopes) ===")
    grid_rho = quant_env["rho"]
    print(f"  quantised : {grid_rho.nunique()} distinct values {sorted(grid_rho.unique())}")
    print(f"  continuous: {cont_env['rho'].round(4).nunique()} distinct values")

    print("\n=== does the ranking change? ===")
    rq = list(quant.sort_values(ascending=False).index)
    rc = list(cont.sort_values(ascending=False).index)
    print(f"  quantised order : {rq}")
    print(f"  continuous order: {rc}")
    print(f"  identical ranking: {rq == rc}")
    if {"tabicl", "tabpfn_client"} <= set(quant.index):
        print(
            f"  tabicl-tabpfn gap: quantised={quant['tabicl']-quant['tabpfn_client']:+.4f}"
            f"  continuous={cont['tabicl']-cont['tabpfn_client']:+.4f}"
        )
    out.to_csv(Path(args.tables) / "aure_continuous.csv")
    print(f"\n[continuous] wrote {args.tables/'aure_continuous.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
