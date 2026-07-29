"""Split AURE into clean-state admissibility and conditional shift tolerance.

Reviewer objection: the paper demonstrates that the published ordering is driven by
clean-state calibration, but never separates the two components, so a reader cannot
see how much of a model's AURE comes from being *admissible at all* versus from
*tolerating shift once admitted*.

The separation is exact rather than approximate. Because rho requires an unbroken
pass-run from lambda=0, any cell failing at lambda=0 contributes rho=0, so

    E[rho] = P(fail at 0)*0 + P(pass at 0)*E[rho | pass at 0]
           = A * T,      A = P(pass at lambda=0)          "clean-state admissibility"
                         T = E[rho | pass at lambda=0]    "conditional shift tolerance"

A is a property of the model's clean state and the failure bar; T is the quantity a
robustness claim is actually about. AURE multiplies them and reports the product,
which is why an ordering in AURE cannot be read as an ordering in shift tolerance.

The script verifies A*T == AURE to numerical tolerance on every basis, then reports
both factors and the ICL-vs-GBDT margin in each, so the reader can see which factor
carries the ordering.

Usage::

    python experiments/decompose_aure.py
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

from tice.envelope.reliability import compute_envelopes  # noqa: E402

ICL = ("tabicl", "tabpfn_client")

BASES = [
    ("pilot (5 ds, 6 axes)", "results/canonical/tables/shift_results.csv", None),
    ("external 44 ds, ICL capped", "results/external/tables_2axis_seed42_merged/shift_results.csv", None),
    ("external 44 ds, ICL uncapped",
     "results/external/tables_2axis_icl_uncapped_merged/shift_results_all44.csv", None),
    ("external subset 12x3 seeds",
     "results/external/tables_2axis_stratified_multiseed/shift_results.csv", None),
]


def decompose(df: pd.DataFrame) -> pd.DataFrame:
    keys = ("model", "dataset_id", "shift_axis")
    if "base_seed" in df.columns and df["base_seed"].nunique() > 1:
        keys += ("base_seed",)
    env = compute_envelopes(df, group_keys=keys)

    zero = df[df.shift_lambda == 0.0].copy()
    zero["_cell"] = list(map(tuple, zero[list(keys)].values))
    passed = set(zero.loc[~zero.failed.astype(bool), "_cell"])
    env["_cell"] = list(map(tuple, env[list(keys)].values))
    env["admitted"] = env["_cell"].isin(passed)

    rows = []
    for model, g in env.groupby("model"):
        a = float(g.admitted.mean())
        t = float(g.loc[g.admitted, "rho"].mean()) if g.admitted.any() else 0.0
        rows.append({"model": model, "aure": float(g.rho.mean()),
                     "admissibility_A": a, "conditional_tolerance_T": t,
                     "A_times_T": a * t, "n_cells": int(len(g))})
    out = pd.DataFrame(rows).sort_values("aure", ascending=False).reset_index(drop=True)

    gap = (out.aure - out.A_times_T).abs().max()
    if gap > 1e-9:
        raise SystemExit(
            f"[decompose] identity A*T = AURE violated by {gap:.2e}; the envelope or the "
            "lambda=0 pass set has been computed inconsistently."
        )
    return out


def margins(out: pd.DataFrame, families: pd.Series) -> dict[str, float]:
    idx = out.set_index("model")
    icl = [m for m in ICL if m in idx.index]
    gbdt = [m for m in idx.index if families.get(m) == "gbdt"]
    if not icl or not gbdt:
        return {}
    return {col: float(idx.loc[icl, col].min() - idx.loc[gbdt, col].max())
            for col in ("aure", "admissibility_A", "conditional_tolerance_T")}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=_ROOT / "results" / "ablations" / "decomposition")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    frames = []
    for label, rel, _ in BASES:
        path = _ROOT / rel
        if not path.exists():
            print(f"[decompose] skipping {label}: no {rel}")
            continue
        df = pd.read_csv(path)
        out = decompose(df)
        fam = df.drop_duplicates("model").set_index("model")["family"]
        out.insert(0, "basis", label)
        frames.append(out)

        print(f"\n=== {label} ===")
        print(out.drop(columns=["basis", "A_times_T"]).round(4).to_string(index=False))
        m = margins(out, fam)
        if m:
            print(f"  min-ICL - best-GBDT:  AURE {m['aure']:+.4f}   "
                  f"admissibility {m['admissibility_A']:+.4f}   "
                  f"conditional tolerance {m['conditional_tolerance_T']:+.4f}")

    if frames:
        allf = pd.concat(frames, ignore_index=True)
        allf.to_csv(args.out / "aure_decomposition.csv", index=False)
        print(f"\n[decompose] wrote {args.out / 'aure_decomposition.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
