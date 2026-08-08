"""Symmetric, like-for-like decomposition of each ICL-vs-GBDT AURE gap into A and T parts.

`decompose_aure.py` reports A and T per model and a margin row that takes the best GBDT
*per column*. That is not like-for-like: the model supplying "best GBDT" in the AURE
column need not be the one supplying it in the T column, so the margin row cannot support
a statement about where any particular gap comes from. On the pilot it is actively
misleading -- TabICL beats CatBoost on T (0.2385 against 0.1577), yet the T-margin reads
-0.0004 because a different model, hist_gbdt, has the highest T.

For a single pair the product difference splits exactly:

    A_i T_i - A_j T_j = ½ (A_i - A_j)(T_i + T_j) + ½ (T_i - T_j)(A_i + A_j)
                        \\___________________/     \\___________________/
                          admissibility part          tolerance part

Both models appear in both terms, so no reference model is chosen and the split is
symmetric in i and j. The identity is asserted at run time.

Usage::

    python experiments/pairwise_decomposition.py
"""

from __future__ import annotations

import argparse
import itertools
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

ICL = ("tabicl", "tabpfn", "tabpfn_client")
BASES = {
    "external 44 ds, 3 seeds (primary)": (
        "results/external/tables_2axis_multiseed_primary_with_tabpfn_local/shift_results.csv",
        ("tabpfn_client",),
    ),
    "pilot (seed-42 grid)": ("results/canonical/tables/shift_results.csv", ()),
}


def factors(path: Path, drop: tuple[str, ...]) -> dict[str, tuple[float, float, float, str]]:
    """Return {model: (AURE, A, T, family)} for one evidence base."""
    d = pd.read_csv(path)
    d = d[~d.model.isin(drop)]
    keys = ("model", "dataset_id", "shift_axis")
    if "base_seed" in d.columns:
        keys += ("base_seed",)
    env = compute_envelopes(d, group_keys=keys)
    idx = list(keys)
    zero = d[d.shift_lambda == 0.0].set_index(idx).failed.astype(bool)
    env = env.set_index(idx)
    env["passed0"] = ~zero.reindex(env.index)
    env = env.reset_index()
    fam = d.drop_duplicates("model").set_index("model").family
    return {m: (g.rho.mean(), g.passed0.mean(), g[g.passed0].rho.mean(), fam.get(m))
            for m, g in env.groupby("model")}


def decompose(f: dict, i: str, j: str) -> tuple[float, float, float]:
    a_i, t_i = f[i][1], f[i][2]
    a_j, t_j = f[j][1], f[j][2]
    gap = a_i * t_i - a_j * t_j
    from_a = 0.5 * (a_i - a_j) * (t_i + t_j)
    from_t = 0.5 * (t_i - t_j) * (a_i + a_j)
    if abs((from_a + from_t) - gap) > 1e-12:
        raise SystemExit(f"[pairwise] decomposition is not exact for {i} vs {j}")
    return gap, from_a, from_t


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)

    for label, (rel, drop) in BASES.items():
        f = factors(_ROOT / rel, drop)
        icl = [m for m in f if m in ICL]
        gb = [m for m in f if f[m][3] == "gbdt"]
        print(f"\n=== {label} ===")
        print(f"{'pair':34s} {'ΔAURE':>8s} {'from A':>9s} {'from T':>9s} {'A share':>9s}")
        rows = []
        for i, j in itertools.product(icl, gb):
            gap, from_a, from_t = decompose(f, i, j)
            share = from_a / gap if gap else float("nan")
            rows.append((gap, from_a, from_t, share))
            print(f"{i + ' − ' + j:34s} {gap:+8.4f} {from_a:+9.4f} {from_t:+9.4f} "
                  f"{share:8.0%}")
        lead = [r for r in rows if r[0] > 0]
        if lead:
            print(f"  ICL ahead on {len(lead)} of {len(rows)} pairs; "
                  f"A-part positive on {sum(1 for r in lead if r[1] > 0)}, "
                  f"T-part positive on {sum(1 for r in lead if r[2] > 0)}")
            print(f"  median share of the gap attributable to A: "
                  f"{pd.Series([r[3] for r in lead]).median():.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
