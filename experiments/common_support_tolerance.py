"""Measure conditional tolerance T on common support, where its conditioning bias is gone.

T = E[rho | pass at lambda=0] is conditioned on each model's *own* admitted set, so a model
with low admissibility has its T measured only where it starts well -- plausibly the easier
cells. Section 6.2 states the direction of that bias; this measures it.

On common support -- the cells where *every* model passes at lambda=0 -- each model is
admissible on every retained cell by construction, so A = 1 and T = AURE there. That is the
bias-free T comparison: identical cells, no conditioning, paired across datasets. The script
asserts A = 1 rather than assuming it.

Two things come out: how much the own-set T comparison was distorted (the gap between the
two margins), and whether the per-pair ordering survives on the smaller, harder cell set.

Usage::

    python experiments/common_support_tolerance.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tice.envelope.reliability import compute_envelopes  # noqa: E402

PRIMARY = _ROOT / ("results/external/tables_2axis_multiseed_primary_with_tabpfn_local/"
                   "shift_results.csv")
GROUP = ("model", "dataset_id", "shift_axis", "base_seed")
CELL = ["dataset_id", "shift_axis", "base_seed"]
ICL = ("tabicl", "tabpfn")
DROP = ("tabpfn_client",)


def factors(frame: pd.DataFrame) -> pd.DataFrame:
    """Envelope radii with a per-cell flag for whether the model was admissible."""
    env = compute_envelopes(frame, group_keys=GROUP)
    zero = frame[frame.shift_lambda == 0.0].set_index(list(GROUP)).failed.astype(bool)
    env = env.set_index(list(GROUP))
    env["admitted"] = ~zero.reindex(env.index)
    return env.reset_index()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=PRIMARY)
    args = p.parse_args(argv)

    d = pd.read_csv(args.results)
    d = d[~d.model.isin(DROP)]

    own = factors(d)
    own_t = {m: g[g.admitted].rho.mean() for m, g in own.groupby("model")}
    own_a = {m: g.admitted.mean() for m, g in own.groupby("model")}

    zero = d[d.shift_lambda == 0.0]
    all_pass = zero.groupby(CELL).failed.apply(lambda s: not s.astype(bool).any())
    keep = list(all_pass[all_pass].index)
    cs = factors(d.set_index(CELL).loc[keep].reset_index())

    fam = d.drop_duplicates("model").set_index("model").family
    gb = [m for m in own_t if fam.get(m) == "gbdt"]
    icl = [m for m in ICL if m in own_t]

    print(f"common support: {len(keep)} of {zero.groupby(CELL).ngroups} cells\n")
    print(f"{'model':11s} {'A (own)':>8s} {'T (own)':>8s} │ {'A (cs)':>7s} {'T (cs)':>8s}")
    for m in icl + gb:
        g = cs[cs.model == m]
        a_cs = g.admitted.mean()
        if abs(a_cs - 1.0) > 1e-9:
            raise SystemExit(f"[common-support] {m} has A={a_cs:.4f} on common support; "
                             f"every model must be admissible there by construction")
        print(f"{m:11s} {own_a[m]:8.3f} {own_t[m]:8.4f} │ {a_cs:7.3f} {g.rho.mean():8.4f}")

    cs_t = {m: cs[cs.model == m].rho.mean() for m in icl + gb}
    m_own = min(own_t[m] for m in icl) - max(own_t[m] for m in gb)
    m_cs = min(cs_t[m] for m in icl) - max(cs_t[m] for m in gb)
    print("\nmin-ICL − best-GBDT on T:")
    print(f"  own admitted sets : {m_own:+.4f}   (biased toward the low-A baselines)")
    print(f"  common support    : {m_cs:+.4f}   (bias removed)")
    print(f"  distortion        : {m_own - m_cs:+.4f}")

    per_ds = cs.groupby(["model", "dataset_id"]).rho.mean().unstack(0)
    print("\npaired per dataset on common support (Wilcoxon signed-rank):")
    for i, j in itertools.product(icl, gb):
        both = per_ds[[i, j]].dropna()
        diff = both[i] - both[j]
        pval = stats.wilcoxon(both[i], both[j], zero_method="wilcox").pvalue
        print(f"  {i:7s} − {j:10s} mean {diff.mean():+.4f}  "
              f"ICL ahead on {int((diff > 0).sum())} of {int((diff != 0).sum())}  "
              f"p = {pval:.3f}")
    print(f"\n  n is small ({len(per_ds)} datasets retain any common-support cell), so these "
          f"tests are\n  reported as directional rather than as evidence of a difference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
