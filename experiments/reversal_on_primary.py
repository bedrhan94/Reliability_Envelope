"""Reference-matched re-analysis on the three-seed primary benchmark.

`ablate_reference_confound.py` runs the two variants on the single-seed and subset bases.
Those predate the three-seed primary run, so the paper's headline (three seeds) and the
re-analysis that qualifies it (single seed) sat on different evidence bases -- a referee
asks about that immediately. This computes both variants on the primary base itself.

No models are re-run: both variants are re-scorings of the stored per-condition table.

    common support   keep only cells where EVERY model passes at lambda=0, so all models
                     are scored on an identical cell set
    self-referenced  rebase all three triggers on each model's own clean state, so no
                     cell can be decided before any shift is applied

Usage::

    python experiments/reversal_on_primary.py
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
from tice.envelope.reliability import compute_envelopes  # noqa: E402
from tice.metrics.utility import evaluate_failure  # noqa: E402

PRIMARY = _ROOT / ("results/external/tables_2axis_multiseed_primary_with_tabpfn_local/"
                   "shift_results.csv")
GROUP = ("model", "dataset_id", "shift_axis", "base_seed")
CELL = ["dataset_id", "shift_axis", "base_seed"]
ICL = ("tabicl", "tabpfn")
# tabpfn_client is quota-limited and superseded by the local-weights rows (see §5.3).
DROP = ("tabpfn_client",)


def margin(frame: pd.DataFrame) -> float:
    """min-ICL minus best gradient-boosted, the convention used throughout the paper."""
    aure = compute_envelopes(frame, group_keys=GROUP).groupby("model").rho.mean()
    fam = frame.drop_duplicates("model").set_index("model")["family"]
    icl = [m for m in ICL if m in aure.index]
    gb = [m for m in aure.index if fam.get(m) == "gbdt"]
    if not icl or not gb:
        raise SystemExit(f"[reversal] need both families; icl={icl} gbdt={gb}")
    return float(aure[icl].min() - aure[gb].max())


def common_support(d: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    zero = d[d.shift_lambda == 0.0]
    passes = zero.groupby(CELL).failed.apply(lambda s: not s.astype(bool).any())
    keep = passes[passes].index
    kept = d.set_index(CELL).loc[list(keep)].reset_index()
    return kept, len(keep), zero.groupby(CELL).ngroups


def self_referenced(d: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    """Rebase all three triggers on each model's own clean state."""
    clean = d[d.shift_lambda == 0.0].set_index(["model", *CELL])
    out = d.copy()
    keys = list(zip(out.model, out.dataset_id, out.shift_axis, out.base_seed, strict=True))
    own_u = [clean.utility.get(k) for k in keys]
    own_ece = [clean.ece.get(k) for k in keys]
    own_nll = [clean.nll_norm.get(k) for k in keys]
    failed = []
    for r, u, e, n in zip(out.itertuples(index=False), own_u, own_ece, own_nll, strict=True):
        res = evaluate_failure(
            utility=r.utility, ece=r.ece, nll_norm=r.nll_norm, reference_utility=u,
            thresholds=Thresholds(tau_utility=th.tau_utility,
                                  tau_ece=e + th.tau_ece, tau_nll=n + th.tau_nll),
            status_ok=(r.status == "ok"))
        failed.append(res.failed)
    out["failed"] = failed
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=PRIMARY)
    args = p.parse_args(argv)

    d = pd.read_csv(args.results)
    d = d[~d.model.isin(DROP)]
    th = Thresholds()

    m_pub = margin(d)
    kept, n_keep, n_all = common_support(d)
    m_cs = margin(kept)
    m_sr = margin(self_referenced(d, th))

    print(f"basis: {args.results.name}  "
          f"({d.dataset_id.nunique()} datasets x {d.model.nunique()} models x "
          f"{d.base_seed.nunique()} seeds)")
    print(f"  published rule   {m_pub:+.4f}")
    print(f"  common support   {m_cs:+.4f}   ({n_keep} of {n_all} cells kept)")
    print(f"  self-referenced  {m_sr:+.4f}")
    print(f"\nreversal holds: common support {m_cs < 0}, self-referenced {m_sr < 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
