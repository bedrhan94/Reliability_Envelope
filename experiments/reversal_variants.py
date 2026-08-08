"""Reference-matched re-analysis on every evidence base, including after each intervention.

`ablate_reference_confound.py` runs the two variants on the single-seed and subset bases.
Those predate the three-seed primary run, so the paper's headline and the re-analysis that
qualifies it sat on different evidence. And the abstract claims the reversal survives
tuning, calibration and baseline strengthening, which the tuning/calibration tables did not
actually show -- they reported the change in *published* AURE, not the reversal after the
intervention. This computes both variants on all four arms so the claim is backed rather
than asserted.

No models are re-run: both variants are re-scorings of stored per-condition tables.

    common support   keep only cells where EVERY model passes at lambda=0, so all models
                     are scored on an identical cell set
    self-referenced  rebase all three triggers on each model's own clean state, so no
                     cell can be decided before any shift is applied

Usage::

    python experiments/reversal_variants.py
    python experiments/reversal_variants.py --arm "tuned baselines"
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
ICL = ("tabicl", "tabpfn", "tabpfn_client")
# tabpfn_client is quota-limited and superseded by the local-weights rows on the primary
# base (§5.3); the older merged arms predate that run and carry it as their only TabPFN.
DROP_ON_PRIMARY = ("tabpfn_client",)

# Every arm on which the paper claims the reversal survives an intervention. The abstract
# says it survives tuning, calibration and strengthening, so each of those has to be shown
# rather than asserted from the change in published AURE alone.
ARMS = {
    "primary (3 seeds)": (PRIMARY, DROP_ON_PRIMARY),
    "tuned baselines": (
        _ROOT / "results/external/tables_2axis_tuned_gbdt_merged/shift_results.csv", ()),
    "calibrated baselines": (
        _ROOT / "results/external/tables_2axis_calibrated_merged/shift_results.csv", ()),
    "strong baselines (3 seeds)": (
        _ROOT / "results/external/tables_2axis_strong_multiseed_merged/shift_results.csv", ()),
}


def margin(frame: pd.DataFrame) -> float:
    """min-ICL minus best gradient-boosted, the convention used throughout the paper."""
    aure = compute_envelopes(frame, group_keys=_group(frame)).groupby("model").rho.mean()
    fam = frame.drop_duplicates("model").set_index("model")["family"]
    icl = [m for m in ICL if m in aure.index]
    gb = [m for m in aure.index if fam.get(m) == "gbdt"]
    if not icl or not gb:
        raise SystemExit(f"[reversal] need both families; icl={icl} gbdt={gb}")
    return float(aure[icl].min() - aure[gb].max())


def _cell(d: pd.DataFrame) -> list[str]:
    """Arms merged before the multiseed run have no base_seed column."""
    return [c for c in CELL if c in d.columns]


def _group(d: pd.DataFrame) -> tuple[str, ...]:
    return tuple(c for c in GROUP if c in d.columns)


def common_support(d: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    CELL_ = _cell(d)
    zero = d[d.shift_lambda == 0.0]
    passes = zero.groupby(CELL_).failed.apply(lambda s: not s.astype(bool).any())
    keep = passes[passes].index
    kept = d.set_index(CELL_).loc[list(keep)].reset_index()
    return kept, len(keep), zero.groupby(CELL_).ngroups


def self_referenced(d: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    """Rebase all three triggers on each model's own clean state."""
    CELL_ = _cell(d)
    clean = d[d.shift_lambda == 0.0].set_index(["model", *CELL_])
    out = d.copy()
    cols = [out.model] + [out[c] for c in CELL_]
    keys = list(zip(*cols, strict=True))
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
    p.add_argument("--arm", choices=sorted(ARMS), help="run one arm instead of all")
    args = p.parse_args(argv)

    names = [args.arm] if args.arm else list(ARMS)
    print(f"{'arm':30s} {'published':>10s} {'common supp':>12s} {'self-ref':>10s} "
          f"{'cells':>10s}")
    rows = []
    for name in names:
        path, drop = ARMS[name]
        if not path.exists():
            print(f"{name:30s}  (missing: {path.name})")
            continue
        d = pd.read_csv(path)
        d = d[~d.model.isin(drop)]
        th = Thresholds()
        m_pub = margin(d)
        kept, n_keep, n_all = common_support(d)
        m_cs = margin(kept)
        m_sr = margin(self_referenced(d, th))
        rows.append((name, m_pub, m_cs, m_sr))
        print(f"{name:30s} {m_pub:+10.4f} {m_cs:+12.4f} {m_sr:+10.4f} "
              f"{f'{n_keep}/{n_all}':>10s}")

    flips = [r for r in rows if r[2] < 0 and r[3] < 0]
    print(f"\nreversal holds on {len(flips)} of {len(rows)} arms "
          f"(both variants negative)")
    for name, m_pub, m_cs, m_sr in rows:
        if not (m_cs < 0 and m_sr < 0):
            print(f"  !! {name}: published {m_pub:+.4f}, common support {m_cs:+.4f}, "
                  f"self-referenced {m_sr:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
