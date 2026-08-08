"""Fold the THREE-seed strong-baseline arm into the multiseed primary and re-score.

The single-seed version of this merge is ``merge_tuned_gbdt.py``, which produced the
published "+0.0227 -> +0.0205" strong-baseline result at seed 42. That number had no
uncertainty attached: one seed cannot say whether a 0.002 change is an effect or noise.
This script runs the same merge at seeds 42/1337/2025 so the margin gets a seed SD.

Two things force a per-seed loop rather than one pass over the concatenated table:

1. ``rescore`` picks the GBDT reference as an argmax over each dataset's lambda=0 rows,
   grouped by ``dataset_id`` alone. Pooling seeds would let one seed's clean utility set
   the bar for another's -- a silent cross-seed leak into the failure rule.
2. rho requires an unbroken pass-run in lambda, so envelopes must also be computed
   within a seed. That is handled by adding ``base_seed`` to ``compute_envelopes``'
   group keys, not by concatenating afterwards.

``tabpfn_client`` is dropped from the base. It is superseded by the local-weights
``tabpfn`` rows and is incomplete anyway (seed 1337: 212 errors; seed 2025: all 528
skipped) because of the PriorLabs daily quota.

**The margin is reported over two ICL sets, and that is the point of this script.**
The published +0.0227 -> +0.0205 pair was computed when the only ICL model in the merged
table was TabICL: TabPFN's external AURE came from ``tables_2axis_multiseed_tabpfn_fill``
and later ``..._tabpfn_local``, single-family arms that contain no gradient-boosted model.
With no reference-family model present, ``best_gbdt``/``reference_utility`` is NaN on every
row, so the ``utility_gap`` criterion cannot fire and those arms are scored under a
two-trigger rule while the paper's rule has three. TabPFN's metrics are bit-identical
either way; only the ``failed`` flag moves, and every one of the 131 cells (8.3%) that
flips is ``utility_gap``. The inflation is +0.0307 -- 0.1195 unreferenced against 0.0888
referenced, larger than the whole ICL-vs-GBDT margin it was being compared against.

Since the margin is a min over the ICL set, a properly referenced TabPFN at 0.0888 becomes
the *carrier* rather than a passenger, and the external margin drops to +0.0144. So this
script asserts the published pair reproduces under ``ICL_PUBLISHED`` (proving the merge
configuration matches the paper's) and reports the corrected margin under ``ICL``. Keeping
both makes the correction auditable instead of a silent renumbering.

Statistics are deliberately NOT computed here: ``multiseed_stats.py`` regenerates the
CIs and paired tests from the envelopes using the same functions the runner uses.

Usage::

    python experiments/merge_strong_multiseed.py
    python experiments/multiseed_stats.py --dir results/external/tables_2axis_strong_multiseed_merged
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from merge_tuned_gbdt import rescore  # noqa: E402

from tice.config import Thresholds  # noqa: E402
from tice.envelope.reliability import compute_aure, compute_envelopes  # noqa: E402

SEEDS = (42, 1337, 2025)
BASE = _ROOT / "results/external/tables_2axis_multiseed_primary_with_tabpfn_local/shift_results.csv"
STRONG = _ROOT / "results/external/tables_2axis_strong_baselines_multiseed"
OUT = _ROOT / "results/external/tables_2axis_strong_multiseed_merged"

DROP_FROM_BASE = ("tabpfn_client",)
REPLACE = ("xgboost", "catboost", "hist_gbdt")
BASELINE_REFERENCE = ("xgboost", "catboost")
REFERENCE = ("xgboost_tuned", "catboost_tuned")
ICL = ("tabicl", "tabpfn")
# The ICL set the published margin was effectively computed over -- TabICL alone, because
# TabPFN's external AURE came from a reference-less arm and was never folded in.
ICL_PUBLISHED = ("tabicl",)

# What the paper states at seed 42 under ICL_PUBLISHED. Asserted, not trusted.
PUBLISHED_SEED42 = (0.022727272727272724, 0.020454545454545475)


def margin(aure: pd.Series, frame: pd.DataFrame,
           icl_set: tuple[str, ...] = ICL) -> float:
    """min-ICL minus best-GBDT, mirroring ``merge_tuned_gbdt._margin``.

    "Best GBDT" is the max over every gbdt-*family* model present, not just the two the
    reference is drawn from: a reader compares against the strongest baseline in the
    table. Taking the *min* over the ICL models is the conservative direction, and it is
    why a correction to TabPFN moves the headline -- once referenced it is the weaker ICL
    model, so it carries the margin.
    """
    fam = frame.drop_duplicates("model").set_index("model")["family"]
    icl = [m for m in icl_set if m in aure.index]
    gb = [m for m in aure.index if fam.get(m) == "gbdt"]
    if not icl or not gb:
        raise SystemExit(f"[merge-strong] margin needs both families; icl={icl} gbdt={gb}")
    return float(aure[icl].min() - aure[gb].max())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)

    base_all = pd.read_csv(BASE)
    if "base_seed" not in base_all.columns:
        raise SystemExit("[merge-strong] base has no base_seed column; not a multiseed table")
    base_all = base_all[~base_all.model.isin(DROP_FROM_BASE)]
    th = Thresholds()

    pub_frames, str_frames, rows = [], [], []
    for seed in SEEDS:
        base = base_all[base_all.base_seed == seed]
        ckpt = STRONG / "checkpoints" / f"seed_{seed}_shift_results.csv"
        if not ckpt.exists():
            raise SystemExit(f"[merge-strong] missing checkpoint for seed {seed}: {ckpt}")
        tuned = pd.read_csv(ckpt)
        if set(tuned.base_seed.unique()) != {seed}:
            raise SystemExit(f"[merge-strong] {ckpt.name} carries seeds "
                             f"{sorted(tuned.base_seed.unique())}, expected [{seed}]")

        datasets = sorted(set(tuned.dataset_id))
        base = base[base.dataset_id.isin(datasets)]

        # Same guard as merge_tuned_gbdt: if our reference pick stops reproducing the
        # pipeline's stored `failed`, the published-vs-strong comparison is contaminated.
        published = rescore(base, BASELINE_REFERENCE, th)
        stored = base["failed"].astype(bool).to_numpy()
        if not (published["failed"].astype(bool).to_numpy() == stored).all():
            raise SystemExit(
                f"[merge-strong] rescore() does not reproduce the stored `failed` column "
                f"at seed {seed}; reference semantics have diverged from the pipeline."
            )

        kept = base[~base.model.isin(REPLACE)]
        cols = [c for c in kept.columns if c in tuned.columns]
        strong = rescore(pd.concat([kept[cols], tuned[cols]], ignore_index=True), REFERENCE, th)

        pub_frames.append(published)
        str_frames.append(strong)

        a_pub = compute_aure(compute_envelopes(published)).set_index("model").aure
        a_str = compute_aure(compute_envelopes(strong)).set_index("model").aure
        m_pub, m_str = margin(a_pub, published), margin(a_str, strong)
        # Same envelopes, margin taken over TabICL alone -- the paper's effective ICL set.
        o_pub = margin(a_pub, published, ICL_PUBLISHED)
        o_str = margin(a_str, strong, ICL_PUBLISHED)
        rows.append({"seed": seed, "margin_published": m_pub, "margin_strong": m_str,
                     "change": m_str - m_pub,
                     "margin_published_iclonly": o_pub, "margin_strong_iclonly": o_str})
        print(f"seed {seed}: margin published {m_pub:+.4f} -> strong {m_str:+.4f} "
              f"(change {m_str - m_pub:+.4f}) | tabicl-only {o_pub:+.4f} -> {o_str:+.4f}")

        if seed == 42:
            got = (round(o_pub, 10), round(o_str, 10))
            want = tuple(round(v, 10) for v in PUBLISHED_SEED42)
            if got != want:
                raise SystemExit(
                    f"[merge-strong] seed 42 does not reproduce the published margins under "
                    f"the TabICL-only ICL set: got {got}, expected {want}. The merge "
                    f"configuration has drifted from the paper's; fix before reporting."
                )
            print("           reproduces the published +0.0227 -> +0.0205 under "
                  "ICL={tabicl}; the corrected pair above includes referenced TabPFN")

    published_all = pd.concat(pub_frames, ignore_index=True)
    strong_all = pd.concat(str_frames, ignore_index=True)
    gk = ("model", "dataset_id", "shift_axis", "base_seed")
    env_pub = compute_envelopes(published_all, group_keys=gk)
    env_str = compute_envelopes(strong_all, group_keys=gk)

    def by_seed(env: pd.DataFrame) -> pd.DataFrame:
        return env.groupby(["base_seed", "model"]).rho.mean().reset_index(name="aure")

    aure_pub, aure_str = by_seed(env_pub), by_seed(env_str)
    comparison = pd.DataFrame({
        "aure_published_baselines": aure_pub.groupby("model").aure.mean(),
        "aure_published_sd": aure_pub.groupby("model").aure.std(ddof=1),
        "aure_strong_baselines": aure_str.groupby("model").aure.mean(),
        "aure_strong_sd": aure_str.groupby("model").aure.std(ddof=1),
        "lambda0_fail_published": published_all[published_all.shift_lambda == 0.0]
            .groupby("model").failed.mean(),
        "lambda0_fail_strong": strong_all[strong_all.shift_lambda == 0.0]
            .groupby("model").failed.mean(),
    }).sort_values("aure_strong_baselines", ascending=False)

    margins = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    strong_all.to_csv(args.out / "shift_results.csv", index=False)
    env_str.to_csv(args.out / "reliability_envelopes.csv", index=False)
    env_pub.to_csv(args.out / "reliability_envelopes_published.csv", index=False)
    aure_str.rename(columns={"base_seed": "seed"}).to_csv(args.out / "aure_by_seed.csv", index=False)
    comparison.to_csv(args.out / "strong_vs_published_multiseed.csv")
    margins.to_csv(args.out / "margin_by_seed.csv", index=False)

    print(f"\ndatasets={published_all.dataset_id.nunique()} seeds={len(SEEDS)} "
          f"rows={len(strong_all)}")
    print(comparison.round(4).to_string())
    mp, ms = margins.margin_published, margins.margin_strong
    print("\n=== corrected: ICL = {tabicl, tabpfn}, both referenced ===")
    print(f"margin published {mp.mean():+.4f} +/- {mp.std(ddof=1):.4f} (seed SD)")
    print(f"margin strong    {ms.mean():+.4f} +/- {ms.std(ddof=1):.4f} (seed SD)")
    print(f"change           {(ms - mp).mean():+.4f} +/- {(ms - mp).std(ddof=1):.4f}")
    op, os_ = margins.margin_published_iclonly, margins.margin_strong_iclonly
    print("\n=== as published: ICL = {tabicl} only (TabPFN was reference-less) ===")
    print(f"margin published {op.mean():+.4f} +/- {op.std(ddof=1):.4f} (seed SD)")
    print(f"margin strong    {os_.mean():+.4f} +/- {os_.std(ddof=1):.4f} (seed SD)")
    print(f"\nreference model per dataset (strong): {strong_all.best_gbdt.value_counts().to_dict()}")
    print(f"[merge-strong] wrote 6 tables to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
