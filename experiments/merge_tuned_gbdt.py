"""Merge the tuned-GBDT run into the 6-model seed-42 table and re-score everything.

Referee objection: the published GBDT baselines run at fixed hyper-parameters
while TabPFN/TabICL are tuning-free, so the comparison is unfairly weak. This
answers it without re-running the ICL models -- their per-condition *metrics* do
not depend on the reference, only the ``failed`` flag does, and that is recomputed
here.

The merge does two things the ``merge_tabpfn_client`` merge does not:

1. the GBDT reference (``best_gbdt`` / ``reference_utility``) is recomputed from
   the **tuned** clean utilities, so every model is scored against the stronger
   bar -- which is the whole point; and
2. the untuned ``xgboost`` / ``catboost`` rows are dropped, so the tuned pair
   replaces them rather than competing alongside them.

Note the direction of the effect: tuning raises ``U_ref`` (ICL trips
``utility_gap`` *more*) and lowers GBDT clean ECE (fewer of their own lambda=0
failures). Both narrow the ICL margin.

Usage::

    python experiments/merge_tuned_gbdt.py \
        --base results/external/tables_2axis_stratified_multiseed/shift_results.csv \
        --tuned results/external/tables_2axis_tuned_gbdt_only/shift_results.csv \
        --out results/external/tables_2axis_tuned_gbdt_merged
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
from tice.envelope.reliability import compute_aure, compute_envelopes  # noqa: E402
from tice.metrics.utility import evaluate_failure  # noqa: E402

REPLACED = ("xgboost", "catboost")
TUNED = ("xgboost_tuned", "catboost_tuned")
ICL = ("tabicl", "tabpfn_client")


def rescore(df: pd.DataFrame, reference_models: tuple[str, ...],
            thresholds: Thresholds) -> pd.DataFrame:
    """Recompute best_gbdt / reference_utility / failed for every row."""
    clean = df[(df.shift_lambda == 0.0) & (df.status == "ok")]
    ref_rows = clean[clean.model.isin(reference_models)]
    best = ref_rows.loc[ref_rows.groupby("dataset_id").utility.idxmax()]
    ref_util = dict(zip(best.dataset_id, best.utility, strict=True))
    ref_name = dict(zip(best.dataset_id, best.model, strict=True))

    out = df.copy()
    out["reference_utility"] = out.dataset_id.map(ref_util)
    out["best_gbdt"] = out.dataset_id.map(ref_name)
    failed, reasons = [], []
    for r in out.itertuples(index=False):
        res = evaluate_failure(
            utility=r.utility, ece=r.ece, nll_norm=r.nll_norm,
            reference_utility=r.reference_utility, thresholds=thresholds,
            status_ok=(r.status == "ok"),
        )
        failed.append(res.failed)
        reasons.append(res.reason_str)
    out["failed"] = failed
    out["failure_reason"] = reasons
    return out


def _aure(df: pd.DataFrame) -> pd.Series:
    return compute_aure(compute_envelopes(df)).set_index("model").aure


def _margin(aure: pd.Series, gbdt: tuple[str, ...]) -> float:
    icl = [m for m in ICL if m in aure.index]
    gb = [m for m in gbdt if m in aure.index]
    return float(aure[icl].min() - aure[gb].max())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--tuned", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42, help="base_seed slice to merge against")
    args = p.parse_args(argv)

    base = pd.read_csv(args.base)
    if "base_seed" in base.columns:
        base = base[base.base_seed == args.seed]
    tuned = pd.read_csv(args.tuned)
    th = Thresholds()

    datasets = sorted(set(tuned.dataset_id))
    base = base[base.dataset_id.isin(datasets)]
    kept = base[~base.model.isin(REPLACED)]
    cols = [c for c in kept.columns if c in tuned.columns]
    merged = pd.concat([kept[cols], tuned[cols]], ignore_index=True)

    untuned_ref = rescore(base, REPLACED, th)
    # Guard: our reference pick is argmax over the dataset's lambda=0 rows, while the
    # pipeline stores the first such row per (dataset, model). Those agree only while
    # the axes' lambda=0 rows are identical (they are -- the shift is a no-op there).
    # If a future run breaks that, the "untuned" arm would silently stop reproducing
    # the published baseline and the tuning effect would be contaminated.
    stored = base["failed"].astype(bool).to_numpy()
    if not (untuned_ref["failed"].astype(bool).to_numpy() == stored).all():
        raise SystemExit(
            "[merge-tuned] rescore() does not reproduce the stored `failed` column on "
            "the untuned rows -- the reference-selection semantics have diverged from "
            "the pipeline; fix before trusting the tuned-vs-untuned comparison."
        )
    tuned_ref = rescore(merged, TUNED, th)

    a_untuned, a_tuned = _aure(untuned_ref), _aure(tuned_ref)
    m_untuned = _margin(a_untuned, REPLACED)
    m_tuned = _margin(a_tuned, TUNED)

    zero_un = untuned_ref[untuned_ref.shift_lambda == 0.0].groupby("model").failed.mean()
    zero_tu = tuned_ref[tuned_ref.shift_lambda == 0.0].groupby("model").failed.mean()

    comparison = pd.DataFrame(
        {
            "aure_untuned_baselines": a_untuned,
            "aure_tuned_baselines": a_tuned,
            "lambda0_fail_untuned": zero_un,
            "lambda0_fail_tuned": zero_tu,
        }
    ).sort_values("aure_tuned_baselines", ascending=False)

    args.out.mkdir(parents=True, exist_ok=True)
    tuned_ref.to_csv(args.out / "shift_results.csv", index=False)
    envelopes = compute_envelopes(tuned_ref)
    envelopes.to_csv(args.out / "reliability_envelopes.csv", index=False)
    compute_aure(envelopes).to_csv(args.out / "aure_summary.csv", index=False)
    comparison.to_csv(args.out / "tuned_vs_untuned.csv")

    print(f"datasets={len(datasets)} rows={len(tuned_ref)} seed={args.seed}")
    print(comparison.round(4).to_string())
    print(f"\nreference model per dataset (tuned): "
          f"{tuned_ref.best_gbdt.value_counts().to_dict()}")
    print(f"min-ICL - best-GBDT margin: untuned {m_untuned:+.4f} -> tuned {m_tuned:+.4f} "
          f"(change {m_tuned - m_untuned:+.4f})")
    print(f"wrote 4 tables to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
