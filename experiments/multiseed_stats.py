"""Recompute multiseed CIs + paired tests on a merged reliability-envelope table.

After ``merge_tabpfn_local.py`` folds the complete local-TabPFN rows into the multiseed
primary, the run's own ``aure_multiseed.csv`` / ``pairwise_tests.csv`` are stale (they
predate the merge). This regenerates both from the merged envelopes using the SAME
functions the runner uses (``summarize_aure`` = bootstrap seed- and dataset-unit CIs;
``pairwise_tests`` = paired Wilcoxon + Holm at both units), so the full-scale multiseed
statistics now include two complete ICL models (tabicl + tabpfn). No models are re-run.

Usage::

    python experiments/multiseed_stats.py \
        --dir results/external/tables_2axis_multiseed_primary_with_tabpfn_local
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(_ROOT / "experiments"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from run_multiseed import _aure_by_dataset, pairwise_tests, summarize_aure  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, required=True)
    args = p.parse_args(argv)

    env = pd.read_csv(args.dir / "reliability_envelopes.csv")
    if "base_seed" not in env.columns:
        raise SystemExit("[stats] envelopes have no base_seed column; not a multiseed table")

    aure_by_seed = (
        env.groupby(["base_seed", "model"]).rho.mean().reset_index(name="aure")
        .rename(columns={"base_seed": "seed"})
    )
    by_dataset = _aure_by_dataset(env[["model", "dataset_id", "rho"]])
    multiseed = summarize_aure(aure_by_seed, by_dataset)
    pairwise = pairwise_tests(aure_by_seed, by_dataset)

    multiseed.to_csv(args.dir / "aure_multiseed.csv", index=False)
    pairwise.to_csv(args.dir / "pairwise_tests.csv", index=False)

    print("=== AURE, dataset-unit 95% bootstrap CI (n=44), seed SD (n=3) ===")
    cols = ["model", "aure_mean", "aure_sd", "dataset_ci95_lo", "dataset_ci95_hi", "n_seeds"]
    print(multiseed[[c for c in cols if c in multiseed.columns]]
          .sort_values("aure_mean", ascending=False).round(4).to_string(index=False))

    icl = {"tabicl", "tabpfn", "tabpfn_client"}
    pw = pairwise.copy()
    mask = pw.apply(
        lambda r: len({r["model_a"], r["model_b"]} & icl) == 1, axis=1
    )  # exactly one ICL model in the pair -> ICL-vs-baseline (or ICL-vs-ICL excluded)
    icl_vs_base = pw[mask]
    print("\n=== ICL-vs-baseline pairs "
          "(dataset unit n=44: mean delta + 95% CI; seed unit n=3: Wilcoxon+Holm) ===")
    print("    (delta = model_a - model_b; positive CI excluding 0 = model_a reliably ahead)")
    show = ["model_a", "model_b", "dataset_mean_delta", "dataset_ci95_lo",
            "dataset_ci95_hi", "seed_wilcoxon_p", "seed_p_holm", "significant"]
    show = [c for c in show if c in pw.columns]
    print((icl_vs_base if not icl_vs_base.empty else pw)[show].round(4).to_string(index=False))
    print(f"\n[stats] wrote aure_multiseed.csv + pairwise_tests.csv to {args.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
