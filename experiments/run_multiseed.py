"""Multi-seed reliability run with variance + significance testing.

Addresses the single-seed threat to validity: reruns the whole shift-stress
pipeline over several base seeds and reports AURE with honest uncertainty.

Two resampling units are reported on purpose, because they answer different
questions and a Q1 reviewer will ask for both:

* **seed unit** (n = #seeds): re-splits/re-shuffles the *same* datasets. Its CI
  captures split + shift + backend jitter but NOT dataset sampling, so it is a
  *within-these-datasets* statement. Higher power for A-vs-B (dataset is held
  fixed and paired out).
* **dataset unit** (n = #datasets): AURE averaged over seeds×axes within each
  dataset, then treated across datasets. Its CI reflects between-dataset
  variability — the generalisation-relevant uncertainty — but n is tiny here
  (5 toys) so it is deliberately wide/underpowered. This is the honest one.

Pairwise model tests are paired at each unit (never the pseudoreplicated
seed×dataset×axis cell pool), with Holm correction across the model-pair family.
All CIs are percentile bootstraps (no normality assumption on a bounded /
quantised / right-censored metric).

**Every number here is conditional on the failure thresholds and these
datasets.** It quantifies noise around the AURE estimator; it does not fix the
estimator's design (quantisation, censoring, arbitrary weights) — that is a
separate remediation.

Writes ``aure_by_seed.csv``, ``rho_by_seed.csv`` (raw per-seed×dataset×axis ρ,
so the analysis can be redone without re-running), ``aure_multiseed.csv``
(seed-unit + dataset-unit summaries) and ``aure_pairwise.csv`` into the output
dir.

Usage::

    python experiments/run_multiseed.py --config configs/experiments/shift_stress.yaml --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Windows consoles default to a legacy code page (e.g. cp1254) that can't encode
# stats glyphs; force UTF-8 so printing never crashes a completed run.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from scipy import stats  # noqa: E402

from tice.config import load_config  # noqa: E402
from tice.pipeline import run_pipeline  # noqa: E402

_BOOT = 10000
_BOOT_SEED = 12345


def _parse_seeds(text: str) -> list[int]:
    seeds = [int(s) for s in text.replace(" ", "").split(",") if s != ""]
    if len(seeds) < 2:
        raise ValueError("need at least 2 seeds for variance/significance")
    return seeds


def bootstrap_ci(values: np.ndarray, conf: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean; no distributional assumption.

    Honest on a bounded/quantised/censored metric where a t-interval is dubious.
    Degenerate (constant / n<2) samples collapse to the point value.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = v.size
    if n < 2 or np.ptp(v) == 0.0:
        return (float(v.mean()) if n else float("nan"),) * 2
    rng = np.random.default_rng(_BOOT_SEED)
    means = rng.choice(v, size=(_BOOT, n), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return float(lo), float(hi)


def _holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (NaNs pass through)."""
    idx = [i for i, p in enumerate(pvals) if p == p]  # non-NaN
    m = len(idx)
    adj = list(pvals)
    order = sorted(idx, key=lambda i: pvals[i])
    running = 0.0
    for rank, i in enumerate(order):
        a = min(1.0, (m - rank) * pvals[i])
        running = max(running, a)  # enforce monotonic non-decreasing
        adj[i] = running
    return adj


def _wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if not np.any(diff != 0):
        return 1.0
    try:
        return float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def collect_over_seeds(config_path: Path, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the pipeline once per seed; return per-seed AURE and per-seed ρ frames."""
    base = load_config(config_path)
    aure_rows: list[pd.DataFrame] = []
    rho_rows: list[pd.DataFrame] = []
    with tempfile.TemporaryDirectory(prefix="tice_multiseed_") as tmp:
        for seed in seeds:
            cfg = replace(base, seed=seed, output_dir=Path(tmp) / f"seed_{seed}")
            print(f"[multiseed] seed={seed} ...", flush=True)
            out = run_pipeline(cfg)
            a = out.aure_summary.copy()
            a.insert(0, "seed", seed)
            aure_rows.append(a)
            e = out.reliability_envelopes.copy()
            e.insert(0, "seed", seed)
            rho_rows.append(e)
    return pd.concat(aure_rows, ignore_index=True), pd.concat(rho_rows, ignore_index=True)


def _aure_by_dataset(rho_by_seed: pd.DataFrame) -> pd.DataFrame:
    """Dataset-unit AURE: mean ρ over seeds×axes within each (model, dataset)."""
    return (
        rho_by_seed.groupby(["model", "dataset_id"]).rho.mean().reset_index(name="aure_ds")
    )


def summarize_aure(aure_by_seed: pd.DataFrame, by_dataset: pd.DataFrame) -> pd.DataFrame:
    """Per-model AURE at both units with bootstrap 95% CIs."""
    rows: list[dict] = []
    seed_g = aure_by_seed.groupby("model", sort=True)
    ds_g = by_dataset.groupby("model", sort=True)
    for model, g in seed_g:
        sv = g["aure"].to_numpy(dtype=float)
        s_lo, s_hi = bootstrap_ci(sv)
        dv = ds_g.get_group(model)["aure_ds"].to_numpy(dtype=float)
        d_lo, d_hi = bootstrap_ci(dv)
        rows.append(
            {
                "model": model,
                "aure_mean": float(sv.mean()),
                "aure_sd": float(sv.std(ddof=1)) if sv.size > 1 else 0.0,
                "seed_ci95_lo": s_lo,
                "seed_ci95_hi": s_hi,
                "n_seeds": int(sv.size),
                "dataset_ci95_lo": d_lo,
                "dataset_ci95_hi": d_hi,
                "n_datasets": int(dv.size),
            }
        )
    return pd.DataFrame(rows).sort_values("aure_mean", ascending=False).reset_index(drop=True)


def pairwise_tests(aure_by_seed: pd.DataFrame, by_dataset: pd.DataFrame) -> pd.DataFrame:
    """Paired model comparison at the seed unit (primary) and dataset unit.

    Seed unit is paired across seeds (n=#seeds); dataset unit across datasets
    (n=#datasets). Holm correction is applied to the seed-unit p-value family.
    The pseudoreplicated seed×dataset×axis cell pool is deliberately NOT used.
    """
    seed_wide = aure_by_seed.pivot_table(index="seed", columns="model", values="aure")
    ds_wide = by_dataset.pivot_table(index="dataset_id", columns="model", values="aure_ds")
    models = list(seed_wide.columns)

    rows: list[dict] = []
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            sp = seed_wide[[a, b]].dropna()
            dp = ds_wide[[a, b]].dropna()
            s_diff = (sp[a] - sp[b]).to_numpy(dtype=float)
            d_diff = (dp[a] - dp[b]).to_numpy(dtype=float)
            s_lo, s_hi = bootstrap_ci(s_diff)
            d_lo, d_hi = bootstrap_ci(d_diff)
            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "seed_mean_delta": float(s_diff.mean()) if s_diff.size else float("nan"),
                    "seed_ci95_lo": s_lo,
                    "seed_ci95_hi": s_hi,
                    "seed_wilcoxon_p": _wilcoxon_p(sp[a].to_numpy(), sp[b].to_numpy()),
                    "n_seeds": int(s_diff.size),
                    "dataset_mean_delta": float(d_diff.mean()) if d_diff.size else float("nan"),
                    "dataset_ci95_lo": d_lo,
                    "dataset_ci95_hi": d_hi,
                    "n_datasets": int(d_diff.size),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["seed_p_holm"] = _holm(df["seed_wilcoxon_p"].tolist())
        # "Significant" only if Holm-corrected AND the seed-unit CI excludes 0.
        df["significant"] = (
            (df["seed_p_holm"] < 0.05)
            & ((df["seed_ci95_lo"] > 0) | (df["seed_ci95_hi"] < 0))
        )
        df = df.sort_values("seed_p_holm").reset_index(drop=True)
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-seed reliability run with statistics.")
    parser.add_argument("--config", type=Path, default=_ROOT / "configs" / "experiments" / "shift_stress.yaml")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--output-dir", type=Path, default=_ROOT / "results" / "tables")
    args = parser.parse_args(argv)

    seeds = _parse_seeds(args.seeds)
    print(f"[multiseed] config={args.config}  seeds={seeds}")
    aure_by_seed, rho_by_seed = collect_over_seeds(args.config, seeds)

    by_dataset = _aure_by_dataset(rho_by_seed)
    summary = summarize_aure(aure_by_seed, by_dataset)
    pairwise = pairwise_tests(aure_by_seed, by_dataset)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aure_by_seed.to_csv(out_dir / "aure_by_seed.csv", index=False)
    rho_by_seed.to_csv(out_dir / "rho_by_seed.csv", index=False)
    summary.to_csv(out_dir / "aure_multiseed.csv", index=False)
    pairwise.to_csv(out_dir / "aure_pairwise.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n[multiseed] AURE with bootstrap 95% CI (seed unit = within-datasets; dataset unit = generalisation):")
    print(summary.round(4).to_string(index=False))
    print("\n[multiseed] pairwise (paired at seed unit, Holm-corrected; dataset unit shown for generalisation):")
    print(pairwise.round(4).to_string(index=False))
    print(
        "\n[multiseed] CAVEAT: all CIs/tests are conditional on the failure thresholds and these "
        f"{summary['n_datasets'].max() if not summary.empty else 0} datasets; ICL (cloud/CUDA) "
        "backend jitter is folded into the seed unit but not the dataset unit."
    )
    print(f"[multiseed] wrote: {out_dir/'aure_multiseed.csv'}, {out_dir/'aure_pairwise.csv'}, "
          f"{out_dir/'aure_by_seed.csv'}, {out_dir/'rho_by_seed.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
