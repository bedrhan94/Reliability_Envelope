"""External stratified MULTISEED runner (Sprint 6).

Runs the shift-stress pipeline on the external stratified subset over several
seeds and emits both the concatenated per-seed tables (with a `seed` column) and
the multi-seed statistics. Reuses `run_pipeline` unchanged (per seed), the
external dataset registry, and the statistics helpers from `run_multiseed.py`
and `make_run_audit.py` — the single-seed pipeline behaviour is untouched.

Reads `seeds:` from the YAML (or --seeds override). Writes into the config's
output_dir:
  shift_results.csv, reliability_envelopes.csv  (with `seed` column, all seeds)
  aure_by_seed.csv        (per-seed per-model AURE)
  aure_summary.csv        (per-model AURE averaged over seeds)
  aure_multiseed.csv      (mean, sd, bootstrap seed-CI and dataset-CI)
  pairwise_tests.csv      (paired Wilcoxon + Holm, seed & dataset units)
  run_audit.csv

Usage::

    TABPFN_TOKEN=<key> python experiments/run_external_multiseed.py \
        --config configs/experiments/shift_stress_external_2axis_stratified_multiseed.yaml
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling experiment scripts

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from make_run_audit import build_audit  # noqa: E402
from run_multiseed import _aure_by_dataset, pairwise_tests, summarize_aure  # noqa: E402

import tice.datasets.external  # noqa: E402,F401  (registers external datasets)
from tice.config import load_config  # noqa: E402
from tice.envelope.reliability import compute_aure  # noqa: E402
from tice.pipeline import run_pipeline  # noqa: E402


def _parse_seeds(raw: dict, override: str | None) -> list[int]:
    if override:
        return [int(s) for s in override.replace(" ", "").split(",") if s]
    seeds = raw.get("seeds") or [raw.get("seed", 42)]
    return [int(s) for s in seeds]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="External stratified multiseed run.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--seeds", type=str, default=None, help="override, e.g. 42,1337,2025")
    args = p.parse_args(argv)

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    seeds = _parse_seeds(raw, args.seeds)
    base = load_config(args.config)
    out_dir = Path(base.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ext-ms] datasets={len(base.datasets)} models={list(base.models)} "
          f"axes={list(base.shift_axes)} seeds={seeds}")

    # Per-seed checkpoints. A multi-hour run that only writes at the end loses
    # everything to one transient cloud error -- which happened twice here. Each
    # completed seed is persisted immediately, and an existing checkpoint is reused
    # instead of recomputed, so a crashed run resumes where it stopped.
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    sr_all, env_all, aure_seed = [], [], []
    with tempfile.TemporaryDirectory(prefix="tice_ext_ms_") as tmp:
        for seed in seeds:
            sr_ck = ckpt_dir / f"seed_{seed}_shift_results.csv"
            en_ck = ckpt_dir / f"seed_{seed}_envelopes.csv"
            if sr_ck.exists() and en_ck.exists():
                print(f"[ext-ms] seed={seed} resumed from checkpoint", flush=True)
                sr, en = pd.read_csv(sr_ck), pd.read_csv(en_ck)
            else:
                cfg = replace(base, seed=seed, output_dir=Path(tmp) / f"seed_{seed}")
                print(f"[ext-ms] seed={seed} ...", flush=True)
                out = run_pipeline(cfg)
                sr = out.shift_results.copy().rename(columns={"seed": "variant_seed"})
                sr["base_seed"] = seed
                en = out.reliability_envelopes.copy()
                en["base_seed"] = seed
                sr.to_csv(sr_ck, index=False)
                en.to_csv(en_ck, index=False)
                print(f"[ext-ms] seed={seed} checkpointed ({len(sr)} rows)", flush=True)
            sr_all.append(sr)
            env_all.append(en)
            a = (compute_aure(en.drop(columns=["base_seed"], errors="ignore"))
                 .assign(seed=seed))
            aure_seed.append(a[["seed", "model", "aure"]])

    shift_results = pd.concat(sr_all, ignore_index=True)
    envelopes = pd.concat(env_all, ignore_index=True)
    aure_by_seed = pd.concat(aure_seed, ignore_index=True)

    by_dataset = _aure_by_dataset(envelopes[["model", "dataset_id", "rho"]])
    multiseed = summarize_aure(aure_by_seed, by_dataset)
    pairwise = pairwise_tests(aure_by_seed, by_dataset)
    aure_summary = (
        aure_by_seed.groupby("model", as_index=False).aure.mean()
        .sort_values("aure", ascending=False)
    )
    audit = build_audit(shift_results)

    shift_results.to_csv(out_dir / "shift_results.csv", index=False)
    envelopes.to_csv(out_dir / "reliability_envelopes.csv", index=False)
    aure_by_seed.to_csv(out_dir / "aure_by_seed.csv", index=False)
    aure_summary.to_csv(out_dir / "aure_summary.csv", index=False)
    multiseed.to_csv(out_dir / "aure_multiseed.csv", index=False)
    pairwise.to_csv(out_dir / "pairwise_tests.csv", index=False)
    audit.to_csv(out_dir / "run_audit.csv", index=False)
    # carry per-seed=first profiles for figure joins
    prof = pd.read_csv(_ROOT / "results" / "external" / "dataset_profiles_external.csv")
    prof[prof.dataset_id.isin(base.datasets)].to_csv(out_dir / "dataset_profiles.csv", index=False)

    run = audit[audit.level == "run"].iloc[0]
    print(f"[ext-ms] rows={len(shift_results)} expected={int(run.total_expected)} "
          f"ok={int(run.ok_rows)} skipped={int(run.skipped_rows)} error={int(run.error_rows)}")
    print("[ext-ms] AURE (mean over seeds):")
    print(aure_summary.round(4).to_string(index=False))
    print(f"[ext-ms] wrote 7 tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
