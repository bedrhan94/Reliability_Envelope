"""Run the shift-stress pipeline on EXTERNAL OpenML datasets (2-axis validation).

Thin wrapper: importing `tice.datasets.external` registers the external datasets
into the shared registry, after which the canonical `run_pipeline` runs unchanged.
Use the 2-axis external configs (label_noise + covariate_shift only). Canonical
results are untouched because these configs write to `results/external/...`.

Usage::

    python experiments/run_external_shift.py --config configs/experiments/shift_stress_external_2axis_smoke.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import tice.datasets.external  # noqa: E402,F401  (registers external datasets on import)
from tice.config import load_config  # noqa: E402
from tice.pipeline import run_pipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="External 2-axis shift-stress run.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)

    config = load_config(args.config)
    if args.output_dir is not None:
        config = config.__class__(**{**config.__dict__, "output_dir": args.output_dir})

    print(f"[external-run] config={args.config}")
    print(f"[external-run] datasets={len(config.datasets)} models={list(config.models)} axes={list(config.shift_axes)}")
    outputs = run_pipeline(config)

    sr = outputs.shift_results
    n_ok = int((sr["status"] == "ok").sum())
    n_sk = int((sr["status"] == "skipped").sum())
    n_er = int((sr["status"] == "error").sum())
    print(f"[external-run] rows={len(sr)} ok={n_ok} skipped={n_sk} error={n_er}")
    if n_er:
        for _, r in sr[sr["status"] == "error"].head(10).iterrows():
            print(f"    ERROR {r['dataset_id']}/{r['model']}/{r['shift_axis']}: {str(r['error_message'])[:120]}")
    for name, path in outputs.paths.items():
        print(f"    - {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
