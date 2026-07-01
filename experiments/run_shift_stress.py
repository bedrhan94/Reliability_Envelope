"""CLI entry point for the controlled shift stress suite (M1-M3).

Usage::

    uv run python experiments/run_shift_stress.py --config configs/experiments/shift_stress.yaml

Runs the dataset profiler, the shift stress sweep, the reliability-envelope
reduction, and AURE, writing four CSVs into the config's ``output_dir``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `tice` importable when invoked as a plain script (no editable install).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tice.config import load_config  # noqa: E402
from tice.pipeline import run_pipeline  # noqa: E402

DEFAULT_CONFIG = _ROOT / "configs" / "experiments" / "shift_stress.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the tabular shift stress suite.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the experiment YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output directory from the config.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.output_dir is not None:
        config = config.__class__(**{**config.__dict__, "output_dir": args.output_dir})

    print(f"[tice] config:        {args.config}")
    print(f"[tice] seed:          {config.seed}")
    print(f"[tice] datasets:      {list(config.datasets)}")
    print(f"[tice] models:        {list(config.models)}")
    print(f"[tice] shift axes:    {list(config.shift_axes)}")
    print(f"[tice] lambda values: {list(config.lambda_values)}")
    print(f"[tice] gbdt reference: {config.gbdt_reference}")

    outputs = run_pipeline(config)

    n_total = len(outputs.shift_results)
    n_ok = int((outputs.shift_results["status"] == "ok").sum())
    n_skipped = int((outputs.shift_results["status"] == "skipped").sum())
    n_error = int((outputs.shift_results["status"] == "error").sum())

    print("\n[tice] done.")
    print(f"  rows: {n_total} (ok={n_ok}, skipped={n_skipped}, error={n_error})")
    print("  outputs:")
    for name, path in outputs.paths.items():
        print(f"    - {name}: {path}")

    if not outputs.aure_summary.empty:
        print("\n[tice] AURE summary:")
        cols = ["model", "aure", "n_envelopes"]
        print(outputs.aure_summary[cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
