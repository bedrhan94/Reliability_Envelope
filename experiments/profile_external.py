"""Fetch + profile the curated external OpenML datasets.

Writes `results/external/dataset_profiles_external.csv` (one row per candidate,
usable or skipped with a reason) and prints the usable/skip summary. Run once
before the smoke / full external runs; the usable ids feed the configs.

Usage::

    python experiments/profile_external.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tice.datasets.external import CANDIDATES, build_profiles, usable_datasets  # noqa: E402


def main() -> int:
    print(f"[external] profiling {len(CANDIDATES)} candidate datasets ...")
    profiles = build_profiles()

    out_dir = _ROOT / "results" / "external"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / "dataset_profiles_external.csv"
    profiles.to_csv(csv, index=False)

    usable = usable_datasets(profiles)
    n_usable = len(usable)
    n_skip = len(profiles) - n_usable
    print(f"\n[external] candidates={len(profiles)}  usable={n_usable}  skipped={n_skip}")
    print("[external] skip reasons:")
    for _, r in profiles[profiles.usable_status == "skip"].iterrows():
        print(f"    {r['dataset_id']:32s} {r['skip_reason']}")
    print(f"\n[external] usable ids ({n_usable}):\n  " + ",".join(usable))
    print(f"\n[external] wrote {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
