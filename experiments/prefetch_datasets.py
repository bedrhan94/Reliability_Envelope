"""Warm the OpenML cache before a long run, with retries.

An 8-hour benchmark that fetches datasets lazily is hostage to a transient network
error: OpenML returned HTTP 503 mid-run twice here, and because the multiseed runner
only writes its tables at the end, the second failure discarded four hours of work.

Fetching every dataset up front removes the network from the long run entirely --
afterwards every ``fetch_openml`` call is a local cache hit. Failures are reported
per dataset with the exception, and the script exits non-zero if any dataset is
still missing, so a run is never started against a partial cache.

Usage::

    python experiments/prefetch_datasets.py --config <config.yaml>
    python experiments/prefetch_datasets.py --all      # every registered external set
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import tice.datasets.external as ext  # noqa: E402


def fetch_with_retry(name: str, openml_id: int, *, attempts: int = 6,
                     base_delay: float = 5.0) -> tuple[bool, str]:
    """Fetch one dataset, backing off exponentially on transient failures."""
    for attempt in range(1, attempts + 1):
        try:
            ds = ext.load_external(name, openml_id)
            return True, f"{len(ds.X)} rows x {ds.X.shape[1]} cols"
        except Exception as exc:  # network, parser, or gateway errors alike
            if attempt == attempts:
                return False, f"{type(exc).__name__}: {exc}"
            delay = base_delay * (2 ** (attempt - 1))
            print(f"    attempt {attempt}/{attempts} failed ({type(exc).__name__}); "
                  f"retrying in {delay:.0f}s")
            time.sleep(delay)
    return False, "unreachable"


def wanted(args) -> list[tuple[str, int]]:
    ids = dict(ext.CANDIDATES)
    if args.config:
        raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        names = list(raw.get("datasets") or [])
    else:
        names = list(ids)
    out = []
    for n in names:
        oid = ids.get(n)
        if oid is None:
            print(f"  ! {n}: no OpenML id registered, skipping")
            continue
        out.append((n, int(oid) if not isinstance(oid, tuple) else int(oid[0])))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--attempts", type=int, default=6)
    args = p.parse_args(argv)

    targets = wanted(args)
    if not targets:
        print("[prefetch] nothing to fetch -- could not resolve dataset ids")
        return 1

    print(f"[prefetch] warming {len(targets)} datasets")
    failed = []
    for i, (name, oid) in enumerate(targets, 1):
        ok, detail = fetch_with_retry(name, oid, attempts=args.attempts)
        mark = "ok  " if ok else "FAIL"
        print(f"  [{i:>2}/{len(targets)}] {mark} {name} (id {oid}): {detail}", flush=True)
        if not ok:
            failed.append(name)

    if failed:
        print(f"\n[prefetch] {len(failed)} dataset(s) still missing: {failed}")
        print("[prefetch] do NOT start the run against a partial cache")
        return 1
    print(f"\n[prefetch] all {len(targets)} datasets cached; the run will not touch the network")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
