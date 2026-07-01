"""Central, deterministic seed management.

Every stochastic step in the pipeline derives its seed from a single base seed
plus a tuple of string keys (dataset id, shift axis, lambda, model, purpose).
The same keys always map to the same seed, so the whole stress suite is
reproducible and order-independent.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np

_MAX_UINT32 = 2**32


def derive_seed(base_seed: int, *keys: object) -> int:
    """Derive a stable 32-bit seed from ``base_seed`` and arbitrary keys.

    The mapping is a SHA-256 hash of the stringified keys, so it does not depend
    on Python's per-process hash randomization and is stable across runs.
    """
    payload = "|".join([str(base_seed), *(str(k) for k in keys)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % _MAX_UINT32


def make_rng(base_seed: int, *keys: object) -> np.random.Generator:
    """Return a NumPy ``Generator`` seeded deterministically from the keys."""
    return np.random.default_rng(derive_seed(base_seed, *keys))


def seed_everything(base_seed: int, extra: Iterable[object] = ()) -> int:
    """Seed Python's ``random`` and NumPy's legacy global RNG.

    Returns the concrete seed used. Library-level randomness (sklearn, xgboost,
    ...) is controlled explicitly through ``random_state`` arguments, but some
    backends consult the global state, so we pin it too.
    """
    import random

    seed = derive_seed(base_seed, *extra)
    random.seed(seed)
    np.random.seed(seed)
    # Seed torch too when available (ICL backends use it). Note: CUDA ops are
    # not bitwise-reproducible, so ICL metrics still jitter in the low decimals.
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    return seed
