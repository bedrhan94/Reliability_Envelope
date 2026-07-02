"""Offline tests for the external OpenML loader (no network)."""

from __future__ import annotations

import pandas as pd

from tice.datasets import external
from tice.datasets.registry import Dataset, available_datasets, load_dataset, register_dataset


def test_candidates_are_well_formed():
    assert len(external.CANDIDATES) >= 40
    assert all(isinstance(v, int) for v in external.CANDIDATES.values())
    # unique names and unique ids
    assert len(set(external.CANDIDATES)) == len(external.CANDIDATES)
    assert len(set(external.CANDIDATES.values())) == len(external.CANDIDATES)


def test_skip_reason_thresholds():
    assert external._skip_reason(100, 10) == f"n_samples<{external.MIN_SAMPLES}"
    assert "too large" in external._skip_reason(999999, 10)
    assert external._skip_reason(1000, 3) == f"n_features<{external.MIN_FEATURES}"
    assert "too wide" in external._skip_reason(1000, 99999)
    assert external._skip_reason(1000, 10) is None


def test_importing_external_registers_candidates():
    # importing the module (done above) must have registered the lazy factories
    registered = set(available_datasets())
    assert external.CANDIDATES.keys() <= registered


def test_register_dataset_roundtrip():
    tiny = Dataset(
        dataset_id="__unit_ext__",
        X=pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]}),
        y=pd.Series([0, 1], name="target"),
    )
    register_dataset("__unit_ext__", lambda: tiny)
    got = load_dataset("__unit_ext__")
    assert got.dataset_id == "__unit_ext__"
    assert got.X.shape == (2, 2)


def test_usable_datasets_filter():
    df = pd.DataFrame(
        {"dataset_id": ["a", "b", "c"], "usable_status": ["usable", "skip", "usable"]}
    )
    assert external.usable_datasets(df) == ["a", "c"]
