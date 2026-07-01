"""Dataset registry and profiling."""

from __future__ import annotations

from tice.datasets.registry import (
    Dataset,
    Split,
    available_datasets,
    load_dataset,
    make_clean_split,
)

__all__ = [
    "Dataset",
    "Split",
    "available_datasets",
    "load_dataset",
    "make_clean_split",
]
