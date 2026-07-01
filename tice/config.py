"""YAML experiment configuration loaded into typed dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_LAMBDA_VALUES: tuple[float, ...] = (0.00, 0.05, 0.10, 0.20, 0.30, 0.40)
DEFAULT_SHIFT_AXES: tuple[str, ...] = (
    "label_noise",
    "class_imbalance",
    "missingness_shift",
    "rare_category_shift",
    "covariate_shift",
    "context_budget",
)


@dataclass(frozen=True)
class Thresholds:
    """Failure-indicator thresholds (item 6 of the spec)."""

    tau_utility: float = 0.03
    tau_ece: float = 0.10
    tau_nll: float = 0.75


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything ``run_shift_stress`` needs to reproduce a run."""

    seed: int = 42
    output_dir: Path = Path("results/tables")
    datasets: tuple[str, ...] = ("breast_cancer", "wine", "iris", "synthetic_mixed")
    models: tuple[str, ...] = ("logreg", "hist_gbdt")
    shift_axes: tuple[str, ...] = DEFAULT_SHIFT_AXES
    lambda_values: tuple[float, ...] = DEFAULT_LAMBDA_VALUES
    test_size: float = 0.3
    ece_bins: int = 15
    thresholds: Thresholds = field(default_factory=Thresholds)
    # Which model family selects the GBDT reference, and how the failure bar
    # tracks the shift. "clean" => fixed clean-ID bar per dataset (default,
    # plain reading of the spec); "matched" => bar recomputed at each lambda.
    gbdt_reference: str = "clean"
    gbdt_reference_models: tuple[str, ...] = ("xgboost", "catboost")
    # Cap the number of train rows handed to memory-bound ICL models.
    max_context_rows: int = 1000

    @property
    def max_lambda(self) -> float:
        return max(self.lambda_values)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML file."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    thresholds_raw = raw.get("thresholds", {}) or {}
    thresholds = Thresholds(
        tau_utility=float(thresholds_raw.get("tau_utility", 0.03)),
        tau_ece=float(thresholds_raw.get("tau_ece", 0.10)),
        tau_nll=float(thresholds_raw.get("tau_nll", 0.75)),
    )

    def _tuple(key: str, default: tuple) -> tuple:
        value = raw.get(key)
        return tuple(value) if value is not None else default

    return ExperimentConfig(
        seed=int(raw.get("seed", 42)),
        output_dir=Path(raw.get("output_dir", "results/tables")),
        datasets=_tuple("datasets", ExperimentConfig.datasets),
        models=_tuple("models", ExperimentConfig.models),
        shift_axes=_tuple("shift_axes", DEFAULT_SHIFT_AXES),
        lambda_values=tuple(
            float(v) for v in _tuple("lambda_values", DEFAULT_LAMBDA_VALUES)
        ),
        test_size=float(raw.get("test_size", 0.3)),
        ece_bins=int(raw.get("ece_bins", 15)),
        thresholds=thresholds,
        gbdt_reference=str(raw.get("gbdt_reference", "clean")),
        gbdt_reference_models=_tuple(
            "gbdt_reference_models", ExperimentConfig.gbdt_reference_models
        ),
        max_context_rows=int(raw.get("max_context_rows", 1000)),
    )
