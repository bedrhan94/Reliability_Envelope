"""End-to-end orchestration: profiles -> shift sweep -> envelopes -> AURE.

Runs every model through every (dataset x shift axis x lambda) condition,
selects the GBDT reference, applies the failure rule, and reduces the per-lambda
table to reliability envelopes and AURE. Writes four CSVs into the configured
output directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tice.config import ExperimentConfig
from tice.datasets.profiler import profile_dataset
from tice.datasets.registry import Split, load_dataset, make_clean_split
from tice.envelope.reliability import compute_aure, compute_envelopes
from tice.metrics.utility import evaluate_failure
from tice.models.registry import get_model_spec
from tice.models.runner import run_model
from tice.seed import derive_seed, make_rng, seed_everything
from tice.shifts.generators import get_shift, is_applicable
from tice.shifts.splits import check_clean_split, make_shift_variant

# Stable column order for the shift results table.
SHIFT_RESULT_COLUMNS = [
    "dataset_id",
    "model",
    "family",
    "shift_axis",
    "shift_lambda",
    "seed",
    "split_policy",
    "clean_split_passed",
    "leakage_risk_level",
    "applicable",
    "status",
    "error_message",
    "accuracy",
    "auc",
    "nll",
    "nll_norm",
    "ece",
    "utility",
    "n_train",
    "best_gbdt",
    "reference_utility",
    "failed",
    "failure_reason",
    "metric_notes",
]


@dataclass(frozen=True)
class PipelineOutputs:
    dataset_profiles: pd.DataFrame
    shift_results: pd.DataFrame
    reliability_envelopes: pd.DataFrame
    aure_summary: pd.DataFrame
    paths: dict[str, Path]


def _select_best_gbdt(
    clean_utilities: dict[str, float],
    families: dict[str, str],
    reference_models: tuple[str, ...],
) -> tuple[str | None, float | None]:
    """best_gbdt = argmax clean-ID utility among GBDT-family reference models.

    Falls back to *any* available GBDT-family model when none of the configured
    reference models ran (keeps the smoke config working without xgb/catboost).
    """
    def _candidates(names: list[str]) -> list[str]:
        return [
            m
            for m in names
            if families.get(m) == "gbdt"
            and m in clean_utilities
            and pd.notna(clean_utilities[m])
        ]

    candidates = _candidates(list(reference_models))
    if not candidates:
        candidates = _candidates(list(clean_utilities))
    if not candidates:
        return None, None
    best = max(candidates, key=lambda m: clean_utilities[m])
    return best, float(clean_utilities[best])


def _run_one(
    dataset_id: str,
    model: str,
    clean_split: Split,
    axis: str,
    lam: float,
    *,
    config: ExperimentConfig,
    clean_split_passed: bool,
) -> dict:
    """Generate the shifted variant, run the model, return a raw result row."""
    variant = make_shift_variant(
        clean_split,
        axis,
        lam,
        base_seed=config.seed,
        clean_split_passed=clean_split_passed,
    )
    run_seed = derive_seed(config.seed, dataset_id, model, lam)
    result = run_model(
        model,
        variant.split,
        seed=run_seed,
        ece_bins=config.ece_bins,
        max_context_rows=config.max_context_rows,
    )
    row = variant.metadata()
    row["applicable"] = True
    row.update(result.as_dict())
    return row


def _probe_split(seed: int) -> Split:
    """Tiny numeric 2-class split used to check a model actually runs.

    Lets us detect models that import but cannot run (e.g. an ICL backend that
    needs interactive license/auth) *once*, instead of failing -- and possibly
    launching a browser -- on every condition.
    """
    rng = make_rng(seed, "preflight_probe")
    n = 40
    x0 = rng.normal(0.0, 1.0, size=(n, 4))
    x1 = rng.normal(1.5, 1.0, size=(n, 4))
    X = pd.DataFrame(
        np.vstack([x0, x1]), columns=[f"f{i}" for i in range(4)]
    )
    y = pd.Series([0] * n + [1] * n, name="target")
    return Split(
        dataset_id="__preflight__",
        X_train=X,
        y_train=y,
        X_test=X.iloc[::2].reset_index(drop=True),
        y_test=y.iloc[::2].reset_index(drop=True),
        categorical_columns=(),
        classes=(0, 1),
    )


def _preflight_models(config: ExperimentConfig) -> dict[str, tuple[bool, str]]:
    """Probe each configured model once; return usable flag + reason."""
    probe = _probe_split(config.seed)
    usable: dict[str, tuple[bool, str]] = {}
    for model in config.models:
        spec = get_model_spec(model)
        if not spec.available():
            usable[model] = (False, f"optional package '{spec.required_package}' not installed")
            continue
        seed = derive_seed(config.seed, "preflight", model)
        result = run_model(
            model, probe, seed=seed, ece_bins=config.ece_bins,
            max_context_rows=config.max_context_rows,
        )
        usable[model] = (result.status_ok, "" if result.status_ok else result.error_message)
    return usable


def _skipped_row(
    *,
    dataset_id: str,
    model: str,
    family: str,
    axis: str,
    lam: float,
    split_policy: str,
    clean_passed: bool,
    seed: int,
    applicable: bool,
    reason: str,
) -> dict:
    return {
        "dataset_id": dataset_id,
        "shift_axis": axis,
        "shift_lambda": float(lam),
        "seed": seed,
        "split_policy": split_policy,
        "clean_split_passed": clean_passed,
        "leakage_risk_level": "low",
        "applicable": applicable,
        "model": model,
        "family": family,
        "status": "skipped",
        "error_message": reason,
        "accuracy": float("nan"),
        "auc": float("nan"),
        "nll": float("nan"),
        "nll_norm": float("nan"),
        "ece": float("nan"),
        "utility": float("nan"),
        "n_train": 0,
        "metric_notes": "",
    }


def run_pipeline(config: ExperimentConfig) -> PipelineOutputs:
    seed_everything(config.seed)
    usable = _preflight_models(config)

    profile_rows: list[dict] = []
    raw_rows: list[dict] = []
    families = {m: get_model_spec(m).family for m in config.models}

    for dataset_id in config.datasets:
        dataset = load_dataset(dataset_id)
        clean_split = make_clean_split(
            dataset, base_seed=config.seed, test_size=config.test_size
        )
        clean_passed = check_clean_split(clean_split)
        profile_rows.append(
            profile_dataset(
                dataset, base_seed=config.seed, test_size=config.test_size, split=clean_split
            ).as_dict()
        )

        for axis in config.shift_axes:
            applicable = is_applicable(axis, clean_split)
            shift = get_shift(axis)
            for model in config.models:
                model_ok, model_reason = usable[model]
                # Decide whether this (axis, model) cell can actually run.
                if not applicable:
                    reason = "shift not applicable: dataset has no categorical features"
                elif not model_ok:
                    reason = f"model unavailable: {model_reason}"
                else:
                    reason = None

                if reason is not None:
                    # Emit skipped rows so the table is complete and honest.
                    for lam in config.lambda_values:
                        raw_rows.append(
                            _skipped_row(
                                dataset_id=dataset_id,
                                model=model,
                                family=families[model],
                                axis=axis,
                                lam=lam,
                                split_policy=shift.split_policy,
                                clean_passed=clean_passed,
                                seed=derive_seed(config.seed, dataset_id, axis, lam),
                                applicable=applicable,
                                reason=reason,
                            )
                        )
                    continue
                for lam in config.lambda_values:
                    raw_rows.append(
                        _run_one(
                            dataset_id,
                            model,
                            clean_split,
                            axis,
                            lam,
                            config=config,
                            clean_split_passed=clean_passed,
                        )
                    )

    shift_results = _finalize_results(pd.DataFrame(raw_rows), config)
    profiles = pd.DataFrame(profile_rows)
    envelopes = compute_envelopes(shift_results)
    aure = compute_aure(envelopes)

    paths = _write_outputs(config.output_dir, profiles, shift_results, envelopes, aure)
    return PipelineOutputs(profiles, shift_results, envelopes, aure, paths)


def _finalize_results(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Attach GBDT reference, failure flags, and order columns."""
    families = {m: get_model_spec(m).family for m in config.models}

    # Per-dataset clean-ID utility for each model (lambda == 0 rows are clean).
    clean_mask = (df["shift_lambda"] == 0.0) & (df["status"] == "ok")
    clean_util: dict[str, dict[str, float]] = {}
    for (ds, model), g in df[clean_mask].groupby(["dataset_id", "model"]):
        clean_util.setdefault(ds, {})[model] = float(g["utility"].iloc[0])

    best_gbdt: dict[str, str | None] = {}
    clean_ref_util: dict[str, float | None] = {}
    for ds in df["dataset_id"].unique():
        best, util = _select_best_gbdt(
            clean_util.get(ds, {}), families, config.gbdt_reference_models
        )
        best_gbdt[ds] = best
        clean_ref_util[ds] = util

    # For "matched" mode, reference is the best GBDT utility at the same condition.
    matched_ref: dict[tuple, float] = {}
    if config.gbdt_reference == "matched":
        gbdt_rows = df[df["family"] == "gbdt"]
        for keys, g in gbdt_rows.groupby(["dataset_id", "shift_axis", "shift_lambda"]):
            util = g["utility"].dropna()
            if not util.empty:
                matched_ref[keys] = float(util.max())

    def _reference(row: pd.Series) -> float:
        ds = row["dataset_id"]
        if config.gbdt_reference == "matched":
            return matched_ref.get(
                (ds, row["shift_axis"], row["shift_lambda"]), float("nan")
            )
        return clean_ref_util.get(ds) if clean_ref_util.get(ds) is not None else float("nan")

    failed_flags: list[bool] = []
    failure_reasons: list[str] = []
    references: list[float] = []
    best_list: list[str] = []
    for _, row in df.iterrows():
        ref = _reference(row)
        references.append(ref if ref is not None else float("nan"))
        best_list.append(best_gbdt.get(row["dataset_id"]) or "")
        res = evaluate_failure(
            utility=row["utility"],
            ece=row["ece"],
            nll_norm=row["nll_norm"],
            reference_utility=ref,
            thresholds=config.thresholds,
            status_ok=(row["status"] == "ok"),
        )
        failed_flags.append(res.failed)
        failure_reasons.append(res.reason_str)

    df = df.copy()
    df["best_gbdt"] = best_list
    df["reference_utility"] = references
    df["failed"] = failed_flags
    df["failure_reason"] = failure_reasons

    for col in SHIFT_RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[SHIFT_RESULT_COLUMNS].sort_values(
        ["dataset_id", "shift_axis", "model", "shift_lambda"]
    ).reset_index(drop=True)


def _write_outputs(
    output_dir: Path,
    profiles: pd.DataFrame,
    shift_results: pd.DataFrame,
    envelopes: pd.DataFrame,
    aure: pd.DataFrame,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "dataset_profiles": output_dir / "dataset_profiles.csv",
        "shift_results": output_dir / "shift_results.csv",
        "reliability_envelopes": output_dir / "reliability_envelopes.csv",
        "aure_summary": output_dir / "aure_summary.csv",
    }
    profiles.to_csv(paths["dataset_profiles"], index=False)
    shift_results.to_csv(paths["shift_results"], index=False)
    envelopes.to_csv(paths["reliability_envelopes"], index=False)
    aure.to_csv(paths["aure_summary"], index=False)
    return paths
