"""Reliability Envelope Radius (item 7) and AURE (item 8).

``rho`` is the largest shift severity a model survives *contiguously* from
lambda=0 upward: the maximum ``lambda`` such that the model has not failed for
every ``lambda' <= lambda``. AURE averages ``rho`` across datasets and shift
axes (per model).
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def envelope_radius(lambda_values: Sequence[float], failed: Sequence[bool]) -> float:
    """Largest lambda with an unbroken run of passes from the smallest lambda.

    * fails at the smallest lambda            -> 0.0
    * never fails                             -> max(lambda_values)
    * fails in the middle (passes again later)-> last lambda before the first
      failure (the "for all lambda' <= lambda" clause caps it there)
    """
    pairs = sorted(zip(lambda_values, failed, strict=True), key=lambda p: p[0])
    last_passed: float | None = None
    for lam, fail in pairs:
        if fail:
            break
        last_passed = float(lam)
    return 0.0 if last_passed is None else last_passed


def _first_crossing(
    lams: Sequence[float], vals: Sequence[float], thr: float, *, fail_above: bool
) -> float | None:
    """Interpolated ``lambda`` at which ``vals`` first crosses ``thr`` into failure.

    ``fail_above``: the criterion fails when the value goes *above* ``thr`` (ECE,
    NLL) vs *below* it (utility). Linear interpolation locates the crossing
    between the two bracketing grid points; a NaN value counts as an immediate
    failure at that lambda. Returns ``None`` if the criterion never fails.
    """
    prev_l: float | None = None
    prev_v: float | None = None
    for lam, v in zip(lams, vals, strict=True):
        lam = float(lam)
        failed = (v != v) or (v > thr if fail_above else v < thr)  # v!=v => NaN
        if failed:
            if prev_l is None or prev_v is None or v != v or prev_v == v:
                return lam  # fails at the first point, or no usable slope
            # prev passed, current fails -> interpolate where value == thr
            frac = (thr - prev_v) / (v - prev_v)
            frac = min(max(frac, 0.0), 1.0)
            return prev_l + frac * (lam - prev_l)
        prev_l, prev_v = lam, v
    return None


def continuous_envelope_radius(
    lambda_values: Sequence[float],
    utility: Sequence[float],
    ece: Sequence[float],
    nll_norm: Sequence[float],
    reference_utility: float,
    *,
    tau_utility: float,
    tau_ece: float,
    tau_nll: float,
) -> float:
    """De-quantised ``rho``: the interpolated lambda of the earliest threshold crossing.

    The grid ``envelope_radius`` snaps rho to one of the sampled lambdas, so two
    models can differ by a whole grid step (or none) for reasons of quantisation
    alone. This locates the failure boundary *between* grid points -- the
    earliest lambda at which any of the three criteria (utility gap / ECE / NLL)
    crosses its threshold -- and reduces to the grid value when snapped. Falls
    back to ``max(lambda)`` (right-censored) when nothing crosses.
    """
    order = sorted(range(len(lambda_values)), key=lambda i: lambda_values[i])
    lams = [float(lambda_values[i]) for i in order]
    crossings = []
    if reference_utility == reference_utility:  # reference defined (not NaN)
        c = _first_crossing(
            lams, [utility[i] for i in order], reference_utility - tau_utility, fail_above=False
        )
        if c is not None:
            crossings.append(c)
    for series, thr in (([ece[i] for i in order], tau_ece), ([nll_norm[i] for i in order], tau_nll)):
        c = _first_crossing(lams, series, thr, fail_above=True)
        if c is not None:
            crossings.append(c)
    return min(crossings) if crossings else max(lams)


def compute_envelopes(
    shift_results: pd.DataFrame,
    *,
    group_keys: tuple[str, ...] = ("model", "dataset_id", "shift_axis"),
) -> pd.DataFrame:
    """Reduce the per-lambda shift results to one ``rho`` per model/dataset/axis.

    Groups with no successful run (every row ``skipped`` or ``error`` -- e.g. a
    missing/unauthenticated model, or a shift not applicable to the dataset) are
    dropped: ``rho`` is only meaningful once the model has produced at least the
    clean baseline. Expects ``shift_lambda``, ``status`` and ``failed`` columns.
    """
    rows: list[dict] = []
    for keys, g in shift_results.groupby(list(group_keys), sort=True):
        if (g["status"] != "ok").all():
            continue
        rho = envelope_radius(g["shift_lambda"].tolist(), g["failed"].astype(bool).tolist())
        row = dict(zip(group_keys, keys if isinstance(keys, tuple) else (keys,), strict=True))
        row.update(
            {
                "rho": rho,
                "n_lambda": int(len(g)),
                "n_failed": int(g["failed"].astype(bool).sum()),
                "max_lambda": float(g["shift_lambda"].max()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def compute_aure(envelopes: pd.DataFrame) -> pd.DataFrame:
    """Average ``rho`` per model (overall and per shift axis)."""
    rows: list[dict] = []
    for model, g in envelopes.groupby("model", sort=True):
        row: dict = {
            "model": model,
            "aure": float(g["rho"].mean()),
            "n_envelopes": int(len(g)),
        }
        for axis, ga in g.groupby("shift_axis", sort=True):
            row[f"aure_{axis}"] = float(ga["rho"].mean())
        rows.append(row)
    return pd.DataFrame(rows)
