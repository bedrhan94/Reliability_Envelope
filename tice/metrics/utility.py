"""Reliability utility score (item 4) and failure indicator (item 6).

    U = 0.35 * AUC + 0.15 * Accuracy - 0.30 * NLL_norm - 0.20 * ECE
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tice.config import Thresholds

W_AUC = 0.35
W_ACC = 0.15
W_NLL = 0.30
W_ECE = 0.20


def reliability_utility(auc: float, accuracy: float, nll_norm: float, ece: float) -> float:
    """Compute the reliability utility ``U``.

    Returns ``nan`` if any component is undefined, so downstream code treats an
    un-evaluable model as failed rather than silently scoring it.
    """
    components = (auc, accuracy, nll_norm, ece)
    if any(c is None or (isinstance(c, float) and math.isnan(c)) for c in components):
        return float("nan")
    return float(W_AUC * auc + W_ACC * accuracy - W_NLL * nll_norm - W_ECE * ece)


@dataclass(frozen=True)
class FailureResult:
    failed: bool
    reasons: tuple[str, ...]

    @property
    def reason_str(self) -> str:
        return ";".join(self.reasons)


def evaluate_failure(
    *,
    utility: float,
    ece: float,
    nll_norm: float,
    reference_utility: float | None,
    thresholds: Thresholds,
    status_ok: bool = True,
) -> FailureResult:
    """Apply the failure rule.

    A model fails when it could not be evaluated, OR its utility falls more than
    ``tau_utility`` below the GBDT reference, OR its ECE exceeds ``tau_ece``, OR
    its normalized NLL exceeds ``tau_nll``.
    """
    reasons: list[str] = []

    if not status_ok or utility is None or math.isnan(utility):
        return FailureResult(True, ("undefined",))

    if (
        reference_utility is not None
        and not math.isnan(reference_utility)
        and utility < reference_utility - thresholds.tau_utility
    ):
        reasons.append("utility_gap")

    if not math.isnan(ece) and ece > thresholds.tau_ece:
        reasons.append("ece")

    if not math.isnan(nll_norm) and nll_norm > thresholds.tau_nll:
        reasons.append("nll")

    return FailureResult(bool(reasons), tuple(reasons))
