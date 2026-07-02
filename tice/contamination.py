"""Contamination / train-test independence diagnostics.

Two contamination vectors matter for this benchmark, and the shipped
``leakage_risk_level`` (exact train/test row duplicates) addresses neither well:

1. **Within-experiment independence.** The exact-duplicate check is blind to
   *near*-duplicates. On the toy sets up to ~98% of test points sit within
   0.5 sd of their nearest train point, so a ``'low'`` exact-dup verdict badly
   overstates how independent the test split is. :func:`proximity_profile` and
   :func:`independence_risk` quantify the near-duplicate structure the exact
   check misses.

2. **Foundation-model pretraining contamination.** TabPFN / TabICL are trained
   on synthetic priors, so they cannot have memorised named public datasets.
   :func:`dataset_exposure` tags dataset provenance; the honest validation is
   the positive control (run the envelope on fresh synthetic data and check the
   ICL>GBDT gap survives — it does, and is in fact larger on synthetic).

Distances are computed on numeric features only, standardised by *train* std,
and reported in per-feature-RMS sd units so a threshold like ``0.5`` reads as
"half a standard deviation, on average, from the nearest training row".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from tice.datasets.registry import Split

# sklearn toy sets are named, public, and widely scraped -> any web-trained
# model could in principle have seen them. Synthetic sets are generated locally
# from a seed and cannot be in any pretraining corpus.
PUBLIC_NAMED = frozenset({"breast_cancer", "wine", "iris", "digits"})

# Default near-duplicate radius (sd units) and the risk cut-offs on the fraction
# of test points inside it.
_DEFAULT_EPS = 0.5
_HI, _MED = 0.5, 0.1


def dataset_exposure(dataset_id: str) -> str:
    """'public_named' (pretraining-exposure risk) or 'synthetic' (uncontaminable)."""
    return "public_named" if dataset_id in PUBLIC_NAMED else "synthetic"


def nearest_train_distance(split: Split) -> np.ndarray:
    """Per-test-row distance to the nearest train row, in per-feature-RMS sd units.

    Returns an empty array when the split has no numeric columns.
    """
    num = list(split.numeric_columns)
    if not num or len(split.X_test) == 0 or len(split.X_train) == 0:
        return np.empty(0, dtype=float)
    # Median-impute (train stats) so datasets with missing values still get a
    # proximity proxy, consistent with the pipeline's numeric imputer.
    imputer = SimpleImputer(strategy="median").fit(split.X_train[num])
    x_tr = imputer.transform(split.X_train[num])
    x_te = imputer.transform(split.X_test[num])
    if x_tr.shape[1] == 0:  # every numeric column was all-NaN
        return np.empty(0, dtype=float)
    scaler = StandardScaler().fit(x_tr)
    x_tr = scaler.transform(x_tr)
    x_te = scaler.transform(x_te)
    nn = NearestNeighbors(n_neighbors=1).fit(x_tr)
    dist, _ = nn.kneighbors(x_te)
    return dist.ravel() / np.sqrt(x_tr.shape[1])


def exact_duplicate_rate(split: Split) -> float:
    """Fraction of test rows that appear verbatim in train (all columns)."""
    if len(split.X_test) == 0:
        return 0.0
    train_keys = set(map(tuple, split.X_train.to_numpy().tolist()))
    hits = sum(1 for row in split.X_test.to_numpy().tolist() if tuple(row) in train_keys)
    return float(hits / len(split.X_test))


def proximity_profile(
    split: Split, eps_sds: tuple[float, ...] = (0.0, 0.1, 0.5)
) -> dict[float, float]:
    """Fraction of test rows within each ``eps`` sd of the nearest train row."""
    dist = nearest_train_distance(split)
    if dist.size == 0:
        return {float(e): float("nan") for e in eps_sds}
    return {float(e): float(np.mean(dist <= e)) for e in eps_sds}


@dataclass(frozen=True)
class ContaminationReport:
    dataset_id: str
    exposure: str
    exact_dup_rate: float
    near_dup_rate: float  # fraction of test within `eps` sd of nearest train
    eps: float
    independence_risk: str  # 'low' | 'medium' | 'high'

    def as_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "exposure": self.exposure,
            "exact_dup_rate": self.exact_dup_rate,
            "near_dup_rate": self.near_dup_rate,
            "near_dup_eps_sd": self.eps,
            "independence_risk": self.independence_risk,
        }


def independence_risk(near_dup_rate: float, *, hi: float = _HI, med: float = _MED) -> str:
    """Upgraded risk that actually varies: keyed on the near-duplicate fraction."""
    if np.isnan(near_dup_rate):
        return "unknown"
    if near_dup_rate >= hi:
        return "high"
    if near_dup_rate >= med:
        return "medium"
    return "low"


def contamination_report(split: Split, *, eps: float = _DEFAULT_EPS) -> ContaminationReport:
    """Full per-split contamination diagnostic (near-dup aware + provenance)."""
    near = proximity_profile(split, (eps,))[float(eps)]
    return ContaminationReport(
        dataset_id=split.dataset_id,
        exposure=dataset_exposure(split.dataset_id),
        exact_dup_rate=exact_duplicate_rate(split),
        near_dup_rate=near,
        eps=float(eps),
        independence_risk=independence_risk(near),
    )
