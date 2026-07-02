"""External OpenML classification datasets for broad validation.

Fetched on demand via ``sklearn.datasets.fetch_openml`` (cached under the sklearn
data home) -- nothing is bundled in the repo. A curated list of OpenML-CC18-style
classification datasets is registered into the shared dataset registry, so the
existing pipeline can run them with no changes; the loader returns the same
``Dataset`` object the built-in toy sets use.

Selection filters (recorded per dataset, never silently dropped):
  * classification only
  * MIN_SAMPLES <= n_samples <= MAX_SAMPLES   (very large sets deferred)
  * MIN_FEATURES <= n_features <= MAX_FEATURES
  * a deterministic stratified split must succeed
Extreme class imbalance is *flagged* (imbalance ratio) but not a skip reason.

`profile_external` returns the audit row; `build_profiles` builds the whole table
(`results/external/dataset_profiles_external.csv`). Registration is lazy: importing
this module registers factories but fetches nothing until a dataset is loaded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tice import contamination
from tice.datasets.registry import Dataset, make_clean_split, register_dataset

SOURCE = "openml"
MIN_SAMPLES, MAX_SAMPLES = 500, 20000
MIN_FEATURES, MAX_FEATURES = 5, 1000
SPLIT_SEED = 42
_IMBALANCE_FLAG = 10.0  # ratio above which we flag (but keep) a dataset

# Curated OpenML classification datasets (name -> OpenML data_id). Chosen to span
# binary/multiclass, numeric/categorical/mixed, with/without missing values, at
# moderate sizes. Some will be filtered out by the size rules above; that is
# recorded in the profile with a skip_reason rather than hidden.
CANDIDATES: dict[str, int] = {
    "kr-vs-kp": 3,
    "breast-w": 15,
    "mfeat-fourier": 14,
    "mfeat-karhunen": 16,
    "mfeat-morphological": 18,
    "mfeat-zernike": 22,
    "mfeat-factors": 12,
    "cmc": 23,
    "credit-approval": 29,
    "credit-g": 31,
    "diabetes": 37,
    "tic-tac-toe": 50,
    "vehicle": 54,
    "spambase": 44,
    "analcatdata_authorship": 458,
    "pc4": 1049,
    "pc3": 1050,
    "kc1": 1067,
    "pc1": 1068,
    "ilpd": 1480,
    "qsar-biodeg": 1494,
    "wdbc": 1510,
    "ozone-level-8hr": 1487,
    "phoneme": 1489,
    "steel-plates-fault": 1504,
    "wall-robot-navigation": 1497,
    "cnae-9": 1468,
    "first-order-theorem-proving": 1475,
    "churn": 40701,
    "texture": 40499,
    "optdigits": 28,
    "pendigits": 32,
    "satimage": 182,
    "segment": 36,
    "vowel": 307,
    "climate-model-simulation-crashes": 40994,
    "car": 40975,
    "dna": 40670,
    "splice": 46,
    "eucalyptus": 188,
    "PhishingWebsites": 4534,
    "GesturePhaseSegmentation": 4538,
    "MiceProtein": 40966,
    "dresses-sales": 23381,
    "banknote-authentication": 1462,
    "balance-scale": 11,
}


def load_external(name: str, openml_id: int) -> Dataset:
    """Fetch an OpenML dataset and wrap it as a :class:`Dataset` (labels factorised)."""
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=openml_id, as_frame=True, parser="auto")
    X = bunch.data.copy()
    y = bunch.target

    keep = y.notna().to_numpy()
    X = X.loc[keep].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True)

    cat_cols = [c for c in X.columns if str(X[c].dtype) in ("category", "object", "string", "bool")]
    for c in cat_cols:
        X[c] = X[c].astype(object)
    for c in X.columns:
        if c not in cat_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    y = pd.Series(pd.factorize(y)[0], name="target")
    return Dataset(
        dataset_id=name, X=X, y=y,
        categorical_columns=tuple(cat_cols), description=f"OpenML data_id={openml_id}",
    )


def _skip_reason(n_samples: int, n_features: int) -> str | None:
    if n_samples < MIN_SAMPLES:
        return f"n_samples<{MIN_SAMPLES}"
    if n_samples > MAX_SAMPLES:
        return f"n_samples>{MAX_SAMPLES} (deferred: too large)"
    if n_features < MIN_FEATURES:
        return f"n_features<{MIN_FEATURES}"
    if n_features > MAX_FEATURES:
        return f"n_features>{MAX_FEATURES} (deferred: too wide)"
    return None


def profile_external(name: str, openml_id: int) -> dict:
    """Fetch + profile one dataset; never raises (records fetch/split failures)."""
    row: dict = {
        "dataset_id": name, "source": SOURCE, "openml_id": openml_id,
        "n_samples": np.nan, "n_features": np.nan, "n_numeric": np.nan,
        "n_categorical": np.nan, "n_classes": np.nan, "class_imbalance_ratio": np.nan,
        "missing_rate": np.nan, "duplicate_ratio": np.nan, "near_duplicate_proxy": np.nan,
        "imbalance_flag": False, "split_seed": SPLIT_SEED,
        "usable_status": "skip", "skip_reason": "",
    }
    try:
        ds = load_external(name, openml_id)
    except Exception as exc:  # noqa: BLE001
        row["skip_reason"] = f"fetch_failed: {type(exc).__name__}: {str(exc)[:120]}"
        return row

    X, y = ds.X, ds.y
    n_samples, n_features = int(len(X)), int(X.shape[1])
    counts = y.value_counts()
    row.update({
        "n_samples": n_samples, "n_features": n_features,
        "n_numeric": len(ds.numeric_columns), "n_categorical": len(ds.categorical_columns),
        "n_classes": int(counts.size),
        "class_imbalance_ratio": float(counts.max() / max(counts.min(), 1)),
        "missing_rate": float(X.isna().to_numpy().mean()),
        "duplicate_ratio": float(X.duplicated().mean()),
    })
    row["imbalance_flag"] = bool(row["class_imbalance_ratio"] >= _IMBALANCE_FLAG)

    reason = _skip_reason(n_samples, n_features)
    if reason is None and row["n_classes"] < 2:
        reason = "n_classes<2"
    if reason is None:
        try:
            # The split is the real usability gate (the pipeline calls it too).
            split = make_clean_split(ds, base_seed=SPLIT_SEED, test_size=0.3)
            row["usable_status"] = "usable"
            # Near-duplicate proxy is best-effort: a failure here (e.g. no numeric
            # columns) leaves it NaN but does NOT make the dataset unusable.
            try:
                row["near_duplicate_proxy"] = contamination.proximity_profile(split, (0.5,))[0.5]
            except Exception:  # noqa: BLE001
                row["near_duplicate_proxy"] = np.nan
        except Exception as exc:  # noqa: BLE001 - e.g. a class with a single member
            reason = f"split_failed: {type(exc).__name__}: {str(exc)[:100]}"
    if reason is not None:
        row["skip_reason"] = reason
        row["usable_status"] = "skip"
    return row


def build_profiles(candidates: dict[str, int] | None = None) -> pd.DataFrame:
    cands = candidates if candidates is not None else CANDIDATES
    rows = []
    for name, oid in cands.items():
        r = profile_external(name, oid)
        rows.append(r)
        print(f"[external] {name:32s} -> {r['usable_status']:6s} "
              f"({r.get('n_samples','?')}x{r.get('n_features','?')}, "
              f"{r['skip_reason'] or 'ok'})", flush=True)
    return pd.DataFrame(rows)


def usable_datasets(profiles: pd.DataFrame) -> list[str]:
    return profiles.loc[profiles["usable_status"] == "usable", "dataset_id"].tolist()


def _register_all() -> None:
    """Register every candidate as a lazy factory (no network until loaded)."""
    for _name, _oid in CANDIDATES.items():
        register_dataset(_name, (lambda n=_name, o=_oid: load_external(n, o)))


_register_all()
