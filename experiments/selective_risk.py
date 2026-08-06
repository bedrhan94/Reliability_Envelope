"""Independent 3rd-party reliability metric: does the confound survive selective classification?

The strongest test of the metric-class claim (§6.10) is whether the confound reproduces on a
reliability metric we did NOT design. We use **selective classification** (El-Yaniv & Wiener,
JMLR 2010; Geifman & El-Yaniv, NeurIPS 2017): a model may abstain on low-confidence inputs, and
its reliability is summarised by the risk-coverage trade-off. We compute standard selective
summaries per condition -- selective accuracy at 80% coverage, coverage at 10% risk, and the area
under the risk-coverage curve (AURC) -- from per-sample predicted probabilities (a fresh run; the
main runs stored only aggregates), then run the SAME envelope protocol using selective accuracy as
the quality metric in place of our utility.

If the confound is a property of the absolute-bar protocol rather than of our utility metric, an
overconfident model should fail the selective bar at lambda=0 exactly as it fails the utility bar,
and reference-matching should reverse the ICL lead again -- now on a third party's metric.

Usage::

    TABPFN_TOKEN=<key> python experiments/selective_risk.py --seeds 42
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sklearn.pipeline import Pipeline  # noqa: E402

import tice.datasets.external  # noqa: E402, F401  (registers external datasets on import)
from tice.datasets.registry import load_dataset, make_clean_split  # noqa: E402
from tice.envelope.reliability import envelope_radius  # noqa: E402
from tice.models.registry import get_model_spec  # noqa: E402
from tice.models.runner import _align_proba  # noqa: E402
from tice.preprocess import build_preprocessor  # noqa: E402
from tice.shifts.generators import get_shift  # noqa: E402

# 8 datasets spanning types/sizes (all-categorical, numeric, mixed, multiclass).
_DATASETS = ("car", "tic-tac-toe", "diabetes", "breast-w",
             "phoneme", "credit-g", "vowel", "segment")
_MODELS = ("logreg", "hist_gbdt", "xgboost", "catboost", "tabicl", "tabpfn")
_AXES = ("label_noise", "covariate_shift")
_LAMBDAS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40)
COVERAGE = 0.8       # selective accuracy at 80% coverage
RISK = 0.10          # coverage at 10% risk
TAU_SEL = 0.03       # envelope tolerance on selective accuracy (mirrors tau_utility)
ICL = {"tabicl", "tabpfn", "tabpfn_client"}


def _selective(probs: np.ndarray, y_true: np.ndarray, classes: np.ndarray) -> dict:
    """Standard selective-classification summaries from per-sample probabilities."""
    conf = probs.max(axis=1)
    pred = classes[probs.argmax(axis=1)]
    correct = (pred == y_true).astype(float)
    order = np.argsort(-conf)  # most confident first
    cs = correct[order]
    n = len(cs)
    cov = np.arange(1, n + 1) / n
    risk = 1.0 - np.cumsum(cs) / np.arange(1, n + 1)  # risk at each coverage level
    k = max(1, int(round(COVERAGE * n)))
    ok = np.where(risk <= RISK)[0]
    return {
        "sel_acc_80": float(cs[:k].mean()),
        "cov_at_risk10": float(cov[ok[-1]]) if len(ok) else 0.0,
        "aurc": float(risk.mean()),
        "accuracy": float(correct.mean()),
    }


def _run_condition(model: str, dataset_id: str, seed: int, axis: str, lam: float) -> dict | None:
    spec = get_model_spec(model)
    if not spec.available():
        return None
    try:
        clean = make_clean_split(load_dataset(dataset_id), base_seed=seed, test_size=0.3)
        shifted = get_shift(axis).apply(clean, lam, seed)
        pre = build_preprocessor(shifted.numeric_columns, shifted.categorical_columns)
        pipe = Pipeline([("pre", pre), ("clf", spec.builder(seed))])
        pipe.fit(shifted.X_train, shifted.y_train)
        classes = np.asarray(clean.classes)
        raw = pipe.predict_proba(shifted.X_test)
        probs = _align_proba(raw, np.asarray(pipe.classes_), classes)
        met = _selective(probs, shifted.y_test.to_numpy(), classes)
        met.update({"model": model, "family": spec.family, "dataset_id": dataset_id,
                    "shift_axis": axis, "shift_lambda": lam, "status": "ok"})
        return met
    except Exception as e:  # noqa: BLE001
        return {"model": model, "family": spec.family, "dataset_id": dataset_id,
                "shift_axis": axis, "shift_lambda": lam, "status": f"error:{type(e).__name__}",
                "sel_acc_80": np.nan}


def _envelope_analysis(df: pd.DataFrame) -> None:
    """Run the envelope protocol on selective accuracy: confound present? reversal?"""
    ok = df[df.status == "ok"].copy()
    cell = ["dataset_id", "shift_axis"]
    # shared reference = best clean GBDT selective accuracy per (dataset, axis)
    clean = ok[ok.shift_lambda == 0.0]
    gclean = clean[clean.family == "gbdt"].groupby(cell)["sel_acc_80"].max().rename("ref")
    ok = ok.join(gclean, on=cell)
    ok["failed"] = ok["sel_acc_80"] < ok["ref"] - TAU_SEL
    # self-referenced: each model vs its own clean selective accuracy
    own = ok[ok.shift_lambda == 0.0].set_index(cell + ["model"])["sel_acc_80"].rename("own")
    ok = ok.join(own, on=cell + ["model"])
    ok["failed_self"] = ok["sel_acc_80"] < ok["own"] - TAU_SEL

    rows = []
    for (model, _ds, _ax), g in ok.groupby(["model", "dataset_id", "shift_axis"]):
        g = g.sort_values("shift_lambda")
        rows.append({"model": model,
                     "rho": envelope_radius(g.shift_lambda.tolist(), g.failed.tolist()),
                     "rho_self": envelope_radius(g.shift_lambda.tolist(), g.failed_self.tolist())})
    per = pd.DataFrame(rows)
    agg = per.groupby("model").agg(aure_sel=("rho", "mean"), aure_self=("rho_self", "mean"))
    z = ok[ok.shift_lambda == 0.0].groupby("model").failed.mean().rename("lambda0_fail")
    agg = agg.join(z).sort_values("aure_sel", ascending=False)

    icl = [m for m in agg.index if m in ICL]
    gbdt = [m for m in agg.index if df[df.model == m].family.iloc[0] == "gbdt"]
    print("\n=== SELECTIVE-CLASSIFICATION envelope (El-Yaniv/Geifman metric, our protocol) ===")
    print(agg.round(4).to_string())
    print(f"\n  lambda=0 fail rate: ICL {agg.loc[icl,'lambda0_fail'].mean():.2f}  "
          f"vs GBDT {agg.loc[gbdt,'lambda0_fail'].mean():.2f}")
    for col, name in [("aure_sel", "published rule"), ("aure_self", "self-referenced")]:
        margin = agg.loc[icl, col].min() - agg.loc[gbdt, col].max()
        print(f"  {name:16s} min-ICL {agg.loc[icl,col].min():.4f}  "
              f"best-GBDT {agg.loc[gbdt,col].max():.4f}  margin {margin:+.4f}")
    print("\n  => confound reproduces on a 3rd-party metric IF: GBDT lambda0-fail >> ICL, "
          "published margin > 0, self-referenced margin < 0.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=str, default="42")
    p.add_argument("--out", type=Path, default=_ROOT / "results" / "ablations" / "selective_risk")
    args = p.parse_args(argv)
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    seeds = [int(s) for s in args.seeds.split(",") if s]

    rows = []
    total = len(_DATASETS) * len(_MODELS) * len(_AXES) * len(_LAMBDAS) * len(seeds)
    done = 0
    for seed in seeds:
        for ds in _DATASETS:
            for model in _MODELS:
                for axis in _AXES:
                    for lam in _LAMBDAS:
                        r = _run_condition(model, ds, seed, axis, lam)
                        done += 1
                        if r is not None:
                            r["seed"] = seed
                            rows.append(r)
            print(f"[selrisk] seed={seed} dataset={ds} done ({done}/{total})", flush=True)
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "selective_metrics.csv", index=False)
    ok = int((df.status == "ok").sum())
    print(f"\n[selrisk] rows={len(df)} ok={ok} "
          f"errors={int((df.status != 'ok').sum())} -> {args.out}")
    _envelope_analysis(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
