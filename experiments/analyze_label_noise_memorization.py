"""Label-noise memorization diagnostic.

Question: under *train*-label noise, is there memorization-like behavior (a model
confidently fitting the corrupted labels) or merely clean-test degradation? We
flip a fraction lambda of TRAIN labels, refit, and compare how the model treats
the flipped rows vs the clean test set.

Per (model, dataset_id, seed, lambda) we record:
  - noisy_train_accuracy      : train accuracy against the labels actually trained on
  - clean_train_accuracy      : train accuracy against the ORIGINAL (pre-flip) labels
  - clean_test_accuracy / clean_test_auc
  - train_test_gap            : noisy_train_accuracy - clean_test_accuracy
  - noisy_label_fit_rate      : on flipped rows, P(predict the flipped/wrong label)
  - clean_label_recovery_rate : on flipped rows, P(predict the original/true label)
  - confidence_on_flipped_samples : mean prob mass on the flipped label, flipped rows

Memorization-like signature = high noisy_label_fit_rate + high
confidence_on_flipped_samples (the model commits to the corrupted labels) while
clean_test degrades. Resistance = high clean_label_recovery_rate instead.

Reuses the pipeline's dataset registry, the exact label_noise generator, the
shared preprocessor, and the metric code. ICL is represented by `tabicl` (local,
no auth); `tabpfn_client` needs the PriorLabs cloud token and is omitted by
default (add it via --models if TABPFN_TOKEN is set).

Caveat: ICL models do in-context learning, so "train accuracy" measures how they
label their own context rows -- a slightly different notion than a GBDT fitting
parameters. Interpreted accordingly.

Usage::

    python experiments/analyze_label_noise_memorization.py --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Springer asks for 600 dpi on combination art; 150 is fine for reading on screen.
# Set TICE_FIG_DPI=600 before running to regenerate at submission resolution.
FIG_DPI = int(os.environ.get("TICE_FIG_DPI", "150"))

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from tice.datasets.registry import load_dataset, make_clean_split  # noqa: E402
from tice.figio import save_figure  # noqa: E402
from tice.metrics.classification import compute_metrics  # noqa: E402
from tice.models.registry import get_model_spec  # noqa: E402
from tice.models.runner import _align_proba  # noqa: E402
from tice.preprocess import build_preprocessor  # noqa: E402
from tice.shifts.generators import get_shift  # noqa: E402

_DEFAULT_MODELS = ("logreg", "hist_gbdt", "xgboost", "catboost", "tabicl")
_DATASETS = ("breast_cancer", "wine", "iris", "digits", "synthetic_mixed")
_LAMBDAS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40)
_FAMILY_COLOR = {"icl": "#d1495b", "gbdt": "#4c72b0", "linear": "#8d99ae"}


def _analyze_condition(model: str, dataset_id: str, seed: int, lam: float) -> dict:
    """Fit `model` on lambda-noised train labels; return the memorization row."""
    spec = get_model_spec(model)
    row = {
        "model": model, "family": spec.family, "dataset_id": dataset_id,
        "seed": seed, "lambda": lam, "status": "ok", "error_message": "",
        "n_train": 0, "n_flipped": 0,
        "noisy_train_accuracy": np.nan, "clean_train_accuracy": np.nan,
        "clean_test_accuracy": np.nan, "clean_test_auc": np.nan, "train_test_gap": np.nan,
        "noisy_label_fit_rate": np.nan, "clean_label_recovery_rate": np.nan,
        "confidence_on_flipped_samples": np.nan,
    }
    if not spec.available():
        row["status"] = "skipped"
        row["error_message"] = f"package '{spec.required_package}' not installed"
        return row
    try:
        dataset = load_dataset(dataset_id)
        clean = make_clean_split(dataset, base_seed=seed, test_size=0.3)
        noisy = get_shift("label_noise").apply(clean, lam, seed)

        y_clean = clean.y_train.to_numpy()
        y_noisy = noisy.y_train.to_numpy()
        flipped = y_noisy != y_clean
        classes = np.asarray(clean.classes)

        pre = build_preprocessor(noisy.numeric_columns, noisy.categorical_columns)
        pipe = Pipeline([("pre", pre), ("clf", spec.builder(seed))])
        pipe.fit(noisy.X_train, y_noisy)

        model_classes = np.asarray(pipe.classes_)
        proba_tr = _align_proba(np.asarray(pipe.predict_proba(noisy.X_train)), model_classes, classes)
        proba_te = _align_proba(np.asarray(pipe.predict_proba(noisy.X_test)), model_classes, classes)
        pred_tr = classes[proba_tr.argmax(axis=1)]

        row["n_train"] = int(len(y_noisy))
        row["n_flipped"] = int(flipped.sum())
        row["noisy_train_accuracy"] = float(np.mean(pred_tr == y_noisy))
        row["clean_train_accuracy"] = float(np.mean(pred_tr == y_clean))

        test = compute_metrics(noisy.y_test.to_numpy(), proba_te, classes)
        row["clean_test_accuracy"] = float(test.accuracy)
        row["clean_test_auc"] = float(test.auc)
        row["train_test_gap"] = row["noisy_train_accuracy"] - row["clean_test_accuracy"]

        if flipped.any():
            cls_idx = {c: i for i, c in enumerate(classes.tolist())}
            f_pred = pred_tr[flipped]
            f_noisy = y_noisy[flipped]
            f_clean = y_clean[flipped]
            row["noisy_label_fit_rate"] = float(np.mean(f_pred == f_noisy))
            row["clean_label_recovery_rate"] = float(np.mean(f_pred == f_clean))
            conf = [proba_tr[np.flatnonzero(flipped)[k], cls_idx[c]] for k, c in enumerate(f_noisy)]
            row["confidence_on_flipped_samples"] = float(np.mean(conf))
    except Exception as exc:  # noqa: BLE001 - record and continue
        row["status"] = "error"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
    return row


def run(models, datasets, seeds, lambdas) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for ds in datasets:
            for model in models:
                for lam in lambdas:
                    rows.append(_analyze_condition(model, ds, seed, lam))
            print(f"[memz] seed={seed} dataset={ds} done", flush=True)
    return pd.DataFrame(rows)


def plot_gap(df: pd.DataFrame, out: Path) -> None:
    ok = df[df["status"] == "ok"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.8))
    # Left: clean-test AUC retention. Right: train/test accuracy gap.
    for col, ax, ylab in (("clean_test_auc", axL, "mean clean-test AUC"),
                          ("train_test_gap", axR, "mean (noisy-train acc - clean-test acc)")):
        piv = ok.groupby(["model", "lambda"])[col].mean().unstack()
        for model in piv.index:
            fam = get_model_spec(model).family
            ax.plot(piv.columns, piv.loc[model], marker="o",
                    color=_FAMILY_COLOR.get(fam, "#8d99ae"),
                    lw=2.4 if fam == "icl" else 1.4, ls="-" if fam == "icl" else "--",
                    label=f"{model} ({fam})")
        ax.set_xlabel("train-label noise severity λ")
        ax.set_ylabel(ylab)
        ax.spines[["top", "right"]].set_visible(False)
    axL.legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle(
        "Train-label noise: ICL models retain clean-test ranking while GBDT baselines degrade faster"
    )
    fig.tight_layout()
    save_figure(fig, out, dpi=FIG_DPI)
    plt.close(fig)


def summarize(df: pd.DataFrame) -> None:
    """Print the memorization-signature comparison (ICL vs GBDT) at high noise."""
    ok = df[(df["status"] == "ok") & (df["lambda"] >= 0.30) & (df["n_flipped"] > 0)]
    if ok.empty:
        print("[memz] no usable rows for the high-noise summary")
        return
    g = ok.groupby("family")[
        ["noisy_label_fit_rate", "clean_label_recovery_rate",
         "confidence_on_flipped_samples", "train_test_gap", "clean_test_auc"]
    ].mean().round(3)
    pd.set_option("display.width", 160)
    print("\n[memz] high-noise (lambda>=0.30) means by family -- memorization signature:")
    print(g.to_string())
    print("\n  memorization-like = high noisy_label_fit_rate + high confidence_on_flipped_samples")
    print("  resistance        = high clean_label_recovery_rate")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Label-noise memorization diagnostic.")
    p.add_argument("--models", type=str, default=",".join(_DEFAULT_MODELS))
    p.add_argument("--datasets", type=str, default=",".join(_DATASETS))
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--out", type=Path, default=_ROOT / "results")
    args = p.parse_args(argv)

    models = tuple(m for m in args.models.split(",") if m)
    datasets = tuple(d for d in args.datasets.split(",") if d)
    seeds = tuple(int(s) for s in args.seeds.split(",") if s != "")

    print(f"[memz] models={list(models)} datasets={list(datasets)} seeds={list(seeds)}")
    df = run(models, datasets, seeds, _LAMBDAS)

    diag_dir = Path(args.out) / "diagnostics"
    fig_dir = Path(args.out) / "figures"
    diag_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv = diag_dir / "label_noise_memorization.csv"
    df.to_csv(csv, index=False)
    plot_gap(df, fig_dir / "label_noise_train_test_gap.png")

    n_ok = int((df["status"] == "ok").sum())
    print(f"[memz] rows={len(df)} ok={n_ok} skipped={(df.status=='skipped').sum()} error={(df.status=='error').sum()}")
    summarize(df)
    print(f"[memz] wrote {csv} and {fig_dir/'label_noise_train_test_gap.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
