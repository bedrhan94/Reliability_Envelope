# Results Manifest — TICE reliability-envelope experiments

This manifest defines which result artefacts are **canonical** (usable for the
paper) and which are **archived/failed** (must not enter paper results). Written
2026-07-02.

> **WARNING.** A local `tabpfn` license/skipped run exists and is **not** part of
> the paper results. The canonical TabPFN result is **`tabpfn_client`** (PriorLabs
> cloud inference). See the archive section below.

---

## 1. Canonical vs archived — at a glance

| Artefact set | Location | Canonical? | Why |
|---|---|---|---|
| Single-seed shift run (seed 42) | `results/canonical/tables/` + live `results/tables/` | ✅ canonical | 6 models incl. **`tabpfn_client`**, 1080 conditions, **0 errors** |
| 10-seed multi-seed aggregates | `results/canonical/tables/aure_multiseed.csv`, `aure_pairwise.csv`, `aure_by_seed.csv`, `rho_by_seed.csv` | ✅ canonical | seeds 0–9, bootstrap CIs + Holm pairwise |
| Threshold-sensitivity | `results/canonical/tables/aure_sensitivity.csv` | ✅ canonical (supporting) | 336-combination robustness check |
| Continuous (de-quantised) AURE | `results/canonical/tables/aure_continuous.csv` | ✅ canonical (supporting) | quantisation robustness check |
| Figures | `results/canonical/figures/` | ✅ canonical | see §6 |
| **Local-TabPFN license run (June 28)** | `results/archive/failed_local_tabpfn_license_run/` | ❌ **archived / do NOT use** | uses local `tabpfn`, all 180 rows `skipped` with `TabPFNLicenseError` |

`results/tables/` and `results/figures/` remain the **live pipeline outputs**
(unchanged, so scripts keep working); `results/canonical/` is a frozen copy for
the paper. `results/diagnostics/` holds post-hoc analyses (see §7).

---

## 2. Canonical experiment provenance

- **Single-seed run:** `python experiments/run_shift_stress.py --config configs/experiments/shift_stress_full.yaml` (seed **42**). Produces `shift_results.csv`, `dataset_profiles.csv`, `reliability_envelopes.csv`, `aure_summary.csv`.
- **Multi-seed run:** `python experiments/run_multiseed.py --config configs/experiments/shift_stress_full.yaml --seeds 0,1,2,3,4,5,6,7,8,9`. Produces `aure_by_seed.csv`, `rho_by_seed.csv`, `aure_multiseed.csv`, `aure_pairwise.csv`.
- **Environment lock:** `requirements-lock.txt` (python 3.10.0; tabpfn-client 0.3.2, tabicl 2.1.1, xgboost 3.2.0, catboost 1.2.10, torch 2.7.1+cu128, …).

### Design grid
- **Models (6):** `logreg`, `hist_gbdt`, `xgboost`, `catboost` (GBDT/linear baselines); **`tabpfn_client`**, `tabicl` (tabular ICL foundation models). Families: linear / gbdt / icl.
- **Datasets (5):** `breast_cancer` (569×30, 2 cls), `wine` (178×13, 3), `iris` (150×4, 3), `digits` (1797×64, 10), `synthetic_mixed` (600×6, 3; the only set with categorical features → the only one on which `rare_category_shift` applies).
- **Shift axes (6):** `label_noise`, `class_imbalance`, `missingness_shift`, `rare_category_shift`, `covariate_shift`, `context_budget`. λ=0 is an exact no-op.
- **Lambda grid (6):** `0.00, 0.05, 0.10, 0.20, 0.30, 0.40`.
- **Conditions:** 6×5×6×6 = **1080** rows (single seed). `skipped` = 144 (rare_category on the 4 non-categorical datasets); errors = **0**.
- **Seeds:** single-seed = **42**; multi-seed = **0–9** (10 seeds).

---

## 3. Canonical AURE (headline = 10-seed with CIs)

Reliability-envelope radius averaged over datasets×axes; higher = more shift-robust.

| Model | Family | AURE (10-seed mean) | seed 95% CI | AURE (single seed 42) |
|---|---|---|---|---|
| tabicl | icl | **0.236** | [0.233, 0.240] | 0.238 |
| tabpfn_client | icl | **0.228** | [0.225, 0.230] | 0.235 |
| catboost | gbdt | 0.185 | [0.175, 0.194] | 0.158 |
| logreg | linear | 0.133 | [0.129, 0.137] | 0.131 |
| hist_gbdt | gbdt | 0.117 | [0.108, 0.125] | 0.090 |
| xgboost | gbdt | 0.111 | [0.097, 0.125] | 0.135 |

**Cite the 10-seed column in the paper** (single-seed GBDT ordering is unstable:
xgboost 0.135→0.111, catboost 0.158→0.185 across the two).

### Significance (Holm-corrected paired tests, from `aure_pairwise.csv`)
- **ICL > every GBDT is significant at both the seed unit and the dataset unit**, including vs the strongest GBDT (catboost): seed `p_holm=0.029`, dataset-unit CI excludes 0.
- **catboost** is the clear best GBDT (significantly above logreg/hist_gbdt/xgboost).
- **logreg ≈ hist_gbdt ≈ xgboost** — mutually indistinguishable (`p_holm` 0.08–0.63).
- **tabicl vs tabpfn_client:** seed-unit significant (`p_holm=0.029`) but Δ=0.009 and the **dataset-unit CI includes 0 → does not generalise**. Treat as **tabicl ≈ tabpfn_client**.

---

## 4. Archived / failed run — do NOT use

`results/archive/failed_local_tabpfn_license_run/` (originally the June-28 backup):
- Model set includes local **`tabpfn`** (not `tabpfn_client`).
- All 180 `tabpfn` rows are `status="skipped"` with `error_message = TabPFNLicenseError`
  ("TabPFN requires a one-time license acceptance … no interactive terminal").
- Because TabPFN was skipped, its AURE is absent and the other models were scored
  against a weaker/different reference. **These numbers are not comparable to the
  canonical run and must not appear in the paper.**

**Local `tabpfn` license/skipped run is not part of the paper results. The
canonical TabPFN result is `tabpfn_client`.**

---

## 5. Which results to use / not use in the paper

**USE (canonical):**
- `results/canonical/tables/aure_multiseed.csv` — headline AURE ± CIs.
- `results/canonical/tables/aure_pairwise.csv` — significance (ICL-tier claim).
- `results/canonical/tables/shift_results.csv` + `reliability_envelopes.csv` — per-condition detail.
- `aure_sensitivity.csv`, `aure_continuous.csv` — robustness of the metric.

**DO NOT USE:**
- Anything under `results/archive/`.
- The single-seed AURE for *fine* model ordering (use it only for illustration; the 10-seed CIs govern ordering claims).
- `tabicl > tabpfn_client` as a headline (see §3).

---

## 6. Figures (`results/canonical/figures/`)

| File | Shows |
|---|---|
| `aure_overall.png` | Single-seed AURE per model, ranked, coloured by family. |
| `aure_overall_ci.png` | **10-seed AURE with two CIs** — dark = seed-unit (within these 5 datasets), light = dataset-unit (generalisation). Overlapping light whiskers ⇒ ordering does not generalise. |
| `aure_by_axis.png` | Per-shift-axis AURE heatmap (where each model degrades). |
| `aure_sensitivity.png` | AURE vs each failure threshold (one-at-a-time) — robustness to the τ constants. |
| `covariate_calibration_collapse.png` | Under covariate shift, AUC stays flat while NLL/ECE explode (calibration collapse, not discrimination loss). |
| `label_noise_robustness.png` | Mean test AUC vs label-noise λ per model — ICL flat, GBDT/linear degrade faster. |
| `label_noise_train_test_gap.png` (see §7) | Train-vs-clean-test behaviour under label noise (memorization-like diagnostic). |

---

## 7. Diagnostics (`results/diagnostics/`)

Post-hoc analyses that support/qualify specific claims (not part of the main grid):
- `label_noise_memorization.csv` — noisy-train vs clean-test accuracy, train/test gap,
  and fit-rate on flipped labels + confidence on flipped samples, per (model, dataset,
  seed, λ). Produced by `experiments/analyze_label_noise_memorization.py` (models:
  logreg, hist_gbdt, xgboost, catboost, **tabicl** as the local ICL representative;
  `tabpfn_client` omitted — needs the cloud token; 5 datasets × 3 seeds × 6 λ = 450 rows, 0 errors).

**Finding (supports a "memorization-like behavior" statement for GBDTs).** At high
train-label noise (λ≥0.30), on flipped training rows:

| family | noisy_label_fit_rate | clean_label_recovery | conf_on_flipped | train_test_gap | clean_test_auc |
|---|---|---|---|---|---|
| gbdt | **0.97** | 0.03 | **0.87** | **+0.24** | 0.879 |
| icl (tabicl) | 0.29 | 0.70 | 0.34 | −0.18 | **0.976** |
| linear (logreg) | 0.17 | 0.73 | 0.27 | −0.17 | 0.914 |

Per model: xgboost 0.999 fit / 0.97 conf, catboost 0.99 / 0.80, hist_gbdt 0.91 / 0.83.
GBDT fit-rate exceeds tabicl's on **all 5 datasets** (0.85–1.00 vs 0.03–0.53). So GBDTs
confidently fit the corrupted labels (positive train/test gap) while tabicl recovers the
*true* label on 70% of flipped rows and keeps the highest clean-test AUC. **Honest
caveat:** logreg resists more than tabicl by fit-rate but only by underfitting (lower
clean-test AUC); the ICL story is "resists noise *and* retains accuracy". The ICL
representative here is `tabicl` only.
