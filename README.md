# Tabular In-Context Reliability Envelope

> **Resuming this project on a new machine? Read [`HANDOFF.md`](HANDOFF.md) first.**
> It covers where the work stands, the one run still outstanding, the traps that have
> already cost time here, and which files are authoritative versus superseded.

> When Tabular In-Context Learning Fails: A Contamination-Aware Reliability
> Envelope under Distribution Shift

Measures **where** tabular in-context / foundation models (TabPFN, TabICL v2)
systematically break down relative to GBDT baselines (XGBoost, CatBoost) under
controlled distribution shift — and turns that into a single number per model,
the **AURE** (Average Reliability Envelope).

This repository covers milestones **M1–M3**: dataset profiler, controlled shift
stress suite, reliability utility, GBDT reference selection, failure indicator,
reliability envelope radius (ρ), and AURE.

## Install

```bash
# core (smoke path — CPU only, no downloads)
pip install -e .

# optional real backends (GBDT + ICL foundation models)
pip install -e ".[gbdt,icl]"   # xgboost, catboost, tabpfn, tabpfn-client, tabicl

# dev tooling
pip install -e ".[dev]"        # pytest, ruff
```

## Run

```bash
# documented interface (once uv is installed)
uv run python experiments/run_shift_stress.py --config configs/experiments/shift_stress.yaml

# equivalent without uv
python experiments/run_shift_stress.py --config configs/experiments/shift_stress.yaml
```

The **smoke config** (`shift_stress.yaml`) uses sklearn toy datasets and
lightweight models (`logreg`, `hist_gbdt`) and runs end-to-end on CPU in
seconds — nothing is downloaded. The **full config**
(`shift_stress_full.yaml`) adds `xgboost`, `catboost`, `tabpfn`, `tabicl`; any
model whose package is missing **or unusable** is recorded as `status="skipped"`
rather than crashing the run. A one-shot preflight probe detects unusable models
(e.g. an ICL backend that can't authenticate) once, so they never run per
condition.

### Enabling TabPFN (PriorLabs license)

TabPFN v2 weights are gated behind a free PriorLabs license + API key, so it
cannot authenticate non-interactively out of the box. To enable it:

1. Register/log in at <https://ux.priorlabs.ai/account> and accept the license
   on the **Licenses** tab.
2. Copy your **API Key** and export it (the runner sets `TABPFN_NO_BROWSER=1`
   automatically, so it will use the token instead of opening a browser):

   ```bash
   export TABPFN_TOKEN="<your-api-key>"
   # gated HuggingFace repo (Prior-Labs/tabpfn_*) may also need HF access:
   export HF_TOKEN="<your-hf-token>"   # or: huggingface-cli login
   ```

3. Re-run the full config. Without a token, TabPFN is cleanly `skipped` and
   excluded from the envelopes/AURE. **TabICL needs no auth** — it downloads its
   checkpoint from HuggingFace on first use.

## Outputs (`results/tables/`)

| File | Contents |
|------|----------|
| `dataset_profiles.csv` | one row per dataset — 12 meta-features |
| `shift_results.csv` | one row per (dataset × model × shift_axis × λ) with metrics, utility, status/error, failure flag |
| `reliability_envelopes.csv` | one ρ per (model × dataset × shift_axis) |
| `aure_summary.csv` | AURE per model (overall + per shift axis) |

## Figures (`results/figures/`)

```bash
python experiments/make_figures.py            # reads results/tables
```

Renders two PNGs from `aure_summary.csv`: `aure_overall.png` (AURE per model,
ranked and coloured by family) and `aure_by_axis.png` (per-shift-axis AURE
heatmap — where each model breaks). Under the published failure rule the ICL
foundation models (TabPFN, TabICL) score highest (AURE ≈ 0.24 on the pilot) versus
the GBDT baselines (≈ 0.13–0.16), with the shared weak axes being `covariate_shift`
and `label_noise`.

> **Read this before quoting AURE.** ρ requires an unbroken pass-run from λ=0 and the
> failure triggers are absolute or shared, so a model whose *clean* state already sits
> near the bar scores ρ=0 before any shift is applied — 43% of cells for
> xgboost/hist_gbdt/logreg versus 7–9% for the ICL models. Under two reference-matched
> re-analyses of the same runs, the ICL lead **reverses** on all three evidence bases.
> AURE conflates out-of-the-box calibration quality with shift tolerance; report it with
> its reference policy. It factors exactly as `E[ρ] = P(pass at λ=0) × E[ρ | pass at λ=0]`,
> and the ICL advantage sits entirely in the first term. Quantified in [`results/ablations/`](results/ablations/) via
> `experiments/ablate_reference_confound.py` and `experiments/tau_lambda0_sensitivity.py`;
> discussion in [`paper/claims.md`](paper/claims.md) §0.

## Key formulas (and where they live)

| Concept | Definition | Location |
|---------|-----------|----------|
| Reliability utility `U` | `0.35*AUC + 0.15*Acc − 0.30*NLL_norm − 0.20*ECE` | [`tice/metrics/utility.py`](tice/metrics/utility.py) |
| `NLL_norm` | `NLL / log(K)` | [`tice/metrics/classification.py`](tice/metrics/classification.py) |
| Failure indicator | `U < U_best_gbdt − τ_u` OR `ECE > τ_ece` OR `NLL_norm > τ_nll` | [`tice/metrics/utility.py`](tice/metrics/utility.py) |
| `best_gbdt` | argmax clean-ID utility over GBDT-family models | [`tice/pipeline.py`](tice/pipeline.py) |
| Envelope radius `ρ` | max λ with an unbroken pass-run from λ=0 | [`tice/envelope/reliability.py`](tice/envelope/reliability.py) |
| `AURE` | mean ρ across datasets × shift axes | [`tice/envelope/reliability.py`](tice/envelope/reliability.py) |

Thresholds (`τ_utility=0.03`, `τ_ece=0.10`, `τ_nll=0.75`) and the GBDT
reference policy (`clean` vs `matched`) are read from the YAML config.

## Shift axes

All axes are deterministic, seed-fixed, and treat **λ=0 as an exact no-op**.

| Axis | Status | Mechanism |
|------|--------|-----------|
| `label_noise` | real | flip a λ-fraction of train labels |
| `class_imbalance` | real | down-sample non-majority train classes by λ |
| `missingness_shift` | real | inject MCAR NaN into test at rate λ |
| `covariate_shift` | real | mean-shift test numerics by `λ·strength·σ` |
| `context_budget` | real | shrink train context to a (1−λ) stratified fraction |
| `rare_category_shift` | conditional | replace λ-fraction of test categoricals with an unseen token — only on datasets with categorical features; otherwise reported `skipped` |

## Tests

```bash
pytest            # 93 tests: profiler, shift generators, envelope, calibration metrics,
                  # reference-confound ablation, run audit
ruff check .
```
