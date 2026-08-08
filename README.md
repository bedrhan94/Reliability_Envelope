# reliability-envelope

> **When Do Tabular In-Context Models Stay Reliable Under Shift?**
> Calibration Confounding in Threshold-Based Reliability Envelopes
>
> Neslihan Aytac, Bedirhan Bedir — Istanbul Topkapi University
> Under review at *Data Mining and Knowledge Discovery*.

A reliability-envelope protocol for tabular models under controlled distribution shift,
and the finding that came out of auditing it. The protocol sweeps a severity parameter λ
per (dataset, model, shift axis), records the largest λ a model survives contiguously
under an explicit failure rule, and averages that radius into one number per model —
**AURE** (Average Reliability Envelope).

**The headline result is about the metric, not the models.** Tabular in-context models
(TabPFN, TabICL) lead the AURE ranking, but the ranking cannot be read as differential
shift tolerance: AURE factors exactly into clean-state *admissibility* × conditional
*tolerance*, and the entire in-context advantage sits in the first factor. Under a
reference-matched failure rule the ordering reverses. The manuscript reporting this is under review and is not distributed here; the full protocol, every experiment script, and all per-condition results are.

The package is `tice`; it covers the dataset profiler, the controlled shift stress
suite, the reliability utility, GBDT reference selection, the failure indicator, the
envelope radius ρ, and AURE.

## Reproducing the results

Every number in the study is recomputable from this repository. The experiment drivers
are under [`experiments/`](experiments/), their configurations under
[`configs/experiments/`](configs/experiments/), and the stored per-condition tables under
[`results/`](results/) — one row per (dataset x model x shift axis x severity x seed) with
all metrics, the failure flag and its reason.

The re-analyses run no models at all; they recompute from those stored tables:

```bash
python experiments/ablate_reference_confound.py   # common support, self-referencing
python experiments/decompose_aure.py              # the A*T factorisation
python experiments/tau_lambda0_sensitivity.py     # 336-point threshold sweep
python experiments/utility_weight_sensitivity.py  # calibration-share sweep
python experiments/continuous_aure.py             # de-quantised radii
```

[`results/external/README.md`](results/external/README.md) maps every results directory to
what produced it, which are superseded, and — importantly — which are merge inputs that
must not be read directly.

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
foundation models score highest — on the pilot TabICL 0.236 and TabPFN 0.228 against
CatBoost 0.185 and 0.111–0.133 for the rest — with the shared weak axes being
`covariate_shift` and `label_noise`. On the three-seed external benchmark the numbers
are much smaller (TabICL 0.0977, TabPFN 0.0888, best GBDT 0.0744): the two are **not**
comparable, since the pilot pools six axes and the external run two.

> **Read this before quoting AURE.** ρ requires an unbroken pass-run from λ=0 and the
> failure triggers are absolute or shared, so a model whose *clean* state already sits
> near the bar scores ρ=0 before any shift is applied — 41–43% of cells for
> xgboost/hist_gbdt/logreg versus 5–8% for the ICL models on the three-seed external
> benchmark. Under two reference-matched
> re-analyses of the same runs, the ICL lead **reverses** on all three evidence bases.
> AURE conflates out-of-the-box calibration quality with shift tolerance; report it with
> its reference policy. It factors exactly as `E[ρ] = P(pass at λ=0) × E[ρ | pass at λ=0]`,
> and the ICL advantage sits entirely in the first term. Quantified in [`results/ablations/`](results/ablations/) via
> `experiments/ablate_reference_confound.py` and `experiments/tau_lambda0_sensitivity.py`;
> quantified in [`results/ablations/`](results/ablations/).
>
> **And never read an AURE off a single-family results directory.** An arm with no
> gradient-boosted model leaves `reference_utility` NaN, so the relative failure criterion
> silently drops out and the radius comes out too high — this inflated TabPFN by +0.031
> before we caught it. Merge into a table containing the reference pool first; see
> [`results/external/README.md`](results/external/README.md).

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
pytest            # 97 tests: profiler, shift generators, envelope, calibration metrics,
                  # reference-confound ablation, run audit
ruff check .
```
