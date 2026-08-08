# External validation

Broad-benchmark check of the two mechanisms found in the 5-dataset pilot, on a
curated set of public OpenML classification datasets. The full 44-dataset run **has
since been executed and merged** (Sprint 2 set up the infrastructure; the run and
merge landed in Sprints 3–4).

## Read this before quoting a number from any directory here

The failure rule has three criteria and one of them is **relative** (utility below the best
clean GBDT by more than τ_u). A directory produced by a **single-family** config — ICL-only
or GBDT-only — has no reference to compare against, so `reference_utility` is NaN on every
row and that criterion silently drops out. Nothing errors; the AURE just comes out too high.
This already cost us a published number: TabPFN read 0.1195 in its own arm against 0.0888
merged, an inflation of +0.0307 (manuscript §5.3 / Table XI, `claims.md` S2,
`limitations.md` item 17).

**Rule: never read an AURE off a `*_only`, `*_fill`, `*_tabpfn_local` or
`*_strong_baselines*` directory.** They are merge inputs. Read numbers off the `*_merged`
directory that folds them against the reference pool. To check any directory,
`df.groupby('model').best_gbdt.notna().mean()` must be 1.0 for every model with
`status == "ok"`. `tables_catcov_multiseed/` reports <100% but is benign — the uncovered
rows are `applicable=False` / `skipped` non-categorical datasets.

## Which directory is which

Rows marked **merge input only** are subject to the warning above.

| Directory | Contents | Status |
|---|---|---|
| `tables_2axis_multiseed_primary_with_tabpfn_local/` | **44 datasets × 2 axes × 6 λ × 3 seeds, all six models complete** (TabPFN from local pinned weights). | **Primary external result — cited by the paper (Tables IX–X).** |
| `tables_2axis_multiseed_primary/` | The same three-seed run before the local-TabPFN merge. | Superseded; merge audit trail. |
| `tables_2axis_multiseed_tabpfn_local/` | `tabpfn` local weights alone, 3 seeds, 1584 rows. | **Merge input only** (no GBDT reference). |
| `tables_2axis_multiseed_tabpfn_fill/` | `tabpfn_client` cloud gap-fill, quota-limited. | **Merge input only**; source of the retracted 0.1199. |
| `tables_2axis_strong_baselines_multiseed/` | 5 strengthened GBDTs × 3 seeds = 7920 rows, equal tuning budget plus `gbdt_ensemble`. | **Merge input only** (GBDT-only). |
| `tables_2axis_strong_multiseed_merged/` | The above merged onto the primary base and re-scored against the strengthened reference (`merge_strong_multiseed.py`). | **Baseline-strength result at three seeds (Table XVIII).** |
| `tables_2axis_strong_baselines/` · `tables_2axis_strong_merged/` | The single-seed predecessors of the two rows above. | Superseded by the three-seed arm; seed 42 reproduces bit-for-bit. |
| `tables_2axis_seed42_merged/` | **44 datasets × 6 models × seed 42 = 3168 rows.** The 5 token-less models plus the merged `tabpfn_client` cloud rows. | Superseded as primary by the three-seed run; still cited for the seed-42 capped/uncapped contrast (Table V). |
| `tables_2axis_seed42_partial/` | The same run before the TabPFN merge (5 models). | Superseded; kept as the merge audit trail. |
| `tables_2axis_seed42_tabpfn_only/` | `tabpfn_client` alone; merged by `merge_tabpfn_client.py`. | **Merge input only.** |
| `tables_2axis_stratified_multiseed/` | **12 stratified datasets × 6 models × 3 seeds = 2592 rows.** | Supplementary seed-stability evidence. |
| `tables_2axis_tuned_gbdt_only/` | `xgboost_tuned` / `catboost_tuned`, 12-dataset subset, seed 42. | Merge input only. |
| `tables_2axis_tuned_gbdt_merged/` | The subset re-scored against **tuned** GBDT baselines (`merge_tuned_gbdt.py`). | Baseline-strength ablation. |
| `tables_2axis_calibrated_only/` | Platt-scaled `logreg`/`hist_gbdt`/`xgboost`/`catboost`, 12-dataset subset, seed 42. | Merge input only. |
| `tables_2axis_calibrated_merged/` | The subset re-scored against **calibrated** baselines (`merge_tuned_gbdt.py --label calibrated`). | Post-hoc calibration ablation. |
| `tables_2axis_icl_uncapped_only/` · `..._merged/` | ICL models re-run uncapped on the 27 large datasets; merged back over the 44-dataset table. | **Context-cap correction — the corrected external result.** |
| `tables_2axis_stratified_multiseed_icl_uncapped/` · `..._merged/` | Same correction for the 12×3-seed subset. | Does not reproduce the correction — see Table 7. |
| `tables_2axis_tuned_gbdt_merged_uncapped/` · `tables_2axis_calibrated_merged_uncapped/` | The tuned and calibrated arms re-merged onto the uncapped ICL base. | Consistency only; within-resolution. |
| `smoke/` | Tiny end-to-end check. | Not a result. |
| `figures_merged/`, `figures_partial/`, `figures_stratified_multiseed/` | Figures for the like-named table dirs. | — |
| `stratified_subset_12.csv` | The 12-dataset subset and its selection rationale. | See limitations. |
| `dataset_profiles_external.csv` | Meta-features for every candidate dataset. | Joined by the figure scripts. |

Re-analyses that re-run no models live in `results/ablations/` (reference-confound
ablation and the τ / λ=0 sweep). See `paper/limitations.md` for what they change.

## Purpose
Test whether the pilot's two headline mechanisms hold beyond the 5 toy datasets:
1. **covariate_shift → calibration collapse** (discrimination holds, NLL/ECE rise).
2. **label_noise → GBDT baselines degrade faster / ICL ranking stability**
   (with the memorization-like signature from `results/diagnostics/`).

## Why only 2 axes
We deliberately restrict to `label_noise` + `covariate_shift` — the two axes tied to
the claims we are validating. The other four axes (class_imbalance, missingness,
rare_category, context_budget) are out of scope for this sprint to keep the run
tractable and the analysis focused. **No 6-axis external run is performed.**

## Dataset selection & filtering
Source: **OpenML** via `sklearn.datasets.fetch_openml(data_id=...)` (cached; nothing
bundled). Curated candidate list + filters live in [`tice/datasets/external.py`](../../tice/datasets/external.py).

Filters (recorded per dataset, never silently dropped):
- classification only; `n_classes >= 2`
- `500 <= n_samples <= 20000` (very large sets deferred to a later sprint)
- `5 <= n_features <= 1000`
- a deterministic **stratified split (seed 42)** must succeed
- extreme class imbalance is *flagged* (`imbalance_flag`, ratio ≥ 10) but **kept**

Profile audit: [`results/external/dataset_profiles_external.csv`](dataset_profiles_external.csv)
(one row per candidate; columns incl. source, openml_id, n_samples/features/numeric/
categorical/classes, class_imbalance_ratio, missing_rate, duplicate_ratio,
near_duplicate_proxy, split_seed, usable_status, skip_reason).

**Result: 46 candidates → 44 usable, 2 skipped.**
- usable span: n_samples 500–11055, n_features 5–856, classes 2–11; 14 mixed/categorical, 5 with missing values, 4 imbalance-flagged.
- **Skip reasons:** `banknote-authentication` (n_features<5), `balance-scale` (n_features<5).
- (Datasets with missing values — breast-w, credit-approval, eucalyptus, MiceProtein —
  are usable: the near-duplicate proxy median-imputes, and the pipeline imputes at fit time.)

## Smoke run (done)
3 diverse datasets (diabetes = numeric-binary, car = all-categorical, vowel =
mixed-11-class) × 2 axes × 6 λ × 3 models (logreg, hist_gbdt, tabicl).

```bash
python experiments/run_external_shift.py --config configs/experiments/shift_stress_external_2axis_smoke.yaml
```

Result: **108 rows, 0 errors, 0 skipped** → `results/external/smoke/`. The loader,
shift generators, categorical/numeric handling, and metric plumbing all work on
external data. Sanity: on `diabetes` label_noise, hist_gbdt degrades fastest
(AUC 0.80→0.57) while logreg/tabicl stay higher — the pilot pattern reproduces even
in smoke. All four external figures render from the smoke outputs.

## Full run (executed — commands for reproduction)
44 datasets × 2 axes × 6 λ × **6 models** × 1 seed (seed 42) = 3168 rows, 0 skipped,
0 error → `tables_2axis_seed42_merged/`. `tabpfn_client` needs the PriorLabs cloud
token; without it that model is cleanly `skipped` and the other five still run.

```bash
# with the cloud TabPFN (full 6-model comparison):
TABPFN_TOKEN=<your-key> python experiments/run_external_shift.py \
  --config configs/experiments/shift_stress_external_2axis.yaml

# or without a token (5 models; tabicl is the ICL representative):
python experiments/run_external_shift.py \
  --config configs/experiments/shift_stress_external_2axis.yaml

# then figures:
python experiments/make_external_figures.py --tables results/external/tables_2axis_seed42
```

Outputs → `results/external/tables_2axis_seed42/` and `results/external/figures/`.

## Known limitations
- **covariate_shift is a no-op on all-categorical datasets** (no numeric columns to
  mean-shift), e.g. `car`, `tic-tac-toe`, `kr-vs-kp` — their covariate envelope is
  trivially maximal. Interpret covariate results on numeric/mixed datasets only:
  38 of 44 in the primary run, 9 of 12 in the stratified subset.
- **1 seed** for the full external run (the pilot's multi-seed CIs are not repeated
  here); treat external AURE as a directional check, not a CI-backed ranking. The
  12-dataset multiseed run is a partial answer — see the next point.
- **The 12-dataset subset is outcome-selected.** It was stratified on meta-features
  *and* balanced on the seed-42 outcome (`seed42_winner` / `seed42_margin` in
  `stratified_subset_12.csv`), and it is easier than the full benchmark (AURE ≈0.11
  vs ≈0.08). It answers "is the seed-42 ranking stable under reseeding", not "what is
  the multiseed external AURE" — never quote its AURE as the latter.
- **Two profile schemas are in play.** `dataset_profiles_external.csv` and the
  multiseed dir's copy carry `n_numeric` / `n_categorical`; the merged dir's
  `dataset_profiles.csv` comes from the pilot profiler and carries
  `numeric_feature_ratio` / `categorical_feature_ratio`. Each figure script must use
  the schema of the directory it reads (`make_stratified_figures.py` filters on
  `n_numeric > 0`, `make_external_figures.py` on the ratio column).
- **The failure rule can decide a cell at λ=0.** With the published thresholds,
  xgboost / hist_gbdt / logreg fail before any shift on 33–44% of cells, which forces
  ρ=0 there. This is a property of the metric, not of these runs — quantified in
  `results/ablations/`.
- **ICL in the smoke = `tabicl` only**; the full config adds `tabpfn_client` (needs token).
- `near_duplicate_proxy` is numeric-only (NaN for all-categorical datasets) and is a
  proxy, not a contamination detector.
- Very large / very wide datasets (>20000 rows, >1000 features) are deferred.
- Reference bar: with `gbdt_reference_models = xgboost, catboost`, if those are absent
  (e.g. smoke) the bar falls back to hist_gbdt — see the main repo's reference-bar note.

## Files
- `tice/datasets/external.py` — loader + curated list + profiler + registration
- `experiments/profile_external.py` — builds the profile audit CSV
- `experiments/run_external_shift.py` — runs the pipeline on external datasets
- `experiments/make_external_figures.py` — the four external figures
- `configs/experiments/shift_stress_external_2axis_smoke.yaml` / `..._2axis.yaml`
- `results/external/dataset_profiles_external.csv`, `results/external/smoke/`

**Canonical results (`results/tables/`, `results/canonical/`) are untouched.**
