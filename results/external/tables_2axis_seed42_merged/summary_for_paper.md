# External merged result — summary for paper (auto-generated)

Source: `results/external/tables_2axis_seed42_merged/` (44 OpenML datasets x 2 axes x 6 lambda x 6 models x seed 42 = 3168 rows, 0 skipped, 0 error).

## AURE (2-axis, external)
| model | AURE |
|---|---|
| tabicl | 0.086 |
| tabpfn_client | 0.081 |
| catboost | 0.076 |
| xgboost | 0.068 |
| hist_gbdt | 0.065 |
| logreg | 0.051 |

## Headline numbers
- TabPFN vs TabICL: per-dataset AURE corr **0.959**, aggregate Delta **+0.006** -> indistinguishable.
- ICL-vs-best-GBDT margin: pilot **+0.043** (6-axis) -> external **+0.005** (2-axis).
- Both ICL above all GBDT/linear on **19/44** datasets.
- label_noise AUC drop smallest for ICL (tabpfn 0.082, tabicl 0.088) vs GBDT 0.13-0.16; tabpfn < mean-GBDT on **40/44**.
- covariate (numeric/mixed): discrimination hurt moderately, calibration amplified sharply (NLL x4-5).

## Message
ICL occupies the top of the aggregate reliability ranking, but on real data the advantage over a strong GBDT (CatBoost) is small (+0.005 AURE) and dataset-dependent; what generalises robustly are the two mechanisms (label-noise ranking stability, covariate calibration collapse).
