# Observed-target scoring protocol, fixed before rescoring

Date: 2026-09-24. This change addresses outcome availability, not selection of a
favourable algorithm result. The source data, model forecasts, chronological
origins and update trajectories remain unchanged.

The inspected validation second-half coverage is:

| Dataset / H | Daily origins | Complete future blocks | Observed target fraction | Complete blocks after first feedback |
|---|---:|---:|---:|---:|
| EIA / 24 h | 183 | 175 | 99.61% | 174 |
| EIA / 168 h | 177 | 151 | 99.60% | 144 |
| ECL / 24 h | 73 | 73 | 100% | 72 |
| ECL / 168 h | 67 | 67 | 100% | 60 |
| Ausgrid / 24 h | 55 | 12 | 95.21% | 11 |
| Ausgrid / 168 h | 49 | 6 | 95.00% | 0 |

The old complete-block score has no post-feedback evaluation support in the
last cell. An unchanged score there cannot diagnose whether parameters or
unscored forecasts responded to updating.

Primary future scoring will use each forecast-target entry whose provider
label is observed, with identical masks across all methods. No missing label
is imputed. MAE and MSE are total observed absolute/squared error divided by
the count of observed forecast-target entries. RMSE is the square root of MSE.
MASE uses the same observed entries and the existing training-only seasonal
scale. Numerators and counts are retained per issued forecast for paired
seed/time-block bootstrap resampling of the ratio estimand.

Complete-block scoring will remain a prespecified sensitivity analysis. Both
total coverage and post-feedback coverage will be reported. Missing targets
never enter adaptation losses; delayed feedback remains strictly earlier
than each issuance time. No future observation mask selects an adaptation
action. This protocol does not make an assumption that unobserved targets
would have the same error distribution as observed ones.

These are development benchmarks. A new EIA July–August 2026 holdout will be
retrieved and evaluated only after the method, hyperparameter grid, outcome
mask and selection rule are fixed. Its outcome values have not been inspected
in this work.
