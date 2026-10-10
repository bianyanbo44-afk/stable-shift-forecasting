# Reproducing QORC v0.3.0

Version: `qorc-v0.3.0`, released 10 October 2026. Study: Learning from delayed feedback with bounded forecast updates.

## Complete evidence

Download `QORC_Reproduction_v0.3.0.zip` from https://github.com/bianyanbo44-afk/stable-shift-forecasting/releases/tag/qorc-v0.3.0 and extract it into a clean directory, preserving relative paths. The archive includes the source models, 60 selected checkpoints, original and new-cohort prepared arrays, calibration selections, per-origin scoring records, update geometry, analysis code and third-party notices. The smaller `QORC_Source_Data_v0.3.0.zip` contains observed-entry sums/counts, paired statistics, selection records, timing measurements and figure-source summaries.

`FILE_SHA256SUMS.txt` covers every reproduction entry. `SHA256SUMS.txt` in the Release covers its downloadable archives and vector figures. Original scientific selections and checkpoints are retained byte-for-byte.

## Environment and replay

Install the root project with analysis dependencies and `requirements-upstream.txt`. The recorded research environment uses Python 3.12.10, CUDA PyTorch 2.6.0+cu124, NumPy 2.2.6 and four CPU threads. The frozen source and controlled COSA replay require CUDA; scalar calibration timings exclude shared source inference.

```bash
python -m pip install -e ".[analysis]" -r requirements-upstream.txt
python research_strengthening_20261009/replay_strengthening.py
```

The default replays Ausgrid, 24 h, seed 11, StableShift, entity transfer. Other recorded cells can be chosen with `--phase`, `--dataset`, `--horizon`, `--source` and `--seed`. The replay writes only new replay results and optional source caches. Source models are not retrained.

## Statistics and figures

```bash
python research_strengthening_20261009/analyze_strengthening.py --phase all
python research_strengthening_20261009/derive_strengthening.py
python research_strengthening_20261009/summarize_control.py
```

Inference is recomputed from per-origin sums/counts using crossed seeds and shared calendar-block draws. Seven-day blocks are primary, with 14/28-day sensitivity analyses and separate Holm families.

Install `research_strengthening_20261009/requirements-figures.txt` for plotting. Set `QNC_DISPLAY_METHOD=QORC`, `QNC_BRITISH_SPELLING=1`, `QNC_FIGURE_DIR` and `QNC_FIGURE_QA_DIR` to existing output directories, then run `draw_strengthening.py --figure accuracy`, `--figure budget`, `--figure transfer`, and `draw_control_revision.py`. `draw_qorc_workflow.py` produces the method schematic; `QNC_SUBMISSION_DIR` sets its output root. The environment names and raw method identifiers retain their archived QNC form.

## Data and interpretation

EIA-930, ECL and Ausgrid are obtained from the recorded provider URLs. Prepared original/new-cohort arrays and provider hashes are included. Original raw ECL/Ausgrid archives are only needed to repeat cohort preparation. Every method uses observed target entries and fully matured feedback. Source-model parameters remain frozen.

Original benchmark and July–August EIA temporal results are exploratory under the expanded search. New entity cohorts use the same historical calendar, archived source models and fixed ascending-ID mapping. Ten constant-training ECL channels remain in MAE/RMSE and feedback; secondary MASE uses the six channels with defined scales. The release contains the complete recorded choices and five budgets.

The query bound concerns one before/after update at the same current query, in joint-channel training-standardised RMS. It does not bound changes between separate issue times. The displacement comparison against the common QORC budget is descriptive and uses all active events.
