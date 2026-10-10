# QORC and StableShift

Code and research materials for **Learning from delayed feedback with bounded forecast updates**.

QORC (Query-aware Online Residual Calibration) learns from matured forecast errors while bounding the change it induces at the current forecast. It uses a closed-form normalised residual update over a frozen source forecaster. The original StableShift forecasting implementation and its earlier study remain available in this repository.

[QORC v0.3.0 research release](https://github.com/bianyanbo44-afk/stable-shift-forecasting/releases/tag/qorc-v0.3.0) · [Source Data](https://github.com/bianyanbo44-afk/stable-shift-forecasting/releases/download/qorc-v0.3.0/QORC_Source_Data_v0.3.0.zip) · [Complete reproduction archive](https://github.com/bianyanbo44-afk/stable-shift-forecasting/releases/download/qorc-v0.3.0/QORC_Reproduction_v0.3.0.zip) · [Reproduction instructions](docs/QORC_REPRODUCIBILITY.md)

## Results and evaluation

The release includes 60 validation selections, 60 original-benchmark evaluations, 20 EIA re-evaluations and 40 fixed-configuration entity-transfer evaluations. It covers three electricity-demand datasets, two frozen sources, 24 h/168 h horizons and five source seeds.

- All 23,375 active QORC updates met the current-query displacement and feedback-loss tolerances.
- Mean benchmark MAE was 4.02% lower than frozen forecasting and 0.74% lower than independently tuned NLMS.
- New-entity MAE was 15.13% lower than frozen forecasting and 2.04% higher than independently tuned NLMS.
- Historical clipping exceeded the matched QORC query budget in 13.00% of benchmark updates and 24.29% of entity-transfer updates.
- The median controlled-COSA/QORC calibration-and-scoring time ratio was 19.6 across 11 active seed-11 settings, excluding shared source inference.

Temporal comparisons revisit development periods and are exploratory. ECL/Ausgrid entities 101–116 use archived sources, fixed positional transfer and configurations selected before cohort preparation. Complete results and five budget profiles accompany the release. QORC names the same rule archived as QNC/`query_expanded`; raw identifiers remain stable.

## Install and reproduce

Use Python 3.12 and an appropriate CUDA PyTorch build for metric replay:

```bash
python -m pip install -e ".[analysis]" -r requirements-upstream.txt
```

Download and extract `QORC_Reproduction_v0.3.0.zip` into a separate directory. It contains the prepared arrays, 60 selected source checkpoints and all recorded trajectories required for replay.

```bash
python research_strengthening_20261009/replay_strengthening.py
python research_strengthening_20261009/analyze_strengthening.py --phase all
python research_strengthening_20261009/derive_strengthening.py
python research_strengthening_20261009/summarize_control.py
```

The default replay regenerates the Ausgrid 24 h seed-11 StableShift entity-transfer cell and compares its metrics with the released records. [Detailed instructions](docs/QORC_REPRODUCIBILITY.md) describe requirements, figures and data preparation.

## Files

| Path | Contents |
|---|---|
| `research_strengthening_20261009/` | Calibration selection, evaluation, paired inference, budget/runtime analysis and figure code |
| `research_strengthening_20261009/analysis/` | Compact complete statistical summaries; full per-origin data are in the release |
| `research_innovation_20260924/` | Original QNC operator and controlled COSA implementation |
| `src/stable_shift_forecasting/` | Frozen forecasting sources, metrics and data interfaces |
| `docs/qorc_figures/` | Five vector figures corresponding to this research version |
| `TERMINOLOGY_MAP.json` | QORC/QNC identifier mapping |

## Citation and licences

Use [CITATION.cff](CITATION.cff) and the versioned release URL for these research materials. Author-created software retains the root MIT licence. Third-party COSA code retains CC BY-NC-SA 4.0 and its original notices; provider terms govern source datasets. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The [StableShift v0.1.0 release](https://github.com/bianyanbo44-afk/stable-shift-forecasting/releases/tag/v0.1.0) and [QNC v0.2.0 release](https://github.com/bianyanbo44-afk/stable-shift-forecasting/releases/tag/qnc-v0.2.0) remain available for the earlier studies. The original StableShift instructions are preserved in [docs/STABLESHIFT_LEGACY_README.md](docs/STABLESHIFT_LEGACY_README.md).
