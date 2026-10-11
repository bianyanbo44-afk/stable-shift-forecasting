# QUART and StableShift

Code, manuscript and reproducibility materials for **QUART: Exact control of forecast revisions in online learning**.

QUART (Query-aware Updates with Adaptive Revision Truncation) learns from matured forecast errors while bounding the revision it induces at the current, unlabelled query. A closed-form rule selects the largest feasible fraction along a normalised residual-learning proposal. The source forecaster remains frozen.

[QUART v0.4.0 release](https://github.com/bianyanbo44-afk/stable-shift-forecasting/releases/tag/quart-v0.4.0) · [Current manuscript](docs/quart/main.pdf) · [Supplementary Information](docs/quart/supplementary.pdf) · [Source Data](https://github.com/bianyanbo44-afk/stable-shift-forecasting/releases/download/quart-v0.4.0/QUART_Source_Data_v0.4.0.zip) · [Complete reproduction archive](https://github.com/bianyanbo44-afk/stable-shift-forecasting/releases/download/quart-v0.4.0/QUART_Reproduction_v0.4.0.zip) · [Reproduction instructions](docs/QUART_REPRODUCIBILITY.md)

## Evidence and scope

The recorded evaluation covers three electricity-demand datasets, two frozen sources, 24 h/168 h horizons and five seeds: 60 original-benchmark cells, 20 EIA re-evaluations and 40 fixed-configuration entity-transfer cells.

- All 23,375 active updates met the current-query displacement and observed-feedback loss tolerances.
- Across 12 benchmark settings, mean relative MAE differences were -4.02% against frozen forecasting and -0.74% against independently tuned NLMS.
- Across eight new-entity settings, the differences were -15.13% against frozen forecasting and +2.04% against fixed NLMS; the ECL 24 h StableShift contrast was +21.62% against NLMS.
- Matched-step historical clipping exceeded the shared budget in 13.00% of benchmark updates and 24.29% of entity-transfer updates.
- The median controlled-COSA/QUART calibration-and-scoring time ratio was 19.6 across 11 active seed-11 settings, excluding shared source inference.

Temporal comparisons revisit development periods and remain exploratory. Entity transfer uses the original calendar, archived source models, fixed positional mapping and configurations selected before scoring. The bound applies to one learning update at a fixed current query in joint training-standardised RMS. Exact control refers to the feasible fraction on the proposal ray. Complete adverse outcomes, block sensitivities and physical-unit budget interpretation remain in the paper and supplement.

## Reproduce

Download `QUART_Reproduction_v0.4.0.zip` from the release and extract it into a separate directory. It preserves all previous computational entries and adds `quart_v0.4.0/` with the current paper, complete Source Data, figure script and bundled QA helpers.

```bash
python -m pip install -r quart_v0.4.0/requirements-figures.txt
python quart_v0.4.0/draw_quart_artwork.py --check-inputs
python quart_v0.4.0/draw_quart_artwork.py
```

For source-model/metric replay, install the original project with the analysis and upstream requirements, using the recorded PyTorch/CUDA environment:

```bash
python -m pip install -e ".[analysis]" -r requirements-upstream.txt
python research_strengthening_20261009/replay_strengthening.py
```

The default replay is the archived Ausgrid 24 h seed-11 StableShift entity-transfer cell. [Detailed instructions](docs/QUART_REPRODUCIBILITY.md) explain the current figure workflow and inherited metric replay.

## Files

| Path | Contents |
|---|---|
| `docs/quart/` | Current manuscript/SI PDFs, LaTeX/Biber sources, 34 verified references, used tables and final figures |
| `scripts/quart/` | Current deterministic figure script, pinned requirements and portable QA helpers |
| `research_strengthening_20261009/` | Original calibration evaluation, paired inference and replay |
| `research_innovation_20260924/` | Original numerical operator and controlled COSA implementation |
| `src/stable_shift_forecasting/` | Frozen forecasting sources, metrics and data interfaces |
| `TERMINOLOGY_MAP.json` | QUART/QURB/QORC/QNC mapping to the unchanged `query_expanded` rule |

## Versions and licences

QUART is the current display name for the rule previously presented as QURB and archived as QORC/QNC/`query_expanded`. This publication synchronization does not change selections, trajectories, checkpoints, bootstrap results or raw experimental identifiers. The [QORC v0.3.0 release](https://github.com/bianyanbo44-afk/stable-shift-forecasting/releases/tag/qorc-v0.3.0), [QNC v0.2.0 release](https://github.com/bianyanbo44-afk/stable-shift-forecasting/releases/tag/qnc-v0.2.0), and [StableShift v0.1.0 release](https://github.com/bianyanbo44-afk/stable-shift-forecasting/releases/tag/v0.1.0) remain accessible. Prior README and citation metadata are retained in `docs/history/`.

Use [CITATION.cff](CITATION.cff) and the versioned release URL for the current research materials. Author-created software retains the root MIT licence. COSA retains CC BY-NC-SA 4.0; other third-party software and provider datasets retain their own terms. The software licence does not relicense manuscript text, figures or third-party data. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
