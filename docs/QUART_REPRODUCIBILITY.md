# Reproducing QUART v0.4.0

Study: **QUART: Exact control of forecast revisions in online learning**. Released 12 October 2026. Public record: https://github.com/bianyanbo44-afk/stable-shift-forecasting/releases/tag/quart-v0.4.0.

## Current files

`QUART_Manuscript.pdf` and `QUART_Supplementary_Information.pdf` are the current paper. `QUART_Manuscript_Source_v0.4.0.zip` includes their LaTeX sources, the prebuilt 34-reference Biber bibliography, all used input tables and five figures. It excludes cover letters, private author-confirmation records and submission-system forms.

`QUART_Source_Data_v0.4.0.zip` is byte-identical to the final manuscript's 06_Source_Data.zip. Its raw labels and provenance remain historical; `TERMINOLOGY_MAP.json` maps them to QUART. `QUART_Reproduction_v0.4.0.zip` preserves every entry of the complete local reproduction package and adds a current `quart_v0.4.0/` interface.

## Figure workflow

After extracting the reproduction archive, run:

```bash
python -m pip install -r quart_v0.4.0/requirements-figures.txt
python quart_v0.4.0/draw_quart_artwork.py --check-inputs
python quart_v0.4.0/draw_quart_artwork.py
```

The script detects `source_data/` and the portable helpers relative to itself. It produces current PDF/SVG/600 dpi PNG/TIFF files in `quart_v0.4.0/source/figures/` and QA records in `quart_v0.4.0/quality/artwork/`. Each displayed statistic is read from the recorded data; no bootstrap, selection or training occurs. At manuscript width, the smallest figure text is approximately 8.2 pt.

To use the Git checkout alone, extract the Source Data archive into a work directory's `source_data/`, set `QUART_SUBMISSION_DIR` to that work directory and run `python scripts/quart/draw_quart_artwork.py`. `QURB_SUBMISSION_DIR` is an inherited input alias, and `NATURE_FIGURE_HELPERS` can override the bundled QA helpers.

## Numerical replay and licences

The original replay entry point remains `research_strengthening_20261009/replay_strengthening.py`. The archive contains all 60 selected source checkpoints, prepared original/new-cohort arrays and per-origin evidence. Install the root project with its original analysis/upstream requirements and the recorded PyTorch/CUDA environment. The archived default replay checks Ausgrid 24 h StableShift seed 11 entity transfer. Other cells are selected with the existing phase/dataset/horizon/source/seed arguments.

The established replay, experimental chronology, provider access and licence details are retained in [QORC_REPRODUCIBILITY.md](QORC_REPRODUCIBILITY.md). This release performs publication synchronization, not a new training or statistical run. Third-party/provider terms and recorded causal/scoring protocols remain unchanged. `SHA256SUMS.txt` covers every downloadable release asset except itself; `QUART_ADDITIONS_SHA256.txt` covers the newly appended reproduction interface.
