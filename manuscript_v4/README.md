# StableShift manuscript and Source Data

**Anchored electricity-demand forecasting with bounded label-free adaptation**

Yanbo Bian, sole author and corresponding author. Computer Science and Technology,
Weihai International Academy, Beijing Jiaotong University, Weihai, China.
Email: 24722081@bjtu.edu.cn. No specific external funding.

Repository: https://github.com/bianyanbo44-afk/stable-shift-forecasting

Versioned evidence: https://github.com/bianyanbo44-afk/stable-shift-forecasting/releases/tag/v0.1.0

## Files

- `main.pdf` and `main.tex`: English article and editable source.
- `supplementary.pdf` and `supplementary.tex`: proof, data preparation, settings and complete statistical comparisons.
- `references.bib`: 28 cited references.
- `figures/`: five figures in vector PDF/SVG and 600 dpi PNG/TIFF.
- `tables/`: editable LaTeX tables.
- `source_data/`: aggregate and seed-level metrics, paired comparisons, run coverage, computational measurements and analysis manifests.
- `highlights.txt`: four manuscript highlights.

## Reproduction

From the repository root, redraw the figures and compile the included tables:

```text
python scripts/draw_manuscript_v4.py
python scripts/compile_manuscript_v4.py
```

LaTeX compilation requires `pdflatex`, `bibtex`, `elsarticle` and the packages
listed in the document sources. Figure generation uses the repository's included
panel-alignment helper. Python dependencies and environment records are described
in the repository README.

To recalculate results, first extract the completed-forecast evidence ZIP from
release v0.1.0 into the repository root. Preserve the archive's relative paths.

```text
python scripts/analyze_manuscript_v4.py
python scripts/build_manuscript_assets_v4.py
python scripts/draw_manuscript_v4.py
python scripts/compile_manuscript_v4.py
```

The evidence archive contains 2,120 completed scenario/variant forecast streams,
including prediction arrays, window metrics, protocols and source snapshots.
Reanalysis starts no training. For data download and training instructions, see
`docs/REPRODUCIBILITY.md` in the repository.

## Data and analysis

The main experiment covers EIA-930, ECL and Ausgrid at 24 h and 168 h, using seeds
11, 22, 33, 44 and 55. EIA observations cover 2023 through June 2026; historical
ECL and Ausgrid observations cover 2011-2014 and 2010-2013. Supplementary 48 h,
TimesNet, representation-penalty and regional-transfer experiments retain their
recorded seed counts. Exact splits, observation masks and evaluation definitions
are given in the supplement and accompanying manifests.

Every analyzed window table and prediction archive is identified in
`source_data/analysis_manifest.json`. The release's SHA-256 manifest identifies
the downloadable archives. Original datasets remain subject to provider terms.
