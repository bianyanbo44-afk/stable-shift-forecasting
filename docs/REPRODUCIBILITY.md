# Reproducing StableShift

Run all commands from the repository root. The release archive preserves the original project-relative paths used by the analysis scripts.

## Completed experimental evidence

Download the evidence ZIP and its `package_manifest.json` from the [v0.1.0 release](https://github.com/bianyanbo44-afk/stable-shift-forecasting/releases/tag/v0.1.0). Verify its SHA-256 value against the manifest, then extract the ZIP into the repository root. The retained result roots are:

- `results/benchmark_v3_formal/`: EIA-930 main evaluation.
- `cloud_returned_20260909/results/cloud_ecl_v3/`: ECL, 24 h.
- `cloud_returned_20260909/results/cloud_ecl_h168_v3/`: ECL, 168 h.
- `cloud_returned_20260909/results/cloud_ausgrid_v3/`: Ausgrid, 24/168 h.
- `cloud_returned_20260909/results/cloud_pjm_v3/`: supplementary archival PJM runs.
- `cloud_returned_20260909/results/benchmark_v3_transfer/`: ERCO/SWPP transfer.

Forecast streams include all scheduled origins, including unscored blocks with unavailable labels. Run protocols, source snapshots and selection records establish the connection between each forecast and its implementation. Training weight checkpoints are retained separately by the author; the public evidence archive contains the saved forecasts needed for the reported analyses.

## Prepare observed data

### Recent regional demand

```bash
python scripts/download_recent_load.py
```

This retrieves the EIA-930 six-month balance files for 2023–2025 and January–June 2026, selects observed Demand in MW, and writes `data/prepared_v2/eia930_2023_2026.npz`. End-of-hour UTC timestamps and observation masks are preserved.

### Historical benchmarks

```bash
python scripts/download_data.py --dataset ecl --output data/raw_ecl.zip --manifest data/ecl_download.json
python scripts/download_data.py --dataset pjm --output data/raw_pjm_kaggle.zip --manifest data/pjm_download.json
python scripts/download_data.py --dataset ausgrid --output data/raw_ausgrid.zip --manifest data/ausgrid_download.json
python scripts/prepare_archival_v3.py
```

The downloader records the actual source URL and checksum. Its Ausgrid default is the distribution mirror used in the recorded data preparation; the provider's documentation is available on the [Ausgrid data page](https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data). Source metadata are retained in `results/data_manifest.json`.

Preparation selects ECL channels MT_001–MT_016 and Ausgrid general-consumption customers 001–016. Four ECL quarter-hour power readings are averaged into each hourly kW value; two Ausgrid half-hour energy readings are summed into hourly kWh. The script also prepares the archival PJM data used in supplementary experiments. The recorded provider terms apply to the original datasets.

## Training entry points

Examples below launch new runs. Use an appropriate CUDA environment and a new output directory when changing a configuration.

```bash
python scripts/run_benchmark_v3.py --config configs/benchmark_v3.json --datasets eia930 --horizons 24 168 --output results/reproduction_eia
python scripts/run_benchmark_v3.py --config configs/benchmark_v3_archival.json --datasets ecl ausgrid --horizons 24 168 --output results/reproduction_archival
python scripts/run_entity_transfer_v3.py --config configs/entity_transfer_v3.json
```

For an exact archived run, use its saved `protocol.json` settings, source snapshot, data manifest and environment. The provided configurations include supplementary methods; the completed seed/method coverage for the paper is recorded in `manuscript_v4/source_data/run_coverage.csv`.

## Statistical analysis

With the completed-evidence archive extracted:

```bash
python scripts/analyze_manuscript_v4.py
python scripts/build_manuscript_assets_v4.py
```

The analysis uses paired seed/time-block resampling and preserves missing scheduled positions. The main paired tables use 5,000 bootstrap resamples; complete window metrics and secondary analyses accompany the release. Figure rendering then reads the generated Source Data.

## Software provenance

The original forecasting implementations are byte-preserved in the repository and archived source snapshots. Time-Series-Library is pinned to commit `4e938a1767106324dd753b2a44832bf870a0252e`. Post-processing and publishing utilities include portability updates for creating output directories and running without optional local figure-audit tools. These updates do not change the model, training or saved forecasts.
