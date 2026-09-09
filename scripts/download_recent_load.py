"""Download actual EIA-930 hourly demand, including 2026 observations."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REGIONS = ["PJM", "MISO", "CISO", "NYIS", "ISNE", "ERCO", "SWPP"]


def download(period: tuple[int, str]) -> dict:
    year, half = period
    name = f"EIA930_BALANCE_{year}_{half}.csv"
    url = f"https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/{name}"
    target = ROOT / "data" / "eia930" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as handle:
            headers = dict(response.headers)
            while block := response.read(1024 * 1024):
                handle.write(block)
        partial.replace(target)
    else:
        headers = {}
    record = {"url": url, "path": str(target.relative_to(ROOT)), "bytes": target.stat().st_size,
              "sha256": hashlib.file_digest(target.open("rb"), "sha256").hexdigest(),
              "retrieved_utc": datetime.now(timezone.utc).isoformat(), "headers": headers}
    print(json.dumps({"downloaded": name, "bytes": record["bytes"]}), flush=True)
    return record


def main() -> None:
    periods = [(year, half) for year in [2023, 2024, 2025] for half in ["Jan_Jun", "Jul_Dec"]]
    periods.append((2026, "Jan_Jun"))
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(download, periods))
    frames = []
    for record in records:
        frame = pd.read_csv(ROOT / record["path"], usecols=["Balancing Authority", "UTC Time at End of Hour", "Demand (MW)"], low_memory=False)
        frame = frame[frame["Balancing Authority"].isin(REGIONS)].copy()
        frame["timestamp"] = pd.to_datetime(frame["UTC Time at End of Hour"], utc=True).dt.tz_localize(None)
        frame["demand"] = pd.to_numeric(frame["Demand (MW)"].astype(str).str.replace(",", "", regex=False), errors="coerce")
        frames.append(frame[["timestamp", "Balancing Authority", "demand"]])
    combined = pd.concat(frames, ignore_index=True)
    pivot = combined.pivot_table(index="timestamp", columns="Balancing Authority", values="demand", aggfunc="mean").reindex(columns=REGIONS)
    # End-of-hour UTC timestamps are preserved, including every missing hour.
    timeline = pd.date_range("2023-01-01 01:00", "2026-07-01 00:00", freq="h")
    pivot = pivot.reindex(timeline)
    observed = pivot.notna() & (pivot >= 0)
    raw = pivot.where(observed)
    out = ROOT / "data/prepared_v2/eia930_2023_2026.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, values=raw.to_numpy(np.float32), timestamps=timeline.to_numpy().astype("datetime64[ns]"),
                        columns=np.asarray(REGIONS), observed=observed.to_numpy(bool))
    manifest = {"source": "U.S. EIA Form EIA-930", "value_field": "Demand (MW)",
                "timezone": "UTC", "timestamp_semantics": "End of hour", "units": "MW",
                "records": records, "output": str(out.relative_to(ROOT)), "shape": list(raw.shape),
                "regions": REGIONS, "missing_by_region": (~observed).sum().to_dict(),
                "first_timestamp": str(timeline[0]), "last_timestamp": str(timeline[-1]),
                "label_policy": "Raw observed demand only; no provider forecasts, adjusted or imputed columns are used.",
                "planned_split": {"train_end_exclusive": "2025-01-01 01:00", "validation_end_exclusive": "2026-01-01 01:00", "test": "2026 first half"},
                "output_sha256": hashlib.file_digest(out.open("rb"), "sha256").hexdigest()}
    (ROOT / "results/eia930_2026_data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "records"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
