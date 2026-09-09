"""Normalize common public load formats into the project NPZ schema."""

from __future__ import annotations

import argparse
import re
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def read_source(path: Path) -> pd.DataFrame:
    if path.suffix.lower() != ".zip":
        return _read_table(path)
    with zipfile.ZipFile(path) as archive:
        candidates = [name for name in archive.namelist() if name.lower().endswith((".csv", ".txt", ".tsv")) and not name.endswith("/")]
        if not candidates:
            raise ValueError(f"no tabular file found in {path}")
        candidate = sorted(candidates, key=lambda item: ("electric" not in item.lower(), len(item)))[0]
        with archive.open(candidate) as handle:
            return _read_table(handle, source_name=candidate)


def _read_table(source: object, source_name: str = "") -> pd.DataFrame:
    """Read common public load exports without assuming one delimiter."""

    suffix = Path(source_name).suffix.lower()
    source_lower = source_name.lower()
    decimal = "," if ("power_consumption" in source_lower or "ld2011" in source_lower) else "."
    if suffix in {".txt", ".tsv"} or "power_consumption" in source_lower or "ld2011" in source_lower:
        return pd.read_csv(source, sep=";", decimal=decimal, na_values=["?", "NA", "N/A"])
    return pd.read_csv(source, sep=None, engine="python", decimal=decimal, na_values=["?", "NA", "N/A"])


def _prepare_ecl(path: Path, target_limit: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        name = next(name for name in archive.namelist() if name.lower().endswith("ld2011_2014.txt"))
        frame = pd.read_csv(archive.open(name), sep=";", decimal=",", na_values=["?", "NA", "N/A"])
    timestamp = pd.to_datetime(frame.iloc[:, 0], errors="raise")
    numeric = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    # Keep the declared leading channels; no future-aware imputation is needed
    # for the ECL archive, whose selected channels are complete.
    if target_limit is not None:
        numeric = numeric.iloc[:, :target_limit]
    values = numeric.to_numpy(dtype=np.float32)
    return values, timestamp.to_numpy(dtype="datetime64[ns]"), np.asarray([str(c) for c in numeric.columns])


def _prepare_pjm(path: Path, target_limit: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            candidates = [n for n in archive.namelist() if n.lower().endswith("pjm_hourly_est.csv")]
            if not candidates:
                candidates = [n for n in archive.namelist() if n.lower().endswith("pjm_load_hourly.csv")]
            if not candidates:
                raise ValueError("PJM archive does not contain a recognized hourly CSV")
            with archive.open(sorted(candidates)[0]) as handle:
                frame = pd.read_csv(handle, parse_dates=["Datetime"])
    else:
        frame = pd.read_csv(path, parse_dates=["Datetime"])
    # PJME and PJMW are the longest, consistently identified regional series.
    preferred = [name for name in ("PJME", "PJMW") if name in frame.columns]
    # Kaggle's per-region files use an explicit ``*_MW`` suffix.
    if not preferred:
        for name in ("PJME", "PJMW"):
            suffixed = f"{name}_MW"
            if suffixed in frame.columns:
                frame = frame.rename(columns={suffixed: name})
                preferred.append(name)
    if target_limit is not None:
        preferred = preferred[:target_limit]
    if not preferred:
        raise ValueError("PJM source does not contain PJME/PJMW columns")
    frame = frame[["Datetime", *preferred]].dropna(subset=preferred).sort_values("Datetime")
    # Retain the longest contiguous hourly block; isolated DST gaps are filled
    # later only within the observed series by resample_hourly.
    timestamps = pd.DatetimeIndex(frame["Datetime"])
    values = frame[preferred].to_numpy(dtype=np.float32)
    return values, timestamps.to_numpy(dtype="datetime64[ns]"), np.asarray(preferred)


def _prepare_ausgrid(path: Path, target_limit: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[pd.DataFrame] = []
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if re.search(r"Solar home 20\d\d-20\d\d\.csv$", name, re.I)]
        for name in sorted(names):
            frame = pd.read_csv(archive.open(name), skiprows=1)
            frame = frame[frame["Consumption Category"].astype(str).str.upper().eq("GC")].copy()
            time_columns = [column for column in frame.columns if re.fullmatch(r"\d{1,2}:\d{2}", str(column))]
            if not time_columns:
                raise ValueError(f"no half-hour columns found in {name}")
            # The files use a fixed daily profile with 48 half-hour readings.
            # Convert each customer-day row into a long timestamp series.
            long = frame.melt(id_vars=["Customer", "date"], value_vars=time_columns, var_name="clock", value_name="load")
            dates = pd.to_datetime(long["date"], dayfirst=True, format="mixed", errors="raise")
            clock = pd.to_timedelta(long["clock"] + ":00")
            long["timestamp"] = dates + clock
            rows.append(long[["timestamp", "Customer", "load"]])
    long = pd.concat(rows, ignore_index=True).sort_values(["timestamp", "Customer"])
    wide = long.pivot_table(index="timestamp", columns="Customer", values="load", aggfunc="mean").sort_index()
    wide.columns = [f"customer_{int(column):03d}" for column in wide.columns]
    # One household has a documented incomplete period in the redistributed
    # 2012-2013 file. Exclude incomplete household channels before applying the
    # target cap; this preserves a complete multivariate panel without filling
    # a long missing block from future observations.
    wide = wide.dropna(axis=1, how="any")
    if target_limit is not None:
        wide = wide.iloc[:, :target_limit]
    # Missing half-hour records are retained as NaN here; the shared hourly
    # resampler fills only bounded gaps after aggregation.
    return wide.to_numpy(dtype=np.float32), wide.index.to_numpy(dtype="datetime64[ns]"), np.asarray([str(c) for c in wide.columns])


def prepare(frame: pd.DataFrame, target_limit: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = {str(c).strip().lower(): c for c in frame.columns}
    date_col = names.get("date")
    time_col = names.get("time")
    if date_col is not None and time_col is not None:
        timestamps = pd.to_datetime(frame[date_col].astype(str) + " " + frame[time_col].astype(str), dayfirst=True, errors="raise")
        frame = frame.drop(columns=[date_col, time_col])
    else:
        time_column = next((names[key] for key in ("timestamp", "datetime", "date", "time") if key in names), None)
        if time_column is None:
            # ECL stores its timestamp in an unnamed first column.
            candidate = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
            if candidate.notna().all():
                time_column = frame.columns[0]
            else:
                raise ValueError("source has no recognized timestamp column")
        timestamps = pd.to_datetime(frame.pop(time_column), errors="raise")
    candidate_columns = [column for column in frame.columns if column not in {date_col, time_col}]
    converted = frame[candidate_columns].copy()
    for column in converted.columns:
        if not pd.api.types.is_numeric_dtype(converted[column]):
            converted[column] = pd.to_numeric(converted[column].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    numeric = converted.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    # Do not fill source gaps here. The chronological split is defined after
    # hourly aggregation, and bounded gap handling is performed by the shared
    # resampler without borrowing values from a later split.
    numeric = numeric.dropna(axis=1, how="all")
    if target_limit is not None:
        numeric = numeric.iloc[:, :target_limit]
    if numeric.shape[1] == 0:
        raise ValueError("source has no numeric targets")
    order = np.argsort(timestamps.to_numpy())
    values = numeric.to_numpy(dtype=np.float32)[order]
    sorted_times = timestamps.to_numpy()[order].astype("datetime64[ns]")
    columns = np.asarray([str(c) for c in numeric.columns])
    return values, sorted_times, columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-limit", type=int)
    args = parser.parse_args()
    if args.dataset.lower() == "ecl":
        values, timestamps, columns = _prepare_ecl(args.input, args.target_limit)
    elif args.dataset.lower() == "pjm":
        values, timestamps, columns = _prepare_pjm(args.input, args.target_limit)
    elif args.dataset.lower() == "ausgrid":
        values, timestamps, columns = _prepare_ausgrid(args.input, args.target_limit)
    else:
        frame = read_source(args.input)
        values, timestamps, columns = prepare(frame, args.target_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, values=values, timestamps=timestamps.astype(str), columns=columns)
    print(f"prepared {args.dataset}: {values.shape} -> {args.output}")


if __name__ == "__main__":
    main()
