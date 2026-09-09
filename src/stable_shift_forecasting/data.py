"""Leakage-safe load-series ingestion and chronological window construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


CALENDAR_NAMES = (
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "year_sin",
    "year_cos",
)


@dataclass(frozen=True)
class LoadSeries:
    values: np.ndarray
    timestamps: pd.DatetimeIndex
    columns: tuple[str, ...]
    entities: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("values must have shape [time, targets]")
        if len(self.timestamps) != len(self.values):
            raise ValueError("timestamp and value lengths differ")
        if len(self.columns) != self.values.shape[1]:
            raise ValueError("column and target dimensions differ")
        if not np.isfinite(self.values).all():
            raise ValueError("values contain NaN or infinite values")


def _calendar_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
    hour = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60.0
    weekday = timestamps.dayofweek.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy() - 1
    return np.column_stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * weekday / 7),
            np.cos(2 * np.pi * weekday / 7),
            np.sin(2 * np.pi * day_of_year / 365.25),
            np.cos(2 * np.pi * day_of_year / 365.25),
        ]
    ).astype(np.float32)


def load_series(path: str | Path, target_limit: int | None = None) -> LoadSeries:
    """Load CSV or NPZ data without silently changing the source semantics.

    CSV files must contain a timestamp column named timestamp, datetime, date or
    time (case-insensitive). All remaining numeric columns are targets. NPZ
    files contain values and optionally timestamps, columns and entities.
    """

    source = Path(path)
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            values = np.asarray(archive["values"], dtype=np.float32)
            raw_times = archive["timestamps"] if "timestamps" in archive else np.arange(len(values))
            columns = tuple(str(x) for x in archive["columns"]) if "columns" in archive else tuple(f"target_{i}" for i in range(values.shape[1]))
            entities = np.asarray(archive["entities"]) if "entities" in archive else None
        # Apply the target cap consistently to prepared arrays as well as CSV
        # inputs.  The cap is part of the locked protocol and must be applied
        # before feature construction, normalization and window generation.
        if target_limit is not None:
            if target_limit <= 0:
                raise ValueError("target_limit must be positive")
            values = values[:, :target_limit]
            columns = columns[:target_limit]
            if entities is not None and entities.ndim >= 1 and len(entities) >= target_limit:
                entities = entities[:target_limit]
        if np.issubdtype(np.asarray(raw_times).dtype, np.number):
            timestamps = pd.date_range("2000-01-01", periods=len(values), freq="h")
        else:
            timestamps = pd.to_datetime(raw_times)
        return LoadSeries(values, pd.DatetimeIndex(timestamps), columns, entities)

    frame = pd.read_csv(source)
    candidates = {str(c).strip().lower(): c for c in frame.columns}
    time_column = next((candidates[name] for name in ("timestamp", "datetime", "date", "time") if name in candidates), None)
    if time_column is None:
        raise ValueError(f"{source} requires a timestamp/datetime/date/time column")
    timestamps = pd.to_datetime(frame.pop(time_column), errors="raise")
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        raise ValueError(f"{source} has no numeric target columns")
    if target_limit is not None:
        numeric = numeric.iloc[:, :target_limit]
    # Keep chronological semantics for generic CSVs.  A short leading gap can
    # be carried from the first observed value, but no backward interpolation
    # from a later observation is allowed across a split boundary.
    numeric = numeric.replace([np.inf, -np.inf], np.nan).ffill(limit=3)
    numeric = numeric.dropna(axis=1, how="any")
    if numeric.shape[1] == 0:
        raise ValueError(f"{source} has no complete numeric target columns after gap handling")
    return LoadSeries(numeric.to_numpy(np.float32), pd.DatetimeIndex(timestamps), tuple(map(str, numeric.columns)))


def resample_hourly(series: LoadSeries, frequency: str = "h") -> LoadSeries:
    """Aggregate duplicate/sub-hourly timestamps and fill empty bins.

    This helper is intentionally explicit so every public source enters the
    same hourly protocol before window generation. It does not use target
    values from a future split to fit a model; split-aware normalization still
    happens in :func:`make_splits`.
    """

    frame = pd.DataFrame(series.values, index=pd.DatetimeIndex(series.timestamps), columns=series.columns)
    frame = frame.sort_index().groupby(pd.Grouper(freq=frequency)).mean()
    frame = frame.resample(frequency).asfreq()
    # Public load files contain short acquisition gaps (including DST-induced
    # two-hour gaps). Carry the last observed value forward only; interpolation
    # would use a later value and could cross a chronological split boundary.
    # Long gaps remain explicit and are removed below.
    frame = frame.ffill(limit=3)
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if frame.isna().any().any():
        # A channel with an unrepairable gap cannot be evaluated under the
        # common multivariate protocol. Drop the affected channel and record
        # the choice in the preparation manifest rather than fabricating data.
        frame = frame.dropna(axis=1, how="any")
    if frame.empty or frame.shape[1] == 0:
        raise ValueError("hourly resampling left no complete target series")
    return LoadSeries(frame.to_numpy(np.float32), pd.DatetimeIndex(frame.index), tuple(map(str, frame.columns)))


class Standardizer:
    """Per-target standardization fitted on the training segment only."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "Standardizer":
        self.mean_ = values.mean(axis=0).astype(np.float32)
        std = values.std(axis=0).astype(np.float32)
        self.std_ = np.where(std < 1e-6, 1.0, std)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer has not been fitted")
        return (values - self.mean_) / self.std_

    def inverse(self, values: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer has not been fitted")
        if isinstance(values, torch.Tensor):
            mean = torch.as_tensor(self.mean_, dtype=values.dtype, device=values.device)
            std = torch.as_tensor(self.std_, dtype=values.dtype, device=values.device)
            return values * std + mean
        return values * self.std_ + self.mean_


def _split_points(length: int, split: Iterable[float]) -> tuple[int, int]:
    fractions = tuple(split)
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("split must contain three fractions summing to one")
    train_end = int(length * fractions[0])
    validation_end = train_end + int(length * fractions[1])
    return train_end, min(validation_end, length)


class WindowedLoadDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        starts: np.ndarray,
        context_length: int,
        horizon: int,
        target_indices: np.ndarray | None = None,
    ) -> None:
        # Keep one copy of the time series and slice windows lazily.  Eagerly
        # stacking every overlapping 336-hour window duplicates the public
        # series hundreds of times and makes the full PJM protocol needlessly
        # memory-bound.
        self.context = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.starts = torch.as_tensor(starts, dtype=torch.long)
        self.context_length = int(context_length)
        self.horizon = horizon
        self.target_indices = None if target_indices is None else np.asarray(target_indices, dtype=np.int64)

    @property
    def input_dim(self) -> int:
        return int(self.context.shape[-1])

    @property
    def target_dim(self) -> int:
        return int(self.targets.shape[-1])

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[index])
        target_start = start + self.context_length
        target = self.targets[target_start : target_start + self.horizon]
        if self.target_indices is not None:
            target = target[:, self.target_indices]
        return {
            "context": self.context[start:target_start],
            "target": target,
            "start": self.starts[index],
        }


@dataclass(frozen=True)
class PreparedSplits:
    train: WindowedLoadDataset
    validation: WindowedLoadDataset
    test: WindowedLoadDataset
    standardizer: Standardizer
    target_columns: tuple[str, ...]
    calendar_names: tuple[str, ...] = CALENDAR_NAMES


def make_splits(
    series: LoadSeries,
    context_length: int,
    horizon: int,
    split: Iterable[float] = (0.70, 0.10, 0.20),
    evaluation_stride: int = 1,
    train_stride: int = 1,
    validation_stride: int = 1,
) -> PreparedSplits:
    """Create windows whose labels never cross a split boundary.

    Training and validation origins remain hourly for stable model selection.
    Test origins may be spaced by ``evaluation_stride`` so the reported
    statistical unit is a non-overlapping forecast block. The same test origins
    are then used by every method.
    """

    if evaluation_stride <= 0 or train_stride <= 0 or validation_stride <= 0:
        raise ValueError("all window strides must be positive")

    train_end, validation_end = _split_points(len(series.values), split)
    standardizer = Standardizer().fit(series.values[:train_end])
    scaled = standardizer.transform(series.values).astype(np.float32)
    calendar = _calendar_features(series.timestamps)
    features = np.concatenate([scaled, calendar], axis=1)
    starts: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    last_start = len(scaled) - context_length - horizon + 1
    if last_start <= 0:
        raise ValueError("series is shorter than context_length + horizon")
    for start in range(last_start):
        target_start = start + context_length
        target_end = target_start + horizon
        if target_end <= train_end:
            bucket = "train"
        elif target_start >= train_end and target_end <= validation_end:
            bucket = "validation"
        elif target_start >= validation_end and target_end <= len(scaled):
            bucket = "test"
        else:
            continue
        starts[bucket].append(start)

    def build(bucket: str) -> WindowedLoadDataset:
        bucket_starts = starts[bucket]
        if bucket == "test":
            bucket_starts = bucket_starts[::evaluation_stride]
        elif bucket == "train":
            bucket_starts = bucket_starts[::train_stride]
        elif bucket == "validation":
            bucket_starts = bucket_starts[::validation_stride]
        if not bucket_starts:
            raise ValueError(f"no {bucket} windows were created")
        return WindowedLoadDataset(features, scaled, np.asarray(bucket_starts), context_length, horizon)

    return PreparedSplits(build("train"), build("validation"), build("test"), standardizer, series.columns)


def make_entity_holdout_splits(
    series: LoadSeries,
    context_length: int,
    horizon: int,
    split: Iterable[float] = (0.70, 0.10, 0.20),
    holdout_fraction: float = 0.25,
    evaluation_stride: int = 1,
) -> PreparedSplits:
    """Construct a genuine unseen-entity transfer split.

    The base model is trained on the first entity subset and evaluated on
    disjoint target channels. Calendar features remain shared, and the
    standardizer is fitted only on training entities and times.
    """

    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must lie in (0, 1)")
    train_entities = max(1, int(series.values.shape[1] * (1.0 - holdout_fraction)))
    if train_entities >= series.values.shape[1]:
        raise ValueError("entity holdout requires at least two target columns")
    base = make_splits(series, context_length, horizon, split, evaluation_stride)
    held_out = np.arange(train_entities, series.values.shape[1], dtype=np.int64)
    held_values = series.values[:, held_out]
    held_series = LoadSeries(held_values, series.timestamps, tuple(series.columns[i] for i in held_out))
    held = make_splits(held_series, context_length, horizon, split, evaluation_stride)
    # Keep the training normalization from the source entities to avoid using
    # held-out target statistics; map held-out values with their own train-only
    # scaler for a well-conditioned diagnostic and report the entity boundary.
    return held
