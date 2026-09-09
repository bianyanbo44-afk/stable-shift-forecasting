"""Chronological observed-label benchmark data, shared by every method."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .data import _calendar_features


class ForecastWindows(Dataset):
    def __init__(self, features, targets, calendar, origins, length, horizon, observed):
        self.features = features
        self.targets = targets
        self.calendar = calendar
        self.origins = np.asarray(origins, dtype=np.int64)
        self.length = length
        self.horizon = horizon
        self.observed = observed

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        origin = int(self.origins[index])
        return {"context": torch.from_numpy(self.features[origin-self.length:origin]),
                "target": torch.from_numpy(self.targets[origin:origin+self.horizon]),
                "future_calendar": torch.from_numpy(self.calendar[origin:origin+self.horizon]),
                "context_observed": torch.from_numpy(self.observed[origin-self.length:origin]),
                "target_observed": torch.from_numpy(self.observed[origin:origin+self.horizon]),
                "origin": origin}


@dataclass
class BenchmarkData:
    features: np.ndarray
    targets: np.ndarray
    calendar: np.ndarray
    observed: np.ndarray
    timestamps: pd.DatetimeIndex
    mean: np.ndarray
    std: np.ndarray
    mase_scale: np.ndarray
    train_end: int
    validation_end: int
    warmup: int
    manifest: dict

    def windows(self, split, length, horizon, stride, include_unobserved=False):
        boundaries = {"train": (max(self.warmup+length, length), self.train_end),
                      "validation": (self.train_end, self.validation_end),
                      "test": (self.validation_end, len(self.targets))}
        first, end = boundaries[split]
        candidates = np.arange(first, end-horizon+1, stride)
        # Missing values may be carried forward in context; labels stay observed.
        origins = candidates if include_unobserved else [o for o in candidates if self.observed[o:o+horizon].all()]
        if len(origins)==0:
            raise ValueError(f"No observed {split} windows, H={horizon}")
        return ForecastWindows(self.features, self.targets, self.calendar, origins, length, horizon, self.observed)


def load_benchmark_data(path: Path, name: str, target_limit=None):
    with np.load(path, allow_pickle=False) as archive:
        raw = np.array(archive["values"], dtype=np.float32)
        stamps = pd.DatetimeIndex(archive["timestamps"])
        columns = np.array(archive["columns"]).astype(str)
        observed = np.array(archive["observed"], bool) if "observed" in archive else np.isfinite(raw)
    if target_limit:
        raw, observed, columns = raw[:, :target_limit], observed[:, :target_limit], columns[:target_limit]
    if not stamps.is_monotonic_increasing or stamps.has_duplicates:
        raise ValueError("Prepared timestamps must be unique and chronological")
    frame = pd.DataFrame(np.where(observed & np.isfinite(raw), raw, np.nan), index=stamps)
    # Retain holes in the time axis instead of compressing elapsed time.
    timeline = pd.date_range(stamps[0], stamps[-1], freq="h")
    frame = frame.reindex(timeline)
    raw = frame.to_numpy(dtype=np.float32)
    observed = np.isfinite(raw)
    if name == "eia930":
        train_end = int(timeline.searchsorted(pd.Timestamp("2025-01-01 01:00")))
        validation_end = int(timeline.searchsorted(pd.Timestamp("2026-01-01 01:00")))
    else:
        train_end = int(.7*len(raw))
        validation_end = train_end+int(.1*len(raw))
    mean = np.nanmean(raw[:train_end], axis=0).astype(np.float32)
    std = np.nanstd(raw[:train_end], axis=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.).astype(np.float32)
    if not np.isfinite(mean).all():
        raise ValueError("A channel has no observed training values")
    filled = frame.ffill().to_numpy(dtype=np.float32)
    valid_rows = np.isfinite(filled).all(axis=1)
    warmup = int(np.flatnonzero(valid_rows)[0])
    # Leading placeholders never enter an eligible context or target window.
    filled[:warmup] = np.nan_to_num(filled[:warmup], nan=0.)
    calendar = _calendar_features(timeline)
    z = ((filled-mean)/std).astype(np.float32)
    lag = 24
    pair_observed = observed[lag:train_end] & observed[:train_end-lag]
    deltas = np.where(pair_observed, np.abs(raw[lag:train_end]-raw[:train_end-lag]), np.nan)
    scales = np.nanmean(deltas, axis=0).astype(np.float64)
    if not (np.isfinite(scales) & (scales > 0)).all():
        raise ValueError("Undefined seasonal MASE scale")
    manifest = {"source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "dataset": name, "columns": columns.tolist(), "shape": list(raw.shape),
        "first_timestamp": str(timeline[0]), "last_timestamp": str(timeline[-1]),
        "train_end_exclusive": str(timeline[train_end]),
        "validation_end_exclusive": str(timeline[validation_end]),
        "leading_excluded_hours": warmup, "missing_by_channel": (~observed).sum(0).tolist(),
        "label_policy": "Exclude windows containing any unobserved future label",
        "input_policy": "Past-only forward fill; leading incomplete contexts excluded",
        "scale_policy": "Observed training values only; channel-wise MASE lag=24"}
    return BenchmarkData(np.c_[z, calendar].astype(np.float32), z, calendar, observed,
                         timeline, mean, std, scales, train_end, validation_end, warmup, manifest)
