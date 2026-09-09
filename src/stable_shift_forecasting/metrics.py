"""Window-level metrics and paired bootstrap inference."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class MetricReport:
    mae: float
    rmse: float
    mase: float
    per_window_mae: np.ndarray
    per_window_rmse: np.ndarray


def parameter_count(model: object) -> int:
    """Return the number of parameters for torch modules or zero for rules."""

    try:
        import torch

        if isinstance(model, torch.nn.Module):
            return int(sum(parameter.numel() for parameter in model.parameters()))
    except ImportError:
        pass
    return 0


def timed_call(callable_object: object, *args: object, repeats: int = 1, **kwargs: object) -> float:
    """Measure wall-clock milliseconds for a callable, without synchronizing APIs."""

    if not callable(callable_object):
        raise TypeError("callable_object must be callable")
    start = perf_counter()
    for _ in range(repeats):
        callable_object(*args, **kwargs)
    return float((perf_counter() - start) * 1000.0 / repeats)


def _validate(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must both have shape [windows, horizon, targets]")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("prediction and target contain non-finite values")
    return prediction, target


def evaluate_predictions(prediction: np.ndarray, target: np.ndarray, training_history: np.ndarray | None = None, seasonal_period: int = 24) -> MetricReport:
    prediction, target = _validate(prediction, target)
    error = prediction - target
    abs_error = np.abs(error)
    squared_error = error**2
    per_window_mae = abs_error.mean(axis=(1, 2))
    per_window_rmse = np.sqrt(squared_error.mean(axis=(1, 2)))
    if training_history is None:
        scale = np.mean(np.abs(np.diff(target, axis=1)))
    else:
        history = np.asarray(training_history, dtype=np.float64)
        if history.ndim != 2 or history.shape[1] != target.shape[2]:
            raise ValueError("training_history must have shape [time, targets]")
        lag = min(seasonal_period, len(history) - 1)
        scale = np.mean(np.abs(history[lag:] - history[:-lag])) if lag > 0 else np.mean(np.abs(np.diff(history, axis=0)))
    scale = max(float(scale), 1e-8)
    return MetricReport(float(abs_error.mean()), float(np.sqrt(squared_error.mean())), float(abs_error.mean() / scale), per_window_mae, per_window_rmse)


def paired_bootstrap_difference(a: np.ndarray, b: np.ndarray, resamples: int = 2000, seed: int = 2026) -> dict[str, float | int]:
    """Return b-a with a paired interval.

    Callers should pass baseline as ``a`` and the proposed method as ``b``
    when they want the exported difference to mean proposed minus baseline.
    """

    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or a.size < 2:
        raise ValueError("paired bootstrap inputs must have equal length >= 2")
    rng = np.random.default_rng(seed)
    differences = b - a
    indices = rng.integers(0, len(differences), size=(resamples, len(differences)))
    estimates = differences[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    # A two-sided paired-bootstrap sign probability. The interval remains the
    # primary report; this p-value is only used for the pre-registered Holm
    # correction across the comparison family.
    p_value = 2.0 * min(float(np.mean(estimates <= 0.0)), float(np.mean(estimates >= 0.0)))
    return {"difference": float(differences.mean()), "ci_low": float(low), "ci_high": float(high), "p_value": float(min(1.0, p_value)), "n_windows": int(len(differences)), "resamples": int(resamples)}


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in original order."""

    values = np.asarray(p_values, dtype=float)
    if (values < 0).any() or (values > 1).any():
        raise ValueError("p-values must be in [0, 1]")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()
