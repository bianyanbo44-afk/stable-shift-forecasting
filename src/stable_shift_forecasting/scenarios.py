"""Deterministic distribution-shift transformations for evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ScenarioResult:
    context: torch.Tensor
    name: str
    shock_mask: np.ndarray | None = None


def _load_feature_count(context: torch.Tensor, target_dim: int) -> int:
    if context.ndim != 3 or context.shape[-1] <= target_dim:
        raise ValueError("context must include target and calendar features")
    return target_dim


def apply_scenario(context: torch.Tensor, target_dim: int, scenario: str, seed: int = 2026, missing_rate: float = 0.10, anomaly_scale: float = 4.0) -> ScenarioResult:
    """Apply a scenario to context only; future targets are never touched."""

    _load_feature_count(context, target_dim)
    transformed = context.clone()
    rng = np.random.default_rng(seed)
    load = transformed[..., :target_dim]
    shock_mask = None
    if scenario in {"normal", "natural_shock"}:
        return ScenarioResult(transformed, scenario)
    if scenario == "missingness":
        mask = torch.as_tensor(rng.random(size=load.shape) < missing_rate, dtype=torch.bool, device=load.device)
        load[mask] = 0.0
    elif scenario == "anomaly":
        length = max(1, load.shape[1] // 12)
        start = load.shape[1] // 2
        signed = torch.sign(load[:, start : start + length].mean(dim=1, keepdim=True)).clamp(min=0.0) * 2.0 - 1.0
        load[:, start : start + length] += signed * anomaly_scale * load.std(dim=1, keepdim=True).clamp_min(1e-4)
    elif scenario == "scale_shift":
        load.mul_(1.15).add_(0.10)
    elif scenario == "shock":
        length = max(1, load.shape[1] // 24)
        start = max(0, load.shape[1] - length)
        load[:, start:] *= 1.35
        shock_mask = np.ones(len(load), dtype=bool)
    elif scenario == "entity_holdout":
        # This is an entity-context holdout stress test: target channels are
        # retained for scoring, while a deterministic subset of entity inputs
        # is hidden at deployment. A strict unseen-entity study is configured
        # separately by training on a reduced column set.
        held_out = np.arange(0, target_dim, 4, dtype=int)
        if held_out.size == 0:
            held_out = np.array([0], dtype=int)
        load[:, :, held_out] = 0.0
        shock_mask = np.zeros(len(load), dtype=bool)
    elif scenario == "controlled_drift":
        load.mul_(1.15).add_(0.10)
        shock_mask = np.ones(len(load), dtype=bool)
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    return ScenarioResult(transformed, scenario, shock_mask)
