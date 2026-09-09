"""Compact reference forecasters used under the locked protocol.

The implementations are deliberately small and self-contained. They are
reference implementations for fair controlled comparisons, not re-creations
of every optional feature in the original papers.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SeasonalNaive:
    def __init__(self, target_dim: int, horizon: int, period: int = 24) -> None:
        self.target_dim = target_dim
        self.horizon = horizon
        self.period = period

    def predict(self, context: torch.Tensor) -> torch.Tensor:
        load = context[..., : self.target_dim]
        repeats = math.ceil(self.horizon / self.period)
        tail = load[:, -self.period :]
        return tail.repeat(1, repeats, 1)[:, : self.horizon]


class Persistence:
    def __init__(self, target_dim: int, horizon: int) -> None:
        self.target_dim = target_dim
        self.horizon = horizon

    def predict(self, context: torch.Tensor) -> torch.Tensor:
        return context[:, -1:, : self.target_dim].repeat(1, self.horizon, 1)


class DLinear(nn.Module):
    def __init__(self, input_dim: int, target_dim: int, horizon: int, context_length: int, kernel_size: int = 25) -> None:
        super().__init__()
        self.target_dim = target_dim
        self.horizon = horizon
        self.kernel_size = min(kernel_size, context_length - 1)
        self.linear_seasonal = nn.Linear(context_length, horizon)
        self.linear_trend = nn.Linear(context_length, horizon)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        values = context[..., : self.target_dim].transpose(1, 2)
        trend = nn.functional.avg_pool1d(nn.functional.pad(values, (self.kernel_size // 2, self.kernel_size - 1 - self.kernel_size // 2), mode="replicate"), self.kernel_size, stride=1)
        seasonal = values - trend
        output = self.linear_seasonal(seasonal) + self.linear_trend(trend)
        return output.transpose(1, 2)


class TiDE(nn.Module):
    def __init__(self, input_dim: int, target_dim: int, horizon: int, context_length: int, hidden: int = 128) -> None:
        super().__init__()
        self.target_dim = target_dim
        self.horizon = horizon
        self.encoder = nn.Sequential(nn.Linear(context_length * input_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, horizon * target_dim))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(context.flatten(1))).view(-1, self.horizon, self.target_dim)


class PatchTST(nn.Module):
    def __init__(self, input_dim: int, target_dim: int, horizon: int, context_length: int, hidden: int = 64, patch: int = 16, layers: int = 2) -> None:
        super().__init__()
        self.target_dim = target_dim
        self.horizon = horizon
        self.patch = min(patch, context_length)
        self.patch_projection = nn.Linear(self.patch * input_dim, hidden)
        encoder_layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=hidden * 2, batch_first=True, dropout=0.1, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(hidden, horizon * target_dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        length = context.shape[1] - context.shape[1] % self.patch
        patches = context[:, -length:].reshape(context.shape[0], length // self.patch, self.patch * context.shape[2])
        encoded = self.encoder(self.patch_projection(patches))
        return self.head(encoded[:, -1]).view(-1, self.horizon, self.target_dim)


class TimesNet(nn.Module):
    """A compact frequency-mixing reference model."""

    def __init__(self, input_dim: int, target_dim: int, horizon: int, context_length: int, hidden: int = 64) -> None:
        super().__init__()
        self.target_dim = target_dim
        self.horizon = horizon
        self.projection = nn.Linear(input_dim, hidden)
        self.mixer = nn.Sequential(nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(), nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU())
        self.head = nn.Linear(hidden, horizon * target_dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        hidden = self.projection(context).transpose(1, 2)
        mixed = self.mixer(hidden).mean(dim=-1)
        return self.head(mixed).view(-1, self.horizon, self.target_dim)


class ITransformer(nn.Module):
    """Variate-token transformer with a temporal projection."""

    def __init__(self, input_dim: int, target_dim: int, horizon: int, context_length: int, hidden: int = 64, layers: int = 2) -> None:
        super().__init__()
        self.target_dim = target_dim
        self.horizon = horizon
        self.temporal = nn.Linear(context_length, hidden)
        encoder_layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=hidden * 2, batch_first=True, dropout=0.1, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        values = context[..., : self.target_dim].transpose(1, 2)
        encoded = self.encoder(self.temporal(values))
        return self.head(encoded).transpose(1, 2)


def build_baseline(name: str, input_dim: int, target_dim: int, horizon: int, context_length: int) -> object:
    factories = {
        "SeasonalNaive": lambda: SeasonalNaive(target_dim, horizon),
        "Persistence": lambda: Persistence(target_dim, horizon),
        "DLinear": lambda: DLinear(input_dim, target_dim, horizon, context_length),
        "PatchTST": lambda: PatchTST(input_dim, target_dim, horizon, context_length),
        "TimesNet": lambda: TimesNet(input_dim, target_dim, horizon, context_length),
        "iTransformer": lambda: ITransformer(input_dim, target_dim, horizon, context_length),
        "TiDE": lambda: TiDE(input_dim, target_dim, horizon, context_length),
    }
    if name not in factories:
        raise ValueError(f"unknown baseline {name}; choose from {sorted(factories)}")
    return factories[name]()
