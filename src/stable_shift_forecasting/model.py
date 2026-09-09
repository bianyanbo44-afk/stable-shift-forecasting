"""Spectrally normalized causal TCN and low-rank drift adapter.

The neural path learns a correction to a last-daily-cycle anchor, which keeps
long-horizon optimization well conditioned while preserving an explicit stable
base and rapidly adapting low-rank branch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils import spectral_norm


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, spectral_bound: float) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = spectral_norm(nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation))
        self.spectral_bound = spectral_bound
        self._eval_scale = None

    def train(self, mode=True):
        if mode or mode != self.training:
            self._eval_scale = None
        return super().train(mode)

    def _load_from_state_dict(self, *args, **kwargs):
        self._eval_scale = None
        return super()._load_from_state_dict(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.pad(x, (self.left_padding, 0))
        output = self.conv(x)
        if self.spectral_bound > 0:
            # The frozen base has constant weights during online deployment.
            # Cache only its scalar norm correction, invalidated by training or
            # loading weights, to avoid an SVD for every forecast origin.
            scale = self._eval_scale
            if self.training or scale is None:
                weight = self.conv.weight
                norm = torch.linalg.matrix_norm(weight.reshape(weight.shape[0], -1), ord=2)
                scale = (self.spectral_bound / norm.detach()).clamp(max=1.)
                if not self.training:
                    self._eval_scale = scale
            output = output * scale
        return output


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float, spectral_bound: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation, spectral_bound)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation, spectral_bound)
        # Normalize each time step independently; sequence-wide GroupNorm can
        # erase the level and phase information needed for load forecasting.
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x).transpose(1, 2)
        x = torch.nn.functional.gelu(self.norm1(x)).transpose(1, 2)
        x = self.dropout(x)
        x = self.conv2(x).transpose(1, 2)
        x = self.norm2(x).transpose(1, 2)
        return torch.nn.functional.gelu(x + residual)


class LowRankAdapter(nn.Module):
    def __init__(self, channels: int, rank: int, scale: float = 1.0) -> None:
        super().__init__()
        self.down = nn.Linear(channels, rank, bias=False)
        self.up = nn.Linear(rank, channels, bias=False)
        self.scale = scale
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.up(self.down(x))


@dataclass(frozen=True)
class ModelOutput:
    prediction: torch.Tensor
    stable_representation: torch.Tensor
    drift_representation: torch.Tensor


class StableShiftForecaster(nn.Module):
    """Forecast model with an explicit frozen-base/adapter-update boundary."""

    def __init__(self, input_dim: int, target_dim: int, horizon: int, context_length: int = 336, hidden_channels: int = 128, levels: int = 4, kernel_size: int = 3, dropout: float = 0.1, adapter_rank: int = 8, spectral_norm_bound: float = 1.0, seasonal_period: int = 24, anchor_cycles: int = 7, cycle_average_weight: float = 0.6, robust_clip: float = 6.0, robust_filter: bool = True, robust_threshold: float = 6.0, robust_window: int = 25) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.horizon = horizon
        self.context_length = context_length
        self.hidden_channels = hidden_channels
        self.seasonal_period = seasonal_period
        self.robust_clip = float(robust_clip)
        self.robust_filter = bool(robust_filter)
        self.robust_threshold = float(robust_threshold)
        self.robust_window = max(3, int(robust_window) | 1)
        self.anchor_cycles = max(1, int(anchor_cycles))
        clipped_cycle_weight = float(max(1e-4, min(1.0 - 1e-4, cycle_average_weight)))
        self.cycle_mix_logit = nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(clipped_cycle_weight)))))
        self.input_projection = nn.Conv1d(input_dim, hidden_channels, 1)
        self.tcn = nn.ModuleList([TCNBlock(hidden_channels, kernel_size, 2**level, dropout, spectral_norm_bound) for level in range(levels)])
        self.stable_projection = nn.Linear(hidden_channels, hidden_channels)
        self.drift_adapter = LowRankAdapter(hidden_channels, adapter_rank)
        self.recent_projection = nn.Sequential(nn.Linear(seasonal_period * target_dim, hidden_channels), nn.GELU(), nn.LayerNorm(hidden_channels))
        # A shared linear decomposition correction follows the strongest
        # compact long-horizon inductive bias in the benchmark.  It starts at
        # zero on top of an exact seasonal-persistence anchor, so optimization
        # cannot produce an arbitrary cold-start forecast scale.
        self.anchor_seasonal = nn.Linear(context_length, horizon)
        self.anchor_trend = nn.Linear(context_length, horizon)
        self.head = nn.Sequential(nn.Linear(hidden_channels * 3, hidden_channels), nn.GELU(), nn.Linear(hidden_channels, horizon * target_dim))
        # Calendar covariates are known exactly at deployment.  The auxiliary
        # task therefore reconstructs masked load values only, rather than
        # rewarding the model for copying deterministic calendar features.
        self.reconstruction_head = nn.Linear(hidden_channels, target_dim)
        # Start from the strong, transparent seasonal prior. The neural path
        # then learns a correction, avoiding a cold-start jump far outside the
        # scale of the standardized targets.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        # A conservative residual gate selected on the validation protocol.
        # It prevents a short training run from overwriting the transparent
        # seasonal prior while retaining a learnable correction path.
        self.residual_scale = 0.5
        nn.init.zeros_(self.anchor_seasonal.weight)
        nn.init.zeros_(self.anchor_seasonal.bias)
        nn.init.zeros_(self.anchor_trend.weight)
        nn.init.zeros_(self.anchor_trend.bias)

    def encode_base(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a context without applying the mutable adapter.

        The base path is independent of the adapter parameters.  Exposing it
        explicitly lets deployment cache the expensive frozen TCN features and
        perform the label-free update on a detached feature tensor.  This is
        mathematically equivalent to :meth:`encode` while avoiding a backward
        pass through the frozen encoder for every forecast block.
        """
        if context.ndim != 3 or context.shape[-1] != self.input_dim:
            raise ValueError(f"context must have shape [batch, time, {self.input_dim}]")
        # Standardized load channels are bounded before entering the learned
        # encoder.  This gives the influence function a finite cap under the
        # anomaly stress test while leaving known calendar covariates intact.
        encoder_context = self._bounded_context(context)
        hidden = self.input_projection(encoder_context.transpose(1, 2))
        for block in self.tcn:
            hidden = block(hidden)
        sequence = hidden.transpose(1, 2)
        stable = self.stable_projection(sequence)
        return sequence, stable

    def encode(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence, stable = self.encode_base(context)
        drift = self.drift_adapter(sequence)
        return sequence, stable, drift

    def forecast_from_encoded(
        self,
        context: torch.Tensor,
        sequence: torch.Tensor,
        stable: torch.Tensor,
        return_representations: bool = False,
    ) -> torch.Tensor | ModelOutput:
        """Produce a forecast from frozen-base features.

        ``sequence`` and ``stable`` must have been produced by
        :meth:`encode_base` for the same context.  The method is public so the
        evaluator can update only ``drift_adapter`` between the feature pass
        and the forecast pass.
        """
        if sequence.ndim != 3 or stable.ndim != 3 or sequence.shape[:2] != context.shape[:2]:
            raise ValueError("encoded features and context have incompatible shapes")
        drift = self.drift_adapter(sequence)
        stable_last = stable[:, -1]
        drift_last = drift[:, -1]
        context = self._bounded_context(context)
        recent = context[..., : self.target_dim]
        if recent.shape[1] >= self.seasonal_period:
            recent = recent[:, -self.seasonal_period :]
        else:
            pad = self.seasonal_period - recent.shape[1]
            recent = torch.cat([recent[:, :1].expand(-1, pad, -1), recent], dim=1)
        recent_embedding = self.recent_projection(recent.flatten(1))
        residual_prediction = self.head(torch.cat([stable_last, drift_last, recent_embedding], dim=-1)).view(-1, self.horizon, self.target_dim)
        # Decompose each target series independently while sharing temporal
        # coefficients across entities, as in a compact DLinear-style anchor.
        values = context[..., : self.target_dim].transpose(1, 2)
        trend = nn.functional.avg_pool1d(
            nn.functional.pad(values, (self.seasonal_period // 2, self.seasonal_period - 1 - self.seasonal_period // 2), mode="replicate"),
            self.seasonal_period,
            stride=1,
        )
        seasonal = values - trend
        seasonal = self._temporal_anchor_input(seasonal)
        trend = self._temporal_anchor_input(trend)
        decomposition_correction = (self.anchor_seasonal(seasonal) + self.anchor_trend(trend)).transpose(1, 2)
        last_cycle = recent[:, -self.seasonal_period :]
        cycle_count = min(self.anchor_cycles, context.shape[1] // self.seasonal_period)
        if cycle_count >= 1:
            cycles = context[..., : self.target_dim][:, -cycle_count * self.seasonal_period :]
            cycle_mean = cycles.reshape(context.shape[0], cycle_count, self.seasonal_period, self.target_dim).mean(dim=1)
        else:
            cycle_mean = last_cycle
        cycle_weight = torch.sigmoid(self.cycle_mix_logit)
        base_cycle = (1.0 - cycle_weight) * last_cycle + cycle_weight * cycle_mean
        persistence_anchor = base_cycle.repeat(1, (self.horizon + self.seasonal_period - 1) // self.seasonal_period, 1)[:, : self.horizon]
        # The initialized output is exactly seasonal persistence.  Both the
        # decomposition correction and learned representation path are additive
        # and can improve it only when supported by training evidence.
        prediction = persistence_anchor + self.residual_scale * (decomposition_correction + residual_prediction)
        if return_representations:
            return ModelOutput(prediction, stable_last, drift_last)
        return prediction

    def forward(self, context: torch.Tensor, return_representations: bool = False) -> torch.Tensor | ModelOutput:
        sequence, stable = self.encode_base(context)
        return self.forecast_from_encoded(context, sequence, stable, return_representations=return_representations)

    def reconstruct(self, context: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.encode_base(context)
        return self.reconstruct_from_encoded(sequence)

    def reconstruct_from_encoded(self, sequence: torch.Tensor) -> torch.Tensor:
        """Reconstruct loads from a detached or live base feature sequence."""
        return self.reconstruction_head(self.drift_adapter(sequence))

    def _temporal_anchor_input(self, values: torch.Tensor) -> torch.Tensor:
        """Resize a target sequence to the anchor's configured context length.

        The production protocol always supplies ``context_length`` points. A
        length-safe path keeps the module usable for compact smoke tests and
        downstream callers without silently changing the production shape.
        """

        if values.shape[-1] == self.context_length:
            return values
        return torch.nn.functional.interpolate(
            values,
            size=self.context_length,
            mode="linear",
            align_corners=False,
        )

    def _bounded_context(self, context: torch.Tensor) -> torch.Tensor:
        if self.robust_clip <= 0 and not self.robust_filter:
            return context
        bounded = context.clone()
        load = bounded[..., : self.target_dim]
        if self.robust_filter and load.shape[1] >= self.robust_window:
            # A Hampel-style local filter suppresses isolated or contiguous
            # input spikes before they can overwrite the stable representation.
            # The rule is parameter-free at deployment: it uses only the
            # current context and a fixed standardized-scale threshold.
            half = self.robust_window // 2
            padded = torch.nn.functional.pad(load.transpose(1, 2), (half, half), mode="replicate")
            neighbourhood = padded.unfold(-1, self.robust_window, 1)
            local_median = neighbourhood.median(dim=-1).values.transpose(1, 2)
            local_mad = (neighbourhood - local_median.transpose(1, 2).unsqueeze(-1)).abs().median(dim=-1).values.transpose(1, 2)
            limit = self.robust_threshold * local_mad.clamp_min(0.20)
            load = torch.where((load - local_median).abs() > limit, local_median, load)
        if self.robust_clip > 0:
            load = load.clamp(-self.robust_clip, self.robust_clip)
        bounded[..., : self.target_dim] = load
        return bounded

    def freeze_base_for_adaptation(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name.startswith("drift_adapter.")

    def unfreeze_all(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = True

    def adapter_state(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().clone() for key, value in self.drift_adapter.state_dict().items()}

    def restore_adapter_state(self, state: dict[str, torch.Tensor]) -> None:
        self.drift_adapter.load_state_dict(state)
