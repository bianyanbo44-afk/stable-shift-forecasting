"""Training and label-free adapter adaptation."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import StableShiftForecaster


@dataclass(frozen=True)
class TrainSummary:
    train_losses: list[float]
    validation_losses: list[float]


def _masked_load_view(
    model: StableShiftForecaster,
    context: torch.Tensor,
    probability: float = 0.15,
    seed: int | None = None,
    observed: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask observed load entries while retaining known calendar features."""

    load = context[..., : model.target_dim]
    if seed is None:
        mask = torch.rand_like(load) < probability
    else:
        generator = torch.Generator(device=load.device)
        generator.manual_seed(int(seed))
        mask = torch.rand(load.shape, dtype=load.dtype, device=load.device, generator=generator) < probability
    # The normal batch/window sizes make an empty mask vanishingly unlikely,
    # but force one element for tiny unit-test inputs and future reuse.
    if observed is not None:
        mask &= observed.to(device=load.device, dtype=torch.bool)
    if not bool(mask.any()):
        eligible = torch.ones_like(mask) if observed is None else observed.to(device=load.device, dtype=torch.bool)
        if not bool(eligible.any()):
            raise ValueError('Masked reconstruction requires at least one observed input')
        mask.reshape(-1)[torch.nonzero(eligible.reshape(-1),as_tuple=False)[0,0]]=True
    masked = context.clone()
    masked[..., : model.target_dim] = load.masked_fill(mask, 0.0)
    return masked, mask


def _masked_mse(reconstruction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared = (reconstruction - target).pow(2)
    weights = mask.to(dtype=squared.dtype)
    return (squared * weights).sum() / weights.sum().clamp_min(1.0)


def _consistency_loss(model: StableShiftForecaster, context: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(context[..., : model.target_dim]) * 0.01
    augmented = context.clone()
    augmented[..., : model.target_dim] = augmented[..., : model.target_dim] + noise
    _, stable_a, _ = model.encode(context)
    _, stable_b, _ = model.encode(augmented)
    return torch.mean((stable_a - stable_b) ** 2)


def train_model(
    model: StableShiftForecaster,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    consistency_weight: float,
    contraction_weight: float,
    max_grad_norm: float,
    device: str,
    reconstruction_weight: float = 0.05,
) -> TrainSummary:
    model.to(device)
    model.unfreeze_all()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    for _ in range(epochs):
        model.train()
        running = 0.0
        count = 0
        for batch in train_loader:
            context = batch["context"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(context, return_representations=True)
            prediction_loss = loss_fn(output.prediction, target)
            consistency = _consistency_loss(model, context) if consistency_weight > 0 else prediction_loss.new_zeros(())
            # Train the same masked reconstruction objective used at
            # deployment. This gives the low-rank branch a meaningful
            # self-supervised signal before it is exposed to online updates.
            if reconstruction_weight > 0:
                masked_context, mask = _masked_load_view(model, context)
                reconstruction = _masked_mse(
                    model.reconstruct(masked_context),
                    context[..., : model.target_dim],
                    mask,
                )
            else:
                reconstruction = prediction_loss.new_zeros(())
            # Normalize the contraction term by the representation scale so it
            # cannot overwhelm the forecast objective early in training.
            contraction = output.stable_representation.pow(2).mean()
            loss = prediction_loss + consistency_weight * consistency + contraction_weight * contraction + reconstruction_weight * reconstruction
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            running += float(loss.detach()) * len(context)
            count += len(context)
        train_losses.append(running / max(1, count))
        model.eval()
        val_running = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in validation_loader:
                context = batch["context"].to(device)
                target = batch["target"].to(device)
                val_running += float(loss_fn(model(context), target)) * len(context)
                val_count += len(context)
        validation_loss = val_running / max(1, val_count)
        validation_losses.append(validation_loss)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return TrainSummary(train_losses, validation_losses)


def adapt_on_context(
    model: StableShiftForecaster,
    context: torch.Tensor,
    steps: int = 1,
    learning_rate: float = 0.01,
    max_grad_norm: float = 1.0,
    mask_probability: float = 0.15,
    adapter_max_frobenius: float = 8.0,
    reference_mean: torch.Tensor | None = None,
    reference_std: torch.Tensor | None = None,
) -> tuple[float, float]:
    """Update only the low-rank adapter using masked reconstruction.

    Returns the reconstruction loss before and after adaptation. No target
    values are accepted by this interface.
    """

    if context.ndim != 3:
        raise ValueError("context must have shape [batch, time, features]")
    model.freeze_base_for_adaptation()
    # Keep the frozen encoder deterministic during a deployment update.  The
    # adapter still receives gradients in eval mode, while dropout cannot turn
    # the same context into a different update on every pass.
    model.eval()
    target = context[..., : model.target_dim].detach()
    masked, mask = _masked_load_view(model, context.detach(), mask_probability)
    # The generic path is kept for callers that do not have cached features.
    # Deployment uses ``adapt_on_encoded`` below to avoid recomputing the
    # frozen encoder for every update.
    base_sequence, _ = model.encode_base(masked)
    return adapt_on_encoded(
        model,
        base_sequence,
        target,
        mask,
        steps=steps,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        adapter_max_frobenius=adapter_max_frobenius,
    )


def adapt_on_encoded(
    model: StableShiftForecaster,
    base_sequence: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    steps: int = 1,
    learning_rate: float = 0.01,
    max_grad_norm: float = 1.0,
    adapter_max_frobenius: float = 8.0,
) -> tuple[float, float]:
    """Adapt the low-rank branch from a cached frozen-base feature tensor.

    ``base_sequence`` is normally detached after a single forward pass over
    the masked context.  No future labels enter this function; ``target`` is
    the observed context load used solely for masked reconstruction.
    """
    if base_sequence.ndim != 3 or target.ndim != 3 or mask.shape != target.shape:
        raise ValueError("encoded sequence, target and mask have incompatible shapes")
    model.freeze_base_for_adaptation()
    model.eval()
    base_sequence = base_sequence.detach()
    target = target.detach()
    mask = mask.detach()
    before_reconstruction = model.reconstruct_from_encoded(base_sequence).detach()
    trainable = [p for p in model.parameters() if p.requires_grad]
    for _ in range(steps):
        for parameter in trainable:
            parameter.grad = None
        reconstructed = model.reconstruct_from_encoded(base_sequence)
        loss = _masked_mse(reconstructed, target, mask)
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        # A hand-written SGD step avoids constructing an optimizer and its
        # state for every one-window deployment block.
        with torch.no_grad():
            for parameter in trainable:
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-learning_rate)
        # Project each low-rank factor to a bounded set.  This makes the
        # sensitivity statement depend on an explicit, inspectable quantity.
        with torch.no_grad():
            for parameter in trainable:
                norm = torch.linalg.vector_norm(parameter)
                if norm > adapter_max_frobenius:
                    parameter.mul_(adapter_max_frobenius / norm)
    after = model.reconstruct_from_encoded(base_sequence).detach()
    model.eval()
    return float(_masked_mse(before_reconstruction, target, mask)), float(_masked_mse(after, target, mask))


def drift_score(context: torch.Tensor, reference_mean: torch.Tensor, reference_std: torch.Tensor, target_dim: int) -> torch.Tensor:
    current = context[..., :target_dim].mean(dim=1)
    return (((current - reference_mean) / reference_std.clamp_min(1e-5)) ** 2).mean(dim=-1).sqrt()
