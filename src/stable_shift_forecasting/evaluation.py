"""End-to-end window evaluation with optional online adaptation."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import Standardizer
from .metrics import evaluate_predictions
from .model import StableShiftForecaster
from .scenarios import apply_scenario
from .training import adapt_on_context, adapt_on_encoded, _masked_load_view, drift_score


def _collect_scenario_windows(
    loader: DataLoader,
    target_dim: int,
    scenario: str,
    missing_rate: float,
    anomaly_scale: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialise one deterministic scenario in chronological origin order.

    Scenario randomness is keyed by the forecast origin, so materialisation is
    invariant to DataLoader batch size.  Keeping this boundary explicit also
    makes it straightforward to cache the frozen encoder representation.
    """
    contexts: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    starts: list[torch.Tensor] = []
    for batch in loader:
        original = batch["context"].to(device)
        start_values = batch["start"].to(device)
        transformed = [
            apply_scenario(
                original[index : index + 1],
                target_dim,
                scenario,
                seed=2026 + int(start_values[index]),
                missing_rate=missing_rate,
                anomaly_scale=anomaly_scale,
            ).context[0]
            for index in range(len(original))
        ]
        contexts.append(torch.stack(transformed, dim=0))
        targets.append(batch["target"].to(device))
        starts.append(start_values)
    if not contexts:
        raise ValueError("evaluation loader contains no windows")
    return torch.cat(contexts, dim=0), torch.cat(targets, dim=0), torch.cat(starts, dim=0)


def _encode_in_batches(model: StableShiftForecaster, contexts: torch.Tensor, batch_size: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode all contexts once through the frozen base path."""
    sequences: list[torch.Tensor] = []
    stable: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(contexts), max(1, int(batch_size))):
            sequence, stable_representation = model.encode_base(contexts[start : start + batch_size])
            sequences.append(sequence.detach())
            stable.append(stable_representation.detach())
    return torch.cat(sequences, dim=0), torch.cat(stable, dim=0)


def predict_windows(
    model: StableShiftForecaster,
    loader: DataLoader,
    device: str,
    standardizer: Standardizer,
    scenario: str = "normal",
    adapt: bool = False,
    adapter_steps: int = 1,
    adapter_learning_rate: float = 0.01,
    adaptation_threshold: float | None = None,
    missing_rate: float = 0.10,
    anomaly_scale: float = 4.0,
    reference_mean: np.ndarray | None = None,
    reference_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.to(device)
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    target_dim = model.target_dim
    ref_mean = torch.as_tensor(reference_mean, dtype=torch.float32, device=device) if reference_mean is not None else None
    ref_std = torch.as_tensor(reference_std, dtype=torch.float32, device=device) if reference_std is not None else None
    # Materialise and encode once.  This preserves chronological adapter state
    # while removing the expensive TCN pass from the per-window update loop.
    context, target, starts = _collect_scenario_windows(
        loader, target_dim, scenario, missing_rate, anomaly_scale, device
    )
    base_sequence, stable_representation = _encode_in_batches(model, context)
    if ref_mean is not None and ref_std is not None:
        score = drift_score(context, ref_mean, ref_std, target_dim)
    else:
        score = torch.zeros(len(context), device=device)
    if adapt:
        selected = torch.ones(len(context), dtype=torch.bool, device=device)
        if adaptation_threshold is not None:
            selected = score > adaptation_threshold
        selected_indices = torch.nonzero(selected, as_tuple=False).flatten().tolist()
        masked_sequences: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        if selected_indices:
            masked_contexts = []
            masks = []
            for index in selected_indices:
                masked_context, mask = _masked_load_view(
                    model,
                    context[index : index + 1].detach(),
                    seed=2026 + int(starts[index]),
                )
                masked_contexts.append(masked_context)
                masks.append(mask)
            masked_context_batch = torch.cat(masked_contexts, dim=0)
            masked_sequence_batch, _ = _encode_in_batches(model, masked_context_batch)
            masked_sequences = {
                index: (masked_sequence_batch[position], masks[position][0])
                for position, index in enumerate(selected_indices)
            }
        # The loop is deliberately chronological and one origin at a time:
        # each update sees only that origin's context and can affect later
        # forecasts, matching the deployment contract.
        for index in range(len(context)):
            if bool(selected[index]):
                masked_sequence, mask = masked_sequences[index]
                adapt_on_encoded(
                    model,
                    masked_sequence.unsqueeze(0),
                    context[index : index + 1, ..., :target_dim].detach(),
                    mask.unsqueeze(0),
                    steps=min(1, int(adapter_steps)),
                    learning_rate=adapter_learning_rate,
                )
            with torch.no_grad():
                prediction = model.forecast_from_encoded(
                    context[index : index + 1],
                    base_sequence[index : index + 1],
                    stable_representation[index : index + 1],
                )
            predictions.append(standardizer.inverse(prediction).detach().cpu().numpy())
    else:
        with torch.no_grad():
            for start in range(0, len(context), 256):
                prediction = model.forecast_from_encoded(
                    context[start : start + 256],
                    base_sequence[start : start + 256],
                    stable_representation[start : start + 256],
                )
                predictions.append(standardizer.inverse(prediction).detach().cpu().numpy())
    return np.concatenate(predictions), standardizer.inverse(target).detach().cpu().numpy(), score.detach().cpu().numpy()


def evaluate_loader(model: StableShiftForecaster, loader: DataLoader, device: str, standardizer: Standardizer, training_history: np.ndarray, scenario: str = "normal", adapt: bool = False, **kwargs: object) -> dict[str, object]:
    prediction, target, scores = predict_windows(model, loader, device, standardizer, scenario=scenario, adapt=adapt, **kwargs)
    report = evaluate_predictions(prediction, target, training_history=training_history)
    result = asdict(report)
    result["prediction"] = prediction
    result["target"] = target
    result["drift_score"] = scores
    return result
