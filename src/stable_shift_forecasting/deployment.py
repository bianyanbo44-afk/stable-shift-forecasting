"""Streaming deployment with input-only, scenario-independent adaptation."""
import math
import numpy as np
import torch
from .model import StableShiftForecaster
from .training import drift_score, _masked_load_view, adapt_on_encoded


class OnlineSession:
    def __init__(self, model, reference_mean, reference_std, threshold, learning_rate=.01,
                 update=True, anchor_only=False):
        self.model = model.eval()
        self.reference_mean = reference_mean
        self.reference_std = reference_std
        self.threshold = float(threshold)
        self.learning_rate = learning_rate
        self.update = update
        self.anchor_only = anchor_only

    def forecast(self, context, origin, observed=None):
        """Only observed context and origin enter the update; never targets/scenarios."""
        model = self.model
        if len(context) != 1:
            raise ValueError("Online deployment requires one chronological origin")
        score = float(drift_score(context, self.reference_mean, self.reference_std, model.target_dim)[0])
        telemetry = {"drift_score": score, "updated": False,
                     "reconstruction_before": np.nan, "reconstruction_after": np.nan}
        if self.anchor_only:
            with torch.no_grad():
                bounded = model._bounded_context(context)[..., :model.target_dim]
                period = model.seasonal_period
                cycles = bounded[:, -model.anchor_cycles*period:].reshape(1, -1, period, model.target_dim)
                alpha = torch.sigmoid(model.cycle_mix_logit)
                base = (1-alpha)*bounded[:, -period:] + alpha*cycles.mean(1)
                return base.repeat(1, math.ceil(model.horizon/period), 1)[:, :model.horizon], telemetry
        if self.update and score > self.threshold:
            model.freeze_base_for_adaptation()
            with torch.no_grad():
                masked, mask = _masked_load_view(model, context, seed=2026+int(origin), observed=observed)
                sequence, _ = model.encode_base(masked)
            before, after = adapt_on_encoded(model, sequence, context[..., :model.target_dim], mask,
                                             steps=1, learning_rate=self.learning_rate)
            telemetry.update(updated=True, reconstruction_before=before, reconstruction_after=after)
        with torch.no_grad():
            return model(context), telemetry
