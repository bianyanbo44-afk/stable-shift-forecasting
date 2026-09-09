"""Adapters for the MIT-licensed Time-Series-Library reference models.

The adapters retain [B,T,D] inputs and supply six known calendar covariates
to architectures that consume them. TiDE also receives future calendar marks.
Its scalar-output LayerNorm is disabled because normalizing one scalar removes
all input dependence from the nonlinear decoder. This local correction is
explicitly recorded with the pinned upstream version.
"""
from __future__ import annotations
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn

VENDOR = Path(__file__).resolve().parents[2] / "vendor_cache" / "Time-Series-Library"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
from models.DLinear import Model as _DLinear
from models.PatchTST import Model as _PatchTST
from models.TimesNet import Model as _TimesNet
from models.iTransformer import Model as _ITransformer
from models.TiDE import Model as _TiDE

def _cfg(d: int, h: int, length=336, width=64) -> SimpleNamespace:
    return SimpleNamespace(task_name="long_term_forecast", seq_len=length, pred_len=h,
        enc_in=d, dec_in=d, c_out=d, d_model=width, n_heads=4, d_ff=width*2,
        e_layers=2, d_layers=1, dropout=.1, factor=3, activation="gelu",
        embed="timeF", freq="s", label_len=48, top_k=5, num_kernels=6,
        moving_avg=25, class_strategy="projection", individual=False, feature_dim=4)

class UpstreamWrapper(nn.Module):
    def __init__(self, name: str, target_dim: int, horizon: int, length=336, width=64):
        super().__init__(); self.target_dim = target_dim; self.horizon = horizon
        self.name = name
        cfg = _cfg(target_dim, horizon, length, width)
        # TSL uses freq solely to select feature dimension (s -> six).
        # Data and forecast horizons remain hourly. Six sine/cosine calendar
        # features are supplied explicitly, never future load values.
        if name == "TiDE":
            cfg.c_out = 16  # TiDE temporal decoder width, not number of load channels
        factory = {"DLinear": _DLinear, "PatchTST": _PatchTST, "TimesNet": _TimesNet,
                   "iTransformer": _ITransformer, "TiDE": _TiDE}[name]
        self.model = factory(cfg)
        if name == "TiDE":
            self.model.temporalDecoder.ln = nn.Identity()
    def forward(self, context: torch.Tensor, future_calendar=None) -> torch.Tensor:
        x = context[..., :self.target_dim].clone()
        marks = context[..., self.target_dim:]
        if marks.shape[-1] != 6:
            raise ValueError("Expected six calendar covariates")
        if self.name == "TiDE" and future_calendar is None:
            raise ValueError("TiDE requires known future calendar covariates")
        y = self.model(x, marks, None, future_calendar)
        return y[..., :self.target_dim]

def build_upstream(name: str, target_dim: int, horizon: int, length=336, width=64) -> UpstreamWrapper:
    return UpstreamWrapper(name, target_dim, horizon, length, width)
