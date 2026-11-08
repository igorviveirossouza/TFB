"""Causal momentum features for the optional Non-stationary input adapter.

This is an input, not a fixed term in the model's output. ``zero`` instantiates
exactly the same projection as ``feature`` but supplies zeros as its ablation.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from ts_benchmark.baselines.momentum_score import LOG_RETURN_KINDS, _positive_integer


def validate_momentum_input_config(config):
    def get(key, default=None):
        return config.get(key, default) if isinstance(config, Mapping) else getattr(config, key, default)

    mode = str(get("momentum_input_mode", "off")).strip().lower()
    if mode not in {"off", "zero", "feature"}:
        raise ValueError("momentum_input_mode deve ser off, zero ou feature.")
    if mode == "off":
        return {"mode": "off", "lookback": None}
    if str(get("momentum_mode", "off")).strip().lower() != "off":
        raise ValueError("Momentum na entrada nao pode ser combinado com momentum residual na saida.")
    lookback = get("momentum_input_lookback")
    if lookback is None:
        lookback = get("seq_len")
    lookback = _positive_integer(lookback, "momentum_input_lookback")
    if str(get("loss_data_kind", "log_return")).strip().lower() not in LOG_RETURN_KINDS:
        raise ValueError("Momentum na entrada exige log-retornos (loss_data_kind=log_return).")
    if str(get("target_mode", "path")).strip().lower() == "cumulative":
        raise ValueError("Momentum na entrada exige target_mode=path.")
    return {"mode": mode, "lookback": lookback}


def momentum_input_features(history, lookback, *, mean=None, scale=None):
    """z_assets(sum of the last L raw observed log returns), shape [B,N].

    ``history`` contains observations ending at the origin. When the outer TFB
    scaler is enabled, its training-only mean/scale must both be supplied. No
    target or future data is accepted by this function.
    """
    lookback = _positive_integer(lookback, "momentum_input_lookback")
    if history.ndim != 3 or history.shape[0] < 1 or history.shape[2] < 2:
        raise ValueError("Momentum na entrada espera historico [B,L,N], com N >= 2.")
    if history.shape[1] < lookback:
        raise ValueError("Historico insuficiente para momentum_input_lookback.")
    if not history.is_floating_point():
        raise ValueError("Historico de momentum deve ser floating point.")
    if (mean is None) != (scale is None):
        raise ValueError("Media e escala do normalizador devem ser fornecidas juntas.")
    raw = history[:, -lookback:, :].detach()
    if raw.dtype in {torch.float16, torch.bfloat16}:
        raw = raw.float()
    if mean is not None:
        mean = torch.as_tensor(mean, dtype=raw.dtype, device=raw.device)
        scale = torch.as_tensor(scale, dtype=raw.dtype, device=raw.device)
        if mean.ndim != 1 or scale.shape != mean.shape or mean.numel() != raw.shape[-1]:
            raise ValueError("Normalizador incompativel com o universo de ativos.")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("Normalizador de momentum deve ser finito e ter escala positiva.")
        raw = raw * scale + mean
    if not torch.isfinite(raw).all():
        raise ValueError("Momentum na entrada exige log-retornos finitos.")
    values = raw.sum(dim=1)
    centered = values - values.mean(dim=-1, keepdim=True)
    return centered / values.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-8)


class MomentumInputProjection(nn.Module):
    """Learned asset-vector projection, added after temporal input normalization."""

    def __init__(self, asset_count, d_model, mode):
        super().__init__()
        if mode not in {"zero", "feature"}:
            raise ValueError("A projecao de momentum exige modo zero ou feature.")
        self.mode = mode
        self.projection = nn.Linear(asset_count, d_model, bias=False)

    def forward(self, features):
        if features.ndim != 2 or features.shape[-1] != self.projection.in_features:
            raise ValueError("Feature momentum deve ter shape [B,N] com os mesmos ativos do modelo.")
        if not torch.isfinite(features).all():
            raise ValueError("Feature momentum nao finita.")
        if self.mode == "zero":
            features = torch.zeros_like(features)
        return self.projection(features).unsqueeze(1)
