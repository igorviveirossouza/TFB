"""MASTER with independent origin batches and a direct multi-horizon head.

Architecture: Li et al., AAAI 2024, arXiv:2312.15235.
Reference core: https://github.com/SJTU-DMTai/MASTER/blob/master/master.py
See LICENSE.upstream for the reference implementation's MIT notice.

This implementation preserves the reference attention scales, normalization and
pooling order. Extensions are an explicit origin-batch axis, a D -> H output
head, separate missingness projections, and optional market-gate ablation. The
network predicts daily economic log returns; it does not normalize labels or
discard extreme labels as in the original experimental training pipeline.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _positive_integer(name: str, value: int, *, allow_zero: bool = False) -> None:
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'} integer")


class _AttentionBlock(nn.Module):
    """Attention over axis -2; every leading axis is an independent group."""

    def __init__(self, width: int, heads: int, dropout: float, *, scaled: bool):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.scale = math.sqrt(self.head_width) if scaled else 1.0
        self.input_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width, width), nn.Dropout(dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        normalized = self.input_norm(values)
        head_shape = (*values.shape[:-1], self.heads, self.head_width)
        query = self.query(normalized).reshape(head_shape).transpose(-3, -2)
        key = self.key(normalized).reshape(head_shape).transpose(-3, -2)
        value = self.value(normalized).reshape(head_shape).transpose(-3, -2)
        weights = torch.softmax((query @ key.transpose(-2, -1)) / self.scale, dim=-1)
        attended = self.attention_dropout(weights) @ value
        attended = attended.transpose(-3, -2).reshape(values.shape)
        # These residuals use the normalized tensors, matching the official core.
        fused = self.ffn_norm(normalized + attended)
        return fused + self.ffn(fused)


class MasterNetwork(nn.Module):
    """Map [batch, lookback, assets, features] to [batch, horizon, assets].

    The market tensor is [batch, lookback, market_features]. Only its final row
    selects feature weights, shared across all stocks and historical steps in
    that origin. Temporal attention is bidirectional *within observed history*;
    no future feature or target tensor is accepted by this module.

    Inputs normally come from MasterStandardizer. False observed-mask entries
    are zeroed again defensively; missingness enters through separate learned
    projections. Audit flags are deliberately not model inputs. With zero
    market features the gate is exactly one and has no trainable parameters.
    """

    def __init__(
        self, n_asset_features: int, n_market_features: int,
        lookback: int, horizon: int, d_model: int = 64,
        t_heads: int = 4, s_heads: int = 2,
        dropout: float = 0.1, gate_beta: float = 1.0,
    ):
        super().__init__()
        for name, value in (
            ("n_asset_features", n_asset_features), ("lookback", lookback),
            ("horizon", horizon), ("d_model", d_model),
            ("t_heads", t_heads), ("s_heads", s_heads),
        ):
            _positive_integer(name, value)
        _positive_integer("n_market_features", n_market_features, allow_zero=True)
        if d_model % t_heads or d_model % s_heads:
            raise ValueError("d_model must be divisible by t_heads and s_heads")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        if not math.isfinite(gate_beta) or gate_beta <= 0:
            raise ValueError("gate_beta must be finite and positive")
        self.n_asset_features = n_asset_features
        self.n_market_features = n_market_features
        self.lookback = lookback
        self.horizon = horizon
        self.gate_beta = float(gate_beta)
        if n_market_features:
            self.market_gate = nn.Linear(n_market_features, n_asset_features)
            self.market_missing = nn.Linear(n_market_features, n_asset_features, bias=False)
            nn.init.zeros_(self.market_missing.weight)
        else:
            self.market_gate = None
            self.market_missing = None
        self.feature_projection = nn.Linear(n_asset_features, d_model)
        self.asset_missing = nn.Linear(n_asset_features, d_model, bias=False)
        nn.init.zeros_(self.asset_missing.weight)

        positions = torch.arange(lookback, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                                * (-math.log(10000.0) / d_model))
        angles = positions * frequencies
        encoding = torch.empty(lookback, d_model)
        encoding[:, 0::2] = torch.sin(angles)
        encoding[:, 1::2] = torch.cos(angles[:, :d_model // 2])
        self.register_buffer("position_encoding", encoding)
        self.temporal = _AttentionBlock(d_model, t_heads, dropout, scaled=False)
        self.cross_asset = _AttentionBlock(d_model, s_heads, dropout, scaled=True)
        self.pool_projection = nn.Linear(d_model, d_model, bias=False)
        self.prediction = nn.Linear(d_model, horizon)

    @staticmethod
    def _observed(values: Tensor, observed: Tensor | None, name: str) -> Tensor:
        if observed is None:
            return torch.ones_like(values, dtype=torch.bool)
        if observed.shape != values.shape or observed.dtype != torch.bool:
            raise ValueError(f"{name} must be a boolean tensor with shape {tuple(values.shape)}")
        if observed.device != values.device:
            raise ValueError(f"{name} must be on the same device as its values")
        return observed

    def feature_weights(self, market_values: Tensor, market_observed: Tensor | None = None) -> Tensor:
        """Return interpretable [batch, features] weights (sum = feature count)."""
        if (market_values.ndim != 3 or market_values.shape[1] != self.lookback
                or market_values.shape[2] != self.n_market_features):
            raise ValueError("market_values must have shape [batch, lookback, n_market_features]")
        mask = self._observed(market_values, market_observed, "market_observed")
        if self.market_gate is None:
            return market_values.new_ones((market_values.shape[0], self.n_asset_features))
        current = torch.where(mask[:, -1], market_values[:, -1], 0.0)
        missing = (~mask[:, -1]).to(market_values.dtype)
        logits = self.market_gate(current) + self.market_missing(missing)
        return self.n_asset_features * torch.softmax(logits / self.gate_beta, dim=-1)

    def forward(
        self, asset_values: Tensor, market_values: Tensor,
        asset_observed: Tensor | None = None, market_observed: Tensor | None = None,
    ) -> Tensor:
        if (asset_values.ndim != 4 or asset_values.shape[1] != self.lookback
                or asset_values.shape[-1] != self.n_asset_features):
            raise ValueError("asset_values must have shape [batch, lookback, assets, n_asset_features]")
        if asset_values.shape[0] < 1 or asset_values.shape[2] < 1:
            raise ValueError("batch and asset dimensions must be nonempty")
        if not asset_values.is_floating_point() or not market_values.is_floating_point():
            raise ValueError("asset_values and market_values must be floating point")
        if market_values.ndim != 3 or market_values.shape[0] != asset_values.shape[0]:
            raise ValueError("market_values must share the asset batch dimension")
        if market_values.device != asset_values.device or market_values.dtype != asset_values.dtype:
            raise ValueError("asset and market values must share dtype and device")
        observed = self._observed(asset_values, asset_observed, "asset_observed")
        values = torch.where(observed, asset_values, 0.0)
        gate = self.feature_weights(market_values, market_observed)[:, None, None, :]
        embedding = self.feature_projection(values * gate)
        embedding = embedding + self.asset_missing((~observed).to(values.dtype))
        embedding = embedding + self.position_encoding[None, :, None, :]
        # [B,N,L,D]: temporal attention cannot mix stocks or origin batches.
        local = self.temporal(embedding.transpose(1, 2))
        # [B,L,N,D]: cross-asset attention cannot mix dates or origin batches.
        correlated = self.cross_asset(local.transpose(1, 2)).transpose(1, 2)
        # Match official pooling: transformed last embedding queries transformed
        # historical embeddings; the weighted values remain untransformed.
        projected = self.pool_projection(correlated)
        scores = (projected * projected[:, :, -1:, :]).sum(dim=-1)
        weights = torch.softmax(scores, dim=-1)
        pooled = (correlated * weights.unsqueeze(-1)).sum(dim=-2)
        return self.prediction(pooled).transpose(1, 2).contiguous()
