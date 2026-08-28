"""Composite trading loss v3: financial aggregation, then cross-sectional standardization.

The model still predicts the full path [B,H,N]. The temporal component is
computed on the complete model-space path. The cross-sectional component first
aggregates financially meaningful block scores of length K and only then
standardizes each block across the N assets.

For each batch element b and trading block j:
    Z[b,j,i] = (S[b,j,i] - mean_i S[b,j,i]) / std_i S[b,j,i]

The same blockwise z-score transformation is used for MSE and ranking losses.
This preserves the financial definition of S while removing arbitrary score
level/scale before the cross-sectional objective.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ts_benchmark.baselines.composite_trading_loss_v2 import (
    COMPOSITE_LOSSES,
    CompositeTradingLossV2,
    _as_bool,
    _cfg,
    build_loss_v1,
    loss_accepts_base_value_v1,
)


def build_loss(config, normalizer_mean=None, normalizer_scale=None):
    loss_name = str(_cfg(config, "loss", "mse")).lower()
    if loss_name not in COMPOSITE_LOSSES:
        return build_loss_v1(
            config,
            normalizer_mean=normalizer_mean,
            normalizer_scale=normalizer_scale,
        )

    trade_window = _cfg(config, "loss_trade_window", None)
    if trade_window is None:
        trade_window = _cfg(config, "loss_k", None)
    if trade_window is None:
        raise ValueError("composite_trading exige loss_trade_window (K) explicitamente.")

    inverse_norm = _as_bool(_cfg(config, "loss_inverse_norm", True)) and _as_bool(
        _cfg(config, "norm", False)
    )

    # Backward-compatible fallback: old experiments used loss_rank_score_normalization.
    cross_score_normalization = str(
        _cfg(
            config,
            "loss_cross_score_normalization",
            _cfg(config, "loss_rank_score_normalization", "zscore"),
        )
    )

    return CompositeTradingLossV3(
        trade_window=int(trade_window),
        data_kind=str(_cfg(config, "loss_data_kind", "log_return")),
        score_kind=str(_cfg(config, "loss_score_kind", "simple_return")),
        temporal_loss=str(_cfg(config, "loss_temporal", "mse")),
        cross_loss=str(_cfg(config, "loss_cross", "mse")),
        cross_lambda=float(_cfg(config, "loss_cross_lambda", 0.5)),
        cross_scale=float(_cfg(config, "loss_cross_scale", 1.0)),
        huber_delta=float(_cfg(config, "loss_huber_delta", 0.5)),
        ranknet_alpha=float(_cfg(config, "loss_ranknet_alpha", 1.0)),
        listnet_tau=float(_cfg(config, "loss_listnet_tau", 1.0)),
        hinge_margin=float(_cfg(config, "loss_hinge_margin", 0.01)),
        cross_score_normalization=cross_score_normalization,
        inverse_norm=inverse_norm,
        normalizer_mean=normalizer_mean,
        normalizer_scale=normalizer_scale,
        track_components=_as_bool(_cfg(config, "loss_track_components", False)),
    )


def loss_accepts_base_value(criterion) -> bool:
    if isinstance(criterion, CompositeTradingLossV3):
        return criterion.needs_base_value
    return loss_accepts_base_value_v1(criterion)


class CompositeTradingLossV3(CompositeTradingLossV2):
    """V2 objective with one normalization rule for every cross loss.

    Financial scores S are always built first. Then each [N]-asset score vector
    is optionally standardized independently inside its batch/trading block.
    """

    def __init__(
        self,
        *,
        trade_window: int,
        data_kind: str,
        score_kind: str = "simple_return",
        temporal_loss: str = "mse",
        cross_loss: str = "mse",
        cross_lambda: float = 0.5,
        cross_scale: float = 1.0,
        huber_delta: float = 0.5,
        ranknet_alpha: float = 1.0,
        listnet_tau: float = 1.0,
        hinge_margin: float = 0.01,
        cross_score_normalization: str = "zscore",
        inverse_norm: bool = True,
        normalizer_mean=None,
        normalizer_scale=None,
        track_components: bool = False,
        eps: float = 1e-8,
    ) -> None:
        normalization = cross_score_normalization.lower()
        if normalization not in {"none", "zscore"}:
            raise ValueError(
                "loss_cross_score_normalization deve ser 'none' ou 'zscore'."
            )

        # V2 already implements the correct return/price block aggregation.
        # Its rank_score_normalization attribute is reused internally only as a
        # validated storage slot; V3 applies it to every cross objective.
        super().__init__(
            trade_window=trade_window,
            data_kind=data_kind,
            score_kind=score_kind,
            temporal_loss=temporal_loss,
            cross_loss=cross_loss,
            cross_lambda=cross_lambda,
            cross_scale=cross_scale,
            huber_delta=huber_delta,
            ranknet_alpha=ranknet_alpha,
            listnet_tau=listnet_tau,
            hinge_margin=hinge_margin,
            rank_score_normalization=normalization,
            inverse_norm=inverse_norm,
            normalizer_mean=normalizer_mean,
            normalizer_scale=normalizer_scale,
            track_components=track_components,
            eps=eps,
        )
        self.cross_score_normalization = normalization

    def _normalize_cross_scores(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cross_score_normalization == "none":
            return pred, target

        def zscore(x: torch.Tensor) -> torch.Tensor:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
            return (x - mean) / std

        return zscore(pred), zscore(target)

    def _cross_loss(
        self,
        pred_scores: torch.Tensor,
        target_scores: torch.Tensor,
    ) -> torch.Tensor:
        # [B,M,N] -> [B*M,N]. Each row remains one independent cross-section.
        pred, target = self._flatten_blocks(pred_scores, target_scores)
        pred, target = self._normalize_cross_scores(pred, target)

        if self.cross_loss_name == "mse":
            return F.mse_loss(pred, target)

        if self.cross_loss_name == "listnet":
            p_true = F.softmax(target / self.listnet_tau, dim=1)
            log_p_pred = F.log_softmax(pred / self.listnet_tau, dim=1)
            return -(p_true * log_p_pred).sum(dim=1).mean()

        pred_diff, target_diff, valid = self._pairwise_upper(pred, target)
        sign = torch.sign(target_diff)

        if self.cross_loss_name == "ranknet":
            values = F.softplus(-self.ranknet_alpha * sign * pred_diff)
            return self._masked_mean(values, valid)

        if self.cross_loss_name == "hinge":
            values = F.relu(self.hinge_margin - sign * pred_diff)
            return self._masked_mean(values, valid)

        # BPR: pairwise logistic with realized ordering as orientation.
        return self._masked_mean(F.softplus(-(sign * pred_diff)), valid)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        base_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Kept explicit for auditability: both losses feed the same backward pass.
        if pred.shape != target.shape:
            raise ValueError(f"Shapes diferentes: pred={pred.shape}; target={target.shape}.")

        temporal = self._temporal_loss(pred, target)
        pred_scores, target_scores = self.aggregator(
            pred,
            target,
            base_value=base_value,
        )
        cross = self._cross_loss(pred_scores, target_scores)
        total = (
            (1.0 - self.cross_lambda) * temporal
            + self.cross_lambda * self.cross_scale * cross
        )
        self._remember(temporal, cross, total, pred_scores)
        return total
