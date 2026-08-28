"""Composite temporal + trading-window loss, with return and price support.

The model emits a full path [B,H,N]. The objective combines a temporal loss on
all H steps with a cross-sectional loss on non-overlapping trading blocks of
length K. H must be exactly divisible by K.

For return datasets, each block score is the accumulated return inside that
block. For price datasets, each block score is the return between the block
start and block end. The first price block starts from the last observed price;
later blocks start from the previous predicted/realized block endpoint.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ts_benchmark.baselines.custom_losses import (
    build_loss as build_loss_v1,
    loss_accepts_base_value as loss_accepts_base_value_v1,
)

COMPOSITE_LOSSES = {"composite_trading"}
TEMPORAL_LOSSES = {"mse", "mae", "huber"}
CROSS_LOSSES = {"mse", "ranknet", "listnet", "bpr", "hinge"}


def _cfg(config, name: str, default=None):
    return getattr(config, name, default)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"true", "1", "yes", "y", "sim", "s"}:
            return True
        if value in {"false", "0", "no", "n", "nao", "não"}:
            return False
    return bool(value)


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

    return CompositeTradingLossV2(
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
        rank_score_normalization=str(
            _cfg(config, "loss_rank_score_normalization", "zscore")
        ),
        inverse_norm=inverse_norm,
        normalizer_mean=normalizer_mean,
        normalizer_scale=normalizer_scale,
        track_components=_as_bool(_cfg(config, "loss_track_components", False)),
    )


def loss_accepts_base_value(criterion) -> bool:
    if isinstance(criterion, CompositeTradingLossV2):
        return criterion.needs_base_value
    return loss_accepts_base_value_v1(criterion)


class TradingBlockScoreAggregatorV2(nn.Module):
    """Map [B,H,N] paths to block scores [B,M,N]."""

    LOG_RETURN_KINDS = {"log_return", "log_returns", "log_retornos"}
    SIMPLE_RETURN_KINDS = {
        "simple_return",
        "simple_returns",
        "return",
        "returns",
        "retornos",
        "retornos_simples",
    }
    PRICE_KINDS = {"price", "prices", "preco", "precos", "preços"}

    def __init__(
        self,
        *,
        trade_window: int,
        data_kind: str,
        score_kind: str,
        inverse_norm: bool,
        normalizer_mean=None,
        normalizer_scale=None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.trade_window = int(trade_window)
        self.data_kind = data_kind.lower()
        self.score_kind = score_kind.lower()
        self.inverse_norm = bool(inverse_norm)
        self.eps = float(eps)

        valid_data = self.LOG_RETURN_KINDS | self.SIMPLE_RETURN_KINDS | self.PRICE_KINDS
        valid_score = self.LOG_RETURN_KINDS | self.SIMPLE_RETURN_KINDS

        if self.trade_window <= 0:
            raise ValueError("loss_trade_window (K) deve ser positivo.")
        if self.data_kind not in valid_data:
            raise ValueError(f"loss_data_kind inválido: {self.data_kind}")
        if self.score_kind not in valid_score:
            raise ValueError("loss_score_kind deve ser log_return ou simple_return.")

        if normalizer_mean is not None:
            self.register_buffer(
                "normalizer_mean",
                torch.as_tensor(normalizer_mean, dtype=torch.float32).view(1, 1, -1),
            )
        else:
            self.normalizer_mean = None

        if normalizer_scale is not None:
            self.register_buffer(
                "normalizer_scale",
                torch.as_tensor(normalizer_scale, dtype=torch.float32).view(1, 1, -1),
            )
        else:
            self.normalizer_scale = None

    @property
    def needs_base_value(self) -> bool:
        return self.data_kind in self.PRICE_KINDS

    def _validate(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[int, int, int]:
        if pred.ndim != 3 or target.ndim != 3:
            raise ValueError("pred e target devem ter formato [B,H,N].")
        if pred.shape != target.shape:
            raise ValueError(f"Shapes diferentes: pred={pred.shape}; target={target.shape}.")
        batch, horizon, n_assets = pred.shape
        if self.trade_window > horizon:
            raise ValueError(f"K={self.trade_window} não pode ser maior que H={horizon}.")
        if horizon % self.trade_window != 0:
            raise ValueError(
                "composite_trading exige K divisor exato de H: "
                f"H={horizon}, K={self.trade_window}, resto={horizon % self.trade_window}."
            )
        return batch, horizon, n_assets

    def _maybe_inverse_norm(self, x: torch.Tensor) -> torch.Tensor:
        if not self.inverse_norm:
            return x
        if self.normalizer_mean is None or self.normalizer_scale is None:
            raise ValueError(
                "loss_inverse_norm=True, mas mean/scale do normalizador não estão disponíveis."
            )
        n_assets = x.shape[-1]
        mean = self.normalizer_mean[..., :n_assets].to(device=x.device, dtype=x.dtype)
        scale = self.normalizer_scale[..., :n_assets].to(device=x.device, dtype=x.dtype)
        if x.ndim == 2:
            return x * scale.squeeze(1) + mean.squeeze(1)
        return x * scale + mean

    def _aggregate_returns(self, x: torch.Tensor) -> torch.Tensor:
        batch, horizon, n_assets = x.shape
        n_blocks = horizon // self.trade_window
        blocks = x.reshape(batch, n_blocks, self.trade_window, n_assets)

        if self.data_kind in self.LOG_RETURN_KINDS:
            block_log = blocks.sum(dim=2)
            if self.score_kind in self.LOG_RETURN_KINDS:
                return block_log
            return torch.expm1(block_log)

        safe = torch.clamp(blocks, min=-1.0 + self.eps)
        if self.score_kind in self.LOG_RETURN_KINDS:
            return torch.log1p(safe).sum(dim=2)
        return torch.prod(1.0 + safe, dim=2) - 1.0

    def _aggregate_prices(
        self,
        x: torch.Tensor,
        base_value: torch.Tensor,
    ) -> torch.Tensor:
        _, horizon, _ = x.shape
        end_idx = torch.arange(
            self.trade_window - 1,
            horizon,
            self.trade_window,
            device=x.device,
        )
        ends = x.index_select(dim=1, index=end_idx)
        starts = torch.cat([base_value.unsqueeze(1), ends[:, :-1, :]], dim=1)

        starts = torch.clamp(starts, min=self.eps)
        ends = torch.clamp(ends, min=self.eps)
        ratio = ends / starts
        if self.score_kind in self.LOG_RETURN_KINDS:
            return torch.log(ratio)
        return ratio - 1.0

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        base_value: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate(pred, target)
        pred_raw = self._maybe_inverse_norm(pred)
        target_raw = self._maybe_inverse_norm(target)

        if self.data_kind in self.PRICE_KINDS:
            if base_value is None:
                raise ValueError("Dataset de preços exige base_value.")
            base_raw = self._maybe_inverse_norm(base_value)
            return (
                self._aggregate_prices(pred_raw, base_raw),
                self._aggregate_prices(target_raw, base_raw),
            )

        return self._aggregate_returns(pred_raw), self._aggregate_returns(target_raw)


class CompositeTradingLossV2(nn.Module):
    """Joint path loss plus blockwise cross-sectional trading loss."""

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
        rank_score_normalization: str = "zscore",
        inverse_norm: bool = True,
        normalizer_mean=None,
        normalizer_scale=None,
        track_components: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.temporal_loss_name = temporal_loss.lower()
        self.cross_loss_name = cross_loss.lower()
        self.cross_lambda = float(cross_lambda)
        self.cross_scale = float(cross_scale)
        self.huber_delta = float(huber_delta)
        self.ranknet_alpha = float(ranknet_alpha)
        self.listnet_tau = float(listnet_tau)
        self.hinge_margin = float(hinge_margin)
        self.rank_score_normalization = rank_score_normalization.lower()
        self.track_components = bool(track_components)
        self.eps = float(eps)
        self.last_components: Dict[str, float] = {}

        if self.temporal_loss_name not in TEMPORAL_LOSSES:
            raise ValueError(f"loss_temporal inválida: {self.temporal_loss_name}")
        if self.cross_loss_name not in CROSS_LOSSES:
            raise ValueError(f"loss_cross inválida: {self.cross_loss_name}")
        if not 0.0 <= self.cross_lambda <= 1.0:
            raise ValueError("loss_cross_lambda deve estar em [0,1].")
        if self.cross_scale <= 0:
            raise ValueError("loss_cross_scale deve ser positivo.")
        if self.huber_delta <= 0:
            raise ValueError("loss_huber_delta deve ser positivo.")
        if self.ranknet_alpha <= 0:
            raise ValueError("loss_ranknet_alpha deve ser positivo.")
        if self.listnet_tau <= 0:
            raise ValueError("loss_listnet_tau deve ser positivo.")
        if self.hinge_margin < 0:
            raise ValueError("loss_hinge_margin deve ser não-negativo.")
        if self.rank_score_normalization not in {"none", "zscore"}:
            raise ValueError("loss_rank_score_normalization deve ser none ou zscore.")

        self.aggregator = TradingBlockScoreAggregatorV2(
            trade_window=trade_window,
            data_kind=data_kind,
            score_kind=score_kind,
            inverse_norm=inverse_norm,
            normalizer_mean=normalizer_mean,
            normalizer_scale=normalizer_scale,
            eps=eps,
        )

    @property
    def trade_window(self) -> int:
        return self.aggregator.trade_window

    @property
    def needs_base_value(self) -> bool:
        return self.aggregator.needs_base_value

    def _temporal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.temporal_loss_name == "mse":
            return F.mse_loss(pred, target)
        if self.temporal_loss_name == "mae":
            return F.l1_loss(pred, target)
        return F.huber_loss(pred, target, delta=self.huber_delta)

    def _normalize_rank_scores(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rank_score_normalization == "none":
            return pred, target

        def zscore(x: torch.Tensor) -> torch.Tensor:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
            return (x - mean) / std

        return zscore(pred), zscore(target)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        selected = values[mask]
        if selected.numel() == 0:
            return values.sum() * 0.0
        return selected.mean()

    @staticmethod
    def _flatten_blocks(
        pred_scores: torch.Tensor, target_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_assets = pred_scores.shape[-1]
        return pred_scores.reshape(-1, n_assets), target_scores.reshape(-1, n_assets)

    @staticmethod
    def _pairwise_upper(
        pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_diff = pred.unsqueeze(2) - pred.unsqueeze(1)
        target_diff = target.unsqueeze(2) - target.unsqueeze(1)
        n_assets = pred.shape[1]
        upper = torch.triu(
            torch.ones((n_assets, n_assets), device=pred.device, dtype=torch.bool),
            diagonal=1,
        ).unsqueeze(0)
        valid = upper & (target_diff != 0)
        return pred_diff, target_diff, valid

    def _cross_loss(self, pred_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        if self.cross_loss_name == "mse":
            return F.mse_loss(pred_scores, target_scores)

        pred, target = self._flatten_blocks(pred_scores, target_scores)
        pred, target = self._normalize_rank_scores(pred, target)

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

    def _remember(
        self,
        temporal: torch.Tensor,
        cross: torch.Tensor,
        total: torch.Tensor,
        pred_scores: torch.Tensor,
    ) -> None:
        if not self.track_components:
            return
        self.last_components = {
            "temporal": float(temporal.detach().cpu()),
            "cross": float(cross.detach().cpu()),
            "cross_scaled": float((self.cross_scale * cross).detach().cpu()),
            "total": float(total.detach().cpu()),
            "H": int(pred_scores.shape[1] * self.trade_window),
            "K": int(self.trade_window),
            "n_blocks": int(pred_scores.shape[1]),
            "n_assets": int(pred_scores.shape[2]),
        }

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        base_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
