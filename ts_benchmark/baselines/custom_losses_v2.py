"""Versão 2, opt-in, das losses financeiras cross-section do TFB.

Este módulo não altera ``custom_losses.py``. Ele é ativado apenas pelo launcher
``scripts/run_benchmark_custom_losses_v2.py`` e delega para a implementação v1
quando a loss solicitada não pertence ao conjunto v2.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ts_benchmark.baselines.custom_losses import (
    build_loss as build_loss_v1,
    loss_accepts_base_value as loss_accepts_base_value_v1,
)


V2_LOSSES = {
    "mse_path_v2",
    "mse_score_v2",
    "ranknet_v2",
    "ranknet_hybrid_v2",
    "listnet_v2",
    "fingat_v2",
}


def _cfg(config, name, default):
    return getattr(config, name, default)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "sim", "s"}:
            return True
        if normalized in {"false", "0", "no", "n", "nao", "não"}:
            return False
    return bool(value)


def build_loss(config, normalizer_mean=None, normalizer_scale=None):
    """Constrói uma loss v2 ou delega integralmente para a implementação v1."""
    loss_name = str(_cfg(config, "loss", "mse")).lower()

    if loss_name not in V2_LOSSES:
        return build_loss_v1(
            config,
            normalizer_mean=normalizer_mean,
            normalizer_scale=normalizer_scale,
        )

    if loss_name == "mse_path_v2":
        return nn.MSELoss()

    return FinancialObjectiveV2(
        loss_name=loss_name,
        data_kind=str(_cfg(config, "loss_data_kind", "log_return")),
        score_kind=str(_cfg(config, "loss_score_kind", "log_return")),
        loss_k=int(_cfg(config, "loss_k", _cfg(config, "horizon", 1))),
        horizon_mode=str(_cfg(config, "loss_horizon_mode", "strict")),
        inverse_norm=_as_bool(_cfg(config, "loss_inverse_norm", True)),
        normalizer_mean=normalizer_mean,
        normalizer_scale=normalizer_scale,
        rank_lambda=float(_cfg(config, "loss_rank_lambda", 0.5)),
        ranknet_alpha=float(_cfg(config, "loss_ranknet_alpha", 1.0)),
        listnet_tau=float(_cfg(config, "loss_listnet_tau", 1.0)),
        score_normalization=str(_cfg(config, "loss_score_normalization", "zscore")),
        hybrid_point_normalization=str(
            _cfg(config, "loss_hybrid_point_normalization", "target_std")
        ),
        fingat_delta=float(_cfg(config, "loss_fingat_delta", 0.2)),
        direction_scale=float(_cfg(config, "loss_direction_scale", 0.01)),
        track_components=_as_bool(_cfg(config, "loss_track_components", False)),
    )


def loss_accepts_base_value(criterion) -> bool:
    if isinstance(criterion, FinancialObjectiveV2):
        return criterion.needs_base_value
    return loss_accepts_base_value_v1(criterion)


class FinancialScoreAggregatorV2(nn.Module):
    """Agrega trajetórias ``[B, H, N]`` em scores financeiros ``[B, N]``."""

    def __init__(
        self,
        *,
        data_kind: str,
        score_kind: str,
        loss_k: int,
        horizon_mode: str,
        inverse_norm: bool,
        normalizer_mean=None,
        normalizer_scale=None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.data_kind = data_kind.lower()
        self.score_kind = score_kind.lower()
        self.loss_k = int(loss_k)
        self.horizon_mode = horizon_mode.lower()
        self.inverse_norm = inverse_norm
        self.eps = eps

        if self.loss_k <= 0:
            raise ValueError("loss_k deve ser positivo.")
        if self.horizon_mode not in {"strict", "prefix"}:
            raise ValueError("loss_horizon_mode deve ser 'strict' ou 'prefix'.")

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
        return self.data_kind in {"price", "prices", "preco", "preços"}

    def _maybe_inverse_norm(self, x: torch.Tensor) -> torch.Tensor:
        if not self.inverse_norm:
            return x
        if self.normalizer_mean is None or self.normalizer_scale is None:
            return x

        n_series = x.shape[-1]
        mean = self.normalizer_mean[..., :n_series].to(device=x.device, dtype=x.dtype)
        scale = self.normalizer_scale[..., :n_series].to(
            device=x.device, dtype=x.dtype
        )
        if x.dim() == 2:
            return x * scale.squeeze(1) + mean.squeeze(1)
        return x * scale + mean

    def _select_horizon(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if pred.ndim != 3 or target.ndim != 3:
            raise ValueError("pred e target devem ter formato [B, H, N].")
        if pred.shape != target.shape:
            raise ValueError(
                f"pred e target devem ter o mesmo shape: {pred.shape} != {target.shape}."
            )

        horizon = pred.shape[1]
        if self.horizon_mode == "strict":
            if horizon != self.loss_k:
                raise ValueError(
                    "Inconsistência H/K: loss_horizon_mode='strict' exige "
                    f"H == loss_k, mas H={horizon} e loss_k={self.loss_k}."
                )
            return pred, target

        if horizon < self.loss_k:
            raise ValueError(
                f"H={horizon} é menor que loss_k={self.loss_k}; não é possível usar prefix."
            )
        return pred[:, : self.loss_k, :], target[:, : self.loss_k, :]

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        base_value: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred, target = self._select_horizon(pred, target)
        pred = self._maybe_inverse_norm(pred)
        target = self._maybe_inverse_norm(target)

        if self.data_kind in {"log_return", "log_returns", "log_retornos"}:
            pred_score = pred.sum(dim=1)
            target_score = target.sum(dim=1)
            if self.score_kind in {
                "simple_return",
                "simple_returns",
                "returns",
                "retornos",
            }:
                pred_score = torch.expm1(pred_score)
                target_score = torch.expm1(target_score)
            return pred_score, target_score

        if self.data_kind in {
            "simple_return",
            "simple_returns",
            "return",
            "returns",
            "retornos",
            "retornos_simples",
        }:
            pred_safe = torch.clamp(pred, min=-1.0 + self.eps)
            target_safe = torch.clamp(target, min=-1.0 + self.eps)
            if self.score_kind in {"log_return", "log_returns", "log_retornos"}:
                return (
                    torch.log1p(pred_safe).sum(dim=1),
                    torch.log1p(target_safe).sum(dim=1),
                )
            return (
                torch.prod(1.0 + pred_safe, dim=1) - 1.0,
                torch.prod(1.0 + target_safe, dim=1) - 1.0,
            )

        if self.data_kind in {"price", "prices", "preco", "preços"}:
            if base_value is None:
                raise ValueError(
                    "loss_data_kind='price' exige base_value: último preço observado "
                    "antes da janela futura."
                )
            base = self._maybe_inverse_norm(base_value)
            base = torch.clamp(base, min=self.eps)
            pred_ratio = torch.clamp(pred[:, -1, :], min=self.eps) / base
            target_ratio = torch.clamp(target[:, -1, :], min=self.eps) / base
            if self.score_kind in {
                "simple_return",
                "simple_returns",
                "returns",
                "retornos",
            }:
                return pred_ratio - 1.0, target_ratio - 1.0
            return torch.log(pred_ratio), torch.log(target_ratio)

        raise ValueError(f"loss_data_kind inválido: {self.data_kind}")


class FinancialObjectiveV2(nn.Module):
    """Objetivos financeiros v2 com H/K explícitos e escalas controladas."""

    def __init__(
        self,
        *,
        loss_name: str,
        data_kind: str,
        score_kind: str,
        loss_k: int,
        horizon_mode: str,
        inverse_norm: bool,
        normalizer_mean=None,
        normalizer_scale=None,
        rank_lambda: float = 0.5,
        ranknet_alpha: float = 1.0,
        listnet_tau: float = 1.0,
        score_normalization: str = "zscore",
        hybrid_point_normalization: str = "target_std",
        fingat_delta: float = 0.2,
        direction_scale: float = 0.01,
        track_components: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.loss_name = loss_name.lower()
        self.rank_lambda = float(rank_lambda)
        self.ranknet_alpha = float(ranknet_alpha)
        self.listnet_tau = float(listnet_tau)
        self.score_normalization = score_normalization.lower()
        self.hybrid_point_normalization = hybrid_point_normalization.lower()
        self.fingat_delta = float(fingat_delta)
        self.direction_scale = float(direction_scale)
        self.track_components = bool(track_components)
        self.eps = eps

        if not 0.0 <= self.rank_lambda <= 1.0:
            raise ValueError("loss_rank_lambda deve estar em [0, 1].")
        if not 0.0 <= self.fingat_delta <= 1.0:
            raise ValueError("loss_fingat_delta deve estar em [0, 1].")
        if self.listnet_tau <= 0:
            raise ValueError("loss_listnet_tau deve ser positivo.")
        if self.direction_scale <= 0:
            raise ValueError("loss_direction_scale deve ser positivo.")
        if self.score_normalization not in {"none", "zscore"}:
            raise ValueError("loss_score_normalization deve ser 'none' ou 'zscore'.")
        if self.hybrid_point_normalization not in {"raw", "target_std"}:
            raise ValueError(
                "loss_hybrid_point_normalization deve ser 'raw' ou 'target_std'."
            )

        self.aggregator = FinancialScoreAggregatorV2(
            data_kind=data_kind,
            score_kind=score_kind,
            loss_k=loss_k,
            horizon_mode=horizon_mode,
            inverse_norm=inverse_norm,
            normalizer_mean=normalizer_mean,
            normalizer_scale=normalizer_scale,
            eps=eps,
        )
        self.last_components = {}

    @property
    def needs_base_value(self) -> bool:
        return self.aggregator.needs_base_value

    def _zscore(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
        return (x - mean) / std

    def _normalized_scores(
        self, pred_score: torch.Tensor, target_score: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.score_normalization == "zscore":
            return self._zscore(pred_score), self._zscore(target_score)
        return pred_score, target_score

    def _upper_pairwise(
        self, pred_score: torch.Tensor, target_score: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_diff = pred_score.unsqueeze(2) - pred_score.unsqueeze(1)
        target_diff = target_score.unsqueeze(2) - target_score.unsqueeze(1)
        n_assets = pred_score.shape[1]
        upper = torch.triu(
            torch.ones(
                (n_assets, n_assets), device=pred_score.device, dtype=torch.bool
            ),
            diagonal=1,
        ).unsqueeze(0)
        valid = upper & (target_diff != 0)
        return pred_diff, target_diff, valid

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        selected = values[mask]
        if selected.numel() == 0:
            return values.sum() * 0.0
        return selected.mean()

    def _ranknet(
        self, pred_score: torch.Tensor, target_score: torch.Tensor
    ) -> torch.Tensor:
        pred_score, target_score = self._normalized_scores(pred_score, target_score)
        pred_diff, target_diff, valid = self._upper_pairwise(
            pred_score, target_score
        )
        sign = torch.sign(target_diff)
        values = F.softplus(-self.ranknet_alpha * sign * pred_diff)
        return self._masked_mean(values, valid)

    def _listnet(
        self, pred_score: torch.Tensor, target_score: torch.Tensor
    ) -> torch.Tensor:
        pred_score, target_score = self._normalized_scores(pred_score, target_score)
        p_true = F.softmax(target_score / self.listnet_tau, dim=1)
        log_p_pred = F.log_softmax(pred_score / self.listnet_tau, dim=1)
        return -(p_true * log_p_pred).sum(dim=1).mean()

    def _score_mse(
        self, pred_score: torch.Tensor, target_score: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(pred_score, target_score)

    def _hybrid_point_loss(
        self, pred_score: torch.Tensor, target_score: torch.Tensor
    ) -> torch.Tensor:
        if self.hybrid_point_normalization == "raw":
            return self._score_mse(pred_score, target_score)
        target_std = target_score.detach().std(
            dim=1, keepdim=True, unbiased=False
        ).clamp_min(self.eps)
        return torch.mean(((pred_score - target_score) / target_std) ** 2)

    def _fingat_rank(
        self, pred_score: torch.Tensor, target_score: torch.Tensor
    ) -> torch.Tensor:
        pred_score, target_score = self._normalized_scores(pred_score, target_score)
        pred_diff, target_diff, valid = self._upper_pairwise(
            pred_score, target_score
        )
        values = F.relu(-(pred_diff * target_diff))
        return self._masked_mean(values, valid)

    def _direction_bce(
        self, pred_score: torch.Tensor, target_score: torch.Tensor
    ) -> torch.Tensor:
        target_move = (target_score > 0).to(dtype=pred_score.dtype)
        logits = pred_score / self.direction_scale
        return F.binary_cross_entropy_with_logits(logits, target_move)

    def _remember(self, **components: torch.Tensor) -> None:
        if not self.track_components:
            return
        self.last_components = {
            name: float(value.detach().cpu()) for name, value in components.items()
        }

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        base_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred_score, target_score = self.aggregator(
            pred, target, base_value=base_value
        )

        if self.loss_name == "mse_score_v2":
            score_mse = self._score_mse(pred_score, target_score)
            self._remember(score_mse=score_mse, total=score_mse)
            return score_mse

        if self.loss_name == "ranknet_v2":
            rank = self._ranknet(pred_score, target_score)
            self._remember(rank=rank, total=rank)
            return rank

        if self.loss_name == "ranknet_hybrid_v2":
            point = self._hybrid_point_loss(pred_score, target_score)
            rank = self._ranknet(pred_score, target_score)
            total = (1.0 - self.rank_lambda) * point + self.rank_lambda * rank
            self._remember(point=point, rank=rank, total=total)
            return total

        if self.loss_name == "listnet_v2":
            listnet = self._listnet(pred_score, target_score)
            self._remember(listnet=listnet, total=listnet)
            return listnet

        if self.loss_name == "fingat_v2":
            rank = self._fingat_rank(pred_score, target_score)
            direction = self._direction_bce(pred_score, target_score)
            total = (1.0 - self.fingat_delta) * rank + self.fingat_delta * direction
            self._remember(rank=rank, direction=direction, total=total)
            return total

        raise ValueError(f"Loss v2 desconhecida: {self.loss_name}")
