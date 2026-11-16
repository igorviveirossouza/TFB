"""Continuous predicted positions supervised by daily true positions.

RankTargetLoss(config=None, **kwargs) reads the usual ``loss_*`` fields from
a dict or configuration object; keyword fields override the configuration.
``n_assets`` (or config.enc_in) is required before constructing active OCE so
its learned cutpoints can enter the optimizer and model checkpoint at once.

forward(pred_positions[B,H,N], target_positions[B,H,N], target_returns=None)
uses smaller predicted positions for better assets. The temporal term compares
continuous positions with the daily integer positions. Cross scores are minus
the MEAN predicted position in each disjoint K-step block. For K>1, true block
positions are ranked from sums of the original float64 log returns, which must
be supplied as target_returns. Positions are never compounded or exponentiated.

There is no score z-score, return scaler or financial expm1 in this objective.
Its contract is independent of whether historical model inputs are ranks or
log returns and of any input-only normalization performed by the adapter.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

from ts_benchmark.baselines.global_ranking_losses import soft_ranks, spearman_loss
from ts_benchmark.baselines.listwise_ranking_losses import listmle_loss
from ts_benchmark.baselines.ordinal_losses import (
    ORDINAL_LOSSES, bucket_pairwise_loss, bucket_listnet_loss, ordinal_cross_entropy,
    positive_temperature, validate_bucket_count,
)
from ts_benchmark.baselines.rank_target_data import stable_descending_ranks_torch

GLOBAL_RANK_TARGET_LOSSES = frozenset({
    "ranknet", "hinge", "listnet", "listmle", "approxndcg", "spearman", "pairwise_mse",
})
RANK_TARGET_CROSS_LOSSES = ORDINAL_LOSSES | GLOBAL_RANK_TARGET_LOSSES


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} deve ser inteiro >= {minimum}.")
    return int(value)


class RankOrderedThresholds(nn.Module):
    """Descending OCE cutpoints initialized between consecutive rank buckets."""

    def __init__(self, bucket_count, n_assets):
        super().__init__()
        self.bucket_count = validate_bucket_count(bucket_count)
        self.n_assets = _integer(n_assets, "n_assets", self.bucket_count)
        boundaries = [(m * self.n_assets + bucket_count - 1) // bucket_count
                      for m in range(1, bucket_count)]
        initial = -torch.tensor(boundaries, dtype=torch.float32) - .5
        self.first = nn.Parameter(initial[:1].clone())
        gaps = initial[:-1] - initial[1:] - 1e-6
        # Stable inverse softplus, including large universes of assets.
        self.raw_gaps = nn.Parameter(gaps + torch.log1p(-torch.exp(-gaps)))

    def forward(self):
        gaps = F.softplus(self.raw_gaps) + 1e-6
        return torch.cat((self.first, self.first - gaps.cumsum(0)))


class RankTargetLoss(nn.Module):
    """(1-lambda)*daily_position_error + lambda*cross_scale*ranking_loss."""

    needs_base_value = False
    cross_construction = "blocks"

    def __init__(self, config=None, **kwargs):
        super().__init__()

        def get(name, default=None):
            if name in kwargs:
                return kwargs[name]
            if isinstance(config, Mapping):
                return config.get(name, default)
            return getattr(config, name, default) if config is not None else default

        self.temporal_loss_name = str(get("loss_temporal", "mse")).lower()
        self.cross_loss_name = str(get("loss_cross", "bucket_hinge")).lower()
        self.cross_lambda = float(get("loss_cross_lambda", .5))
        self.cross_scale = float(get("loss_cross_scale", 1.))
        self.trade_window = _integer(get("loss_trade_window", get("loss_k", 1)), "loss_trade_window")
        n_assets = get("n_assets", get("enc_in", None))
        self.n_assets = None if n_assets is None else _integer(n_assets, "n_assets")
        if not math.isfinite(self.cross_lambda) or not 0 <= self.cross_lambda <= 1:
            raise ValueError("loss_cross_lambda deve estar em [0,1].")
        if not math.isfinite(self.cross_scale) or self.cross_scale <= 0:
            raise ValueError("loss_cross_scale deve ser finito e positivo.")
        if self.temporal_loss_name not in {"mse", "mae", "huber"}:
            raise ValueError("loss_temporal deve ser mse, mae ou huber.")
        if self.cross_loss_name not in RANK_TARGET_CROSS_LOSSES:
            raise ValueError("Loss de ranking de posicoes desconhecida: " + self.cross_loss_name)
        self.huber_delta = positive_temperature(get("loss_huber_delta", .5), "loss_huber_delta")
        track = get("loss_track_components", False)
        self.track_components = track if isinstance(track, bool) else str(track).lower() in {"true", "1"}
        self.last_components = {}
        self.ordinal_thresholds = None
        if self.cross_lambda == 0:
            return

        if self.cross_loss_name in ORDINAL_LOSSES:
            self.bucket_count = validate_bucket_count(get("loss_bucket_count", 3))
            if self.n_assets is not None and self.bucket_count > self.n_assets:
                raise ValueError("loss_bucket_count nao pode superar n_assets.")
        if self.cross_loss_name == "ordinal_ce":
            if self.n_assets is None:
                raise ValueError("OCE exige n_assets (ou enc_in) antes de criar o otimizador.")
            self.ordinal_tau = positive_temperature(get("loss_ordinal_tau", 1.), "loss_ordinal_tau")
            self.ordinal_thresholds = RankOrderedThresholds(self.bucket_count, self.n_assets)
        if self.cross_loss_name in {"bucket_ranknet", "bucket_listnet"}:
            self.bucket_tau = positive_temperature(get("loss_bucket_tau", 1.), "loss_bucket_tau")
        if self.cross_loss_name == "bucket_listnet":
            self.bucket_target_tau = positive_temperature(get("loss_bucket_target_tau", 1.), "loss_bucket_target_tau")
        if self.cross_loss_name == "ranknet":
            self.ranknet_alpha = positive_temperature(get("loss_ranknet_alpha", 1.), "loss_ranknet_alpha")
        if self.cross_loss_name in {"hinge", "bucket_hinge"}:
            self.hinge_margin = float(get("loss_hinge_margin", .1))
            if not math.isfinite(self.hinge_margin) or self.hinge_margin < 0:
                raise ValueError("loss_hinge_margin deve ser finita e nao negativa.")
        for name in ("listnet", "listmle", "approxndcg", "spearman"):
            if self.cross_loss_name == name:
                setattr(self, name + "_tau", positive_temperature(get("loss_" + name + "_tau", 1.), "loss_" + name + "_tau"))

    def _validate(self, predicted, positions):
        if (predicted.ndim != 3 or predicted.shape != positions.shape
                or any(size < 1 for size in predicted.shape)):
            raise ValueError("Previsoes e alvos devem ter mesmo formato nao vazio [B,H,N].")
        if not predicted.is_floating_point() or not torch.isfinite(predicted).all():
            raise ValueError("Posicoes previstas devem ser continuas e finitas.")
        if predicted.device != positions.device:
            raise ValueError("Previsoes e alvos devem estar no mesmo dispositivo.")
        n = predicted.shape[-1]
        if self.n_assets is not None and n != self.n_assets:
            raise ValueError("Numero de ativos difere de n_assets configurado.")
        labels = positions.detach().to(torch.float64)
        expected = torch.arange(1, n + 1, device=labels.device, dtype=labels.dtype).expand_as(labels)
        if not torch.isfinite(labels).all() or not torch.equal(labels.sort(dim=-1).values, expected):
            raise ValueError("Cada alvo diario deve ser uma permutacao inteira de 1..N.")
        return labels.to(torch.long)

    def _cross(self, scores, positions):
        name = self.cross_loss_name
        n = scores.shape[-1]
        if name in ORDINAL_LOSSES:
            if n < self.bucket_count:
                raise ValueError("loss_bucket_count nao pode superar o numero de ativos.")
            classes = torch.div((positions - 1) * self.bucket_count, n, rounding_mode="floor") + 1
            if name == "ordinal_ce":
                return ordinal_cross_entropy(scores, classes, self.ordinal_thresholds, self.ordinal_tau)
            if name == "bucket_listnet":
                return bucket_listnet_loss(scores, classes, self.bucket_count,
                    temperature=self.bucket_tau, target_temperature=self.bucket_target_tau)
            return bucket_pairwise_loss(scores, classes, kind=name,
                temperature=getattr(self, "bucket_tau", 1.), margin=getattr(self, "hinge_margin", .1))

        target_scores = -positions.to(scores.dtype)
        if n < 2:
            return scores.sum() * 0.
        if name == "listnet":
            # softmax(-position/tau) == softmax((N-position)/tau).
            probability = F.softmax(target_scores / self.listnet_tau, dim=-1)
            return -(probability * F.log_softmax(scores / self.listnet_tau, dim=-1)).sum(-1).mean()
        if name == "listmle":
            return listmle_loss(scores, target_scores, tau=self.listmle_tau)
        if name == "spearman":
            return spearman_loss(scores, target_scores, tau=self.spearman_tau)
        if name == "approxndcg":
            predicted_ranks = soft_ranks(scores, self.approxndcg_tau).clamp_min(1.)
            relevance = (n - positions.to(scores.dtype)) / (n - 1)
            gains = torch.pow(2., relevance) - 1.
            ideal_positions = torch.arange(1, n + 1, device=scores.device, dtype=scores.dtype)
            ideal = (gains.sort(dim=-1, descending=True).values / torch.log2(1 + ideal_positions)).sum(-1)
            dcg = (gains / torch.log2(1 + predicted_ranks)).sum(-1)
            return (1 - dcg / ideal.clamp_min(torch.finfo(scores.dtype).tiny)).mean()

        valid = positions.unsqueeze(2) < positions.unsqueeze(1)
        difference = scores.unsqueeze(2) - scores.unsqueeze(1)
        if name == "ranknet":
            values = F.softplus(-self.ranknet_alpha * difference)
        elif name == "hinge":
            values = F.relu(self.hinge_margin - difference)
        else:  # pairwise_mse uses exact differences in rank units.
            target_difference = target_scores.unsqueeze(2) - target_scores.unsqueeze(1)
            values = (difference - target_difference).square()
        row_losses = torch.where(valid, values, torch.zeros_like(values)).sum((1, 2)) / valid.sum((1, 2))
        return row_losses.mean()

    def forward(self, pred_positions, target_positions, target_returns=None):
        labels = self._validate(pred_positions, target_positions)
        prediction = (pred_positions.float() if pred_positions.dtype in (torch.float16, torch.bfloat16)
                      else pred_positions)
        temporal = cross = None
        if self.cross_lambda != 1:
            target = labels.to(prediction.dtype)
            if self.temporal_loss_name == "mse":
                temporal = F.mse_loss(prediction, target)
            elif self.temporal_loss_name == "mae":
                temporal = F.l1_loss(prediction, target)
            else:
                temporal = F.huber_loss(prediction, target, delta=self.huber_delta)
        if self.cross_lambda != 0:
            b, h, n = prediction.shape
            k = self.trade_window
            if h % k:
                raise ValueError("loss_trade_window K deve dividir H exatamente.")
            scores = -prediction.reshape(b, h // k, k, n).mean(2)
            if k > 1:
                if target_returns is None:
                    raise ValueError("K>1 exige target_returns com os log-retornos originais.")
                if target_returns.shape != prediction.shape:
                    raise ValueError("target_returns deve ter mesmo formato [B,H,N].")
                returns = target_returns.detach().to(device=prediction.device, dtype=torch.float64)
                if not torch.isfinite(returns).all():
                    raise ValueError("target_returns deve conter log-retornos finitos.")
                totals = returns.reshape(b, h // k, k, n).sum(2)
                block_positions = stable_descending_ranks_torch(totals)
            else:
                block_positions = labels
            cross = self._cross(scores.reshape(-1, n), block_positions.reshape(-1, n))
        if self.cross_lambda == 0:
            total = temporal
        elif self.cross_lambda == 1:
            total = self.cross_scale * cross
        else:
            total = (1 - self.cross_lambda) * temporal + self.cross_lambda * self.cross_scale * cross
        if self.track_components:
            self.last_components = {
                "temporal": None if temporal is None else float(temporal.detach()),
                "cross": None if cross is None else float(cross.detach()),
                "total": float(total.detach()), "H": prediction.shape[1], "K": self.trade_window,
                "n_assets": prediction.shape[2], "cross_construction": "blocks",
                "target_kind": "rank_positions", "prediction_orientation": "lower_is_better",
            }
        return total
