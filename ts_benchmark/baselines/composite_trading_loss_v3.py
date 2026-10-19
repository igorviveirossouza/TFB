"""Composite trading loss v3: financial aggregation, then cross-sectional standardization.

The model still predicts the full path [B,H,N]. The temporal component is
computed on the complete model-space path. The cross-sectional component first
aggregates financially meaningful block scores of length K and only then
standardizes each block across the N assets. K must divide H exactly, so the
cross-sectional blocks cover the complete forecast horizon without a remainder.

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
from ts_benchmark.baselines.proximity_losses import (
    PROXIMITY_LOSSES,
    validate_cross_mask,
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
        cross_delta=float(_cfg(config, "loss_cross_delta", 0.0)),
        cross_mask=_cfg(config, "loss_cross_mask", None),
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
        cross_delta: float = 0.0,
        cross_mask=None,
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

        cross_loss_name = cross_loss.lower()
        proximity_mask = (
            validate_cross_mask(cross_mask)
            if cross_loss_name in PROXIMITY_LOSSES
            else None
        )

        # V2 already implements the correct return/price block aggregation.
        # Its rank_score_normalization attribute is reused internally only as a
        # validated storage slot; V3 applies it to every cross objective.
        super().__init__(
            trade_window=trade_window,
            data_kind=data_kind,
            score_kind=score_kind,
            temporal_loss=temporal_loss,
            cross_loss=PROXIMITY_LOSSES.get(cross_loss_name, cross_loss_name),
            cross_lambda=cross_lambda,
            cross_scale=cross_scale,
            huber_delta=huber_delta,
            ranknet_alpha=ranknet_alpha,
            listnet_tau=listnet_tau,
            hinge_margin=hinge_margin,
            cross_delta=cross_delta,
            rank_score_normalization=normalization,
            inverse_norm=inverse_norm,
            normalizer_mean=normalizer_mean,
            normalizer_scale=normalizer_scale,
            track_components=track_components,
            eps=eps,
        )
        self.cross_loss_name = cross_loss_name
        # Original objectives deliberately ignore this new hyperparameter.
        self.cross_mask = proximity_mask
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
        # Delta is evaluated on the realized financial score before z-scoring,
        # so its unit remains the original score unit (e.g. simple return).
        pred_raw, target_raw = self._flatten_blocks(pred_scores, target_scores)

        if self.cross_loss_name in PROXIMITY_LOSSES:
            return self._proximity_cross_loss(pred_raw, target_raw)

        if self.cross_loss_name == "mse":
            pred, target = self._normalize_cross_scores(pred_raw, target_raw)
            return F.mse_loss(pred, target)

        if self.cross_loss_name == "listnet":
            pred, target = self._normalize_cross_scores(pred_raw, target_raw)
            p_true = F.softmax(target / self.listnet_tau, dim=1)
            log_p_pred = F.log_softmax(pred / self.listnet_tau, dim=1)
            return -(p_true * log_p_pred).sum(dim=1).mean()

        _, target_diff_raw, valid = self._pairwise_upper(pred_raw, target_raw)
        valid = valid & (target_diff_raw.abs() > self.cross_delta)

        pred, target = self._normalize_cross_scores(pred_raw, target_raw)
        pred_diff, target_diff, _ = self._pairwise_upper(pred, target)

        if self.cross_loss_name == "pairwise_mse":
            values = (pred_diff - target_diff).square()
            return self._masked_mean(values, valid)

        sign = torch.sign(target_diff)

        if self.cross_loss_name == "ranknet":
            values = F.softplus(-self.ranknet_alpha * sign * pred_diff)
            return self._masked_mean(values, valid)

        if self.cross_loss_name == "hinge":
            values = F.relu(self.hinge_margin - sign * pred_diff)
            return self._masked_mean(values, valid)

        # BPR: pairwise logistic with realized ordering as orientation.
        return self._masked_mean(F.softplus(-(sign * pred_diff)), valid)

    def _proximity_cross_loss(
        self,
        pred_raw: torch.Tensor,
        target_raw: torch.Tensor,
    ) -> torch.Tensor:
        """Proximity-weighted pairs, including exact realized-score ties.

        Normalize weights independently within each [N]-asset cross-section,
        then average its loss over batch elements and trading blocks. A uniform
        mask with no eligible pair contributes a differentiable zero. This
        intentionally does not use V2's tie-excluding pair validity helper.

        As in the original V3, optional z-scoring couples all asset scores:
        an individual tie term favors equal normalized scores, but does not
        guarantee a nonzero gradient for every degenerate cross-section (e.g.
        two distinct predicted scores always normalize to a fixed gap).
        """
        n_assets = pred_raw.shape[1]
        if n_assets < 2:
            return pred_raw.sum() * 0.0

        rows, cols = torch.triu_indices(
            n_assets, n_assets, offset=1, device=pred_raw.device
        )
        # Float64 protects the financial-unit mask when subtracting large
        # float32 values or dividing by a very small Gaussian bandwidth.
        realized = target_raw.to(torch.float64)
        realized_diff = realized[:, rows] - realized[:, cols]
        gap = realized_diff.abs()
        mask_name, bandwidth = self.cross_mask
        if mask_name == "uniform":
            weights = (gap <= bandwidth).to(pred_raw.dtype)
        else:
            # exp(log(w) - max(log(w))) cancels in the weighted mean. Compute
            # differences of squares in factored form so every closest pair
            # has weight 1 even when ALL original Gaussian weights underflow.
            min_gap = gap.amin(dim=1, keepdim=True)
            distance_from_min = gap - min_gap
            relative_energy = (
                (distance_from_min / bandwidth)
                * ((gap + min_gap) / bandwidth)
                * 0.5
            )
            relative_energy = torch.where(
                distance_from_min == 0,
                torch.zeros_like(relative_energy),
                relative_energy,
            )
            weights = torch.exp(-relative_energy).to(pred_raw.dtype)

        pred, target = self._normalize_cross_scores(pred_raw, target_raw)
        pred_diff = pred[:, rows] - pred[:, cols]
        target_diff = target[:, rows] - target[:, cols]
        sign = realized_diff.sign().to(pred_diff.dtype)
        base_loss = PROXIMITY_LOSSES[self.cross_loss_name]
        if base_loss == "hinge":
            values = torch.where(
                realized_diff == 0,
                pred_diff.abs(),
                F.relu(self.hinge_margin - sign * pred_diff),
            )
        elif base_loss == "ranknet":
            values = F.binary_cross_entropy_with_logits(
                self.ranknet_alpha * pred_diff,
                (1.0 + sign) * 0.5,
                reduction="none",
            )
        else:
            values = (pred_diff - target_diff).square()

        # At least one Gaussian weight is exactly 1. For uniform masks only,
        # an empty section has numerator 0 and denominator clamped to 1.
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return ((weights * values).sum(dim=1) / denominator).mean()

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
        if self.track_components:
            self.last_components.update(H=int(pred.shape[1]),
                                        cross_steps=int(pred_scores.shape[1] * self.trade_window),
                                        cross_tail_steps=int(pred.shape[1] % self.trade_window))
        return total
