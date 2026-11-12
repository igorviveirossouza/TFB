"""Composite trading loss v3: financial aggregation, then cross-sectional standardization.

The model still predicts the full path [B,H,N]. The temporal component is
computed on the complete model-space path. The cross-sectional component first
aggregates financially meaningful block scores of length K and only then
standardizes each block across the N assets. K must divide H exactly, so the
cross-sectional blocks cover the complete forecast horizon without a remainder.

Opt-in ``origin_anchored_windows`` instead ranks cumulative prefixes of the
same forecast path (e.g. its first 1, 5 and 25 days), equally weighted. It is
architecture independent and preserves the existing block objective by default.

For each batch element b and trading block j:
    Z[b,j,i] = (S[b,j,i] - mean_i S[b,j,i]) / std_i S[b,j,i]

The same blockwise z-score transformation is used for MSE and ranking losses.
This preserves the financial definition of S while removing arbitrary score
level/scale before the cross-sectional objective.
"""
from __future__ import annotations

import math
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
    kernel_parameters,
)
from ts_benchmark.baselines.momentum_score import (
    LOG_RETURN_KINDS,
    canonical_score_kind,
    validate_momentum_config,
)
from ts_benchmark.baselines.global_ranking_losses import (
    spearman_loss,
    validate_spearman_tau,
)

from ts_benchmark.baselines.listwise_ranking_losses import (
    validate_temperature, financial_pair_weights, listnet_proximity_loss,
    listmle_loss, approxndcg_loss,
)
from ts_benchmark.baselines.ranking_mixture import (
    validate_rank_mix_config, validate_rank_mix_proximity_components,
)
from ts_benchmark.baselines.hybrid_ranking_losses import ranking_mixture_loss


def _scaled_log_weights(log_weights, dtype):
    """Rescaling cancels in a weighted mean, including extremely small weights."""
    maximum = log_weights.amax(dim=1, keepdim=True)
    shift = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    return torch.exp(log_weights - shift).to(dtype)


def _log_complement(log_energy):
    """log(1-exp(-E)) from log(E), retaining complements below float64 range."""
    return torch.where(
        log_energy < -36.0,
        log_energy,
        (-torch.expm1(-torch.exp(log_energy))).log(),
    )


def proximity_weights(gap, mask, dtype):
    """Pair weights, rescaled per cross-section, with exact legacy special cases.

    Natural finite-support kernels and the intermediate band may have no active
    pair. Such rows remain zero. Complements use the original kernel, not its
    rescaled weights. As in the existing complement contract, ties stay zero.
    """
    mask_name, bandwidth = mask[:2]
    mode = mask[2] if len(mask) >= 3 else "natural"
    family, shape = kernel_parameters(mask)
    # Route equivalent family specifications through the original arithmetic.
    if family == "student" and shape[0] == 1:
        mask_name = "cauchy"
    elif family == "generalized_gaussian" and shape[0] == 2:
        mask_name = "gaussian"
    elif family == "compact" and shape[1] == 0:
        mask_name = "uniform"
    if mask_name == "uniform":
        eligible = gap <= bandwidth if mode == "natural" else gap > bandwidth
        return eligible.to(dtype)
    if mask_name == "gaussian" and mode == "natural":
        min_gap = gap.amin(dim=1, keepdim=True)
        distance_from_min = gap - min_gap
        relative_energy = ((distance_from_min / bandwidth)
                           * ((gap + min_gap) / bandwidth) * 0.5)
        relative_energy = torch.where(distance_from_min == 0,
                                      torch.zeros_like(relative_energy), relative_energy)
        return torch.exp(-relative_energy).to(dtype)
    log_u = gap.log() - math.log(bandwidth)
    if mask_name in {"cauchy", "gaussian"}:
        # Original Cauchy / complementary-Gaussian path, deliberately unchanged.
        log_ratio_squared = 2.0 * log_u
        if mask_name == "cauchy":
            log_weights = -F.softplus(log_ratio_squared if mode == "natural" else -log_ratio_squared)
        else:
            log_weights = _log_complement(log_ratio_squared - math.log(2.0))
        return _scaled_log_weights(log_weights, dtype)
    if family == "student":
        nu = shape[0]
        coefficient = (nu + 1.0) / 2.0
        log_ratio = 2.0 * log_u - math.log(nu)
        soft = F.softplus(log_ratio)
        if mode == "natural":
            difference = soft - soft.amin(dim=1, keepdim=True)
            return torch.exp(-coefficient * difference).to(dtype)
        log_soft = torch.where(log_ratio < -36.0, log_ratio, soft.log())
        log_weights = _log_complement(math.log(coefficient) + log_soft)
    elif family == "generalized_gaussian":
        p = shape[0]
        log_energy = p * log_u - math.log(p)
        if mode == "natural":
            minimum = log_u.amin(dim=1, keepdim=True)
            log_difference = p * log_u + torch.log(-torch.expm1(p * (minimum - log_u))) - math.log(p)
            relative = torch.where(log_u == minimum, torch.zeros_like(log_u), torch.exp(log_difference))
            return torch.exp(-relative).to(dtype)
        log_weights = _log_complement(log_energy)
    elif family == "compact":
        p, q = shape
        inside = gap < bandwidth  # q>0: boundary has exactly zero natural weight.
        log_power = p * log_u
        log_one_minus = torch.log(-torch.expm1(log_power.clamp_max(0.0)))
        if mode == "natural":
            valid_log = torch.where(inside, log_one_minus, torch.full_like(log_u, -torch.inf))
            maximum = valid_log.amax(dim=1, keepdim=True)
            shift = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
            log_weights = torch.where(inside, q * (log_one_minus - shift),
                                      torch.full_like(log_u, -torch.inf))
        else:
            # E=-q*log(1-u^p); log(E)~log(q)+p*log(u) near the centre.
            log_e = torch.where(log_power < -36.0, math.log(q) + log_power,
                                math.log(q) + (-log_one_minus).log())
            log_weights = torch.where(inside, _log_complement(log_e), torch.zeros_like(log_u))
    elif family == "cosine":
        inside = gap < bandwidth
        u = torch.exp(log_u.clamp_max(0.0))
        if mode == "natural":
            log_weights = torch.where(inside, torch.cos(math.pi * u / 2.0).log(),
                                      torch.full_like(log_u, -torch.inf))
        else:
            # 1-cos(pi*u/2)=2*sin(pi*u/4)^2; use its logarithmic limit near zero.
            log_sin = torch.where(log_u < -18.0, log_u + math.log(math.pi / 4.0),
                                  torch.sin(math.pi * u / 4.0).log())
            log_weights = torch.where(inside, math.log(2.0) + 2.0 * log_sin,
                                      torch.zeros_like(log_u))
    elif family == "intermediate_band":
        p = shape[0]
        # phi(u)=u-1-log(u), nonnegative; use a series around u=1 to avoid
        # cancellation and retain a complementary zero exactly at the peak.
        x = log_u
        series = x.square() * (0.5 + x / 6.0 + x.square() / 24.0 + x.pow(3) / 120.0)
        phi = torch.where(x.abs() < 1e-3, series, torch.expm1(x) - x).clamp_min(0.0)
        phi = torch.where(gap > 0, phi, torch.full_like(phi, torch.inf))
        if mode == "natural":
            reference = phi.argmin(dim=1, keepdim=True)
            # If every u overflows, monotonic right tail selects the smallest u.
            finite = torch.isfinite(phi).any(dim=1, keepdim=True)
            positive_log_u = torch.where(gap > 0, x, torch.full_like(x, torch.inf))
            reference = torch.where(finite, reference, positive_log_u.argmin(dim=1, keepdim=True))
            ref_x = x.gather(1, reference)
            ref_phi = phi.gather(1, reference)
            delta = phi - ref_phi
            # Stable right-tail difference when both exponentials overflow.
            dx = x - ref_x
            du = torch.exp(torch.maximum(x, ref_x) + torch.log(-torch.expm1(-dx.abs()))) * dx.sign()
            tail_delta = torch.where(dx == 0, torch.zeros_like(dx), du - dx)
            delta = torch.where(torch.isfinite(ref_phi), delta, tail_delta).clamp_min(0.0)
            relative = p * delta
            log_weights = torch.where(gap > 0, -relative, torch.full_like(x, -torch.inf))
        else:
            log_weights = _log_complement(math.log(p) + phi.log())
    else:
        raise ValueError("Familia de kernel desconhecida: " + family)
    if mode == "complementary":
        log_weights = torch.where(gap > 0, log_weights, torch.full_like(log_weights, -torch.inf))
    return _scaled_log_weights(log_weights, dtype)



def build_loss(config, normalizer_mean=None, normalizer_scale=None):
    momentum = validate_momentum_config(config)
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
    if (_cfg(config, "loss_cross_construction", "blocks") == "origin_anchored_windows"
            and _as_bool(_cfg(config, "norm", False)) and not inverse_norm):
        raise ValueError("origin_anchored_windows exige loss_inverse_norm=True quando norm=True.")

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
        spearman_tau=float(_cfg(config, "loss_spearman_tau", 1.0)),
        listmle_tau=float(_cfg(config, "loss_listmle_tau", 1.0)),
        approxndcg_tau=float(_cfg(config, "loss_approxndcg_tau", 1.0)),
        rank_mix_weights=_cfg(config, "loss_rank_mix_weights", None),
        rank_mix_mode=_cfg(config, "loss_rank_mix_mode", "global"),
        rank_mix_regions=_cfg(config, "loss_rank_mix_regions", None),
        rank_mix_fraction=_cfg(config, "loss_rank_mix_fraction", 1.0 / 3.0),
        rank_mix_proximity_components=_cfg(config, "loss_rank_mix_proximity_components", None),
        hinge_margin=float(_cfg(config, "loss_hinge_margin", 0.01)),
        cross_delta=float(_cfg(config, "loss_cross_delta", 0.0)),
        cross_mask=_cfg(config, "loss_cross_mask", None),
        cross_score_normalization=cross_score_normalization,
        inverse_norm=inverse_norm,
        normalizer_mean=normalizer_mean,
        normalizer_scale=normalizer_scale,
        track_components=_as_bool(_cfg(config, "loss_track_components", False)),
        momentum_mode=momentum["mode"],
        cross_construction=str(_cfg(config, "loss_cross_construction", "blocks")),
        rank_windows=_cfg(config, "loss_rank_windows", None),
    )


def loss_accepts_base_value(criterion) -> bool:
    if isinstance(criterion, CompositeTradingLossV3):
        return criterion.needs_base_value
    return loss_accepts_base_value_v1(criterion)


class CompositeTradingLossV3(CompositeTradingLossV2):
    """V2 objective with one normalization rule for every cross loss.

    Financial scores S are always built first. Then each [N]-asset score vector
    is optionally standardized independently inside its batch/trading window.
    ``blocks`` retains the historical disjoint aggregation; the opt-in
    ``origin_anchored_windows`` matches the restored shared-head experiment.
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
        spearman_tau: float = 1.0,
        listmle_tau: float = 1.0,
        approxndcg_tau: float = 1.0,
        rank_mix_weights=None,
        rank_mix_mode: str = "global",
        rank_mix_regions=None,
        rank_mix_fraction: float = 1.0 / 3.0,
        rank_mix_proximity_components=None,
        hinge_margin: float = 0.01,
        cross_delta: float = 0.0,
        cross_mask=None,
        cross_score_normalization: str = "zscore",
        inverse_norm: bool = True,
        normalizer_mean=None,
        normalizer_scale=None,
        track_components: bool = False,
        eps: float = 1e-8,
        momentum_mode: str = "off",
        cross_construction: str = "blocks",
        rank_windows=None,
    ) -> None:
        momentum_mode = str(momentum_mode).lower()
        if cross_construction not in {"blocks", "origin_anchored_windows"}:
            raise ValueError("loss_cross_construction deve ser blocks ou origin_anchored_windows.")
        if cross_construction == "origin_anchored_windows":
            if str(data_kind).lower() not in LOG_RETURN_KINDS:
                raise ValueError("origin_anchored_windows exige dados em log-retornos.")
            if momentum_mode != "off":
                raise ValueError("origin_anchored_windows exige momentum_mode=off.")
            if trade_window != 1:
                raise ValueError("origin_anchored_windows exige K_loss=1; use loss_rank_windows para as janelas.")
            if (not isinstance(rank_windows, (list, tuple)) or not rank_windows
                    or any(type(k) is not int or k < 1 for k in rank_windows)
                    or len(set(rank_windows)) != len(rank_windows)):
                raise ValueError("loss_rank_windows exige uma lista nao vazia de inteiros positivos unicos.")
            canonical_score_kind(score_kind)
        if momentum_mode not in {"off", "residual_output"}:
            raise ValueError("momentum_mode deve ser off ou residual_output.")
        if momentum_mode == "residual_output":
            if not 0 < float(cross_lambda) <= 1:
                raise ValueError("Momentum exige 0 < loss_cross_lambda <= 1.")
            if str(data_kind).lower() not in LOG_RETURN_KINDS:
                raise ValueError("Momentum exige entrada em log-retornos.")
            canonical_score_kind(score_kind)
        normalization = cross_score_normalization.lower()
        if normalization not in {"none", "zscore"}:
            raise ValueError(
                "loss_cross_score_normalization deve ser 'none' ou 'zscore'."
            )

        cross_loss_name = cross_loss.lower()
        # V2 supplies path aggregation and common parameter validation. New
        # full-list objectives are implemented only in V3, without changing
        # the accepted objectives or numerical behavior of the legacy class.
        base_cross_loss = PROXIMITY_LOSSES.get(cross_loss_name, cross_loss_name)
        spearman_tau = validate_spearman_tau(spearman_tau)
        listmle_tau = validate_temperature(listmle_tau, "loss_listmle_tau")
        approxndcg_tau = validate_temperature(approxndcg_tau, "loss_approxndcg_tau")
        if momentum_mode == "residual_output" and normalization == "none" and cross_loss_name in {
            "mse", "pairwise_mse", "pairwise_mse_proximity"
        }:
            raise ValueError("Momentum exige zscore para losses de magnitude: mse e pairwise_mse (inclusive proximity).")
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
            cross_loss="ranknet" if base_cross_loss in {"spearman", "listmle", "approxndcg", "rankmix"} else base_cross_loss,
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
        self.momentum_mode = momentum_mode
        self.cross_construction = cross_construction
        # Inactive options must not change the legacy blocks calculation.
        self.rank_windows = tuple(rank_windows) if cross_construction == "origin_anchored_windows" else ()
        self.spearman_tau = spearman_tau
        self.listmle_tau = listmle_tau
        self.approxndcg_tau = approxndcg_tau
        # Opt-in: stale or unrelated mixture options cannot change legacy runs.
        self.rank_mix_config = (validate_rank_mix_config(
            rank_mix_weights, rank_mix_mode, rank_mix_regions, rank_mix_fraction)
            if base_cross_loss == "rankmix" else None)
        self.rank_mix_proximity_components = (
            validate_rank_mix_proximity_components(rank_mix_proximity_components)
            if cross_loss_name == "rankmix_proximity" else None)

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

        if self.cross_loss_name in {"rankmix", "rankmix_proximity"}:
            pred, target = self._normalize_cross_scores(pred_raw, target_raw.detach())
            weights = None
            if self.cross_loss_name == "rankmix_proximity":
                dtype = torch.float32 if pred.dtype in (torch.float16, torch.bfloat16) else pred.dtype
                weights = financial_pair_weights(target_raw, self.cross_mask, dtype)
            return ranking_mixture_loss(
                pred, target_raw, target_values=target, pair_weights=weights,
                proximity_components=self.rank_mix_proximity_components,
                listnet_tau=self.listnet_tau, listmle_tau=self.listmle_tau,
                approxndcg_tau=self.approxndcg_tau, **self.rank_mix_config)

        if self.cross_loss_name in PROXIMITY_LOSSES:
            return self._proximity_cross_loss(pred_raw, target_raw)

        if self.cross_loss_name == "spearman":
            pred, _ = self._normalize_cross_scores(pred_raw, target_raw)
            return spearman_loss(pred, target_raw, tau=self.spearman_tau, eps=self.eps)

        if self.cross_loss_name in {"listmle", "approxndcg"}:
            pred, _ = self._normalize_cross_scores(pred_raw, target_raw)
            function = listmle_loss if self.cross_loss_name == "listmle" else approxndcg_loss
            tau = self.listmle_tau if self.cross_loss_name == "listmle" else self.approxndcg_tau
            return function(pred, target_raw, tau=tau)

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
        """Natural/complementary weights from realized financial gaps.

        ListNet/ListMLE/ApproxNDCG use absolute kernel values in their listwise
        objectives. The normalized weighted-pair contract below applies to
        the earlier pair losses and Spearman.

        Normalize weights independently within each [N]-asset cross-section,
        then average its loss over batch elements and trading blocks. A mask
        with no eligible pair contributes a differentiable zero. Natural masks
        retain their value at ties (zero for the intermediate band); complements
        give ties exactly zero weight. This
        intentionally does not use V2's tie-excluding pair validity helper.

        As in the original V3, optional z-scoring couples all asset scores:
        an individual tie term favors equal normalized scores, but does not
        guarantee a nonzero gradient for every degenerate cross-section (e.g.
        two distinct predicted scores always normalize to a fixed gap).
        """
        n_assets = pred_raw.shape[1]
        if n_assets < 2:
            return pred_raw.sum() * 0.0

        if self.cross_loss_name in {"listnet_proximity", "listmle_proximity", "approxndcg_proximity"}:
            # These listwise adaptations require absolute kernel weights.
            # Pairwise rescaling below is retained unchanged for legacy losses.
            weight_dtype = (torch.float32 if pred_raw.dtype in (torch.float16, torch.bfloat16)
                            else pred_raw.dtype)
            weights = financial_pair_weights(target_raw, self.cross_mask, weight_dtype)
            pred, target = self._normalize_cross_scores(pred_raw, target_raw.detach())
            if self.cross_loss_name == "listnet_proximity":
                return listnet_proximity_loss(pred, target, pair_weights=weights, tau=self.listnet_tau)
            if self.cross_loss_name == "listmle_proximity":
                return listmle_loss(pred, target_raw, pair_weights=weights, tau=self.listmle_tau)
            return approxndcg_loss(pred, target_raw, pair_weights=weights, tau=self.approxndcg_tau)

        rows, cols = torch.triu_indices(
            n_assets, n_assets, offset=1, device=pred_raw.device
        )
        # Float64 protects the financial-unit mask when subtracting large
        # float32 values or dividing by a very small Gaussian bandwidth.
        realized = target_raw.to(torch.float64)
        realized_diff = realized[:, rows] - realized[:, cols]
        gap = realized_diff.abs()
        weights = proximity_weights(gap, self.cross_mask, pred_raw.dtype)

        pred, target = self._normalize_cross_scores(pred_raw, target_raw)
        if self.cross_loss_name == "spearman_proximity":
            return spearman_loss(pred, target_raw, tau=self.spearman_tau,
                                 pair_weights=weights, eps=self.eps)
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

        # Every nonempty section has maximum weight 1 after rescaling. Empty
        # sections have numerator 0 and denominator clamped to 1. Rescaling
        # cancels in the weighted mean, with no floor on any original weight.
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return ((weights * values).sum(dim=1) / denominator).mean()

    def _forward_origin_windows(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Rank equally weighted cumulative windows starting at each origin.

        This is the shared-head objective used by NSTaskAdapter, expressed as
        a loss so any native forecasting architecture can use it. The temporal
        target remains the complete model-space path. Ranking uses restored
        economic log returns; a window K never observes target steps after K.
        """
        if pred.ndim != 3 or pred.shape[1] < 1:
            raise ValueError("origin_anchored_windows exige tensores [B,H,N] com H positivo.")
        windows = tuple(k for k in self.rank_windows if k <= pred.shape[1])
        if not windows:
            raise ValueError("Nenhuma loss_rank_windows cabe no horizonte H.")

        temporal = cross = None
        if self.cross_lambda != 1.0:
            temporal = self._temporal_loss(pred, target)
        if self.cross_lambda != 0.0:
            raw_pred = self.aggregator._maybe_inverse_norm(pred)
            raw_target = self.aggregator._maybe_inverse_norm(target)
            scores, labels = [], []
            for k in windows:
                score = raw_pred[:, :k].sum(1)
                label = raw_target[:, :k].sum(1)
                if canonical_score_kind(self.aggregator.score_kind) == "simple_return":
                    score, label = torch.expm1(score), torch.expm1(label)
                scores.append(score)
                labels.append(label)
            cross = self._cross_loss(torch.stack(scores, 1), torch.stack(labels, 1))

        if self.cross_lambda == 0.0:
            total = temporal
        elif self.cross_lambda == 1.0:
            total = self.cross_scale * cross
        else:
            total = (1.0 - self.cross_lambda) * temporal + self.cross_lambda * self.cross_scale * cross
        if self.track_components:
            # None denotes an inactive objective, not an artificially zero error.
            self.last_components = {
                "temporal": None if temporal is None else float(temporal.detach().cpu()),
                "cross": None if cross is None else float(cross.detach().cpu()),
                "cross_scaled": None if cross is None else float((self.cross_scale * cross).detach().cpu()),
                "total": float(total.detach().cpu()),
                "H": int(pred.shape[1]), "K": 1, "n_assets": int(pred.shape[2]),
                "cross_construction": self.cross_construction,
                "rank_windows": list(windows), "n_rank_windows": len(windows),
                "cross_active": self.cross_lambda != 0.0,
            }
        return total

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        base_value: Optional[torch.Tensor] = None,
        *,
        momentum_history: Optional[torch.Tensor] = None,
        momentum_module=None,
    ) -> torch.Tensor:
        # Kept explicit for auditability: both losses feed the same backward pass.
        if pred.shape != target.shape:
            raise ValueError(f"Shapes diferentes: pred={pred.shape}; target={target.shape}.")
        if self.momentum_mode == "off":
            if momentum_history is not None or momentum_module is not None:
                raise ValueError("Contexto momentum fornecido com momentum_mode=off.")
        elif momentum_history is None or momentum_module is None:
            raise ValueError("Momentum habilitado exige momentum_history e momentum_module.")

        if self.cross_construction == "origin_anchored_windows":
            return self._forward_origin_windows(pred, target)

        temporal = self._temporal_loss(pred, target)
        pred_scores, target_scores = self.aggregator(
            pred,
            target,
            base_value=base_value,
        )
        momentum_result = None
        if self.momentum_mode == "residual_output":
            # The same training-only scaler restores both historical returns
            # and forecast/target block scores; future targets never enter m.
            history_raw = self.aggregator._maybe_inverse_norm(momentum_history)
            momentum_result = momentum_module(history_raw, pred_scores)
            pred_scores = momentum_result["final"]
        cross = self._cross_loss(pred_scores, target_scores)
        # An inactive objective must not contaminate a pure endpoint through
        # 0 * inf/nan (and lambda=1 does not depend on the temporal criterion).
        if self.cross_lambda == 0.0:
            total = temporal
        elif self.cross_lambda == 1.0:
            total = self.cross_scale * cross
        else:
            total = ((1.0 - self.cross_lambda) * temporal
                     + self.cross_lambda * self.cross_scale * cross)
        if momentum_result is not None:
            # Penalize residual amplitude outside lambda. Normalizing this
            # correction again would erase the amplitude being regularized.
            total = total + momentum_result["penalty"]
        self._remember(temporal, cross, total, pred_scores)
        if self.track_components:
            self.last_components.update(H=int(pred.shape[1]),
                                        cross_steps=int(pred_scores.shape[1] * self.trade_window),
                                        cross_tail_steps=int(pred.shape[1] % self.trade_window))
            if momentum_result is not None:
                self.last_components.update(
                    momentum_penalty=float(momentum_result["penalty"].detach()),
                    momentum_correction_energy=float(momentum_result["correction"].detach().square().mean()),
                    momentum_gain=float(momentum_module.gain.detach()),
                    momentum_lookback=int(momentum_module.lookback),
                )
        return total

