"""Listwise objectives and financial-gap Proximity adaptations.

All matrices are cross-sectional: [B,N,N] indexes anchor i and candidate j.
Weights depend only on detached realized returns. Unlike the pairwise weighted
means, these objectives require the actual kernel in [0,1], not rescaled values.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ts_benchmark.baselines.global_ranking_losses import soft_ranks, target_midranks
from ts_benchmark.baselines.proximity_losses import kernel_parameters, validate_cross_mask


def validate_temperature(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name + " deve ser finito e positivo.")
    return value


def unscaled_proximity_weights(gap, mask, dtype=None):
    """Actual kernel values, including aliases and complementary tie-zero rule.

    Compute in float64 and log space; underflow to zero is intentional here.
    Relative rescaling would change ListMLE risk sets and ApproxNDCG ranks.
    """
    mask = validate_cross_mask(mask)
    gap = gap.detach().to(torch.float64)
    name, bandwidth = mask[:2]
    mode = mask[2] if len(mask) >= 3 else "natural"
    family, shape = kernel_parameters(mask)
    if family == "compact" and shape[1] == 0:
        weights = (gap <= bandwidth if mode == "natural" else gap > bandwidth).to(torch.float64)
    else:
        log_u = gap.log() - math.log(bandwidth)
        if family == "student":
            nu = shape[0]
            log_natural = -((nu + 1.0) / 2.0) * F.softplus(2 * log_u - math.log(nu))
        elif family == "generalized_gaussian":
            p = shape[0]
            log_natural = -torch.exp(p * log_u - math.log(p))
        elif family == "compact":
            p, q = shape
            inside = gap < bandwidth
            log_power = (p * log_u).clamp_max(0)
            # log(1-exp(x)): log1p retains tiny complements near the centre;
            # expm1 retains precision close to the compact support boundary.
            log_one_minus = torch.where(
                log_power < -math.log(2), torch.log1p(-log_power.exp()),
                torch.log(-torch.expm1(log_power)))
            log_natural = torch.where(
                inside, q * log_one_minus,
                torch.full_like(gap, -torch.inf))
        elif family == "cosine":
            inside = gap < bandwidth
            u = torch.exp(log_u.clamp_max(0))
            log_natural = torch.where(inside, torch.cos(math.pi * u / 2).log(),
                                      torch.full_like(gap, -torch.inf))
        elif family == "intermediate_band":
            p = shape[0]
            x = log_u
            series = x.square() * (0.5 + x / 6 + x.square() / 24 + x.pow(3) / 120)
            phi = torch.where(x.abs() < 1e-3, series, torch.expm1(x) - x).clamp_min(0)
            phi = torch.where(torch.isfinite(x), phi, torch.full_like(phi, torch.inf))
            log_natural = -p * phi
        else:
            raise ValueError("Familia de kernel desconhecida: " + family)
        if mode == "natural":
            weights = log_natural.exp()
        elif family == "cosine":
            # 1-cos(x)=2*sin(x/2)^2 keeps small, representable complements.
            weights = torch.where(inside, 2 * torch.sin(math.pi * u / 4).square(),
                                  torch.ones_like(gap))
        else:
            weights = -torch.expm1(log_natural)
        if mode == "complementary":
            weights = torch.where(gap > 0, weights, torch.zeros_like(weights))
    return weights.clamp(0, 1).to(dtype or torch.float64)


def financial_pair_weights(target, mask, dtype):
    realized = target.detach().to(torch.float64)
    gap = (realized.unsqueeze(2) - realized.unsqueeze(1)).abs()
    weights = unscaled_proximity_weights(gap, mask, dtype)
    # The financial gap and kernel are mathematically symmetric. Vectorized
    # transcendental kernels can nevertheless round equal ij/ji arguments
    # differently in distinct SIMD lanes, especially float64 tail lanes. Use
    # one canonical value per unordered pair, preserving raw kernel scale and
    # exact symmetry required by the anchored ApproxNDCG rank construction.
    upper = torch.triu(weights, diagonal=1)
    return upper + upper.transpose(1, 2)


def _inputs(scores, target, tau, name):
    if scores.ndim != 2 or scores.shape != target.shape:
        raise ValueError("Loss listwise exige scores e target com mesmo formato [B,N].")
    tau = validate_temperature(tau, name)
    if scores.dtype in (torch.float16, torch.bfloat16):
        scores = scores.float()
    return scores, target.detach(), tau


def _weights(scores, pair_weights, *, symmetric=False):
    batch, n = scores.shape
    if pair_weights is None:
        return scores.new_ones((batch, n, n))
    if pair_weights.shape != (batch, n, n):
        raise ValueError("Pesos listwise exigem formato [B,N,N].")
    weights = pair_weights.detach().to(device=scores.device, dtype=scores.dtype)
    if not bool(torch.isfinite(weights).all()) or bool(((weights < 0) | (weights > 1)).any()):
        raise ValueError("Pesos listwise devem ser finitos em [0,1].")
    if symmetric and not torch.equal(weights, weights.transpose(1, 2)):
        raise ValueError("ApproxNDCG exige pesos simetricos.")
    return weights


def _masked_log(weights):
    positive = weights > 0
    safe = torch.where(positive, weights, torch.ones_like(weights))
    return safe.log().masked_fill(~positive, -torch.inf)


def listnet_proximity_loss(scores, target, *, pair_weights, tau=1.0):
    """Mean anchor-conditional ListNet cross entropy; singleton masks yield zero.

    Each anchor is retained with weight one. All other entries use the actual
    kernel, in both target and predicted probabilities, normalized separately.
    Full unit off-diagonal weights recover ordinary top-one ListNet.
    """
    scores, target, tau = _inputs(scores, target, tau, "loss_listnet_tau")
    n = scores.shape[1]
    if n < 2:
        return scores.sum() * 0
    weights = _weights(scores, pair_weights)
    eye = torch.eye(n, device=scores.device, dtype=torch.bool)[None]
    active = (weights.masked_fill(eye, 0) > 0).any(dim=-1)
    weights = weights.masked_fill(eye, 1)
    log_weights = _masked_log(weights)
    target_logits = target.to(scores.dtype).unsqueeze(1) / tau + log_weights
    predicted_logits = scores.unsqueeze(1) / tau + log_weights
    true_probability = F.softmax(target_logits, dim=-1)
    predicted_log_probability = F.log_softmax(predicted_logits, dim=-1)
    predicted_log_probability = torch.where(weights > 0, predicted_log_probability,
                                           torch.zeros_like(predicted_log_probability))
    values = -(true_probability * predicted_log_probability).sum(dim=-1)
    values = torch.where(active, values, torch.zeros_like(values))
    return (values.sum(dim=-1) / active.sum(dim=-1).clamp_min(1)).mean()


def listmle_loss(scores, target, *, pair_weights=None, tau=1.0):
    """Mean ListMLE stage loss; equal-return assets never compete with each other.

    Each i competes with strictly lower realized returns, and its own risk-set
    mass is one. With distinct labels and unit weights this is standard ListMLE
    divided by N-1; tie handling is a permutation-invariant extension. Reduction
    uses the number of strict-lower stages, independently of the kernel.
    """
    scores, target, tau = _inputs(scores, target, tau, "loss_listmle_tau")
    n = scores.shape[1]
    if n < 2:
        return scores.sum() * 0
    weights = _weights(scores, pair_weights)
    lower = target.unsqueeze(1) < target.unsqueeze(2)
    active = lower.any(dim=-1)
    logits = (scores.unsqueeze(1) - scores.unsqueeze(2)) / tau
    logits = logits + _masked_log(weights.masked_fill(~lower, 0))
    # The fixed self term exp(0) makes empty risk sets safe and exactly zero.
    values = torch.logsumexp(torch.cat([torch.zeros_like(logits[..., :1]), logits], dim=-1), dim=-1)
    values = torch.where(active, values, torch.zeros_like(values))
    return (values.sum(dim=-1) / active.sum(dim=-1).clamp_min(1)).mean()


def approxndcg_loss(scores, target, *, pair_weights=None, tau=1.0):
    """ApproxNDCG with nonnegative percentile-rank gains for signed returns.

    Proximity fixes inactive pair comparisons at their realized ordering:
        r_i = r_i_true + sum_j w_ij * (sigmoid((s_j-s_i)/tau) - c_ji_true).
    Symmetric weights in [0,1] preserve feasible expected ranks. IDCG remains
    the conventional ideal discounted gain, and all unit weights recover the
    ordinary soft-rank approximation. This is a training loss, not an IC metric.
    """
    scores, target, tau = _inputs(scores, target, tau, "loss_approxndcg_tau")
    n = scores.shape[1]
    if n < 2:
        return scores.sum() * 0
    true_ranks = target_midranks(target).to(scores.dtype)
    relevance = (n - true_ranks) / (n - 1)
    gains = torch.expm1(relevance * math.log(2))
    positions = torch.arange(1, n + 1, device=scores.device, dtype=scores.dtype)
    ideal = (gains.sort(dim=-1, descending=True).values / torch.log2(1 + positions)).sum(dim=-1)
    informative = target.amax(dim=-1) > target.amin(dim=-1)
    if pair_weights is None:
        predicted_ranks = soft_ranks(scores, tau)
    else:
        weights = _weights(scores, pair_weights, symmetric=True)
        eye = torch.eye(n, device=scores.device, dtype=torch.bool)[None]
        weights = weights.masked_fill(eye, 0)
        actual = ((target.unsqueeze(1) > target.unsqueeze(2)).to(scores.dtype)
                  + 0.5 * (target.unsqueeze(1) == target.unsqueeze(2)).to(scores.dtype))
        predicted = torch.sigmoid((scores.unsqueeze(1) - scores.unsqueeze(2)) / tau)
        predicted_ranks = true_ranks + (weights * (predicted - actual)).sum(dim=-1)
        distinct = target.unsqueeze(1) != target.unsqueeze(2)
        informative = informative & ((weights > 0) & distinct).any(dim=-1).any(dim=-1)
    # Rounding can place a mathematically feasible boundary just below rank one.
    predicted_ranks = predicted_ranks.clamp_min(1)
    dcg = (gains / torch.log2(1 + predicted_ranks)).sum(dim=-1)
    values = 1 - dcg / ideal.clamp_min(torch.finfo(scores.dtype).tiny)
    values = torch.where(informative, values, scores.sum(dim=-1) * 0)
    return values.mean()

