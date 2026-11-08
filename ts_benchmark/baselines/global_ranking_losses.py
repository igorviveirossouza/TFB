"""Differentiable full-list rank correlation, optionally weighted by pairs.

Predicted ranks use pairwise sigmoid comparisons (O(N^2)); they are not the
isotonic soft-rank operator. Realized labels use exact average ranks at ties.
The weighted objective is a normalized inner product of rank differences;
with equal pair weights it reduces algebraically to Pearson rank correlation.
"""
from __future__ import annotations

import math

import torch


def validate_spearman_tau(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("loss_spearman_tau deve ser finito e positivo.")
    return value


def soft_ranks(scores: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Descending soft ranks: 1 + sum_{j != i} sigmoid((s_j-s_i)/tau)."""
    tau = validate_spearman_tau(tau)
    comparisons = (scores.unsqueeze(1) - scores.unsqueeze(2)) / tau
    # Each self-comparison contributes exactly 1/2, removed here.
    return 0.5 + torch.sigmoid(comparisons).sum(dim=-1)


def target_midranks(target: torch.Tensor) -> torch.Tensor:
    """Descending exact ranks with average rank for equal realized returns."""
    labels = target.detach()
    higher = labels.unsqueeze(1) > labels.unsqueeze(2)
    equal = labels.unsqueeze(1) == labels.unsqueeze(2)
    # equal includes self; 1 + (equal_count - 1)/2 = (equal_count + 1)/2.
    return 0.5 + higher.sum(dim=-1).to(target.dtype) + 0.5 * equal.sum(dim=-1).to(target.dtype)


def spearman_loss(
    scores: torch.Tensor,
    target: torch.Tensor,
    *,
    tau: float = 1.0,
    pair_weights: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean 1-correlation across independent [N]-asset cross-sections.

pair_weights follows torch.triu_indices(N,N,1), one [P] row per section.
Weights and realized ranks are labels, detached from autograd. Sections with
no weighted target-rank variation (including all ties or an empty mask) carry
no ranking information and contribute differentiable zero. Constant predicted
ranks have correlation zero and retain a finite learning gradient. The epsilon
floor applies to squared norms *before* sqrt, avoiding its undefined derivative
at zero. Half precision inputs are promoted for stable reductions.
"""
    if scores.ndim != 2 or target.shape != scores.shape:
        raise ValueError("Spearman exige scores e target com mesmo formato [B,N].")
    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("Spearman exige eps finito e positivo.")
    tau = validate_spearman_tau(tau)
    n_assets = scores.shape[1]
    if n_assets < 2:
        return scores.sum() * 0.0
    if scores.dtype in (torch.float16, torch.bfloat16):
        scores = scores.float()
    predicted = soft_ranks(scores, tau)
    realized = target_midranks(target).to(scores.dtype)
    rows, cols = torch.triu_indices(n_assets, n_assets, 1, device=scores.device)
    predicted_diff = predicted[:, rows] - predicted[:, cols]
    realized_diff = realized[:, rows] - realized[:, cols]
    if pair_weights is None:
        weights = torch.ones_like(predicted_diff)
    else:
        if pair_weights.shape != predicted_diff.shape:
            raise ValueError("Spearman exige um peso por par i<j e por cross-section.")
        weights = pair_weights.detach().to(device=scores.device, dtype=scores.dtype)
    # Normalize only the weights, preserving both correlation and epsilon's
    # scale under a common rescaling of all weights within a section.
    weight_sum = weights.sum(dim=1, keepdim=True)
    weights = weights / weight_sum.clamp_min(torch.finfo(weights.dtype).tiny)
    covariance = (weights * predicted_diff * realized_diff).sum(dim=1)
    predicted_energy = (weights * predicted_diff.square()).sum(dim=1)
    target_energy = (weights * realized_diff.square()).sum(dim=1)
    denominator = (predicted_energy.clamp_min(eps * eps).sqrt()
                   * target_energy.clamp_min(eps * eps).sqrt())
    correlation = (covariance / denominator).clamp(-1.0, 1.0)
    loss = torch.where(target_energy > 0, 1.0 - correlation, predicted_energy * 0.0)
    return loss.mean()
