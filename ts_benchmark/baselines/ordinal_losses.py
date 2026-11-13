"""Cross-sectional quantile targets and ordinal objectives.

Buckets are 1-based (1 is best). Ties are broken stably by input-channel
order, using rank0 * T // N + 1. This keeps group sizes within one asset
and makes the convention reproducible in the R evaluation.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

ORDINAL_LOSSES = frozenset({
    "ordinal_ce", "bucket_ranknet", "bucket_hinge", "bucket_listnet",
})
BUCKET_TIE_POLICY = "stable_channel_order"


def validate_bucket_count(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError("loss_bucket_count deve ser inteiro >= 2.")
    return value


def positive_temperature(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} deve ser finito e positivo.")
    return value


def bucket_labels(target: torch.Tensor, bucket_count: int) -> torch.Tensor:
    """Return deterministic balanced classes for each independent [N] row."""
    bucket_count = validate_bucket_count(bucket_count)
    if target.ndim != 2 or target.shape[1] < bucket_count:
        raise ValueError("Alvos devem ser [secoes, ativos], com 2 <= T <= N.")
    if not torch.isfinite(target).all():
        raise ValueError("Alvos ordinais devem ser finitos.")
    with torch.no_grad():
        order = torch.argsort(target.detach(), dim=1, descending=True, stable=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(1, order, torch.arange(target.shape[1], device=target.device)
                       .expand_as(order))
        return torch.div(ranks * bucket_count, target.shape[1], rounding_mode="floor") + 1


class OrderedThresholds(nn.Module):
    """Learned descending cutpoints; no random initialization or extra MLP."""

    def __init__(self, bucket_count: int):
        super().__init__()
        self.bucket_count = validate_bucket_count(bucket_count)
        # Start at standard-normal quantiles, independent of future outcomes.
        probabilities = 1 - torch.arange(1, bucket_count, dtype=torch.float32) / bucket_count
        initial = torch.distributions.Normal(0., 1.).icdf(probabilities)
        self.first = nn.Parameter(initial[:1].clone())
        gaps = initial[:-1] - initial[1:]
        self.raw_gaps = nn.Parameter(torch.log(torch.expm1(gaps)))

    def forward(self):
        gaps = F.softplus(self.raw_gaps) + 1e-6
        return torch.cat((self.first, self.first - gaps.cumsum(0)))


def attach_ordinal_parameters(model, criterion):
    """Include OCE cutpoints in the model optimizer and native checkpoints.

    The criterion and model reference the same small module. It never affects
    the forecasting forward pass. Pure temporal controls attach nothing.
    """
    head = getattr(criterion, "ordinal_thresholds", None)
    if head is None:
        return
    owner = model.module if isinstance(model, nn.DataParallel) else model
    parameter = next(model.parameters())
    head.to(device=parameter.device, dtype=parameter.dtype)
    owner.add_module("ordinal_thresholds", head)


def ordinal_cross_entropy(scores, classes, thresholds, temperature=1.):
    """BCE over 1[c <= m], with P(c <= m) = sigmoid((s-theta_m)/tau)."""
    temperature = positive_temperature(temperature, "loss_ordinal_tau")
    cuts = thresholds().to(device=scores.device, dtype=scores.dtype)
    labels = classes.unsqueeze(-1) <= torch.arange(
        1, cuts.numel() + 1, device=scores.device)
    logits = (scores.unsqueeze(-1) - cuts) / temperature
    return F.binary_cross_entropy_with_logits(logits, labels.to(scores.dtype))


def bucket_pairwise_loss(scores, classes, *, kind, temperature=1., margin=.1):
    """Equal-weight mean over oriented pairs c_i<c_j, then over rows."""
    if kind not in {"bucket_ranknet", "bucket_hinge"}:
        raise ValueError("Loss pairwise ordinal desconhecida.")
    temperature = positive_temperature(temperature, "loss_bucket_tau")
    if not math.isfinite(float(margin)) or float(margin) < 0:
        raise ValueError("loss_hinge_margin deve ser finita e nao negativa.")
    valid = classes.unsqueeze(2) < classes.unsqueeze(1)
    difference = scores.unsqueeze(2) - scores.unsqueeze(1)
    values = (F.softplus(-difference / temperature) if kind == "bucket_ranknet"
              else F.relu(float(margin) - difference))
    # A standalone call can contain one occupied class. Such a row contributes
    # differentiable zero, never NaN, and is not counted as an informative row.
    counts = valid.sum(dim=(1, 2))
    totals = torch.where(valid, values, torch.zeros_like(values)).sum(dim=(1, 2))
    active = counts > 0
    return (totals / counts.clamp_min(1)).sum() / active.sum().clamp_min(1)


def bucket_listnet_loss(scores, classes, bucket_count, *, temperature=1., target_temperature=1.):
    """ListNet with relevance T-c; members of a class have equal target mass."""
    temperature = positive_temperature(temperature, "loss_bucket_tau")
    target_temperature = positive_temperature(target_temperature, "loss_bucket_target_tau")
    relevance = (bucket_count - classes).to(scores.dtype)
    target = F.softmax(relevance / target_temperature, dim=1)
    return -(target * F.log_softmax(scores / temperature, dim=1)).sum(1).mean()
