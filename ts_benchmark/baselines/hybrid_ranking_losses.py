"""Global and tail-directed mixtures on a single full-universe score vector.

Regional selection uses realized labels only, detached from autograd. Selecting
the first m=floor(N*fraction) assets shares the boundary mass equally at ties.
This differs deliberately from alphabetical tie breaking used in descriptive
tercile membership tables and preserves permutation invariance in training.

Regional ListNet predicts probabilities over the entire universe (or each
anchor's financial-kernel neighborhood), not just the selected tail. Regional
ListMLE retains the entire remaining risk set. Regional ApproxNDCG computes
the ranks of selected gains against all assets. Thus a globally misplaced tail
is penalized even when its internal ordering is correct.

Component reductions preserve their meanings: cross entropy for ListNet,
mean active stage for ListMLE, and 1-normalized DCG for ApproxNDCG. Equal mix
coefficients are equal objective coefficients, not equal gradient magnitudes.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ts_benchmark.baselines.global_ranking_losses import soft_ranks, target_midranks
from ts_benchmark.baselines.listwise_ranking_losses import (
    _inputs, _masked_log, _weights, approxndcg_loss, listmle_loss,
    listnet_proximity_loss,
)
from ts_benchmark.baselines.ranking_mixture import (
    RANK_MIX_COMPONENTS, validate_rank_mix_config,
)


def tail_selection(target: torch.Tensor, fraction=1.0 / 3.0) -> torch.Tensor:
    """Fractional top-tail membership; call with -target to select the bottom."""
    if target.ndim != 2:
        raise ValueError("Selecao regional exige target [B,N].")
    # Validate the public helper, including NaN and bool, without duplicating
    # the contract used by launcher, model factory and provenance.
    fraction = validate_rank_mix_config(fraction=fraction)["fraction"]
    labels = target.detach()
    n = labels.shape[1]
    if not n:
        return torch.zeros_like(labels)
    m = max(1, int(math.floor(n * fraction + 1e-12)))
    higher = (labels.unsqueeze(1) > labels.unsqueeze(2)).sum(dim=-1)
    equal = (labels.unsqueeze(1) == labels.unsqueeze(2)).sum(dim=-1)
    return ((m - higher).to(labels.dtype) / equal.clamp_min(1)).clamp(0, 1)


def _oriented(scores, labels, target_values, region):
    if region not in {"top", "bottom"}:
        raise ValueError("Regiao deve ser top ou bottom.")
    if region == "bottom":
        return -scores, -labels, -target_values
    return scores, labels, target_values


def regional_listnet_loss(scores, target, *, target_values=None, region="top",
                         fraction=1.0 / 3.0, pair_weights=None, tau=1.0):
    """Tail-target cross entropy with full-universe predicted denominators.

    Proximity averages anchor-conditional cross entropies over selected
    informative anchors. Each anchor keeps self mass one and all other
    candidates keep their original kernel weights. The target distribution
    includes only selected-tail mass; predicted probabilities retain all
    weighted candidates. Unit weights recover the unmasked loss exactly up to
    floating point reduction. A neighborhood without distinct labels is zero.
    """
    scores, target, tau = _inputs(scores, target, tau, "loss_listnet_tau")
    target_values = target if target_values is None else target_values.detach()
    if target_values.shape != target.shape:
        raise ValueError("target_values deve ter o mesmo formato de target.")
    scores, target, target_values = _oriented(scores, target, target_values, region)
    n = scores.shape[1]
    if n < 2:
        return scores.sum() * 0.0
    selection = tail_selection(target, fraction).to(scores.dtype)
    informative = target.amax(-1) > target.amin(-1)
    target_logits = target_values.to(scores.dtype) / tau + _masked_log(selection)
    if pair_weights is None:
        q = F.softmax(target_logits, dim=-1)
        values = -(q * F.log_softmax(scores / tau, dim=-1)).sum(-1)
        return torch.where(informative, values, scores.sum(-1) * 0).mean()
    weights = _weights(scores, pair_weights)
    eye = torch.eye(n, dtype=torch.bool, device=scores.device)[None]
    weights = weights.masked_fill(eye, 1)
    distinct = target.unsqueeze(1) != target.unsqueeze(2)
    active = ((weights > 0) & distinct).any(-1) & (selection > 0)
    # Nonselected anchors can have an empty target support under compact
    # kernels. Their self entry is a harmless dummy for finite softmax; such
    # rows have zero reduction mass and cannot affect scores or gradients.
    q_weights = weights * selection.unsqueeze(1)
    dummy = eye & ~(q_weights > 0).any(-1, keepdim=True)
    q_weights = torch.where(dummy, torch.ones_like(q_weights), q_weights)
    q = F.softmax(target_values.to(scores.dtype).unsqueeze(1) / tau
                  + _masked_log(q_weights), dim=-1)
    logp = F.log_softmax(scores.unsqueeze(1) / tau + _masked_log(weights), dim=-1)
    logp = torch.where(weights > 0, logp, torch.zeros_like(logp))
    values = -(q * logp).sum(-1)
    mass = selection * active.to(scores.dtype)
    per_row = ((values * mass).sum(-1)
               / mass.sum(-1).clamp_min(torch.finfo(scores.dtype).tiny))
    return torch.where(informative, per_row, scores.sum(-1) * 0).mean()


def regional_listmle_loss(scores, target, *, region="bottom", fraction=1.0 / 3.0,
                         pair_weights=None, tau=1.0):
    """Selected stages of ListMLE; opposite-tail assets stay in risk sets.

    For region=bottom, signs of scores and labels are both reversed. This is
    a new worst-first partial ListMLE, not the original full-list ListMLE.
    Equal labels never compete. Reduction is by selected informative stage
    mass and is independent of kernel sparsity, as in the existing ListMLE.
    """
    scores, target, tau = _inputs(scores, target, tau, "loss_listmle_tau")
    scores, target, _ = _oriented(scores, target, target, region)
    if scores.shape[1] < 2:
        return scores.sum() * 0.0
    selection = tail_selection(target, fraction).to(scores.dtype)
    weights = _weights(scores, pair_weights)
    lower = target.unsqueeze(1) < target.unsqueeze(2)
    mass = selection * lower.any(-1).to(scores.dtype)
    logits = ((scores.unsqueeze(1) - scores.unsqueeze(2)) / tau
              + _masked_log(weights.masked_fill(~lower, 0)))
    values = torch.logsumexp(torch.cat([torch.zeros_like(logits[..., :1]), logits], -1), -1)
    return ((values * mass).sum(-1) / mass.sum(-1).clamp_min(1)).mean()


def regional_approxndcg_loss(scores, target, *, region="top", fraction=1.0 / 3.0,
                            pair_weights=None, tau=1.0):
    """NDCG on realized-tail gains with soft ranks evaluated across all assets.

    Gains outside the selected tail are zero. This is a target-gain restriction,
    not a hard cutoff on predicted ranks. Kernel-weighted ranks follow the
    existing anchored-comparison Proximity definition and unit weights recover
    the plain variant. Bottom mode reverses scores and labels before ranking.
    """
    scores, target, tau = _inputs(scores, target, tau, "loss_approxndcg_tau")
    scores, target, _ = _oriented(scores, target, target, region)
    n = scores.shape[1]
    if n < 2:
        return scores.sum() * 0.0
    ranks = target_midranks(target).to(scores.dtype)
    selection = tail_selection(target, fraction).to(scores.dtype)
    gains = selection * torch.expm1((n - ranks) * (math.log(2) / (n - 1)))
    positions = torch.arange(1, n + 1, dtype=scores.dtype, device=scores.device)
    ideal = (gains.sort(-1, descending=True).values / torch.log2(1 + positions)).sum(-1)
    informative = target.amax(-1) > target.amin(-1)
    if pair_weights is None:
        predicted_ranks = soft_ranks(scores, tau)
    else:
        weights = _weights(scores, pair_weights, symmetric=True)
        eye = torch.eye(n, dtype=torch.bool, device=scores.device)[None]
        weights = weights.masked_fill(eye, 0)
        actual = ((target.unsqueeze(1) > target.unsqueeze(2)).to(scores.dtype)
                  + 0.5 * (target.unsqueeze(1) == target.unsqueeze(2)).to(scores.dtype))
        predicted = torch.sigmoid((scores.unsqueeze(1) - scores.unsqueeze(2)) / tau)
        predicted_ranks = ranks + (weights * (predicted - actual)).sum(-1)
        informative = informative & (((weights > 0)
                                     & (target.unsqueeze(1) != target.unsqueeze(2))
                                     & (gains > 0).unsqueeze(2)).any(-1).any(-1))
    dcg = (gains / torch.log2(1 + predicted_ranks.clamp_min(1))).sum(-1)
    values = 1 - dcg / ideal.clamp_min(torch.finfo(scores.dtype).tiny)
    return torch.where(informative, values, scores.sum(-1) * 0).mean()


def ranking_mixture_loss(scores, target, *, target_values=None, weights=None,
                         mode="global", regions=None, fraction=1.0 / 3.0,
                         pair_weights=None, listnet_tau=1.0, listmle_tau=1.0,
                         approxndcg_tau=1.0):
    """Weighted sum in fixed ListNet, ListMLE, ApproxNDCG component order.

    target contains original financial scores (ordering and mask semantics).
    target_values contains optionally normalized target scores for ListNet.
    Global mode calls the existing objectives without changing their tie or
    reduction conventions, including ListNet's uniform target at all ties.
    """
    config = validate_rank_mix_config(weights, mode, regions, fraction)
    if scores.ndim != 2 or target.shape != scores.shape:
        raise ValueError("Mixture exige scores e target com formato [B,N].")
    if scores.dtype in (torch.float16, torch.bfloat16):
        scores = scores.float()
    target_values = target.detach() if target_values is None else target_values.detach()
    if target_values.shape != target.shape:
        raise ValueError("target_values deve ter o mesmo formato de target.")
    total = scores.sum() * 0.0
    for name, coefficient, region in zip(RANK_MIX_COMPONENTS, config["weights"], config["regions"]):
        if coefficient == 0:
            continue
        if config["mode"] == "regional":
            kwargs = dict(region=region, fraction=config["fraction"], pair_weights=pair_weights)
            if name == "listnet":
                value = regional_listnet_loss(scores, target, target_values=target_values,
                                             tau=listnet_tau, **kwargs)
            elif name == "listmle":
                value = regional_listmle_loss(scores, target, tau=listmle_tau, **kwargs)
            else:
                value = regional_approxndcg_loss(scores, target, tau=approxndcg_tau, **kwargs)
        elif name == "listnet":
            if pair_weights is None:
                pred, labels, tau = _inputs(scores, target_values, listnet_tau, "loss_listnet_tau")
                value = -(F.softmax(labels.to(pred.dtype) / tau, -1)
                          * F.log_softmax(pred / tau, -1)).sum(-1).mean()
            else:
                value = listnet_proximity_loss(scores, target_values, pair_weights=pair_weights,
                                               tau=listnet_tau)
        elif name == "listmle":
            value = listmle_loss(scores, target, pair_weights=pair_weights, tau=listmle_tau)
        else:
            value = approxndcg_loss(scores, target, pair_weights=pair_weights, tau=approxndcg_tau)
        total = total + coefficient * value
    return total
