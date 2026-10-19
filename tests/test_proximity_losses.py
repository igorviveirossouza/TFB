"""Numerical, gradient and compatibility contracts for proximity objectives."""
from __future__ import annotations

import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ts_benchmark.baselines.composite_trading_loss_v2 import CROSS_LOSSES
from ts_benchmark.baselines.composite_trading_loss_v3 import (
    CompositeTradingLossV3,
    build_loss,
)
from ts_benchmark.baselines.proximity_losses import (
    PROXIMITY_LOSSES,
    validate_cross_mask,
)


NAMES = tuple(PROXIMITY_LOSSES)


def criterion(name, mask=("uniform", 0.125), **kwargs):
    options = dict(
        trade_window=1,
        data_kind="log_return",
        cross_loss=name,
        cross_mask=mask,
        cross_score_normalization="none",
        inverse_norm=False,
        cross_lambda=1.0,
        hinge_margin=0.2,
        ranknet_alpha=1.7,
    )
    options.update(kwargs)
    return CompositeTradingLossV3(**options)


def oracle(pred, target, name, mask, normalization="none", delta=None):
    """Independent explicit pair loop, also retaining autograd for comparison.

    delta selects the legacy objective (one mean over all eligible pairs).
    Otherwise use a normalized weighted mean per section, then mean sections.
    """
    pred_rows = pred.reshape(-1, pred.shape[-1])
    target_rows = target.reshape(-1, target.shape[-1])
    section_losses, all_legacy_pairs = [], []
    for raw_p, raw_t in zip(pred_rows, target_rows):
        p, t = raw_p, raw_t
        if normalization == "zscore":
            p = (p - p.mean()) / p.std(unbiased=False).clamp_min(1e-8)
            t = (t - t.mean()) / t.std(unbiased=False).clamp_min(1e-8)
        terms, weights = [], []
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                gap = float(abs(raw_t[i] - raw_t[j]))
                if delta is not None and gap <= delta:
                    continue
                weight = (
                    float(gap <= mask[1])
                    if mask[0] == "uniform"
                    else math.exp(-gap * gap / (2 * mask[1] ** 2))
                )
                sign = float(torch.sign(raw_t[i] - raw_t[j]))
                difference = p[i] - p[j]
                base_name = PROXIMITY_LOSSES.get(name, name)
                if base_name == "hinge":
                    term = (
                        abs(difference)
                        if sign == 0
                        else torch.clamp(0.2 - sign * difference, min=0)
                    )
                elif base_name == "ranknet":
                    x = 1.7 * difference
                    if sign == 0:
                        # Symmetric two-class cross entropy for target 1/2.
                        term = (F.softplus(x) + F.softplus(-x)) / 2
                    else:
                        term = F.softplus(-sign * x)
                else:
                    term = (difference - (t[i] - t[j])) ** 2
                all_legacy_pairs.append(term)
                terms.append(weight * term)
                weights.append(weight)
        if sum(weights):
            section_losses.append(sum(terms) / sum(weights))
        else:
            section_losses.append(raw_p.sum() * 0)
    if delta is not None:
        return sum(all_legacy_pairs) / len(all_legacy_pairs) if all_legacy_pairs else pred.sum() * 0
    return torch.stack(section_losses).mean()


@pytest.mark.parametrize("value, expected", [
    (["uniform", 0], ("uniform", 0.0)),
    ((" Gaussian ", 0.025), ("gaussian", 0.025)),
    (["uniform", 2], ("uniform", 2.0)),
])
def test_mask_validation_canonicalizes(value, expected):
    assert validate_cross_mask(value) == expected


@pytest.mark.parametrize("value", [
    None, "uniform", [], ["uniform"], ["uniform", 1, 2],
    [1, 0.1], ["gauss", 0.1], ["uniform", True], ["gaussian", False],
    ["uniform", "0.1"], ["uniform", None], ["uniform", float("nan")],
    ["gaussian", float("inf")], ["uniform", -0.1], ["gaussian", 0],
    ["uniform", 10 ** 400], {"uniform": 0.1},
])
def test_mask_validation_rejects_malformed(value):
    with pytest.raises(ValueError, match="loss_cross_mask"):
        validate_cross_mask(value)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("mask", [("uniform", 0.125), ("gaussian", 0.25)])
@pytest.mark.parametrize("normalization", ["none", "zscore"])
def test_values_and_gradients_match_explicit_pair_oracle(name, mask, normalization):
    p = torch.tensor([
        [[0.4, 0.1, -0.3, 0.8], [0.3, 0.0, -0.1, 0.9]],
        [[0.5, 0.1, 0.0, -0.4], [0.2, 0.4, -0.5, 0.7]],
    ], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([
        [[0.125, 0.125, 0.25, 0.75], [0.0, 0.5, 1.0, 2.0]],
        [[0.0, 0.0625, 0.125, 0.5], [0.1, 0.1, 0.1, 0.1]],
    ], dtype=torch.float64)
    actual = criterion(name, mask, cross_score_normalization=normalization)._cross_loss(p, t)
    expected = oracle(p, t, name, mask, normalization)
    torch.testing.assert_close(actual, expected)
    actual_grad, = torch.autograd.grad(actual, p, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected, p)
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("name", NAMES)
def test_ties_have_corrective_gradient_and_minimum_at_equal_scores(name):
    p = torch.tensor([[[0.4, -0.2]]], dtype=torch.float64, requires_grad=True)
    t = torch.zeros_like(p)
    loss = criterion(name, ("uniform", 0))._cross_loss(p, t)
    loss.backward()
    assert p.grad[0, 0, 0] > 0
    assert p.grad[0, 0, 1] < 0
    equal_p = torch.zeros_like(p, requires_grad=True)
    minimum = criterion(name, ("uniform", 0))._cross_loss(equal_p, t)
    assert loss.item() > minimum.item()
    expected_minimum = math.log(2) if name == "ranknet_proximity" else 0.0
    assert minimum.item() == pytest.approx(expected_minimum)
    minimum.backward()
    torch.testing.assert_close(equal_p.grad, torch.zeros_like(equal_p))


@pytest.mark.parametrize("name", NAMES)
def test_tied_pair_attraction_with_actual_zscore_and_distinct_third_asset(name):
    p = torch.tensor([[[0.4, -0.2, 2.0]]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float64)
    fn = criterion(name, ("uniform", 0.1), cross_score_normalization="zscore")
    loss = fn._cross_loss(p, t)
    loss.backward()
    assert p.grad[0, 0, 0] > 0
    assert p.grad[0, 0, 1] < 0
    updated = p.detach() - 0.01 * p.grad
    assert fn._cross_loss(updated, t).item() < loss.item()
    assert abs(updated[0, 0, 0] - updated[0, 0, 1]) < abs(p[0, 0, 0] - p[0, 0, 1])


@pytest.mark.parametrize("name", NAMES)
def test_near_ties_are_not_relabelled_as_ties(name):
    p = torch.tensor([[[0.0, 0.0]]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([[[0.0, 1e-12]]], dtype=torch.float64)
    loss = criterion(name, ("uniform", 1e-10))._cross_loss(p, t)
    loss.backward()
    assert p.grad[0, 0, 0] > 0
    assert p.grad[0, 0, 1] < 0
    assert criterion(name, ("uniform", 0))._cross_loss(p, t).item() == 0.0


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("n_assets", [1, 3])
def test_no_pairs_returns_differentiable_zero(name, n_assets):
    p = torch.arange(n_assets, dtype=torch.float64).reshape(1, 1, -1).requires_grad_()
    t = p.detach() * 2
    loss = criterion(name, ("uniform", 0.1))._cross_loss(p, t)
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(p.grad, torch.zeros_like(p))


def test_gaussian_single_asset_returns_differentiable_zero():
    p = torch.tensor([[[0.5]]], requires_grad=True)
    loss = criterion("ranknet_proximity", ("gaussian", 0.1))._cross_loss(p, p.detach())
    loss.backward()
    assert loss.item() == 0
    assert p.grad.item() == 0


@pytest.mark.parametrize("name", NAMES)
def test_uniform_threshold_is_inclusive_and_in_raw_financial_units(name):
    # Only the first two assets lie within a=0.125 before z-scoring.
    p = torch.tensor([[[0.3, -0.1, 0.2]]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([[[0.0, 0.125, 0.5]]], dtype=torch.float64)
    actual = criterion(name, ("uniform", 0.125), cross_score_normalization="zscore")._cross_loss(p, t)
    expected = oracle(p, t, name, ("uniform", 0.125), "zscore")
    torch.testing.assert_close(actual, expected)
    assert actual.item() > 0


@pytest.mark.parametrize("name", NAMES)
def test_delta_does_not_affect_new_objectives(name):
    p = torch.tensor([[[0.5, -0.1, 0.2]]])
    t = torch.tensor([[[0.0, 0.01, 0.0]]])
    a = criterion(name, cross_delta=0)._cross_loss(p, t)
    b = criterion(name, cross_delta=100)._cross_loss(p, t)
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("bandwidth", [1e-100, 1e-300, 5e-324])
def test_gaussian_does_not_silently_zero_when_all_absolute_weights_underflow(name, bandwidth):
    p = torch.tensor([[[0.2, 0.0, 0.6]]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([[[0.0, 1.0, 3.0]]], dtype=torch.float64)
    actual = criterion(name, ("gaussian", bandwidth))._cross_loss(p, t)
    # In this limit only the closest pair (0, 1) carries normalized weight.
    expected = criterion(name, ("uniform", 1))._cross_loss(p, t)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(p.grad).all()


def test_gaussian_extreme_float32_financial_scales_stay_finite():
    p = torch.tensor([[[0.2, 0.0, 0.6]]], requires_grad=True)
    t = torch.tensor([[[-3e38, 0.0, 3e38]]])
    actual = criterion("ranknet_proximity", ("gaussian", 0.1))._cross_loss(p, t)
    expected = (F.softplus(1.7 * (p[0, 0, 0] - p[0, 0, 1]))
                + F.softplus(1.7 * (p[0, 0, 1] - p[0, 0, 2]))) / 2
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize("name", NAMES)
def test_blocks_are_averaged_equally_even_with_empty_or_differently_sized_masks(name):
    p = torch.tensor([[[0.2, 0.1, 0.4], [0.0, 0.2, 0.8], [0.1, 0.5, -0.5]]], dtype=torch.float64)
    t = torch.tensor([[[0.0, 0.0, 0.1], [0.0, 0.0, 1.0], [0.0, 1.0, 2.0]]], dtype=torch.float64)
    loss_fn = criterion(name, ("uniform", 0.15))
    expected = sum(loss_fn._cross_loss(p[:, j:j + 1], t[:, j:j + 1]) for j in range(3)) / 3
    torch.testing.assert_close(loss_fn._cross_loss(p, t), expected)


@pytest.mark.parametrize("name", ["hinge", "ranknet", "pairwise_mse"])
@pytest.mark.parametrize("normalization", ["none", "zscore"])
@pytest.mark.parametrize("delta", [0.0, 0.125, 100])
def test_original_loss_values_gradients_and_tie_exclusion_are_unchanged(name, normalization, delta):
    p = torch.tensor([[[0.3, -0.5, 0.1], [0.5, 0.2, -0.1]]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([[[0.0, 0.0, 0.5], [0.0, 0.125, 0.25]]], dtype=torch.float64)
    # An invalid mask is deliberately ignored by original losses.
    actual = criterion(name, mask="ignored", cross_delta=delta, cross_score_normalization=normalization)._cross_loss(p, t)
    expected = oracle(p, t, name, ("uniform", 1), normalization, delta=delta)
    torch.testing.assert_close(actual, expected)
    actual_grad, = torch.autograd.grad(actual, p, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected, p)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_v2_registry_is_not_modified():
    assert not set(PROXIMITY_LOSSES).intersection(CROSS_LOSSES)


@pytest.mark.parametrize("name", NAMES)
def test_factory_requires_and_passes_mask(name):
    config = SimpleNamespace(loss="composite_trading", loss_trade_window=1, loss_cross=name)
    with pytest.raises(ValueError, match="loss_cross_mask"):
        build_loss(config)
    config.loss_cross_mask = ["gaussian", 0.05]
    built = build_loss(config)
    assert built.cross_loss_name == name
    assert built.cross_mask == ("gaussian", 0.05)


@pytest.mark.parametrize("name", NAMES)
def test_forward_aggregates_all_blocks_after_inverse_normalization(name):
    # Exact binary mean/scale keep the independent reconstruction exact.
    mean, scale = [0.0, 0.125, -0.125], [0.25, 0.5, 0.25]
    p = torch.tensor([[[0.1, 0.2, 0.4], [-0.1, 0.0, 0.2],
                       [0.2, -0.3, 0.0], [0.1, 0.3, -0.2]]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([[[0.0, 0.1, 0.4], [0.2, -0.1, 0.2],
                       [0.3, -0.2, 0.1], [0.1, 0.2, -0.1]]], dtype=torch.float64)
    mask = ("gaussian", 0.2)
    fn = criterion(name, mask, trade_window=2, inverse_norm=True,
                   normalizer_mean=mean, normalizer_scale=scale,
                   cross_score_normalization="zscore", cross_lambda=0.4,
                   cross_scale=1.3, track_components=True)
    actual = fn(p, t)
    p_raw = p * torch.tensor(scale) + torch.tensor(mean)
    t_raw = t * torch.tensor(scale) + torch.tensor(mean)
    p_scores = torch.exp(p_raw.reshape(1, 2, 2, 3).sum(dim=2)) - 1
    t_scores = torch.exp(t_raw.reshape(1, 2, 2, 3).sum(dim=2)) - 1
    expected = 0.6 * (p - t).square().mean() + 0.4 * 1.3 * oracle(p_scores, t_scores, name, mask, "zscore")
    torch.testing.assert_close(actual, expected)
    actual_grad, = torch.autograd.grad(actual, p, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected, p)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert fn.last_components["cross_steps"] == 4
    assert fn.last_components["cross_tail_steps"] == 0
    assert fn.last_components["n_blocks"] == 2
    with pytest.raises(ValueError, match="divisor exato"):
        fn(p[:, :3], t[:, :3])
