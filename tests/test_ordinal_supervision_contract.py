"""Audit the distinction between discrete supervision and continuous outputs.

These tests specify observable training semantics, not a new objective:
ordinal targets are classes, differentiable predictions are scores/logits,
and only the temporal component retains continuous-return supervision.
"""
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ts_benchmark.baselines import composite_trading_loss_v3 as loss_module
from ts_benchmark.baselines.composite_trading_loss_v3 import CompositeTradingLossV3
from ts_benchmark.baselines.ordinal_losses import (
    ORDINAL_LOSSES, OrderedThresholds, bucket_labels, ordinal_cross_entropy,
)


def objective(name, **kwargs):
    config = dict(trade_window=1, data_kind="log_return", score_kind="log_return",
        cross_loss=name, cross_lambda=1., inverse_norm=False,
        cross_score_normalization="zscore", bucket_count=3, hinge_margin=.1)
    config.update(kwargs)
    return CompositeTradingLossV3(**config).double()


@pytest.mark.parametrize("name", sorted(ORDINAL_LOSSES))
@pytest.mark.parametrize("protocol", ["blocks", "origin_anchored_windows"])
@pytest.mark.parametrize("score_kind", ["log_return", "simple_return"])
def test_same_classes_ignore_return_magnitudes_and_order_inside_each_class(name, protocol, score_kind):
    first = torch.tensor([[[.06, .04, .02, .01, -.01, -.02]]], dtype=torch.float64).repeat(1, 2, 1)
    # Both magnitudes and within-class ordering change, but class identities do not.
    second = torch.tensor([[[.5, 100., -1., .4, -1000., -5.]]], dtype=torch.float64).repeat(1, 2, 1)
    assert bucket_labels(first[:, 0], 3).tolist() == [[1, 1, 2, 2, 3, 3]]
    assert torch.equal(bucket_labels(first[:, 0], 3), bucket_labels(second[:, 0], 3))
    options = (dict(cross_construction=protocol, rank_windows=[1, 2])
               if protocol == "origin_anchored_windows" else dict(trade_window=2))
    criterion = objective(name, score_kind=score_kind, **options)
    seed = torch.tensor([[[.3, -.1, .4, -.8, .1, .2],
                          [-.2, .5, .1, .0, -.5, .8]]], dtype=torch.float64)
    values, gradients = [], []
    for actual in (first, second):
        prediction = seed.clone().requires_grad_()
        value = criterion(prediction, actual, target_economic=actual)
        gradients.append(torch.autograd.grad(value, prediction)[0])
        values.append(value.detach())
    torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
    assert gradients[0].abs().sum() > 0


@pytest.mark.parametrize("name", sorted(ORDINAL_LOSSES))
def test_labels_come_only_from_actual_economic_returns_not_prediction_or_model_space_target(name, monkeypatch):
    actual = torch.tensor([[[.03, .01, -.02, .04, 0., -.01]]], dtype=torch.float64)
    expected = torch.tensor([[1, 2, 3, 1, 2, 3]])
    observed = []
    def inspect_target(values, count):
        labels = bucket_labels(values, count)
        observed.append(labels)
        torch.testing.assert_close(values, actual[:, 0], rtol=0, atol=0)
        return labels
    monkeypatch.setattr(loss_module, "bucket_labels", inspect_target)
    criterion = objective(name)
    prediction = torch.tensor([[[1.2, .2, -1.2, .8, -.1, -.9]]], dtype=torch.float64)
    # A deliberately uninformative model-space target cannot replace raw labels.
    target_model_space = torch.zeros_like(actual)
    correct = criterion(prediction, target_model_space, target_economic=actual)
    inverted = criterion(-prediction, target_model_space, target_economic=actual)
    assert len(observed) == 2 and all(torch.equal(c, expected) for c in observed)
    assert inverted > correct


@pytest.mark.parametrize("name", sorted(ORDINAL_LOSSES))
def test_hard_decoding_before_loss_disconnects_network_gradients(name):
    actual = torch.tensor([[[.03, .01, -.02, .04, 0., -.01]]], dtype=torch.float64)
    output = torch.tensor([[[1.2, .2, -1.2, .8, -.1, -.9]]],
                          dtype=torch.float64, requires_grad=True)
    # A high class number means worse performance, so negate it for score orientation.
    hard_scores = -bucket_labels(output[:, 0], 3).to(output.dtype).unsqueeze(1)
    criterion = objective(name)
    value = criterion(hard_scores, actual, target_economic=actual)
    gradient = (torch.autograd.grad(value, output, allow_unused=True)[0]
                if value.requires_grad else None)
    assert gradient is None
    # OCE still has learnable cutpoints, but that does not reconnect the network.
    if name == "ordinal_ce":
        assert value.requires_grad


def test_oce_encodes_one_to_t_classes_as_ordered_cumulative_probabilities():
    head = OrderedThresholds(3).double()
    scores = torch.tensor([[2., 0., -2.]], dtype=torch.float64, requires_grad=True)
    classes = torch.tensor([[1, 2, 3]])
    z = torch.tensor([[[1., 1.], [0., 1.], [0., 0.]]], dtype=torch.float64)
    logits = (scores.unsqueeze(-1) - head()) / .7
    q = logits.sigmoid()
    probabilities = torch.cat((q[:, :, :1], q[:, :, 1:] - q[:, :, :-1], 1-q[:, :, -1:]), -1)
    torch.testing.assert_close(ordinal_cross_entropy(scores, classes, head, .7),
                               F.binary_cross_entropy_with_logits(logits, z), rtol=0, atol=0)
    assert torch.all(q[:, :, 1:] >= q[:, :, :-1])
    assert torch.all(probabilities >= 0)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones_like(scores))
    # Threshold decoding is an inference decision, not an input to the BCE.
    median_class = 1 + (q < .5).sum(-1)
    assert torch.equal(median_class, classes)
    loss = ordinal_cross_entropy(scores, classes, head, .7)
    score_gradient, first_cut_gradient = torch.autograd.grad(loss, (scores, head.first))
    assert score_gradient.abs().sum() > 0 and torch.isfinite(first_cut_gradient).all()


def test_continuous_return_target_is_retained_only_in_temporal_component():
    first = torch.tensor([[[.06, .04, .02, .01, -.01, -.02]]], dtype=torch.float64)
    second = first * 100
    prediction = torch.tensor([[[.03, -.01, .04, -.08, .01, .02]]], dtype=torch.float64)
    assert torch.equal(bucket_labels(first[:, 0], 3), bucket_labels(second[:, 0], 3))
    temporal = objective("bucket_ranknet", cross_lambda=0.)
    assert torch.equal(temporal(prediction, first, target_economic=first), F.mse_loss(prediction, first))
    assert temporal(prediction, first) != temporal(prediction, second)
    mixed = objective("bucket_ranknet", cross_lambda=.5)
    difference = (mixed(prediction, second, target_economic=second)
                  - mixed(prediction, first, target_economic=first))
    expected = .5 * (F.mse_loss(prediction, second) - F.mse_loss(prediction, first))
    torch.testing.assert_close(difference, expected, rtol=1e-12, atol=1e-12)
