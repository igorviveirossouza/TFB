"""Numeric contracts for discrete cross-sectional targets and all four losses."""
from pathlib import Path
import math
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ts_benchmark.baselines.ordinal_losses import (
    ORDINAL_LOSSES, OrderedThresholds, bucket_labels, ordinal_cross_entropy,
    bucket_pairwise_loss, bucket_listnet_loss,
)
from ts_benchmark.baselines.composite_trading_loss_v3 import CompositeTradingLossV3, build_loss
from ts_benchmark.baselines import composite_trading_loss_v3 as loss_module
from ts_benchmark.baselines.utils import forecasting_data_provider, DatasetForTransformer


def criterion(name, **kwargs):
    options = dict(trade_window=1, data_kind="log_return", score_kind="log_return",
                   cross_loss=name, cross_lambda=1., inverse_norm=False,
                   cross_score_normalization="none", bucket_count=3)
    options.update(kwargs)
    return CompositeTradingLossV3(**options)


def test_bucket_balance_direction_and_stable_ties():
    labels = bucket_labels(torch.tensor([[3., 1., 1., 2., -1., 8., 0., -2.]]), 3)
    assert labels.tolist() == [[1, 2, 2, 1, 3, 1, 2, 3]]
    for n, t in [(55, 3), (55, 10), (6, 6)]:
        labels = bucket_labels(torch.zeros(2, n), t)
        expected = torch.arange(n) * t // n + 1
        assert torch.equal(labels[0], expected)
        counts = torch.bincount(labels[0])[1:]
        assert counts.max() - counts.min() <= 1


@pytest.mark.parametrize("count", [1, 0, -1, True, 2.5])
def test_invalid_counts(count):
    with pytest.raises(ValueError, match="loss_bucket_count"):
        criterion("ordinal_ce", bucket_count=count)


def test_no_missing_assets_or_nonfinite_targets_silently_bucketed():
    with pytest.raises(ValueError, match="T <= N"):
        bucket_labels(torch.ones(1, 2), 3)
    with pytest.raises(ValueError, match="finitos"):
        bucket_labels(torch.tensor([[1., float("nan"), 0.]]), 3)


def test_pairwise_uses_cross_bucket_pairs_and_correct_orientation():
    scores = torch.tensor([[2., 1., -1., -2.]], dtype=torch.float64, requires_grad=True)
    classes = torch.tensor([[1, 1, 2, 2]])
    expected = sum(math.log1p(math.exp(-d / .7)) for d in [3., 4., 2., 3.]) / 4
    loss = bucket_pairwise_loss(scores, classes, kind="bucket_ranknet", temperature=.7)
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert (scores.grad[0, :2] < 0).all() and (scores.grad[0, 2:] > 0).all()
    swapped = scores.detach()[:, [1, 0, 3, 2]]
    assert bucket_pairwise_loss(swapped, classes, kind="bucket_ranknet", temperature=.7) == loss.detach()
    assert bucket_pairwise_loss(scores.detach(), classes, kind="bucket_hinge", margin=2.5).item() == .125
    assert bucket_pairwise_loss(-scores.detach(), classes, kind="bucket_hinge", margin=2.5).item() > 2.5


def test_empty_pair_set_is_finite_differentiable_zero():
    scores = torch.randn(1, 4, requires_grad=True)
    loss = bucket_pairwise_loss(scores, torch.ones(1, 4), kind="bucket_ranknet")
    loss.backward()
    assert loss.item() == 0 and torch.equal(scores.grad, torch.zeros_like(scores))


def test_oce_matches_cumulative_binary_labels_and_ordered_probabilities():
    head = OrderedThresholds(3).double()
    scores = torch.tensor([[1., 0., -1.]], dtype=torch.float64, requires_grad=True)
    classes = torch.tensor([[1, 2, 3]])
    labels = torch.tensor([[[1., 1.], [0., 1.], [0., 0.]]], dtype=torch.float64)
    logits = (scores.unsqueeze(-1) - head()) / .8
    expected = F.binary_cross_entropy_with_logits(logits, labels)
    loss = ordinal_cross_entropy(scores, classes, head, .8)
    assert torch.equal(loss, expected)
    assert (logits.sigmoid()[..., 0] <= logits.sigmoid()[..., 1]).all()
    loss.backward()
    assert scores.grad[0, 0] < 0 and scores.grad[0, -1] > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    with torch.no_grad():
        head.raw_gaps.fill_(-100)
    assert (head()[:-1] > head()[1:]).all()


def test_listnet_uses_bucket_relevance_and_two_temperatures():
    scores = torch.tensor([[3., 1., -2., -3.]], requires_grad=True)
    classes = torch.tensor([[1, 1, 2, 3]])
    p = torch.softmax(torch.tensor([[2., 2., 1., 0.]]) / .6, 1)
    expected = -(p * torch.log_softmax(scores / .9, 1)).sum()
    actual = bucket_listnet_loss(scores, classes, 3, temperature=.9, target_temperature=.6)
    assert torch.equal(actual, expected)
    actual.backward()
    assert torch.isfinite(scores.grad).all()


@pytest.mark.parametrize("name", sorted(ORDINAL_LOSSES))
@pytest.mark.parametrize("construction", ["blocks", "origin_anchored_windows"])
def test_mixture_endpoints_and_temporal_control_ignore_inactive_objective(name, construction, monkeypatch):
    kwargs = (dict(cross_construction=construction, rank_windows=[1, 2])
              if construction == "origin_anchored_windows" else {})
    prediction = torch.tensor([[[.1, .2, -.2], [.3, .1, -.1]]], requires_grad=True)
    target = torch.tensor([[[.2, -.1, .1], [.1, .4, -.2]]])
    pure_temporal = criterion(name, cross_lambda=0, **kwargs)
    monkeypatch.setattr(pure_temporal, "_cross_loss", lambda *args: pytest.fail("inactive cross evaluated"))
    assert torch.equal(pure_temporal(prediction, target), F.mse_loss(prediction, target))
    assert not list(pure_temporal.parameters())
    pure_ordinal = criterion(name, cross_lambda=1, **kwargs)
    monkeypatch.setattr(pure_ordinal, "_temporal_loss", lambda *args: pytest.fail("inactive temporal evaluated"))
    cross = pure_ordinal(prediction, target)
    combined = criterion(name, cross_lambda=.4, cross_scale=1.7, **kwargs)
    assert combined(prediction, target).item() == pytest.approx(
        (.6 * F.mse_loss(prediction, target) + .4 * 1.7 * cross).detach().item())


@pytest.mark.parametrize("name", sorted(ORDINAL_LOSSES))
def test_aggregation_restores_economic_scale_before_classes(name):
    prediction = torch.tensor([[[.2, -.1, .3], [.4, .1, -.2]]])
    target = torch.tensor([[[.3, -.2, .1], [.2, .4, -.1]]])
    mean = torch.tensor([.5, -1., 2.]); scale = torch.tensor([2., 4., .3])
    model_pred = (prediction - mean) / scale
    model_target = (target - mean) / scale
    raw = criterion(name, trade_window=2)
    normalized = criterion(name, trade_window=2, inverse_norm=True,
                           normalizer_mean=mean, normalizer_scale=scale)
    assert normalized(model_pred, model_target).item() == pytest.approx(raw(prediction, target).item(), rel=1e-5)


@pytest.mark.parametrize("field", ["ordinal_tau", "bucket_tau", "bucket_target_tau"])
@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf")])
def test_temperature_validation(field, value):
    with pytest.raises(ValueError, match="finito e positivo"):
        criterion("ordinal_ce", **{field: value})


def test_factory_forwards_all_ordinal_options():
    config = SimpleNamespace(loss="composite_trading", loss_cross="bucket_listnet", loss_trade_window=1,
        loss_bucket_count=5, loss_ordinal_tau=.7, loss_bucket_tau=.8, loss_bucket_target_tau=.9)
    result = build_loss(config)
    assert (result.bucket_count, result.ordinal_tau, result.bucket_tau, result.bucket_target_tau) == (5, .7, .8, .9)


@pytest.mark.parametrize("name", sorted(ORDINAL_LOSSES))
@pytest.mark.parametrize("construction", ["blocks", "origin_anchored_windows"])
@pytest.mark.parametrize("score_kind", ["simple_return", "log_return"])
def test_raw_economic_targets_preserve_exact_ties_after_normalization(name, construction, score_kind, monkeypatch):
    rng = np.random.default_rng(4)
    mean = rng.normal(0, .001, 6)
    scale = rng.uniform(.01, .04, 6)
    economic = torch.zeros((1, 2, 6), dtype=torch.float64, requires_grad=True)
    model_target = torch.tensor((economic.detach().numpy() - mean) / scale, dtype=torch.float32)
    prediction = torch.tensor([[[.1, .4, -.3, .7, -.8, .5]]], requires_grad=True).expand(1, 2, 6)
    kwargs = dict(trade_window=2) if construction == "blocks" else dict(
        cross_construction=construction, rank_windows=[1, 2])
    objective = criterion(name, score_kind=score_kind, inverse_norm=True,
        normalizer_mean=mean, normalizer_scale=scale, **kwargs)
    # Reproducer: a float32 inverse transform spuriously splits an exact tie.
    restored = objective.aggregator._maybe_inverse_norm(model_target)
    assert not torch.equal(restored, torch.zeros_like(restored))
    seen = []
    def record_labels(values, count):
        seen.append(values.detach().clone())
        assert values.dtype == torch.float64
        assert torch.equal(values, torch.zeros_like(values))
        labels = bucket_labels(values, count)
        assert labels.tolist() == [[1, 1, 2, 2, 3, 3]] * len(values)
        return labels
    monkeypatch.setattr(loss_module, "bucket_labels", record_labels)
    loss = objective(prediction, model_target, target_economic=economic)
    loss.backward()
    assert seen and economic.grad is None


@pytest.mark.parametrize("shuffle", [False, True])
def test_raw_target_loader_is_aligned_with_each_shuffled_forecast_window(shuffle):
    raw = pd.DataFrame(np.arange(60, dtype=np.float64).reshape(20, 3) / 100.,
        columns=list("ABC"), index=pd.date_range("2024-01-01", periods=20, name="date"))
    normalized = (raw - np.array([.1, -.2, 1.])) / np.array([.2, .5, .7])
    config = SimpleNamespace(seq_len=4, pred_len=3, label_len=2, freq="d", num_workers=0)
    dataset, loader = forecasting_data_provider(normalized, config, 1, 3, shuffle, False, raw_targets=raw)
    legacy = DatasetForTransformer(normalized, history_len=4, prediction_len=3, label_len=2, freq="d")
    assert len(legacy[0]) == 4 and len(dataset[0]) == 5
    for actual, original in zip(dataset[0][:4], legacy[0]):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    starts = []
    for x, y, xm, ym, raw_y in loader:
        assert raw_y.dtype == torch.float64 and x.dtype == y.dtype == torch.float32
        for sample in range(len(x)):
            # Locate the first target by its exact source value, independently
            # of DataLoader iteration order, then check its preceding inputs.
            start = int(round(float(raw_y[sample, 0, 0]) * 100 / 3))
            starts.append(start)
            torch.testing.assert_close(raw_y[sample], torch.tensor(raw.iloc[start:start+3].values), rtol=0, atol=0)
            torch.testing.assert_close(x[sample], torch.tensor(normalized.iloc[start-4:start].values, dtype=torch.float32), rtol=0, atol=0)
            torch.testing.assert_close(y[sample], torch.tensor(normalized.iloc[start-2:start+3].values, dtype=torch.float32), rtol=0, atol=0)
    assert sorted(starts) == list(range(4, 18))
    with pytest.raises(ValueError, match="alinhar"):
        DatasetForTransformer(normalized, raw_targets=raw.iloc[::-1])


def test_raw_labels_accumulate_original_double_precision_returns(monkeypatch):
    raw = torch.tensor([[[.1, .1 + 1e-10, .1 - 1e-10],
                         [.2, .2 + 1e-10, .2 - 1e-10]]], dtype=torch.float64)
    objective = criterion("bucket_ranknet", trade_window=2)
    seen = []
    def record_labels(values, count):
        seen.append(values.detach().clone())
        return bucket_labels(values, count)
    monkeypatch.setattr(loss_module, "bucket_labels", record_labels)
    objective(torch.zeros_like(raw, dtype=torch.float32), raw.float(), target_economic=raw)
    torch.testing.assert_close(seen[0], raw.sum(1), rtol=0, atol=0)
    assert bucket_labels(seen[0], 3).tolist() == [[2, 1, 3]]
