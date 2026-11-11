"""Origin-anchored ranking parity, isolation and legacy-block compatibility."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from ts_benchmark.baselines.composite_trading_loss_v3 import CompositeTradingLossV3, build_loss
from ts_benchmark.baselines.ns_task_heads import NSTaskAdapter


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def criterion(**kwargs):
    options = dict(
        trade_window=1, data_kind="log_return", score_kind="simple_return",
        cross_loss="rankmix_proximity", cross_lambda=.5, cross_scale=1.7,
        cross_mask=["cauchy", .025, "natural"],
        rank_mix_weights=[.5, .5, 0], rank_mix_mode="global",
        inverse_norm=True, normalizer_mean=[.002, -.004, .01, .001, -.002, .005],
        normalizer_scale=[.01, .02, .03, .04, .05, .015],
        cross_construction="origin_anchored_windows", rank_windows=[1, 5, 10, 15, 20, 25],
        track_components=True,
    )
    options.update(kwargs)
    return CompositeTradingLossV3(**options).double()


def sample(horizon):
    generator = torch.Generator().manual_seed(2026 + horizon)
    return (torch.randn(2, horizon, 6, dtype=torch.float64, generator=generator),
            torch.randn(2, horizon, 6, dtype=torch.float64, generator=generator))


@pytest.mark.parametrize("horizon", [1, 25])
@pytest.mark.parametrize("lam", [0., .1, .5, 1.])
@pytest.mark.parametrize("loss", ["rankmix", "rankmix_proximity"])
@pytest.mark.parametrize("score_kind", ["log_return", "simple_return"])
@pytest.mark.parametrize("mode", ["global", "regional"])
def test_same_value_and_gradient_as_existing_shared_head_objective(horizon, lam, loss, score_kind, mode):
    new = criterion(cross_lambda=lam, cross_loss=loss, score_kind=score_kind, rank_mix_mode=mode)
    old = criterion(cross_lambda=lam, cross_loss=loss, score_kind=score_kind,
                    rank_mix_mode=mode, cross_construction="blocks")
    prediction, target = sample(horizon)
    prediction = prediction.requires_grad_()
    reference_prediction = prediction.detach().clone().requires_grad_()
    # Execute the existing adapter method itself, without constructing a network.
    reference_adapter = SimpleNamespace(
        task_lambda=lam, study="heads", head_mode="shared",
        rank_windows=tuple(k for k in new.rank_windows if k <= horizon),
        config=SimpleNamespace(loss_score_kind=score_kind),
    )
    expected = NSTaskAdapter._criterion_loss(reference_adapter, old, reference_prediction, target)
    actual = new(prediction, target)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(prediction.grad, reference_prediction.grad, rtol=0, atol=0)
    assert new.last_components["H"] == horizon
    assert new.last_components["rank_windows"] == list(reference_adapter.rank_windows)
    assert new.last_components["n_rank_windows"] == len(reference_adapter.rank_windows)
    assert new.last_components["cross_construction"] == "origin_anchored_windows"


@pytest.mark.parametrize("window", [1, 5, 25])
@pytest.mark.parametrize("lam", [0., .5, 1.])
def test_default_and_explicit_blocks_retain_legacy_value_and_gradient(window, lam):
    base = dict(trade_window=window, data_kind="log_return", cross_loss="ranknet_proximity",
                cross_mask=["cauchy", .025, "natural"], cross_lambda=lam,
                inverse_norm=False, cross_score_normalization="zscore")
    implicit = CompositeTradingLossV3(**base).double()
    explicit = CompositeTradingLossV3(**base, cross_construction="blocks", rank_windows=[1, 5]).double()
    p, t = sample(25)
    p = p.requires_grad_(); q = p.detach().clone().requires_grad_()
    actual = implicit(p, t); expected = explicit(q, t)
    actual.backward(); expected.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    scores, labels = implicit.aggregator(p, t)
    temporal = implicit._temporal_loss(p, t)
    cross = implicit._cross_loss(scores, labels)
    legacy = temporal if lam == 0 else (cross if lam == 1 else (1-lam)*temporal+lam*cross)
    torch.testing.assert_close(actual, legacy, rtol=0, atol=0)


def test_prefix_window_one_ignores_later_days_but_legacy_k_one_does_not():
    origin = criterion(cross_lambda=1, rank_windows=[1], cross_loss="ranknet")
    blocks = criterion(cross_lambda=1, cross_construction="blocks", cross_loss="ranknet")
    p, t = sample(25)
    altered = t.clone(); altered[:, 1:] *= -100
    torch.testing.assert_close(origin(p, t), origin(p, altered), rtol=0, atol=0)
    assert not torch.isclose(blocks(p, t), blocks(p, altered))
    p.requires_grad_(); origin(p, t).backward()
    assert p.grad[:, :1].abs().sum() > 0
    assert torch.count_nonzero(p.grad[:, 1:]) == 0


def test_equal_weight_over_valid_windows_and_horizon_filter():
    full = criterion(cross_lambda=1, cross_scale=1, rank_windows=[1, 5, 25, 100])
    p, t = sample(25)
    expected = torch.stack([criterion(cross_lambda=1, cross_scale=1, rank_windows=[k])(p, t)
                            for k in [1, 5, 25]]).mean()
    torch.testing.assert_close(full(p, t), expected, rtol=1e-14, atol=1e-14)
    assert full.last_components["rank_windows"] == [1, 5, 25]


def test_inactive_objectives_are_not_evaluated():
    p, t = sample(25)
    temporal = criterion(cross_lambda=0)
    with patch.object(temporal, "_cross_loss", side_effect=AssertionError("inactive ranking")), \
            patch.object(temporal.aggregator, "_maybe_inverse_norm", side_effect=AssertionError("inactive scores")):
        torch.testing.assert_close(temporal(p, t), (p-t).square().mean(), rtol=0, atol=0)
    assert temporal.last_components["cross"] is None
    ranking = criterion(cross_lambda=1)
    with patch.object(ranking, "_temporal_loss", side_effect=AssertionError("inactive temporal")):
        assert torch.isfinite(ranking(p, t))
    assert ranking.last_components["temporal"] is None


@pytest.mark.parametrize("changes", [
    dict(cross_construction="unknown"), dict(rank_windows=None), dict(rank_windows=[]),
    dict(rank_windows="1 5"), dict(rank_windows=[1, 1]), dict(rank_windows=[0, 1]),
    dict(rank_windows=[-1]), dict(rank_windows=[True]), dict(rank_windows=[1.0]),
    dict(rank_windows=[float("nan")]), dict(trade_window=5), dict(data_kind="price"),
    dict(data_kind="simple_return"), dict(momentum_mode="residual_output"),
])
def test_invalid_origin_configuration_fails(changes):
    with pytest.raises(ValueError):
        criterion(**changes)


def test_no_window_fits_horizon_and_shapes_are_rejected():
    loss = criterion(rank_windows=[5])
    p, t = sample(1)
    with pytest.raises(ValueError, match="Nenhuma"):
        loss(p, t)
    with pytest.raises(ValueError, match="Shapes"):
        loss(p, t[:, :, :3])
    with pytest.raises(ValueError, match="B,H,N"):
        loss(p[:, 0], t[:, 0])


def test_factory_passes_protocol_and_preserves_native_temporal_control():
    config = SimpleNamespace(
        loss="composite_trading", loss_cross="rankmix_proximity", loss_cross_lambda=.8,
        loss_trade_window=1, loss_data_kind="log_return", loss_score_kind="simple_return",
        loss_cross_mask=["cauchy", .025, "natural"], norm=True, loss_inverse_norm=True,
        loss_cross_construction="origin_anchored_windows", loss_rank_windows=[1, 5, 25],
    )
    loss = build_loss(config, [0.] * 6, [0.01] * 6)
    assert loss.cross_construction == "origin_anchored_windows"
    assert loss.rank_windows == (1, 5, 25)
    assert loss.aggregator.inverse_norm
    p, t = sample(25)
    assert torch.isfinite(loss(p, t))
    config.loss_inverse_norm = False
    with pytest.raises(ValueError, match="loss_inverse_norm=True"):
        build_loss(config)
    config.norm = False  # Raw economic inputs do not require a scaler inversion.
    assert torch.isfinite(build_loss(config)(p * .01, t * .01))
    config.loss = "mse"
    temporal = build_loss(config)
    assert isinstance(temporal, torch.nn.MSELoss)
    torch.testing.assert_close(temporal(p, t), (p-t).square().mean(), rtol=0, atol=0)


@pytest.mark.parametrize("name", ["Nonstationary_Transformer", "iTransformer", "simpleFORMER"])
@pytest.mark.parametrize("lam", [0., .5, 1.])
def test_native_architectures_fit_and_export_without_extra_heads(name, lam, monkeypatch):
    """Exercise the native adapters with the same opt-in factory as B3 workers."""
    import importlib
    import numpy as np
    import pandas as pd
    from ts_benchmark.baselines import deep_forecasting_model_base as deep_base
    from ts_benchmark.baselines.composite_trading_loss_v3 import loss_accepts_base_value
    from ts_benchmark.baselines.time_series_library.adapters_for_transformers import TransformerAdapter

    monkeypatch.setattr(deep_base, "build_loss", build_loss)
    monkeypatch.setattr(deep_base, "loss_accepts_base_value", loss_accepts_base_value)
    model_class = getattr(importlib.import_module(
        "ts_benchmark.baselines.time_series_library.models." + name), name)
    torch.manual_seed(2026)
    adapter = TransformerAdapter(
        name, model_class, seq_len=8, horizon=5, norm=True, batch_size=8,
        num_epochs=1, patience=1, lr=.001, lradj="constant", d_model=8, d_ff=16,
        e_layers=1, d_layers=1, n_heads=2, p_hidden_dims=[8, 8], p_hidden_layers=2,
        dropout=0., parallel_strategy=None, validation_mode="ordered_sample_mean",
        loss="mse" if lam == 0 else "composite_trading", loss_temporal="mse",
        loss_cross="rankmix_proximity", loss_cross_lambda=lam, loss_trade_window=1,
        loss_data_kind="log_return", loss_score_kind="simple_return",
        loss_inverse_norm=True, loss_cross_mask=["cauchy", .025, "natural"],
        loss_cross_construction="origin_anchored_windows", loss_rank_windows=[1, 5, 25],
    )
    data = pd.DataFrame(np.random.default_rng(42).normal(0, .02, (80, 6)),
                        columns=list("ABCDEF"),
                        index=pd.date_range("2000-01-01", periods=80, name="date"))
    adapter.forecast_fit(data, train_ratio_in_tv=.75)
    predicted = adapter.forecast(5, data)
    assert predicted.shape == (5, 6) and np.isfinite(predicted).all()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.model.parameters())
    assert not any("ranking_head" in key or "ranking_projection" in key
                   for key in adapter.model.state_dict())
