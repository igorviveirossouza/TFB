"""Real CPU fitting, validation, checkpoint selection and prediction for RankMix.

These tests use the native TFB adapters and both supported experiment models;
they do not replace a model's forward pass or the forecasting data loader.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ts_benchmark.baselines import deep_forecasting_model_base as base
from ts_benchmark.baselines.composite_trading_loss_v3 import (
    CompositeTradingLossV3, build_loss,
)
from ts_benchmark.baselines.time_series_library.adapters_for_transformers import TransformerAdapter
from ts_benchmark.baselines.time_series_library.models.Nonstationary_Transformer import Nonstationary_Transformer
from ts_benchmark.baselines.time_series_library.models.simpleFORMER import simpleFORMER


class RecordingAdapter(TransformerAdapter):
    def _init_model(self):
        model = super()._init_model()
        self.initial_parameters = {k: v.detach().clone()
                                   for k, v in model.named_parameters()}
        return model

    def _init_criterion_and_optimizer(self):
        criterion, optimizer = super()._init_criterion_and_optimizer()
        self.training_criterion = criterion
        return criterion, optimizer


@pytest.fixture(autouse=True)
def cpu_single_thread(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(base, "get_device", lambda: torch.device("cpu"))
    # This is the same factory substitution made by the production V3 entrypoint.
    monkeypatch.setattr(base, "build_loss", build_loss)
    yield
    torch.set_num_threads(previous)


def dataset():
    values = np.random.default_rng(191).normal(0, .02, (64, 6))
    return pd.DataFrame(values, columns=list("ABCDEF"),
        index=pd.date_range("2024-01-01", periods=64, name="date"))


def make_model(name, architecture, **overrides):
    options = dict(
        seq_len=8, horizon=2, norm=True, batch_size=8, num_epochs=1,
        lr=.001, lradj="constant", loss="composite_trading",
        loss_temporal="huber", loss_cross="rankmix", loss_cross_lambda=.8,
        loss_cross_mask=["cauchy", .025, "natural"],
        loss_rank_mix_weights=[.5, .5, 0.], loss_rank_mix_mode="regional",
        loss_rank_mix_regions=["top", "bottom", "top"],
        loss_rank_mix_fraction=1/3,
        loss_listnet_tau=.8, loss_listmle_tau=.9, loss_approxndcg_tau=.6,
        loss_huber_delta=.7, loss_trade_window=2,
        loss_data_kind="log_return", loss_score_kind="simple_return",
        momentum_mode="off", momentum_input_mode="off",
        validation_mode="ordered_sample_mean", parallel_strategy=None,
        d_model=8, d_ff=16, n_heads=2, e_layers=1, d_layers=1,
        p_hidden_dims=[8, 8], p_hidden_layers=2, dropout=0.)
    options.update(overrides)
    return RecordingAdapter(name, architecture, **options)


@pytest.mark.parametrize("name,architecture", [
    ("Nonstationary_Transformer", Nonstationary_Transformer),
    ("simpleFORMER", simpleFORMER),
])
@pytest.mark.parametrize("mode", ["global", "regional"])
@pytest.mark.parametrize("cross_loss", ["rankmix", "rankmix_proximity"])
@pytest.mark.parametrize("lam,temporal", [(.8, "huber"), (1., "mse")])
@pytest.mark.parametrize("weights", [[.5, .5, 0.], [.2, .3, .5]])
def test_native_rankmix_fit_and_forecast(name, architecture, mode, cross_loss,
                                       lam, temporal, weights):
    data = dataset()
    torch.manual_seed(2026)
    model = make_model(name, architecture, loss_rank_mix_mode=mode,
        loss_rank_mix_weights=weights, loss_cross=cross_loss,
        loss_cross_lambda=lam, loss_temporal=temporal)
    model.forecast_fit(data, train_ratio_in_tv=.75)

    criterion = model.training_criterion
    assert isinstance(criterion, CompositeTradingLossV3)
    assert criterion.cross_loss_name == cross_loss
    assert criterion.cross_lambda == lam
    assert criterion.temporal_loss_name == temporal
    assert criterion.rank_mix_config == {
        "weights": weights, "mode": mode,
        "regions": ["top", "bottom", "top"], "fraction": 1/3}
    assert criterion.listnet_tau == .8 and criterion.listmle_tau == .9
    assert criterion.approxndcg_tau == .6
    assert criterion.momentum_mode == "off"
    assert not model._momentum_enabled() and not model._momentum_input_enabled()
    assert isinstance(model.model, architecture)
    assert model.check_point
    assert not any("momentum" in key or "rank_mix" in key for key in model.check_point)
    assert all(torch.isfinite(value).all() for value in model.check_point.values())
    assert any(not torch.equal(model.check_point[key], initial)
               for key, initial in model.initial_parameters.items())
    gradients = [p.grad for p in model.model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)

    predictions = model.forecast(2, data)
    assert predictions.shape == (2, 6)
    assert np.isfinite(predictions).all()


@pytest.mark.parametrize("name,architecture", [
    ("Nonstationary_Transformer", Nonstationary_Transformer),
    ("simpleFORMER", simpleFORMER),
])
def test_lambda_zero_fits_match_native_mse(name, architecture):
    data = dataset()
    predictions, checkpoints = [], []
    for overrides in [
        {"loss": "mse", "loss_temporal": "mse"},
        {"loss_cross": "rankmix_proximity", "loss_cross_lambda": 0.,
         "loss_temporal": "mse", "loss_rank_mix_weights": [.1, .2, .7]},
    ]:
        torch.manual_seed(2026)
        model = make_model(name, architecture, **overrides)
        model.forecast_fit(data, train_ratio_in_tv=.75)
        predictions.append(model.forecast(2, data))
        checkpoints.append(model.check_point)
    assert checkpoints[0].keys() == checkpoints[1].keys()
    for key in checkpoints[0]:
        torch.testing.assert_close(checkpoints[0][key], checkpoints[1][key],
                                   rtol=0, atol=0)
    np.testing.assert_array_equal(predictions[0], predictions[1])
