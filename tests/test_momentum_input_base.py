"""Actual NS fit/forecast and stable validation tests for the input experiment."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ts_benchmark.baselines import deep_forecasting_model_base as base
from ts_benchmark.baselines.composite_trading_loss_v3 import build_loss
from ts_benchmark.baselines.momentum_input import momentum_input_features
from ts_benchmark.baselines.time_series_library.adapters_for_transformers import TransformerAdapter
from ts_benchmark.baselines.time_series_library.models.Nonstationary_Transformer import Nonstationary_Transformer


def settings(**overrides):
    values = dict(seq_len=8, horizon=2, norm=True, batch_size=8, num_epochs=2,
                  lr=.001, lradj="constant", loss="composite_trading", loss_cross="ranknet_proximity",
                  loss_cross_lambda=1., loss_cross_mask=["cauchy", .025, "natural"], loss_trade_window=2,
                  loss_data_kind="log_return", loss_score_kind="simple_return", momentum_input_mode="feature",
                  momentum_input_lookback=12, validation_mode="ordered_sample_mean", parallel_strategy=None,
                  d_model=8, d_ff=16, n_heads=2, e_layers=1, d_layers=1, p_hidden_dims=[8, 8],
                  p_hidden_layers=2, dropout=0.)
    values.update(overrides)
    return values


def market():
    values = np.random.default_rng(191).normal(0, .02, (64, 4))
    return pd.DataFrame(values, columns=list("ABCD"), index=pd.date_range("2024-01-01", periods=64, name="date"))


@pytest.fixture(autouse=True)
def cpu_single_thread(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(base, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(base, "build_loss", build_loss)
    yield
    torch.set_num_threads(previous)


class RecordingNS(TransformerAdapter):
    def __init__(self, **kwargs):
        super().__init__("Nonstationary_Transformer", Nonstationary_Transformer, **settings(**kwargs))
        self.recorded_inputs = []

    def _process(self, input, target, input_mark, target_mark, *, momentum_features=None):
        self.recorded_inputs.append((input.detach().clone(), momentum_features.detach().clone()))
        return super()._process(input, target, input_mark, target_mark, momentum_features=momentum_features)


@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("mode", ["zero", "feature"])
def test_actual_ns_fit_feature_history_checkpoint_and_both_forecast_paths(norm, mode, monkeypatch):
    loaders = []
    original = base.forecasting_data_provider
    def recording_provider(*args, **kwargs):
        result = original(*args, **kwargs)
        loaders.append((kwargs, result[1]))
        return result
    monkeypatch.setattr(base, "forecasting_data_provider", recording_provider)
    torch.manual_seed(19)
    model = RecordingNS(norm=norm, momentum_input_mode=mode)
    data = market()
    model.forecast_fit(data, train_ratio_in_tv=.75)
    assert isinstance(loaders[0][1].sampler, SequentialSampler)
    assert loaders[0][0]["shuffle"] is False
    assert model.win_size == 12 and model.config.seq_len == 8
    assert "momentum_input.projection.weight" in model.check_point
    assert not any("momentum_score" in key for key in model.check_point)
    np.testing.assert_allclose(model.scaler.mean_, data.iloc[:48].mean().to_numpy())
    assert all(x.shape[1] == 8 and f.shape == (len(x), 4) for x, f in model.recorded_inputs)
    prediction = model.forecast(2, data)
    features = model.recorded_inputs[-1][1]
    expected_features = momentum_input_features(torch.tensor(data.iloc[-12:].to_numpy()[None], dtype=torch.float32), 12)
    torch.testing.assert_close(features, expected_features, atol=1e-6, rtol=1e-5)
    assert prediction.shape == (2, 4) and np.isfinite(prediction).all()
    class Maker:
        requested_length = None
        def make_batch(self, batch_size, win_size):
            self.requested_length = win_size
            return {"input": data.iloc[-win_size:].to_numpy()[None], "time_stamps": data.index.to_numpy()[-win_size:][None], "covariates": None}
    maker = Maker()
    batched = model.batch_forecast(2, maker)
    assert maker.requested_length == 12
    np.testing.assert_allclose(batched[0], prediction, rtol=1e-6, atol=1e-8)
    with pytest.raises(ValueError, match="horizon"):
        model.forecast(3, data)
    with pytest.raises(ValueError, match="horizon"):
        model.batch_forecast(3, maker)
    with pytest.raises(ValueError, match="universo"):
        model.forecast(2, data.iloc[:, ::-1])


def test_future_target_values_cannot_affect_input_feature_or_ns_prediction():
    torch.manual_seed(3)
    model = RecordingNS(norm=False, num_epochs=1)
    model.forecast_fit(market(), train_ratio_in_tv=.75)
    model.model.eval()
    history = torch.randn(2, 12, 4)
    target = torch.randn(2, 6, 4)
    input_mark = torch.zeros(2, 12, 3); target_mark = torch.zeros(2, 6, 3)
    with torch.no_grad():
        first = model._process_with_history(history, target, input_mark, target_mark)["output"]
        feature = model.recorded_inputs[-1][1]
        target[:, -2:] = 1e9
        second = model._process_with_history(history, target, input_mark, target_mark)["output"]
    torch.testing.assert_close(feature, model.recorded_inputs[-1][1], rtol=0, atol=0)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


class LossValidationAdapter(base.DeepForecastingModelBase):
    def __init__(self, mode):
        super().__init__({}, seq_len=1, horizon=1, norm=False, validation_mode=mode)
        self.model = torch.nn.Identity()
    def _process(self, input, target, input_mark, target_mark):
        return {"output": input}


def test_validation_weights_last_seven_origins_by_samples_not_by_batch():
    # 343 origins = 21 batches of 16 plus 7. Deliberately large final errors
    # make the historical 16/7 overweighting measurable without optimizer noise.
    prediction = torch.zeros(343, 1, 2); prediction[-7:] = 2
    target = torch.zeros_like(prediction); marks = torch.zeros(343, 1, 3)
    dataset = TensorDataset(prediction, target, marks, marks)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    stable = LossValidationAdapter("ordered_sample_mean")
    legacy = LossValidationAdapter("legacy")
    assert stable.validate(loader, 2, torch.nn.MSELoss()) == pytest.approx(4 * 7 / 343)
    assert legacy.validate(loader, 2, torch.nn.MSELoss()) == pytest.approx(4 / 22)
    # For this per-origin objective, changing batch size leaves the result fixed.
    assert stable.validate(DataLoader(dataset, batch_size=31), 2, torch.nn.MSELoss()) == pytest.approx(4 * 7 / 343)


def test_unsupported_architecture_and_exogenous_inputs_fail_before_training():
    model = RecordingNS()
    model._model_name = "TimesNet"
    with pytest.raises(ValueError, match="apenas para Nonstationary"):
        model.forecast_fit(market(), train_ratio_in_tv=.75)
    model._model_name = "Nonstationary_Transformer"
    with pytest.raises(ValueError, match="exogenas"):
        model.forecast_fit(market(), covariates={"exog": market()}, train_ratio_in_tv=.75)


def test_input_off_preserves_legacy_rng_parameters_and_forecasts():
    results = []
    for options in ({}, {"momentum_input_lookback": -20}):
        torch.manual_seed(77)
        model = TransformerAdapter("Nonstationary_Transformer", Nonstationary_Transformer,
                                   **settings(momentum_input_mode="off", **options))
        model.forecast_fit(market(), train_ratio_in_tv=.75)
        results.append((model.forecast(2, market()), torch.random.get_rng_state().clone(), model.check_point))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])
    for key in results[0][2]:
        torch.testing.assert_close(results[0][2][key], results[1][2][key], rtol=0, atol=0)
