"""CPU integration tests using the actual shared training/forecast lifecycle."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ts_benchmark.baselines import deep_forecasting_model_base as base
from ts_benchmark.baselines.composite_trading_loss_v3 import build_loss


def settings(**overrides):
    config = dict(seq_len=8, horizon=2, norm=True, batch_size=8, num_epochs=2,
                  lr=.01, lradj="constant", loss="composite_trading",
                  loss_cross="ranknet_proximity", loss_cross_lambda=.8,
                  loss_cross_mask=["cauchy", .025, "natural"],
                  loss_trade_window=1, loss_data_kind="log_return",
                  loss_score_kind="simple_return", momentum_mode="residual_output",
                  momentum_lookback=12, momentum_eta=.1, parallel_strategy=None)
    config.update(overrides)
    return config


def market():
    values = np.random.default_rng(19).normal(0, .02, (64, 4))
    return pd.DataFrame(values, columns=["A", "B", "C", "D"], index=pd.date_range("2024-01-01", periods=64, freq="D", name="date"))


@pytest.fixture(autouse=True)
def cpu_single_thread(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(base, "get_device", lambda: torch.device("cpu"))
    yield
    torch.set_num_threads(previous)


class TinyAdapter(base.DeepForecastingModelBase):
    """A small backbone exercises the unmodified shared fit/validation loops."""
    def __init__(self, **kwargs):
        super().__init__({}, **settings(**kwargs))
        self.native_inputs = []
        self.loss_histories = []

    def _init_model(self):
        return torch.nn.Linear(self.config.seq_len, self.config.horizon)

    def _process(self, input, target, input_mark, target_mark):
        self.native_inputs.append((self.model.training, input.detach().clone(), input_mark.detach().clone()))
        return {"output": self.model(input.transpose(1, 2)).transpose(1, 2)}

    def _criterion_loss(self, criterion, output, target, base_value=None, history=None):
        self.loss_histories.append((self.model.training, history.detach().clone()))
        return super()._criterion_loss(criterion, output, target, base_value=base_value, history=history)

    def _init_criterion_and_optimizer(self):
        criterion, optimizer = super()._init_criterion_and_optimizer()
        self.optimizer_parameter_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        return criterion, optimizer


def test_train_validation_full_history_native_crop_and_gain_optimizer_checkpoint():
    torch.manual_seed(6)
    model = TinyAdapter()
    model.forecast_fit(market(), train_ratio_in_tv=.75)
    assert model.win_size == 12
    assert model.config.seq_len == 8
    assert model._data_config().seq_len == 12
    assert id(model._momentum_module().gain) in model.optimizer_parameter_ids
    assert "momentum_score.gain" in model.check_point
    assert abs(model._momentum_module().gain.item()) > 0
    assert {training for training, _ in model.loss_histories} == {False, True}
    assert len(model.loss_histories) == len(model.native_inputs)
    for (training, native, marks), (loss_training, history) in zip(model.native_inputs, model.loss_histories):
        assert training == loss_training
        assert native.shape[1] == marks.shape[1] == 8
        assert history.shape[1] == 12
        torch.testing.assert_close(native, history[:, -8:, :], rtol=0, atol=0)
    # Only the training block defines the external scaler.
    np.testing.assert_allclose(model.scaler.mean_, market().iloc[:48].mean().to_numpy())


def test_forecast_and_batch_export_best_gain_and_keep_returns_separate(tmp_path):
    torch.manual_seed(7)
    path = tmp_path / "momentum_state.json"
    model = TinyAdapter(momentum_state_path=str(path))
    data = market()
    model.forecast_fit(data, train_ratio_in_tv=.75)
    best_gain = float(model.check_point["momentum_score.gain"])
    with torch.no_grad():
        model._momentum_module().gain.fill_(99.)
    predicted = model.forecast(2, data)
    assert predicted.shape == (2, 4)
    assert np.isfinite(predicted).all()
    state = json.loads(path.read_text())
    assert state["gain"] == best_gain
    assert state["asset_names"] == list(data.columns)
    # forecast returns the raw backbone path, not the dimensionless final score.
    scaled_input = model.scaler.transform(data.iloc[-8:].to_numpy())
    with torch.no_grad():
        expected_scaled = model.model(torch.tensor(scaled_input, dtype=torch.float32).T).T.numpy()
    np.testing.assert_allclose(predicted, model.scaler.inverse_transform(expected_scaled), rtol=1e-6, atol=1e-8)

    class Maker:
        requested_length = None

        def make_batch(self, batch_size, win_size):
            self.requested_length = win_size
            return {"input": data.iloc[-win_size:].to_numpy()[None], "time_stamps": data.index.to_numpy()[-win_size:][None], "covariates": None}

    maker = Maker()
    batched = model.batch_forecast(2, maker)
    assert maker.requested_length == 12
    np.testing.assert_allclose(batched[0], predicted, rtol=1e-6, atol=1e-8)


def test_off_mode_keeps_native_windows_parameters_and_predictions(monkeypatch):
    # The existing runner installs V3 for composite experiments. Off mode keeps
    # using that legacy builder rather than changing the caller's policy.
    monkeypatch.setattr(base, "build_loss", build_loss)
    results = []
    for inactive_options in ({}, {"momentum_lookback": 0, "momentum_eta": float("nan")}):
        torch.manual_seed(16)
        model = TinyAdapter(momentum_mode="off", **inactive_options)
        model.forecast_fit(market(), train_ratio_in_tv=.75)
        results.append((model, model.forecast(2, market())))
        assert not hasattr(model.model, "momentum_score")
        assert model.win_size == model.config.seq_len == 8
        assert all(history.shape[1] == 8 for _, history in model.loss_histories)
    np.testing.assert_array_equal(results[0][1], results[1][1])
    for key in results[0][0].model.state_dict():
        torch.testing.assert_close(results[0][0].model.state_dict()[key], results[1][0].model.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("gain_init,schedule", [(0., "type1"), (.1, "type1"), (.1, "constant")])
def test_diagnostic_experiment_records_initial_gradient_lr_and_selected_gain(tmp_path, gain_init, schedule):
    torch.manual_seed(6)
    model = TinyAdapter(momentum_gain_init=gain_init, momentum_diagnostics=True,
                        loss_cross_lambda=1., lradj=schedule, num_epochs=3,
                        momentum_state_path=str(tmp_path / "momentum_state.json"))
    model.forecast_fit(market(), train_ratio_in_tv=.75)
    record = json.loads((tmp_path / "momentum_training_diagnostics.json").read_text())
    assert record["completed"] is True
    assert record["checkpoint_selection"] == "validation_loss"
    assert record["gain_init"] == gain_init
    assert record["gradient_objective"].startswith("total_loss")
    assert len(record["epochs"]) == 3
    assert [e["learning_rates_start"][0] for e in record["epochs"]] == pytest.approx(
        [.01, .01, .005] if schedule == "type1" else [.01] * 3)
    first = record["epochs"][0]
    assert first["gain_start"] == pytest.approx(gain_init)
    assert first["first_batch_gradient_norms"]["gain"] > 0
    assert (first["first_batch_gradient_norms"]["backbone"] > 0) == (gain_init != 0.)
    for row in record["epochs"]:
        assert np.isfinite(row["train_loss"]) and np.isfinite(row["validation_loss"])
        for sample in ("train_components", "validation_components"):
            assert {"temporal", "cross", "total", "momentum_penalty"} <= set(row[sample])
            # Lambda1 and no auxiliary loss: cross + eta*correction^2.
            assert row[sample]["total"] == pytest.approx(
                row[sample]["cross_scaled"] + row[sample]["momentum_penalty"], rel=1e-6)
    selected = record["epochs"][record["best_epoch"] - 1]
    assert selected["checkpoint_selected"] is True
    assert record["best_gain"] == selected["gain_end"]
    assert record["best_gain"] == float(model.check_point["momentum_score.gain"])
    model.forecast(2, market())
    state = json.loads((tmp_path / "momentum_state.json").read_text())
    assert state["gain"] == record["best_gain"]
    assert state.get("gain_init", 0.) == gain_init


def test_diagnostics_do_not_change_training_rng_checkpoint_or_forecast(tmp_path):
    results = []
    for enabled in (False, True):
        torch.manual_seed(71)
        model = TinyAdapter(momentum_gain_init=.1, momentum_diagnostics=enabled,
                            momentum_state_path=str(tmp_path / str(enabled) / "momentum_state.json"))
        model.forecast_fit(market(), train_ratio_in_tv=.75)
        results.append((model, model.forecast(2, market()), torch.random.get_rng_state()))
    assert not (tmp_path / "False" / "momentum_training_diagnostics.json").exists()
    np.testing.assert_array_equal(results[0][1], results[1][1])
    assert torch.equal(results[0][2], results[1][2])
    for key in results[0][0].check_point:
        torch.testing.assert_close(results[0][0].check_point[key], results[1][0].check_point[key], rtol=0, atol=0)


def test_diagnostics_record_early_stop_epoch_and_keep_best_checkpoint(tmp_path, monkeypatch):
    torch.manual_seed(6)
    model = TinyAdapter(momentum_diagnostics=True, num_epochs=5, patience=1,
                        momentum_state_path=str(tmp_path / "momentum_state.json"))
    validation_losses = iter([1., 2.])
    monkeypatch.setattr(model, "validate", lambda *args: next(validation_losses))
    model.forecast_fit(market(), train_ratio_in_tv=.75)
    record = json.loads((tmp_path / "momentum_training_diagnostics.json").read_text())
    assert len(record["epochs"]) == 2
    assert record["best_epoch"] == 1
    assert record["epochs"][-1]["checkpoint_selected"] is False
    assert record["epochs"][-1]["gain_end"] != record["best_gain"]
    assert record["best_gain"] == float(model.check_point["momentum_score.gain"])


@pytest.mark.parametrize("architecture", ["Nonstationary_Transformer", "DUET", "TimesNet", "FEDformer"])
def test_native_architecture_cpu_training_validation_and_forecast(architecture, tmp_path):
    torch.manual_seed(25)
    kwargs = settings(num_epochs=1, d_model=8, d_ff=16, n_heads=2, e_layers=1,
                      d_layers=1, hidden_size=8, num_experts=2, dropout=0.,
                      fc_dropout=0., moving_avg=3, top_k=2, num_kernels=2, p_hidden_dims=[8, 8],
                      p_hidden_layers=2, momentum_state_path=str(tmp_path / "state.json"))
    if architecture == "FEDformer":
        # Native FourierBlock uses eight heads internally.
        kwargs["n_heads"] = 8
    if architecture == "DUET":
        from ts_benchmark.baselines.duet.duet import DUET
        model = DUET(**kwargs)
    else:
        from ts_benchmark.baselines.time_series_library.adapters_for_transformers import TransformerAdapter
        from importlib import import_module
        model_class = getattr(import_module("ts_benchmark.baselines.time_series_library.models." + architecture), architecture)
        model = TransformerAdapter(architecture, model_class, **kwargs)
    model.forecast_fit(market(), train_ratio_in_tv=.75)
    assert "momentum_score.gain" in model.check_point
    assert torch.isfinite(model.check_point["momentum_score.gain"])
    prediction = model.forecast(2, market())
    assert prediction.shape == (2, 4)
    assert np.isfinite(prediction).all()
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["gain"] == float(model.check_point["momentum_score.gain"])
