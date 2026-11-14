"""Dynamic architecture selection must preserve the requested objective."""
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ts_benchmark.baselines import deep_forecasting_model_base as base
from ts_benchmark.baselines import ordinal_model_adapter as adapter_module
from ts_benchmark.baselines.ordinal_model_adapter import OrdinalModel, validate_model, initialize_objective
from ts_benchmark.baselines.composite_trading_loss_v3 import CompositeTradingLossV3
from ts_benchmark.baselines.ordinal_losses import ORDINAL_LOSSES, OrderedThresholds
from ts_benchmark.models.model_loader import ModelFactory, get_models


@pytest.fixture(autouse=True)
def single_cpu(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(base, "get_device", lambda: torch.device("cpu"))
    yield
    torch.set_num_threads(previous)


def data():
    return pd.DataFrame(np.random.default_rng(87).normal(0, .02, (64, 6)),
        columns=list("ABCDEF"), index=pd.date_range("2024-01-01", periods=64, name="date"))


def options(name="time_series_library.DLinear", adapter="transformer_adapter", **kwargs):
    result = dict(_ordinal_model_name=name, _ordinal_model_adapter=adapter,
        seq_len=8, pred_len=2, horizon=2, norm=True, batch_size=8, num_epochs=1,
        lr=.001, lradj="constant", loss="composite_trading", loss_temporal="mse",
        loss_cross="ordinal_ce", loss_cross_lambda=.8, loss_bucket_count=3,
        loss_trade_window=1, loss_data_kind="log_return", loss_score_kind="simple_return",
        loss_cross_construction="origin_anchored_windows", loss_rank_windows=[1, 2],
        validation_mode="ordered_sample_mean", parallel_strategy=None,
        d_model=8, d_ff=16, n_heads=2, e_layers=1, d_layers=1, hidden_size=8,
        patch_len=4, stride=2, dropout=0.)
    result.update(kwargs)
    return result


@pytest.mark.parametrize("name,adapter", [
    ("time_series_library.DLinear", "transformer_adapter"),
    ("time_series_library.PatchTST", "transformer_adapter"),
    ("amplifier.Amplifier", None),
    ("duet.DUET", None),
])
@pytest.mark.parametrize("loss", sorted(ORDINAL_LOSSES))
def test_real_architectures_train_the_requested_ordinal_loss(name, adapter, loss):
    kwargs = options(name, adapter, loss_cross=loss, num_experts=2, k=1,
                     fc_dropout=0., CI=True)
    preflight = validate_model(**kwargs)
    assert preflight["raw_economic_targets"]
    torch.manual_seed(2026)
    model = OrdinalModel(**kwargs)
    model.forecast_fit(data(), train_ratio_in_tv=.75)
    objective = model._ordinal_training_criterion
    assert isinstance(objective, CompositeTradingLossV3)
    assert objective.cross_loss_name == loss
    assert model.check_point
    forecast = model.forecast(2, data())
    assert forecast.shape == (2, 6) and np.isfinite(forecast).all()
    if loss == "ordinal_ce":
        head = model.model.ordinal_thresholds
        assert head is objective.ordinal_thresholds
        assert not torch.equal(head.first.detach(), OrderedThresholds(3).first.detach())
        assert any(key.startswith("ordinal_thresholds.") for key in model.check_point)


@pytest.mark.parametrize("name,adapter", [
    ("time_series_library.DLinear", "transformer_adapter"),
    ("amplifier.Amplifier", None),
    ("time_series_library.Nonstationary_Transformer", "transformer_adapter"),
])
def test_temporal_endpoint_matches_native_mse_and_is_pickleable(name, adapter):
    kwargs = options(name, adapter, loss="mse", loss_cross_lambda=0.,
                     p_hidden_dims=[8, 8], p_hidden_layers=2)
    native_hyper = {k: v for k, v in kwargs.items() if not k.startswith("_ordinal_")}
    spec = {"model_name": name, "model_hyper_params": native_hyper}
    if adapter:
        spec["adapter"] = adapter
    checkpoints, forecasts = [], []
    for factory in [get_models({"models": [spec]})[0], lambda: OrdinalModel(**kwargs)]:
        torch.manual_seed(2026)
        model = factory()
        model.forecast_fit(data(), train_ratio_in_tv=.75)
        checkpoints.append(model.check_point)
        forecasts.append(model.forecast(2, data()))
    for key in checkpoints[0]:
        torch.testing.assert_close(checkpoints[0][key], checkpoints[1][key], rtol=0, atol=0)
    np.testing.assert_array_equal(forecasts[0], forecasts[1])
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.forecast(2, data()), forecasts[1])


class NativeScheduler(base.DeepForecastingModelBase):
    """A native wrapper with its own optimizer/scheduler and hardcoded loss."""
    def __init__(self, **kwargs):
        super().__init__({}, **kwargs)

    def _init_model(self):
        return nn.Linear(6, 6)

    def _init_criterion_and_optimizer(self):
        self.saw_cutpoints_before_optimizer = hasattr(self.model, "ordinal_thresholds")
        optimizer = torch.optim.SGD(self.model.parameters(), lr=.03, momentum=.9)
        self.native_optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=.03,
            total_steps=20, pct_start=.3)
        return nn.L1Loss(), optimizer

    def _process(self, input, target, input_mark, target_mark):
        return {"output": self.model(input[:, -self.config.horizon:])}

    def _post_process(self, output, target):
        raise AssertionError("native loss weighting must not alter ordinal experiment scores")


def test_native_scheduler_optimizer_and_cutpoint_registration_are_preserved():
    model = NativeScheduler(**{k: v for k, v in options().items() if not k.startswith("_ordinal_")})
    model.model = model._init_model()
    model.scaler.fit(data())
    criterion, optimizer = initialize_objective(model)
    assert model.saw_cutpoints_before_optimizer
    assert isinstance(criterion, CompositeTradingLossV3)
    assert optimizer is model.native_optimizer is model.scheduler.optimizer
    assert isinstance(optimizer, torch.optim.SGD)
    assert len(optimizer.param_groups) == 1
    registered = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all(id(p) in registered for p in criterion.ordinal_thresholds.parameters())
    loss = criterion(torch.randn(2, 2, 6, requires_grad=True), torch.randn(2, 2, 6),
                     target_economic=torch.randn(2, 2, 6, dtype=torch.float64))
    loss.backward()
    optimizer.step()
    model.scheduler.step()
    assert optimizer.state and model.scheduler.last_epoch == 1


def test_native_loss_postprocessing_is_bypassed_only_for_this_experiment(monkeypatch):
    monkeypatch.setattr(adapter_module, "get_models", lambda config: [
        ModelFactory("NativeScheduler", NativeScheduler, config["models"][0]["model_hyper_params"])])
    model = OrdinalModel(**options(loss="mse", loss_cross_lambda=0.))
    model.forecast_fit(data(), train_ratio_in_tv=.75)
    assert isinstance(model._ordinal_training_criterion, nn.MSELoss)
    assert not model.saw_cutpoints_before_optimizer
    assert not hasattr(model.train_data_loader.dataset, "raw_targets") or model.train_data_loader.dataset.raw_targets is None


def test_preflight_does_not_initialize_network_or_fit(monkeypatch):
    monkeypatch.setattr(NativeScheduler, "_init_model", lambda self: pytest.fail("network initialized in preflight"))
    monkeypatch.setattr(adapter_module, "get_models", lambda config: [
        ModelFactory("NativeScheduler", NativeScheduler, config["models"][0]["model_hyper_params"])])
    assert validate_model(**options())["native_optimizer_override"]


def test_statistical_model_is_rejected_by_trainer_capability():
    with pytest.raises(ValueError, match="estatisticos"):
        validate_model(**options("snaive.SNaive", None))


@pytest.mark.parametrize("method", ["forecast_fit", "validate", "_criterion_loss"])
def test_independent_training_flow_is_rejected_instead_of_ignoring_loss(method, monkeypatch):
    custom = type("IndependentTrainer", (NativeScheduler,), {method: lambda *args, **kwargs: None})
    monkeypatch.setattr(adapter_module, "get_models", lambda config: [
        ModelFactory("IndependentTrainer", custom, config["models"][0]["model_hyper_params"])])
    with pytest.raises(ValueError, match=f"redefine {method}"):
        validate_model(**options())
