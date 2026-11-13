"""Real CPU optimization and checkpoint round trips for the ordinal task."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ts_benchmark.baselines import deep_forecasting_model_base as base
from ts_benchmark.baselines.ordinal_losses import ORDINAL_LOSSES, attach_ordinal_parameters
from ts_benchmark.baselines.composite_trading_loss_v3 import build_loss
from ts_benchmark.baselines.time_series_library.adapters_for_transformers import TransformerAdapter
from ts_benchmark.baselines.time_series_library.models.Nonstationary_Transformer import Nonstationary_Transformer
from ts_benchmark.baselines.time_series_library.models.simpleFORMER import simpleFORMER


class RecordingAdapter(TransformerAdapter):
    def _init_criterion_and_optimizer(self):
        criterion, optimizer = super()._init_criterion_and_optimizer()
        self.training_criterion = criterion
        self.initial_thresholds = ({key: value.detach().clone()
            for key, value in criterion.ordinal_thresholds.state_dict().items()}
            if getattr(criterion, "ordinal_thresholds", None) is not None else {})
        self.optimizer_parameter_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.economic_batches = []
        return criterion, optimizer

    def _criterion_loss(self, *args, **kwargs):
        raw = kwargs.get("target_economic")
        if raw is not None:
            self.economic_batches.append((self.model.training, raw.detach().cpu().clone()))
        return super()._criterion_loss(*args, **kwargs)


@pytest.fixture(autouse=True)
def single_cpu_thread(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(base, "get_device", lambda: torch.device("cpu"))
    yield
    torch.set_num_threads(previous)


def dataset():
    values = np.random.default_rng(185).normal(0, .02, (64, 6))
    return pd.DataFrame(values, columns=list("ABCDEF"),
                        index=pd.date_range("2024-01-01", periods=64, name="date"))


def model(name="Nonstationary_Transformer", architecture=Nonstationary_Transformer, **kwargs):
    options = dict(seq_len=8, horizon=2, norm=True, batch_size=8, num_epochs=1,
        lr=.001, lradj="constant", loss="composite_trading", loss_temporal="mse",
        loss_cross="ordinal_ce", loss_cross_lambda=.8, loss_bucket_count=3,
        loss_trade_window=1, loss_data_kind="log_return", loss_score_kind="simple_return",
        loss_cross_construction="origin_anchored_windows", loss_rank_windows=[1, 2],
        momentum_mode="off", validation_mode="ordered_sample_mean", parallel_strategy=None,
        d_model=8, d_ff=16, n_heads=2, e_layers=1, d_layers=1,
        p_hidden_dims=[8, 8], p_hidden_layers=2, dropout=0.)
    options.update(kwargs)
    return RecordingAdapter(name, architecture, **options)


@pytest.mark.parametrize("cross_loss", sorted(ORDINAL_LOSSES))
@pytest.mark.parametrize("lam", [.5, 1.])
def test_native_ns_fit_forecast_and_oce_parameters_in_best_checkpoint(cross_loss, lam):
    data = dataset()
    torch.manual_seed(2026)
    adapter = model(loss_cross=cross_loss, loss_cross_lambda=lam)
    adapter.forecast_fit(data, train_ratio_in_tv=.75)
    assert adapter.check_point
    assert all(torch.isfinite(v).all() for v in adapter.check_point.values())
    predictions = adapter.forecast(2, data)
    assert predictions.shape == (2, 6) and np.isfinite(predictions).all()
    assert {training for training, _ in adapter.economic_batches} == {True, False}
    for training, raw_batch in adapter.economic_batches:
        assert raw_batch.dtype == torch.float64
        for raw in raw_batch:
            starts = [i for i in range(len(data)-1)
                      if np.array_equal(data.iloc[i:i+2].values, raw.numpy())]
            assert len(starts) == 1
            assert (8 <= starts[0] <= 46) if training else (48 <= starts[0] <= 62)
    if cross_loss == "ordinal_ce":
        head = adapter.training_criterion.ordinal_thresholds
        assert head is adapter.model.ordinal_thresholds
        assert all(id(p) in adapter.optimizer_parameter_ids for p in head.parameters())
        assert any(not torch.equal(value, adapter.initial_thresholds[key])
                   for key, value in head.state_dict().items())
        assert any(key.startswith("ordinal_thresholds.") for key in adapter.check_point)
        selected = {key: value.detach().clone() for key, value in head.state_dict().items()}
        with torch.no_grad():
            head.first.add_(3)
        restored_predictions = adapter.forecast(2, data)
        np.testing.assert_array_equal(restored_predictions, predictions)
        for key, value in head.state_dict().items():
            torch.testing.assert_close(value, selected[key], rtol=0, atol=0)
        # A freshly initialized architecture can load the same complete state.
        fresh = adapter._init_model()
        new_criterion = build_loss(adapter.config, adapter.scaler.mean_, adapter.scaler.scale_)
        attach_ordinal_parameters(fresh, new_criterion)
        fresh.load_state_dict(adapter.check_point)
        torch.testing.assert_close(new_criterion.ordinal_thresholds(), head(), rtol=0, atol=0)
    else:
        assert not any(key.startswith("ordinal_thresholds.") for key in adapter.check_point)


@pytest.mark.parametrize("name,architecture", [
    ("Nonstationary_Transformer", Nonstationary_Transformer), ("simpleFORMER", simpleFORMER),
])
def test_temporal_endpoint_identical_to_native_mse(name, architecture):
    outputs, checkpoints = [], []
    data = dataset()
    for options in [{"loss": "mse"}, {"loss_cross_lambda": 0.}]:
        torch.manual_seed(2026)
        adapter = model(name, architecture, **options)
        adapter.forecast_fit(data, train_ratio_in_tv=.75)
        assert not adapter.economic_batches
        outputs.append(adapter.forecast(2, data))
        checkpoints.append(adapter.check_point)
    assert checkpoints[0].keys() == checkpoints[1].keys()
    for key in checkpoints[0]:
        torch.testing.assert_close(checkpoints[0][key], checkpoints[1][key], rtol=0, atol=0)
    np.testing.assert_array_equal(outputs[0], outputs[1])
