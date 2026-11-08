"""Tiny real fit/validation/forecast through both experiment architectures."""
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
        self.initial_parameters = {k: v.detach().clone() for k, v in model.named_parameters()}
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
    # The experiment worker selects V3; mirror that entry point explicitly.
    monkeypatch.setattr(base, "build_loss", build_loss)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("name, architecture", [
    ("Nonstationary_Transformer", Nonstationary_Transformer),
    ("simpleFORMER", simpleFORMER),
])
@pytest.mark.parametrize("cross_loss", [
    name + suffix for name in ("spearman", "listnet", "listmle", "approxndcg")
    for suffix in ("", "_proximity")
])
@pytest.mark.parametrize("lam, temporal", [(.8, "huber"), (1., "mse")])
def test_actual_fit_checkpoint_and_forecast(name, architecture, cross_loss, lam, temporal):
    values = np.random.default_rng(191).normal(0, .02, (64, 4))
    data = pd.DataFrame(values, columns=list("ABCD"),
        index=pd.date_range("2024-01-01", periods=64, name="date"))
    torch.manual_seed(2026)
    model = RecordingAdapter(name, architecture,
        seq_len=8, horizon=2, norm=True, batch_size=8, num_epochs=1,
        lr=.001, lradj="constant", loss="composite_trading",
        loss_temporal=temporal, loss_cross=cross_loss, loss_cross_lambda=lam,
        loss_cross_mask=["cauchy", .025], loss_spearman_tau=.7,
        loss_listnet_tau=.8, loss_listmle_tau=.9, loss_approxndcg_tau=.6,
        loss_huber_delta=.7,
        loss_trade_window=2, loss_data_kind="log_return", loss_score_kind="simple_return",
        momentum_mode="off", momentum_input_mode="off",
        validation_mode="ordered_sample_mean", parallel_strategy=None,
        d_model=8, d_ff=16, n_heads=2, e_layers=1, d_layers=1,
        p_hidden_dims=[8, 8], p_hidden_layers=2, dropout=0.)
    model.forecast_fit(data, train_ratio_in_tv=.75)

    criterion = model.training_criterion
    assert isinstance(criterion, CompositeTradingLossV3)
    assert criterion.cross_loss_name == cross_loss
    assert criterion.temporal_loss_name == temporal
    assert criterion.cross_lambda == lam and criterion.spearman_tau == .7
    assert criterion.listnet_tau == .8 and criterion.listmle_tau == .9
    assert criterion.approxndcg_tau == .6 and criterion.huber_delta == .7
    assert criterion.momentum_mode == "off"
    assert model.check_point
    assert not any("momentum" in key for key in model.check_point)
    assert all(torch.isfinite(value).all() for value in model.check_point.values())
    assert any(not torch.equal(model.check_point[key], initial)
               for key, initial in model.initial_parameters.items())
    gradients = [p.grad for p in model.model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)

    prediction = model.forecast(2, data)
    assert prediction.shape == (2, 4)
    assert np.isfinite(prediction).all()
