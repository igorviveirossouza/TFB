#!/usr/bin/env python3
"""CPU checks for simpleMLP/simpleFORMER and the existing TFB loss adapter.

Run from the repository root: python scripts/verify_simple_models.py
"""
import contextlib
import io
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ts_benchmark.baselines import deep_forecasting_model_base as deep_base
from ts_benchmark.baselines.composite_trading_loss_v3 import (
    CompositeTradingLossV3,
    build_loss,
    loss_accepts_base_value,
)
from ts_benchmark.baselines.time_series_library import simpleMLP, simpleFORMER
from ts_benchmark.models.model_loader import get_model_info


MODELS = (simpleMLP, simpleFORMER)


def config(**overrides):
    values = dict(
        seq_len=8, pred_len=4, horizon=4, enc_in=4, c_out=4,
        d_model=16, e_layers=2, d_ff=32, n_heads=4, dropout=0.0,
        use_norm=True, norm_eps=1e-5, temporal_hidden_dim=0,
        task_name="short_term_forecast",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class SimpleModelChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)

    def test_forecast_shapes_and_no_future_decoder_leakage(self):
        for cls in MODELS:
            for lb, horizon, channels in ((1, 1, 1), (32, 24, 66), (104, 5, 4), (246, 1, 4)):
                with self.subTest(model=cls.__name__, lb=lb, horizon=horizon):
                    model = cls(config(seq_len=lb, pred_len=horizon, enc_in=channels, c_out=channels)).eval()
                    x = torch.randn(2, lb, channels)
                    prediction = model(x)
                    self.assertEqual(prediction.shape, (2, horizon, channels))
                    torch.testing.assert_close(
                        prediction, model(x, None, torch.full((2, horizon, channels), float("nan")), None)
                    )
                    self.assertTrue(torch.isfinite(model(torch.ones_like(x))).all())

    def test_window_normalization_is_reversed_before_loss(self):
        for cls in MODELS:
            for enabled in (True, False):
                with self.subTest(model=cls.__name__, normalization=enabled):
                    model = cls(config(use_norm=enabled)).eval()
                    with torch.no_grad():
                        model.output_projection.weight.zero_()
                        model.output_projection.bias.fill_(2.0)
                    x = torch.randn(3, 8, 4) * 3 + 7
                    if enabled:
                        expected = x.mean(1, keepdim=True) + 2 * torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
                    else:
                        expected = x.new_full((3, 1, 4), 2.0)
                    torch.testing.assert_close(model(x), expected.expand(3, 4, 4))
                    model(x + 20)  # No cached normalization state may leak between calls.
                    torch.testing.assert_close(model(x), expected.expand(3, 4, 4))

    def test_all_existing_cross_losses_backpropagate(self):
        for cls in MODELS:
            for kind in ("log_return", "simple_return", "price"):
                for loss in ("mse", "pairwise_mse", "ranknet", "hinge", "bpr", "listnet"):
                    with self.subTest(model=cls.__name__, kind=kind, loss=loss):
                        model = cls(config())
                        x = torch.randn(2, 8, 4)
                        target = torch.randn(2, 4, 4)
                        criterion = CompositeTradingLossV3(
                            trade_window=2, data_kind=kind, cross_loss=loss,
                            cross_lambda=1.0, hinge_margin=10.0, cross_delta=0.001,
                            inverse_norm=True,
                            normalizer_mean=np.full(4, 100.0 if kind == "price" else 0.001),
                            normalizer_scale=np.full(4, 2.0 if kind == "price" else 0.02),
                        )
                        criterion(model(x), target, base_value=x[:, -1]).backward()
                        grads = [p.grad for p in model.parameters() if p.requires_grad]
                        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
                        self.assertGreater(model.input_projection.weight.grad.abs().sum().item(), 0)

    def test_lambda_zero_and_empty_dead_zone(self):
        for cls in MODELS:
            model = cls(config())
            pred = model(torch.randn(2, 8, 4))
            target = torch.randn_like(pred)
            options = dict(trade_window=2, data_kind="log_return", score_kind="log_return", inverse_norm=False)
            zero = CompositeTradingLossV3(cross_lambda=0, cross_loss="mse", **options)
            torch.testing.assert_close(zero(pred, target), F.mse_loss(pred, target))
            for name in ("pairwise_mse", "hinge", "ranknet", "bpr"):
                empty = CompositeTradingLossV3(cross_lambda=0.8, cross_loss=name, cross_delta=1e6, **options)
                torch.testing.assert_close(empty(pred, target), 0.2 * F.mse_loss(pred, target))

    def test_default_architecture_and_channel_dependence(self):
        for cls in MODELS:
            model = cls(config(d_model=512, d_ff=2048, n_heads=8)).eval()
            self.assertEqual(len(model.blocks), 2)
            self.assertFalse(next(model.blocks[0].parameters()) is next(model.blocks[1].parameters()))
            x = torch.randn(1, 8, 4, requires_grad=True)
            model(x)[0, 0, 0].backward()
            self.assertGreater(x.grad[0, :, 1:].abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            simpleFORMER(config(d_model=15, n_heads=4))

    def test_real_tfb_adapter_fit_and_forecast_with_both_objectives(self):
        values = np.random.default_rng(2026).normal(0, 0.02, (128, 4))
        frame = pd.DataFrame(values, index=pd.date_range("2020-01-01", periods=128, freq="D"))
        frame.index.name = "date"
        for cls in MODELS:
            for objective in ("mse", "composite_trading"):
                with self.subTest(model=cls.__name__, objective=objective):
                    info = get_model_info(dict(
                        model_name="time_series_library." + cls.__name__, adapter="transformer_adapter"
                    ))
                    params = vars(config())
                    params.update(norm=True, loss=objective, loss_cross="pairwise_mse",
                                  loss_cross_lambda=0.5, loss_trade_window=2,
                                  loss_cross_delta=0.005, loss_data_kind="log_return",
                                  batch_size=16, num_epochs=1, patience=1, lr=0.0001,
                                  parallel_strategy=None)
                    adapter = info["model_factory"](**params)
                    # The exact same two factory bindings as the v3 launcher.
                    with patch.object(deep_base, "build_loss", build_loss), patch.object(
                        deep_base, "loss_accepts_base_value", loss_accepts_base_value
                    ), contextlib.redirect_stdout(io.StringIO()):
                        adapter.forecast_fit(frame, train_ratio_in_tv=0.8)
                        forecast = adapter.forecast(4, frame)
                    self.assertEqual(forecast.shape, (4, 4))
                    self.assertTrue(np.isfinite(forecast).all())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
