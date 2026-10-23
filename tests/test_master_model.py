"""MASTER training/TFB integration using the audited synthetic R-shaped panel."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from ts_benchmark.baselines.master import MASTER
from ts_benchmark.models.model_loader import get_model_info

from test_master_data import ExportFixture


@pytest.fixture(autouse=True)
def small_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def panel(tmp_path):
    exported = ExportFixture(tmp_path)
    bundle = exported.load()
    frame = pd.DataFrame(bundle.targets, columns=bundle.assets,
                         index=pd.date_range("2000-01-02", periods=bundle.n_dates))
    return exported, frame


def model_for(exported, **kwargs):
    params = dict(master_bundle_dir=exported.directory, master_returns_path=exported.returns,
                  seq_len=4, pred_len=5, d_model=8, t_heads=2, s_heads=2, dropout=0.0,
                  batch_size=8, num_epochs=2, patience=2, device="cpu", seed=17,
                  master_max_train_batches=2, master_max_validation_batches=2)
    params.update(kwargs)
    return MASTER(**params)


class ArrayBatchMaker:
    def __init__(self, panel, starts):
        self.panel, self.starts = panel, starts

    def make_batch(self, batch_size, win_size):
        starts = self.starts[:batch_size]
        return {"input": np.stack([self.panel.iloc[j-win_size:j].values for j in starts]),
                "time_stamps": np.stack([self.panel.index[j-win_size:j].values for j in starts]),
                "covariates": None}


@pytest.mark.parametrize("horizon,loss,cross", [(1, "mse", "ranknet"),
    (5, "composite_trading", "ranknet"), (5, "composite_trading", "hinge_proximity")])
def test_train_predict_and_batch_multihorizon(panel, horizon, loss, cross):
    exported, frame = panel
    model = model_for(exported, pred_len=horizon, loss=loss, loss_cross=cross,
                      loss_cross_mask=["gaussian", .025, "complementary"])
    assert model.forecast_fit(frame.iloc[:38], train_ratio_in_tv=.75) is model
    assert model.audit_["split"] == {"n_dates": 48, "n_pretest": 38, "n_train": 28,
                                     "n_validation": 10, "n_test": 10}
    assert model.audit_["limited_run"] is True
    assert model.audit_["best_epoch"] in {1, 2}
    assert model.audit_["history"][0]["train"]["n_windows"] == 16
    assert model.audit_["window_counts"]["train"] == 28 - 4 - horizon + 1
    assert len(model.validation_dataset) == 10 - horizon + 1
    pred = model.forecast(horizon, frame.iloc[:38])
    batch = model.batch_forecast(horizon, ArrayBatchMaker(frame, [38, 39]))
    assert pred.shape == (horizon, 3)
    assert batch.shape == (2, horizon, 3)
    assert np.isfinite(batch).all()
    np.testing.assert_allclose(batch[0], pred, atol=1e-8)
    if loss == "composite_trading":
        assert {"temporal", "cross", "cross_scaled", "total"}.issubset(model.history_[0]["train"])
        assert model.criterion.aggregator.inverse_norm is True
        np.testing.assert_allclose(model.criterion.aggregator.normalizer_scale.cpu().numpy().ravel(),
                                   model.target_scaler.scale_, rtol=1e-6)


def test_normalizers_use_only_training_prefix_and_forecast_only_past(panel):
    exported, frame = panel
    model = model_for(exported).forecast_fit(frame.iloc[:38], train_ratio_in_tv=.75)
    np.testing.assert_allclose(model.target_scaler.mean_, frame.iloc[:28].mean().values)
    np.testing.assert_allclose(model.target_scaler.scale_, frame.iloc[:28].std(ddof=0).values)
    assert model.feature_scaler.train_end == 28
    assert model.audit_["target_normalization"]["n_samples_seen"] == 28
    future = model.forecast(5, frame.iloc[:38])
    model.prepared.asset_values[38:] = 1e6
    model.prepared.market_values[38:] = -1e6
    model.bundle.targets[38:] = 300.0
    np.testing.assert_array_equal(model.forecast(5, frame.iloc[:38]), future)
    # Inference returns raw units after the network-space prediction.
    with torch.no_grad():
        for parameter in model.model.parameters():
            parameter.zero_()
    np.testing.assert_allclose(model.forecast(5, frame.iloc[:38]),
                               np.broadcast_to(model.target_scaler.mean_, (5, 3)))


@pytest.mark.parametrize("kwargs", [dict(asset_blocks=["retornos"], use_market=False),
                                     dict(master_asset_blocks=["retornos"], master_use_market=False)])
def test_ablations_change_actual_network_inputs(panel, kwargs):
    exported, frame = panel
    model = model_for(exported, **kwargs).forecast_fit(frame.iloc[:38], train_ratio_in_tv=.75)
    assert model.bundle.asset_features == ("retorno_log", "retorno_media_3")
    assert model.model.n_asset_features == 2
    assert model.model.n_market_features == 0
    assert model.model.market_gate is None
    assert model.prepared.market_values.shape == (48, 0)
    assert np.isfinite(model.forecast(5, frame.iloc[:38])).all()
    assert not model.audit_["audit_flags_as_inputs"]


@pytest.mark.parametrize("kind", ["trading_dates", "trading_steps"])
def test_original_calendar_and_explicit_asset_permutation(panel, kind):
    exported, frame = panel
    frame.index = exported.dates if kind == "trading_dates" else np.arange(1, 49)
    frame = frame.iloc[:, ::-1]
    model = model_for(exported).forecast_fit(frame.iloc[:38], train_ratio_in_tv=.75)
    assert model.index_kind == kind
    assert model.bundle.assets == tuple(frame.columns)
    assert model.forecast(5, frame.iloc[:38]).shape == (5, 3)


def test_rejects_unverified_targets_calendar_and_covariates(panel):
    exported, frame = panel
    bad = frame.iloc[:38].copy()
    bad.iloc[0, 0] += .01
    with pytest.raises(ValueError, match="frozen economic"):
        model_for(exported).forecast_fit(bad)
    bad = frame.iloc[:38].copy()
    bad.index += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="Training index"):
        model_for(exported).forecast_fit(bad)
    with pytest.raises(ValueError, match="validation split"):
        model_for(exported).forecast_fit(frame.iloc[:38], train_ratio_in_tv=1.)
    with pytest.raises(ValueError, match="untouched test"):
        model_for(exported).forecast_fit(frame)
    with pytest.raises(ValueError, match="covariates exclusively"):
        model_for(exported).forecast_fit(frame.iloc[:38], covariates={"exog": frame.values})
    model = model_for(exported).forecast_fit(frame.iloc[:38], train_ratio_in_tv=.75)
    with pytest.raises(ValueError, match="horizon"):
        model.forecast(1, frame.iloc[:38])
    with pytest.raises(ValueError, match="asset order"):
        model.forecast(5, frame.iloc[:38, ::-1])
    bad = frame.iloc[:38].copy()
    bad.iloc[-1, 0] += 1.
    with pytest.raises(ValueError, match="Prediction history"):
        model.forecast(5, bad)
    bad = frame.iloc[:38].drop(frame.index[36])
    with pytest.raises(ValueError, match="consecutive audited"):
        model.forecast(5, bad)


@pytest.mark.parametrize("kwargs,match", [
    ({"horizon": 1}, "must agree"),
    ({"loss": "composite_trading", "loss_trade_window": 2}, "must divide"),
    ({"loss": "composite_trading", "loss_trade_window": 1.5}, "integer"),
    ({"loss": "composite_trading", "loss_inverse_norm": False}, "Financial losses"),
    ({"loss_data_kind": "price"}, "economic log returns"),
    ({"loss": "unknown"}, "Loss desconhecida"),
    ({"master_use_market": "False"}, "must be a boolean"),
    ({"master_max_train_batches": 0}, "integer"),
    ({"master_max_validation_batches": 1.5}, "integer"),
    ({"lradj": "unimplemented"}, "lradj"),
    ({"asset_blocks": ["retornos"], "master_asset_blocks": ["ohlcv"]}, "must agree"),
])
def test_bad_configuration_fails_early(panel, kwargs, match):
    exported, _ = panel
    with pytest.raises(ValueError, match=match):
        model_for(exported, **kwargs)


def test_checkpoint_audit_and_same_forecast_after_reload(panel, tmp_path):
    exported, frame = panel
    audit_path, checkpoint = tmp_path / "audit.json", tmp_path / "model.pt"
    model = model_for(exported, master_audit_path=audit_path,
                      master_checkpoint_path=checkpoint).forecast_fit(frame.iloc[:38], train_ratio_in_tv=.75)
    disk_audit = json.loads(audit_path.read_text())
    assert disk_audit == model.audit_
    assert disk_audit["bundle_report"]["assets"] == list(exported.assets)
    original = model.forecast(5, frame.iloc[:38])
    restored = MASTER.load_checkpoint(checkpoint)
    np.testing.assert_array_equal(restored.forecast(5, frame.iloc[:38]), original)
    np.testing.assert_array_equal(restored.feature_scaler.asset_mean, model.feature_scaler.asset_mean)
    assert restored.audit_["best_epoch"] == model.audit_["best_epoch"]
    # A newly rehashed, structurally valid bundle still cannot replace the
    # checkpoint's exact original inputs.
    data = exported.read(exported.names["ativos"])
    data.loc[data.index[0], "ohlcv_log_close_open"] += .0001
    exported.write(exported.names["ativos"], data)
    exported.refresh_quality()
    exported.rehash()
    with pytest.raises(ValueError, match="hashes differ"):
        MASTER.load_checkpoint(checkpoint)


def test_raw_target_option_and_scheduler(panel):
    exported, frame = panel
    model = model_for(exported, norm=False, lradj="type1", num_epochs=3, patience=3,
                      loss="composite_trading", loss_cross="pairwise_mse", loss_trade_window=5)
    model.forecast_fit(frame.iloc[:38], train_ratio_in_tv=.75)
    assert model.criterion.aggregator.inverse_norm is False
    assert model.criterion.aggregator.normalizer_mean is None
    assert [item["lr"] for item in model.history_] == [1e-4, 1e-4, 5e-5]
    with torch.no_grad():
        for parameter in model.model.parameters():
            parameter.zero_()
    np.testing.assert_array_equal(model.forecast(5, frame.iloc[:38]), np.zeros((5, 3)))


def test_failed_refit_cannot_forecast_with_stale_weights(panel):
    exported, frame = panel
    model = model_for(exported).forecast_fit(frame.iloc[:38], train_ratio_in_tv=.75)
    with pytest.raises(ValueError):
        model.forecast_fit(frame.iloc[:38], train_ratio_in_tv=1.)
    with pytest.raises(ValueError, match="forecast_fit"):
        model.forecast(5, frame.iloc[:38])


def test_checkpoint_loader_handles_older_torch_signature(panel, tmp_path, monkeypatch):
    exported, frame = panel
    model = model_for(exported).forecast_fit(frame.iloc[:38], train_ratio_in_tv=.75)
    checkpoint = tmp_path / "legacy_api.pt"
    model.save_checkpoint(checkpoint)
    native_load = torch.load
    calls = []

    def legacy_load(path, map_location):
        calls.append(map_location)
        return native_load(path, map_location=map_location, weights_only=True)

    monkeypatch.setattr(torch, "load", legacy_load)
    restored = MASTER.load_checkpoint(checkpoint)
    assert calls == ["cpu"]
    np.testing.assert_array_equal(restored.forecast(5, frame.iloc[:38]), model.forecast(5, frame.iloc[:38]))


def test_registered_model_resolves_without_generic_transformer_adapter():
    assert get_model_info({"model_name": "master.MASTER", "adapter": None}) is MASTER
