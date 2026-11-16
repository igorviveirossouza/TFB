"""Rank-target training keeps future truth and input scaling on separate paths."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ts_benchmark.baselines.rank_target_data import stable_descending_ranks_numpy
from ts_benchmark.baselines.rank_target_model import (
    RankInputTransform, RankTargetDataset, RankTargetForecaster,
)


@pytest.fixture(autouse=True)
def single_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def frame():
    rng = np.random.default_rng(2027)
    data = rng.normal(size=(64, 6)) * np.arange(1, 7)[None, :] * .01
    # Real trading dates need not have an inferable frequency.
    dates = pd.bdate_range("2024-01-01", periods=65).delete(10)
    return pd.DataFrame(data, index=dates, columns=list("ABCDEF"))


def options(**kwargs):
    result = dict(model_name="time_series_library.models.TSMixer.TSMixer", model_adapter="transformer_adapter",
        seq_len=8, horizon=3, batch_size=8, num_epochs=1, patience=2,
        lr=.001, lradj="constant", seed=12, loss="MSE", loss_temporal="mse",
        loss_cross="bucket_hinge", loss_cross_lambda=.5, loss_bucket_count=3,
        loss_trade_window=1, parallel_strategy=None, d_model=8, d_ff=16,
        n_heads=2, e_layers=1, d_layers=1, p_hidden_dims=[8, 8], p_hidden_layers=2,
        patch_len=4, dropout=0., device="cpu")
    result.update(kwargs)
    return result


@pytest.mark.parametrize("input_kind", ["rank", "log_return"])
@pytest.mark.parametrize("normalization", ["none", "standard", "window", "both"])
def test_dataset_decoder_only_sees_observed_input(input_kind, normalization):
    data = frame()
    source = stable_descending_ranks_numpy(data.values) if input_kind == "rank" else data.values
    transform = RankInputTransform(normalization).fit(source[:40])
    dataset = RankTargetDataset(data, np.array([31]), seq_len=8, horizon=3,
        label_len=4, input_kind=input_kind, normalizer=transform)
    inputs, decoder, _, _, targets, returns = dataset[0]
    np.testing.assert_array_equal(decoder[:4], inputs[-4:])
    assert np.count_nonzero(decoder[4:]) == 0
    np.testing.assert_array_equal(targets, stable_descending_ranks_numpy(data.values[32:35]))
    np.testing.assert_array_equal(returns, data.values[32:35])
    changed = data.copy()
    changed.iloc[32:] = changed.iloc[32:].values[:, ::-1] * 1000
    future_changed = RankTargetDataset(changed, np.array([31]), seq_len=8, horizon=3,
        label_len=4, input_kind=input_kind, normalizer=transform)
    for before, after in zip(dataset[0][:4], future_changed[0][:4]):
        np.testing.assert_array_equal(before, after)
    inference = RankTargetDataset(data, np.array([31]), seq_len=8, horizon=3,
        label_len=4, input_kind=input_kind, normalizer=transform, supervised=False)
    assert not np.any(inference[0][4]) and not np.any(inference[0][5])


@pytest.mark.parametrize("normalization", ["none", "standard", "window", "both"])
def test_training_split_and_input_statistics_exclude_validation(normalization):
    data = frame()
    data.iloc[48:] = data.iloc[48:].values * 1000 + 100
    model = RankTargetForecaster(**options(input_kind="log_return", input_normalization=normalization))
    model.forecast_fit(data, .75)
    assert model.train_dataset_.origins[-1] + 3 == 47
    assert model.valid_dataset_.origins[0] + 1 == 48
    if normalization in {"standard", "both"}:
        np.testing.assert_allclose(model.input_mean_, data.values[:48].mean(0), atol=0, rtol=0)
        np.testing.assert_allclose(model.input_scale_, data.values[:48].std(0), atol=0, rtol=0)
    else:
        np.testing.assert_array_equal(model.input_mean_, np.zeros(6))
        np.testing.assert_array_equal(model.input_scale_, np.ones(6))
    # Output is exactly the forward tensor; return mean/variance are not
    # reused to denormalize a rank prediction, even in standard/both modes.
    origins = np.array([52, 53])
    predictions = model.predict_origins(data, origins)
    batch = next(iter(model._loader(model._dataset(data, origins, supervised=False))))
    with torch.no_grad():
        raw_output, _ = model._forward(batch)
    np.testing.assert_array_equal(predictions, raw_output.numpy())
    assert model.fit_metadata_["output_inverse_normalization"] is False
    assert model.config.norm is False


@pytest.mark.parametrize("architecture", ["TSMixer", "iTransformer", "Nonstationary_Transformer"])
@pytest.mark.parametrize("cross,weight", [("bucket_hinge", 0.), ("bucket_ranknet", 1.), ("ordinal_ce", .5)])
def test_native_backbones_train_continuous_rank_outputs(architecture, cross, weight):
    model = RankTargetForecaster(**options(model_name=("time_series_library.models.TSMixer.TSMixer" if architecture == "TSMixer" else "time_series_library." + architecture),
        loss_cross=cross, loss_cross_lambda=weight))
    model.forecast_fit(frame(), .75)
    forecast = model.predict_origins(frame(), [52, 53, 54])
    assert forecast.shape == (3, 3, 6) and np.isfinite(forecast).all()
    assert not np.all(forecast == np.round(forecast))
    if cross == "ordinal_ce":
        head = model.criterion.ordinal_thresholds
        assert model.model.ordinal_thresholds is head
        assert any(key.startswith("ordinal_thresholds.") for key in model.check_point)
        optimized = {id(p) for group in model.optimizer_.param_groups for p in group["params"]}
        assert all(id(p) in optimized for p in head.parameters())
        assert model.fit_metadata_["ordinal_thresholds"] == head().detach().tolist()


def test_predict_does_not_depend_on_future_returns_or_targets():
    data = frame()
    model = RankTargetForecaster(**options(input_kind="log_return", input_normalization="both"))
    model.forecast_fit(data.iloc[:48], .75)
    before = model.predict_origins(data, [52])
    data.iloc[53:] = data.iloc[53:].values[:, ::-1] * 10000 - 100
    np.testing.assert_array_equal(before, model.predict_origins(data, [52]))


def test_training_ranks_keep_float64_order_before_cast():
    data = frame()
    data.iloc[32] = np.array([1., 1. + 1e-10, 0., -.1, -.2, -.3])
    dataset = RankTargetDataset(data, np.array([31]), seq_len=8, horizon=3, label_len=4)
    assert tuple(dataset[0][4][0, :2]) == (2, 1)


def test_input_normalizer_preserves_constant_channels():
    transform = RankInputTransform("standard").fit(np.ones((10, 3)))
    np.testing.assert_array_equal(transform.scale_, np.ones(3))
    np.testing.assert_array_equal(transform.transform_window(np.ones((4, 3))), np.zeros((4, 3)))


def test_forecast_keeps_requested_origin_order_and_handles_empty():
    data = frame()
    model = RankTargetForecaster(**options())
    model.forecast_fit(data, 1.)
    prediction = model.predict_origins(data, [53, 52, 53], batch_size=2)
    np.testing.assert_allclose(prediction[0], prediction[2], rtol=1e-6, atol=1e-6)
    assert model.predict_origins(data, []).shape == (0, 3, 6)


@pytest.mark.parametrize("name,adapter", [
    ("duet.DUET", None),
    ("time_series_library.Crossformer", "transformer_adapter"),
    ("time_series_library.TimesNet", "transformer_adapter"),
])
def test_more_native_backbones_with_log_return_to_rank_and_k3(name, adapter):
    model = RankTargetForecaster(**options(model_name=name, model_adapter=adapter,
        input_kind="log_return", input_normalization="standard", loss_trade_window=3,
        loss_cross="ranknet", loss_cross_lambda=1., num_experts=2, k=1,
        fc_dropout=0., CI=True, hidden_size=8, top_k=2, num_kernels=2))
    model.forecast_fit(frame(), .75)
    assert np.isfinite(model.predict_origins(frame(), [52, 53])).all()


def test_validation_includes_auxiliary_native_objective(monkeypatch):
    model = RankTargetForecaster(**options())
    model.forecast_fit(frame(), .75)
    loader = model._loader(model.valid_dataset_)
    original = model._objective
    baseline = model._validate(loader)
    monkeypatch.setattr(model, "_objective", lambda batch: (original(batch)[0], 3.))
    assert model._validate(loader) == pytest.approx(baseline + 3.)


def test_fractional_origins_are_not_silently_truncated():
    model = RankTargetForecaster(**options())
    model.forecast_fit(frame(), .75)
    with pytest.raises(ValueError, match="inteiros"):
        model.predict_origins(frame(), [52.5])


@pytest.mark.parametrize("mode,expected", [("ordered_sample_mean", 5. / 3.), ("legacy", 1.5)])
def test_validation_mode_honors_sample_or_batch_mean(monkeypatch, mode, expected):
    model = RankTargetForecaster(**options(validation_mode=mode))
    model.model = torch.nn.Identity()
    model.criterion = torch.nn.Identity()
    model.amp_enabled_ = False
    batches = [(torch.zeros(2, 8, 6),), (torch.zeros(1, 8, 6),)]
    monkeypatch.setattr(model, "_objective", lambda batch: (torch.tensor(float(batch[0].shape[0])), 0.))
    assert model._validate(batches) == pytest.approx(expected)


def test_new_trainer_defaults_to_ordered_sample_validation():
    assert RankTargetForecaster(**options()).config.validation_mode == "ordered_sample_mean"
