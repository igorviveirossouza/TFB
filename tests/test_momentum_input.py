"""Causal feature and matched native architecture tests for momentum input."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ts_benchmark.baselines.momentum_input import (
    MomentumInputProjection, momentum_input_features, validate_momentum_input_config,
)
from ts_benchmark.baselines.time_series_library.adapters_for_transformers import MODEL_HYPER_PARAMS
from ts_benchmark.baselines.time_series_library.models.Nonstationary_Transformer import Nonstationary_Transformer


@pytest.fixture(autouse=True)
def cpu_single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(mode="off", **kwargs):
    values = dict(MODEL_HYPER_PARAMS)
    values.update(seq_len=8, pred_len=2, label_len=4, enc_in=4, dec_in=4, c_out=4,
                  d_model=8, d_ff=16, n_heads=2, e_layers=1, d_layers=1, dropout=0.,
                  p_hidden_dims=[8, 8], p_hidden_layers=2, momentum_input_mode=mode)
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_momentum_uses_only_selected_observed_tail_and_restores_outer_scaler():
    raw = torch.tensor([[[100., -100., 12.], [1., 2., 3.], [.5, -1., 2.]]], dtype=torch.float64)
    expected = torch.tensor([[1.5, 1., 5.]], dtype=torch.float64)
    expected = (expected - expected.mean(-1, keepdim=True)) / expected.std(-1, keepdim=True, unbiased=False)
    direct = momentum_input_features(raw, 2)
    torch.testing.assert_close(direct, expected)
    mean = np.array([.1, .2, .3]); scale = np.array([.25, 3., 2.])
    normalized = (raw - torch.tensor(mean)) / torch.tensor(scale)
    torch.testing.assert_close(momentum_input_features(normalized, 2, mean=mean, scale=scale), expected)
    raw[:, 0] = -777.
    torch.testing.assert_close(momentum_input_features(raw, 2), direct)
    assert not momentum_input_features(raw.requires_grad_(), 2).requires_grad


def test_constant_cross_section_is_zero_and_half_does_not_underflow_epsilon():
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        features = momentum_input_features(torch.ones(2, 5, 4, dtype=dtype), 5)
        assert torch.isfinite(features).all()
        torch.testing.assert_close(features, torch.zeros_like(features))


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "nan"])
def test_bad_lookback_rejected(bad):
    with pytest.raises(ValueError):
        validate_momentum_input_config({"momentum_input_mode": "feature", "momentum_input_lookback": bad})


@pytest.mark.parametrize("settings", [
    {"momentum_mode": "residual_output"}, {"loss_data_kind": "price"}, {"target_mode": "cumulative"},
])
def test_invalid_input_contract_rejected(settings):
    with pytest.raises(ValueError):
        validate_momentum_input_config(dict(momentum_input_mode="feature", seq_len=8, **settings))


def test_inactive_input_configuration_ignored_without_rng_or_parameters():
    state = torch.random.get_rng_state().clone()
    assert validate_momentum_input_config({"momentum_input_mode": "off", "momentum_input_lookback": -7}) == {"mode": "off", "lookback": None}
    assert torch.equal(state, torch.random.get_rng_state())
    native = Nonstationary_Transformer(config())
    assert native.momentum_input is None
    assert not any("momentum_input" in name for name in native.state_dict())


def test_zero_and_feature_models_have_identical_initial_weights_and_native_weights_match():
    models = {}
    for mode in ("off", "zero", "feature"):
        torch.manual_seed(17)
        models[mode] = Nonstationary_Transformer(config(mode)).eval()
    for key, value in models["off"].state_dict().items():
        torch.testing.assert_close(value, models["zero"].state_dict()[key], rtol=0, atol=0)
    for key, value in models["zero"].state_dict().items():
        torch.testing.assert_close(value, models["feature"].state_dict()[key], rtol=0, atol=0)
    x = torch.randn(2, 8, 4); dec = torch.randn(2, 6, 4)
    features = torch.tensor([[1., -1., .5, -.5], [-1., .4, 1., -.4]])
    base = models["off"](x, None, dec, None)
    zero = models["zero"](x, None, dec, None, momentum_features=features)
    torch.testing.assert_close(base, zero, rtol=0, atol=0)
    actual = models["feature"](x, None, dec, None, momentum_features=features)
    assert not torch.allclose(actual, zero)
    actual.square().mean().backward()
    projection = models["feature"].momentum_input.projection
    assert projection.weight.grad is not None and projection.weight.grad.norm() > 0
    assert models["feature"].enc_embedding.value_embedding.tokenConv.weight.grad.norm() > 0
    zero.square().mean().backward()
    torch.testing.assert_close(models["zero"].momentum_input.projection.weight.grad,
                               torch.zeros_like(models["zero"].momentum_input.projection.weight))


def test_feature_injection_survives_temporal_centering_and_is_not_fixed_output_momentum():
    torch.manual_seed(19)
    model = Nonstationary_Transformer(config("feature")).eval()
    x = torch.randn(2, 8, 4); dec = torch.zeros(2, 6, 4)
    f = torch.tensor([[1., 0., -1., 0.], [0., 1., 0., -1.]])
    captures = []
    hook = model.encoder.register_forward_pre_hook(lambda module, args: captures.append(args[0].detach().clone()))
    first = model(x, None, dec, None, momentum_features=f)
    second = model(x, None, dec, None, momentum_features=2 * f)
    hook.remove()
    difference = captures[1] - captures[0]
    expected = model.momentum_input(f).expand_as(difference)
    torch.testing.assert_close(difference, expected, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(first, second)
    # There is no residual gate, penalty or output addition. Zeroing the learned
    # input projection removes all dependence on the supplied momentum feature.
    with torch.no_grad():
        model.momentum_input.projection.weight.zero_()
    torch.testing.assert_close(model(x, None, dec, None, momentum_features=f),
                               model(x, None, dec, None, momentum_features=7 * f), rtol=0, atol=0)
    assert not any("gain" in name or "momentum_score" in name for name, _ in model.named_parameters())


def test_momentum_projection_missing_or_wrong_feature_is_rejected():
    model = Nonstationary_Transformer(config("feature"))
    x = torch.randn(2, 8, 4); dec = torch.zeros(2, 6, 4)
    with pytest.raises(ValueError, match="exige momentum_features"):
        model(x, None, dec, None)
    with pytest.raises(ValueError, match="shape"):
        model(x, None, dec, None, momentum_features=torch.zeros(2, 5))
    with pytest.raises(ValueError, match="mesmo batch"):
        model(x, None, dec, None, momentum_features=torch.zeros(1, 4))
    with pytest.raises(ValueError, match="forecasting"):
        Nonstationary_Transformer(config("feature", task_name="anomaly_detection"))
