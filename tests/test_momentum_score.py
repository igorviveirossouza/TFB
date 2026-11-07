"""Numerical and integration contracts for the optional momentum residual."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ts_benchmark.baselines.momentum_score import MomentumScore, validate_momentum_config
from ts_benchmark.baselines.composite_trading_loss_v3 import CompositeTradingLossV3, build_loss
from ts_benchmark.baselines.composite_trading_loss_v2 import CROSS_LOSSES
from ts_benchmark.baselines.proximity_losses import PROXIMITY_LOSSES


def config(**overrides):
    values = dict(momentum_mode="residual_output", seq_len=4, momentum_eta=0.2,
                  loss="composite_trading", loss_cross_lambda=0.8,
                  loss_trade_window=2, loss_data_kind="log_return", norm=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def loss(**overrides):
    values = dict(trade_window=2, data_kind="log_return", score_kind="simple_return",
                  cross_loss="ranknet", cross_lambda=0.8, inverse_norm=False,
                  momentum_mode="residual_output")
    values.update(overrides)
    return CompositeTradingLossV3(**values)


def inputs():
    history = torch.tensor([[[.90, .20, -.80, .31], [.02, -.01, .04, .00],
                             [.01, .03, -.02, .01], [-.01, .02, .01, -.03]]], dtype=torch.float64)
    pred = torch.tensor([[[.01, -.02, .04, -.01], [.02, .01, -.01, .03],
                          [.01, .01, -.02, .03], [-.01, .02, .01, -.04]]], dtype=torch.float64)
    target = torch.tensor([[[-.01, .02, .03, .04], [.03, -.04, .02, .01],
                            [.03, .02, -.01, .04], [.02, -.01, .03, .00]]], dtype=torch.float64)
    return history, pred, target


def test_numpy_oracle_last_lookback_and_zero_initialization():
    history, scores, _ = inputs()
    state = torch.random.get_rng_state()
    module = MomentumScore(3, eta=.7).double()
    assert torch.equal(state, torch.random.get_rng_state())
    summed = history.numpy()[:, -3:, :].sum(axis=1)
    expected_m = (summed - summed.mean(axis=-1, keepdims=True)) / summed.std(axis=-1, keepdims=True)
    result = module(history, scores)
    np.testing.assert_allclose(result["momentum"].numpy(), expected_m)
    np.testing.assert_allclose(result["final"].detach().numpy(), np.broadcast_to(expected_m[:, None, :], scores.shape))
    assert result["penalty"].item() == 0
    assert torch.count_nonzero(result["correction"]) == 0

    with torch.no_grad():
        module.gain.fill_(.3)
    expected_delta = .3 * (scores.numpy() - scores.numpy().mean(axis=-1, keepdims=True)) / scores.numpy().std(axis=-1, keepdims=True)
    result = module(history, scores)
    np.testing.assert_allclose(result["correction"].detach().numpy(), expected_delta)
    np.testing.assert_allclose(result["final"].detach().numpy(), expected_m[:, None, :] + expected_delta)
    assert result["penalty"].item() == pytest.approx(.7 * np.mean(expected_delta ** 2))


def test_penalty_controls_amplitude_and_has_nonzero_gain_gradient():
    history, scores, _ = inputs()
    module = MomentumScore(3, eta=.4).double()
    penalties = []
    for gain in (0., .5, 1., 2.):
        with torch.no_grad():
            module.gain.fill_(gain)
        penalty = module(history, scores)["penalty"]
        penalties.append(penalty.item())
    assert penalties == pytest.approx([0, .1, .4, 1.6])
    penalty.backward()
    assert module.gain.grad.item() == pytest.approx(2 * .4 * 2)


def test_gain_learns_first_and_backbone_receives_gradient_after_gain_moves():
    history, prediction, target = inputs()
    model = MomentumScore(3).double()
    criterion = loss(cross_lambda=1.)
    pred = prediction.clone().requires_grad_()
    criterion(pred, target, momentum_history=history, momentum_module=model).backward()
    assert torch.isfinite(model.gain.grad)
    assert abs(model.gain.grad.item()) > 1e-8
    assert torch.count_nonzero(pred.grad) == 0
    model.zero_grad()
    with torch.no_grad():
        model.gain.fill_(.4)
    pred = prediction.clone().requires_grad_()
    criterion(pred, target, momentum_history=history, momentum_module=model).backward()
    assert torch.isfinite(pred.grad).all()
    assert torch.count_nonzero(pred.grad) > 0


@pytest.mark.parametrize("assets", [1, 4])
def test_constant_cross_sections_and_ties_have_finite_gradients(assets):
    history = torch.ones(2, 4, assets, dtype=torch.float64)
    scores = torch.ones(2, 3, assets, dtype=torch.float64, requires_grad=True)
    module = MomentumScore(3, eta=.2).double()
    with torch.no_grad():
        module.gain.fill_(.3)
    result = module(history, scores)
    (result["final"].square().mean() + result["penalty"]).backward()
    assert torch.equal(result["final"], torch.zeros_like(scores))
    assert torch.isfinite(scores.grad).all()
    assert torch.isfinite(module.gain.grad)


def test_history_before_lookback_and_future_targets_do_not_change_momentum():
    history, pred, target = inputs()
    module = MomentumScore(3)
    before = module(history, pred)["final"]
    altered = history.clone()
    altered[:, 0, :] = 1e5
    torch.testing.assert_close(module(altered, pred)["final"], before, rtol=0, atol=0)
    criterion = loss()
    captured = []
    criterion._cross_loss = lambda predicted, realized: captured.append(predicted.detach().clone()) or predicted.sum() * 0
    criterion(pred, target, momentum_history=history, momentum_module=module)
    criterion(pred, target + 5, momentum_history=history, momentum_module=module)
    torch.testing.assert_close(captured[0], captured[1], rtol=0, atol=0)


def test_inverse_scaler_raw_target_units_and_temporal_term_are_preserved():
    history, pred_raw, target_raw = inputs()
    mean = torch.tensor([.01, -.02, .03, .01], dtype=torch.float64)
    scale = torch.tensor([.02, .04, .01, .03], dtype=torch.float64)
    pred, target, scaled_history = [(x - mean) / scale for x in (pred_raw, target_raw, history)]
    criterion = loss(inverse_norm=True, normalizer_mean=mean, normalizer_scale=scale,
                     track_components=True, cross_loss="ranknet_proximity", cross_mask=["cauchy", .025, "natural"])
    # Buffers are stored as float32 by the existing aggregator; convert to double
    # and use their actual values in the independent restoration oracle.
    criterion.double()
    restored_history = scaled_history * criterion.aggregator.normalizer_scale + criterion.aggregator.normalizer_mean
    restored_pred = pred * criterion.aggregator.normalizer_scale + criterion.aggregator.normalizer_mean
    restored_target = target * criterion.aggregator.normalizer_scale + criterion.aggregator.normalizer_mean
    expected_target = torch.expm1(restored_target.reshape(1, 2, 2, 4).sum(dim=2))
    expected_pred = torch.expm1(restored_pred.reshape(1, 2, 2, 4).sum(dim=2))
    module = MomentumScore(3, eta=.6).double()
    with torch.no_grad():
        module.gain.fill_(.2)
    combined = module(restored_history, expected_pred)
    expected_cross = criterion._cross_loss(combined["final"], expected_target)
    temporal = (pred - target).square().mean()
    actual = criterion(pred, target, momentum_history=scaled_history, momentum_module=module)
    expected = .2 * temporal + .8 * expected_cross + combined["penalty"]
    torch.testing.assert_close(actual, expected)
    assert criterion.last_components["temporal"] == pytest.approx(temporal.item())
    assert criterion.last_components["momentum_penalty"] == pytest.approx(combined["penalty"].item())


@pytest.mark.parametrize("cross_loss", sorted(CROSS_LOSSES | set(PROXIMITY_LOSSES)))
@pytest.mark.parametrize("normalization", ["none", "zscore"])
def test_disabled_mode_is_exactly_original_loss_and_gradients(cross_loss, normalization):
    _, predicted, target = inputs()
    criterion = loss(momentum_mode="off", cross_loss=cross_loss, cross_mask=["cauchy", .025, "natural"],
                     cross_score_normalization=normalization)
    pred = predicted.clone().requires_grad_()
    actual = criterion(pred, target)
    actual_grad, = torch.autograd.grad(actual, pred)
    # Original V3 arithmetic before the residual branch was introduced.
    pred = predicted.clone().requires_grad_()
    temporal = criterion._temporal_loss(pred, target)
    scores, target_scores = criterion.aggregator(pred, target)
    expected = (1 - criterion.cross_lambda) * temporal + criterion.cross_lambda * criterion.cross_scale * criterion._cross_loss(scores, target_scores)
    expected_grad, = torch.autograd.grad(expected, pred)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("changes", [
    {"momentum_mode": "input"}, {"momentum_lookback": 0}, {"momentum_lookback": 2.5},
    {"momentum_lookback": float("inf")}, {"momentum_eta": -1}, {"momentum_eta": float("nan")},
    {"momentum_eta": float("inf")}, {"loss": "mse"}, {"loss_cross_lambda": 0},
    {"loss_cross_lambda": 1.1}, {"loss_data_kind": "price"}, {"loss_score_kind": "price"},
    {"norm": True, "loss_inverse_norm": False}, {"target_mode": "cumulative"},
])
def test_incompatible_configs_rejected_before_build(changes):
    with pytest.raises(ValueError):
        build_loss(config(**changes))


def test_valid_configuration_and_inactive_backward_compatibility():
    assert validate_momentum_config(config()) == {"mode": "residual_output", "lookback": 4, "eta": .2}
    assert validate_momentum_config(config(momentum_lookback="6"))["lookback"] == 6
    assert build_loss(config(loss_cross_lambda=1.)).momentum_mode == "residual_output"
    disabled = config(momentum_mode="off", momentum_lookback=-1, momentum_eta=float("nan"), loss="mse")
    assert validate_momentum_config(disabled) == {"mode": "off", "lookback": None, "eta": 0.}
    assert isinstance(build_loss(disabled), torch.nn.MSELoss)


@pytest.mark.parametrize("cross_loss", ["mse", "pairwise_mse", "pairwise_mse_proximity"])
def test_dimensionless_momentum_requires_normalization_for_magnitude_losses(cross_loss):
    with pytest.raises(ValueError, match="losses de magnitude"):
        build_loss(config(loss_cross=cross_loss, loss_cross_score_normalization="none"))
    with pytest.raises(ValueError, match="losses de magnitude"):
        loss(cross_loss=cross_loss, cross_score_normalization="none")


def test_context_and_shapes_fail_explicitly():
    history, pred, target = inputs()
    module = MomentumScore(3)
    with pytest.raises(ValueError, match="exige momentum_history"):
        loss()(pred, target)
    with pytest.raises(ValueError, match="momentum_mode=off"):
        loss(momentum_mode="off")(pred, target, momentum_history=history, momentum_module=module)
    with pytest.raises(ValueError, match="insuficiente"):
        module(history[:, :2], pred)
    with pytest.raises(ValueError, match="mesmo batch"):
        module(history, pred[:, :, :2])
    with pytest.raises(ValueError, match="espera history_raw"):
        module(history.squeeze(0), pred)


def test_state_payload_and_checkpoint_roundtrip():
    history, pred, _ = inputs()
    module = MomentumScore(3, eta=.2).double()
    with torch.no_grad():
        module.gain.fill_(.37)
    payload = json.loads(json.dumps(module.state_payload("simple_returns", ["A", "B", "C", "D"]), allow_nan=False))
    assert payload["version"] == "momentum_residual_v1"
    assert payload["gain"] == pytest.approx(.37)
    assert payload["score_kind"] == "simple_return"
    restored = MomentumScore(payload["lookback"], payload["eta"], payload["eps"]).double()
    restored.load_state_dict(module.state_dict())
    torch.testing.assert_close(restored(history, pred)["final"], module(history, pred)["final"], rtol=0, atol=0)


def test_half_precision_constant_sections_remain_finite():
    result = MomentumScore(3)(torch.ones(1, 3, 4, dtype=torch.float16), torch.ones(1, 2, 4, dtype=torch.float16))
    assert torch.isfinite(result["final"]).all()
    assert torch.isfinite(result["penalty"])
