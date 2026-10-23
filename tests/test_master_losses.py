"""MASTER reaches every current TFB objective without changing its loss formulas."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ts_benchmark.baselines.composite_trading_loss_v2 import CROSS_LOSSES, TEMPORAL_LOSSES
from ts_benchmark.baselines.composite_trading_loss_v3 import build_loss
from ts_benchmark.baselines.master.network import MasterNetwork
from ts_benchmark.baselines.proximity_losses import PROXIMITY_LOSSES


MEAN = [0.001, -0.002, 0.0005]
SCALE = [0.01, 0.02, 0.03]
HORIZON_BLOCKS = [(1, 1), (5, 5), (25, 5)]
CROSS_CASES = [pytest.param(name, None, id=name) for name in sorted(CROSS_LOSSES)]
CROSS_CASES += [
    pytest.param(name, (kernel, 0.025, mode), id=f"{name}-{kernel}-{mode}")
    for name in sorted(PROXIMITY_LOSSES)
    for kernel in ("uniform", "gaussian", "cauchy")
    for mode in ("natural", "complementary")
]


@pytest.fixture(scope="module", autouse=True)
def bounded_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def network_case(horizon):
    # Separate origins contain whole cross-sections; missing feature channels
    # exercise both the value and missingness paths through the real network.
    with torch.random.fork_rng():
        torch.manual_seed(2026)
        network = MasterNetwork(4, 2, lookback=4, horizon=horizon,
                                d_model=8, t_heads=2, s_heads=2, dropout=0)
        assets = torch.randn(2, 4, 3, 4)
        market = torch.randn(2, 4, 2)
        asset_observed = torch.ones_like(assets, dtype=torch.bool)
        market_observed = torch.ones_like(market, dtype=torch.bool)
        asset_observed[0, 1, 0, 2] = False
        market_observed[1, -1, 0] = False
        prediction = network(assets, market, asset_observed, market_observed)
        target = torch.randn(2, horizon, 3) * 0.7
    prediction.retain_grad()
    return network, prediction, target


def criterion(temporal, *, cross=None, mask=None, k=1):
    config = SimpleNamespace(
        loss="composite_trading" if cross is not None else temporal,
        loss_temporal=temporal, loss_cross=cross,
        loss_trade_window=k, loss_cross_lambda=0.5,
        loss_data_kind="log_return", loss_score_kind="simple_return",
        loss_cross_score_normalization="zscore",
        loss_cross_mask=mask, loss_track_components=True,
        norm=True, loss_inverse_norm=True,
    )
    return build_loss(config, normalizer_mean=MEAN, normalizer_scale=SCALE)


def assert_gradients(network, prediction, loss):
    assert prediction.shape[1] == network.horizon
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    # Every output step contributes to training; no implicit H=1 selection or
    # last-horizon-only reduction is allowed on the model output.
    assert (prediction.grad.abs().sum(dim=(0, 2)) > 0).all()
    for name, parameter in network.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for block in (network.prediction, network.market_gate,
                  network.temporal, network.cross_asset):
        assert any(parameter.grad.abs().sum() > 0 for parameter in block.parameters())


@pytest.mark.parametrize("temporal", sorted(TEMPORAL_LOSSES))
@pytest.mark.parametrize("horizon", [1, 5, 25])
def test_temporal_objectives_train_all_master_horizons(temporal, horizon):
    network, prediction, target = network_case(horizon)
    loss = criterion(temporal)(prediction, target)
    assert prediction.shape == (2, horizon, 3)
    assert_gradients(network, prediction, loss)


@pytest.mark.parametrize("cross,mask", CROSS_CASES)
@pytest.mark.parametrize("temporal", sorted(TEMPORAL_LOSSES))
@pytest.mark.parametrize("horizon,k", HORIZON_BLOCKS)
def test_composite_and_proximity_objectives_train_master(temporal, cross, mask, horizon, k):
    network, prediction, target = network_case(horizon)
    objective = criterion(temporal, cross=cross, mask=mask, k=k)
    loss = objective(prediction, target)
    assert_gradients(network, prediction, loss)
    assert objective.last_components["H"] == horizon
    assert objective.last_components["K"] == k
    assert objective.last_components["n_blocks"] == horizon // k
    assert objective.last_components["cross_steps"] == horizon
    assert objective.last_components["cross_tail_steps"] == 0


@pytest.mark.parametrize("horizon,k", [(5, 1), (25, 25)])
def test_financial_scores_receive_inverse_normalized_complete_path(horizon, k, monkeypatch):
    network, prediction, target = network_case(horizon)
    objective = criterion("mse", cross="ranknet_proximity",
                          mask=("gaussian", 0.025, "natural"), k=k)
    observed = {}
    original = objective._cross_loss

    def capture(pred_scores, target_scores):
        observed["prediction"] = pred_scores.detach().clone()
        observed["target"] = target_scores.detach().clone()
        return original(pred_scores, target_scores)

    monkeypatch.setattr(objective, "_cross_loss", capture)
    loss = objective(prediction, target)
    mean, scale = torch.tensor(MEAN), torch.tensor(SCALE)
    # Cross scores are financially aggregated raw log returns. The cross loss
    # applies its own z-score only after receiving these complete block scores.
    expected_prediction = torch.expm1(
        (prediction.detach() * scale + mean).reshape(2, horizon // k, k, 3).sum(dim=2)
    )
    expected_target = torch.expm1(
        (target * scale + mean).reshape(2, horizon // k, k, 3).sum(dim=2)
    )
    torch.testing.assert_close(observed["prediction"], expected_prediction)
    torch.testing.assert_close(observed["target"], expected_target)
    assert_gradients(network, prediction, loss)


@pytest.mark.parametrize("cross,mask", [
    ("pairwise_mse", None),
    ("hinge_proximity", ("cauchy", 0.025, "complementary")),
])
def test_nondivisible_trade_window_is_rejected_for_master_outputs(cross, mask):
    _, prediction, target = network_case(5)
    with pytest.raises(ValueError, match="divisor exato"):
        criterion("mse", cross=cross, mask=mask, k=2)(prediction, target)
    assert prediction.grad is None
