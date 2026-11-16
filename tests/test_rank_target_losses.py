"""Contract tests for the independent continuous-position experiment."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ts_benchmark.baselines.ordinal_losses import attach_ordinal_parameters
from ts_benchmark.baselines.rank_target_data import (
    stable_descending_ranks_numpy, stable_descending_ranks_torch,
)
from ts_benchmark.baselines.rank_target_losses import (
    RANK_TARGET_CROSS_LOSSES, RankOrderedThresholds, RankTargetLoss,
)


LOSSES = sorted(RANK_TARGET_CROSS_LOSSES)


def criterion(name, **kwargs):
    config = dict(loss_cross=name, loss_cross_lambda=1., n_assets=6,
                  loss_bucket_count=3)
    config.update(kwargs)
    return RankTargetLoss(**config)


def positions():
    return torch.arange(1, 7, dtype=torch.float64).reshape(1, 1, 6)


def test_original_float64_returns_determine_stable_ranks_and_ties():
    raw = np.array([[[0., 0., 1e-12, -1e-12],
                     [1., 1. + 1e-12, 1., 1. - 1e-12]]])
    expected = np.array([[[2, 3, 1, 4], [2, 1, 3, 4]]])
    np.testing.assert_array_equal(stable_descending_ranks_numpy(raw), expected)
    np.testing.assert_array_equal(stable_descending_ranks_numpy(raw * 8.), expected)
    tensor = torch.tensor(raw, requires_grad=True)
    actual = stable_descending_ranks_torch(tensor)
    assert actual.dtype == torch.long
    assert not actual.requires_grad
    np.testing.assert_array_equal(actual.numpy(), expected)
    assert np.array_equal(raw, tensor.detach().numpy())


@pytest.mark.parametrize("values", [[], [np.nan, 1.], [np.inf, 1.]])
def test_invalid_return_inputs_fail_explicitly(values):
    with pytest.raises(ValueError):
        stable_descending_ranks_numpy(values)
    with pytest.raises(ValueError):
        stable_descending_ranks_torch(values)


@pytest.mark.parametrize("name", LOSSES)
def test_all_objectives_prefer_correct_positions_to_inverted_positions(name):
    target = positions()
    loss = criterion(name)
    good = loss(target.clone(), target)
    reversed_loss = loss(7. - target, target)
    assert torch.isfinite(good) and torch.isfinite(reversed_loss)
    assert good < reversed_loss


@pytest.mark.parametrize("name", LOSSES)
def test_k1_supervises_all_25_days_with_finite_gradients(name):
    target = positions().expand(2, 25, 6)
    predicted = torch.full((2, 25, 6), 3.5, dtype=torch.float64, requires_grad=True)
    loss = criterion(name)
    value = loss(predicted, target)
    value.backward()
    assert torch.isfinite(predicted.grad).all()
    assert torch.all(predicted.grad.abs().sum(-1) > 0)
    # Changing the final day alone changes exactly its share of the mean.
    amended = predicted.detach().clone()
    better_day = target[:, :1].clone()
    if name == "bucket_listnet":
        # Equal mass inside each bucket is its optimum, not six distinct
        # positions with a sharper distribution than the three-class target.
        better_day[:] = torch.tensor([1., 1., 2., 2., 3., 3.])
    amended[:, -1, :] = better_day[:, 0, :]
    per_day_improvement = loss(predicted.detach()[:, :1], target[:, :1]) - loss(better_day, target[:, :1])
    assert per_day_improvement > 0
    torch.testing.assert_close(value.detach() - loss(amended, target), per_day_improvement / 25.)


@pytest.mark.parametrize("name", LOSSES)
def test_daily_supervision_ignores_return_magnitude_and_legacy_scalers(name, monkeypatch):
    # This experiment uses positions directly, even if an old config carries
    # irrelevant return-space normalization fields.
    def forbidden(*args, **kwargs):
        raise AssertionError("financial expm1 must never process positions")

    monkeypatch.setattr(torch, "expm1", forbidden)
    loss = criterion(name, loss_inverse_norm=True,
                     loss_norm_mean=[100.] * 6, loss_norm_std=[1e8] * 6,
                     loss_score_kind="simple_return", loss_cross_score_normalization="zscore")
    raw1 = torch.tensor([.1, .02, .01, 0., -.1, -.2], dtype=torch.float64).reshape(1, 1, 6)
    raw2 = raw1 * 1e6 + 90.
    ranks1, ranks2 = stable_descending_ranks_torch(raw1), stable_descending_ranks_torch(raw2)
    predicted = torch.tensor([2.5, 1.8, 3.1, 6.2, 4.7, 5.4], dtype=torch.float64).reshape(1, 1, 6).requires_grad_()
    a = loss(predicted, ranks1, target_returns=raw1)
    grad_a, = torch.autograd.grad(a, predicted)
    b = loss(predicted, ranks2, target_returns=raw2)
    grad_b, = torch.autograd.grad(b, predicted)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(grad_a, grad_b, rtol=0, atol=0)


@pytest.mark.parametrize("temporal", ["mse", "mae", "huber"])
def test_temporal_term_compares_daily_raw_positions_not_return_units(temporal):
    target = positions()
    predicted = target + 2.
    loss = criterion("ordinal_ce", loss_cross_lambda=0., loss_temporal=temporal,
                     loss_huber_delta=.5, loss_trade_window=2,
                     loss_bucket_count=-99, n_assets=None)
    loss._cross = lambda *args: pytest.fail("lambda=0 must not evaluate cross")
    expected = {"mse": 4., "mae": 2., "huber": .875}[temporal]
    assert loss.ordinal_thresholds is None
    assert loss(predicted, target, target_returns=torch.full_like(target, float("nan"))).item() == expected


def test_pure_cross_never_evaluates_temporal_and_mixture_has_exact_weights(monkeypatch):
    target = positions()
    prediction = 7. - target
    pure = criterion("hinge", loss_cross_scale=2.3)
    mixed = criterion("hinge", loss_cross_lambda=.4, loss_cross_scale=2.3)
    temporal = F.mse_loss(prediction, target)
    expected_cross = pure(prediction, target)
    torch.testing.assert_close(mixed(prediction, target), .6 * temporal + .4 * expected_cross)
    monkeypatch.setattr(F, "mse_loss", lambda *args, **kwargs: pytest.fail("lambda=1 must not evaluate temporal"))
    torch.testing.assert_close(pure(prediction, target), expected_cross)


@pytest.mark.parametrize("name", ["ordinal_ce", "bucket_ranknet", "bucket_hinge", "bucket_listnet"])
def test_ordinal_targets_do_not_order_assets_inside_the_same_bucket(name):
    target = positions()
    swapped = torch.tensor([2., 1., 4., 3., 6., 5.], dtype=torch.float64).reshape(1, 1, 6)
    loss = criterion(name)
    torch.testing.assert_close(loss(target, target), loss(swapped, target))
    global_loss = criterion("hinge")
    assert global_loss(target, target) == 0.
    assert global_loss(swapped, target) > 0.


def test_listnet_global_target_mass_uses_rank_distances():
    target = positions()
    prediction = torch.tensor([2., 1., 4., 6., 5., 3.], dtype=torch.float64).reshape(1, 1, 6)
    tau = 2.7
    expected = -(F.softmax(-target / tau, -1) * F.log_softmax(-prediction / tau, -1)).sum(-1).mean()
    torch.testing.assert_close(criterion("listnet", loss_listnet_tau=tau)(prediction, target), expected)


def test_k2_uses_mean_positions_and_ranks_summed_raw_returns():
    # Daily ranks average to [2,2,2]; raw block returns rank [3,1,2].
    raw = torch.tensor([[[100., 2., 0.], [-99., 2., 3.]]], dtype=torch.float64, requires_grad=True)
    target = stable_descending_ranks_torch(raw)
    prediction = torch.tensor([[[3., 1., 2.], [3., 1., 2.]]], dtype=torch.float64, requires_grad=True)
    loss = criterion("pairwise_mse", n_assets=3, loss_trade_window=2)
    assert loss(prediction, target, target_returns=raw) == 0.
    reverse = (4. - prediction).detach().requires_grad_()
    wrong = loss(reverse, target, target_returns=raw)
    wrong.backward()
    assert wrong > 0 and torch.all(reverse.grad.abs().sum(-1) > 0)
    assert raw.grad is None
    with pytest.raises(ValueError, match="target_returns"):
        loss(prediction, target)


def test_k2_raw_returns_keep_float64_tie_resolution():
    raw = torch.tensor([[[.1, .1, .1], [.1, .1 + 1e-12, .1]]], dtype=torch.float64)
    target = stable_descending_ranks_torch(raw)
    expected = torch.tensor([[[2., 1., 3.], [2., 1., 3.]]], dtype=torch.float64)
    loss = criterion("pairwise_mse", n_assets=3, loss_trade_window=2)
    assert loss(expected, target, target_returns=raw) == 0.


def test_k2_uses_disjoint_blocks_and_daily_temporal_errors():
    raw = torch.tensor([[[.1, .2, .3], [.3, .2, .1],
                         [.1, .4, .2], [.1, .4, .2]]], dtype=torch.float64)
    target = stable_descending_ranks_torch(raw)
    predicted = torch.tensor([[[1., 2., 3.], [1., 2., 3.],
                               [3., 1., 2.], [3., 1., 2.]]], dtype=torch.float64)
    loss = criterion("pairwise_mse", n_assets=3, loss_trade_window=2,
                     loss_cross_lambda=.3, loss_track_components=True)
    actual = loss(predicted, target, target_returns=raw)
    # Both block rankings are perfect. Daily MSE still sees the second day.
    torch.testing.assert_close(actual, .7 * F.mse_loss(predicted, target.double()))
    assert loss.last_components["cross"] == 0.
    assert loss.last_components["H"] == 4
    assert loss.last_components["K"] == 2


def test_unequal_bucket_sizes_and_initial_cutpoints_follow_exact_rank_boundaries(monkeypatch):
    import ts_benchmark.baselines.rank_target_losses as module

    observed = {}
    original = module.ordinal_cross_entropy

    def capture(scores, classes, thresholds, temperature):
        observed["classes"] = classes.detach().clone()
        return original(scores, classes, thresholds, temperature)

    monkeypatch.setattr(module, "ordinal_cross_entropy", capture)
    loss = criterion("ordinal_ce", n_assets=7)
    target = torch.arange(1., 8.).reshape(1, 1, 7)
    loss(target, target)
    assert observed["classes"].tolist() == [[1, 1, 1, 2, 2, 3, 3]]
    torch.testing.assert_close(loss.ordinal_thresholds(), torch.tensor([-3.5, -5.5]))
    torch.testing.assert_close(RankOrderedThresholds(5, 55)(), torch.tensor([-11.5, -22.5, -33.5, -44.5]))


def test_oce_cutpoints_receive_optimizer_updates_and_checkpoint_roundtrip():
    torch.manual_seed(27)
    model = nn.Linear(2, 6)
    config = SimpleNamespace(loss_cross="ordinal_ce", loss_cross_lambda=1., loss_bucket_count=3, enc_in=6)
    loss = RankTargetLoss(config)
    attach_ordinal_parameters(model, loss)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    x = torch.tensor([[1., -.5]])
    y = positions().float()
    before = loss.ordinal_thresholds().detach().clone()
    optimizer.zero_grad()
    loss(model(x).unsqueeze(1), y).backward()
    assert loss.ordinal_thresholds.first.grad.abs().sum() > 0
    assert loss.ordinal_thresholds.raw_gaps.grad.abs().sum() > 0
    optimizer.step()
    after = loss.ordinal_thresholds().detach()
    assert not torch.equal(before, after)
    assert torch.all(after[:-1] > after[1:])
    clone_model = nn.Linear(2, 6)
    clone_loss = RankTargetLoss(config)
    attach_ordinal_parameters(clone_model, clone_loss)
    clone_model.load_state_dict(model.state_dict())
    torch.testing.assert_close(clone_loss(clone_model(x).unsqueeze(1), y), loss(model(x).unsqueeze(1), y))
    assert "ordinal_thresholds.first" in model.state_dict()


@pytest.mark.parametrize("bad", [torch.tensor([[[1., 1., 3., 4., 5., 6.]]]),
                                  torch.tensor([[[1., 2., 3., 4., 5., 6.1]]])])
def test_targets_must_be_true_integer_rank_permutations(bad):
    with pytest.raises(ValueError, match="permutacao"):
        criterion("hinge")(positions(), bad)


def test_incompatible_shapes_windows_and_missing_head_size_fail():
    target = positions()
    with pytest.raises(ValueError, match="dividir H"):
        criterion("hinge", loss_trade_window=2)(target, target)
    with pytest.raises(ValueError, match="n_assets"):
        criterion("ordinal_ce", n_assets=None)
    with pytest.raises(ValueError, match="bucket_count"):
        criterion("bucket_hinge", loss_bucket_count=7)
    with pytest.raises(ValueError, match="finita"):
        criterion("hinge")(torch.full_like(target, float("inf")), target)
    with pytest.raises(ValueError, match="formato"):
        criterion("hinge")(target.squeeze(0), target.squeeze(0))


@pytest.mark.parametrize("name", sorted(RANK_TARGET_CROSS_LOSSES - {"ordinal_ce", "bucket_ranknet", "bucket_hinge", "bucket_listnet"}))
def test_single_asset_global_cross_is_differentiable_zero(name):
    predicted = torch.tensor([[[2.]]], requires_grad=True)
    target = torch.ones_like(predicted)
    value = criterion(name, n_assets=1)(predicted, target)
    value.backward()
    assert value.item() == 0 and predicted.grad.item() == 0
