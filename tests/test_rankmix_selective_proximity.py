"""Selective financial-kernel weighting preserves mixture and legacy contracts."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ts_benchmark.baselines.ranking_mixture import (
    RANK_MIX_COMPONENTS, validate_rank_mix_proximity_components,
)
from ts_benchmark.baselines.hybrid_ranking_losses import (
    ranking_mixture_loss, regional_listnet_loss, regional_listmle_loss,
)
from ts_benchmark.baselines.listwise_ranking_losses import (
    financial_pair_weights, listnet_proximity_loss, listmle_loss, approxndcg_loss,
)
from ts_benchmark.baselines.composite_trading_loss_v3 import CompositeTradingLossV3, build_loss


def sample():
    pred = torch.tensor([[.7, -.4, .2, .9, -.1, .4], [.1, .9, -.3, .5, -.6, -.2]],
                        dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[.03, -.01, .02, .06, -.02, .01], [.01, .04, -.04, .02, -.02, .03]],
                          dtype=torch.float64)
    weights = financial_pair_weights(target, ['cauchy', .025], pred.dtype)
    return pred, target, weights


def criterion(name='rankmix_proximity', **kwargs):
    config = dict(trade_window=1, data_kind='log_return', score_kind='log_return',
                  cross_loss=name, cross_lambda=1., inverse_norm=False,
                  cross_score_normalization='none', cross_mask=['cauchy', .025])
    config.update(kwargs)
    return CompositeTradingLossV3(**config)


@pytest.mark.parametrize('value', [[], (), 'listnet', ['other'], ['listnet_proximity'],
    [True], [None], [1], ['listnet', 'LISTNET'], ['listnet', ' approxndcg ', 'approxndcg']])
def test_invalid_selectors_rejected(value):
    with pytest.raises(ValueError, match='loss_rank_mix_proximity_components'):
        validate_rank_mix_proximity_components(value)
    pred, target, weights = sample()
    with pytest.raises(ValueError, match='loss_rank_mix_proximity_components'):
        ranking_mixture_loss(pred, target, pair_weights=weights, proximity_components=value)
    with pytest.raises(ValueError, match='loss_rank_mix_proximity_components'):
        criterion(rank_mix_proximity_components=value)


def test_selector_canonical_order_and_legacy_default():
    assert validate_rank_mix_proximity_components() == list(RANK_MIX_COMPONENTS)
    assert validate_rank_mix_proximity_components(('ApproxNDCG', ' LISTNET ')) == ['listnet', 'approxndcg']


@pytest.mark.parametrize('mode', ['global', 'regional'])
def test_half_listnet_proximity_half_original_listmle_exact(mode):
    pred, target, weights = sample()
    if mode == 'global':
        expected = (.5 * listnet_proximity_loss(pred, target, pair_weights=weights)
                    + .5 * listmle_loss(pred, target))
    else:
        expected = (.5 * regional_listnet_loss(pred, target, pair_weights=weights, region='top')
                    + .5 * regional_listmle_loss(pred, target, region='bottom'))
    actual = ranking_mixture_loss(pred, target, weights=[.5, .5, 0.], mode=mode,
                                  pair_weights=weights, proximity_components=['listnet'])
    assert torch.equal(actual, expected)
    actual_grad, = torch.autograd.grad(actual, pred, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected, pred)
    assert torch.isfinite(actual_grad).all()
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-14, rtol=1e-13)


@pytest.mark.parametrize('mode', ['global', 'regional'])
@pytest.mark.parametrize('mix', [[.5, .5, 0.], [.2, .3, .5]])
def test_omitted_selector_exactly_matches_all_components(mode, mix):
    pred, target, weights = sample()
    kwargs = dict(mode=mode, weights=mix, pair_weights=weights)
    old = ranking_mixture_loss(pred, target, **kwargs)
    explicit_all = ranking_mixture_loss(pred, target, proximity_components=list(RANK_MIX_COMPONENTS), **kwargs)
    assert torch.equal(old, explicit_all)
    old_grad, = torch.autograd.grad(old, pred, retain_graph=True)
    all_grad, = torch.autograd.grad(explicit_all, pred)
    assert torch.equal(old_grad, all_grad)


@pytest.mark.parametrize('name,index,function', [('listmle', 1, listmle_loss),
                                               ('approxndcg', 2, approxndcg_loss)])
def test_other_components_can_be_selected(name, index, function):
    pred, target, weights = sample()
    mix = [0., 0., 0.]
    mix[index] = 1.
    expected = function(pred, target, pair_weights=weights)
    actual = ranking_mixture_loss(pred, target, weights=mix, pair_weights=weights,
                                  proximity_components=[name])
    assert torch.equal(actual, expected)
    assert torch.isfinite(torch.autograd.grad(actual, pred)[0]).all()


@pytest.mark.parametrize('mode', ['global', 'regional'])
def test_zero_weight_selected_component_does_not_modify_active_original(mode):
    pred, target, weights = sample()
    actual = ranking_mixture_loss(pred, target, weights=[0., 1., 0.], mode=mode,
                                  pair_weights=weights, proximity_components=['listnet'])
    expected = ranking_mixture_loss(pred, target, weights=[0., 1., 0.], mode=mode)
    assert torch.equal(actual, expected)
    assert torch.isfinite(torch.autograd.grad(actual, pred)[0]).all()


@pytest.mark.parametrize('mode', ['global', 'regional'])
@pytest.mark.parametrize('lam', [0., .5, .8, 1.])
def test_outer_lambda_preserves_temporal_and_selective_coefficients(mode, lam):
    pred, target, weights = sample()
    c = criterion(cross_lambda=lam, rank_mix_mode=mode, rank_mix_proximity_components=['listnet'])
    cross = ranking_mixture_loss(pred, target, mode=mode, pair_weights=weights,
                                 proximity_components=['listnet'])
    expected = (1 - lam) * F.mse_loss(pred, target) + lam * cross
    actual = c(pred[:, None], target[:, None])
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-13)
    assert torch.isfinite(torch.autograd.grad(actual, pred)[0]).all()


def test_original_rankmix_always_ignores_kernel_selector():
    pred, target, _ = sample()
    plain = criterion('rankmix')._cross_loss(pred[:, None], target[:, None])
    selected = criterion('rankmix', rank_mix_proximity_components=['listnet'])._cross_loss(pred[:, None], target[:, None])
    irrelevant = criterion('rankmix', rank_mix_proximity_components=['invalid'])._cross_loss(pred[:, None], target[:, None])
    assert torch.equal(plain, selected) and torch.equal(plain, irrelevant)


def test_factory_forwards_selective_config_without_changing_mixture_identity():
    c = build_loss(SimpleNamespace(loss='composite_trading', loss_cross='rankmix_proximity',
        loss_trade_window=1, loss_cross_lambda=1., norm=False,
        loss_score_kind='log_return', loss_cross_score_normalization='none',
        loss_cross_mask=['cauchy', .025], loss_rank_mix_weights=[.5, .5, 0.],
        loss_rank_mix_mode='global', loss_rank_mix_proximity_components=['listnet']))
    assert c.rank_mix_proximity_components == ['listnet']
    assert c.rank_mix_config == dict(weights=[.5, .5, 0.], mode='global',
                                    regions=['top', 'bottom', 'top'], fraction=1. / 3.)
    pred, target, _ = sample()
    expected = criterion(rank_mix_proximity_components=['listnet'])(pred[:, None], target[:, None])
    actual = c(pred[:, None], target[:, None])
    assert torch.equal(actual, expected)


def test_kernel_weights_are_not_learned():
    pred, target, weights = sample()
    weights.requires_grad_(True)
    actual = ranking_mixture_loss(pred, target, pair_weights=weights,
                                  proximity_components=['listnet'])
    pred_grad, weight_grad = torch.autograd.grad(actual, (pred, weights), allow_unused=True)
    assert torch.isfinite(pred_grad).all()
    assert weight_grad is None
