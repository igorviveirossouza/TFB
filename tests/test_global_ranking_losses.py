"""Full-list rank objective: mathematical, numerical and hybrid contracts."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ts_benchmark.baselines.global_ranking_losses import (
    soft_ranks, target_midranks, spearman_loss,
)
from ts_benchmark.baselines.composite_trading_loss_v3 import (
    CompositeTradingLossV3, build_loss,
)


def criterion(name="spearman", **kwargs):
    config = dict(trade_window=1, data_kind="log_return", score_kind="log_return",
                  cross_loss=name, cross_lambda=1., inverse_norm=False,
                  cross_score_normalization="none", cross_mask=["cauchy", .025])
    config.update(kwargs)
    return CompositeTradingLossV3(**config)


def sample(requires_grad=False):
    p = torch.tensor([[.4, -.2, .15, .3], [-.1, .45, .8, .23]],
                     dtype=torch.float64, requires_grad=requires_grad)
    t = torch.tensor([[.03, .01, .01, -.02], [.1, .04, .06, .01]], dtype=torch.float64)
    return p, t


def independent_ranks(p, t, tau):
    """Explicit scalar loops, no broadcasting or helper reuse."""
    a = torch.stack([1 + sum(torch.sigmoid((p[j] - p[i]) / tau)
                             for j in range(len(p)) if j != i) for i in range(len(p))])
    b = t.new_tensor([1 + sum(float(t[j] > t[i]) + .5 * float(t[j] == t[i])
                             for j in range(len(t)) if j != i) for i in range(len(t))])
    return a, b


def test_exact_midranks_and_soft_limit():
    scores = torch.tensor([[9., 2., 2., -1.]], dtype=torch.float64)
    expected = torch.tensor([[1., 2.5, 2.5, 4.]], dtype=torch.float64)
    torch.testing.assert_close(target_midranks(scores), expected)
    torch.testing.assert_close(soft_ranks(scores, .001), expected)
    torch.testing.assert_close(soft_ranks(scores, 1).sum(1), scores.new_tensor([10.]))


@pytest.mark.parametrize("tau", [.1, 1., 3.])
def test_plain_matches_pearson_of_ranks(tau):
    p, t = sample(True)
    expected = []
    for pred, target in zip(p, t):
        a, b = independent_ranks(pred, target, tau)
        expected.append(1 - torch.corrcoef(torch.stack([a, b]))[0, 1])
    expected = torch.stack(expected).mean()
    actual = spearman_loss(p, t, tau=tau)
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-12)
    ga, = torch.autograd.grad(actual, p, retain_graph=True)
    ge, = torch.autograd.grad(expected, p)
    torch.testing.assert_close(ga, ge, atol=1e-12, rtol=1e-11)


def test_proximity_matches_independent_raw_cauchy_pair_correlation():
    p, t = sample(True)
    expected_rows = []
    for pred, target in zip(p, t):
        a, b = independent_ranks(pred, target, 1.)
        num, pa, pb = 0., 0., 0.
        for i in range(len(pred)):
            for j in range(i+1, len(pred)):
                weight = 1 / (1 + ((target[i] - target[j]) / .025)**2)
                da, db = a[i] - a[j], b[i] - b[j]
                num = num + weight * da * db
                pa = pa + weight * da.square()
                pb = pb + weight * db.square()
        expected_rows.append(1 - num / (pa.sqrt() * pb.sqrt()))
    expected = torch.stack(expected_rows).mean()
    actual = criterion("spearman_proximity")._cross_loss(p[:, None], t[:, None])
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-12)
    ga, = torch.autograd.grad(actual, p, retain_graph=True)
    ge, = torch.autograd.grad(expected, p)
    torch.testing.assert_close(ga, ge, atol=1e-12, rtol=1e-11)


@pytest.mark.parametrize("normalization", ["none", "zscore"])
def test_full_uniform_proximity_equals_plain_value_and_gradient(normalization):
    p, t = sample(True)
    plain = criterion(cross_score_normalization=normalization)._cross_loss(p[:, None], t[:, None])
    prox = criterion("spearman_proximity", cross_mask=["uniform", 10.],
                     cross_score_normalization=normalization)._cross_loss(p[:, None], t[:, None])
    assert torch.equal(plain, prox)
    ga, = torch.autograd.grad(plain, p, retain_graph=True)
    gb, = torch.autograd.grad(prox, p)
    assert torch.equal(ga, gb)


@pytest.mark.parametrize("name", ["spearman", "spearman_proximity"])
def test_permutation_invariance_and_gradient(name):
    p, t = sample(True)
    loss = criterion(name)
    value = loss._cross_loss(p[:, None], t[:, None])
    order = torch.tensor([3, 1, 0, 2])
    permuted = loss._cross_loss(p[:, None, order], t[:, None, order])
    torch.testing.assert_close(value, permuted)
    ga, = torch.autograd.grad(value, p, retain_graph=True)
    gb, = torch.autograd.grad(permuted, p)
    torch.testing.assert_close(ga, gb)


@pytest.mark.parametrize("name", ["spearman", "spearman_proximity"])
def test_gradient_finite_differences(name):
    p, t = sample(True)
    loss = criterion(name, cross_score_normalization="zscore")
    assert torch.autograd.gradcheck(lambda x: loss._cross_loss(x[:, None], t[:, None]), (p,), eps=1e-6)


@pytest.mark.parametrize("name", ["spearman", "spearman_proximity"])
@pytest.mark.parametrize("case", ["constant_target", "constant_prediction", "single_asset"])
def test_degenerate_rows_are_finite_and_have_defined_gradient(name, case):
    p, t = sample()
    if case == "constant_target": t = torch.zeros_like(t)
    if case == "constant_prediction": p = torch.zeros_like(p)
    if case == "single_asset": p, t = p[:, :1], t[:, :1]
    p.requires_grad_()
    value = criterion(name)._cross_loss(p[:, None], t[:, None])
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(p.grad).all()
    if case == "constant_prediction":
        assert value == 1
        assert p.grad.abs().sum() > 0
    else:
        assert value == 0
        assert p.grad.abs().sum() == 0


@pytest.mark.parametrize("mask", [["uniform", 0.], ["uniform", 10., "complementary"],
                                  ["compact", .00001, "natural", 2., 1.],
                                  ["cosine", .00001]])
def test_empty_or_ties_only_mask_contributes_zero(mask):
    p, t = sample(True)
    value = criterion("spearman_proximity", cross_mask=mask)._cross_loss(p[:, None], t[:, None])
    value.backward()
    assert value == 0
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() == 0


@pytest.mark.parametrize("mask", [["student", .025, "natural", 10.],
                                  ["generalized_gaussian", .025, "natural", 4.],
                                  ["intermediate_band", .025, "natural", 2.],
                                  ["cauchy", 1e-300, "complementary"],
                                  ["gaussian", 1e-300, "natural"]])
def test_kernel_families_and_extreme_weights(mask):
    p, t = sample(True)
    value = criterion("spearman_proximity", cross_mask=mask)._cross_loss(p[:, None], t[:, None])
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(p.grad).all()
    assert 0 <= value <= 2


@pytest.mark.parametrize("name", ["spearman", "spearman_proximity"])
@pytest.mark.parametrize("lam", [0., .5, .8, 1.])
def test_hybrid_endpoints_and_aggregation(name, lam):
    p, t = sample(True)
    p = torch.stack([p, p * 1.3], dim=1)
    t = torch.stack([t, t * .7], dim=1)
    loss = criterion(name, trade_window=2, cross_lambda=lam, cross_scale=2.)
    actual = loss(p, t)
    expected = (1-lam) * F.mse_loss(p, t) + lam * 2 * loss._cross_loss(p.sum(1, keepdim=True), t.sum(1, keepdim=True))
    torch.testing.assert_close(actual, expected)
    if lam == 1:
        huber = criterion(name, trade_window=2, cross_lambda=lam, cross_scale=2., temporal_loss="huber")
        alternative = huber(p, t)
        assert torch.equal(actual, alternative)
        ga, = torch.autograd.grad(actual, p, retain_graph=True)
        gb, = torch.autograd.grad(alternative, p)
        assert torch.equal(ga, gb)


def test_pure_ranking_ignores_overflowing_temporal_mse():
    p = torch.tensor([[[1e20, -1e20, 2e20, -2e20]]], requires_grad=True)
    t = torch.tensor([[[.01, -.02, .03, .04]]])
    assert torch.isinf(F.mse_loss(p, t))
    value = criterion(cross_lambda=1)(p, t)
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(p.grad).all()


def test_raw_target_gaps_and_rank_labels_have_distinct_roles():
    p, t = sample()
    # Positive rescaling preserves exact target ranks and the plain objective.
    plain = criterion(cross_score_normalization="zscore")
    torch.testing.assert_close(plain._cross_loss(p[:, None], t[:, None]),
                               plain._cross_loss(p[:, None], 10*t[:, None]))
    prox = criterion("spearman_proximity", cross_score_normalization="zscore")
    assert not torch.isclose(prox._cross_loss(p[:, None], t[:, None]),
                             prox._cross_loss(p[:, None], 10*t[:, None]))


@pytest.mark.parametrize("tau", [0., -1., float("nan"), float("inf")])
def test_temperature_validation(tau):
    with pytest.raises(ValueError, match="loss_spearman_tau"):
        criterion(spearman_tau=tau)


def test_builder_passes_temperature_and_preserves_name():
    config = SimpleNamespace(loss="composite_trading", loss_trade_window=1,
        loss_cross="spearman_proximity", loss_cross_mask=["cauchy", .025],
        loss_spearman_tau=.25, loss_cross_lambda=1., norm=False)
    loss = build_loss(config)
    assert loss.cross_loss_name == "spearman_proximity"
    assert loss.spearman_tau == .25
