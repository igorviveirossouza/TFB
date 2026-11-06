"""Kernel-family contracts and independent numerical/gradient oracles."""
import math
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ts_benchmark.baselines.proximity_losses import validate_cross_mask
from ts_benchmark.baselines.composite_trading_loss_v3 import CompositeTradingLossV3, proximity_weights

SPECS = [
    ('cauchy', ()), ('student', (10.,)),
    ('generalized_gaussian', (1.,)), ('generalized_gaussian', (2.,)),
    ('generalized_gaussian', (4.,)), ('compact', (2., 0.)),
    ('compact', (1., 1.)), ('compact', (2., 1.)),
    ('compact', (2., 2.)), ('compact', (2., 3.)),
    ('cosine', ()), ('intermediate_band', (2.,)),
]


def oracle(u, family, shape):
    if family == 'cauchy': return 1 / (1 + u*u)
    if family == 'student': return (1 + u*u/shape[0]) ** (-(shape[0]+1)/2)
    if family == 'generalized_gaussian': return math.exp(-u**shape[0]/shape[0])
    if family == 'compact':
        p, q = shape
        return (1-u**p)**q if u <= 1 else 0.
    if family == 'cosine': return math.cos(math.pi*u/2) if u < 1 else 0.
    if family == 'intermediate_band': return (u*math.exp(1-u))**shape[0]
    raise AssertionError(family)


@pytest.mark.parametrize('family,shape', SPECS)
@pytest.mark.parametrize('mode', ['natural', 'complementary'])
def test_weights_match_direct_formula(family, shape, mode):
    gap = torch.tensor([[0., .01, .05, .125, .25, .5, 1.]], dtype=torch.float64)
    mask = validate_cross_mask([family, .25, mode, *shape])
    actual = proximity_weights(gap, mask, torch.float64)
    expected = torch.tensor([[oracle(float(x)/.25, family, shape) for x in gap[0]]], dtype=torch.float64)
    if mode == 'complementary':
        expected = 1-expected
        expected[gap == 0] = 0
    expected /= expected.max(dim=1, keepdim=True).values.clamp_min(1e-300)
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-12)


@pytest.mark.parametrize('alias,family,shape', [
    ('cauchy', 'student', (1.,)), ('gaussian', 'generalized_gaussian', (2.,)),
    ('laplace', 'generalized_gaussian', (1.,)), ('uniform', 'compact', (2.,0.)),
    ('triangular', 'compact', (1.,1.)), ('epanechnikov', 'compact', (2.,1.)),
    ('biweight', 'compact', (2.,2.)), ('triweight', 'compact', (2.,3.)),
])
@pytest.mark.parametrize('mode', ['natural', 'complementary'])
def test_alias_equivalence(alias, family, shape, mode):
    gap = torch.tensor([[0.,.1,.25,.4], [.01,.04,.2,.5]], dtype=torch.float64)
    a = proximity_weights(gap, validate_cross_mask([alias,.25,mode]),torch.float64)
    b = proximity_weights(gap, validate_cross_mask([family,.25,mode,*shape]),torch.float64)
    assert torch.equal(a,b)


def test_legacy_configuration_identity_and_new_explicit_shapes():
    for name in ['cauchy','gaussian','uniform']:
        assert validate_cross_mask([name,.25]) == (name,.25)
        assert validate_cross_mask([name,.25,'natural']) == (name,.25)
        assert validate_cross_mask([name,.25,'complementary']) == (name,.25,'complementary')
    assert validate_cross_mask(['student',.25,'natural',10]) == ('student',.25,'natural',10.)
    assert validate_cross_mask(['compact',0,'natural',2,0]) == ('compact',0.,'natural',2.,0.)


@pytest.mark.parametrize('value', [
    ['student',.25], ['student',.25,10], ['student',.25,'natural'],
    ['student',.25,'natural',0], ['student',.25,'natural',True],
    ['student',.25,'natural',float('inf')], ['student',float('nan'),'natural',10],
    ['compact',.25,'natural',2,-1], ['compact',.25,'natural',0,1],
    ['compact',0,'natural',2,1], ['cosine',0], ['laplace',-1],
    ['gaussian',.25,'natural',2], ['intermediate_band',.25,'natural',-2],
])
def test_invalid_shape_contract(value):
    with pytest.raises(ValueError): validate_cross_mask(value)


@pytest.mark.parametrize('family,shape',SPECS)
@pytest.mark.parametrize('mode',['natural','complementary'])
@pytest.mark.parametrize('bandwidth',[1e-300,.25,1e300])
def test_extreme_scales_and_empty_rows_have_finite_weighted_loss_gradient(family,shape,mode,bandwidth):
    loss = CompositeTradingLossV3(trade_window=1,data_kind='log_return',
        cross_loss='ranknet_proximity',cross_lambda=1.,cross_score_normalization='none',
        inverse_norm=False,cross_mask=[family,bandwidth,mode,*shape])
    pred = torch.tensor([[.4,-.2,.1,.3],[.05,.12,-.01,.3]], dtype=torch.float64,requires_grad=True)
    target = torch.tensor([[0.,0.,0.,0.],[0.,.01,.1,.5]],dtype=torch.float64)
    value=loss._proximity_cross_loss(pred,target)
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(pred.grad).all()
    weights=proximity_weights(torch.tensor([[0.,0.,0.],[.01,.1,1.]],dtype=torch.float64),loss.cross_mask,torch.float64)
    assert torch.isfinite(weights).all() and (weights>=0).all() and (weights<=1).all()
    if family not in {'compact','cosine'} and mode=='natural':
        assert weights[1].max()==1, 'No false empty row from absolute underflow'
    if mode=='complementary': assert torch.equal(weights[0],torch.zeros_like(weights[0]))


def test_band_peak_and_quadratic_at_origin():
    gap=torch.tensor([[0.,.001,.002,1.,2.,20.]],dtype=torch.float64)
    w=proximity_weights(gap,validate_cross_mask(['intermediate_band',1.,'natural',2]),torch.float64)
    assert w[0,0]==0 and w[0,3]==1
    assert 3.98 < w[0,2]/w[0,1] < 4.0
    assert w[0,5]<w[0,4]<w[0,3]


def test_generalized_small_p_and_student_large_nu():
    gap=torch.tensor([[.1, .2, .5, 1.]],dtype=torch.float64)
    w=proximity_weights(gap,validate_cross_mask(['generalized_gaussian',1.,'natural',1e-10]),torch.float64)
    torch.testing.assert_close(w,.1/gap,rtol=1e-8,atol=1e-10)
    student=proximity_weights(gap,validate_cross_mask(['student',1.,'natural',1e10]),torch.float64)
    gaussian=proximity_weights(gap,validate_cross_mask(['gaussian',1.]),torch.float64)
    torch.testing.assert_close(student,gaussian,rtol=1e-8,atol=1e-10)


def test_cosine_and_compact_boundary_exact_zero():
    gap=torch.tensor([[0.,1.,2.]],dtype=torch.float64)
    for mask in [['cosine',1.],['compact',1.,'natural',2,1]]:
        w=proximity_weights(gap,validate_cross_mask(mask),torch.float64)
        assert torch.equal(w,torch.tensor([[1.,0.,0.]],dtype=torch.float64))
