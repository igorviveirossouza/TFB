"""Mixture supervision, regional membership, gradients and legacy contracts."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ts_benchmark.baselines.ranking_mixture import validate_rank_mix_config
from ts_benchmark.baselines.hybrid_ranking_losses import (
    tail_selection, ranking_mixture_loss, regional_listnet_loss,
    regional_listmle_loss, regional_approxndcg_loss,
)
from ts_benchmark.baselines.listwise_ranking_losses import (
    financial_pair_weights, listnet_proximity_loss, listmle_loss, approxndcg_loss,
)
from ts_benchmark.baselines.composite_trading_loss_v3 import CompositeTradingLossV3, build_loss


def sample():
    p = torch.tensor([[.7, -.4, .2, .9, -.1, .4], [.1, .9, -.3, .5, -.6, -.2]],
                     dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[.03, -.01, .02, .06, -.02, .01], [.01, .04, -.04, .02, -.02, .03]],
                     dtype=torch.float64)
    return p, y


def criterion(name="rankmix", **kwargs):
    config = dict(trade_window=1, data_kind="log_return", score_kind="log_return",
                  cross_loss=name, cross_lambda=1., inverse_norm=False,
                  cross_score_normalization="none", cross_mask=["cauchy", .025])
    config.update(kwargs)
    return CompositeTradingLossV3(**config)


@pytest.mark.parametrize("weights", [[1, 0], [0, 0, 0], [1, 1, 0], [-.1, 1.1, 0],
                                     [float('nan'), .5, .5], [float('inf'), 0, 0],
                                     [True, 0, 0], ['.5', .5, 0], '.5 .5 0'])
def test_reject_invalid_simplex(weights):
    with pytest.raises(ValueError):
        validate_rank_mix_config(weights=weights)


@pytest.mark.parametrize("field,value", [('mode','local'),('regions',['top']),
    ('regions',['top','middle','top']),('fraction',0),('fraction',.51),
    ('fraction',float('nan')),('fraction',True)])
def test_reject_invalid_regional_options(field, value):
    with pytest.raises(ValueError):
        validate_rank_mix_config(**{field:value})


def test_valid_simplex_not_silently_renormalized():
    x = validate_rank_mix_config([.2,.3,.5], 'REGIONAL', ['TOP','BOTTOM','TOP'], .2)
    assert x == dict(weights=[.2,.3,.5],mode='regional',regions=['top','bottom','top'],fraction=.2)
    assert validate_rank_mix_config()['weights'] == [.5,.5,0.]


def test_tail_count_and_boundary_ties():
    y = torch.arange(55., dtype=torch.float64)[None]
    selection = tail_selection(y)
    assert selection.sum() == 18
    assert torch.equal(selection[0,:37],torch.zeros(37,dtype=torch.float64))
    y = torch.tensor([[6.,5.,5.,5.,2.,1.]],dtype=torch.float64,requires_grad=True)
    expected = torch.tensor([[1.,1/3,1/3,1/3,0.,0.]],dtype=torch.float64)
    actual = tail_selection(y)
    torch.testing.assert_close(actual,expected)
    assert not actual.requires_grad


@pytest.mark.parametrize('prox',[False,True])
@pytest.mark.parametrize('weights',[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[.2,.3,.5]])
def test_global_equals_existing_objectives_and_gradients(prox,weights):
    p,y=sample()
    w=financial_pair_weights(y,['cauchy',.025],p.dtype) if prox else None
    net=(listnet_proximity_loss(p,y,pair_weights=w) if prox else
         -(F.softmax(y,-1)*F.log_softmax(p,-1)).sum(-1).mean())
    expected=weights[0]*net+weights[1]*listmle_loss(p,y,pair_weights=w)+weights[2]*approxndcg_loss(p,y,pair_weights=w)
    actual=ranking_mixture_loss(p,y,weights=weights,pair_weights=w)
    torch.testing.assert_close(actual,expected,atol=1e-14,rtol=1e-13)
    ga,=torch.autograd.grad(actual,p,retain_graph=True)
    ge,=torch.autograd.grad(expected,p)
    torch.testing.assert_close(ga,ge,atol=1e-14,rtol=1e-13)


@pytest.mark.parametrize('mode',['global','regional'])
@pytest.mark.parametrize('normalization',['none','zscore'])
def test_full_uniform_mask_matches_plain_mixture(mode,normalization):
    p,y=sample()
    kwargs=dict(rank_mix_mode=mode,rank_mix_weights=[.2,.3,.5],cross_score_normalization=normalization)
    plain=criterion(**kwargs)._cross_loss(p[:,None],y[:,None])
    prox=criterion('rankmix_proximity',cross_mask=['uniform',100.],**kwargs)._cross_loss(p[:,None],y[:,None])
    torch.testing.assert_close(plain,prox,atol=2e-14,rtol=1e-13)
    a,=torch.autograd.grad(plain,p,retain_graph=True)
    b,=torch.autograd.grad(prox,p)
    torch.testing.assert_close(a,b,atol=1e-13,rtol=1e-12)


@pytest.mark.parametrize('function',[regional_listnet_loss,regional_listmle_loss,regional_approxndcg_loss])
@pytest.mark.parametrize('region',['top','bottom'])
def test_preserving_internal_order_but_misplacing_entire_tail_is_penalized(function,region):
    y=torch.tensor([[9.,8.,7.,6.,5.,4.,3.,2.,1.]],dtype=torch.float64)
    good=y.clone()
    bad=good.clone()
    if region=='top':
        bad[:,:3]-=20
    else:
        bad[:,-3:]+=20
    assert function(bad,y,region=region)>function(good,y,region=region)


@pytest.mark.parametrize('prox',[False,True])
def test_regional_listnet_matches_independent_full_universe_formula(prox):
    p,y=sample()
    w=financial_pair_weights(y,['cauchy',.025],p.dtype) if prox else None
    totals=[]
    for b in range(len(p)):
        top=sorted(range(p.shape[1]),key=lambda i:float(y[b,i]),reverse=True)[:2]
        anchor_values=[]
        for a in top:
            weights=[p.new_tensor(1.) if not prox or a==j else w[b,a,j] for j in range(p.shape[1])]
            qden=sum(weights[j]*torch.exp(y[b,j]) for j in top)
            pden=sum(weights[j]*torch.exp(p[b,j]) for j in range(p.shape[1]))
            anchor_values.append(-sum(weights[j]*torch.exp(y[b,j])/qden*
                                     torch.log(weights[j]*torch.exp(p[b,j])/pden) for j in top))
        totals.append(torch.stack(anchor_values).mean())
    expected=torch.stack(totals).mean()
    actual=regional_listnet_loss(p,y,pair_weights=w)
    torch.testing.assert_close(actual,expected,atol=1e-13,rtol=1e-12)


@pytest.mark.parametrize('prox',[False,True])
def test_bottom_listmle_matches_independent_worst_first_partial_likelihood(prox):
    p,y=sample()
    w=financial_pair_weights(y,['cauchy',.025],p.dtype) if prox else None
    totals=[]
    for b in range(len(p)):
        order=sorted(range(p.shape[1]),key=lambda i:float(y[b,i]))
        terms=[]
        for pos in range(2):
            i=order[pos]
            terms.append(torch.log(1+sum((1 if w is None else w[b,i,j])*
                            torch.exp(p[b,i]-p[b,j]) for j in order[pos+1:])))
        totals.append(torch.stack(terms).mean())
    torch.testing.assert_close(regional_listmle_loss(p,y,pair_weights=w),torch.stack(totals).mean(),atol=1e-13,rtol=1e-12)


def test_regional_ndcg_matches_independent_all_asset_ranks():
    p,y=sample()
    vals=[]
    for pred,target in zip(p,y):
        order=sorted(range(len(pred)),key=lambda i:float(target[i]),reverse=True)
        g={i:2**((len(pred)-1-order.index(i))/(len(pred)-1))-1 for i in order[:2]}
        dcg=sum(g[i]/torch.log2(2+sum(torch.sigmoid(pred[j]-pred[i]) for j in range(len(pred)) if j!=i)) for i in order[:2])
        ideal=sum(g[i]/math.log2(pos+2) for pos,i in enumerate(order[:2]))
        vals.append(1-dcg/ideal)
    torch.testing.assert_close(regional_approxndcg_loss(p,y),torch.stack(vals).mean(),atol=1e-13,rtol=1e-12)


@pytest.mark.parametrize('function',[regional_listnet_loss,regional_listmle_loss,regional_approxndcg_loss])
def test_direction_reversal(function):
    p,y=sample()
    torch.testing.assert_close(function(p,y,region='bottom'),function(-p,-y,region='top'))


@pytest.mark.parametrize('mode',['global','regional'])
@pytest.mark.parametrize('kernel',[['cauchy',.025],['uniform',0.],['gaussian',.025],['cauchy',.025,'complementary']])
def test_permutation_invariance_including_boundary_ties(mode,kernel):
    p,y=sample()
    y[0]=torch.tensor([.03,.02,.02,.02,-.01,-.02])
    perm=torch.tensor([5,2,0,4,1,3])
    w=financial_pair_weights(y,kernel,p.dtype)
    left=ranking_mixture_loss(p,y,weights=[.2,.3,.5],mode=mode,pair_weights=w)
    right=ranking_mixture_loss(p[:,perm],y[:,perm],weights=[.2,.3,.5],mode=mode,pair_weights=w[:,perm][:,:,perm])
    torch.testing.assert_close(left,right,atol=1e-13,rtol=1e-12)
    gl,=torch.autograd.grad(left,p,retain_graph=True)
    gr,=torch.autograd.grad(right,p)
    torch.testing.assert_close(gl,gr,atol=1e-13,rtol=1e-12)


@pytest.mark.parametrize('mode',['global','regional'])
def test_empty_pair_weights_zero_loss_and_gradient(mode):
    p,y=sample()
    actual=ranking_mixture_loss(p,y,weights=[.2,.3,.5],mode=mode,pair_weights=p.new_zeros((2,6,6)))
    assert actual.item()==0
    grad,=torch.autograd.grad(actual,p)
    assert torch.count_nonzero(grad)==0


@pytest.mark.parametrize('function',[regional_listnet_loss,regional_listmle_loss,regional_approxndcg_loss])
def test_regional_all_ties_zero(function):
    p,y=sample()
    value=function(p,torch.zeros_like(y))
    assert value.item()==0
    g,=torch.autograd.grad(value,p)
    assert torch.count_nonzero(g)==0


@pytest.mark.parametrize('mode',['global','regional'])
def test_mixture_gradcheck(mode):
    p,y=sample()
    w=financial_pair_weights(y,['cauchy',.025],p.dtype)
    assert torch.autograd.gradcheck(lambda x:ranking_mixture_loss(x,y,weights=[.2,.3,.5],mode=mode,pair_weights=w),(p,))


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('mode',['global','regional'])
def test_extreme_finite_scores(dtype,mode):
    p=torch.tensor([[1000.,-1000.,500.,-500.,0.,1.]],dtype=dtype,requires_grad=True)
    y=torch.tensor([[.03,.02,.01,-.01,-.02,-.03]],dtype=dtype)
    w=financial_pair_weights(y,['uniform',.005],dtype)
    value=ranking_mixture_loss(p,y,weights=[.2,.3,.5],mode=mode,pair_weights=w)
    assert torch.isfinite(value)
    grad,=torch.autograd.grad(value,p)
    assert torch.isfinite(grad).all()


@pytest.mark.parametrize('name',['rankmix','rankmix_proximity'])
@pytest.mark.parametrize('mode',['global','regional'])
@pytest.mark.parametrize('lam',[0.,.5,.8,1.])
def test_outer_lambda_contract(name,mode,lam):
    p,y=sample()
    c=criterion(name,cross_lambda=lam,rank_mix_mode=mode,rank_mix_weights=[.2,.3,.5])
    actual=c(p[:,None],y[:,None])
    cross=c._cross_loss(p[:,None],y[:,None])
    expected=(1-lam)*F.mse_loss(p,y)+lam*cross
    torch.testing.assert_close(actual,expected)
    if lam==1:
        h=criterion(name,temporal_loss='huber',rank_mix_mode=mode,rank_mix_weights=[.2,.3,.5])
        torch.testing.assert_close(actual,h(p[:,None],y[:,None]))


def test_factory_and_legacy_opt_out():
    config=SimpleNamespace(loss='composite_trading',loss_cross='rankmix_proximity',
        loss_trade_window=1,loss_cross_lambda=1.,norm=False,
        loss_cross_mask=['cauchy',.025],loss_rank_mix_weights=[0,.5,.5],
        loss_rank_mix_mode='regional',loss_rank_mix_regions=['top','bottom','top'],
        loss_rank_mix_fraction=.2)
    c=build_loss(config)
    assert c.rank_mix_config==dict(weights=[0,.5,.5],mode='regional',regions=['top','bottom','top'],fraction=.2)
    p,y=sample()
    plain=criterion('listnet')._cross_loss(p[:,None],y[:,None])
    legacy=criterion('listnet',rank_mix_weights=[1,2,3],rank_mix_mode='invalid')._cross_loss(p[:,None],y[:,None])
    assert torch.equal(plain,legacy)


def test_regional_labels_detached():
    p,y=sample()
    y.requires_grad_(True)
    value=ranking_mixture_loss(p,y,mode='regional',weights=[.2,.3,.5])
    gp,gy=torch.autograd.grad(value,(p,y),allow_unused=True)
    assert gy is None and torch.isfinite(gp).all()


@pytest.mark.parametrize('mode',['global','regional'])
def test_half_precision_promotion_precedes_zero_accumulator(mode):
    # A finite half-precision vector can overflow during a preliminary sum.
    p=torch.full((1,55),10000.,dtype=torch.float16,requires_grad=True)
    y=torch.arange(55,dtype=torch.float32)[None]/100
    value=ranking_mixture_loss(p,y,weights=[.2,.3,.5],mode=mode)
    assert value.dtype==torch.float32 and torch.isfinite(value)
    gradient,=torch.autograd.grad(value,p)
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize('kernel',[['cauchy',.025],['cauchy',.025,'complementary'],
                                  ['gaussian',.025],['gaussian',.025,'complementary']])
def test_financial_weights_exact_symmetry_in_float64_tail_lanes(kernel):
    # N=9 exposes non-full vectorization tails. Identical ij/ji gap values may
    # round differently through log/exp without canonical pair mirroring.
    generator=torch.Generator().manual_seed(984)
    for _ in range(12):
        p=torch.randn((6,9),dtype=torch.float64,generator=generator,requires_grad=True)
        y=torch.randn((6,9),dtype=torch.float64,generator=generator)*.02
        y[0,1:4]=.01
        y[2,:]=.005
        weights=financial_pair_weights(y,kernel,p.dtype)
        assert torch.equal(weights,weights.transpose(1,2))
        assert torch.count_nonzero(weights.diagonal(dim1=1,dim2=2))==0
        value=ranking_mixture_loss(p,y,weights=[.2,.3,.5],mode='regional',pair_weights=weights)
        grad,=torch.autograd.grad(value,p)
        assert torch.isfinite(value) and torch.isfinite(grad).all()


def test_explicit_asymmetric_weights_still_rejected_by_ndcg():
    p,y=sample()
    weights=p.new_ones((2,6,6))
    weights[:,0,1]=.5
    with pytest.raises(ValueError,match='simetricos'):
        regional_approxndcg_loss(p,y,pair_weights=weights)
