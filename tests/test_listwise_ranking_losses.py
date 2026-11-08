"""Independent math, gradients and edge-case checks for global Proximity losses."""
from pathlib import Path
import math
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ts_benchmark.baselines.listwise_ranking_losses import (
    unscaled_proximity_weights, financial_pair_weights,
    listnet_proximity_loss, listmle_loss, approxndcg_loss,
)
from ts_benchmark.baselines.composite_trading_loss_v3 import CompositeTradingLossV3, build_loss


def criterion(name, **kwargs):
    config = dict(trade_window=1, data_kind="log_return", score_kind="log_return",
                  cross_loss=name, cross_lambda=1., inverse_norm=False,
                  cross_score_normalization="none", cross_mask=["cauchy", .025])
    config.update(kwargs)
    return CompositeTradingLossV3(**config)


def sample(requires_grad=False):
    p = torch.tensor([[.4, -.2, .15, .3], [-.1, .45, .8, .23]], dtype=torch.float64,
                     requires_grad=requires_grad)
    y = torch.tensor([[.03, .01, .01, -.02], [.1, .04, .06, -.01]], dtype=torch.float64)
    return p, y


def listnet_oracle(p, y, w, tau):
    batch = []
    for pred, target, weight in zip(p, y, w):
        anchors = []
        for i in range(len(pred)):
            if not any(weight[i,j] > 0 for j in range(len(pred)) if i != j):
                continue
            mass = torch.stack([pred.new_tensor(1.) if i == j else weight[i,j] for j in range(len(pred))])
            true_prob = mass * (target / tau).exp(); true_prob = true_prob / true_prob.sum()
            pred_prob = mass * (pred / tau).exp(); pred_prob = pred_prob / pred_prob.sum()
            anchors.append(-sum(true_prob[j] * pred_prob[j].log() for j in range(len(pred)) if mass[j] > 0))
        batch.append(torch.stack(anchors).mean() if anchors else pred.sum()*0)
    return torch.stack(batch).mean()


def listmle_oracle(p, y, w, tau):
    batch = []
    for pred, target, weight in zip(p, y, w):
        anchors = []
        for i in range(len(pred)):
            lower = [j for j in range(len(pred)) if target[j] < target[i]]
            if lower:
                anchors.append(torch.log(1 + sum(weight[i,j] * torch.exp((pred[j]-pred[i])/tau) for j in lower)))
        batch.append(torch.stack(anchors).mean() if anchors else pred.sum()*0)
    return torch.stack(batch).mean()


def ndcg_oracle(p, y, w, tau):
    batch = []
    for pred, target, weight in zip(p, y, w):
        n = len(pred)
        true_ranks = [1 + sum(float(target[j] > target[i]) + .5*float(target[j] == target[i])
                              for j in range(n) if j != i) for i in range(n)]
        gains = [2 ** ((n-r)/(n-1)) - 1 for r in true_ranks]
        ideal = sum(g/math.log2(i+2) for i,g in enumerate(sorted(gains, reverse=True)))
        ranks = [true_ranks[i] + sum(weight[i,j] * (torch.sigmoid((pred[j]-pred[i])/tau)
                      - float(target[j] > target[i]) - .5*float(target[j] == target[i]))
                    for j in range(n) if j != i) for i in range(n)]
        batch.append(1 - sum(g/torch.log2(1+r) for g,r in zip(gains,ranks)) / ideal)
    return torch.stack(batch).mean()


@pytest.mark.parametrize("name,oracle", [("listnet_proximity",listnet_oracle),
                                          ("listmle_proximity",listmle_oracle),
                                          ("approxndcg_proximity",ndcg_oracle)])
def test_raw_cauchy_matches_independent_anchor_formulas_and_gradients(name, oracle):
    p,y = sample(True)
    w = 1 / (1 + ((y[:,:,None] - y[:,None,:]) / .025).square())
    expected = oracle(p,y,w,1.)
    actual = criterion(name)._cross_loss(p[:,None], y[:,None])
    torch.testing.assert_close(actual,expected,atol=1e-13,rtol=1e-12)
    ga, = torch.autograd.grad(actual,p,retain_graph=True)
    ge, = torch.autograd.grad(expected,p)
    torch.testing.assert_close(ga,ge,atol=1e-12,rtol=1e-11)


@pytest.mark.parametrize("name", ["listnet","listmle","approxndcg"])
@pytest.mark.parametrize("normalization", ["none","zscore"])
def test_unit_weights_recover_plain_loss_and_gradient(name, normalization):
    p,y = sample(True)
    plain = criterion(name,cross_score_normalization=normalization)._cross_loss(p[:,None],y[:,None])
    prox = criterion(name+"_proximity",cross_score_normalization=normalization,
                     cross_mask=["uniform",1e9])._cross_loss(p[:,None],y[:,None])
    torch.testing.assert_close(plain,prox,atol=1e-14,rtol=1e-13)
    ga, = torch.autograd.grad(plain,p,retain_graph=True)
    gb, = torch.autograd.grad(prox,p)
    torch.testing.assert_close(ga,gb,atol=1e-13,rtol=1e-12)


def test_listmle_plain_matches_sorted_plackett_luce_probability():
    p = torch.tensor([[.3,-.5,.1,.7]],dtype=torch.float64,requires_grad=True)
    y = torch.tensor([[2.,1.,4.,3.]],dtype=torch.float64)
    ordered = p[0,torch.argsort(y[0],descending=True)]
    expected = sum(torch.logsumexp(ordered[i:],dim=0)-ordered[i] for i in range(3))/3
    torch.testing.assert_close(listmle_loss(p,y),expected)


@pytest.mark.parametrize("name",["listnet_proximity","listmle","listmle_proximity","approxndcg","approxndcg_proximity"])
def test_permutation_invariant_value_and_gradient_including_target_ties(name):
    p,y=sample(True)
    order=torch.tensor([2,0,3,1])
    objective=criterion(name)
    value=objective._cross_loss(p[:,None],y[:,None])
    other=objective._cross_loss(p[:,None,order],y[:,None,order])
    torch.testing.assert_close(value,other)
    ga,=torch.autograd.grad(value,p,retain_graph=True)
    gb,=torch.autograd.grad(other,p)
    torch.testing.assert_close(ga,gb)


@pytest.mark.parametrize("name",["listnet_proximity","listmle","listmle_proximity","approxndcg","approxndcg_proximity"])
def test_gradcheck_and_targets_detached(name):
    p,y=sample(True); y.requires_grad_()
    objective=criterion(name)
    assert torch.autograd.gradcheck(lambda x: objective._cross_loss(x[:,None],y[:,None]),(p,),eps=1e-6,atol=1e-5)
    loss=objective._cross_loss(p[:,None],y[:,None])
    g_pred,g_target=torch.autograd.grad(loss,(p,y),allow_unused=True)
    assert torch.isfinite(g_pred).all() and g_target is None


@pytest.mark.parametrize("name",["listnet_proximity","listmle_proximity","approxndcg_proximity"])
@pytest.mark.parametrize("mask",[["uniform",0.],["gaussian",1e-300],["intermediate_band",1e-300,"natural",2.]])
def test_no_active_pairs_yield_zero_finite_gradient(name, mask):
    p,y=sample(True)
    y=torch.tensor([[.3,.1,.4,-.2],[.1,.4,.6,-.1]],dtype=torch.float64)
    value=criterion(name,cross_mask=mask)._cross_loss(p[:,None],y[:,None])
    assert value.item()==0.
    value.backward()
    assert torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)==0


@pytest.mark.parametrize("name",["listmle","listmle_proximity","approxndcg","approxndcg_proximity"])
def test_all_tied_labels_have_no_order_signal(name):
    p,y=sample(True); y=torch.ones_like(y)
    loss=criterion(name)._cross_loss(p[:,None],y[:,None])
    assert loss.item()==0.
    loss.backward(); assert torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)==0


@pytest.mark.parametrize("name",["listnet_proximity","listmle","listmle_proximity","approxndcg","approxndcg_proximity"])
def test_constant_predictions_and_single_asset_are_finite(name):
    p,y=sample(); p=torch.zeros_like(p,requires_grad=True)
    loss=criterion(name)._cross_loss(p[:,None],y[:,None]); loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(p.grad).all()
    single=p[:,:1]
    value=criterion(name)._cross_loss(single[:,None],y[:,:1,None].transpose(1,2))
    assert value.item()==0


@pytest.mark.parametrize("name",["listnet_proximity","listmle","listmle_proximity","approxndcg","approxndcg_proximity"])
def test_large_finite_scores_keep_stable_log_sums(name):
    p,y=sample(True)
    p=(p*1e4).detach().requires_grad_()
    value=criterion(name)._cross_loss(p[:,None],y[:,None]); value.backward()
    assert torch.isfinite(value) and torch.isfinite(p.grad).all()


MASKS=[["cauchy",.025],["gaussian",.025],["uniform",.025],["student",.025,"natural",10.],
       ["generalized_gaussian",.025,"natural",1.],["generalized_gaussian",.025,"natural",4.],
       ["compact",.025,"natural",1.,1.],["compact",.025,"natural",2.,2.],
       ["cosine",.025],["intermediate_band",.025,"natural",2.]]


@pytest.mark.parametrize("mask",MASKS)
@pytest.mark.parametrize("mode",["natural","complementary"])
def test_all_kernel_weights_are_unscaled_bounded_finite_and_detached(mask, mode):
    mask=[mask[0],mask[1],mode,*mask[3:]]
    gaps=torch.tensor([0.,1e-300,.005,.025,.05,1e300,float('inf')],dtype=torch.float64,requires_grad=True)
    weights=unscaled_proximity_weights(gaps,mask)
    assert torch.isfinite(weights).all() and (weights>=0).all() and (weights<=1).all()
    assert weights.requires_grad is False
    if mode=="complementary": assert weights[0]==0
    # No per-section normalization: the Cauchy weight at gap=A is 1/2.
    if mask[0]=="cauchy": assert weights[3].item()==pytest.approx(.5)


@pytest.mark.parametrize("alias,family",[("cauchy",["student",.025,"natural",1.]),
                                         ("gaussian",["generalized_gaussian",.025,"natural",2.]),
                                         ("laplace",["generalized_gaussian",.025,"natural",1.]),
                                         ("uniform",["compact",.025,"natural",2.,0.]),
                                         ("triangular",["compact",.025,"natural",1.,1.]),
                                         ("epanechnikov",["compact",.025,"natural",2.,1.]),
                                         ("biweight",["compact",.025,"natural",2.,2.]),
                                         ("triweight",["compact",.025,"natural",2.,3.])])
def test_unscaled_kernel_aliases(alias,family):
    gap=torch.linspace(0,.1,51,dtype=torch.float64)
    torch.testing.assert_close(unscaled_proximity_weights(gap,[alias,.025]),
                               unscaled_proximity_weights(gap,family),rtol=0,atol=0)


def test_uniform_zero_bandwidth_has_legacy_boundary_semantics():
    gap=torch.tensor([0.,1e-100,1.],dtype=torch.float64)
    assert torch.equal(unscaled_proximity_weights(gap,["uniform",0.]),torch.tensor([1.,0.,0.],dtype=torch.float64))
    assert torch.equal(unscaled_proximity_weights(gap,["uniform",0.,"complementary"]),torch.tensor([0.,1.,1.],dtype=torch.float64))


def test_approxndcg_bound_and_true_order_with_arbitrary_symmetric_masks():
    generator=torch.Generator().manual_seed(80)
    p=torch.randn(300,7,generator=generator,dtype=torch.float64)*15
    y=torch.randn(300,7,generator=generator,dtype=torch.float64)
    w=torch.rand(300,7,7,generator=generator,dtype=torch.float64)
    w=(w+w.transpose(1,2))/2
    # Check independently per row rather than allowing averages to hide negatives.
    for i in range(len(p)):
        value=approxndcg_loss(p[i:i+1],y[i:i+1],pair_weights=w[i:i+1],tau=.1)
        assert -1e-13 <= value <= 1+1e-13
        perfect=approxndcg_loss(y[i:i+1]*1e4,y[i:i+1],pair_weights=w[i:i+1],tau=.01)
        assert abs(perfect)<1e-10


def test_approxndcg_signed_returns_and_label_scale_invariance_without_kernel():
    p,y=sample()
    first=approxndcg_loss(p,y)
    torch.testing.assert_close(first,approxndcg_loss(p,100*y-1000))


@pytest.mark.parametrize("key",["listmle_tau","approxndcg_tau"])
@pytest.mark.parametrize("bad",[0,-1,float('inf'),float('nan')])
def test_invalid_temperature_is_rejected(key,bad):
    with pytest.raises(ValueError,match="finito e positivo"):
        criterion("listmle",**{key:bad})


def test_builder_propagates_temperatures_and_new_losses():
    for name in ("listmle","listmle_proximity","approxndcg","approxndcg_proximity","listnet_proximity"):
        config=SimpleNamespace(loss="composite_trading",loss_cross=name,loss_trade_window=1,
                               loss_cross_mask=["cauchy",.025],loss_listmle_tau=.4,
                               loss_approxndcg_tau=.3,loss_listnet_tau=.2)
        result=build_loss(config)
        assert result.cross_loss_name==name and result.listmle_tau==.4 and result.approxndcg_tau==.3


def test_approxndcg_rejects_non_symmetric_or_out_of_range_weights():
    p,y=sample(); w=torch.ones(2,4,4,dtype=p.dtype)
    w[:,0,1]=.3
    with pytest.raises(ValueError,match="simetricos"): approxndcg_loss(p,y,pair_weights=w)
    w[:]=1.1
    with pytest.raises(ValueError,match=r"\[0,1\]"): approxndcg_loss(p,y,pair_weights=w)


@pytest.mark.parametrize("mask,expected",[
    (["compact",1.,"complementary",2.,1.],1e-20),
    (["compact",1.,"complementary",2.,3.],3e-20),
    (["cosine",1.,"complementary"],math.pi**2/8*1e-20),
])
def test_tiny_complementary_weights_survive_float32(mask,expected):
    value=unscaled_proximity_weights(torch.tensor([1e-10],dtype=torch.float64),mask,torch.float32)
    assert value.item()>0
    assert value.item()==pytest.approx(expected,rel=1e-6,abs=0)


def test_approxndcg_mask_selecting_only_ties_has_no_ordinal_signal():
    p=torch.tensor([[.6,-.2,.1]],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([[.03,.03,-.05]],dtype=torch.float64)
    loss=criterion('approxndcg_proximity',cross_mask=['uniform',0.])._cross_loss(p[:,None],y[:,None])
    assert loss.item()==0
    loss.backward()
    assert torch.count_nonzero(p.grad)==0
