"""CPU integration, objective routing, scientific contracts and Slurm mocks."""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import ns_capacity_heads as runner
from ts_benchmark.baselines.ns_task_heads import NSTaskAdapter, SeparateTaskNetwork
from ts_benchmark.baselines.time_series_library.models.Nonstationary_Transformer import Nonstationary_Transformer
from ts_benchmark.baselines.time_series_library.adapters_for_transformers import TransformerAdapter


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(('NS_', 'RANK_', 'MOMENTUM_', 'SLURM_')) or key in {
                'MODEL','LOOKBACKS_VALUES','HORIZON_VALUES','K_LOSS_VALUES','LAMBDAS_VALUES',
                'TEMPORAL_LOSSES_VALUES','TRAIN_VALUES','SEEDS_VALUES','CROSS_LOSSES_VALUES','TASK_HEAD_MODES'}:
            monkeypatch.delenv(key)
    torch.set_num_threads(1)


def data():
    return pd.DataFrame(np.random.default_rng(491).normal(0, .02, (80, 5)),
                        columns=list('ABCDE'), index=pd.date_range('2000-01-01', periods=80, name='date'))


def adapter(study='heads', mode='separate', lam=.5, loss='rankmix_proximity', **kw):
    config = dict(seq_len=8,horizon=2,norm=True,batch_size=8,num_epochs=1,patience=1,
                  lr=.001,lradj='constant',d_model=8,d_ff=16,e_layers=1,d_layers=1,n_heads=2,
                  p_hidden_dims=[8,8],p_hidden_layers=2,dropout=0.,loss_temporal='mse',
                  loss_cross=loss,loss_cross_lambda=lam,loss_trade_window=1,loss_data_kind='log_return',
                  loss_score_kind='simple_return',loss_cross_mask=['cauchy',.025,'natural'],
                  validation_mode='ordered_sample_mean')
    config.update(kw)
    return NSTaskAdapter(study=study,head_mode=mode,rank_windows=[1,2],**config)


def test_default_counts_and_grid_overrides(monkeypatch):
    assert len(runner.tasks(runner.configuration('capacity'))) == 60
    assert len(runner.tasks(runner.configuration('heads'))) == 54
    monkeypatch.setenv('TEMPORAL_LOSSES_VALUES','mse huber')
    assert len(runner.tasks(runner.configuration('capacity'))) == 96
    assert len(runner.tasks(runner.configuration('heads'))) == 84
    monkeypatch.setenv('MODEL','DUET')
    with pytest.raises(ValueError): runner.configuration('heads')


def test_validation_and_no_duplicate_controls(monkeypatch):
    monkeypatch.setenv('RANK_MIX_MODES','global regional')
    cfg=runner.configuration('heads'); tasks=runner.tasks(cfg)
    assert len([t for t in tasks if t['lambda_value']==0])==6
    monkeypatch.setenv('NS_CAPACITY_SPECS','{"bad":{"d_model":13}}')
    with pytest.raises(ValueError): runner.configuration('heads')


@pytest.mark.parametrize('mode', ['shared','separate'])
@pytest.mark.parametrize('loss', ['rankmix','rankmix_proximity','ranknet','ranknet_proximity','listnet','listmle','approxndcg','spearman'])
def test_native_fit_and_gradient_routing(mode,loss):
    torch.manual_seed(2026)
    m=adapter(mode=mode,lam=1,loss=loss)
    m.forecast_fit(data(),train_ratio_in_tv=.75)
    pred=m.forecast(2,data())
    assert pred.shape==(2,5) and np.isfinite(pred).all()
    assert m.check_point and all(torch.isfinite(v).all() for v in m.check_point.values())
    if mode=='separate':
        assert m.export_rank_scores.shape==(2,5) and np.isfinite(m.export_rank_scores).all()
        assert m.model.decoder.projection.weight.grad is None  # Pure rank never trains temporal output.
        assert m.model.ranking_head[0].weight.grad.abs().sum()>0
        assert m.model.encoder.attn_layers[0].attention.query_projection.weight.grad.abs().sum()>0
    else:
        assert m.export_rank_scores is None


def test_temporal_endpoint_is_identical_between_studies():
    results=[]
    for study in ['capacity','heads']:
        torch.manual_seed(17);m=adapter(study=study,mode='shared',lam=0)
        m.forecast_fit(data(),train_ratio_in_tv=.75)
        results.append(m.forecast(2,data()))
    np.testing.assert_array_equal(*results)
    torch.manual_seed(17)
    options=vars(adapter(mode='shared',lam=0).config).copy()
    options.update(loss='mse',parallel_strategy=None)
    native=TransformerAdapter('Nonstationary_Transformer',Nonstationary_Transformer,**options)
    native.forecast_fit(data(),train_ratio_in_tv=.75)
    np.testing.assert_array_equal(results[0],native.forecast(2,data()))


def test_native_initialization_is_preserved():
    options=adapter().config
    options.enc_in=options.dec_in=options.c_out=5;options.label_len=4;options.pred_len=2
    torch.manual_seed(91);base=Nonstationary_Transformer(options);rng=torch.get_rng_state()
    torch.manual_seed(91);dual=SeparateTaskNetwork(options)
    assert torch.equal(rng,torch.get_rng_state())
    for name,value in base.state_dict().items(): torch.testing.assert_close(value,dual.state_dict()[name],rtol=0,atol=0)


def test_mixed_head_gradients_and_no_future_input():
    m=adapter();m.forecast_fit(data(),train_ratio_in_tv=.75)
    assert m.model.decoder.projection.weight.grad.abs().sum()>0
    assert m.model.ranking_head[0].weight.grad.abs().sum()>0
    # Decoder only consumes the observed label segment. Future placeholders
    # are zeroed by the native adapter, regardless of supplied target values.
    m.model.eval();x=torch.randn(2,8,5);y=torch.randn(2,6,5)
    marks=torch.zeros(2,8,3);ymarks=torch.zeros(2,6,3)
    with torch.no_grad():
        a=m._process(x,y,marks,ymarks)['output'].clone();sa=m.latest_rank_scores.clone()
        y[:,-2:]=1000
        b=m._process(x,y,marks,ymarks)['output'];sb=m.latest_rank_scores
    torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(sa,sb,rtol=0,atol=0)


def create_fixture(root, monkeypatch, study='heads'):
    source=root/'source.csv'; root.mkdir(parents=True,exist_ok=True)
    panel=data().copy();panel.index=np.arange(1,len(panel)+1);panel.index.name='date'
    panel.reset_index().melt(id_vars='date',var_name='cols',value_name='data').to_csv(source,index=False)
    small=dict(d_model=8,d_ff=16,e_layers=1,d_layers=1,n_heads=2,p_hidden_dims=[8,8],dropout=0.)
    for k,v in dict(NS_CAPACITY_SPECS=json.dumps({'base':small}),LOOKBACKS_VALUES='8',HORIZON_VALUES='1 2',
                    NS_NUM_EPOCHS='1',NS_PATIENCE='1',NS_BATCH_SIZE='8',NS_LRADJ='constant',
                    NS_RETURNS_FILE=str(source),RANK_WINDOWS_VALUES='1 2').items():monkeypatch.setenv(k,v)
    cfg=runner.configuration(study)
    calls=[]
    def fake(command,**kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command,0,'MaxArraySize = 1001\n' if command[0]=='scontrol' else '12345\n','')
    with patch.object(runner.subprocess,'run',side_effect=fake):
        runner.submit(cfg,root/'predictions');runner.submit(cfg,root/'predictions')
    assert len([c for c in calls if c[0]=='sbatch'])==1
    plan=root/'predictions/experiment_config.json'
    return plan,cfg


def test_frozen_worker_export_and_resume(tmp_path,monkeypatch):
    plan,cfg=create_fixture(tmp_path,monkeypatch)
    records=json.loads(plan.read_text())['tasks']
    selected=[0,1,2,6]  # Temporal, shared mixed, separate mixed and separate pure ranking.
    for ix in selected:runner.run_worker(plan,ix)
    for ix in selected:
        folder=plan.parent/('fit='+records[ix]['fit_id']);record=json.loads((folder/'result.json').read_text())
        assert record['parameter_count']>0
        assert all(runner.sha(folder/p)==v for p,v in record['files_sha256'].items())
        if records[ix]['head_mode']=='separate':
            scores=pd.read_parquet(folder/'ranking_scores.parquet')
            assert set(scores.k_eval)==set(records[ix]['rank_windows'])
            if records[ix]['lambda_value']==1:
                assert not record['return_forecast_valid']
                assert not (folder/'part.parquet').exists()
        runner.run_worker(plan,ix)
    frozen=json.loads(plan.read_text())
    (plan.parent/frozen['source_file']).write_text('changed')
    with pytest.raises(ValueError,match='alterada'):runner.run_worker(plan,0)
