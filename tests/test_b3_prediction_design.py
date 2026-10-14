"""16-pair loss coverage and one fitted VAR producing every forecast horizon."""
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO))
import b3_experiment as experiment
import b3_experiment_io as experiment_io

HORIZONS = [1, 5, 10, 15, 20, 25]
PAIRS = [(h, k) for h in HORIZONS for k in HORIZONS if k <= h and h % k == 0]


@pytest.fixture(scope='module')
def config():
    result = subprocess.run(['bash', str(REPO/'scripts/run_composite_trading_experiment.sh'), '--print-config'],
                            check=True, capture_output=True, text=True,
                            env=dict(os.environ, TFB_ROOT=str(REPO), PYTHON_BIN=sys.executable))
    return json.loads(result.stdout)


@pytest.mark.parametrize('horizon,k', PAIRS)
@pytest.mark.parametrize('cross_loss', ['ranknet', 'hinge', 'listnet', 'pairwise_mse'])
def test_every_requested_pair_runs_forward_and_backward(config, horizon, k, cross_loss):
    torch = pytest.importorskip('torch')
    from ts_benchmark.baselines.composite_trading_loss_v3 import build_loss
    task = dict(modelo='simpleMLP', experimento='composto', lookback=32,
                pred_len=horizon, k=k, loss=cross_loss, cross_lambda=.5, cross_delta=0)
    _, hyper, _, _ = experiment.model_spec(config, task)
    criterion = build_loss(SimpleNamespace(**hyper), normalizer_mean=[0]*3, normalizer_scale=[1]*3)
    generator = torch.Generator().manual_seed(2026)
    prediction = (torch.randn(2, horizon, 3, generator=generator)*.01).requires_grad_()
    target = torch.randn(2, horizon, 3, generator=generator)*.01
    loss = criterion(prediction, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
    components = criterion.last_components
    assert components['H'] == horizon
    assert components['K'] == k
    assert components['n_blocks'] == horizon//k
    assert components['cross_steps'] == horizon//k*k
    assert components['cross_tail_steps'] == 0
    assert components['cross_steps'] == horizon


@pytest.mark.parametrize('horizon,k', PAIRS)
@pytest.mark.parametrize('cross_lambda', [0.1, 0.5, 0.8])
def test_listnet_values_and_gradients_are_identical_for_all_deltas(config, horizon, k, cross_lambda):
    torch = pytest.importorskip('torch')
    from ts_benchmark.baselines.composite_trading_loss_v3 import build_loss
    generator = torch.Generator().manual_seed(2026)
    values = torch.randn(2, horizon, 3, generator=generator) * .01
    target = torch.randn(2, horizon, 3, generator=generator) * .01
    results = []
    for delta in (0, .025, .05, .1):
        task = dict(modelo='simpleMLP', experimento='composto', lookback=32,
                    pred_len=horizon, k=k, loss='listnet',
                    cross_lambda=cross_lambda, cross_delta=delta)
        _, hyper, _, _ = experiment.model_spec(config, task)
        criterion = build_loss(SimpleNamespace(**hyper), normalizer_mean=[0]*3, normalizer_scale=[1]*3)
        prediction = values.clone().requires_grad_()
        loss = criterion(prediction, target)
        gradient = torch.autograd.grad(loss, prediction)[0]
        assert torch.isfinite(loss) and torch.isfinite(gradient).all()
        results.append((loss.detach(), gradient))
    for loss, gradient in results[1:]:
        torch.testing.assert_close(loss, results[0][0], rtol=0, atol=0)
        torch.testing.assert_close(gradient, results[0][1], rtol=0, atol=0)


@pytest.mark.parametrize('cross_loss', ['ranknet', 'hinge', 'pairwise_mse'])
def test_pairwise_losses_really_use_delta_in_values_and_gradients(cross_loss):
    torch = pytest.importorskip('torch')
    from ts_benchmark.baselines.composite_trading_loss_v3 import CompositeTradingLossV3
    results = []
    for delta in (0, .025, .05, .1):
        criterion = CompositeTradingLossV3(
            trade_window=1, data_kind='log_return',
            cross_loss=cross_loss, cross_delta=delta, hinge_margin=.1,
            cross_score_normalization='zscore', inverse_norm=False,
        )
        # Raw realized score gaps straddle every threshold; predictions reverse
        # the order so hinge pairs remain active, exercising the actual mask.
        prediction = torch.tensor([[[.03, .08, -.02, -.08]]], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[[0, .02, .06, .18]]], dtype=torch.float64)
        loss = criterion._cross_loss(prediction, target)
        gradient = torch.autograd.grad(loss, prediction)[0]
        results.append((loss.detach(), gradient))
    for i, (loss, gradient) in enumerate(results):
        for other_loss, other_gradient in results[i+1:]:
            assert not torch.equal(loss, other_loss)
            assert not torch.equal(gradient, other_gradient)


@pytest.mark.parametrize('horizon,k', [(15,10), (20,15), (25,10), (25,15), (25,20)])
@pytest.mark.parametrize('cross_loss', ['ranknet', 'hinge', 'listnet', 'pairwise_mse'])
def test_nondivisible_pairs_are_rejected_before_backpropagation(horizon, k, cross_loss):
    torch = pytest.importorskip('torch')
    from ts_benchmark.baselines.composite_trading_loss_v3 import CompositeTradingLossV3
    from ts_benchmark.baselines.composite_trading_loss_v2 import TradingBlockScoreAggregatorV2
    prediction = torch.zeros(1, horizon, 3, requires_grad=True)
    target = torch.zeros_like(prediction)
    criterion = CompositeTradingLossV3(trade_window=k, data_kind='log_return',
                                      cross_loss=cross_loss, cross_lambda=.5,
                                      inverse_norm=False)
    with pytest.raises(ValueError, match='divisor exato'):
        criterion(prediction, target)
    assert prediction.grad is None
    legacy = TradingBlockScoreAggregatorV2(trade_window=k, data_kind='log_return',
                                         score_kind='log_return', inverse_norm=False)
    with pytest.raises(ValueError, match='divisor exato'):
        legacy(prediction, target)


def test_k_does_not_change_temporal_or_var_models(config):
    modified = copy.deepcopy(config)
    modified['trade_windows'] = [7, 13, 100]
    for mode in ['temporal', 'var']:
        before = experiment.make_tasks(config, mode)
        after = experiment.make_tasks(modified, mode)
        assert before == after
        assert all(t['k'] is None for t in before)
        assert [experiment.model_spec(config, t) for t in before] == [experiment.model_spec(modified, t) for t in after]
    assert len(experiment.make_tasks(config, 'var')) == 3


def test_var_fits_once_per_cut_and_exports_all_six_horizons(config, tmp_path, monkeypatch):
    pytest.importorskip('statsmodels')
    pytest.importorskip('pyarrow')
    from ts_benchmark.baselines.self_impl.VAR.VAR import VAR
    cfg = copy.deepcopy(config)
    fake_root = tmp_path/'checkout'
    relatives = ['scripts/b3_experiment.py', 'scripts/b3_experiment_io.py',
                 'scripts/run_benchmark_composite_trading_loss_v3.py',
                 'ts_benchmark/baselines/composite_trading_loss_v2.py',
                 'ts_benchmark/baselines/composite_trading_loss_v3.py',
                 'ts_benchmark/baselines/self_impl/VAR/VAR.py',
                 'ts_benchmark/baselines/deep_forecasting_model_base.py']
    for relative in relatives:
        dest = fake_root/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO/relative, dest)
    rng = np.random.default_rng(2026)
    returns = rng.normal(0, .01, (400, 2))
    prices = 100*np.exp(np.vstack([np.zeros(2), returns.cumsum(axis=0)]))
    source = tmp_path/'prices.csv'
    pd.DataFrame({'date':np.arange(1, 402), 'Z3':prices[:, 0], 'A4':prices[:, 1]}).to_csv(source, index=False)
    monkeypatch.setattr(experiment, 'ROOT', fake_root)
    monkeypatch.setenv('DATASET_PATH', str(source))
    monkeypatch.setattr(experiment.subprocess, 'check_output', lambda *a, **kw:'test-commit\n')
    tasks = experiment.make_tasks(cfg, 'var')
    out = tmp_path/'out'
    metadata = experiment.prepare(tmp_path/'run_var_experiment.sh', cfg, tasks, out)
    panel = experiment_io.load_panel(Path(metadata['dataset']))
    fitted = []
    original_fit = VAR.forecast_fit
    def counted_fit(model, train, **kwargs):
        fitted.append(len(train))
        pd.testing.assert_frame_equal(train, panel.iloc[:len(train)])
        return original_fit(model, train, **kwargs)
    monkeypatch.setattr(VAR, 'forecast_fit', counted_fit)
    for index, task in enumerate(tasks):
        experiment.worker(cfg, tasks, out, index)
        paths = experiment.task_output_paths(cfg, task, out)
        assert len(paths) == 6
        cut = 400*task['treino']//100
        frames = {}
        for h, path in paths.items():
            frame = pd.read_parquet(path/'part.parquet')
            count = 400-cut-int(h)+1
            assert frame['arquivo'].nunique() == count
            assert len(frame) == count*int(h)
            assert (frame['step']-frame['origin_step'] == frame['h']).all()
            assert frame['step'].max() == 401  # All feasible test targets, including the end.
            assert 'k_dir' not in str(path)
            frames[int(h)] = frame[['Z3', 'A4']].to_numpy().reshape(count, int(h), 2)
        longest = frames[25]
        for h in HORIZONS:
            np.testing.assert_allclose(frames[h][:len(longest)], longest[:, :h], rtol=0, atol=0)
        directory = out/experiment.group_path(task)
        completed = list(csv.DictReader((directory/'completed_tasks.csv').open()))
        assert len(completed) == 1  # One completed task per cut, six output files.
        assert json.loads(completed[0]['forecast_horizons']) == HORIZONS
        assert len(json.loads(completed[0]['parquet_sha256'])) == 6
        experiment.worker(cfg, tasks, out, index)  # Resume: no second fit.
        assert len(fitted) == index+1
    assert fitted == [280, 320, 360]
