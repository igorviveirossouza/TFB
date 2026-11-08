"""Complete global-ranking grid, endpoint reuse and terminal configuration."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / 'scripts/run_ns_global_ranking_isolado.sh'


def environment(**overrides):
    env = dict(os.environ)
    names = {'MODEL', 'TRAIN_VALUES', 'LOOKBACKS_VALUES', 'HORIZON_VALUES',
             'K_LOSS_VALUES', 'POSITIVE_LAMBDAS', 'LAMBDAS_VALUES',
             'TEMPORAL_LOSSES_VALUES', 'SEEDS_VALUES', 'CROSS_LOSSES_VALUES',
             'K_EVAL_VALUES', 'DELTAS_VALUES', 'KERNEL_SPECS', 'EXPERIMENT_TAG',
             'TFB_ROOT', 'PRED_ROOT', 'BACKTEST_ROOT'}
    for name in list(env):
        if name.startswith(('NS_', 'MOMENTUM_')) or name in names:
            env.pop(name)
    env.update(overrides)
    return env


def run(*args, script=WRAPPER, **env):
    return subprocess.run(['bash', str(script), *args], env=environment(**env),
                          text=True, capture_output=True)


def fixture(tmp_path, real=False):
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'bin').mkdir()
    (tmp_path / 'returns.csv').write_text('fixture\n')
    (tmp_path / 'bin/sbatch').write_text('#!/bin/bash\nexit 99\n')
    (tmp_path / 'bin/sbatch').chmod(0o755)
    mock = '''import json, os, pathlib, sys
path=pathlib.Path(os.environ['CAPTURE'])
keys=['RUN_TAG','OUT_ROOT','TEMPORAL_LOSS','HUBER_DELTA','SPEARMAN_TAU','LISTNET_TAU','LISTMLE_TAU','APPROXNDCG_TAU','SEED','MODELS','CROSS_LOSSES','CROSS_MASK','HORIZONS','TRADE_WINDOWS','LAMBDAS','B3_AFTER_JOB','GRES','MAX_CONCURRENT_JOBS','MOMENTUM_MODE','MOMENTUM_INPUT_MODE']
row=dict(action=sys.argv[1], **{k:os.environ.get(k) for k in keys})
with path.open('a') as stream: stream.write(json.dumps(row)+'\\n')
if sys.argv[1]=='submit':
    counter=path.with_suffix('.counter')
    job=int(counter.read_text())+1 if counter.exists() else 1001
    counter.write_text(str(job))
    print(f'Job {job}: array local 0-1; tarefas globais 0-1')
    if os.environ.get('FAIL_BRANCH') and os.environ['RUN_TAG'].endswith(os.environ['FAIL_BRANCH']): sys.exit(1)
'''
    (tmp_path / 'mock.py').write_text(mock)
    for name in ('run_temporal_experiment.sh', 'run_composite_trading_experiment.sh'):
        if real:
            shutil.copy(ROOT / 'scripts' / name, tmp_path / 'scripts' / name)
        else:
            (tmp_path / 'scripts' / name).write_text('#!/bin/bash\nexec "$PYTHON_BIN" "$TFB_ROOT/mock.py" "$@"\n')
    shutil.copy(ROOT / 'scripts/b3_experiment_env.sh', tmp_path / 'scripts/b3_experiment_env.sh')
    (tmp_path / 'scripts/b3_experiment.py').write_text(
        "import json,os\nprint(json.dumps(dict(config=json.loads(os.environ['B3_INLINE_CONFIG']), out_root=os.environ['OUT_ROOT'], argv=__import__('sys').argv)))\n")
    env = dict(TFB_ROOT=str(tmp_path), NS_PYTHON_BIN=sys.executable,
               NS_VENV_PATH=str(tmp_path / 'no_venv'),
               NS_RETURNS_FILE=str(tmp_path / 'returns.csv'),
               CAPTURE=str(tmp_path / 'capture.jsonl'),
               PRED_ROOT=str(tmp_path / 'pred'), BACKTEST_ROOT=str(tmp_path / 'backtest'),
               PATH=str(tmp_path / 'bin') + os.pathsep + os.environ['PATH'])
    return env


def records(tmp_path):
    return [json.loads(line) for line in (tmp_path / 'capture.jsonl').read_text().splitlines()]


def test_default_count_and_benchmark_only():
    result = run('--plan')
    assert result.returncode == 0, result.stderr
    assert 'TOTAL: 156 treinos unicos; 8 ramos' in result.stdout
    assert 'Controles lambda=0: 6; mistos: 120; ranking puro lambda=1: 30' in result.stdout
    assert 'Momentum: benchmark externo, 0 treinos' in result.stdout
    assert 'cauchy 0.025 natural' in result.stdout


@pytest.mark.parametrize('lambdas,total', [('0',6),('.5 .8',120),('1',30),('0 1',36)])
def test_endpoints_are_independently_selectable(lambdas,total):
    result=run('--plan', LAMBDAS_VALUES=lambdas)
    assert result.returncode==0,result.stderr
    assert f'TOTAL: {total} treinos' in result.stdout


def test_seed_models_horizons_kernels_can_all_be_changed():
    result=run('--plan', MODEL='DUET', LOOKBACKS_VALUES='32',
               HORIZON_VALUES='1 25', K_LOSS_VALUES='1',
               TEMPORAL_LOSSES_VALUES='mse', SEEDS_VALUES='2026 2027',
               KERNEL_SPECS='cauchy;student 10')
    assert result.returncode==0,result.stderr
    assert 'TOTAL: 184 treinos' in result.stdout
    assert 'student 0.025 natural 10' in result.stdout


@pytest.mark.parametrize('env', [dict(MOMENTUM_MODE='residual_output'),
                                  dict(MOMENTUM_INPUT_MODE='feature'),
                                  dict(LAMBDAS_VALUES='0 .5 0.50 1'),
                                  dict(SEEDS_VALUES='4294967296'),
                                  dict(TRAIN_VALUES='100'),
                                  dict(NS_SPEARMAN_TAU='0'),
                                  dict(NS_LISTNET_TAU='nan'),
                                  dict(NS_LISTMLE_TAU='0'),
                                  dict(NS_APPROXNDCG_TAU='-1'),
                                  dict(NS_MASK_A='inf'),
                                  dict(TEMPORAL_LOSSES_VALUES='mse mse')])
def test_invalid_or_confounding_parameters_are_rejected_in_plan(env):
    assert run('--plan',**env).returncode!=0


def test_actual_launcher_configs_produce156_unique_effective_tasks(tmp_path):
    env=fixture(tmp_path,real=True)
    result=run('--dry-run', **env, NS_GRES='', NS_HUBER_DELTA='0.3',NS_SPEARMAN_TAU='0.2',
               NS_LISTNET_TAU='.4',NS_LISTMLE_TAU='.5',NS_APPROXNDCG_TAU='.6')
    assert result.returncode==0,result.stderr
    rows=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    assert len(rows)==8
    spec=importlib.util.spec_from_file_location('global_test_b3',ROOT/'scripts/b3_experiment.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    paths=[]; scientific=[]
    for row in rows:
        config=row['config']
        mode=row['argv'][1]
        tasks=module.make_tasks(config,mode)
        for task in tasks:
            _,hyper,_,_=module.model_spec(config,task)
            paths.append(str(Path(row['out_root'])/module.output_path(task)))
            scientific.append((task['modelo'],task['lookback'],task['pred_len'],task['k'],
                task['loss'],task['cross_lambda'],
                config['loss']['loss_temporal'] if task['cross_lambda']<1 else None,
                config['strategy']['seed'],task.get('mask_type')))
            assert hyper.get('momentum_mode','off')=='off'
            assert hyper.get('momentum_input_mode','off')=='off'
            assert hyper['loss_spearman_tau']==.2
            assert hyper['loss_huber_delta']==.3
            assert hyper['loss_listnet_tau']==.4
            assert hyper['loss_listmle_tau']==.5
            assert hyper['loss_approxndcg_tau']==.6
            if mode=='temporal' and task['loss']=='huber':
                assert hyper['huber_delta']==.3
        assert row['out_root'].startswith(str(tmp_path/'pred'))
    assert len(paths)==len(set(paths))==len(set(scientific))==156
    assert sum('/temporal_loss=none/' in p for p in paths)==30


def test_submission_chain_and_retry_do_not_duplicate_jobs(tmp_path):
    env=fixture(tmp_path)
    first=run('--train', **env, NS_GRES='',NS_MAX_CONCURRENT_JOBS='3')
    assert first.returncode==0,first.stderr
    submitted=[row for row in records(tmp_path) if row['action']=='submit']
    assert len(submitted)==8
    assert [r['B3_AFTER_JOB'] for r in submitted]==['']+[str(n) for n in range(1001,1008)]
    assert all(r['GRES']=='' and r['MAX_CONCURRENT_JOBS']=='3' for r in submitted)
    second=run('--train',**env,NS_GRES='',NS_MAX_CONCURRENT_JOBS='3')
    assert second.returncode==0,second.stderr
    assert len([row for row in records(tmp_path) if row['action']=='submit'])==8
    changed=run('--train',**env,NS_SPEARMAN_TAU='.3')
    assert changed.returncode!=0
    assert 'Parametros alterados' in changed.stderr


def test_parallel_arrays_have_no_generated_dependency(tmp_path):
    env=fixture(tmp_path)
    result=run('--train',**env,NS_PARALLEL_BRANCHES='1')
    assert result.returncode==0,result.stderr
    submitted=[row for row in records(tmp_path) if row['action']=='submit']
    assert all(r['B3_AFTER_JOB']=='' for r in submitted)


def test_partial_failure_blocks_unsafe_retry(tmp_path):
    env=fixture(tmp_path)
    result=run('--train',**env,FAIL_BRANCH='composto/sem_kernel')
    assert result.returncode!=0
    second=run('--train',**env)
    assert second.returncode!=0
    assert 'incerta/parcial' in second.stderr
    assert len([row for row in records(tmp_path) if row['action']=='submit'])==2


def test_model_list_doubles_the_default_grade():
    result=run('--plan',MODEL='Nonstationary_Transformer simpleFORMER')
    assert result.returncode==0,result.stderr
    assert 'TOTAL: 312 treinos' in result.stdout


def test_each_new_listwise_pair_can_be_selected_alone():
    for loss in ('listnet','listmle','approxndcg'):
        result=run('--plan',CROSS_LOSSES_VALUES=f'{loss} {loss}_proximity')
        assert result.returncode==0,result.stderr
        assert 'TOTAL: 36 treinos' in result.stdout


def test_listwise_delta_is_inactive_and_does_not_repeat_training(tmp_path):
    env=fixture(tmp_path,real=True)
    result=run('--dry-run',**env,DELTAS_VALUES='0 .01 .1',
               CROSS_LOSSES_VALUES='listnet listnet_proximity listmle listmle_proximity approxndcg approxndcg_proximity')
    assert result.returncode==0,result.stderr
    assert 'TOTAL: 96 treinos' in result.stdout
    rows=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    spec=importlib.util.spec_from_file_location('global_listwise_b3',ROOT/'scripts/b3_experiment.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tasks=[]
    for row in rows:
        tasks.extend(module.make_tasks(row['config'],row['argv'][1]))
    assert len(tasks)==96
    assert all(t['cross_delta']==0 for t in tasks)
