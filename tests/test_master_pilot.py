"""The full pilot submits exactly three fits and no Python on the login host."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import b3_experiment as experiment

CAPTURE_KEYS = '''TFB_ROOT PYTHON_BIN RUN_TAG OUT_ROOT RESULT_ROOT LOG_ROOT
TRAIN_PERCENTAGES LOOKBACKS HORIZONS TRADE_WINDOWS TEMPORAL_LOSS CROSS_LOSSES
LAMBDAS DELTAS CROSS_MASK NUM_EPOCHS PATIENCE BATCH_SIZE LEARNING_RATE
TRAIN_RATIO_IN_TV SEED D_MODEL T_HEADS S_HEADS DROPOUT GATE_BETA LRADJ
ASSET_BLOCKS USE_MARKET DEVICE PARTITION TIME_LIMIT MEMORY CPUS_PER_TASK
MAX_CONCURRENT_JOBS ARRAY_CHUNK_SIZE NODELIST GRES MANTER_CSV'''.split()


def capture_submissions(tmp_path, arguments=()):
    binary = tmp_path / 'bin'
    binary.mkdir()
    sbatch = binary / 'sbatch'
    sbatch.write_text('''#!/usr/bin/env bash
set -eu
mode="${LOG_ROOT##*/}"
printf '%s\\n' "$@" > "$CAPTURE_ROOT/$mode.args"
for key in ''' + ' '.join(CAPTURE_KEYS) + '''; do
  printf '%s=%s\\n' "$key" "${!key}"
done > "$CAPTURE_ROOT/$mode.env"
printf '12345\\n'
''')
    sbatch.chmod(0o755)
    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'],
               TFB_ROOT=str(ROOT), CAPTURE_ROOT=str(tmp_path),
               PYTHON_BIN='/must/not/run/python',
               MASTER_PILOT_LOG_ROOT=str(tmp_path / 'logs'),
               MASTER_PILOT_RESULT_ROOT=str(tmp_path / 'results'),
               MASTER_PILOT_OUT_ROOT=str(tmp_path / 'predictions'),
               MASTER_PILOT_TAG='master_pilot_test',
               MASTER_PILOT_PARTITION='gorgonas',
               MASTER_PILOT_TIME_LIMIT='08:00:00', MASTER_PILOT_MEMORY='32G',
               # Prior experiment environment must not expand/re-route this pilot.
               OUT_ROOT='/stale/output', RESULT_ROOT='/stale/result',
               LOG_ROOT='/stale/logs', RUN_TAG='stale', DEVICE='cuda',
               GRES='gpu:0', NODELIST='old-node', TRAIN_PERCENTAGES='70 80 90',
               LOOKBACKS='32 128 256', HORIZONS='1 5 25',
               CROSS_LOSSES='hinge listnet', LAMBDAS='0.1 0.5 0.8')
    env.pop('SLURM_JOB_ID', None)
    result = subprocess.run(['bash', str(ROOT / 'scripts/run_master_pilot.sh'), *arguments],
                            env=env, text=True, capture_output=True, check=True)
    captured = {}
    for mode in ('temporal', 'composto'):
        captured[mode] = dict(line.split('=', 1) for line in
                              (tmp_path / f'{mode}.env').read_text().splitlines())
    return result, captured


@pytest.mark.parametrize('arguments', [(), ('--dry-run',), ('--prepare',)])
def test_login_only_submits_two_preparations_with_isolated_paths(tmp_path, arguments):
    result, captured = capture_submissions(tmp_path, arguments)
    assert result.stdout.count('12345') == 2
    for mode, env in captured.items():
        args = (tmp_path / f'{mode}.args').read_text().splitlines()
        assert '--partition=gorgonas_dev' in args
        assert args[-len(arguments)-2:] == ['--on-node', mode, *arguments]
        assert env['PYTHON_BIN'] == '/must/not/run/python'
        assert env['OUT_ROOT'] == str(tmp_path / 'predictions')
        assert env['LOG_ROOT'] == str(tmp_path / 'logs' / mode)
        assert env['RESULT_ROOT'] == str(tmp_path / 'results' / mode)
        assert env['DEVICE'] == 'cpu' and env['GRES'] == '' and env['NODELIST'] == ''
        assert env['PARTITION'] == 'gorgonas' and env['TIME_LIMIT'] == '08:00:00'
        assert env['MAX_CONCURRENT_JOBS'] == '1'


def test_actual_configs_have_three_full_fits_and_all_test_origins(tmp_path):
    _, captured = capture_submissions(tmp_path)
    total = 0
    for mode, env in captured.items():
        result = subprocess.run(['bash', str(ROOT / 'scripts/run_master_experiment.sh'),
                                 '--on-node', mode, '--print-config'],
                                env={**os.environ, **env, 'SLURM_JOB_ID': 'test',
                                     'PYTHON_BIN': sys.executable},
                                text=True, capture_output=True, check=True)
        config = json.loads(result.stdout)
        tasks = experiment.make_tasks(config, mode)
        total += len(tasks)
        assert len(tasks) == (1 if mode == 'temporal' else 2)
        assert {t['loss'] for t in tasks} == ({'mse'} if mode == 'temporal' else
                                             {'ranknet', 'ranknet_proximity'})
        assert config['training'] == {'num_epochs': 100, 'patience': 10, 'lr': 0.0001}
        for task in tasks:
            assert task['lookback'] == 32 and task['pred_len'] == 5 and task['treino'] == 80
            _, hyper, _, _ = experiment.model_spec(config, task)
            assert hyper.get('master_max_train_batches') is None
            assert hyper.get('master_max_validation_batches') is None
            assert hyper['batch_size'] == 32 and hyper['device'] == 'cpu'
            assert hyper['master_use_market'] is True
            assert hyper['master_asset_blocks'] == ['retornos', 'ohlcv', 'atividade']
            split = experiment.split_counts(config, task, 3664)
            assert split['n_train'] - 32 - 5 + 1 == 2528
            assert split['n_validation'] - 5 + 1 == 363
            assert split['n_windows'] == 729
            if mode == 'composto':
                assert task['k'] == 1 and task['cross_lambda'] == .5
            if task['loss'] == 'ranknet_proximity':
                assert hyper['loss_cross_mask'][:2] == ['gaussian', .025]
                assert len(hyper['loss_cross_mask']) == 2 or hyper['loss_cross_mask'][2] == 'natural'
    assert total == 3


def test_unknown_pilot_action_fails_before_submission(tmp_path):
    result = subprocess.run(['bash', str(ROOT / 'scripts/run_master_pilot.sh'), 'smoke'],
                            text=True, capture_output=True)
    assert result.returncode == 2
    assert 'Uso:' in result.stderr
