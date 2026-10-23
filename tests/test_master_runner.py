"""MASTER uses B3's existing design and never starts Python on the login host."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import b3_experiment as experiment


@pytest.fixture
def config():
    env = dict(os.environ, TFB_ROOT=str(ROOT), PYTHON_BIN=sys.executable, SLURM_JOB_ID='test')
    result = subprocess.run(['bash', str(ROOT / 'scripts/run_master_experiment.sh'),
                             '--on-node', 'composto', '--print-config'],
                            env=env, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_master_registered_without_generic_transformer_adapter(config):
    config['strategy']['seed'] = 73
    task = experiment.make_tasks(config, 'composto')[0]
    name, hyper, adapter, deterministic = experiment.model_spec(config, task)
    assert name == 'master.MASTER' and adapter is None and deterministic == 'full'
    assert hyper['seed'] == 73
    assert hyper['norm'] is True
    assert hyper['master_expected_assets'] == 55
    assert hyper['master_expected_dates'] == 3664
    assert hyper['horizon'] == hyper['pred_len'] == task['pred_len']


def test_all_composite_choices_and_multihorizon_cover(config):
    assert set(config['cross_losses']) == {'mse', 'bpr', 'ranknet', 'hinge', 'listnet',
        'pairwise_mse', 'hinge_proximity', 'ranknet_proximity', 'pairwise_mse_proximity'}
    config['trade_windows'] = [1, 5, 10, 15, 20, 25]
    tasks = experiment.make_tasks(config, 'composto')
    assert {t['pred_len'] for t in tasks} == {1, 5, 10, 15, 20, 25}
    assert all(t['pred_len'] % t['k'] == 0 for t in tasks)
    assert {t['k'] for t in tasks} == {1, 5, 10, 15, 20, 25}


@pytest.mark.parametrize('temporal', ['mse', 'mae', 'huber'])
def test_temporal_label_matches_actual_loss(config, temporal):
    config['loss']['loss_temporal'] = temporal
    tasks = experiment.make_tasks(config, 'temporal')
    for task in tasks:
        assert task['loss'] == temporal and task['k'] is None
        assert f'loss={temporal}' in str(experiment.output_path(task))
        assert experiment.model_spec(config, task)[1]['loss'] == temporal


def test_inactive_mse_delta_does_not_duplicate_tasks(config):
    config['cross_losses'] = ['mse']
    config['deltas'] = [0, .025, .05]
    assert {t['cross_delta'] for t in experiment.make_tasks(config, 'composto')} == {0}


def test_master_accepts_partial_training_batches(config):
    config['lookbacks'], config['horizons'] = [4], [5]
    task = experiment.make_tasks(config, 'temporal')[0]
    experiment.validate_lengths(config, [task], dict(n_rows=48, n_assets=3))


def test_worker_preserves_master_launcher_and_existing_others(config, monkeypatch):
    monkeypatch.setenv('B3_WORKER_LAUNCHER', str(ROOT / 'scripts/run_master_experiment.sh'))
    command = experiment.sbatch_command('composto', 11, 4)
    assert command[-6:] == [str(ROOT / 'scripts/run_master_experiment.sh'), '--on-node',
                           'composto', 'worker', '--offset', '11']
    monkeypatch.delenv('B3_WORKER_LAUNCHER')
    assert experiment.sbatch_command('composto', 11, 4)[-4:] == [
        str(ROOT / 'scripts/run_composite_trading_experiment.sh'), 'worker', '--offset', '11']


def test_checkpoint_and_audit_survive_tfb_temporary_results(config, tmp_path):
    task = experiment.make_tasks(config, 'temporal')[0]
    command = experiment.benchmark_command(config, task, 'panel.csv', 3664,
                                          tmp_path / 'temporary', tmp_path / 'task')
    hyper = json.loads(command[command.index('--model-hyper-params') + 1])
    assert hyper['master_checkpoint_path'] == str(tmp_path / 'task/master_checkpoint.pt')
    assert hyper['master_audit_path'] == str(tmp_path / 'task/master_training.json')
    assert '--adapter' not in command


def test_changed_feature_bytes_invalidate_design(config, monkeypatch, tmp_path):
    bundle = tmp_path / 'features'
    bundle.mkdir()
    feature = bundle / 'ativos_features.csv'
    feature.write_text('original')
    source = tmp_path / 'returns.csv'
    source.write_text('returns')
    config['model_defaults']['MASTER'].update(master_bundle_dir=str(bundle), master_returns_path=str(source))
    original_hash = experiment.file_hash
    monkeypatch.setattr(experiment, 'file_hash',
                        lambda path: original_hash(path) if Path(path).exists() else 'upstream_fixture')
    metadata = dict(source=str(source), source_sha256=original_hash(source),
                    dataset_sha256='prepared', source_data_kind='log_return', step_offset=0)
    first = experiment.provenance(config, metadata)
    feature.write_text('modified')
    second = experiment.provenance(config, metadata)
    assert experiment.digest(first) != experiment.digest(second)
    assert 'ts_benchmark/baselines/master/master.py' in first['code_sha256']
    assert 'ts_benchmark/baselines/custom_losses.py' in first['code_sha256']
    source.write_text('modified returns')
    assert experiment.digest(second) != experiment.digest(experiment.provenance(config, metadata))
    wrong = dict(metadata, source=str(tmp_path / 'other.csv'))
    with pytest.raises(ValueError, match='mesmo painel bruto'):
        experiment.provenance(config, wrong)


@pytest.mark.parametrize('mode', ['smoke', 'temporal', 'composto'])
def test_login_launcher_only_submits_sbatch(mode, tmp_path):
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    sbatch = fake_bin / 'sbatch'
    sbatch.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE"\nprintf "12345\\n"\n')
    sbatch.chmod(0o755)
    capture = tmp_path / 'arguments.txt'
    env = dict(os.environ, PATH=str(fake_bin) + os.pathsep + os.environ['PATH'],
               CAPTURE=str(capture), LOG_ROOT=str(tmp_path / 'logs'),
               PYTHON_BIN='/must/not/run/python')
    result = subprocess.run(['bash', str(ROOT / 'scripts/run_master_experiment.sh'), mode],
                            env=env, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == '12345'
    arguments = capture.read_text().splitlines()
    assert '--partition=gorgonas_dev' in arguments
    assert arguments[-2:] == ['--on-node', mode]


def test_on_node_flag_without_allocation_is_rejected(tmp_path):
    env = dict(os.environ, PYTHON_BIN='/must/not/run/python')
    env.pop('SLURM_JOB_ID', None)
    result = subprocess.run(['bash', str(ROOT / 'scripts/run_master_experiment.sh'),
                             '--on-node', 'temporal', '--print-config'],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'somente dentro de job Slurm' in result.stderr
