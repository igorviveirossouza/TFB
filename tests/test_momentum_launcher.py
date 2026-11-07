"""Momentum opt-in contracts across shell, task identity and worker completion."""
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import b3_experiment as experiment


def shell_config(**overrides):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('MOMENTUM_', 'B3_')) and k != 'LRADJ'}
    env.update(TFB_ROOT=str(ROOT), PYTHON_BIN=sys.executable,
               VENV_PATH='/nonexistent/venv', MODELS='Nonstationary_Transformer',
               LOOKBACKS='32 64', HORIZONS='1 5', TRADE_WINDOWS='1',
               CROSS_LOSSES='ranknet ranknet_proximity', LAMBDAS='0.5 1',
               CROSS_MASK='cauchy 0.025 natural')
    env.update(overrides)
    return subprocess.run(['bash', str(ROOT / 'scripts/run_composite_trading_experiment.sh'),
                           '--print-config'], env=env, text=True, capture_output=True)


@pytest.fixture
def config():
    result = shell_config()
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def load_inline(config, monkeypatch):
    monkeypatch.setenv('B3_INLINE_CONFIG', json.dumps(config))
    monkeypatch.setenv('B3_CONFIG_SOURCE', str(ROOT / 'scripts/run_composite_trading_experiment.sh'))
    return experiment.load_config()[1]


@pytest.mark.parametrize('mode', ['composto', 'temporal', 'var'])
def test_off_preserves_config_tasks_paths_and_hyperparameters(config, monkeypatch, mode):
    changed = copy.deepcopy(config)
    changed['loss'].update(momentum_mode='off', momentum_lookback=512, momentum_eta=.25,
                           momentum_gain_init=.1, momentum_diagnostics=True)
    changed = load_inline(changed, monkeypatch)
    assert changed == config
    before = experiment.make_tasks(config, mode)
    after = experiment.make_tasks(changed, mode)
    assert before == after
    for old, new in zip(before, after):
        assert experiment.output_path(old) == experiment.output_path(new)
        assert experiment.model_spec(config, old) == experiment.model_spec(changed, new)
        assert not any(k.startswith('momentum_') for k in new)


def test_shell_values_reach_tasks_and_only_enabled_hyperparameters(config):
    result = shell_config(MOMENTUM_MODE='residual_output', MOMENTUM_LOOKBACK='512', MOMENTUM_ETA='.01')
    assert result.returncode == 0, result.stderr
    enabled = json.loads(result.stdout)
    tasks = experiment.make_tasks(enabled, 'composto')
    assert len(tasks) == len(experiment.make_tasks(config, 'composto'))
    for task in tasks:
        assert task['momentum_lookback'] == 512 and task['momentum_eta'] == .01
        path = str(experiment.output_path(task))
        assert '/momentum=residual_output/momentum_lb=512/momentum_eta=0.01/modelo=' in path
        hyper = experiment.model_spec(enabled, task)[1]
        assert hyper['momentum_mode'] == 'residual_output'
        assert hyper['seq_len'] == task['lookback']
        assert 'momentum_state_path' not in hyper
        command = experiment.benchmark_command(enabled, task, 'data.csv', 4000, Path('tmp'), Path('out'))
        runtime = json.loads(command[command.index('--model-hyper-params') + 1])
        assert runtime['momentum_state_path'] == 'out/momentum_state.json'
    temporal = experiment.make_tasks(enabled, 'temporal')
    assert temporal == experiment.make_tasks(config, 'temporal')
    for task in temporal:
        assert experiment.model_spec(enabled, task) == experiment.model_spec(config, task)


def test_default_momentum_lookback_tracks_model_and_paths_distinguish_eta(config):
    config['loss']['momentum_mode'] = 'residual_output'
    tasks = experiment.make_tasks(config, 'composto')
    assert all(t['momentum_lookback'] == t['lookback'] for t in tasks)
    first = tasks[0]
    assert experiment.output_path(first) != experiment.output_path(dict(first, momentum_eta=.000000001))
    assert experiment.output_path(first) != experiment.output_path(dict(first, momentum_lookback=512))


@pytest.mark.parametrize('overrides', [
    {'MOMENTUM_MODE': 'resdual_output'}, {'MOMENTUM_LOOKBACK': '0'},
    {'MOMENTUM_LOOKBACK': '1.5'}, {'MOMENTUM_LOOKBACK': '-1'},
    {'MOMENTUM_ETA': '-.01'}, {'MOMENTUM_ETA': 'nan'},
    {'MOMENTUM_ETA': 'inf'}, {'MOMENTUM_ETA': '1e999'},
    {'MOMENTUM_GAIN_INIT': 'nan'}, {'MOMENTUM_GAIN_INIT': 'inf'},
    {'MOMENTUM_GAIN_INIT': '1e999'}, {'MOMENTUM_GAIN_INIT': 'true'},
    {'MOMENTUM_DIAGNOSTICS': 'yes'}, {'MOMENTUM_DIAGNOSTICS': '2'},
    {'LRADJ': 'typo'}, {'LRADJ': 'TST'},
])
def test_invalid_shell_settings_fail_before_training(overrides):
    result = shell_config(**overrides)
    assert result.returncode != 0


@pytest.mark.parametrize('key,value', [
    ('momentum_mode', 'unknown'), ('momentum_mode', []), ('momentum_lookback', True),
    ('momentum_lookback', 0), ('momentum_lookback', 1.5),
    ('momentum_eta', True), ('momentum_eta', -.5),
    ('momentum_eta', float('nan')), ('momentum_etas', .1),
    ('momentum_gain_init', True), ('momentum_gain_init', float('nan')),
    ('momentum_gain_init', float('inf')), ('momentum_gain_init', '.1'),
    ('momentum_diagnostics', 'true'), ('momentum_diagnostics', 1),
])
def test_invalid_inline_settings_fail(config, monkeypatch, key, value):
    config['loss'][key] = value
    with pytest.raises(ValueError, match='[Mm][Oo][Mm][Ee][Nn][Tt][Uu][Mm]'):
        load_inline(config, monkeypatch)


@pytest.mark.parametrize('model', ['MASTER', 'VAR', 'simpleMLP'])
def test_unverified_architectures_rejected(config, monkeypatch, model):
    config['models'] = [model]
    config['model_defaults'][model] = {}
    config['loss']['momentum_mode'] = 'residual_output'
    with pytest.raises(ValueError, match='não validado'):
        load_inline(config, monkeypatch)


def test_effective_history_is_validated_before_submission(config):
    config['loss'].update(momentum_mode='residual_output', momentum_lookback=4000)
    with pytest.raises(ValueError, match='Treino insuficiente'):
        experiment.validate_lengths(config, experiment.make_tasks(config, 'composto'),
                                    {'n_rows': 4000, 'n_assets': 55})


def test_price_input_rejected_before_training(config, monkeypatch):
    config['loss']['momentum_mode'] = 'residual_output'
    config['source_data_kind'] = 'price'
    with pytest.raises(ValueError, match='SOURCE_DATA_KIND=log_return'):
        load_inline(config, monkeypatch)


@pytest.mark.parametrize('loss', ['mse', 'pairwise_mse', 'pairwise_mse_proximity'])
def test_value_losses_require_comparable_normalization(config, monkeypatch, loss):
    config['cross_losses'] = [loss]
    config['loss'].update(momentum_mode='residual_output', loss_cross_score_normalization='none')
    with pytest.raises(ValueError, match='exige zscore'):
        load_inline(config, monkeypatch)
    config['loss']['loss_cross_score_normalization'] = 'zscore'
    load_inline(config, monkeypatch)


def test_new_module_hashes_only_for_enabled_design(config, monkeypatch):
    monkeypatch.setattr(experiment, 'file_hash', lambda path: str(path))
    metadata = dict(source_sha256='source', dataset_sha256='dataset', step_offset=1)
    old = experiment.provenance(config, metadata)['code_sha256']
    config['loss']['momentum_mode'] = 'residual_output'
    new = experiment.provenance(config, metadata)['code_sha256']
    assert set(new) - set(old) == {'ts_benchmark/baselines/momentum_score.py',
                                   'ts_benchmark/baselines/momentum_score_io.py'}


@pytest.mark.parametrize('wrapper,function', [
    ('run_ns_kloss_isolado.sh', 'training_launcher'),
    ('run_ns_kernels_isolado.sh', 'train_launcher'),
])
def test_wrappers_force_temporal_off_and_forward_composite(tmp_path, wrapper, function):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    for name in ('run_temporal_experiment.sh', 'run_composite_trading_experiment.sh', 'b3_experiment_env.sh'):
        shutil.copy(ROOT / 'scripts' / name, scripts / name)
    (scripts / 'b3_experiment.py').write_text("import os\nprint(os.environ['B3_INLINE_CONFIG'])\n")
    copy_path = scripts / wrapper
    copy_path.write_text((ROOT / 'scripts' / wrapper).read_text().replace('main "$@"', ''))
    if function == 'train_launcher':
        body = 'source "$1"; grid; train_launcher 0 --print-config; train_launcher 1 --print-config'
    else:
        body = 'source "$1"; training_launcher temporal --print-config; training_launcher composto --print-config'
    env = dict(os.environ, TFB_ROOT=str(tmp_path), NS_PYTHON_BIN=sys.executable,
               NS_VENV_PATH='/nonexistent/venv', MOMENTUM_MODE='residual_output',
               MOMENTUM_LOOKBACK='128', MOMENTUM_ETA='.2')
    result = subprocess.run(['bash', '-c', body, 'wrapper', str(copy_path)], env=env,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    control, compound = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    assert not any(key.startswith('momentum_') for key in control['loss'])
    assert {key: compound['loss'][key] for key in experiment.MOMENTUM_KEYS if key in compound['loss']} == dict(
        momentum_mode='residual_output', momentum_lookback=128, momentum_eta=.2)


def test_kernel_submission_fingerprint_preserves_off_and_control(tmp_path):
    script = tmp_path / 'wrapper.sh'
    script.write_text((ROOT / 'scripts/run_ns_kernels_isolado.sh').read_text().replace('main "$@"', ''))
    body = '''source "$1"; grid
branch_fingerprint 0; branch_fingerprint 1
MOMENTUM_MODE=residual_output; MOMENTUM_ETA=.1
branch_fingerprint 0; branch_fingerprint 1
MOMENTUM_ETA=.2; branch_fingerprint 1
'''
    result = subprocess.run(['bash', '-c', body, 'wrapper', str(script)],
                            env=dict(os.environ, MOMENTUM_MODE='off'), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    control_off, comp_off, control_on, comp_on, comp_eta = result.stdout.splitlines()
    assert control_off == control_on
    assert len({comp_off, comp_on, comp_eta}) == 3


def test_worker_exports_score_before_completion_and_validates_on_resume(config, tmp_path, monkeypatch):
    import pandas as pd
    import b3_experiment_io as io
    config['loss'].update(momentum_mode='residual_output', momentum_lookback=32, momentum_eta=.1)
    task = experiment.make_tasks(config, 'composto')[0]
    out = tmp_path / 'out'
    group = out / experiment.group_path(task)
    group.mkdir(parents=True)
    source = tmp_path / 'data.csv'; source.write_text('raw')
    metadata = dict(dataset=str(source), calendar=str(source),
                    dataset_sha256=experiment.file_hash(source), calendar_sha256=experiment.file_hash(source),
                    step_offset=1)
    monkeypatch.setattr(experiment, 'provenance', lambda *args: {})
    (group / 'run_config.json').write_text(json.dumps(dict(metadata=metadata, design_sha256=experiment.digest({}))))
    header = dict(task, slurm_job_id='', parquet_sha256='')
    (group / 'completed_tasks.csv').write_text(experiment.csv_text([header]).splitlines()[0] + '\n')
    monkeypatch.setattr(experiment, 'audit_row', lambda *args: dict(task))
    monkeypatch.setattr(io, 'load_panel', lambda path: pd.DataFrame([[0., .1]] * 100))
    monkeypatch.setattr(experiment.subprocess, 'run', lambda *args, **kwargs: None)
    monkeypatch.setenv('RESULT_ROOT', str(tmp_path / 'results'))
    output = out / experiment.output_path(task)
    events = []
    def raw_export(*args, **kwargs):
        (output / 'part.parquet').write_bytes(b'forecast')
        events.append('raw')
    def score_export(*args, **kwargs):
        assert (output / 'part.parquet').exists()
        assert kwargs['step_offset'] == 1
        events.append('scores')
    def validate(path):
        assert path == output
        events.append('validate')
    monkeypatch.setattr(io, 'export_predictions', raw_export)
    monkeypatch.setattr(io, 'export_momentum_scores', score_export, raising=False)
    monkeypatch.setattr(io, 'validate_momentum_export', validate, raising=False)
    experiment.worker(config, [task], out, 0)
    assert events == ['raw', 'scores']
    rows = list(csv.DictReader((group / 'completed_tasks.csv').open()))
    assert rows[0]['momentum_mode'] == 'residual_output'
    experiment.worker(config, [task], out, 0)
    assert events == ['raw', 'scores', 'validate']
    def corrupt(path):
        raise ValueError('momentum sidecar altered')
    monkeypatch.setattr(io, 'validate_momentum_export', corrupt)
    with pytest.raises(ValueError, match='sidecar altered'):
        experiment.worker(config, [task], out, 0)


@pytest.mark.parametrize('gain', ['0.1', '-0.1'])
@pytest.mark.parametrize('lradj', ['type1', 'constant'])
def test_gain_diagnostics_and_schedule_reach_actual_runtime(config, gain, lradj):
    result = shell_config(MOMENTUM_MODE='residual_output', MOMENTUM_LOOKBACK='512',
                          MOMENTUM_ETA='.01', MOMENTUM_GAIN_INIT=gain,
                          MOMENTUM_DIAGNOSTICS='1', LRADJ=lradj)
    assert result.returncode == 0, result.stderr
    enabled = json.loads(result.stdout)
    task = experiment.make_tasks(enabled, 'composto')[0]
    assert task['momentum_gain_init'] == float(gain)
    assert task['momentum_diagnostics'] is True
    assert f"momentum_gain_init={float(gain)!r}" in str(experiment.output_path(task))
    old = dict(task)
    old.pop('momentum_gain_init')
    assert experiment.output_path(task) != experiment.output_path(old)
    command = experiment.benchmark_command(enabled, task, 'data.csv', 4000, Path('tmp'), Path('out'))
    runtime = json.loads(command[command.index('--model-hyper-params') + 1])
    assert runtime['momentum_gain_init'] == float(gain)
    assert runtime['momentum_diagnostics'] is True
    assert runtime['lradj'] == lradj
    control = experiment.make_tasks(enabled, 'temporal')[0]
    hyper = experiment.model_spec(enabled, control)[1]
    assert not any(k.startswith('momentum_') for k in hyper)
    assert hyper['lradj'] == lradj


def test_neutral_options_preserve_old_enabled_config_identity(monkeypatch):
    result = shell_config(MOMENTUM_MODE='residual_output', MOMENTUM_LOOKBACK='512', MOMENTUM_ETA='.01')
    assert result.returncode == 0, result.stderr
    default = json.loads(result.stdout)
    assert 'momentum_gain_init' not in default['loss']
    assert 'momentum_diagnostics' not in default['loss']
    assert 'lradj' not in default['training']
    explicit = shell_config(MOMENTUM_MODE='residual_output', MOMENTUM_LOOKBACK='512',
                            MOMENTUM_ETA='.01', MOMENTUM_GAIN_INIT='0.0',
                            MOMENTUM_DIAGNOSTICS='false', LRADJ='')
    assert explicit.returncode == 0, explicit.stderr
    assert json.loads(explicit.stdout) == default
    inline = copy.deepcopy(default)
    inline['loss'].update(momentum_gain_init=0.0, momentum_diagnostics=False)
    assert load_inline(inline, monkeypatch) == default


@pytest.mark.parametrize('value', ['typo', [], None, True])
def test_invalid_inline_learning_rate_schedule(config, monkeypatch, value):
    config['training']['lradj'] = value
    with pytest.raises(ValueError, match='LRADJ'):
        load_inline(config, monkeypatch)


def test_retry_fingerprint_tracks_only_effective_new_options(tmp_path):
    script = tmp_path / 'wrapper.sh'
    script.write_text((ROOT / 'scripts/run_ns_kernels_isolado.sh').read_text().replace('main "$@"', ''))
    body = '''source "$1"; grid
branch_fingerprint 0; branch_fingerprint 1
MOMENTUM_GAIN_INIT=0.0; MOMENTUM_DIAGNOSTICS=false
branch_fingerprint 0; branch_fingerprint 1
MOMENTUM_GAIN_INIT=.1
branch_fingerprint 0; branch_fingerprint 1
MOMENTUM_DIAGNOSTICS=1
branch_fingerprint 0; branch_fingerprint 1
NS_LRADJ=constant
branch_fingerprint 0; branch_fingerprint 1
'''
    env = {k: v for k, v in os.environ.items() if not k.startswith(('MOMENTUM_', 'NS_'))}
    env.update(MOMENTUM_MODE='residual_output')
    result = subprocess.run(['bash', '-c', body, 'wrapper', str(script)],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    control, comp, control0, comp0, control1, comp1, control2, comp2, control3, comp3 = result.stdout.splitlines()
    assert control == control0 == control1 == control2
    assert comp == comp0
    assert len({comp, comp1, comp2, comp3}) == 4
    assert control3 != control
