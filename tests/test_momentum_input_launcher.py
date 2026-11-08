"""Opt-in momentum feature contracts through environment, identity and workers."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from test_momentum_launcher import shell_config, load_inline, config
import b3_experiment as experiment

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('mode', ['zero', 'feature'])
def test_feature_round_trip_from_export_to_runtime(config, mode):
    result = shell_config(MOMENTUM_INPUT_MODE=mode, MOMENTUM_INPUT_LOOKBACK='512',
                          VALIDATION_MODE='ordered_sample_mean')
    assert result.returncode == 0, result.stderr
    enabled = json.loads(result.stdout)
    assert enabled['training']['validation_mode'] == 'ordered_sample_mean'
    assert enabled['loss']['momentum_input_mode'] == mode
    tasks = experiment.make_tasks(enabled, 'composto')
    assert len(tasks) == len(experiment.make_tasks(config, 'composto'))
    for task in tasks:
        assert task['momentum_input_lookback'] == 512
        assert f'/momentum_input={mode}/momentum_input_lb=512/modelo=' in str(experiment.output_path(task))
        command = experiment.benchmark_command(enabled, task, 'data.csv', 4000,
                                                Path('tmp'), Path('out'))
        hyper = json.loads(command[command.index('--model-hyper-params') + 1])
        assert hyper['momentum_input_mode'] == mode
        assert hyper['momentum_input_lookback'] == 512
        assert hyper['validation_mode'] == 'ordered_sample_mean'
        assert hyper['seq_len'] == task['lookback']
        assert 'momentum_mode' not in hyper
        assert 'momentum_state_path' not in hyper
    assert experiment.make_tasks(enabled, 'temporal') == experiment.make_tasks(config, 'temporal')
    for task in experiment.make_tasks(enabled, 'temporal'):
        assert not any(key.startswith('momentum_') for key in experiment.model_spec(enabled, task)[1])


def test_off_canonicalizes_to_legacy(config, monkeypatch):
    changed = copy.deepcopy(config)
    changed['loss'].update(momentum_input_mode='off', momentum_input_lookback=512)
    assert load_inline(changed, monkeypatch) == config
    result = shell_config(MOMENTUM_INPUT_MODE='off', MOMENTUM_INPUT_LOOKBACK='512')
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == config


def test_resolved_lookback_and_mode_are_distinct_identities(config):
    config['loss'].update(momentum_input_mode='feature')
    tasks = experiment.make_tasks(config, 'composto')
    assert all(task['momentum_input_lookback'] == task['lookback'] for task in tasks)
    task = tasks[0]
    assert len({experiment.output_path(task),
                experiment.output_path(dict(task, momentum_input_mode='zero')),
                experiment.output_path(dict(task, momentum_input_lookback=512))}) == 3


@pytest.mark.parametrize('overrides', [
    {'MOMENTUM_INPUT_MODE': 'on'}, {'MOMENTUM_INPUT_LOOKBACK': '0'},
    {'MOMENTUM_INPUT_LOOKBACK': '-1'}, {'MOMENTUM_INPUT_LOOKBACK': '1.5'},
    {'MOMENTUM_INPUT_LOOKBACK': 'true'}, {'VALIDATION_MODE': 'sample_mean'},
    {'MOMENTUM_INPUT_MODE': 'feature', 'MOMENTUM_MODE': 'residual_output'},
    {'MOMENTUM_INPUT_MODE': 'zero', 'MOMENTUM_MODE': 'residual_output'},
])
def test_invalid_exports_rejected(overrides):
    result = shell_config(**overrides)
    assert result.returncode != 0


@pytest.mark.parametrize('key,value', [
    ('momentum_input_mode', []), ('momentum_input_mode', 'on'),
    ('momentum_input_lookback', True), ('momentum_input_lookback', 0),
    ('momentum_input_lookback', 1.5), ('momentum_input_lookbacks', 512),
])
def test_invalid_inline_options_rejected(config, monkeypatch, key, value):
    config['loss'][key] = value
    with pytest.raises(ValueError, match='[Mm]omentum|MOMENTUM'):
        load_inline(config, monkeypatch)


@pytest.mark.parametrize('model', ['DUET', 'TimesNet', 'FEDformer', 'MASTER'])
def test_only_supported_architecture_allowed(config, monkeypatch, model):
    config['models'] = [model]
    config['model_defaults'][model] = {}
    config['loss']['momentum_input_mode'] = 'feature'
    with pytest.raises(ValueError, match='nao validado'):
        load_inline(config, monkeypatch)


@pytest.mark.parametrize('field', ['source_data_kind', 'loss_data_kind'])
def test_only_raw_log_returns_allowed(config, monkeypatch, field):
    config['loss']['momentum_input_mode'] = 'feature'
    (config if field == 'source_data_kind' else config['loss'])[field] = 'price'
    with pytest.raises(ValueError, match='log_return'):
        load_inline(config, monkeypatch)


def test_history_validation_includes_auxiliary_lookback(config):
    config['loss'].update(momentum_input_mode='zero', momentum_input_lookback=4000)
    with pytest.raises(ValueError, match='Treino insuficiente'):
        experiment.validate_lengths(config, experiment.make_tasks(config, 'composto'),
                                    {'n_rows': 4000, 'n_assets': 55})


def test_provenance_records_every_input_specific_module(config, monkeypatch):
    monkeypatch.setattr(experiment, 'file_hash', lambda path: str(path))
    meta = dict(source_sha256='source', dataset_sha256='dataset', step_offset=1)
    old = experiment.provenance(config, meta)['code_sha256']
    config['loss']['momentum_input_mode'] = 'feature'
    new = experiment.provenance(config, meta)['code_sha256']
    assert set(new) - set(old) == {
        'ts_benchmark/baselines/momentum_input.py',
        'ts_benchmark/baselines/momentum_score.py',
        'ts_benchmark/baselines/time_series_library/adapters_for_transformers.py',
        'ts_benchmark/baselines/time_series_library/models/Nonstationary_Transformer.py',
    }


def test_kernel_wrapper_forwards_input_and_stable_validation_only_as_intended(tmp_path):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    for name in ('run_temporal_experiment.sh', 'run_composite_trading_experiment.sh', 'b3_experiment_env.sh'):
        shutil.copy(ROOT / 'scripts' / name, scripts / name)
    (scripts / 'b3_experiment.py').write_text("import os\nprint(os.environ['B3_INLINE_CONFIG'])\n")
    wrapper = scripts / 'wrapper.sh'
    wrapper.write_text((ROOT / 'scripts/run_ns_kernels_isolado.sh').read_text().replace('main "$@"', ''))
    env = dict(os.environ, TFB_ROOT=str(tmp_path), NS_PYTHON_BIN=sys.executable,
               NS_VENV_PATH='/nonexistent/venv', MODEL='Nonstationary_Transformer',
               MOMENTUM_MODE='off', MOMENTUM_INPUT_MODE='feature', MOMENTUM_INPUT_LOOKBACK='512',
               NS_VALIDATION_MODE='ordered_sample_mean')
    result = subprocess.run(['bash', '-c',
        'source "$1"; grid; train_launcher 0 --print-config; train_launcher 1 --print-config',
        'wrapper', str(wrapper)], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    temporal, compound = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    assert not any(key.startswith('momentum_') for key in temporal['loss'])
    assert compound['loss']['momentum_input_mode'] == 'feature'
    assert compound['loss']['momentum_input_lookback'] == 512
    assert temporal['training']['validation_mode'] == compound['training']['validation_mode'] == 'ordered_sample_mean'


def test_fingerprints_distinguish_input_ablations_but_preserve_off(tmp_path):
    wrapper = tmp_path / 'wrapper.sh'
    wrapper.write_text((ROOT / 'scripts/run_ns_kernels_isolado.sh').read_text().replace('main "$@"', ''))
    body = '''source "$1"; grid
branch_fingerprint 0; branch_fingerprint 1
MOMENTUM_INPUT_MODE=feature; MOMENTUM_INPUT_LOOKBACK=512
branch_fingerprint 0; branch_fingerprint 1
MOMENTUM_INPUT_MODE=zero; branch_fingerprint 1
MOMENTUM_INPUT_LOOKBACK=256; branch_fingerprint 1
MOMENTUM_INPUT_MODE=off; branch_fingerprint 1
NS_VALIDATION_MODE=ordered_sample_mean; branch_fingerprint 1
'''
    env = dict(os.environ, MOMENTUM_MODE='off', MOMENTUM_INPUT_MODE='off', NS_VALIDATION_MODE='')
    result = subprocess.run(['bash', '-c', body, 'wrapper', str(wrapper)],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    control, plain, control_input, feature, zero512, zero256, off_again, stable = result.stdout.splitlines()
    assert control == control_input
    assert plain == off_again
    assert len({plain, feature, zero512, zero256, stable}) == 5


@pytest.mark.parametrize('option,value', [('target_mode', 'cumulative'), ('task_name', 'classification')])
def test_non_path_or_non_forecasting_models_fail_preflight(config, monkeypatch, option, value):
    config['loss']['momentum_input_mode'] = 'feature'
    config['model_defaults']['Nonstationary_Transformer'][option] = value
    with pytest.raises(ValueError, match='target_mode|forecast'):
        load_inline(config, monkeypatch)
