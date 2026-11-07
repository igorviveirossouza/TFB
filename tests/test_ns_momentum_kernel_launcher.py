"""Exercise the dedicated launcher without submitting jobs or requiring market data."""
import json
import os
from pathlib import Path
import subprocess
import sys

import test_ns_kernel_launcher as fixtures

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/run_ns_momentum_kernels_isolado.sh'


def run(*arguments, env=None):
    inherited = {key: value for key, value in os.environ.items()
                 if not key.startswith(('NS_', 'MOMENTUM_', 'B3_')) and key not in {
                     'TFB_ROOT', 'EXPERIMENT_TAG', 'MODEL', 'TRAIN_VALUES',
                     'LOOKBACKS_VALUES', 'HORIZON_VALUES', 'K_LOSS_VALUES',
                     'POSITIVE_LAMBDAS', 'CROSS_LOSSES_VALUES', 'K_EVAL_VALUES',
                     'DELTAS_VALUES', 'KERNEL_SPECS'}}
    inherited.update(env or {})
    return subprocess.run(['bash', str(SCRIPT), *arguments], env=inherited,
                          text=True, capture_output=True, timeout=30)


def test_default_plan_counts_shared_controls_and_momentum():
    result = run('--plan')
    assert result.returncode == 0, result.stderr
    assert 'TOTAL: 160 treinos unicos; 14 ramos selecionados.' in result.stdout
    assert 'Momentum residual nos compostos: LB=LB do modelo; eta=0.01' in result.stdout
    assert 'cauchy 0.25 natural' in result.stdout
    assert 'student 0.25 natural 10' in result.stdout
    assert 'intermediate_band 0.25 natural 2' in result.stdout
    assert 'isolados/ns_momentum_kernels_a0.25_t80_v1' in result.stdout
    help_result = run('--help')
    assert help_result.returncode == 0, help_result.stderr
    assert SCRIPT.name in help_result.stdout
    assert 'Lambda0 continua temporal puro' in help_result.stdout


def test_exports_override_defaults_and_empty_gpu_is_preserved(tmp_path):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    capture = scripts / 'run_ns_kernels_isolado.sh'
    capture.write_text('exec "$CAPTURE_PYTHON" -c '\
        + "'import json,os,sys; print(json.dumps(dict(env=dict(os.environ),args=sys.argv[1:])))' \"$@\"\n")
    overrides = dict(TFB_ROOT=str(tmp_path), CAPTURE_PYTHON=sys.executable,
        EXPERIMENT_TAG='isolados/test_mom', MODEL='DUET TimesNet', TRAIN_VALUES='80 90',
        LOOKBACKS_VALUES='32 64', HORIZON_VALUES='1 5', K_LOSS_VALUES='1',
        POSITIVE_LAMBDAS='0.5 1', CROSS_LOSSES_VALUES='hinge hinge_proximity',
        KERNEL_SPECS='student 5;cosine', NS_MASK_A='0.10', NS_MASK_MODE='complementary',
        MOMENTUM_MODE='residual_output', MOMENTUM_LOOKBACK='128', MOMENTUM_ETA='0',
        MOMENTUM_GAIN_INIT='0.1', MOMENTUM_DIAGNOSTICS='1', NS_LRADJ='constant',
        NS_PARTITION='gorgonas_dev', NS_GRES='', NS_NODELIST='gorgona6',
        NS_MAX_CONCURRENT_JOBS='3', NS_PARALLEL_BRANCHES='1', NS_AFTER_JOB='123',
        NS_BATCH_SIZE='8', NS_NUM_EPOCHS='20', NS_PATIENCE='4', NS_LEARNING_RATE='0.0002',
        NS_SEED='2027', NS_RANKNET_ALPHA='2', NS_CPU_WORKERS='4', NS_CPU_MEMORY='8G')
    result = run('--backtest', 'planejar', env=overrides)
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record['args'] == ['--backtest', 'planejar']
    for name, value in overrides.items():
        assert record['env'][name] == value, name


def test_actual_dry_run_keeps_temporal_off_and_composite_active(tmp_path):
    helper = fixtures.KernelLauncherTests()
    root, _, env = helper.fixture(tmp_path, real_launchers=True)
    env.update(NS_ONLY_BRANCHES='temporal sem_kernel cauchy',
               MOMENTUM_LOOKBACK='512', MOMENTUM_ETA='0.02')
    result = run('--dry-run', env=env)
    assert result.returncode == 0, result.stderr
    configs = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    assert len(configs) == 3
    assert not any(key.startswith('momentum_') for key in configs[0]['loss'])
    for config in configs[1:]:
        assert config['loss']['momentum_mode'] == 'residual_output'
        assert config['loss']['momentum_lookback'] == 512
        assert config['loss']['momentum_eta'] == .02
        assert config['lambdas'] == [.5, .8, 1.]
        assert config['lookbacks'] == [256, 512]
        assert config['horizons'] == [1, 25]
    assert configs[1]['cross_losses'] == ['ranknet']
    assert configs[2]['cross_losses'] == ['ranknet_proximity']
    assert configs[2]['loss']['loss_cross_mask'] == ['cauchy', .25, 'natural']
    assert 'TOTAL: 28 treinos unicos; 3 ramos selecionados.' in result.stdout
    assert not (root / 'store').exists()  # Dry-run never submits or creates outputs.


def test_smaller_grid_and_disabled_momentum_are_editable():
    result = run('--plan', env=dict(MODEL='DUET', LOOKBACKS_VALUES='32',
        HORIZON_VALUES='1 5', K_LOSS_VALUES='1', POSITIVE_LAMBDAS='0.5 1',
        CROSS_LOSSES_VALUES='hinge hinge_proximity', KERNEL_SPECS='cauchy;cosine',
        MOMENTUM_MODE='off', NS_MAX_CONCURRENT_JOBS='2', NS_PARALLEL_BRANCHES='1'))
    assert result.returncode == 0, result.stderr
    assert 'TOTAL: 14 treinos unicos; 4 ramos selecionados.' in result.stdout
    assert 'Momentum residual nos compostos' not in result.stdout
    assert 'Ramos paralelos: ate 8 tarefas simultaneas.' in result.stdout


def test_actual_diagnostic_overrides_reach_selected_proximity_only(tmp_path):
    helper = fixtures.KernelLauncherTests()
    root, _, env = helper.fixture(tmp_path, real_launchers=True)
    env.update(NS_ONLY_BRANCHES='cauchy', KERNEL_SPECS='cauchy',
               LOOKBACKS_VALUES='256', HORIZON_VALUES='25', K_LOSS_VALUES='1',
               POSITIVE_LAMBDAS='1', MOMENTUM_LOOKBACK='512', MOMENTUM_ETA='.01',
               MOMENTUM_GAIN_INIT='.1', MOMENTUM_DIAGNOSTICS='1', NS_LRADJ='constant')
    result = run('--dry-run', env=env)
    assert result.returncode == 0, result.stderr
    configs = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    assert len(configs) == 1
    config = configs[0]
    assert config['loss']['momentum_gain_init'] == .1
    assert config['loss']['momentum_diagnostics'] is True
    assert config['training']['lradj'] == 'constant'
    assert config['cross_losses'] == ['ranknet_proximity']
    assert config['lambdas'] == [1.]
    assert 'TOTAL: 1 treinos unicos; 1 ramos selecionados.' in result.stdout
    assert not (root / 'store').exists()
