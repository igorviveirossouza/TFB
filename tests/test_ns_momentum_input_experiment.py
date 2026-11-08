"""The isolated input experiment submits exactly the paired design requested."""
import json
import os
from pathlib import Path
import subprocess

import pytest
import test_ns_kernel_launcher as fixtures

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/run_ns_momentum_input_isolado.sh'


def run(*arguments, env=None):
    inherited = {k: v for k, v in os.environ.items()
                 if not k.startswith(('INPUT_', 'NS_', 'MOMENTUM_', 'B3_')) and k not in {
                     'TFB_ROOT', 'EXPERIMENT_TAG', 'MODEL', 'TRAIN_VALUES',
                     'LOOKBACKS_VALUES', 'HORIZON_VALUES', 'K_LOSS_VALUES',
                     'POSITIVE_LAMBDAS', 'CROSS_LOSSES_VALUES', 'K_EVAL_VALUES',
                     'DELTAS_VALUES', 'KERNEL_SPECS'}}
    inherited.update(env or {})
    return subprocess.run(['bash', str(SCRIPT), *arguments], env=inherited,
                          text=True, capture_output=True, timeout=40)


def fixture(tmp_path, real=False):
    return fixtures.KernelLauncherTests().fixture(tmp_path, real_launchers=real)


def records(root):
    return [json.loads(line) for line in (root / 'capture.jsonl').read_text().splitlines()]


def test_eight_fit_design_and_no_old_export_contamination():
    result = run('--plan', env=dict(MODEL='DUET', LOOKBACKS_VALUES='32 256 512',
        HORIZON_VALUES='1 25', K_LOSS_VALUES='1', POSITIVE_LAMBDAS='.5 .8 1',
        CROSS_LOSSES_VALUES='hinge hinge_proximity', KERNEL_SPECS='cosine;uniform',
        MOMENTUM_MODE='residual_output', MOMENTUM_ETA='5', MOMENTUM_GAIN_INIT='.1',
        MOMENTUM_INPUT_MODE='off', MOMENTUM_INPUT_LOOKBACK='32', NS_SEED='7',
        NS_ONLY_BRANCHES='temporal', NS_LRADJ='constant', NS_VALIDATION_MODE='legacy'))
    assert result.returncode == 0, result.stderr
    assert 'EXPERIMENTO INPUT TOTAL: 8 treinos; 2 sementes; 2 entradas; 2 losses.' in result.stdout
    assert result.stdout.count('TOTAL: 2 treinos unicos; 2 ramos selecionados.') == 4
    assert result.stdout.count('LB: 512; H: 25; K_loss: 25; lambdas compostos: 1') == 4
    assert result.stdout.count('cauchy 0.025 natural') == 4
    assert 'temporal ' not in result.stdout
    assert 'type1' in result.stdout


def test_actual_inline_configs_match_all_four_pairs(tmp_path):
    root, _, env = fixture(tmp_path, real=True)
    result = run('--dry-run', env=dict(env, NS_GRES='', NS_NODELIST='gorgona6'))
    assert result.returncode == 0, result.stderr
    configs = [json.loads(x) for x in result.stdout.splitlines() if x.startswith('{')]
    assert len(configs) == 8
    seen = []
    for c in configs:
        assert c['models'] == ['Nonstationary_Transformer']
        assert c['lookbacks'] == [512] and c['horizons'] == [25]
        assert c['trade_windows'] == [25] and c['lambdas'] == [1.0]
        assert c['loss']['momentum_input_lookback'] == 512
        assert c['loss'].get('momentum_mode', 'off') == 'off'
        assert 'momentum_eta' not in c['loss']
        assert c['training']['validation_mode'] == 'ordered_sample_mean'
        assert c['training']['lradj'] == 'type1'
        seen.append((c['strategy']['seed'], c['loss']['momentum_input_mode'], c['cross_losses'][0]))
    assert set(seen) == {(s, m, loss) for s in (2026, 2027) for m in ('zero', 'feature')
                        for loss in ('ranknet', 'ranknet_proximity')}
    assert not (root / 'store').exists()


def test_dynamic_count_and_loss_kernel_overrides():
    result = run('--plan', env=dict(INPUT_SEEDS='0 42 99', INPUT_MODES='feature',
        INPUT_LOOKBACKS_VALUES='256 512', INPUT_CROSS_LOSSES_VALUES='hinge_proximity hinge',
        INPUT_KERNEL_SPEC='student 10', INPUT_MASK_A='0.1'))
    assert result.returncode == 0, result.stderr
    assert 'EXPERIMENTO INPUT TOTAL: 12 treinos; 3 sementes; 1 entradas; 2 losses.' in result.stdout
    assert result.stdout.count('student 0.1 natural 10') == 3


def test_preflight_then_eight_chained_submissions_and_idempotent_retry(tmp_path):
    root, _, env = fixture(tmp_path)
    result = run('--train', env=dict(env, NS_GRES=''))
    assert result.returncode == 0, result.stderr
    rows = records(root)
    assert all(x['action'] == '--dry-run' for x in rows[:8])
    submitted = [x for x in rows if x['action'] == 'submit']
    assert len(submitted) == 8
    assert [x['B3_AFTER_JOB'] for x in submitted] == [''] + [str(s) for s in range(1001, 1008)]
    assert all(x['GRES'] == '' for x in submitted)
    assert len({x['RUN_TAG'] for x in submitted}) == 8
    result = run('--train', env=dict(env, NS_GRES=''))
    assert result.returncode == 0, result.stderr
    assert sum(x['action'] == 'submit' for x in records(root)) == 8


def test_parallel_arrays_have_no_intergroup_dependency(tmp_path):
    root, _, env = fixture(tmp_path)
    result = run('--train', env=dict(env, INPUT_PARALLEL='1', NS_AFTER_JOB='999'))
    assert result.returncode == 0, result.stderr
    submitted = [x for x in records(root) if x['action'] == 'submit']
    assert len(submitted) == 8
    assert [x['B3_AFTER_JOB'] for x in submitted] == ['999'] * 8


def test_final_preflight_failure_submits_nothing(tmp_path):
    root, _, env = fixture(tmp_path)
    mock = root / 'mock.py'
    mock.write_text(mock.read_text() + "\nif sys.argv[1]=='--dry-run' and '/seed=2027/entrada=feature/' in os.environ['RUN_TAG']: sys.exit(23)\n")
    result = run('--train', env=env)
    assert result.returncode != 0
    assert all(x['action'] == '--dry-run' for x in records(root))
    assert not (root / 'store').exists()


@pytest.mark.parametrize('overrides', [
    {'INPUT_MODEL': 'DUET'}, {'INPUT_SEEDS': '2026 2026'}, {'INPUT_SEEDS': '-1'},
    {'INPUT_SEEDS': '4294967296'}, {'INPUT_MODES': 'off'}, {'INPUT_MODES': 'zero zero'},
    {'INPUT_CROSS_LOSSES_VALUES': 'ranknet hinge_proximity'},
    {'INPUT_CROSS_LOSSES_VALUES': 'ranknet'}, {'INPUT_KERNEL_SPEC': 'cauchy;cosine'},
    {'INPUT_PARALLEL': '2'}, {'INPUT_MOMENTUM_LOOKBACK': '0'},
])
def test_invalid_design_fails_before_submission(overrides):
    result = run('--plan', env=overrides)
    assert result.returncode != 0
