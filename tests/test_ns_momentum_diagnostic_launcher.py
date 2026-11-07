"""Three-condition diagnosis: real configuration wiring without cluster submission."""
import json
import os
from pathlib import Path
import subprocess

import test_ns_kernel_launcher as fixtures

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/run_ns_momentum_diagnostico.sh'


def run(*arguments, env=None):
    inherited = {key: value for key, value in os.environ.items()
                 if not key.startswith(('DIAG_', 'NS_', 'MOMENTUM_', 'B3_')) and key not in {
                     'TFB_ROOT', 'EXPERIMENT_TAG', 'MODEL', 'TRAIN_VALUES',
                     'LOOKBACKS_VALUES', 'HORIZON_VALUES', 'K_LOSS_VALUES',
                     'POSITIVE_LAMBDAS', 'CROSS_LOSSES_VALUES', 'K_EVAL_VALUES',
                     'DELTAS_VALUES', 'KERNEL_SPECS'}}
    inherited.update(env or {})
    return subprocess.run(['bash', str(SCRIPT), *arguments], env=inherited,
                          text=True, capture_output=True, timeout=30)


def fixture(tmp_path, real_launchers=False):
    return fixtures.KernelLauncherTests().fixture(tmp_path, real_launchers=real_launchers)


def records(root):
    return [json.loads(line) for line in (root / 'capture.jsonl').read_text().splitlines()]


def test_default_plan_is_exactly_three_without_extra_controls():
    result = run('--plan')
    assert result.returncode == 0, result.stderr
    assert 'DIAGNOSTICO TOTAL: 3 treinos; 3 condicoes.' in result.stdout
    assert result.stdout.count('TOTAL: 1 treinos unicos; 1 ramos selecionados.') == 3
    assert 'ganho inicial=0; lradj=type1' in result.stdout
    assert 'ganho inicial=0.1; lradj=type1' in result.stdout
    assert 'ganho inicial=0.1; lradj=constant' in result.stdout
    assert 'cauchy 0.025 natural' in result.stdout
    assert '/controle' in result.stdout and '/ganho_inicial' in result.stdout
    assert 'sem_kernel ' not in result.stdout
    help_result = run('--help')
    assert help_result.returncode == 0
    assert 'TOTAL DEFAULT: 3 treinos' in help_result.stdout
    assert 'NAO fixa o no/GPU' in help_result.stdout


def test_previous_scientific_exports_cannot_expand_or_disable_diagnosis():
    result = run('--plan', env=dict(MODEL='DUET TimesNet', LOOKBACKS_VALUES='32 256 512',
        HORIZON_VALUES='1 5 25', POSITIVE_LAMBDAS='0.5 0.8 1', K_LOSS_VALUES='1 5 25',
        CROSS_LOSSES_VALUES='hinge hinge_proximity', KERNEL_SPECS='cauchy;cosine',
        NS_ONLY_BRANCHES='temporal sem_kernel', NS_LRADJ='constant',
        MOMENTUM_MODE='off', MOMENTUM_LOOKBACK='32', MOMENTUM_GAIN_INIT='2'))
    assert result.returncode == 0, result.stderr
    assert 'DIAGNOSTICO TOTAL: 3 treinos; 3 condicoes.' in result.stdout
    assert 'LB: 256; H: 25; K_loss: 1; lambdas compostos: 1' in result.stdout
    assert 'LB=512; eta=0.01' in result.stdout
    assert 'Modelos: Nonstationary_Transformer' in result.stdout


def test_real_configs_have_only_requested_interventions(tmp_path):
    root, _, env = fixture(tmp_path, real_launchers=True)
    env.update(NS_GRES='', NS_NODELIST='gorgona6', NS_PARTITION='gorgonas',
               NS_SEED='2027', EXPERIMENT_TAG='momentum/diagnostic_test')
    result = run('--dry-run', env=env)
    assert result.returncode == 0, result.stderr
    configs = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    assert len(configs) == 3
    for config in configs:
        assert config['cross_losses'] == ['ranknet_proximity']
        assert config['lookbacks'] == [256]
        assert config['horizons'] == [25]
        assert config['lambdas'] == [1.]
        assert config['loss']['loss_cross_mask'] == ['cauchy', .025, 'natural']
        assert config['loss']['momentum_mode'] == 'residual_output'
        assert config['loss']['momentum_lookback'] == 512
        assert config['loss']['momentum_eta'] == .01
        assert config['loss']['momentum_diagnostics'] is True
    assert [x['loss'].get('momentum_gain_init', 0) for x in configs] == [0, .1, .1]
    # lradj is a training hyperparameter, shared by every architecture in a condition.
    assert [x['training']['lradj'] for x in configs] == ['type1', 'type1', 'constant']
    assert not (root / 'store').exists()


def test_scientific_overrides_keep_each_condition_comparable():
    result = run('--plan', env=dict(DIAG_MODEL='DUET', DIAG_HORIZON_VALUES='1 25',
        DIAG_LAMBDAS='0.5 1', DIAG_CROSS_LOSS='hinge_proximity',
        DIAG_KERNEL_SPEC='student 10', DIAG_MASK_A='0.10', DIAG_GAIN_INIT='0.2'))
    assert result.returncode == 0, result.stderr
    assert 'DIAGNOSTICO TOTAL: 12 treinos; 3 condicoes.' in result.stdout
    assert result.stdout.count('student 0.10 natural 10') == 3
    assert 'ganho inicial=0.2; lradj=type1' in result.stdout
    assert 'Modelos: DUET' in result.stdout


def test_submissions_preflight_all_chain_and_do_not_duplicate(tmp_path):
    root, _, env = fixture(tmp_path)
    env.update(NS_GRES='', EXPERIMENT_TAG='momentum/diagnostic_test')
    first = run('--train', env=env)
    assert first.returncode == 0, first.stderr
    rows = records(root)
    submitted = [x for x in rows if x['action'] == 'submit']
    assert len(submitted) == 3
    assert [x['B3_AFTER_JOB'] for x in submitted] == ['', '1001', '1002']
    assert all(x['GRES'] == '' and x['CROSS_LOSSES'] == 'ranknet_proximity' for x in rows)
    assert [x['RUN_TAG'].split('/')[-3] for x in submitted] == [
        'controle', 'ganho_inicial', 'ganho_lr_constante']
    assert all(x['action'] == '--dry-run' for x in rows[:3])
    second = run('--train', env=env)
    assert second.returncode == 0, second.stderr
    assert sum(x['action'] == 'submit' for x in records(root)) == 3


def test_failed_preflight_never_submits_any_condition(tmp_path):
    root, _, env = fixture(tmp_path)
    mock = root / 'mock.py'
    mock.write_text(mock.read_text() + "\nif sys.argv[1]=='--dry-run' and '/ganho_lr_constante/' in os.environ['RUN_TAG']: sys.exit(23)\n")
    result = run('--train', env=env)
    assert result.returncode != 0
    assert all(x['action'] == '--dry-run' for x in records(root))
    assert not (root / 'store').exists()


def test_parallel_conditions_share_only_requested_initial_dependency(tmp_path):
    root, _, env = fixture(tmp_path)
    env.update(DIAG_PARALLEL='1', NS_AFTER_JOB='999')
    result = run('--train', env=env)
    assert result.returncode == 0, result.stderr
    submitted = [x for x in records(root) if x['action'] == 'submit']
    assert len(submitted) == 3
    assert [x['B3_AFTER_JOB'] for x in submitted] == ['999'] * 3


def test_condition_selection_and_invalid_controls():
    selected = run('--plan', env=dict(DIAG_CONDITIONS='ganho_inicial'))
    assert selected.returncode == 0, selected.stderr
    assert 'DIAGNOSTICO TOTAL: 1 treinos; 1 condicoes.' in selected.stdout
    for overrides in ({'DIAG_CONDITIONS': 'controle controle'},
                      {'DIAG_CONDITIONS': 'unknown'}, {'DIAG_PARALLEL': '2'},
                      {'DIAG_CROSS_LOSS': 'ranknet'}, {'DIAG_KERNEL_SPEC': 'cauchy;cosine'}):
        result = run('--plan', env=overrides)
        assert result.returncode != 0
