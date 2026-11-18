"""No rank-target work may execute on the submission/login host."""
from pathlib import Path
import os
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'scripts/run_rank_target_experiment.sh'


def executable(path, content):
    path.write_text('#!/bin/bash\nset -euo pipefail\n' + content)
    path.chmod(0o755)
    return str(path)


def records(path):
    return path.read_bytes().decode().rstrip('\0').split('\0')


@pytest.fixture
def dispatch(tmp_path):
    bindir = tmp_path / 'mock bin'
    bindir.mkdir()
    # Do not allow a developer's allocation variables to affect the tests.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('SLURM', 'NS_', 'RANK_', 'ORDINAL_'))
           and key not in {'TFB_ROOT', 'PYTHON_BIN', 'VENV_PATH'}}
    env.update(PATH=str(bindir) + ':' + os.environ.get('PATH', ''),
               TFB_ROOT=str(ROOT), SRUN_LOG=str(tmp_path / 'srun.log'),
               PYTHON_LOG=str(tmp_path / 'python.log'),
               PYTHON_CONTEXT=str(tmp_path / 'python_context.log'),
               VENV_LOG=str(tmp_path / 'venv.log'),
               NS_PARTITION='test_partition', NS_TIME_LIMIT='00:12:34',
               NS_CPUS_PER_TASK='2', NS_MEMORY='7G', NS_NODELIST='compute[2-3]',
               CUSTOM_EXPERIMENT_VALUE='rank -> rank; a value with spaces',
               NS_MODEL_DEFAULTS_JSON='{"TSMixer": {"d_model": 32}}')
    python = executable(bindir / 'mock python', '''
printf '%s\\0' "$@" > "$PYTHON_LOG"
printf '%s\\0' "${SLURM_JOB_ID:-}" "${SLURMD_NODENAME:-}" "$CUSTOM_EXPERIMENT_VALUE" "$NS_MODEL_DEFAULTS_JSON" > "$PYTHON_CONTEXT"
exit "${PYTHON_EXIT:-0}"
''')
    venv = tmp_path / 'venv with spaces.sh'
    venv.write_text('printf \'%s\\n\' "${SLURMD_NODENAME:-LOGIN}" >> "$VENV_LOG"\n')
    env.update(NS_PYTHON_BIN=python, RANK_PLAN_PYTHON=python,
               NS_VENV_PATH=str(venv))
    executable(bindir / 'hostname', 'printf \'%s\\n\' "${ACTUAL_HOST:-test-compute}"\n')
    executable(bindir / 'srun', '''
[[ ! -e "$PYTHON_LOG" && ! -e "$VENV_LOG" ]] || exit 91
printf '%s\\0' "$@" > "$SRUN_LOG"
if [[ "${SRUN_EXECUTE:-0}" != 1 ]]; then exit "${SRUN_EXIT:-0}"; fi
while [[ "$#" -gt 0 && "$1" != bash && "$1" != /bin/bash ]]; do shift; done
[[ "$#" -gt 0 ]] || exit 92
export SLURM_JOB_ID=5678 SLURMD_NODENAME=test-compute
exec "$@"
''')
    return env, tmp_path


def launch(env, *args):
    return subprocess.run(['/bin/bash', str(LAUNCHER), *args], env=env,
                          cwd=ROOT, capture_output=True, text=True)


@pytest.mark.parametrize('args', [(), ('--trainplan',), ('--dry-run',),
    ('--print-config',), ('--submission-status',), ('--prepare',), ('--train',),
    ('--local',), ('--worker', '/snapshot with spaces/design.json', '--branch', '2', '--index', '3')])
def test_every_action_hands_off_before_python_or_environment_activation(dispatch, args):
    env, tmp_path = dispatch
    result = launch(env, *args)
    assert result.returncode == 0, result.stderr
    call = records(tmp_path / 'srun.log')
    step = call.index('--compute-step')
    assert call[step + 1:] == list(args)
    assert '--export=ALL' in call and '--ntasks=1' in call
    assert not (tmp_path / 'python.log').exists()
    assert not (tmp_path / 'venv.log').exists()


@pytest.mark.parametrize('action', ['--trainplan', '--prepare', '--train'])
def test_coordinator_cpu_resources_are_passed_to_slurm(dispatch, action):
    env, tmp_path = dispatch
    env['NS_GRES'] = 'gpu:2'
    assert launch(env, action).returncode == 0
    call = records(tmp_path / 'srun.log')
    for value in ['--partition=test_partition', '--time=00:12:34',
                  '--cpus-per-task=2', '--mem=7G', '--nodelist=compute[2-3]', '--gres=none']:
        assert value in call
    assert '--gres=gpu:2' not in call


@pytest.mark.parametrize('gres', ['gpu:2', ''])
@pytest.mark.parametrize('action', ['--local', '--worker=/snapshot with spaces/design.json'])
def test_training_requests_configured_training_resources(dispatch, gres, action):
    env, tmp_path = dispatch
    env['NS_GRES'] = gres
    assert launch(env, action).returncode == 0
    call = records(tmp_path / 'srun.log')
    assert ('--gres=' + (gres or 'none')) in call


def test_salloc_job_id_on_login_does_not_authorize_local_execution(dispatch):
    env, tmp_path = dispatch
    env['SLURM_JOB_ID'] = '1234'
    result = launch(env, '--train')
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'srun.log').exists()
    call = records(tmp_path / 'srun.log')
    assert '--jobid=1234' in call
    assert not any(value.startswith('--partition=') for value in call)
    assert not (tmp_path / 'python.log').exists()


@pytest.mark.parametrize('internal_flag', ['--compute-step', '--on-node'])
def test_internal_flags_cannot_bypass_the_node_guard(dispatch, internal_flag):
    env, tmp_path = dispatch
    result = launch(env, internal_flag, '--train')
    assert result.returncode != 0
    assert not (tmp_path / 'python.log').exists()
    assert not (tmp_path / 'venv.log').exists()
    assert not (tmp_path / 'srun.log').exists()


@pytest.mark.parametrize('internal_flag', ['--compute-step', '--on-node'])
def test_actual_login_host_is_rejected_even_with_slurm_variables(dispatch, internal_flag):
    env, tmp_path = dispatch
    env.update(SLURM_JOB_ID='1234', SLURMD_NODENAME='test-compute', ACTUAL_HOST='phocus4')
    result = launch(env, internal_flag, '--train')
    assert result.returncode != 0
    assert not (tmp_path / 'python.log').exists()
    assert not (tmp_path / 'venv.log').exists()
    assert not (tmp_path / 'srun.log').exists()


def test_failed_allocation_has_no_local_fallback_and_preserves_exit_code(dispatch):
    env, tmp_path = dispatch
    env['SRUN_EXIT'] = '37'
    result = launch(env, '--train')
    assert result.returncode == 37
    assert not (tmp_path / 'python.log').exists()
    assert not (tmp_path / 'venv.log').exists()


def test_missing_slurm_has_no_local_fallback(dispatch):
    env, tmp_path = dispatch
    bindir = Path(env['PATH'].split(':')[0])
    (bindir / 'srun').unlink()
    env['PATH'] = str(bindir)
    result = launch(env, '--train')
    assert result.returncode != 0
    assert 'srun' in result.stderr.lower()
    assert not (tmp_path / 'python.log').exists()
    assert not (tmp_path / 'venv.log').exists()


def test_handoff_preserves_exports_arguments_and_activates_environment_on_node(dispatch):
    env, tmp_path = dispatch
    env['SRUN_EXECUTE'] = '1'
    args = ['--worker', '/snapshot with spaces/design.json', '--branch', '2', '--index', '3']
    result = launch(env, *args)
    assert result.returncode == 0, result.stderr
    assert records(tmp_path / 'python.log') == [str(ROOT / 'scripts/rank_target_experiment.py'), *args]
    assert records(tmp_path / 'python_context.log') == ['5678', 'test-compute',
        env['CUSTOM_EXPERIMENT_VALUE'], env['NS_MODEL_DEFAULTS_JSON']]
    assert (tmp_path / 'venv.log').read_text().splitlines() == ['test-compute']


def test_exported_login_hostname_does_not_block_compute_step(dispatch):
    env, tmp_path = dispatch
    env.update(SRUN_EXECUTE='1', HOSTNAME='phocus4')
    result = launch(env, '--train')
    assert result.returncode == 0, result.stderr
    assert records(tmp_path / 'python_context.log')[:2] == ['5678', 'test-compute']


def test_existing_compute_job_executes_local_mode_without_nested_allocation(dispatch):
    env, tmp_path = dispatch
    env.update(SLURM_JOB_ID='1234', SLURMD_NODENAME='test-compute', PYTHON_EXIT='23')
    result = launch(env, '--local')
    assert result.returncode == 23
    assert not (tmp_path / 'srun.log').exists()
    assert records(tmp_path / 'python.log')[-1] == '--local'
    assert (tmp_path / 'venv.log').read_text().splitlines() == ['test-compute']


def test_frozen_worker_keeps_frozen_python_and_venv(dispatch):
    env, tmp_path = dispatch
    env.update(SLURM_JOB_ID='1234', SLURMD_NODENAME='test-compute',
               PYTHON_BIN=env['NS_PYTHON_BIN'], VENV_PATH=env['NS_VENV_PATH'],
               NS_PYTHON_BIN='/must/not/use/unfrozen/python', NS_VENV_PATH='/must/not/use/unfrozen/venv')
    result = launch(env, '--on-node', '--worker', '/frozen snapshot.json', '--branch', '1', '--index', '2')
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / 'srun.log').exists()
    assert records(tmp_path / 'python.log') == [str(ROOT / 'scripts/rank_target_experiment.py'),
        '--worker', '/frozen snapshot.json', '--branch', '1', '--index', '2']
    assert (tmp_path / 'venv.log').read_text().splitlines() == ['test-compute']


def test_direct_python_cli_refuses_login_before_planning(dispatch):
    env, tmp_path = dispatch
    env['MODEL'] = 'INVALID_MODEL_SHOULD_NOT_BE_RESOLVED'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/rank_target_experiment.py'), '--trainplan'],
                            env=env, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'slurm' in result.stderr.lower()
    assert 'INVALID_MODEL_SHOULD_NOT_BE_RESOLVED' not in result.stderr


def test_python_guard_refuses_phocus4_even_with_inherited_slurm_context(monkeypatch, capsys):
    from scripts import rank_target_experiment as rank
    monkeypatch.setenv('SLURM_JOB_ID', '1234')
    monkeypatch.setenv('SLURMD_NODENAME', 'test-compute')
    monkeypatch.setattr(rank.socket, 'gethostname', lambda: 'phocus4.cluster')
    with pytest.raises(SystemExit) as error:
        rank.main(['--trainplan'])
    assert error.value.code == 2
    assert 'Slurm' in capsys.readouterr().err
