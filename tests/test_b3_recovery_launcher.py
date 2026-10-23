"""Login-host submission stays in shell and freezes recovery tooling."""
import os
from pathlib import Path
import shutil
import stat
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'scripts/recover_composite_experiments.sh'


def launch(tmp_path, arguments=(), extra_environment=None):
    repo = tmp_path / 'repo with spaces'
    scripts = repo / 'scripts'
    scripts.mkdir(parents=True)
    shutil.copy2(LAUNCHER, scripts / LAUNCHER.name)
    for name in ('b3_recovery.py', 'b3_recovery_audit.py', 'b3_recovery_snapshot.py'):
        (scripts / name).write_text('# frozen helper: ' + name + '\n')
    binary = tmp_path / 'bin'
    binary.mkdir()
    sbatch = binary / 'sbatch'
    sbatch.write_text('''#!/usr/bin/env bash
set -eu
printf '%s\\n' "$@" > "$CAPTURE_ROOT/args"
env > "$CAPTURE_ROOT/environment"
printf '999;cluster\\n'
''')
    sbatch.chmod(0o755)
    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'],
               TFB_ROOT=str(repo), CAPTURE_ROOT=str(tmp_path),
               RECOVERY_ROOT=str(tmp_path / 'recovery with spaces'),
               RECOVERY_OUT_BASE=str(tmp_path / 'predictions'),
               RECOVERY_RESULT_BASE=str(tmp_path / 'results'),
               PYTHON_BIN='/python/must/not/run/on/login',
               PARTITION='medusas_dev', NODELIST='medusa6', GRES='gpu:1',
               SBATCH_PARTITION='medusas_dev', SBATCH_NODELIST='medusa6',
               SBATCH_GRES='gpu:1', SBATCH_ARRAY_INX='0-99', SBATCH_MEM_PER_CPU='60G')
    for name in list(env):
        if name.startswith('RECOVERY_WORKER_') or name == 'RECOVERY_MAX_CONCURRENT':
            env.pop(name)
    env.update(extra_environment or {})
    result = subprocess.run(['bash', str(scripts / LAUNCHER.name), *arguments],
                            env=env, text=True, capture_output=True)
    return result, repo


@pytest.mark.parametrize('arguments', [(), ('audit',), ('resume',)])
def test_login_submits_cpu_coordinator_without_python(tmp_path, arguments):
    result, repo = launch(tmp_path, arguments)
    assert result.returncode == 0, result.stderr
    args = (tmp_path / 'args').read_text().splitlines()
    env = dict(line.split('=', 1) for line in
               (tmp_path / 'environment').read_text().splitlines() if '=' in line)
    assert '--partition=gorgonas_dev' in args
    assert '--time=01:30:00' in args
    assert '--no-requeue' in args
    assert '--cpus-per-task=3' in args and '--mem=8G' in args
    assert not any(arg.startswith(('--gres', '--nodelist', '--array')) for arg in args)
    assert not any(name.startswith('SBATCH_') for name in env)
    assert env['RECOVERY_WORKER_PARTITION'] == 'gorgonas'
    assert env['RECOVERY_WORKER_GRES'] == ''
    assert env['RECOVERY_WORKER_NODELIST'] == ''
    assert env['RECOVERY_WORKER_MEMORY'] == ''
    assert env['RECOVERY_MAX_CONCURRENT'] == '2'
    assert env['PYTHON_BIN'] == '/python/must/not/run/on/login'
    state = Path(args[args.index('--state-dir') + 1])
    assert env['RECOVERY_TOOL_DIR'] == str(state / 'tools')
    assert str(state / 'tools' / LAUNCHER.name) in args
    assert str(repo / 'scripts' / LAUNCHER.name) not in args
    assert args[args.index('--on-node') + 1] == (arguments[0] if arguments else 'audit')
    assert args[args.index('--repo') + 1] == str(repo)
    for helper in (state / 'tools').iterdir():
        assert not helper.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    original = repo / 'scripts/b3_recovery.py'
    original.write_text('changed during MASTER development\n')
    assert (state / 'tools/b3_recovery.py').read_text().startswith('# frozen helper:')
    assert '999;cluster' in result.stdout
    assert 'coordenador_999.{out,err}' in result.stdout


def test_explicit_recovery_resources_and_subset_are_preserved(tmp_path):
    result, _ = launch(tmp_path, ('resume', '--run-tag', 'grade_a', '--run-tag', 'grade_b'),
                       {'RECOVERY_WORKER_TIME': '08:00:00',
                        'RECOVERY_WORKER_MEMORY': '16G',
                        'RECOVERY_MAX_CONCURRENT': '1',
                        'RECOVERY_AUDIT_MEMORY': '12G'})
    assert result.returncode == 0, result.stderr
    args = (tmp_path / 'args').read_text().splitlines()
    env = dict(line.split('=', 1) for line in
               (tmp_path / 'environment').read_text().splitlines() if '=' in line)
    assert args[-4:] == ['--run-tag', 'grade_a', '--run-tag', 'grade_b']
    assert '--mem=12G' in args
    assert env['RECOVERY_WORKER_TIME'] == '08:00:00'
    assert env['RECOVERY_WORKER_MEMORY'] == '16G'
    assert env['RECOVERY_MAX_CONCURRENT'] == '1'


@pytest.mark.parametrize('arguments', [('unknown',), ('audit', '--unknown'),
                                        ('audit', '--run-tag'), ('audit', '--run-tag', '')])
def test_invalid_arguments_do_not_submit(tmp_path, arguments):
    result, _ = launch(tmp_path, arguments)
    assert result.returncode == 2
    assert not (tmp_path / 'args').exists()


def test_on_node_requires_slurm_allocation(tmp_path):
    env = dict(os.environ, PYTHON_BIN='/python/must/not/run/on/login')
    env.pop('SLURM_JOB_ID', None)
    result = subprocess.run(['bash', str(LAUNCHER), '--on-node', 'audit'],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert 'alocação Slurm' in result.stderr


def test_slurm_spooled_script_uses_frozen_helper_and_preserves_arguments(tmp_path):
    result, _ = launch(tmp_path, ('audit',))
    assert result.returncode == 0, result.stderr
    args = (tmp_path / 'args').read_text().splitlines()
    state = Path(args[args.index('--state-dir') + 1])
    # Slurm runs its saved script from a spool directory, so BASH_SOURCE no
    # longer locates the staged Python files.
    spool = tmp_path / 'var/spool/slurmd/job999'
    spool.mkdir(parents=True)
    spooled_script = spool / 'slurm_script'
    shutil.copy2(state / 'tools' / LAUNCHER.name, spooled_script)
    interpreter = tmp_path / 'fake_python'
    interpreter.write_text('#!/usr/bin/env bash\n'
                           'test "$PYTHONDONTWRITEBYTECODE" = 1\n'
                           'printf "%s\\n" "$@"\n')
    interpreter.chmod(0o755)
    result = subprocess.run(['bash', str(spooled_script),
                             '--on-node', 'audit', '--repo', '/repo with spaces',
                             '--state-dir', str(state)],
                            env={**os.environ, 'SLURM_JOB_ID': '999',
                                 'RECOVERY_TOOL_DIR': str(state / 'tools'),
                                 'PYTHON_BIN': str(interpreter)},
                            text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == [str(state / 'tools/b3_recovery.py'),
                                          'audit', '--repo', '/repo with spaces',
                                          '--state-dir', str(state)]


@pytest.mark.parametrize('tool_dir', ['', 'relative/path', '/missing/frozen/tools'])
def test_on_node_rejects_missing_frozen_tool_directory(tmp_path, tool_dir):
    result = subprocess.run(['bash', str(LAUNCHER), '--on-node', 'audit'],
                            env={**os.environ, 'SLURM_JOB_ID': '999',
                                 'RECOVERY_TOOL_DIR': tool_dir,
                                 'PYTHON_BIN': '/python/must/not/run'},
                            text=True, capture_output=True)
    assert result.returncode == 2
    assert 'RECOVERY_TOOL_DIR' in result.stderr
