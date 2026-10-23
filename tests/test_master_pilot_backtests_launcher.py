"""The login host only submits; Slurm runs an immutable source revision on CPU."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'scripts/run_master_pilot_backtests.sh'


def capture(tmp_path, **overrides):
    binary = tmp_path / 'bin'
    binary.mkdir()
    batch = binary / 'sbatch'
    batch.write_text('''#!/usr/bin/env bash
set -eu
printf '%s\\n' "$@" > "$CAPTURE/args"
for key in TFB_ROOT MASTER_EVAL_COMMIT MASTER_EVAL_PREDICTIONS MASTER_EVAL_MARKET MASTER_EVAL_RETURNS MASTER_EVAL_RUN_DIR MASTER_EVAL_WORKERS PYTHON_BIN; do
  printf '%s=%s\\n' "$key" "${!key}" >> "$CAPTURE/env"
done
printf '81234;cluster\\n'
''')
    batch.chmod(0o755)
    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'],
               CAPTURE=str(tmp_path), TFB_ROOT=str(ROOT),
               PYTHON_BIN='/must/not/run/python',
               MASTER_EVAL_LOG_DIR=str(tmp_path / 'logs'),
               # Settings from the training/recovery launchers cannot redirect this job.
               PARTITION='medusas_dev', NODELIST='medusa6', GRES='gpu:1',
               PREDICTIONS_ROOT='/stale/predictions', OUT_ROOT='/stale/output',
               RUN_TAG='stale', WORKERS='16', TRAIN_PERCENTAGES='70 90',
               HORIZONS='25', SLIPPAGE_BPS='50', COST_BPS='20')
    for key in ('MASTER_EVAL_PARTITION', 'MASTER_EVAL_WORKERS', 'MASTER_EVAL_MEMORY',
                'MASTER_EVAL_TIME', 'MASTER_EVAL_COMMIT', 'SLURM_JOB_ID'):
        env.pop(key, None)
    env.update(overrides)
    result = subprocess.run(['bash', str(LAUNCHER)], env=env, text=True, capture_output=True)
    return result, tmp_path


def test_login_only_submits_cpu_job_with_fixed_pilot_defaults(tmp_path):
    result, directory = capture(tmp_path)
    assert result.returncode == 0, result.stderr
    args = (directory / 'args').read_text().splitlines()
    env = dict(line.split('=', 1) for line in (directory / 'env').read_text().splitlines())
    assert '--partition=gorgonas_dev' in args
    assert '--cpus-per-task=2' in args and '--mem=16G' in args
    assert '--time=01:30:00' in args
    assert args[-2:] == [str(LAUNCHER), '--on-node']
    assert not any(a.startswith(('--gres=', '--nodelist=', '--array=')) for a in args)
    assert env['PYTHON_BIN'] == '/must/not/run/python'
    assert env['MASTER_EVAL_WORKERS'] == '2'
    assert env['MASTER_EVAL_PREDICTIONS'].endswith('master_piloto_l32_h5_k1_t80_v1')
    assert env['MASTER_EVAL_RUN_DIR'].endswith('master_piloto_l32_h5_k1_t80_s0_c0_cdi_v1')
    assert env['MASTER_EVAL_COMMIT'] == subprocess.check_output(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    assert 'avaliacao_81234.{out,err}' in result.stdout


def test_named_resource_overrides_reach_sbatch(tmp_path):
    result, directory = capture(tmp_path, MASTER_EVAL_PARTITION='gorgonas',
                                MASTER_EVAL_WORKERS='4', MASTER_EVAL_TIME='08:00:00',
                                MASTER_EVAL_MEMORY='24G')
    assert result.returncode == 0
    args = (directory / 'args').read_text().splitlines()
    for option in ('--partition=gorgonas', '--cpus-per-task=4', '--time=08:00:00', '--mem=24G'):
        assert option in args


@pytest.mark.parametrize('workers', ['0', '17', '2;echo bad'])
def test_bad_resources_never_submit(tmp_path, workers):
    result, directory = capture(tmp_path, MASTER_EVAL_WORKERS=workers)
    assert result.returncode == 2
    assert not (directory / 'args').exists()


@pytest.mark.parametrize('arguments', [['smoke'], ['--on-node', 'extra']])
def test_invalid_action_never_submits(arguments):
    result = subprocess.run(['bash', str(LAUNCHER), *arguments], text=True, capture_output=True)
    assert result.returncode == 2
    assert 'Uso:' in result.stderr


def test_on_node_requires_slurm_allocation(tmp_path):
    env = dict(os.environ, TFB_ROOT=str(ROOT), MASTER_EVAL_LOG_DIR=str(tmp_path),
               PYTHON_BIN='/must/not/run/python')
    env.pop('SLURM_JOB_ID', None)
    result = subprocess.run(['bash', str(LAUNCHER), '--on-node'], env=env,
                            text=True, capture_output=True)
    assert result.returncode == 2
    assert 'no de computacao' in result.stderr


def test_slurm_spool_uses_submitted_revision_after_checkout_changes(tmp_path):
    """Run the copied shell as Slurm does; later source changes cannot leak in."""
    import sys

    repo, binary, source = tmp_path / 'repo', tmp_path / 'bin', tmp_path / 'source'
    repo.mkdir()
    binary.mkdir()
    source.mkdir()
    for directory in ('scripts', 'ts_benchmark', 'config'):
        (repo / directory).mkdir()
        (repo / directory / 'marker').write_text('tracked')
    evaluator = repo / 'scripts/evaluate_master_pilot.py'
    evaluator.write_text('print("FROZEN_VERSION")\n')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Fixture', '-c',
                    'user.email=fixture@example.invalid', 'commit', '-qm', 'frozen'], check=True)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    evaluator.write_text('raise RuntimeError("LIVE_CHECKOUT_WAS_USED")\n')
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Fixture', '-c',
                    'user.email=fixture@example.invalid', 'commit', '-qm', 'later'], check=True)
    # Redirect only shell scratch creation to the fixture; archive/extraction/Python are real.
    for command, text in {
        'mkdir': '#!/usr/bin/env bash\nexit 0\n',
        'mktemp': '#!/usr/bin/env bash\nprintf "%s\\n" "$FIXTURE_SOURCE"\n',
        'hostname': '#!/usr/bin/env bash\nprintf "gorgona3\\n"\n',
    }.items():
        path = binary / command
        path.write_text(text)
        path.chmod(0o755)
    spool = tmp_path / 'slurm_script'
    spool.write_text(LAUNCHER.read_text())
    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'],
               TFB_ROOT=str(repo), PYTHON_BIN=sys.executable, SLURM_JOB_ID='fixture',
               MASTER_EVAL_COMMIT=commit, MASTER_EVAL_WORKERS='2',
               MASTER_EVAL_RUN_DIR='/snfs2/fixture-master-evaluation', FIXTURE_SOURCE=str(source))
    result = subprocess.run(['bash', str(spool), '--on-node'], env=env,
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'FROZEN_VERSION' in result.stdout
    assert 'LIVE_CHECKOUT_WAS_USED' not in result.stdout
