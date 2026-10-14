"""Editable shell GPU requests reach sbatch without submitting or training jobs."""
from __future__ import annotations

from collections import Counter
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import b3_experiment as experiment

MODES = tuple(experiment.LAUNCHERS)
GRES_VALUES = (None, '', 'gpu:1', 'gpu:a100:1', 'gpu:2')


def shell_env(tmp_path, gres=None):
    # Do not inherit a user's scheduler/config overrides or activate a real venv.
    env = dict(PATH=os.environ['PATH'], TFB_ROOT=str(REPO),
               VENV_PATH=str(tmp_path / 'missing-venv'), PYTHON_BIN=sys.executable,
               PYTHONDONTWRITEBYTECODE='1', OUT_ROOT=str(tmp_path / 'predictions'),
               RESULT_ROOT=str(tmp_path / 'results'), LOG_ROOT=str(tmp_path / 'logs with spaces'),
               DATASET_PATH=str(tmp_path / 'missing-dataset.csv'),
               PARTITION='medusas_shr', TIME_LIMIT='12:34:56', NODELIST='medusa3',
               CPUS_PER_TASK='4', MEMORY='16G', MAX_CONCURRENT_JOBS='2', ARRAY_CHUNK_SIZE='1000',
               CUDA_VISIBLE_DEVICES='GPU-slurm-allocation')
    if gres is not None:
        env['GRES'] = gres
    return env


def shell_run(mode, env, action='--dry-run', launcher=None):
    launcher = launcher or REPO / 'scripts' / experiment.LAUNCHERS[mode]
    return subprocess.run(['bash', str(launcher), action], env=env, check=True,
                          capture_output=True, text=True)


def preview(mode, env, launcher=None):
    result = shell_run(mode, env, launcher=launcher)
    line = next(line for line in result.stdout.splitlines() if line.startswith('sbatch --'))
    return shlex.split(line)


def assert_gres(command, mode, gres):
    expected = ('' if mode == 'var' else 'gpu:1') if gres is None else gres
    assert [arg for arg in command if arg.startswith('--gres=')] == (
        [f'--gres={expected}'] if expected else [])
    assert not any(arg.startswith('--gpus') or 'CUDA_VISIBLE_DEVICES' in arg for arg in command)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('gres', GRES_VALUES)
def test_shell_resource_overrides_reach_dry_run_without_writes(tmp_path, mode, gres):
    env = shell_env(tmp_path, gres)
    command = preview(mode, env)
    assert_gres(command, mode, gres)
    assert command[command.index('-p') + 1] == env['PARTITION']
    for option, value in (('time', env['TIME_LIMIT']), ('cpus-per-task', '4'),
                          ('nodelist', 'medusa3'), ('mem', '16G'), ('job-name', f'b3_{mode}'),
                          ('output', f"{env['LOG_ROOT']}/{mode}_%A_%a.out"),
                          ('error', f"{env['LOG_ROOT']}/{mode}_%A_%a.err")):
        assert f'--{option}={value}' in command
    assert next(arg for arg in command if arg.startswith('--array=')).endswith('%2')
    assert command[-3:] == ['worker', '--offset', '0']
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('gres', ('', 'gpu:tesla:2'))
def test_gres_can_be_edited_in_script_without_export_or_environment(tmp_path, mode, gres):
    original = REPO / 'scripts' / experiment.LAUNCHERS[mode]
    text, count = re.subn(r'^GRES=.*$', f'GRES="{gres}"', original.read_text(), flags=re.MULTILINE)
    assert count == 1
    edited = tmp_path / original.name
    edited.write_text(text)
    env = shell_env(tmp_path)
    assert 'GRES' not in env
    assert_gres(preview(mode, env, edited), mode, gres)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('gres', (None, '', 'gpu:a100:2'))
def test_submit_uses_same_builder_and_preserves_arrays_logs_and_cuda(tmp_path, mode, gres):
    env = shell_env(tmp_path, gres)
    env.update(GPU_PARTITION=env['PARTITION'], GPU_TIME=env['TIME_LIMIT'], MAX_GPU_JOBS='2')
    calls = []

    def fake_run(command, **kwargs):
        if command[0] == 'scontrol':
            return SimpleNamespace(stdout='MaxArraySize = 2\n')
        assert command[0] == 'sbatch'
        assert Path(env['LOG_ROOT']).is_dir()  # Must exist before submitting.
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == env['CUDA_VISIBLE_DEVICES']
        assert kwargs['env'].get('GRES') == env.get('GRES')
        assert kwargs['check'] is True
        calls.append(command)
        return SimpleNamespace(stdout=f'{900 + len(calls)};test-cluster\n')

    with patch.dict(os.environ, env, clear=True), \
         patch.object(experiment.shutil, 'which', side_effect=lambda name: f'/mock/{name}'), \
         patch.object(experiment.subprocess, 'run', side_effect=fake_run), \
         redirect_stdout(io.StringIO()):
        experiment.submit(mode, [None] * 5, tmp_path / 'predictions', tmp_path / 'config.sh')
        assert len(calls) == 3
        for command, offset, count, previous in zip(calls, (0, 2, 4), (2, 2, 1), (None, '901', '902')):
            assert command == experiment.sbatch_command(mode, offset, count, previous)
            assert_gres(command, mode, gres)
            assert f'--array=0-{count-1}%2' in command
            assert command[-2:] == ['--offset', str(offset)]
            dependencies = [arg for arg in command if arg.startswith('--dependency=')]
            assert dependencies == ([f'--dependency=afterany:{previous}'] if previous else [])


@pytest.mark.parametrize('mode', MODES)
def test_direct_python_call_keeps_legacy_resource_default(mode):
    with patch.dict(os.environ, {}, clear=True):
        assert_gres(experiment.sbatch_command(mode, 0, 1), mode, None)


@pytest.mark.parametrize('value', ('0', '-1', 'invalid'))
def test_invalid_concurrency_fails_before_submission(value):
    with patch.dict(os.environ, {'MAX_GPU_JOBS': value}, clear=True):
        with pytest.raises(ValueError):
            experiment.sbatch_command('temporal', 0, 1)


@pytest.mark.parametrize('mode,count', (('temporal', 324), ('composto', 33696), ('var', 3)))
def test_resource_changes_do_not_change_training_grid(tmp_path, mode, count):
    before = json.loads(shell_run(mode, shell_env(tmp_path), '--print-config').stdout)
    after = json.loads(shell_run(mode, shell_env(tmp_path, 'gpu:a100:2'), '--print-config').stdout)
    assert before == after
    tasks = experiment.make_tasks(after, mode)
    assert len(tasks) == len({experiment.output_path(task) for task in tasks}) == count
    assert [task['array_index'] for task in tasks] == list(range(count))
    if mode == 'composto':
        assert all(task['pred_len'] % task['k'] == 0 for task in tasks)
        assert Counter(task['loss'] for task in tasks) == {
            'ranknet': 10368, 'hinge': 10368, 'pairwise_mse': 10368, 'listnet': 2592}
        assert all(task['cross_delta'] == 0 for task in tasks if task['loss'] == 'listnet')


@pytest.mark.parametrize('name', (*experiment.LAUNCHERS.values(), 'b3_experiment_env.sh'))
def test_shell_syntax(name):
    subprocess.run(['bash', '-n', str(REPO / 'scripts' / name)], check=True)
