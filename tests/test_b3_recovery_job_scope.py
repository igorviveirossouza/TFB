"""A shared Slurm job name must not block unrelated experiment recovery."""
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import b3_recovery as recovery

REPO = Path('/sonic_home/igor.viveiros/src/TFB')
OUT = Path('/snfs2/igor.viveiros/previsoes')
TAG = 'comp_lb32-128-256-512_gaussian_0.025_complementary_k1_t80_v5'


def details(launcher, log_directory, *, error_directory=None, job='71017_1'):
    error_directory = error_directory or log_directory
    return (
        f'JobId={job} JobName=b3_composto UserId=igor.viveiros(1001) '
        f'JobState=RUNNING Command={REPO}/scripts/{launcher} '
        f'WorkDir={REPO} StdOut={log_directory}/composto_71017_1.out '
        f'StdErr={error_directory}/composto_71017_1.err'
    )


def mock_scheduler(monkeypatch, information, queue='71017_1|b3_composto|RUNNING'):
    calls = []

    def output(command):
        calls.append(command)
        if command[0] == 'squeue':
            return queue
        assert command[0] == 'scontrol'
        assert command[-1] in information
        value = information[command[-1]]
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(recovery, 'command_output', output)
    return calls


def test_master_composto_can_run_while_old_grades_are_recovered(monkeypatch):
    record = details('run_master_experiment.sh', REPO / 'logs/master/master_piloto_l32_h5_k1_t80_v1/composto')
    calls = mock_scheduler(monkeypatch, {'71017_1': record})
    recovery.check_no_original_workers([OUT / TAG])
    assert any(call[0] == 'scontrol' for call in calls)


def test_other_standard_composite_grade_does_not_block_selected_grade(monkeypatch):
    record = details('run_composite_trading_experiment.sh', REPO / 'logs/composto_uniform_a0.1_k1_t80_v1')
    mock_scheduler(monkeypatch, {'71017_1': record})
    recovery.check_no_original_workers([OUT / TAG])


@pytest.mark.parametrize('state', ['RUNNING', 'PENDING', 'COMPLETING'])
def test_selected_grade_is_blocked_in_every_active_state(monkeypatch, state):
    record = details('run_composite_trading_experiment.sh', REPO / 'logs' / TAG)
    mock_scheduler(monkeypatch, {'71017_1': record}, queue=f'71017_1|b3_composto|{state}')
    with pytest.raises(RuntimeError, match='71017_1'):
        recovery.check_no_original_workers([OUT / TAG])


def test_selected_log_tag_overrides_master_launcher_name(monkeypatch):
    record = details('run_master_experiment.sh', REPO / 'logs/master' / TAG / 'composto')
    mock_scheduler(monkeypatch, {'71017_1': record})
    with pytest.raises(RuntimeError, match='71017_1'):
        recovery.check_no_original_workers([OUT / TAG])


def test_one_selected_log_and_one_other_log_cannot_bypass_guard(monkeypatch):
    record = details('run_composite_trading_experiment.sh', REPO / 'logs' / TAG,
                     error_directory=REPO / 'logs/another_grade')
    mock_scheduler(monkeypatch, {'71017_1': record})
    with pytest.raises(RuntimeError, match='71017_1'):
        recovery.check_no_original_workers([OUT / TAG])


@pytest.mark.parametrize('record', [
    'JobId=71017_1 JobName=b3_composto JobState=RUNNING',
    details('unknown_wrapper.sh', REPO / 'logs/another_grade'),
    details('run_composite_trading_experiment.sh', Path('/tmp/custom_logs')),
    details('run_master_experiment.sh', Path('/tmp/custom_logs')),
    details('run_composite_trading_experiment.sh', REPO / 'logs/grade_a',
            error_directory=REPO / 'logs/grade_b'),
])
def test_unidentified_or_conflicting_scope_blocks_instead_of_guessing(monkeypatch, record):
    mock_scheduler(monkeypatch, {'71017_1': record})
    with pytest.raises(RuntimeError, match='71017_1'):
        recovery.check_no_original_workers([OUT / TAG])


def test_exact_tag_matching_does_not_confuse_prefixes(monkeypatch):
    record = details('run_composite_trading_experiment.sh', REPO / 'logs' / (TAG + '_other'))
    mock_scheduler(monkeypatch, {'71017_1': record})
    recovery.check_no_original_workers([OUT / TAG])


def test_empty_queue_needs_no_job_lookup(monkeypatch):
    calls = mock_scheduler(monkeypatch, {}, queue='')
    recovery.check_no_original_workers([OUT / TAG])
    assert all(call[0] != 'scontrol' for call in calls)


def test_non_composite_job_needs_no_lookup(monkeypatch):
    calls = mock_scheduler(monkeypatch, {}, queue='71536|b3_recovery_resume|RUNNING\n123|b3_temporal|RUNNING')
    recovery.check_no_original_workers([OUT / TAG])
    assert all(call[0] != 'scontrol' for call in calls)


def test_unrelated_master_does_not_hide_selected_original_job(monkeypatch):
    master = details('run_master_experiment.sh', REPO / 'logs/master/pilot/composto')
    original = details('run_composite_trading_experiment.sh', REPO / 'logs' / TAG, job='71018_4')
    mock_scheduler(monkeypatch, {'71017_1': master, '71018_4': original},
                   queue='71017_1|b3_composto|RUNNING\n71018_4|b3_composto|PENDING')
    with pytest.raises(RuntimeError, match='71018_4'):
        recovery.check_no_original_workers([OUT / TAG])


def test_failed_slurm_lookup_never_silently_ignores_active_job(monkeypatch):
    failure = subprocess.CalledProcessError(1, ['scontrol', 'show', 'job', '71017_1'])
    mock_scheduler(monkeypatch, {'71017_1': failure})
    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        recovery.check_no_original_workers([OUT / TAG])


def test_job_ending_between_queue_and_lookup_does_not_block(monkeypatch):
    queues = iter(['71017_1|b3_composto|RUNNING', ''])

    def output(command):
        if command[0] == 'squeue':
            return next(queues)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(recovery, 'command_output', output)
    recovery.check_no_original_workers([OUT / TAG])


def test_master_command_arguments_are_not_confused_with_slurm_fields(monkeypatch):
    record = details('run_master_experiment.sh --on-node composto worker --offset 0',
                     REPO / 'logs/master/pilot/composto')
    mock_scheduler(monkeypatch, {'71017_1': record})
    recovery.check_no_original_workers([OUT / TAG])
