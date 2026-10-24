"""Resolve only an explicitly identified rejected sbatch; preserve every record.

The fixture mirrors the old report whose CalledProcessError lost stderr. An
unconfirmed submission alone must never be interpreted as a rejected job.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import b3_recovery as recovery
from test_b3_recovery import coordinator_fixture, make_run
from test_b3_recovery_launcher import launch

TOKEN = 'e8b6c31e7a8f4eaab2dec9cdc32b1b33'
OTHER_TOKEN = 'f' * 32


def rejected_fixture(tmp_path, token=TOKEN):
    run, state = make_run(tmp_path)
    root = Path(run['out_root'])
    plan = state / run['run_tag'] / 'execution_plan.json'
    recovery.atomic_json(plan, {'out_root': str(root), 'missing_indices': [55, 59, 1000, 1295]})
    command = ['sbatch', '--parsable', '--partition=medusas', '--gres=gpu:1',
               '--array=0-3%2', '--comment=b3-recovery:' + token,
               str(plan.parent / 'worker.sh'), '--plan', str(plan)]
    record = dict(token=token, indices=[55, 59, 1000, 1295], execution_plan=str(plan),
                  command=command, job_id=None)
    journal = root / '.b3_recovery_submissions.json'
    recovery.atomic_json(journal, [record])
    report = state / 'report.json'
    recovery.atomic_json(report, {
        'status': 'SUBMISSION_INCOMPLETE', 'action': 'resume',
        'error': str(subprocess.CalledProcessError(1, command)),
        'runs': [dict(run_tag=run['run_tag'], out_root=str(root), submission_records=[record])],
    })
    return root, journal, report, record


def test_explicit_rejection_keeps_original_evidence_and_unblocks_verifier(tmp_path, monkeypatch):
    root, journal, report, record = rejected_fixture(tmp_path)
    report_before = report.read_bytes()
    design = {p: p.read_bytes() for p in (root / recovery.GROUP).iterdir()}
    monkeypatch.setattr(recovery, 'command_output', lambda command: '')
    recovery.reconcile_rejected_submissions([root], [TOKEN])
    updated = json.loads(journal.read_text())
    assert len(updated) == 1
    assert all(updated[0][key] == value for key, value in record.items())
    assert updated[0]['rejection_resolution']['kind'] == 'user_confirmed_rejection'
    assert report.read_bytes() == report_before
    assert {p: p.read_bytes() for p in design} == design
    recovery.verify_previous_submissions(root)
    after = journal.read_bytes()
    recovery.reconcile_rejected_submissions([root], [TOKEN])
    assert journal.read_bytes() == after


@pytest.mark.parametrize('mutation', ['no_failure_report', 'wrong_report_status',
                                    'different_token', 'different_command', 'accepted_job'])
def test_ambiguous_or_accepted_intent_cannot_be_declared_rejected(tmp_path, monkeypatch, mutation):
    root, journal, report, record = rejected_fixture(tmp_path)
    saved = json.loads(report.read_text())
    if mutation == 'no_failure_report':
        report.unlink()
    elif mutation == 'wrong_report_status':
        saved['status'] = 'SUBMITTED'
        recovery.atomic_json(report, saved)
    elif mutation == 'different_token':
        saved['runs'][0]['submission_records'][0]['token'] = OTHER_TOKEN
        recovery.atomic_json(report, saved)
    elif mutation == 'different_command':
        saved['runs'][0]['submission_records'][0]['command'][2] = '--partition=gorgonas'
        recovery.atomic_json(report, saved)
    else:
        record['job_id'] = '71538'
        recovery.atomic_json(journal, [record])
    before = journal.read_bytes()
    monkeypatch.setattr(recovery, 'command_output', lambda command: '')
    with pytest.raises((RuntimeError, ValueError, OSError)):
        recovery.reconcile_rejected_submissions([root], [TOKEN])
    assert journal.read_bytes() == before


def test_active_matching_token_blocks_resolution(tmp_path, monkeypatch):
    root, journal, _, _ = rejected_fixture(tmp_path)
    before = journal.read_bytes()
    monkeypatch.setattr(recovery, 'command_output',
                        lambda command: '71538_[0-3]|b3-recovery:' + TOKEN)
    with pytest.raises((RuntimeError, ValueError)):
        recovery.reconcile_rejected_submissions([root], [TOKEN])
    assert journal.read_bytes() == before


def test_unmarked_null_intent_remains_blocked(tmp_path, monkeypatch):
    root, journal, _, _ = rejected_fixture(tmp_path)
    before = journal.read_bytes()
    monkeypatch.setattr(recovery, 'command_output', lambda command: '')
    recovery.reconcile_rejected_submissions([root], [])
    with pytest.raises(RuntimeError, match='confirmação incompleta'):
        recovery.verify_previous_submissions(root)
    assert journal.read_bytes() == before


def test_all_tokens_validated_before_any_journal_is_changed(tmp_path, monkeypatch):
    root_a, journal_a, _, _ = rejected_fixture(tmp_path / 'a')
    root_b, journal_b, report_b, _ = rejected_fixture(tmp_path / 'b', OTHER_TOKEN)
    report_b.unlink()
    before = {journal_a: journal_a.read_bytes(), journal_b: journal_b.read_bytes()}
    monkeypatch.setattr(recovery, 'command_output', lambda command: '')
    with pytest.raises((RuntimeError, ValueError, OSError)):
        recovery.reconcile_rejected_submissions([root_a, root_b], [TOKEN, OTHER_TOKEN])
    assert {p: p.read_bytes() for p in before} == before


def test_duplicate_record_tokens_cannot_resolve_multiple_submissions(tmp_path, monkeypatch):
    root, journal, _, record = rejected_fixture(tmp_path)
    recovery.atomic_json(journal, [record, copy.deepcopy(record)])
    before = journal.read_bytes()
    monkeypatch.setattr(recovery, 'command_output', lambda command: '')
    with pytest.raises((RuntimeError, ValueError)):
        recovery.reconcile_rejected_submissions([root], [TOKEN])
    assert journal.read_bytes() == before


@pytest.mark.parametrize('token', ['invalid', '', TOKEN[:-1], '../' + TOKEN])
def test_invalid_resolution_token_is_rejected(tmp_path, monkeypatch, token):
    root, journal, _, _ = rejected_fixture(tmp_path)
    before = journal.read_bytes()
    monkeypatch.setattr(recovery, 'command_output', lambda command: '')
    with pytest.raises((RuntimeError, ValueError)):
        recovery.reconcile_rejected_submissions([root], [token])
    assert journal.read_bytes() == before


def test_login_passes_explicit_rejection_to_frozen_coordinator_without_python(tmp_path):
    result, _ = launch(tmp_path, ('resume', '--reject-token', TOKEN),
                       {'RECOVERY_WORKER_PARTITION': 'medusas_dev',
                        'RECOVERY_WORKER_GRES': 'gpu:1', 'RECOVERY_WORKER_NODELIST': 'medusa6'})
    assert result.returncode == 0, result.stderr
    args = (tmp_path / 'args').read_text().splitlines()
    assert args[-2:] == ['--reject-token', TOKEN]
    assert '--partition=gorgonas_dev' in args
    env = dict(line.split('=', 1) for line in (tmp_path / 'environment').read_text().splitlines()
               if '=' in line)
    assert env['RECOVERY_WORKER_PARTITION'] == 'medusas_dev'
    assert env['RECOVERY_WORKER_GRES'] == 'gpu:1'


def test_sbatch_failure_keeps_stderr_stdout_and_returncode_for_reconciliation(tmp_path, monkeypatch):
    run, state = make_run(tmp_path)
    message = 'sbatch: error: invalid partition specified: medusas\n'

    def rejected(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, output='', stderr=message)

    monkeypatch.setattr(recovery.subprocess, 'run', rejected)
    with pytest.raises(RuntimeError, match='invalid partition specified: medusas'):
        recovery.submit_run(run, state, 1000, recovery.worker_resources())
    root = Path(run['out_root'])
    journal = root / '.b3_recovery_submissions.json'
    saved = json.loads(journal.read_text())
    assert len(saved) == 1 and saved[0]['job_id'] is None
    assert saved[0]['indices'] == [55, 59, 1000, 1295]
    assert saved[0]['submission_failure'] == {'returncode': 1, 'stdout': '', 'stderr': message}
    # A recorded failure still needs explicit confirmation, not automatic skip.
    monkeypatch.setattr(recovery, 'command_output', lambda command: '')
    with pytest.raises(RuntimeError, match='confirmação incompleta'):
        recovery.verify_previous_submissions(root)
    recovery.reconcile_rejected_submissions([root], [saved[0]['token']])
    recovery.verify_previous_submissions(root)


@pytest.mark.parametrize('stdout', ['71538\n', '71538;cluster\n', 'ambiguous response\n'])
def test_nonzero_sbatch_with_stdout_is_ambiguous_and_cannot_be_resolved(tmp_path, monkeypatch, stdout):
    root, journal, _, record = rejected_fixture(tmp_path)
    record['submission_failure'] = {'returncode': 1, 'stdout': stdout, 'stderr': 'failed'}
    recovery.atomic_json(journal, [record])
    before = journal.read_bytes()
    monkeypatch.setattr(recovery, 'command_output', lambda command: '')
    with pytest.raises((RuntimeError, ValueError)):
        recovery.reconcile_rejected_submissions([root], [TOKEN])
    assert journal.read_bytes() == before


@pytest.mark.parametrize('resolution', [
    {'kind': 'skip_anyway'},
    {'kind': 'user_confirmed_rejection'},
    {'kind': 'user_confirmed_rejection', 'token': TOKEN},
    {'kind': 'user_confirmed_rejection', 'token': OTHER_TOKEN, 'evidence': {'returncode': 1}},
])
def test_incomplete_resolution_does_not_bypass_previous_submission_guard(tmp_path, monkeypatch, resolution):
    root, journal, _, record = rejected_fixture(tmp_path)
    record['rejection_resolution'] = resolution
    recovery.atomic_json(journal, [record])
    monkeypatch.setattr(recovery, 'command_output', lambda command: '')
    with pytest.raises(RuntimeError):
        recovery.verify_previous_submissions(root)


def test_resolved_rejection_reaudits_and_submits_only_current_missing_tasks(tmp_path, monkeypatch):
    verify_previous = recovery.verify_previous_submissions
    args = coordinator_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(recovery, 'verify_previous_submissions', verify_previous)
    args.reject_token = [TOKEN]
    first_root = Path(args.out_base) / 'one'
    old_plan = tmp_path / 'old_request/one/execution_plan.json'
    recovery.atomic_json(old_plan, {'out_root': str(first_root)})
    command = ['sbatch', '--partition=medusas', '--comment=b3-recovery:' + TOKEN]
    original = dict(token=TOKEN, job_id=None, command=command,
                    execution_plan=str(old_plan), indices=[0, 1, 2])
    journal = first_root / '.b3_recovery_submissions.json'
    recovery.atomic_json(journal, [original])
    recovery.atomic_json(old_plan.parent.parent / 'report.json', {
        'status': 'SUBMISSION_INCOMPLETE',
        'error': str(subprocess.CalledProcessError(1, command)),
        'runs': [dict(out_root=str(first_root), submission_records=[original])],
    })
    # Snapshot creation and audits are isolated fixture inputs; submission and
    # reconciliation use production code, with only Slurm replaced by a recorder.
    def snapshot(_repo, _commit, path):
        control = Path(path) / '.b3_recovery'
        control.mkdir(parents=True, exist_ok=True)
        for name in ('snapshot.json', 'dataset.json'):
            (control / name).write_text('{}')

    monkeypatch.setattr(recovery, 'create_snapshot', snapshot)
    monkeypatch.setattr(recovery, 'command_output',
                        lambda cmd: 'MaxArraySize = 1001' if cmd[0] == 'scontrol' else '')
    audit_process = recovery.subprocess.run
    submissions = []

    def scheduler_or_audit(cmd, **kwargs):
        if cmd[0] != 'sbatch':
            return audit_process(cmd, **kwargs)
        path = Path(cmd[cmd.index('--indices') + 1])
        assert json.loads(path.read_text()) == [2]  # 0 and 1 now completed.
        assert '--partition=medusas_dev' in cmd and '--gres=gpu:1' in cmd
        submissions.append(cmd)
        return SimpleNamespace(stdout=str(70000 + len(submissions)) + '\n')

    monkeypatch.setenv('RECOVERY_WORKER_PARTITION', 'medusas_dev')
    monkeypatch.setenv('RECOVERY_WORKER_GRES', 'gpu:1')
    monkeypatch.setattr(recovery.subprocess, 'run', scheduler_or_audit)
    original_files = {p: p.read_bytes() for root in (Path(args.out_base) / t for t in args.run_tag)
                      for p in (root / recovery.GROUP).iterdir()}
    assert recovery.coordinator(args) == 0
    assert len(submissions) == 2
    report = json.loads((Path(args.state_dir) / 'report.json').read_text())
    assert report['status'] == 'SUBMITTED'
    assert report['resolved_rejections'][0]['token'] == TOKEN
    assert [r['jobs'][0]['job_id'] for r in report['runs']] == ['70001', '70002']
    records = json.loads(journal.read_text())
    assert len(records) == 2
    assert all(records[0][key] == value for key, value in original.items())
    assert recovery.rejection_resolved(records[0])
    assert records[1]['indices'] == [2] and records[1]['job_id'] == '70001'
    assert {p: p.read_bytes() for p in original_files} == original_files
