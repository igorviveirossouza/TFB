"""Recovery orchestration must never schedule or overwrite proven work."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import b3_recovery as recovery
import b3_recovery_audit as audit


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    # worker deliberately sets process-level variables for its child benchmark.
    # Keep those changes inside each test, including direct os.environ updates.
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    for key in list(os.environ):
        if key.startswith('RECOVERY_'):
            monkeypatch.delenv(key)


def test_defaults_match_four_user_runs():
    assert recovery.DEFAULT_TAGS == [
        'comp_lb32-128-256-512_gaussian_0.025_complementary_k1_t80_v5',
        'comp_lb32-128-256-512_uniform_0.025_complementary_k1_t80_v5',
        'comp_lb32-128-256-512_cauchy_0.025_complementary_k1_t80_v5',
        'comp_lb32-128-256-512_cauchy_0.025_natural_k1_t80_v5',
    ]


@pytest.mark.parametrize('host,job', [('phocus4', '123'), ('gorgona1', None)])
def test_processing_refused_without_compute_allocation(monkeypatch, host, job):
    monkeypatch.setattr(recovery.socket, 'gethostname', lambda: host)
    if job is None:
        monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    else:
        monkeypatch.setenv('SLURM_JOB_ID', job)
    with pytest.raises(RuntimeError, match='phocus4'):
        recovery.require_compute()


def test_scheduler_environment_does_not_inherit_old_resources(monkeypatch, tmp_path):
    monkeypatch.setenv('SBATCH_GRES', 'gpu:1')
    monkeypatch.setenv('SBATCH_NODELIST', 'medusa6')
    monkeypatch.setenv('GRES', 'gpu:1')
    monkeypatch.setenv('NODELIST', 'medusa6')
    monkeypatch.setenv('PARTITION', 'medusas_dev')
    monkeypatch.setenv('B3_INLINE_CONFIG', '{"old": true}')
    monkeypatch.setenv('PYTHONPATH', '/mutable/development')
    env = recovery.clean_env(tmp_path)
    assert not any(key.startswith('SBATCH_') for key in env)
    assert 'B3_INLINE_CONFIG' not in env
    assert env['TFB_ROOT'] == str(tmp_path)
    assert '/mutable/development' not in env['PYTHONPATH']
    resources = recovery.worker_resources()
    assert resources['partition'] == 'gorgonas'
    assert resources['gres'] == resources['nodelist'] == ''


def make_run(tmp_path, missing=(55, 59, 1000, 1295)):
    out_root = tmp_path / 'out' / 'example'
    original = out_root / recovery.GROUP
    original.mkdir(parents=True)
    (original / 'run_config.json').write_text('{}')
    (original / 'design_composite_trading.csv').write_text('array_index\n')
    state = tmp_path / 'state'
    (state / 'example').mkdir(parents=True)
    snapshot = tmp_path / 'snapshot' / '.b3_recovery'
    snapshot.mkdir(parents=True)
    for name in ('snapshot.json', 'dataset.json'):
        (snapshot / name).write_text('{}')
    run = dict(run_tag='example', out_root=str(out_root), snapshot=str(tmp_path / 'snapshot'),
               result_root=str(tmp_path / 'results'), audit=dict(missing_indices=list(missing)))
    return run, state


def test_submission_maps_only_sparse_missing_indices_and_records_before_sbatch(monkeypatch, tmp_path):
    run, state = make_run(tmp_path)
    calls = []

    def sbatch(command, **kwargs):
        journal = json.loads((Path(run['out_root']) / '.b3_recovery_submissions.json').read_text())
        assert journal[-1]['job_id'] is None
        assert journal[-1]['command'] == command
        assert not any(x.startswith('--gres') or x.startswith('--nodelist') for x in command)
        assert '--partition=gorgonas' in command
        assert not any(k.startswith('SBATCH_') for k in kwargs['env'])
        calls.append(command)
        return SimpleNamespace(stdout=str(500 + len(calls)) + '\n')

    monkeypatch.setattr(recovery.subprocess, 'run', sbatch)
    jobs = recovery.submit_run(run, state, 3, recovery.worker_resources())
    assert [job['count'] for job in jobs] == [3, 1]
    assert json.loads((state / 'example' / 'indices_0.json').read_text()) == [55, 59, 1000]
    assert json.loads((state / 'example' / 'indices_1.json').read_text()) == [1295]
    assert '--array=0-2%2' in calls[0]
    assert '--array=0-0%2' in calls[1]
    assert '--dependency=afterany:501' in calls[1]
    plan = json.loads((state / 'example' / 'execution_plan.json').read_text())
    assert plan['missing_indices'] == [55, 59, 1000, 1295]
    assert 54 not in plan['missing_indices']
    journal = json.loads((Path(run['out_root']) / '.b3_recovery_submissions.json').read_text())
    assert [record['job_id'] for record in journal] == ['501', '502']


def test_fully_complete_run_submits_nothing(monkeypatch, tmp_path):
    run, state = make_run(tmp_path, missing=())
    monkeypatch.setattr(recovery.subprocess, 'run', lambda *a, **k: pytest.fail('sbatch called'))
    assert recovery.submit_run(run, state, 1000, recovery.worker_resources()) == []
    assert not (Path(run['out_root']) / '.b3_recovery_submissions.json').exists()


def test_ambiguous_sbatch_response_preserves_intent_and_blocks_duplicate(monkeypatch, tmp_path):
    run, state = make_run(tmp_path)
    monkeypatch.setattr(recovery.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='unknown reply'))
    with pytest.raises(RuntimeError, match='Resposta inesperada'):
        recovery.submit_run(run, state, 1000, recovery.worker_resources())
    monkeypatch.setattr(recovery, 'command_output', lambda *a: '')
    with pytest.raises(RuntimeError, match='confirmação incompleta'):
        recovery.verify_previous_submissions(run['out_root'])


def test_partial_submission_keeps_accepted_job_and_uncertain_chunk_in_journal(monkeypatch, tmp_path):
    run, state = make_run(tmp_path)
    replies = iter(['123\n', 'ambiguous reply'])
    monkeypatch.setattr(recovery.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=next(replies)))
    with pytest.raises(RuntimeError, match='Resposta inesperada'):
        recovery.submit_run(run, state, 3, recovery.worker_resources())
    records = json.loads((Path(run['out_root']) / '.b3_recovery_submissions.json').read_text())
    assert records[0]['job_id'] == '123'
    assert records[0]['indices'] == [55, 59, 1000]
    assert records[1]['job_id'] is None
    assert records[1]['indices'] == [1295]


@pytest.mark.parametrize('accounting,allowed', [
    ('123_0|COMPLETED\n123_1|CANCELLED by 1001\n', True),
    ('123_0|TIMEOUT\n', True), ('', False), ('123_0|RUNNING\n', False),
])
def test_resubmission_needs_terminal_accounting(monkeypatch, tmp_path, accounting, allowed):
    recovery.atomic_json(tmp_path / '.b3_recovery_submissions.json', [dict(job_id='123', token='abc')])
    monkeypatch.setattr(recovery, 'command_output', lambda cmd: '' if cmd[0] == 'squeue' else accounting)
    if allowed:
        recovery.verify_previous_submissions(tmp_path)
    else:
        with pytest.raises(RuntimeError, match='não confirmado'):
            recovery.verify_previous_submissions(tmp_path)


def test_active_prior_submission_is_never_scheduled_again(monkeypatch, tmp_path):
    recovery.atomic_json(tmp_path / '.b3_recovery_submissions.json', [dict(job_id='123', token='abc')])
    monkeypatch.setattr(recovery, 'command_output', lambda cmd: '123_[0-7]|PENDING')
    with pytest.raises(RuntimeError, match='ativa'):
        recovery.verify_previous_submissions(tmp_path)


def test_raw_artifact_inventory_includes_nested_archives_but_not_empty_tempdirs(tmp_path):
    (tmp_path / 'task_empty').mkdir()
    raw = tmp_path / 'task_killed' / 'nested' / 'model.csv.tar.gz'
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b'original saved forecast')
    (tmp_path / 'unrelated.log').write_text('not a worker temporary result')
    assert recovery.raw_artifacts(tmp_path) == [str(raw)]


def coordinator_fixture(tmp_path, monkeypatch, blocked_tag=None, raw_tag=None):
    tags = ['one', 'two']
    out = tmp_path / 'out'
    for tag in tags:
        original = out / tag / recovery.GROUP
        original.mkdir(parents=True)
        recovery.atomic_json(original / 'run_config.json', dict(git_commit='a' * 40, metadata=dict(dataset_sha256='b' * 64)))
        for name in ('completed_tasks.csv', 'design_composite_trading.csv'):
            (original / name).write_text('array_index\n')
    monkeypatch.setattr(recovery, 'check_no_original_workers', lambda roots: None)
    monkeypatch.setattr(recovery, 'verify_previous_submissions', lambda root: None)
    monkeypatch.setattr(recovery, 'create_snapshot', lambda *a: None)
    monkeypatch.setattr(recovery, 'materialize_dataset', lambda *a: None)
    monkeypatch.setattr(recovery, 'raw_artifacts', lambda path: ['surviving.csv.tar.gz'] if Path(path).name == raw_tag else [])

    def run_audit(cmd, **kwargs):
        tag = Path(cmd[cmd.index('--out-root') + 1]).name
        report = dict(complete_indices=[0, 1], missing_indices=[2],
                      blocked=[dict(array_index=3, reason='orphan')] if tag == blocked_tag else [])
        recovery.atomic_json(cmd[cmd.index('--report') + 1], report)
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    monkeypatch.setattr(recovery.subprocess, 'run', run_audit)
    args = SimpleNamespace(action='resume', state_dir=str(tmp_path / 'state'), run_tag=tags,
                           out_base=str(out), result_base=str(tmp_path / 'raw'), repo=str(tmp_path / 'repo'))
    return args


@pytest.mark.parametrize('blocked_tag,raw_tag', [('two', None), (None, 'two')])
def test_one_blocked_grade_prevents_all_training_submissions(monkeypatch, tmp_path, blocked_tag, raw_tag):
    args = coordinator_fixture(tmp_path, monkeypatch, blocked_tag, raw_tag)
    monkeypatch.setattr(recovery, 'submit_run', lambda *a: pytest.fail('no array may be submitted'))
    assert recovery.coordinator(args) == 2
    report = json.loads((Path(args.state_dir) / 'report.json').read_text())
    assert report['status'] == 'BLOCKED'
    assert len(report['runs']) == 2
    assert report['runs'][0]['audit']['complete_indices'] == [0, 1]


def test_coordinator_respects_cluster_max_array_size(monkeypatch, tmp_path):
    args = coordinator_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(recovery, 'command_output', lambda cmd: 'MaxArraySize = 500')
    calls = []
    monkeypatch.setattr(recovery, 'submit_run', lambda run, state, size, resources: calls.append((run['run_tag'], size)) or [])
    assert recovery.coordinator(args) == 0
    assert calls == [('one', 500), ('two', 500)]


def test_initial_block_reason_is_visible_in_coordinator_output(monkeypatch, tmp_path, capsys):
    args = coordinator_fixture(tmp_path, monkeypatch)

    def blocked(roots):
        assert roots == [Path(args.out_base) / tag for tag in args.run_tag]
        raise RuntimeError('conflito na grade selecionada: 71017_1')

    monkeypatch.setattr(recovery, 'check_no_original_workers', blocked)
    assert recovery.coordinator(args) == 2
    assert 'BLOCKED: conflito na grade selecionada: 71017_1' in capsys.readouterr().out


def test_coordinator_reports_partial_submission_and_journal_blocks_repeat(monkeypatch, tmp_path):
    verify_previous = recovery.verify_previous_submissions
    args = coordinator_fixture(tmp_path, monkeypatch)
    args.run_tag = ['one']
    fake_audit = recovery.subprocess.run

    def snapshot(repo, commit, destination):
        controls = destination / '.b3_recovery'
        controls.mkdir(parents=True)
        for name in ('snapshot.json', 'dataset.json'):
            (controls / name).write_text('{}')

    monkeypatch.setattr(recovery, 'create_snapshot', snapshot)
    monkeypatch.setattr(recovery, 'command_output', lambda cmd: 'MaxArraySize = 1')
    replies = iter(['123\n', 'ambiguous reply'])

    def run(cmd, **kwargs):
        if cmd[0] == 'sbatch':
            return SimpleNamespace(stdout=next(replies))
        result = fake_audit(cmd, **kwargs)
        path = cmd[cmd.index('--report') + 1]
        value = json.loads(Path(path).read_text())
        value['missing_indices'] = [2, 5]
        recovery.atomic_json(path, value)
        return result

    monkeypatch.setattr(recovery.subprocess, 'run', run)
    assert recovery.coordinator(args) == 2
    report = json.loads((Path(args.state_dir) / 'report.json').read_text())
    assert report['status'] == 'SUBMISSION_INCOMPLETE'
    records = report['runs'][0]['submission_records']
    assert [record['job_id'] for record in records] == ['123', None]
    monkeypatch.setattr(recovery, 'command_output', lambda cmd: '' if cmd[0] == 'squeue' else '123_0|CANCELLED\n')
    with pytest.raises(RuntimeError, match='confirmação incompleta'):
        verify_previous(Path(args.out_base) / 'one')


def worker_fixture(tmp_path, monkeypatch, statuses=('missing', 'complete')):
    original = tmp_path / 'out' / recovery.GROUP
    original.mkdir(parents=True)
    metadata = {}
    for name in ('dataset', 'calendar'):
        path = tmp_path / (name + '.csv')
        path.write_text(name)
        metadata[name] = str(path)
        metadata[name + '_sha256'] = recovery.sha(path)
    saved = dict(config={'frozen': True}, metadata=metadata, design_sha256='signature')
    recovery.atomic_json(original / 'run_config.json', saved)
    (original / 'design_composite_trading.csv').write_text('original design')
    controls = tmp_path / 'snapshot' / '.b3_recovery'
    controls.mkdir(parents=True)
    for name in ('snapshot.json', 'dataset.json'):
        (controls / name).write_text('{}')
    plan = dict(snapshot=str(tmp_path / 'snapshot'), out_root=str(tmp_path / 'out'),
                result_root=str(tmp_path / 'results'), missing_indices=[55, 1295],
                resources=recovery.worker_resources(),
                snapshot_control_sha256={name: recovery.sha(controls / name)
                                         for name in ('snapshot.json', 'dataset.json')},
                tools_sha256={name: recovery.sha(ROOT / 'scripts' / name) for name in recovery.TOOLS},
                run_config_sha256=recovery.sha(original / 'run_config.json'),
                design_sha256=recovery.sha(original / 'design_composite_trading.csv'))
    plan_path, index_path = tmp_path / 'plan.json', tmp_path / 'indices.json'
    recovery.atomic_json(plan_path, plan)
    recovery.atomic_json(index_path, [55, 1295])
    args = SimpleNamespace(plan=str(plan_path), plan_sha=recovery.sha(plan_path),
                           indices=str(index_path), indices_sha=recovery.sha(index_path))
    calls = []
    tasks = [dict(array_index=i) for i in range(1296)]
    runner = SimpleNamespace(make_tasks=lambda config, mode: tasks, provenance=lambda *a: 'prov',
                             digest=lambda value: 'signature', output_path=lambda task: Path(f'task-{task["array_index"]}'),
                             worker=lambda config, tasks, out_root, index: calls.append(index))
    monkeypatch.setattr(recovery, 'verify_snapshot', lambda path: None)
    monkeypatch.setattr(recovery, 'verify_dataset', lambda path: None)
    monkeypatch.setattr(recovery, 'load_runner', lambda path: runner)
    status_iter = iter(statuses)
    monkeypatch.setattr(audit, 'audit_task', lambda *a, **k: dict(status=next(status_iter)))
    monkeypatch.setenv('SLURM_ARRAY_TASK_ID', '1')
    return args, calls, original


def test_worker_maps_local_array_index_to_original_task(monkeypatch, tmp_path):
    args, calls, _ = worker_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv('SBATCH_GRES', 'gpu:1')
    monkeypatch.setenv('B3_INLINE_CONFIG', '{"old": true}')
    recovery.worker(args)
    assert calls == [1295]
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert os.environ['TFB_ROOT'] == str(tmp_path / 'snapshot')
    assert 'SBATCH_GRES' not in os.environ
    assert 'B3_INLINE_CONFIG' not in os.environ


def test_worker_skips_if_completed_after_planning(monkeypatch, tmp_path):
    args, calls, _ = worker_fixture(tmp_path, monkeypatch, statuses=('complete',))
    recovery.worker(args)
    assert calls == []


def test_worker_never_retrains_orphan_or_bad_output(monkeypatch, tmp_path):
    args, calls, _ = worker_fixture(tmp_path, monkeypatch, statuses=('blocked',))
    with pytest.raises(RuntimeError, match='preservar resultados'):
        recovery.worker(args)
    assert calls == []


@pytest.mark.parametrize('target', ['plan', 'indices', 'original', 'tool_hash', 'dataset', 'snapshot_control'])
def test_worker_rejects_changed_evidence_before_training(monkeypatch, tmp_path, target):
    args, calls, original = worker_fixture(tmp_path, monkeypatch)
    if target in {'plan', 'indices'}:
        Path(getattr(args, target)).write_text('changed')
    elif target == 'original':
        (original / 'run_config.json').write_text('changed')
    elif target == 'tool_hash':
        plan = json.loads(Path(args.plan).read_text())
        plan['tools_sha256']['b3_recovery.py'] = '0' * 64
        recovery.atomic_json(args.plan, plan)
        args.plan_sha = recovery.sha(args.plan)
    elif target == 'snapshot_control':
        (tmp_path / 'snapshot' / '.b3_recovery' / 'snapshot.json').write_text('changed')
    else:
        (tmp_path / 'dataset.csv').write_text('changed')
    with pytest.raises(ValueError):
        recovery.worker(args)
    assert calls == []


def test_fresh_process_imports_original_commit_after_development_changes(tmp_path):
    """Real Git archive + fresh interpreter; no mocked Python import isolation."""
    repo = tmp_path / 'development'
    (repo / 'scripts').mkdir(parents=True)
    (repo / 'ts_benchmark').mkdir()
    (repo / 'ts_benchmark' / '__init__.py').write_text('')
    marker = repo / 'ts_benchmark' / 'marker.py'
    marker.write_text('VERSION = "original"\n')
    (repo / 'scripts' / 'b3_experiment.py').write_text(
        'from pathlib import Path\n'
        'from ts_benchmark.marker import VERSION\n'
        'ROOT = Path(__file__).resolve().parents[1]\n')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c',
                    'user.email=test@example.invalid', 'commit', '-qm', 'original'], check=True)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    frozen = recovery.create_snapshot(repo, commit, tmp_path / 'frozen')
    marker.write_text('VERSION = "MASTER development"\n')
    program = (
        'import json, sys\n'
        'sys.path.insert(0, sys.argv[1])\n'
        'import b3_recovery\n'
        'runner = b3_recovery.load_runner(sys.argv[2])\n'
        'print(json.dumps([runner.VERSION, str(runner.ROOT)]))\n')
    result = subprocess.run([sys.executable, '-B', '-c', program, str(ROOT / 'scripts'), str(frozen)],
                            capture_output=True, text=True, check=True,
                            env=dict(os.environ, PYTHONPATH=str(repo)))
    assert json.loads(result.stdout) == ['original', str(frozen)]
    recovery.verify_snapshot(frozen)
