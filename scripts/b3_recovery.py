#!/usr/bin/env python3
"""Resume original composite designs using verified, isolated execution snapshots.

No experiment is prepared again. Original config, design and result files remain
authoritative. All Python entry points require a compute allocation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import uuid

from b3_recovery_snapshot import create_snapshot, materialize_dataset, verify_snapshot, verify_dataset

DEFAULT_TAGS = [
    f'comp_lb32-128-256-512_{mask}_0.025_{mode}_k1_t80_v5'
    for mask, mode in [('gaussian', 'complementary'), ('uniform', 'complementary'),
                       ('cauchy', 'complementary'), ('cauchy', 'natural')]]
GROUP = Path('treino=80/experimento=composto')
TOOLS = ('b3_recovery.py', 'b3_recovery_audit.py', 'b3_recovery_snapshot.py')
TERMINAL = {'BOOT_FAIL', 'CANCELLED', 'COMPLETED', 'DEADLINE', 'FAILED',
            'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'TIMEOUT', 'REVOKED'}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def locked(path):
    with Path(path).open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def require_compute():
    host = socket.gethostname().split('.')[0].lower()
    if not os.environ.get('SLURM_JOB_ID') or host.startswith('phocus'):
        raise RuntimeError('Execute pelo launcher Slurm: nenhum processamento Python na phocus4.')


def command_output(command):
    return subprocess.check_output(command, text=True).strip()


def clean_env(snapshot=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith('SBATCH_')}
    for key in ('B3_INLINE_CONFIG', 'B3_CONFIG', 'B3_CONFIG_SOURCE', 'B3_WORKER_LAUNCHER',
                'TFB_ROOT', 'PYTHONPATH'):
        env.pop(key, None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    if snapshot:
        env['TFB_ROOT'] = str(snapshot)
        env['PYTHONPATH'] = os.pathsep.join([str(snapshot), str(Path(snapshot) / 'scripts')])
    return env


def load_runner(snapshot):
    snapshot = Path(snapshot).resolve()
    sys.dont_write_bytecode = True
    if any(name == 'ts_benchmark' or name.startswith('ts_benchmark.') for name in sys.modules):
        raise RuntimeError('Importação prévia de TFB: o processo precisa começar isolado.')
    sys.path[:0] = [str(snapshot / 'scripts'), str(snapshot)]
    spec = importlib.util.spec_from_file_location('b3_experiment', snapshot / 'scripts/b3_experiment.py')
    runner = importlib.util.module_from_spec(spec)
    sys.modules['b3_experiment'] = runner
    spec.loader.exec_module(runner)
    return runner


def raw_artifacts(result_root):
    """A killed worker can leave a complete raw forecast before Parquet export."""
    root = Path(result_root)
    return [str(p) for directory in sorted(root.glob('task_*')) if directory.is_dir()
            for p in sorted(directory.rglob('*')) if p.is_file()]


def slurm_job_fields(details):
    # scontrol --oneliner leaves command arguments and spaces in paths intact.
    matches = list(re.finditer(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=', details))
    fields = {}
    for position, match in enumerate(matches):
        key = match.group(1)
        if key in fields:
            raise ValueError('Resposta Slurm contém mais de um registro ou campo duplicado: ' + key)
        end = matches[position + 1].start() if position + 1 < len(matches) else len(details)
        fields[key] = details[match.end():end].strip()
    return fields


def original_job_scope(details, out_roots):
    """Identify the run from its launcher and log paths, never the shared name.

    The original composite launcher fixes both output and log paths from the
    same RUN_TAG. MASTER uses a distinct launcher/log tree. Unrecognized layouts
    remain blocked instead of being guessed safe.
    """
    fields = slurm_job_fields(details)
    tags = {Path(root).name for root in out_roots}
    logs = [Path(fields.get(key, '')) for key in ('StdOut', 'StdErr')]
    if any(tags.intersection(path.parts) for path in logs):
        return 'selected'
    if (any(not path.is_absolute() for path in logs) or logs[0].parent != logs[1].parent
            or not logs[0].name.startswith('composto_') or not logs[1].name.startswith('composto_')
            or logs[0].suffix != '.out' or logs[1].suffix != '.err'):
        return 'unknown'
    try:
        command = shlex.split(fields.get('Command', ''))
    except ValueError:
        return 'unknown'
    if not command:
        return 'unknown'
    launcher = Path(command[0]).name
    parent = logs[0].parent
    if launcher == 'run_composite_trading_experiment.sh' and parent.parent.name == 'logs':
        return 'unrelated'
    if launcher == 'run_master_experiment.sh':
        if any(p.name == 'master' and p.parent.name == 'logs' for p in [parent, *parent.parents]):
            return 'unrelated'
    return 'unknown'


def original_workers_in_queue():
    rows = command_output(['squeue', '--me', '--array', '--noheader', '--format=%i|%j|%T'])
    active = []
    for line in rows.splitlines():
        columns = [part.strip() for part in line.split('|')]
        if len(columns) != 3:
            raise RuntimeError('Resposta de squeue não reconhecida: ' + line)
        if columns[1] in {'b3_composto', 'composto'}:
            active.append(columns)
    return active


def check_no_original_workers(out_roots):
    conflicts, unknown = [], []
    for job_id, name, state in original_workers_in_queue():
        try:
            details = command_output(['scontrol', '--oneliner', 'show', 'job', job_id])
        except subprocess.CalledProcessError:
            # The job may finish between squeue and scontrol. Confirm absence;
            # an inspection failure for a still-active job cannot be ignored.
            if job_id not in {row[0] for row in original_workers_in_queue()}:
                continue
            unknown.append(job_id)
            continue
        try:
            fields = slurm_job_fields(details)
            if fields.get('JobState') in TERMINAL:
                continue
            scope = original_job_scope(details, out_roots)
        except ValueError:
            scope = 'unknown'
        if scope == 'selected':
            conflicts.append(f'{job_id}|{name}|{state}')
        elif scope == 'unknown':
            unknown.append(job_id)
    if conflicts:
        raise RuntimeError('Ainda há jobs ativos nas grades selecionadas; preservar o trabalho em curso: '
                           + '; '.join(conflicts))
    if unknown:
        raise RuntimeError('Não foi possível identificar a grade dos jobs ativos ' + ', '.join(unknown)
                           + '. Inspecione: scontrol show job ' + unknown[0])


def verify_previous_submissions(out_root):
    path = Path(out_root) / '.b3_recovery_submissions.json'
    if not path.exists():
        return
    records = json.loads(path.read_text())
    # Querying a specific already-purged ID can make squeue exit nonzero.
    queued = command_output(['squeue', '--me', '--noheader', '--format=%i|%T'])
    active_ids = {line.split('|')[0].strip().split('_')[0]
                  for line in queued.splitlines() if '|' in line}
    for record in records:
        if not record.get('job_id'):
            raise RuntimeError('Submissão anterior com confirmação incompleta; conferir token '
                               + record['token'] + ' antes de tentar novamente.')
        job = record['job_id']
        if job in active_ids:
            raise RuntimeError('Retomada anterior ainda ativa: ' + job)
        accounting = command_output(['sacct', '--noheader', '--parsable2', '--allocations',
                                     '--jobs=' + job, '--format=JobIDRaw,State'])
        states = [line.split('|')[1].strip().split()[0].rstrip('+')
                  for line in accounting.splitlines() if '|' in line and line.split('|')[1].strip()]
        if not states or any(state not in TERMINAL for state in states):
            raise RuntimeError('Encerramento da retomada anterior não confirmado pelo sacct: ' + job)


def audit_child(args):
    snapshot, out_root = Path(args.snapshot), Path(args.out_root)
    verify_snapshot(snapshot)
    verify_dataset(snapshot)
    from b3_recovery_audit import audit_run
    # Load the sealed recovery helper before historical scripts enter sys.path.
    runner = load_runner(snapshot)
    plan_path = out_root / GROUP / 'run_config.json'
    saved = json.loads(plan_path.read_text())
    config = saved['config']
    if config['train_percentages'] != [80] or 'MASTER' in config['models'] or 'VAR' in config['models']:
        raise ValueError('Esta retomada aceita somente as grades compostas originais de treino=80, sem MASTER/VAR.')
    tasks = runner.make_tasks(config, 'composto')
    report = audit_run(runner, config, tasks, out_root, saved)
    report.update(original_commit=saved['git_commit'], design_sha256=saved['design_sha256'],
                  run_config_sha256=sha(plan_path), total_tasks=len(tasks),
                  config=config, metadata=saved['metadata'], snapshot=str(snapshot))
    atomic_json(args.report, report)


def worker_resources():
    options = {
        'partition': os.environ.get('RECOVERY_WORKER_PARTITION', 'gorgonas'),
        'time': os.environ.get('RECOVERY_WORKER_TIME', '01:30:00'),
        'cpus': int(os.environ.get('RECOVERY_WORKER_CPUS', '3')),
        'memory': os.environ.get('RECOVERY_WORKER_MEMORY', ''),
        'gres': os.environ.get('RECOVERY_WORKER_GRES', ''),
        'nodelist': os.environ.get('RECOVERY_WORKER_NODELIST', ''),
        'concurrent': int(os.environ.get('RECOVERY_MAX_CONCURRENT', '2')),
    }
    if options['cpus'] < 1 or options['concurrent'] < 1:
        raise ValueError('CPUs e concorrência devem ser positivos.')
    if not options['partition'] or not options['time']:
        raise ValueError('Partição e tempo são obrigatórios.')
    return options


def chunks(indices, size):
    if size < 1:
        raise ValueError('Tamanho de array inválido.')
    return [indices[start:start + size] for start in range(0, len(indices), size)]


def build_sbatch(plan_path, index_path, count, resources, launcher, logs, token, previous=None):
    command = ['sbatch', '--parsable', '--export=ALL', '--no-requeue',
               '--nodes=1', '--ntasks=1',
               '--partition=' + resources['partition'], '--time=' + resources['time'],
               '--cpus-per-task=' + str(resources['cpus']),
               f'--array=0-{count - 1}%{resources["concurrent"]}', '--job-name=b3_resume',
               '--comment=b3-recovery:' + token, '--chdir=' + str(plan_path.parent),
               '--output=' + str(logs / 'resume_%A_%a.out'),
               '--error=' + str(logs / 'resume_%A_%a.err')]
    for key in ('gres', 'nodelist'):
        if resources[key]:
            command.append('--' + key + '=' + resources[key])
    if resources['memory']:
        command.append('--mem=' + resources['memory'])
    if previous:
        command.append('--dependency=afterany:' + previous)
    command.extend([str(launcher), '--plan', str(plan_path), '--plan-sha', sha(plan_path),
                    '--indices', str(index_path), '--indices-sha', sha(index_path)])
    return command


def submit_run(run, state, chunk_size, resources):
    out_root = Path(run['out_root'])
    run_dir = state / run['run_tag']
    tools_dir = Path(__file__).resolve().parent
    missing = run['audit']['missing_indices']
    if not missing:
        return []
    original = out_root / GROUP
    plan = dict(schema_version=1, snapshot=run['snapshot'], out_root=str(out_root),
                result_root=run['result_root'], run_tag=run['run_tag'],
                run_config_sha256=sha(original / 'run_config.json'),
                design_sha256=sha(original / 'design_composite_trading.csv'),
                missing_indices=missing, resources=resources,
                snapshot_control_sha256={name: sha(Path(run['snapshot']) / '.b3_recovery' / name)
                                         for name in ('snapshot.json', 'dataset.json')},
                tools_sha256={name: sha(tools_dir / name) for name in TOOLS})
    plan_path = run_dir / 'execution_plan.json'
    atomic_json(plan_path, plan)
    launcher = run_dir / 'worker.sh'
    launcher.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
                        'export PYTHONDONTWRITEBYTECODE=1\nexec ' + shlex.join([
                            sys.executable, str(Path(__file__).resolve()), 'worker']) + ' "$@"\n')
    launcher.chmod(0o555)
    logs = run_dir / 'logs'
    logs.mkdir(exist_ok=True)
    journal_path = out_root / '.b3_recovery_submissions.json'
    records = json.loads(journal_path.read_text()) if journal_path.exists() else []
    previous = None
    jobs = []
    for chunk_number, indices in enumerate(chunks(missing, chunk_size)):
        index_path = run_dir / f'indices_{chunk_number}.json'
        atomic_json(index_path, indices)
        token = uuid.uuid4().hex
        command = build_sbatch(plan_path, index_path, len(indices), resources, launcher, logs, token, previous)
        record = dict(token=token, indices=indices, execution_plan=str(plan_path),
                      command=command, job_id=None)
        records.append(record)
        # Persist intent BEFORE sbatch. An ambiguous submission blocks a second
        # attempt instead of risking another training of the same task.
        atomic_json(journal_path, records)
        result = subprocess.run(command, env=clean_env(), check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split(';')[0]
        if not job_id.isdigit():
            raise RuntimeError('Resposta inesperada de sbatch; conferir token ' + token + ': ' + result.stdout)
        record['job_id'] = previous = job_id
        atomic_json(journal_path, records)
        jobs.append(dict(job_id=job_id, count=len(indices), indices_file=str(index_path)))
        print(f'{run["run_tag"]}: job {job_id}, somente {len(indices)} tarefas faltantes.', flush=True)
    return jobs


def coordinator(args):
    state = Path(args.state_dir).resolve()
    state.mkdir(parents=True, exist_ok=True)
    tags = args.run_tag or DEFAULT_TAGS
    if len(set(tags)) != len(tags) or any(not re.fullmatch(r'[A-Za-z0-9_.-]+', tag) for tag in tags):
        raise ValueError('RUN_TAG inválido ou duplicado.')
    resources = worker_resources()
    report = dict(schema_version=1, action=args.action, status='AUDITING', runs=[],
                  python=sys.executable, python_version=sys.version, resources=resources)
    report_path = state / 'report.json'
    atomic_json(report_path, report)
    out_roots = [Path(args.out_base).resolve() / tag for tag in tags]
    try:
        check_no_original_workers(out_roots)
        with ExitStack() as stack:
            # Serialize all coordinators for these roots, including audit-only.
            for tag in sorted(tags):
                root = Path(args.out_base).resolve() / tag
                if not root.is_dir():
                    raise FileNotFoundError('Grade original não encontrada: ' + str(root))
                stack.enter_context(locked(root / '.b3_recovery.lock'))
                verify_previous_submissions(root)
            for tag in tags:
                out_root = Path(args.out_base).resolve() / tag
                run_dir = state / tag
                run_dir.mkdir(exist_ok=True)
                run = dict(run_tag=tag, out_root=str(out_root),
                           result_root=str(Path(args.result_base).resolve() / tag))
                report['runs'].append(run)
                try:
                    original_path = out_root / GROUP / 'run_config.json'
                    saved = json.loads(original_path.read_text())
                    # Copy only manifests for diagnostics; never rewrite originals.
                    for name in ('run_config.json', 'completed_tasks.csv', 'design_composite_trading.csv'):
                        source = out_root / GROUP / name
                        (run_dir / name).write_bytes(source.read_bytes())
                    snapshot = state / 'snapshots' / (saved['git_commit'] + '-' + saved['metadata']['dataset_sha256'][:16])
                    create_snapshot(Path(args.repo).resolve(), saved['git_commit'], snapshot)
                    materialize_dataset(snapshot, saved['metadata'])
                    run['snapshot'] = str(snapshot)
                    child_report = run_dir / 'audit.json'
                    child = [sys.executable, str(Path(__file__).resolve()), '_audit',
                             '--snapshot', str(snapshot), '--out-root', str(out_root), '--report', str(child_report)]
                    result = subprocess.run(child, env=clean_env(snapshot), text=True, capture_output=True)
                    (run_dir / 'audit.stdout').write_text(result.stdout)
                    (run_dir / 'audit.stderr').write_text(result.stderr)
                    if result.returncode:
                        raise RuntimeError('Auditoria isolada falhou: ' + result.stderr[-12000:])
                    run['audit'] = json.loads(child_report.read_text())
                    run['raw_artifacts'] = raw_artifacts(run['result_root'])
                    run['blocked'] = bool(run['audit']['blocked'] or run['raw_artifacts'])
                    print(f'{tag}: concluídas={len(run["audit"]["complete_indices"])}; '
                          f'faltantes={len(run["audit"]["missing_indices"])}; '
                          f'bloqueios={len(run["audit"]["blocked"])}; '
                          f'arquivos brutos pendentes={len(run["raw_artifacts"])}.', flush=True)
                except Exception as exc:
                    run.update(blocked=True, error=str(exc))
                    print(f'{tag}: BLOQUEADO: {exc}', flush=True)
                atomic_json(report_path, report)
            blocked = any(run['blocked'] for run in report['runs'])
            report['status'] = 'BLOCKED' if blocked else 'READY'
            atomic_json(report_path, report)
            if args.action == 'resume' and not blocked:
                check_no_original_workers(out_roots)
                settings = command_output(['scontrol', 'show', 'config'])
                match = re.search(r'MaxArraySize\s*=\s*(\d+)', settings)
                if not match:
                    raise RuntimeError('Não foi possível determinar MaxArraySize; nenhum array será submetido.')
                size = min(1000, int(match.group(1)))
                report['status'] = 'SUBMITTING'
                atomic_json(report_path, report)
                for run in report['runs']:
                    run['jobs'] = submit_run(run, state, size, resources)
                    atomic_json(report_path, report)
                report['status'] = 'SUBMITTED'
    except Exception as exc:
        if report['status'] == 'SUBMITTING':
            report['status'] = 'SUBMISSION_INCOMPLETE'
            for run in report['runs']:
                journal = Path(run['out_root']) / '.b3_recovery_submissions.json'
                if journal.exists():
                    plan_path = str(state / run['run_tag'] / 'execution_plan.json')
                    run['submission_records'] = [entry for entry in json.loads(journal.read_text())
                                                 if entry.get('execution_plan') == plan_path]
        else:
            report['status'] = 'BLOCKED'
        report['error'] = str(exc)
        print(f'{report["status"]}: {exc}', flush=True)
    finally:
        atomic_json(report_path, report)
        print('Relatório: ' + str(report_path), flush=True)
    return 2 if report['status'] in {'BLOCKED', 'SUBMISSION_INCOMPLETE'} else 0


def worker(args):
    plan_path = Path(args.plan)
    if sha(plan_path) != args.plan_sha or sha(args.indices) != args.indices_sha:
        raise ValueError('Plano/índices da retomada foram alterados.')
    plan = json.loads(plan_path.read_text())
    for name, expected in plan['tools_sha256'].items():
        if sha(Path(__file__).resolve().parent / name) != expected:
            raise ValueError('Ferramenta de retomada alterada: ' + name)
    indices = json.loads(Path(args.indices).read_text())
    local_index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    if not 0 <= local_index < len(indices) or len(set(indices)) != len(indices):
        raise ValueError('Índices de array inválidos.')
    index = indices[local_index]
    if index not in plan['missing_indices']:
        raise ValueError('Tarefa não estava autorizada como faltante na auditoria.')
    snapshot = Path(plan['snapshot'])
    for name, expected in plan['snapshot_control_sha256'].items():
        if sha(snapshot / '.b3_recovery' / name) != expected:
            raise ValueError('Manifesto do snapshot alterado: ' + name)
    verify_snapshot(snapshot)
    verify_dataset(snapshot)
    out_root = Path(plan['out_root'])
    directory = out_root / GROUP
    if sha(directory / 'run_config.json') != plan['run_config_sha256'] or sha(directory / 'design_composite_trading.csv') != plan['design_sha256']:
        raise ValueError('Manifesto original alterado depois da auditoria.')
    saved = json.loads((directory / 'run_config.json').read_text())
    from b3_recovery_audit import audit_task
    runner = load_runner(snapshot)
    config = saved['config']
    tasks = runner.make_tasks(config, 'composto')
    if runner.digest(runner.provenance(config, saved['metadata'])) != saved['design_sha256']:
        raise ValueError('Snapshot não reproduz a assinatura original.')
    for key in ('dataset', 'calendar'):
        if sha(saved['metadata'][key]) != saved['metadata'][key + '_sha256']:
            raise ValueError('Dados originais foram alterados: ' + key)
    task = tasks[index]
    output = out_root / runner.output_path(task)
    output.mkdir(parents=True, exist_ok=True)
    with locked(output / '.b3_recovery_task.lock'):
        status = audit_task(runner, config, task, out_root, saved)
        if status['status'] == 'complete':
            print(f'TASK={index}: já concluída e verificada; nenhum treino.', flush=True)
            return
        if status['status'] != 'missing':
            raise RuntimeError('Tarefa bloqueada; preservar resultados: ' + json.dumps(status))
        environment = clean_env(snapshot)
        os.environ.clear()
        os.environ.update(environment)
        os.environ['RESULT_ROOT'] = plan['result_root']
        os.environ['OUT_ROOT'] = str(out_root)
        if not plan['resources']['gres']:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            os.environ[key] = str(plan['resources']['cpus'])
        runner.worker(config, tasks, out_root, index)
        verify_snapshot(snapshot)
        verify_dataset(snapshot)
        final = audit_task(runner, config, task, out_root, saved)
        if final['status'] != 'complete':
            raise RuntimeError('A saída produzida não passou na auditoria: ' + json.dumps(final))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for action in ('audit', 'resume'):
        command = sub.add_parser(action)
        command.add_argument('--repo', required=True)
        command.add_argument('--state-dir', required=True)
        command.add_argument('--out-base', default='/snfs2/igor.viveiros/previsoes')
        command.add_argument('--result-base', default='/snfs2/igor.viveiros/result')
        command.add_argument('--run-tag', action='append')
    command = sub.add_parser('_audit')
    for name in ('snapshot', 'out-root', 'report'):
        command.add_argument('--' + name, required=True)
    command = sub.add_parser('worker')
    for name in ('plan', 'plan-sha', 'indices', 'indices-sha'):
        command.add_argument('--' + name, required=True)
    args = parser.parse_args()
    require_compute()
    if args.action in {'audit', 'resume'}:
        return coordinator(args)
    if args.action == '_audit':
        audit_child(args)
    else:
        worker(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
