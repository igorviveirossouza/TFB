"""Read-only evidence audit before recovering a frozen B3 experiment.

Only outputs with their original completion record, matching design, exact
Parquet hash and full rolling schema are reusable. An orphan is BLOCKED, never
silently classified as a missing task. The caller must first make the original
jobs quiescent; a task still training can legitimately have no output yet.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


def _rows(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f'Missing or duplicate CSV headers: {path}')
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f'Malformed CSV record: {path}')
    return reader.fieldnames, rows


def _index_rows(rows, expected_indices, label):
    indexed = {}
    for row in rows:
        raw = row.get('array_index', '')
        if not re.fullmatch(r'0|[1-9][0-9]*', raw):
            raise ValueError(f'{label}: invalid array_index={raw!r}')
        index = int(raw)
        if index not in expected_indices:
            raise ValueError(f'{label}: unexpected array_index={index}')
        if index in indexed:
            raise ValueError(f'{label}: duplicate array_index={index}')
        indexed[index] = row
    return indexed


def _csv_row(row):
    # This is csv.DictWriter's value conversion, used by the original runner.
    return {key: '' if value is None else str(value) for key, value in row.items()}


def _matching_row(actual, expected, label):
    mismatches = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatches:
        raise ValueError(f'{label}: fields differ from frozen design: {", ".join(mismatches)}')


def _snapshot(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def _validate_inputs(runner, config, plan, panel=None):
    if plan.get('config') != config:
        raise ValueError('Requested configuration differs from saved run_config.json')
    if not plan.get('git_commit'):
        raise ValueError('Saved plan lacks its original git_commit')
    signature = plan.get('design_sha256', '')
    if not re.fullmatch('[0-9a-f]{64}', signature):
        raise ValueError('Saved plan has an invalid design_sha256')
    metadata = plan['metadata']
    if runner.digest(runner.provenance(config, metadata)) != signature:
        raise ValueError('Candidate code/configuration does not reproduce original design_sha256')
    for key in ('source', 'dataset', 'calendar'):
        path = Path(metadata[key])
        if not path.is_file():
            raise ValueError(f'Frozen {key} file is absent: {path}')
        if runner.file_hash(path) != metadata[f'{key}_sha256']:
            raise ValueError(f'Frozen {key} hash differs: {path}')
    targets = metadata.get('targets')
    if (not isinstance(targets, list) or not targets or len(targets) != len(set(targets))
            or len(targets) != metadata['n_assets']):
        raise ValueError('Saved target order/asset count is invalid')
    if type(metadata.get('step_offset')) is not int or metadata['step_offset'] not in (1, 2):
        raise ValueError('Saved step_offset must be 1 or 2')
    # Read-only validation; importing this helper does not prepare/rewrite data.
    if panel is None:
        from b3_experiment_io import load_panel
        panel = load_panel(Path(metadata['dataset']))
    if len(panel) != metadata['n_rows'] or list(panel.columns) != targets:
        raise ValueError('Prepared dataset dimensions/order differ from saved metadata')


def _parquet(runner, path, expected_hash, metadata, cut, horizon):
    if not isinstance(expected_hash, str) or not re.fullmatch('[0-9a-f]{64}', expected_hash):
        raise ValueError(f'Invalid completion hash for {path}')
    if not path.is_file():
        raise ValueError(f'Completion record exists but Parquet is missing: {path}')
    before = _snapshot(path)
    if runner.file_hash(path) != expected_hash:
        raise ValueError(f'Parquet hash differs from completion record: {path}')
    frame = pd.read_parquet(path)
    targets = metadata['targets']
    count = metadata['n_rows'] - cut - horizon + 1
    if count < 1:
        raise ValueError('Frozen split has no valid prediction origins')
    expected_columns = ['arquivo', *targets, 'h', 'origin_step', 'step']
    if list(frame.columns) != expected_columns:
        raise ValueError(f'Parquet schema/asset order differs: {path}')
    if len(frame) != count * horizon:
        raise ValueError(f'Parquet row count differs: expected {count*horizon}, got {len(frame)}: {path}')
    if any(not pd.api.types.is_numeric_dtype(frame[column])
           or pd.api.types.is_bool_dtype(frame[column]) for column in targets):
        raise ValueError(f'Parquet predictions must be numeric: {path}')
    if not np.isfinite(frame[targets].to_numpy()).all():
        raise ValueError(f'Parquet has nonfinite predictions: {path}')
    if any(not pd.api.types.is_integer_dtype(frame[column]) for column in ('h', 'origin_step', 'step')):
        raise ValueError(f'Parquet step columns must be integers: {path}')
    first = cut + metadata['step_offset'] - 1
    origins = np.repeat(np.arange(first, first + count), horizon)
    steps = np.tile(np.arange(1, horizon + 1), count)
    names = np.repeat([f'janela_{i:06d}.csv' for i in range(count)], horizon)
    for column, values in (('arquivo', names), ('h', steps),
                           ('origin_step', origins), ('step', origins + steps)):
        if not np.array_equal(frame[column].to_numpy(), values):
            raise ValueError(f'Parquet {column} ordering/coverage differs: {path}')
    if _snapshot(path) != before or runner.file_hash(path) != expected_hash:
        raise ValueError(f'Parquet changed during the audit: {path}')


def _orphan_entries(paths):
    entries = []
    for directory in paths.values():
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            entries.append(str(directory))
        elif directory.is_dir():
            # Empty task locks are normal even when failure preceded training.
            # Every other entry, including an empty artifacts directory or an
            # atomic-write temporary file, needs explicit human investigation.
            entries.extend(str(path) for path in sorted(directory.iterdir())
                           if not (path.name in {'.task.lock', '.b3_recovery_task.lock'}
                                   and not path.is_symlink() and path.is_file()
                                   and path.stat().st_size == 0))
    return entries


def _group_records(runner, config, group_tasks, out_root, saved_plan):
    """Validate manifest evidence without reading any task prediction file."""
    directory = out_root / runner.group_path(group_tasks[0])
    group_indices = {task['array_index'] for task in group_tasks}
    disk_plan = json.loads((directory / 'run_config.json').read_text())
    for key in ('config', 'metadata', 'design_sha256', 'git_commit'):
        if disk_plan.get(key) != saved_plan.get(key):
            raise ValueError(f'Group run_config.json differs in {key}: {directory}')
    manifests = [directory / name for name in
                 ('design_composite_trading.csv', 'desing_composite_trading.csv')
                 if (directory / name).exists()]
    if not manifests:
        raise ValueError(f'Original design manifest is absent: {directory}')
    designs = []
    for path in manifests:
        _, rows = _rows(path)
        indexed = _index_rows(rows, group_indices, str(path))
        if set(indexed) != group_indices:
            raise ValueError(f'Design manifest task count/indices differ: {path}')
        designs.append(indexed)
    if any(design != designs[0] for design in designs[1:]):
        raise ValueError(f'Design manifest aliases disagree: {directory}')
    expected = {t['array_index']: _csv_row(runner.audit_row(
        config, t, saved_plan['metadata'], out_root, saved_plan['design_sha256']))
        for t in group_tasks}
    for index, row in expected.items():
        _matching_row(designs[0][index], row, f'Design task {index}')
    headers, rows = _rows(directory / 'completed_tasks.csv')
    if not {'array_index', 'parquet_sha256', 'design_sha256'}.issubset(headers):
        raise ValueError(f'Completion manifest lacks required columns: {directory}')
    return expected, _index_rows(rows, group_indices, 'completed_tasks.csv')


def _task_status(runner, config, task, out_root, saved_plan, expected, completed):
    index = task['array_index']
    result = {'array_index': index, 'status': 'blocked'}
    try:
        paths = runner.task_output_paths(config, task, out_root)
        if index not in completed:
            orphans = _orphan_entries(paths)
            if orphans:
                result.update(reason='Outputs exist without original completion record', evidence=orphans)
            else:
                result['status'] = 'missing'
            return result
        row = completed[index]
        _matching_row(row, expected[index], f'Completed task {index}')
        if task['modelo'] == 'VAR':
            hashes = json.loads(row['parquet_sha256'])
            if not isinstance(hashes, dict) or set(hashes) != set(paths):
                raise ValueError('VAR completion hashes do not cover exactly the requested horizons')
        else:
            hashes = {str(task['pred_len']): row['parquet_sha256']}
        metadata = saved_plan['metadata']
        cut = runner.split_counts(config, task, metadata['n_rows'])['n_pretest']
        for horizon, path in paths.items():
            _parquet(runner, path / 'part.parquet', hashes[horizon], metadata, cut, int(horizon))
        result['status'] = 'complete'
    except Exception as exc:
        result['reason'] = str(exc)
    return result


def audit_task(runner, config, task, out_root, saved_plan, *, panel=None):
    """Recheck one frozen task; never reads other tasks' prediction Parquets.

    The caller must serialize recovery of this task and ensure original jobs
    are quiescent. The full group design is checked, including duplicate or
    foreign completion indices, but only this task's outputs are read.
    """
    result = {'array_index': task['array_index'], 'status': 'blocked',
              'requires_quiescent_run': True}
    try:
        _validate_inputs(runner, config, saved_plan, panel=panel)
        group = runner.group_path(task)
        generated = [t for t in runner.make_tasks(config, task['experimento'])
                     if runner.group_path(t) == group]
        matching = [t for t in generated if t['array_index'] == task['array_index']]
        if matching != [task]:
            raise ValueError('Requested task differs from generated frozen design')
        expected, completed = _group_records(runner, config, generated, Path(out_root), saved_plan)
        result.update(_task_status(runner, config, task, Path(out_root), saved_plan, expected, completed))
        _validate_inputs(runner, config, saved_plan, panel=panel)
    except Exception as exc:
        result.update(status='blocked', reason=str(exc))
    return result


def audit_run(runner, config, tasks, out_root, saved_plan):
    """Audit the complete generated task list without changing any source file.

    ``runner`` is the original/pinned b3_experiment module. ``saved_plan`` is
    the original run_config.json object. For a configuration containing several
    split groups, every group's stored plan must agree with this object. The
    report deliberately blocks orphans rather than claiming they need retraining.
    """
    out_root = Path(out_root)
    indices = [task['array_index'] for task in tasks]
    if not tasks or any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != len(indices):
        raise ValueError('Pass a nonempty generated task list with unique integer indices')
    report = {'complete_indices': [], 'missing_indices': [], 'blocked': [],
              'requires_quiescent_run': True, 'design_sha256': saved_plan.get('design_sha256'),
              'out_root': str(out_root), 'policy': 'NO_ORPHAN_ADOPTION_NO_OVERWRITE'}

    def block(group_tasks, reason, **extra):
        report['blocked'].extend(dict(array_index=t['array_index'], reason=str(reason), **extra)
                                 for t in group_tasks)

    try:
        _validate_inputs(runner, config, saved_plan)
    except Exception as exc:
        block(tasks, exc, scope='run')
    else:
        metadata = saved_plan['metadata']
        signature = saved_plan['design_sha256']
        groups = {}
        for task in tasks:
            groups.setdefault(runner.group_path(task), []).append(task)
        for group, group_tasks in groups.items():
            try:
                expected, completed = _group_records(runner, config, group_tasks, out_root, saved_plan)
            except Exception as exc:
                block(group_tasks, exc, scope='group')
                continue
            for task in group_tasks:
                result = _task_status(runner, config, task, out_root, saved_plan, expected, completed)
                if result['status'] == 'blocked':
                    result.pop('status')
                    report['blocked'].append(dict(result, scope='task'))
                else:
                    report[result['status'] + '_indices'].append(task['array_index'])

        # A concurrent code/data rewrite invalidates the whole read-only snapshot.
        try:
            _validate_inputs(runner, config, saved_plan)
        except Exception as exc:
            report['complete_indices'].clear()
            report['missing_indices'].clear()
            report['blocked'].clear()
            block(tasks, f'Inputs/code changed during audit: {exc}', scope='run')

    report['complete_indices'].sort()
    report['missing_indices'].sort()
    report['blocked'].sort(key=lambda item: item['array_index'])
    report['counts'] = {'tasks': len(tasks), 'complete': len(report['complete_indices']),
                        'missing': len(report['missing_indices']), 'blocked': len(report['blocked'])}
    report['status'] = 'BLOCKED' if report['blocked'] else 'AUDITED'
    return report


def inventory_raw_results(result_root):
    """List surviving raw artifacts without decoding, adopting or deleting them.

    Result directories cannot be mapped to task identities just from their
    temporary names. The caller must retain this evidence and resolve lineage
    before deciding that any corresponding task needs another fit.
    """
    root = Path(result_root)
    if not root.exists():
        return {'root': str(root), 'exists': False, 'files': [], 'count': 0}
    if not root.is_dir():
        raise ValueError(f'Raw result root is not a directory: {root}')
    files = [{'path': str(path), 'size': path.stat().st_size}
             for path in sorted(root.rglob('*')) if path.is_file()]
    return {'root': str(root), 'exists': True, 'files': files, 'count': len(files),
            'policy': 'INVENTORY_ONLY_LINEAGE_NOT_ASSUMED'}
