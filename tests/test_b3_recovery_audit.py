"""Recovery only reuses provable complete outputs; orphans never become work."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import b3_experiment as runner
from b3_experiment_io import export_prediction_windows, load_panel, prepare_dataset
from b3_recovery_audit import audit_run, audit_task, inventory_raw_results


@pytest.fixture
def experiment(tmp_path):
    source = tmp_path / 'returns.csv'
    pd.DataFrame({'date': range(1, 41), 'Z3': np.arange(40) / 1000,
                  'A4': -np.arange(40) / 2000}).to_csv(source, index=False)
    metadata = prepare_dataset(source, tmp_path / 'prepared.csv', source_data_kind='log_return')
    config = dict(train_percentages=[80], models=['simpleMLP'], lookbacks=[4],
                  horizons=[1, 2], trade_windows=[1], cross_losses=['mse'], lambdas=[.5],
                  deltas=[0], var_lags=1, model_defaults={'simpleMLP': {}}, training={},
                  loss={'loss_temporal': 'mse'},
                  strategy={'train_ratio_in_tv': .8, 'seed': 2026, 'stride': 1},
                  source_data_kind='log_return')
    tasks = runner.make_tasks(config, 'composto')
    out = tmp_path / 'outputs'
    plan = dict(config=config, metadata=metadata, git_commit='fixture-original-commit',
                design_sha256=runner.digest(runner.provenance(config, metadata)))
    state = dict(config=config, tasks=tasks, out=out, plan=plan, rows=[],
                 panel=load_panel(Path(metadata['dataset'])))
    write_manifests(state)
    return state


def write_manifests(state):
    tasks, out, plan = state['tasks'], state['out'], state['plan']
    directory = out / runner.group_path(tasks[0])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'run_config.json').write_text(json.dumps(plan))
    rows = [runner.audit_row(state['config'], task, plan['metadata'], out,
                             plan['design_sha256']) for task in tasks]
    for name in ('design_composite_trading.csv', 'desing_composite_trading.csv'):
        (directory / name).write_text(runner.csv_text(rows))
    with (directory / 'completed_tasks.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[*rows[0], 'slurm_job_id', 'parquet_sha256'])
        writer.writeheader()
        writer.writerows(state['rows'])


def finish(state, index=0, record=True):
    task, config, plan, panel = state['tasks'][index], state['config'], state['plan'], state['panel']
    cut = runner.split_counts(config, task, len(panel))['n_pretest']
    hashes = {}
    for horizon, path in runner.task_output_paths(config, task, state['out']).items():
        h = int(horizon)
        actual = np.stack([panel.iloc[i:i+h].to_numpy() for i in range(cut, len(panel)-h+1)])
        exported = export_prediction_windows(actual, actual + .001, panel, cut, h, path, step_offset=1)
        hashes[horizon] = exported['parquet_sha256']
    if record:
        row = runner.audit_row(config, task, plan['metadata'], state['out'], plan['design_sha256'])
        row.update(slurm_job_id='original-array_0', parquet_sha256=(json.dumps(hashes)
                   if task['modelo'] == 'VAR' else hashes[str(task['pred_len'])]))
        state['rows'].append(row)
        write_manifests(state)


def audit(state):
    return audit_run(runner, state['config'], state['tasks'], state['out'], state['plan'])


def output(state, index=0):
    return next(iter(runner.task_output_paths(state['config'], state['tasks'][index], state['out']).values()))


def test_complete_and_missing_are_distinguished_without_writing(experiment):
    finish(experiment)
    path = output(experiment) / 'part.parquet'
    before = path.read_bytes(), path.stat().st_mtime_ns
    report = audit(experiment)
    assert report['complete_indices'] == [0]
    assert report['missing_indices'] == [1]
    assert report['counts'] == dict(tasks=2, complete=1, missing=1, blocked=0)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert report['requires_quiescent_run'] is True


@pytest.mark.parametrize('evidence', ['part.parquet', 'part.csv', 'artifacts', 'dangling-link'])
def test_orphan_evidence_is_blocked_never_silently_retrained(experiment, evidence):
    path = output(experiment)
    path.mkdir(parents=True)
    if evidence == 'artifacts':
        (path / evidence).mkdir()
    elif evidence == 'dangling-link':
        (path / evidence).symlink_to(path / 'absent')
    else:
        (path / evidence).write_bytes(b'original evidence')
    result = audit(experiment)
    assert result['complete_indices'] == []
    assert result['missing_indices'] == [1]
    assert result['blocked'][0]['array_index'] == 0
    assert 'without original completion record' in result['blocked'][0]['reason']


def test_finished_parquet_without_completion_record_is_orphan(experiment):
    finish(experiment, record=False)
    assert audit(experiment)['missing_indices'] == [1]
    assert len(audit(experiment)['blocked']) == 1


def test_empty_locks_allow_missing_but_nonempty_lock_blocks(experiment):
    path = output(experiment)
    path.mkdir(parents=True)
    for name in ('.task.lock', '.b3_recovery_task.lock'):
        (path / name).touch()
    assert audit(experiment)['missing_indices'] == [0, 1]
    (path / '.task.lock').write_text('unexpected content')
    assert audit(experiment)['missing_indices'] == [1]


@pytest.mark.parametrize('damage', ['missing', 'hash', 'columns', 'steps', 'truncated', 'nonfinite'])
def test_completed_evidence_failures_block_retraining(experiment, damage):
    finish(experiment)
    path = output(experiment) / 'part.parquet'
    if damage == 'missing':
        path.unlink()
    elif damage == 'hash':
        path.write_bytes(path.read_bytes() + b'changed')
    else:
        frame = pd.read_parquet(path)
        if damage == 'columns':
            frame = frame[['arquivo', 'A4', 'Z3', 'h', 'origin_step', 'step']]
        elif damage == 'steps':
            frame.loc[0, 'step'] += 1
        elif damage == 'truncated':
            frame = frame.iloc[:-1]
        else:
            frame.loc[0, 'A4'] = np.nan
        frame.to_parquet(path, index=False)
        experiment['rows'][0]['parquet_sha256'] = runner.file_hash(path)
        write_manifests(experiment)
    report = audit(experiment)
    assert report['complete_indices'] == []
    assert report['missing_indices'] == [1]
    assert report['blocked'][0]['array_index'] == 0


def test_duplicate_completion_indices_block_entire_group(experiment):
    finish(experiment)
    experiment['rows'].append(dict(experiment['rows'][0]))
    write_manifests(experiment)
    report = audit(experiment)
    assert report['missing_indices'] == report['complete_indices'] == []
    assert report['counts']['blocked'] == 2


@pytest.mark.parametrize('damage', ['missing_design_row', 'wrong_output_path', 'source_mutation', 'provenance'])
def test_frozen_design_and_input_failures_block_all(experiment, damage):
    group = experiment['out'] / runner.group_path(experiment['tasks'][0])
    if damage in {'missing_design_row', 'wrong_output_path'}:
        for name in ('design_composite_trading.csv', 'desing_composite_trading.csv'):
            frame = pd.read_csv(group / name, keep_default_na=False)
            if damage == 'missing_design_row':
                frame = frame.iloc[:1]
            else:
                frame.loc[0, 'output_dir'] = '/unrelated/output'
            frame.to_csv(group / name, index=False)
    elif damage == 'source_mutation':
        source = Path(experiment['plan']['metadata']['source'])
        source.write_text(source.read_text() + '\n')
    else:
        experiment['plan']['design_sha256'] = 'f' * 64
    report = audit(experiment)
    assert report['counts']['blocked'] == 2
    assert report['complete_indices'] == report['missing_indices'] == []


def test_one_task_audit_reads_no_other_prediction_files(experiment, monkeypatch):
    finish(experiment, 0)
    finish(experiment, 1)
    original = pd.read_parquet
    paths = []
    def read(path, *args, **kwargs):
        paths.append(Path(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, 'read_parquet', read)
    result = audit_task(runner, experiment['config'], experiment['tasks'][0],
                        experiment['out'], experiment['plan'], panel=experiment['panel'])
    assert result['status'] == 'complete'
    assert paths == [output(experiment) / 'part.parquet']


def test_var_requires_all_horizons_even_if_max_horizon_completed(experiment):
    experiment['tasks'] = runner.make_tasks(experiment['config'], 'var')
    write_manifests(experiment)
    finish(experiment)
    assert audit(experiment)['complete_indices'] == [0]
    paths = runner.task_output_paths(experiment['config'], experiment['tasks'][0], experiment['out'])
    (paths['1'] / 'part.parquet').unlink()
    report = audit(experiment)
    assert report['missing_indices'] == report['complete_indices'] == []
    assert report['counts']['blocked'] == 1


def test_raw_inventory_preserves_evidence_without_adoption(tmp_path):
    path = tmp_path / 'task_survived' / 'raw.pkl'
    path.parent.mkdir()
    path.write_bytes(b'opaque prediction evidence')
    report = inventory_raw_results(tmp_path)
    assert report['count'] == 1
    assert report['files'] == [{'path': str(path), 'size': path.stat().st_size}]
    assert path.read_bytes() == b'opaque prediction evidence'
