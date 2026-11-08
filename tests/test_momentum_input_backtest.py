"""Momentum input remains a raw-forecast contract with audited feature identity."""
import copy
import csv
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import b3_experiment as producer
from backtest_composto.io import (discover_composite_tasks, _expected_tasks,
                                  _momentum_input_options, load_composite_predictions,
                                  load_composite_ranking_scores, momentum_snapshot)
from test_backtest_composto_io import (make_composite_fixture, publish_composite_task,
                                       rewrite_registry)
from test_backtest_composto_pipeline import cli, prepare_args, read_parquet


def input_fixture(tmp_path, mode='feature', lookback=3):
    f = make_composite_fixture(tmp_path, losses=('ranknet', 'ranknet_proximity'),
                               deltas=(0,), cross_mask=['cauchy', .025], available_indices=())
    f.config['models'] = ['Nonstationary_Transformer']
    f.config['model_defaults'] = {'Nonstationary_Transformer': {'batch_size': 16, 'norm': True}}
    f.config['loss'].update(momentum_input_mode=mode, momentum_input_lookback=lookback,
                           loss_data_kind='log_return')
    f.config['training']['validation_mode'] = 'ordered_sample_mean'
    f.producer_tasks = producer.make_tasks(f.config, 'composto')
    f.design_sha256 = producer.digest(producer.provenance(f.config, f.metadata))
    group = f.root/'treino=50'/'experimento=composto'
    (group/'run_config.json').write_text(json.dumps(dict(config=f.config, metadata=f.metadata,
                                                        design_sha256=f.design_sha256)))
    row = producer.audit_row(f.config, f.producer_tasks[0], f.metadata, f.root, f.design_sha256)
    row.update(slurm_job_id='', parquet_sha256='')
    with (group/'completed_tasks.csv').open('w', newline='') as stream:
        csv.DictWriter(stream, fieldnames=list(row)).writeheader()
    for i in range(len(f.producer_tasks)):
        publish_composite_task(f, i)
    return f


@pytest.mark.parametrize('mode', ['zero', 'feature'])
@pytest.mark.parametrize('lookback', [None, 3])
def test_input_discovery_matches_producer_without_sidecars(tmp_path, mode, lookback):
    f = input_fixture(tmp_path, mode, lookback)
    tasks, info = discover_composite_tasks(f.root, [50], f.market, f.identity['retornos_sha256'])
    assert info['expected_complete']
    assert len(tasks) == len(f.producer_tasks)
    for task, original in zip(tasks, f.producer_tasks):
        assert task['relpath'] == str(producer.output_path(original))
        assert task['momentum_input_mode'] == mode
        assert task['momentum_input_lookback'] == (lookback or task['lookback'])
        assert 'momentum_input=' + mode in task['relpath']
        assert 'momentum_mode' not in task
        folder = Path(task['path']).parent
        assert not (folder/'ranking_scores.parquet').exists()
        assert not (folder/'momentum_state.json').exists()
        origins, predictions, accumulated = load_composite_predictions(task, f.market)
        assert predictions.shape[0] == len(origins)
        assert momentum_snapshot(task, f.market) == {}
        assert load_composite_ranking_scores(task, f.market, origins) is None


def test_input_off_preserves_cpu_task_identities(tmp_path):
    f = make_composite_fixture(tmp_path, available_indices=())
    original = _expected_tasks(f.config, 50)
    changed = copy.deepcopy(f.config)
    changed['loss'].update(momentum_input_mode='off', momentum_input_lookback=512)
    assert _expected_tasks(changed, 50) == original


@pytest.mark.parametrize('key,value', [
    ('momentum_input_mode', 'on'), ('momentum_input_mode', []),
    ('momentum_input_lookback', 0), ('momentum_input_lookback', True),
    ('momentum_input_lookback', 1.5), ('momentum_input_lookback', '512'),
])
def test_invalid_feature_options_fail(tmp_path, key, value):
    f = make_composite_fixture(tmp_path, available_indices=())
    f.config['loss'][key] = value
    with pytest.raises(ValueError, match='momentum_input_'):
        _momentum_input_options(f.config)


@pytest.mark.parametrize('change', ['model', 'source', 'loss_kind', 'residual'])
def test_input_unsupported_contracts_fail(tmp_path, change):
    f = make_composite_fixture(tmp_path, available_indices=())
    f.config['models'] = ['Nonstationary_Transformer']
    f.config['loss']['momentum_input_mode'] = 'feature'
    if change == 'model':
        f.config['models'] = ['DUET']
    elif change == 'source':
        f.config['source_data_kind'] = 'price'
    elif change == 'loss_kind':
        f.config['loss']['loss_data_kind'] = 'price'
    else:
        f.config['loss']['momentum_mode'] = 'residual_output'
    with pytest.raises(ValueError, match='Momentum de entrada'):
        _momentum_input_options(f.config)


@pytest.mark.parametrize('mode', ['zero', 'feature'])
@pytest.mark.parametrize('field,value', [('momentum_input_mode', 'off'),
                                       ('momentum_input_lookback', '7')])
def test_input_row_tampering_fails(tmp_path, mode, field, value):
    f = input_fixture(tmp_path, mode)
    rewrite_registry(f, lambda rows: rows.assign(**{field: value}))
    with pytest.raises(ValueError, match='Momentum de entrada divergente'):
        discover_composite_tasks(f.root, [50], f.market, f.identity['retornos_sha256'])


def test_cpu_pipeline_persists_input_identity_and_raw_scores(tmp_path):
    f = input_fixture(tmp_path, 'feature')
    run = tmp_path/'run'
    cli(*prepare_args(f, run))
    cli('run', '--run-dir', run, '--workers', '1')
    cli('consolidate', '--run-dir', run)
    ic = read_parquet(run/'resumos'/'ic_acumulado.parquet')
    assert set(ic.momentum_input_mode) == {'feature'}
    assert set(ic.momentum_input_lookback) == {3}
    assert 'momentum_mode' not in ic
    status = pd.read_csv(run/'resumos'/'status_tarefas.csv')
    assert set(status.momentum_input_mode) == {'feature'}
    assert set(status.momentum_input_lookback) == {3}
    assert set(status.status) == {'COMPLETA'}
