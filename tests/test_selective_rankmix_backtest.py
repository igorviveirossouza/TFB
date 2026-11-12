"""Frozen CPU backtests retain selective-Proximity identity and output metadata."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
import b3_experiment as producer
from backtest_composto import io as consumer
from backtest_composto.pipeline import with_meta
from ts_benchmark.baselines.ranking_mixture import validate_rank_mix_proximity_components
from test_backtest_composto_io import make_composite_fixture, discover, rewrite_registry
from test_backtest_composto_pipeline import cli, prepare_args, read_parquet
from test_global_ranking_backtest import publish_contract


CONFIG_KEY = 'loss_rank_mix_proximity_components'
TASK_KEY = 'rank_mix_proximity_components'


def selective_fixture(tmp_path, selector=('listnet',), horizons=(1, 3), protocol=None):
    fixture = make_composite_fixture(tmp_path, losses=('listnet',), horizons=horizons,
                                     available_indices=())
    fixture.config.update(cross_losses=['rankmix', 'rankmix_proximity'],
                          lambdas=[1.], trade_windows=[1])
    fixture.config['loss'].update(loss_cross_mask=['cauchy', .025],
                                  loss_rank_mix_weights=[.5, .5, 0.],
                                  loss_rank_mix_mode='global')
    if selector is not None:
        fixture.config['loss'][CONFIG_KEY] = list(selector)
    if protocol is not None:
        fixture.config['loss'].update(loss_cross_construction=protocol,
                                      loss_rank_windows=[1, 3] if protocol == 'origin_anchored_windows' else [])
    publish_contract(fixture, 'composto')
    return fixture


@pytest.mark.parametrize('selector', [None, ['listnet'], ['approxndcg', 'listnet'],
                                      ['ListMLE'], ['approxndcg', 'listmle', 'listnet']])
def test_cpu_selector_validator_matches_training_canonicalization(selector):
    config = {'cross_losses': ['rankmix_proximity'], 'loss': {CONFIG_KEY: selector}}
    expected = validate_rank_mix_proximity_components(selector)
    assert consumer._rank_mix_proximity_options(config) == (
        None if len(expected) == 3 else expected)


@pytest.mark.parametrize('selector', [[], ['listnet', 'listnet'], ['bad'], 'listnet', [True]])
def test_cpu_rejects_invalid_selector(selector):
    config = {'cross_losses': ['rankmix_proximity'], 'loss': {CONFIG_KEY: selector}}
    with pytest.raises(ValueError, match=CONFIG_KEY):
        consumer._rank_mix_proximity_options(config)


def test_producer_consumer_round_trip_selective_path_audit_and_parquet_schema(tmp_path):
    fixture = selective_fixture(tmp_path)
    tasks, discovery = discover(fixture)
    assert discovery['expected_complete']
    assert len(tasks) == len(fixture.producer_tasks) == 4
    tables = []
    for task, original in zip(tasks, fixture.producer_tasks):
        assert task['relpath'] == str(producer.output_path(original))
        assert task.get(TASK_KEY) == original.get(TASK_KEY)
        if task['loss'] == 'rankmix_proximity':
            assert 'proximity_on=listnet' in Path(task['relpath']).parts
        else:
            assert TASK_KEY not in task
        task.update(id=0, task_signature='fixture')
        tables.append(with_meta(pd.DataFrame({'value': [1.]}), task,
                                dict(tasks=tasks, slippage_bps=0., cost_bps=0.)))
    assert all(table.schema == tables[0].schema for table in tables)
    assert tables[0][TASK_KEY].to_pylist() == [None]
    assert tables[-1][TASK_KEY].to_pylist() == ['["listnet"]']


def test_legacy_and_all_explicit_keep_cpu_task_identity_and_old_schema(tmp_path):
    fixture = selective_fixture(tmp_path, selector=None)
    old = consumer._expected_tasks(fixture.config, 50)
    all_explicit = copy.deepcopy(fixture.config)
    all_explicit['loss'][CONFIG_KEY] = ['approxndcg', 'listmle', 'listnet']
    assert consumer._expected_tasks(all_explicit, 50) == old
    assert consumer._RANK_MIX_FIELDS == ('weights', 'mode', 'regions', 'fraction')
    tasks, _ = discover(fixture)
    for task in tasks:
        assert TASK_KEY not in task
        task.update(id=0, task_signature='fixture')
        table = with_meta(pd.DataFrame({'value': [1.]}), task,
                          dict(tasks=tasks, slippage_bps=0., cost_bps=0.))
        assert TASK_KEY not in table.column_names


@pytest.mark.parametrize('replacement', ['', '["listmle"]', 'not-json'])
def test_cpu_rejects_missing_or_tampered_selector_audit(tmp_path, replacement):
    fixture = selective_fixture(tmp_path)
    def mutate(frame):
        frame.loc[frame.loss.eq('rankmix_proximity'), TASK_KEY] = replacement
        return frame
    rewrite_registry(fixture, mutate)
    with pytest.raises(ValueError, match='Proximity seletiva'):
        discover(fixture)


def test_cpu_rejects_selector_on_original_control(tmp_path):
    fixture = selective_fixture(tmp_path)
    def mutate(frame):
        frame.loc[frame.loss.eq('rankmix'), TASK_KEY] = '["listnet"]'
        return frame
    rewrite_registry(fixture, mutate)
    with pytest.raises(ValueError, match='Proximity seletiva ativa fora'):
        discover(fixture)


def test_cpu_rejects_inconsistent_selected_model_hyperparameter(tmp_path):
    fixture = selective_fixture(tmp_path)
    def mutate(frame):
        index = frame.index[frame.loss.eq('rankmix_proximity')][0]
        hyper = json.loads(frame.loc[index, 'model_hyper_params'])
        hyper[CONFIG_KEY] = ['listmle']
        frame.loc[index, 'model_hyper_params'] = json.dumps(hyper)
        return frame
    rewrite_registry(fixture, mutate)
    with pytest.raises(ValueError, match='Hiperparametros'):
        discover(fixture)


@pytest.mark.parametrize('protocol', [None, 'blocks', 'origin_anchored_windows'])
def test_frozen_backtest_runs_resumes_and_retains_selector_in_statistics(tmp_path, protocol):
    fixture = selective_fixture(tmp_path, horizons=(3,), protocol=protocol)
    run = tmp_path / 'backtest'
    cli(*prepare_args(fixture, run))
    cli('run', '--run-dir', run, '--workers', '1')
    result = read_parquet(run / 'resumos/ic_acumulado.parquet')
    selected = result.loss.eq('rankmix_proximity')
    assert selected.any() and (~selected).any()
    assert result.loc[selected, TASK_KEY].eq('["listnet"]').all()
    assert result.loc[~selected, TASK_KEY].isna().all()
    status = pd.read_csv(run / 'resumos/status_tarefas.csv')
    assert status.status.eq('COMPLETA').all()
    assert status.loc[status.loss.eq('rankmix_proximity'), TASK_KEY].eq('["listnet"]').all()
    assert status.loc[status.loss.eq('rankmix'), TASK_KEY].isna().all()
    if protocol is not None:
        assert result.cross_construction.eq(protocol).all()
        assert status.cross_construction.eq(protocol).all()
        expected = [1, 3] if protocol == 'origin_anchored_windows' else []
        assert result.rank_windows.map(json.loads).apply(lambda value: value == expected).all()
        assert result.ranking_supervision_active.all()
    else:
        assert 'cross_construction' not in result.columns
    contract = (run / 'contrato.json').read_bytes()
    cli('prepare', '--run-dir', run)
    cli('run', '--run-dir', run, '--workers', '1')
    assert (run / 'contrato.json').read_bytes() == contract
    pd.testing.assert_frame_equal(result, read_parquet(run / 'resumos/ic_acumulado.parquet'))


def test_origin_windows_contract_is_paired_with_producer_and_truncated_by_h(tmp_path):
    fixture = selective_fixture(tmp_path, protocol='origin_anchored_windows')
    tasks, _ = discover(fixture)
    for task, original in zip(tasks, fixture.producer_tasks):
        expected = producer.ranking_protocol_settings(fixture.config, original)
        assert consumer._ranking_protocol_settings(fixture.config, task) == expected
        assert all(task[key] == value for key, value in expected.items())
        assert task['rank_windows'] == ([1] if task['pred_len'] == 1 else [1, 3])


@pytest.mark.parametrize('field,value', [('rank_windows', '[1]'),
                                       ('cross_construction', 'blocks'),
                                       ('ranking_supervision_active', 'False'),
                                       ('cross_blocks', '1')])
def test_origin_protocol_audit_rejects_tampering(tmp_path, field, value):
    fixture = selective_fixture(tmp_path, horizons=(3,), protocol='origin_anchored_windows')
    def mutate(frame):
        frame.loc[0, field] = value
        return frame
    rewrite_registry(fixture, mutate)
    with pytest.raises(ValueError, match='ranking|Blocos'):
        discover(fixture)
