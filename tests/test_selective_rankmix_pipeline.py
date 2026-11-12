"""Selective Proximity preserves legacy identities and freezes its own contract."""
from __future__ import annotations

import copy
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
import b3_experiment as producer


CONFIG_KEY = 'loss_rank_mix_proximity_components'
TASK_KEY = 'rank_mix_proximity_components'


def launcher_result(**changes):
    env = dict(os.environ)
    for key in ('B3_INLINE_CONFIG', 'B3_CONFIG', 'RANK_MIX_PROXIMITY_SELECTIVE',
                'RANK_MIX_PROXIMITY_COMPONENTS', 'CROSS_CONSTRUCTION', 'RANK_WINDOWS'):
        env.pop(key, None)
    env.update(TFB_ROOT=str(ROOT), PYTHON_BIN=sys.executable, VENV_PATH='/nonexistent',
               TRAIN_PERCENTAGES='90', MODELS='Nonstationary_Transformer', LOOKBACKS='32',
               HORIZONS='1 25', TRADE_WINDOWS='1', LAMBDAS='0.5 1', DELTAS='0',
               CROSS_LOSSES='rankmix rankmix_proximity', SOURCE_DATA_KIND='log_return',
               CROSS_MASK='cauchy 0.025 natural', RANK_MIX_WEIGHTS='0.5 0.5 0',
               RANK_MIX_MODE='global', RANK_MIX_REGIONS='top bottom top',
               RANK_MIX_FRACTION='0.3333333333333333')
    env.update(changes)
    return subprocess.run(
        ['bash', str(ROOT / 'scripts/run_composite_trading_experiment.sh'), '--print-config'],
        env=env, capture_output=True, text=True)


def launcher_config(**changes):
    result = launcher_result(**changes)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def metadata():
    return dict(n_rows=3664, n_assets=55, source='/fixture/returns.csv',
                step_offset=1, source_sha256='source', dataset_sha256='dataset',
                source_data_kind='log_return')


def test_absent_and_explicit_all_keep_legacy_config_task_and_hyperparameter_identity():
    legacy = launcher_config()
    explicit = launcher_config(RANK_MIX_PROXIMITY_SELECTIVE='1',
                               RANK_MIX_PROXIMITY_COMPONENTS='approxndcg listmle listnet')
    assert CONFIG_KEY not in legacy['loss']
    assert producer.RANK_MIX_FIELDS == ('weights', 'mode', 'regions', 'fraction')
    assert explicit == legacy
    assert producer.digest(explicit) == producer.digest(legacy)
    for mode in ('temporal', 'composto'):
        old_tasks = producer.make_tasks(legacy, mode)
        assert producer.make_tasks(explicit, mode) == old_tasks
        for task in old_tasks:
            assert TASK_KEY not in task
            assert 'proximity_on=' not in str(producer.output_path(task))
            assert CONFIG_KEY not in producer.model_spec(legacy, task)[1]
            assert TASK_KEY not in producer.audit_row(legacy, task, metadata(), Path('/out'), 'sig')


def test_a_selector_export_alone_does_not_change_legacy_launchers():
    assert launcher_config(RANK_MIX_PROXIMITY_COMPONENTS='listnet') == launcher_config()


def test_partial_selection_is_canonical_and_isolated_from_original_and_temporal():
    plain = launcher_config()
    selective = launcher_config(RANK_MIX_PROXIMITY_SELECTIVE='1',
                                RANK_MIX_PROXIMITY_COMPONENTS='ListMLE listnet')
    assert selective['loss'][CONFIG_KEY] == ['listnet', 'listmle']
    assert producer.digest(selective) != producer.digest(plain)
    for mode in ('temporal', 'composto'):
        old_tasks = producer.make_tasks(plain, mode)
        tasks = producer.make_tasks(selective, mode)
        assert len(tasks) == len(old_tasks)
        for old, task in zip(old_tasks, tasks):
            if task['loss'] == 'rankmix_proximity':
                assert task[TASK_KEY] == ['listnet', 'listmle']
                assert 'proximity_on=listnet_listmle' in producer.output_path(task).parts
                assert producer.output_path(task) != producer.output_path(old)
                assert producer.model_spec(selective, task)[1][CONFIG_KEY] == task[TASK_KEY]
            else:
                assert task == old
                assert producer.output_path(task) == producer.output_path(old)
                assert producer.model_spec(selective, task) == producer.model_spec(plain, old)
                assert CONFIG_KEY not in producer.model_spec(selective, task)[1]


def test_all_and_listnet_forecasts_have_distinct_task_identities_and_paths():
    all_config = launcher_config(CROSS_LOSSES='rankmix_proximity')
    only_listnet = launcher_config(CROSS_LOSSES='rankmix_proximity',
                                  RANK_MIX_PROXIMITY_SELECTIVE='1',
                                  RANK_MIX_PROXIMITY_COMPONENTS='listnet')
    all_tasks = producer.make_tasks(all_config, 'composto')
    selective_tasks = producer.make_tasks(only_listnet, 'composto')
    assert set(map(producer.output_path, all_tasks)).isdisjoint(
        map(producer.output_path, selective_tasks))
    for old, new in zip(all_tasks, selective_tasks):
        assert producer.digest(old) != producer.digest(new)
        assert new[TASK_KEY] == ['listnet']


def test_original_only_ignores_a_selector_even_in_direct_json(monkeypatch):
    config = launcher_config(CROSS_LOSSES='rankmix')
    config['loss'][CONFIG_KEY] = ['not-a-component']
    monkeypatch.setenv('B3_INLINE_CONFIG', json.dumps(config))
    monkeypatch.setenv('B3_CONFIG_SOURCE', str(ROOT / 'scripts/run_composite_trading_experiment.sh'))
    _, normalized = producer.load_config()
    assert CONFIG_KEY not in normalized['loss']
    for task in producer.make_tasks(config, 'composto'):
        assert TASK_KEY not in task
        assert CONFIG_KEY not in producer.model_spec(config, task)[1]


@pytest.mark.parametrize('selector', ['', 'listnet listnet', 'ranknet', 'listnet unknown'])
def test_invalid_selectors_fail_before_preparation(selector):
    result = launcher_result(RANK_MIX_PROXIMITY_SELECTIVE='1',
                             RANK_MIX_PROXIMITY_COMPONENTS=selector)
    assert result.returncode != 0
    assert CONFIG_KEY in result.stderr


def test_frozen_config_wins_over_changed_terminal_selection(monkeypatch):
    config = launcher_config(RANK_MIX_PROXIMITY_SELECTIVE='1',
                             RANK_MIX_PROXIMITY_COMPONENTS='listnet')
    monkeypatch.setenv('B3_INLINE_CONFIG', json.dumps(config))
    monkeypatch.setenv('B3_CONFIG_SOURCE', str(ROOT / 'scripts/run_composite_trading_experiment.sh'))
    monkeypatch.setenv('RANK_MIX_PROXIMITY_COMPONENTS', 'listmle')
    _, restored = producer.load_config()
    assert restored == config


def test_audit_and_success_sidecars_record_partial_selection(tmp_path):
    config = launcher_config(RANK_MIX_PROXIMITY_SELECTIVE='1',
                             RANK_MIX_PROXIMITY_COMPONENTS='listnet')
    tasks = producer.make_tasks(config, 'composto') + producer.make_tasks(config, 'temporal')
    rows = [producer.audit_row(config, task, metadata(), tmp_path, 'sig') for task in tasks]
    assert all(set(row) == set(rows[0]) for row in rows)
    recovered = list(csv.DictReader(io.StringIO(producer.csv_text(rows))))
    for task, row in zip(tasks, recovered):
        expected = ['listnet'] if task['loss'] == 'rankmix_proximity' else None
        assert (json.loads(row[TASK_KEY]) if row[TASK_KEY] else None) == expected
        hyper = json.loads(row['model_hyper_params'])
        assert hyper.get(CONFIG_KEY) == expected
        output = tmp_path / producer.output_path(task)
        producer.record_execution_metadata({str(task['pred_len']): output}, 'parquet', 'sig',
                                           extra={'cross_construction': 'blocks'}, task=task)
        saved = json.loads((output / 'execution_metadata.json').read_text())
        assert saved.get(TASK_KEY) == expected
        assert saved['parquet_sha256'] == 'parquet'
        assert saved['cross_construction'] == 'blocks'


def test_direct_configs_also_canonicalize_explicit_all_without_mutating_task_schema():
    legacy = launcher_config()
    explicit = copy.deepcopy(legacy)
    explicit['loss'][CONFIG_KEY] = ['listmle', 'approxndcg', 'listnet']
    assert producer.make_tasks(explicit, 'composto') == producer.make_tasks(legacy, 'composto')
    for task in producer.make_tasks(explicit, 'composto'):
        assert producer.model_spec(explicit, task) == producer.model_spec(legacy, task)
