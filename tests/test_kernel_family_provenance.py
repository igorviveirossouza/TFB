"""Kernel shape survives producer, immutable paths, CSV audits and CPU outputs."""
import copy
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
import b3_experiment as producer
from backtest_composto.io import _cross_mask, _expected_tasks
from ts_benchmark.baselines.proximity_losses import validate_cross_mask
from test_b3_proximity_experiment import config, shell_config
from test_backtest_composto_io import discover, make_composite_fixture, rewrite_registry
from test_backtest_composto_pipeline import cli, prepare_args, read_parquet
from backtest_cpu.io import read_json

MASKS = [
    ['student', .25, 'natural', 10],
    ['generalized_gaussian', .25, 'natural', 1],
    ['generalized_gaussian', .25, 'complementary', 4],
    ['compact', .25, 'natural', 2, 0],
    ['compact', .25, 'complementary', 2, 3],
    ['intermediate_band', .25, 'natural', 2],
    ['cosine', .25], ['laplace', .25], ['triangular', .25],
    ['epanechnikov', .25], ['biweight', .25], ['triweight', .25],
]


@pytest.mark.parametrize('mask', MASKS)
def test_producer_consumer_shapes_paths_and_hyperparameters(config, mask):
    config['cross_losses'] = ['ranknet', 'ranknet_proximity']
    config['loss']['loss_cross_mask'] = mask
    canonical = validate_cross_mask(mask)
    assert _cross_mask(config) == canonical
    original = producer.make_tasks(config, 'composto')
    expected = _expected_tasks(config, 80)
    assert len(original) == len(expected)
    for src, dst in zip(original, expected):
        assert str(producer.output_path(src)) == dst['relpath']
        if src['loss'] == 'ranknet':
            assert 'mask_shape' not in src
            assert 'loss_cross_mask' not in producer.model_spec(config, src)[1]
        else:
            assert src.get('mask_shape', []) == list(canonical[3:])
            assert src.get('mask_shape', []) == dst.get('mask_shape', [])
            assert producer.model_spec(config, src)[1]['loss_cross_mask'] == list(canonical)
            if len(canonical) > 3:
                assert '/shape='+'_'.join(repr(x) for x in canonical[3:])+'/' in dst['relpath']


@pytest.mark.parametrize('mask', MASKS)
def test_registry_accepts_real_shape_roundtrip(tmp_path, mask):
    f = make_composite_fixture(tmp_path, losses=('ranknet', 'ranknet_proximity'), cross_mask=mask)
    tasks, _ = discover(f)
    assert tasks
    for task in tasks:
        if task['loss'].endswith('_proximity'):
            assert task.get('mask_shape', []) == list(validate_cross_mask(mask)[3:])


@pytest.mark.parametrize('change', ['missing', 'wrong', 'extra', 'hyper'])
def test_changed_shape_cannot_be_misidentified_as_original_forecast(tmp_path, change):
    f = make_composite_fixture(tmp_path, losses=('ranknet_proximity',),
                               cross_mask=['compact', .25, 'natural', 2, 3])
    def mutate(df):
        if change == 'missing':
            return df.drop(columns='mask_shape')
        if change == 'hyper':
            value = json.loads(df.loc[0, 'model_hyper_params'])
            value['loss_cross_mask'][-1] = 2.
            df.loc[0, 'model_hyper_params'] = json.dumps(value)
        else:
            df.loc[0, 'mask_shape'] = '[2.0,2.0]' if change == 'wrong' else '[2.0,3.0,4.0]'
        return df
    rewrite_registry(f, mutate)
    with pytest.raises(ValueError):
        discover(f)


def test_parameter_changes_have_distinct_paths_but_identical_plain_control(config):
    config['cross_losses'] = ['ranknet', 'ranknet_proximity']
    config['loss']['loss_cross_mask'] = ['student', .25, 'natural', 1]
    first = producer.make_tasks(config, 'composto')
    config['loss']['loss_cross_mask'][-1] = 10
    second = producer.make_tasks(config, 'composto')
    for old, new in zip(first, second):
        if old['loss'] == 'ranknet':
            assert old == new
            assert producer.model_spec(config, old) == producer.model_spec(config, new)
        else:
            assert producer.output_path(old) != producer.output_path(new)


@pytest.mark.parametrize('mask', [ ['student', .25, 'natural', 10],
                                  ['compact', .25, 'natural', 2, 3]])
def test_cpu_parquets_keep_kernel_shape_and_resume(tmp_path, mask):
    f = make_composite_fixture(tmp_path, losses=('ranknet_proximity',), cross_mask=mask,
                               available_indices=(0,))
    run = tmp_path/'run'
    cli(*prepare_args(f, run))
    cli('run', '--run-dir', run, '--workers', '1')
    plan = read_json(run/'plano.json')
    encoded = json.dumps(list(validate_cross_mask(mask)[3:]), separators=(',', ':'))
    for task in plan['tasks']:
        assert task['mask_shape'] == list(validate_cross_mask(mask)[3:])
        for path in (run/'tasks'/task['relpath']).glob('*.parquet'):
            table = read_parquet(path)
            assert 'mask_shape' in table
            assert table.mask_shape.eq(encoded).all()
    status = pd.read_csv(run/'resumos/status_tarefas.csv')
    assert status.mask_shape.eq(encoded).all()
    old = (run/'contrato.json').read_bytes()
    cli('prepare', '--run-dir', run)
    cli('run', '--run-dir', run, '--workers', '1')
    assert (run/'contrato.json').read_bytes() == old


def test_branch_dependency_does_not_override_internal_array_chain(monkeypatch):
    monkeypatch.delenv('B3_AFTER_JOB', raising=False)
    assert not any(x.startswith('--dependency=') for x in producer.sbatch_command('composto', 0, 1))
    monkeypatch.setenv('B3_AFTER_JOB', '12345')
    assert '--dependency=afterany:12345' in producer.sbatch_command('composto', 0, 1)
    assert '--dependency=afterany:23456' in producer.sbatch_command('composto', 1, 1, '23456')
    monkeypatch.setenv('B3_AFTER_JOB', '12; rm bad')
    with pytest.raises(ValueError, match='ID Slurm numerico'):
        producer.sbatch_command('composto', 0, 1)
