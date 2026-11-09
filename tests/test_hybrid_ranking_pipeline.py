"""Mixture contract from environment through producer and frozen CPU consumer."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import b3_experiment as producer
from backtest_composto import io as consumer
from backtest_composto.pipeline import with_meta
from ts_benchmark.baselines.ranking_mixture import validate_rank_mix_config
from test_backtest_composto_io import make_composite_fixture, discover, rewrite_registry
from test_global_ranking_backtest import publish_contract
from test_backtest_composto_pipeline import cli, prepare_args, read_parquet


def config_from_shell(**overrides):
    env = dict(os.environ, TFB_ROOT=str(ROOT), PYTHON_BIN=sys.executable,
               CROSS_LOSSES='rankmix rankmix_proximity',
               MODELS='Nonstationary_Transformer', LOOKBACKS='32', HORIZONS='1 25',
               TRADE_WINDOWS='1 5', LAMBDAS='0.5 0.8 1', DELTAS='0 0.1',
               RANK_MIX_WEIGHTS='0.25 0.5 0.25', RANK_MIX_MODE='regional',
               RANK_MIX_REGIONS='top bottom top', RANK_MIX_FRACTION='0.25')
    env.update(overrides)
    result = subprocess.run(['bash', str(ROOT / 'scripts/run_composite_trading_experiment.sh'),
                             '--print-config'], env=env, capture_output=True, text=True)
    return result


def test_editable_parameters_reach_config_tasks_and_model_without_delta_duplicates():
    result = config_from_shell()
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    assert config['loss']['loss_rank_mix_weights'] == [.25, .5, .25]
    assert config['loss']['loss_rank_mix_mode'] == 'regional'
    tasks = producer.make_tasks(config, 'composto')
    # Three valid H/K pairs, three positive lambdas, two loss variants.
    assert len(tasks) == 18 * len(config['train_percentages'])
    assert len(set(map(producer.output_path, tasks))) == len(tasks)
    for task in tasks:
        _, hyper, _, _ = producer.model_spec(config, task)
        assert task['cross_delta'] == 0
        assert task['rank_mix_weights'] == hyper['loss_rank_mix_weights'] == [.25, .5, .25]
        assert task['rank_mix_fraction'] == hyper['loss_rank_mix_fraction'] == .25
        assert ('loss_cross_mask' in hyper) == (task['loss'] == 'rankmix_proximity')


@pytest.mark.parametrize('weights', ['1 1 0', '0.5 0.5', 'nan 0 1', '-0.1 0.6 0.5'])
def test_invalid_weight_vectors_fail_before_submission(weights):
    result = config_from_shell(RANK_MIX_WEIGHTS=weights)
    assert result.returncode != 0
    assert 'rank_mix_weights' in result.stderr


def test_inactive_mixture_does_not_leak_into_original_or_temporal_hyperparameters():
    plain = config_from_shell(CROSS_LOSSES='listnet listnet_proximity')
    assert plain.returncode == 0, plain.stderr
    config = json.loads(plain.stdout)
    assert not any(key.startswith('loss_rank_mix_') for key in config['loss'])
    config['cross_losses'] = ['listnet', 'rankmix']
    config['loss'].update(loss_rank_mix_weights=[.25, .5, .25], loss_rank_mix_mode='regional')
    for mode in ['composto', 'temporal']:
        tasks = producer.make_tasks(config, mode)
        for task in tasks:
            _, hyper, _, _ = producer.model_spec(config, task)
            assert any(key.startswith('loss_rank_mix_') for key in hyper) == (task['loss'] == 'rankmix')


@pytest.mark.parametrize('key,new_value', [
    ('loss_rank_mix_weights', [.0, .5, .5]),
    ('loss_rank_mix_mode', 'global'),
    ('loss_rank_mix_regions', ['bottom', 'bottom', 'top']),
    ('loss_rank_mix_fraction', .4),
])
def test_direct_launcher_paths_distinguish_scientific_mixture_settings(key, new_value):
    config = json.loads(config_from_shell().stdout)
    original = producer.make_tasks(config, 'composto')[0]
    changed = copy.deepcopy(config)
    changed['loss'][key] = new_value
    revised = producer.make_tasks(changed, 'composto')[0]
    assert producer.output_path(original) != producer.output_path(revised)


@pytest.mark.parametrize('mode', ['global', 'regional'])
def test_producer_consumer_round_trip_mixed_registry_and_parquet_metadata(tmp_path, mode):
    f = make_composite_fixture(tmp_path, losses=('listnet',), available_indices=())
    f.config.update(cross_losses=['listnet', 'rankmix', 'rankmix_proximity'],
                    deltas=[0., .25, .5], lambdas=[.5, 1.])
    f.config['loss'].update(loss_rank_mix_mode=mode, loss_rank_mix_weights=[.25, .5, .25],
                           loss_rank_mix_regions=['top', 'bottom', 'top'],
                           loss_rank_mix_fraction=.25, loss_cross_mask=['cauchy', .025])
    publish_contract(f, 'composto')
    tasks, info = discover(f)
    assert info['expected_complete']
    assert len(tasks) == len(f.producer_tasks) == 18
    for task, original in zip(tasks, f.producer_tasks):
        assert task['relpath'] == str(producer.output_path(original))
        assert not task['delta_effective']
        if task['loss'] != 'listnet':
            assert task['rank_mix_mode'] == mode
            assert task['rank_mix_weights'] == [.25, .5, .25]
        else:
            assert 'rank_mix_mode' not in task
        task.update(id=0, task_signature='fixture')
    plan = dict(tasks=tasks, slippage_bps=0., cost_bps=0.)
    plain = with_meta(pd.DataFrame({'value': [1.]}), tasks[0], plan)
    mix = with_meta(pd.DataFrame({'value': [1.]}), tasks[-1], plan)
    assert plain.schema == mix.schema
    assert plain['rank_mix_weights'].to_pylist() == [None]
    assert json.loads(mix['rank_mix_weights'][0].as_py()) == [.25, .5, .25]


@pytest.mark.parametrize('field', ['rank_mix_weights', 'rank_mix_mode', 'rank_mix_regions', 'rank_mix_fraction'])
def test_consumer_rejects_tampered_mixture_audit(tmp_path, field):
    f = make_composite_fixture(tmp_path, losses=('listnet',), available_indices=())
    f.config.update(cross_losses=['rankmix_proximity'], lambdas=[1.])
    f.config['loss'].update(loss_cross_mask=['cauchy', .025])
    publish_contract(f, 'composto')
    replacement = {'rank_mix_weights': '[0,0.5,0.5]', 'rank_mix_mode': 'regional',
                   'rank_mix_regions': '["bottom","bottom","top"]', 'rank_mix_fraction': '0.25'}
    def mutate(frame):
        frame.loc[0, field] = replacement[field]
        return frame
    rewrite_registry(f, mutate)
    with pytest.raises(ValueError, match='Mistura de ranking'):
        discover(f)


@pytest.mark.parametrize('settings', [
    {}, dict(weights=[.2, .3, .5], mode=' Regional ', regions=[' Top ', 'BOTTOM', 'top'], fraction=.25),
    dict(weights=[1., 0., 0.]), dict(weights=[0., .5, .5]),
])
def test_cpu_validator_agrees_with_training_contract(settings):
    expected = validate_rank_mix_config(**settings)
    config = {'loss': {'loss_rank_mix_' + key: value for key, value in settings.items()}}
    assert consumer._rank_mix_options(config) == expected


def test_provenance_records_new_sources_only_for_active_mixtures(tmp_path):
    f = make_composite_fixture(tmp_path, losses=('listnet',), available_indices=())
    old = producer.provenance(f.config, f.metadata)['code_sha256']
    f.config['cross_losses'] = ['rankmix']
    new = producer.provenance(f.config, f.metadata)['code_sha256']
    paths = {'ts_benchmark/baselines/ranking_mixture.py',
             'ts_benchmark/baselines/hybrid_ranking_losses.py'}
    assert paths.isdisjoint(old)
    assert paths <= set(new)


def test_frozen_cpu_execution_consolidates_mixtures_and_resumes(tmp_path):
    f = make_composite_fixture(tmp_path, losses=('listnet',), horizons=(3,),
                               available_indices=())
    f.config.update(cross_losses=['listnet', 'rankmix', 'rankmix_proximity'],
                    lambdas=[1.], trade_windows=[1])
    f.config['loss'].update(loss_rank_mix_mode='regional',
                           loss_rank_mix_weights=[.5, .5, 0.],
                           loss_cross_mask=['cauchy', .025])
    publish_contract(f, 'composto')
    run = tmp_path / 'run'
    cli(*prepare_args(f, run))
    cli('run', '--run-dir', run, '--workers', '1')
    result = read_parquet(run / 'resumos/ic_acumulado.parquet')
    mixture = result.loss.str.startswith('rankmix')
    assert mixture.any() and (~mixture).any()
    assert result.loc[mixture, 'rank_mix_mode'].eq('regional').all()
    assert result.loc[mixture, 'rank_mix_weights'].eq('[0.5,0.5,0.0]').all()
    assert result.loc[~mixture, 'rank_mix_mode'].isna().all()
    status = pd.read_csv(run / 'resumos/status_tarefas.csv')
    assert status.loc[status.loss.str.startswith('rankmix'), 'rank_mix_mode'].eq('regional').all()
    old = (run / 'contrato.json').read_bytes()
    cli('prepare', '--run-dir', run)
    cli('run', '--run-dir', run, '--workers', '1')
    assert (run / 'contrato.json').read_bytes() == old
    pd.testing.assert_frame_equal(result, read_parquet(run / 'resumos/ic_acumulado.parquet'))
