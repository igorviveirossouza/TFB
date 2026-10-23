"""Producer/consumer parity, frozen-contract resumption, and complete CPU outputs."""
import json
from pathlib import Path
import shutil
import sys

import pandas as pd
import pytest

from test_backtest_composto_io import (
    ROOT, discover, make_composite_fixture, publish_composite_task, rewrite_registry,
)
from test_backtest_composto_pipeline import (
    assert_consolidated_matches_tasks, cli, fingerprint, prepare_args, read_parquet,
)
from backtest_composto.io import _cross_mask, _expected_tasks
from backtest_composto.pipeline import VERSION
from backtest_cpu.io import read_json
from ts_benchmark.baselines.proximity_losses import validate_cross_mask

LOSSES = ('hinge', 'ranknet', 'pairwise_mse', 'listnet',
          'hinge_proximity', 'ranknet_proximity', 'pairwise_mse_proximity')


@pytest.mark.parametrize('state', ['empty', 'orphan', 'missing_registry', 'missing_parquet'])
def test_empty_plan_explains_publication_state_without_accepting_orphans(tmp_path, state):
    f = make_composite_fixture(tmp_path, losses=('hinge_proximity',),
                               cross_mask=['uniform', .025], available_indices=())
    group = f.root/'treino=50'/'experimento=composto'
    if state != 'empty':
        row = publish_composite_task(f, 0, register=state == 'missing_parquet')
        if state == 'missing_parquet':
            (Path(row['output_dir'])/'part.parquet').unlink()
        elif state == 'missing_registry':
            (group/'completed_tasks.csv').unlink()
    tasks, info = discover(f)
    assert tasks == []
    detail = info['groups'][0]
    assert detail['unregistered_parquets'] == int(state in ('orphan', 'missing_registry'))
    assert detail['registered_missing'] == int(state == 'missing_parquet')
    assert detail['losses'] == ['hinge_proximity']
    assert detail['cross_mask'] == ['uniform', .025]
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
              for p in f.root.rglob('*') if p.is_file()}
    run = tmp_path/'diagnostic_run'
    result = cli(*prepare_args(f, run), '--dry-run')
    assert not run.exists()
    assert str(f.root) in result.stdout and str(group/'completed_tasks.csv') in result.stdout
    assert 'hinge_proximity' in result.stdout and 'uniform' in result.stdout
    if state in ('orphan', 'missing_registry'):
        assert '1 Parquets da grade ainda sem registro' in result.stdout
    elif state == 'missing_parquet':
        assert 'registradas_sem_arquivo=1' in result.stdout
        assert 'Arquivo registrado ausente' in result.stdout
    else:
        assert 'consulte sacct' in result.stdout
    assert before == {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
                      for p in f.root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('mask', [['uniform', 0], ['uniform', .025], ['gaussian', .025],
                                  ['gaussian', .02500000001], ['cauchy', .025]] +
                         [[kernel, .025, mode] for kernel in ('uniform', 'gaussian', 'cauchy')
                          for mode in ('natural', 'complementary')])
def test_grid_and_paths_match_training_without_delta_duplicates(tmp_path, mask):
    f = make_composite_fixture(tmp_path, percentages=(50, 75), losses=LOSSES,
                               cross_mask=mask, available_indices=())
    expected = [task for pct in (50, 75) for task in _expected_tasks(f.config, pct)]
    assert len(expected) == len(f.producer_tasks)
    from b3_experiment import output_path
    for task, original in zip(expected, f.producer_tasks):
        assert task['training_array_index'] == original['array_index']
        assert task['relpath'] == str(output_path(original))
        if task['loss'].endswith('_proximity'):
            assert not task['delta_effective'] and task['cross_delta'] == 0
            assert [task['mask_type'], task['mask_a']] == mask[:2]
            assert task.get('mask_mode', 'natural') == (mask[2] if len(mask) == 3 else 'natural')
            assert ('mask_mode' in task) == (mask[-1] == 'complementary')
    # Actual CSV auditing for every loss, including originals with inactive mask.
    for loss in LOSSES:
        publish_composite_task(f, next(t['array_index'] for t in f.producer_tasks if t['loss'] == loss))
    tasks, info = discover(f)
    assert len(tasks) == len(LOSSES)
    assert info['expected'] == len(expected)


@pytest.mark.parametrize('mask', [None, [], ['uniform'], ['bad', 1], ['gaussian', 0],
                                  ['uniform', -1], ['uniform', True], ['uniform', '0.1'],
                                  ['uniform', float('inf')], ['gaussian', float('nan')],
                                  ['cauchy', 0], ['cauchy', -1, 'complementary'],
                                  ['gaussian', 0, 'complementary'], ['uniform', 1, 'bad'],
                                  ['uniform', 1, None], ['uniform', 1, True],
                                  ['uniform', 1, 'natural', 'extra']])
def test_mask_validation_matches_training(mask):
    with pytest.raises(ValueError):
        validate_cross_mask(mask)
    with pytest.raises(ValueError):
        _cross_mask({'loss': {'loss_cross_mask': mask}})


@pytest.mark.parametrize('mask,expected', [
    (['cauchy', .025], ('cauchy', .025)),
    ([' Cauchy ', .025, ' Natural '], ('cauchy', .025)),
    ([' GAUSSIAN ', .025, ' Complementary '], ('gaussian', .025, 'complementary')),
    (['uniform', 0, 'complementary'], ('uniform', 0., 'complementary')),
])
def test_valid_mask_canonicalization_matches_training(mask, expected):
    assert _cross_mask({'loss': {'loss_cross_mask': mask}}) == expected
    assert validate_cross_mask(mask) == expected


@pytest.mark.parametrize('kernel', ['uniform', 'gaussian', 'cauchy'])
def test_natural_identity_is_unchanged_and_complementary_paths_are_distinct(tmp_path, kernel):
    f = make_composite_fixture(tmp_path, losses=('hinge_proximity',),
                               cross_mask=[kernel, .025], available_indices=())
    old = _expected_tasks(f.config, 50)
    f.config['loss']['loss_cross_mask'] = [kernel, .025, 'natural']
    assert _expected_tasks(f.config, 50) == old
    f.config['loss']['loss_cross_mask'] = [kernel, .025, 'complementary']
    complementary = _expected_tasks(f.config, 50)
    assert {t['relpath'] for t in old}.isdisjoint({t['relpath'] for t in complementary})
    for natural, complement in zip(old, complementary):
        assert 'mask_mode' not in natural
        assert complement['mask_mode'] == 'complementary'
        assert complement['training_array_index'] == natural['training_array_index']
        assert '/mask_mode=complementary/' in complement['relpath']


@pytest.mark.parametrize('field,value', [('mask_type', 'gaussian'), ('mask_a', '.05'),
                                        ('mask_effective', 'False'), ('cross_delta', '.1'),
                                        ('delta_effective', 'True'), ('mask_mode', 'complementary')])
def test_proximity_audit_rejects_wrong_mask_or_delta(tmp_path, field, value):
    f = make_composite_fixture(tmp_path, losses=('hinge_proximity',), cross_mask=['uniform', .025])
    def change(frame):
        frame.loc[0, field] = value
        return frame
    rewrite_registry(f, change)
    with pytest.raises(ValueError):
        discover(f)


def test_legacy_csv_without_mask_columns_remains_supported(tmp_path):
    f = make_composite_fixture(tmp_path)
    before, _ = discover(f)
    rewrite_registry(f, lambda frame: frame.drop(columns=['mask_type', 'mask_a', 'mask_effective',
                                                         'mask_mode'], errors='ignore'))
    after, _ = discover(f)
    assert after == before


@pytest.mark.parametrize('kernel', ['uniform', 'gaussian'])
def test_legacy_proximity_audit_without_mode_retains_identity(tmp_path, kernel):
    f = make_composite_fixture(tmp_path, losses=('ranknet_proximity',), cross_mask=[kernel, .025])
    before, _ = discover(f)
    rewrite_registry(f, lambda frame: frame.drop(columns=['mask_mode']))
    after, _ = discover(f)
    assert after == before
    assert 'mask_mode' not in after[0]


@pytest.mark.parametrize('mutation', ['missing_mode', 'wrong_mode', 'missing_hyper_mode',
                                     'wrong_hyper_mode', 'wrong_kernel'])
def test_complementary_requires_consistent_audit_and_hyperparameters(tmp_path, mutation):
    f = make_composite_fixture(tmp_path, losses=('ranknet_proximity',),
                               cross_mask=['cauchy', .025, 'complementary'])
    def change(frame):
        if mutation == 'missing_mode':
            return frame.drop(columns=['mask_mode'])
        if mutation == 'wrong_mode':
            frame.loc[0, 'mask_mode'] = 'natural'
        else:
            hyper = json.loads(frame.loc[0, 'model_hyper_params'])
            if mutation == 'missing_hyper_mode':
                hyper['loss_cross_mask'].pop()
            elif mutation == 'wrong_hyper_mode':
                hyper['loss_cross_mask'][2] = 'natural'
            else:
                hyper['loss_cross_mask'][0] = 'gaussian'
            frame.loc[0, 'model_hyper_params'] = json.dumps(hyper)
        return frame
    rewrite_registry(f, change)
    with pytest.raises(ValueError):
        discover(f)


def test_original_loss_rejects_active_mask_mode(tmp_path):
    f = make_composite_fixture(tmp_path)
    def change(frame):
        frame.loc[0, 'mask_mode'] = 'natural'
        return frame
    rewrite_registry(f, change)
    with pytest.raises(ValueError, match='Mascara ativa'):
        discover(f)


@pytest.mark.parametrize('mask', [['uniform', .025], ['gaussian', .025],
                                  ['cauchy', .025, 'natural']] +
                         [[kernel, .025, 'complementary'] for kernel in ('uniform', 'gaussian', 'cauchy')])
def test_complete_backtest_preserves_mask_in_every_output_and_resumes(tmp_path, mask):
    f = make_composite_fixture(tmp_path, losses=LOSSES, cross_mask=mask, available_indices=())
    for loss in LOSSES:
        task = next(t for t in f.producer_tasks if t['loss'] == loss and t['pred_len'] == 3 and t['k'] == 1)
        publish_composite_task(f, task['array_index'])
    run = tmp_path/'run'
    cli(*prepare_args(f, run))
    cli('run', '--run-dir', run, '--workers', '2')
    plan = read_json(run/'plano.json')
    assert len(plan['tasks']) == len(LOSSES)
    assert_consolidated_matches_tasks(run, plan)
    for task in plan['tasks']:
        for path in (run/'tasks'/task['relpath']).glob('*.parquet'):
            table = read_parquet(path)
            assert {'mask_type', 'mask_a', 'mask_mode', 'mask_effective'} <= set(table.columns)
            effective = task['loss'].endswith('_proximity')
            assert table.mask_effective.eq(effective).all()
            if effective:
                assert table.mask_type.eq(mask[0]).all()
                assert table.mask_a.eq(mask[1]).all()
                assert table.mask_mode.eq(mask[2] if len(mask) == 3 else 'natural').all()
            else:
                assert table.mask_type.isna().all() and table.mask_a.isna().all() and table.mask_mode.isna().all()
    statuses = pd.read_csv(run/'resumos/status_tarefas.csv')
    assert set(statuses.status) == {'COMPLETA'}
    assert statuses.mask_effective.sum() == 3
    assert statuses.loc[statuses.mask_effective, 'mask_mode'].eq(
        mask[2] if len(mask) == 3 else 'natural').all()
    assert statuses.loc[~statuses.mask_effective, 'mask_mode'].isna().all()
    before = fingerprint(run, plan['tasks'])
    contract = (run/'contrato.json').read_bytes()
    # Only run identity is needed for resuming prepare; no source/options repeated.
    cli('prepare', '--run-dir', run)
    cli('run', '--run-dir', run, '--workers', '2')
    cli('consolidate', '--run-dir', run)
    assert fingerprint(run, plan['tasks']) == before
    assert (run/'contrato.json').read_bytes() == contract


def test_existing_contract_uses_its_own_code_after_checkout_update(tmp_path):
    f = make_composite_fixture(tmp_path, losses=('ranknet',), available_indices=(1,))
    checkout = tmp_path/'previous_scripts'
    checkout.mkdir()
    for name in ('backtest_cpu', 'backtest_composto'):
        shutil.copytree(ROOT/'scripts'/name, checkout/name, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(ROOT/'scripts/run_backtests_composto.py', checkout/'run_backtests_composto.py')
    pipeline = checkout/'backtest_composto/pipeline.py'
    pipeline.write_text(pipeline.read_text().replace(VERSION, 'previous-contract'))
    run = tmp_path/'old_run'
    cli(*prepare_args(f, run), entrypoint=checkout/'run_backtests_composto.py')
    cli('run', '--run-dir', run, '--workers', '1')
    plan = read_json(run/'plano.json')
    before = fingerprint(run, plan['tasks'])
    contract = (run/'contrato.json').read_bytes()
    assert json.loads(contract)['schema_version'] == 'previous-contract'
    cli('prepare', '--run-dir', run, '--dry-run')
    publish_composite_task(f, 0)
    cli('prepare', '--run-dir', run)
    cli('run', '--run-dir', run, '--workers', '1')
    cli('consolidate', '--run-dir', run)
    assert (run/'contrato.json').read_bytes() == contract
    assert fingerprint(run, plan['tasks']) == before
    assert read_json(run/'resumos/estado_execucao.json')['COMPLETA'] == 2
    # Explicit incompatible CLI changes still fail the original contract checks.
    assert cli('prepare', '--run-dir', run, '--cost-bps', '5', check=False).returncode == 1
    snapshot = run/'codigo/backtest_composto/io.py'
    snapshot.write_text(snapshot.read_text() + '\n# altered snapshot\n')
    result = cli('prepare', '--run-dir', run, '--dry-run', check=False)
    assert result.returncode == 1 and 'Codigo congelado' in result.stderr
    assert (run/'contrato.json').read_bytes() == contract
