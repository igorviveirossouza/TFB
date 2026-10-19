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
from backtest_cpu.io import read_json
from ts_benchmark.baselines.proximity_losses import validate_cross_mask

LOSSES = ('hinge', 'ranknet', 'pairwise_mse', 'listnet',
          'hinge_proximity', 'ranknet_proximity', 'pairwise_mse_proximity')


@pytest.mark.parametrize('mask', [['uniform', 0], ['uniform', .025], ['gaussian', .025],
                                  ['gaussian', .02500000001]])
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
            assert [task['mask_type'], task['mask_a']] == mask
    # Actual CSV auditing for every loss, including originals with inactive mask.
    for loss in LOSSES:
        publish_composite_task(f, next(t['array_index'] for t in f.producer_tasks if t['loss'] == loss))
    tasks, info = discover(f)
    assert len(tasks) == len(LOSSES)
    assert info['expected'] == len(expected)


@pytest.mark.parametrize('mask', [None, [], ['uniform'], ['bad', 1], ['gaussian', 0],
                                  ['uniform', -1], ['uniform', True], ['uniform', '0.1'],
                                  ['uniform', float('inf')], ['gaussian', float('nan')]])
def test_mask_validation_matches_training(mask):
    with pytest.raises(ValueError):
        validate_cross_mask(mask)
    with pytest.raises(ValueError):
        _cross_mask({'loss': {'loss_cross_mask': mask}})


@pytest.mark.parametrize('field,value', [('mask_type', 'gaussian'), ('mask_a', '.05'),
                                        ('mask_effective', 'False'), ('cross_delta', '.1'),
                                        ('delta_effective', 'True')])
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
    rewrite_registry(f, lambda frame: frame.drop(columns=['mask_type', 'mask_a', 'mask_effective']))
    after, _ = discover(f)
    assert after == before


@pytest.mark.parametrize('mask', [['uniform', .025], ['gaussian', .025]])
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
            assert {'mask_type', 'mask_a', 'mask_effective'} <= set(table.columns)
            effective = task['loss'].endswith('_proximity')
            assert table.mask_effective.eq(effective).all()
            if effective:
                assert table.mask_type.eq(mask[0]).all()
                assert table.mask_a.eq(mask[1]).all()
            else:
                assert table.mask_type.isna().all() and table.mask_a.isna().all()
    statuses = pd.read_csv(run/'resumos/status_tarefas.csv')
    assert set(statuses.status) == {'COMPLETA'}
    assert statuses.mask_effective.sum() == 3
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
    pipeline.write_text(pipeline.read_text().replace('2026-10-19-composto-proximity-v2', 'previous-contract'))
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
