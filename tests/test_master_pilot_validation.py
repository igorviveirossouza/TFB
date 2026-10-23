"""Saved pilot evidence is checked without training or changing its outputs."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import b3_experiment as runner
from b3_experiment_io import export_prediction_windows, load_panel, prepare_dataset
import validate_master_pilot as validator


@pytest.fixture
def pilot(tmp_path):
    source = tmp_path / 'returns.csv'
    pd.DataFrame(dict(date=range(1, 81), Z3=np.sin(np.arange(80)) / 100,
                      A4=np.cos(np.arange(80)) / 100)).to_csv(source, index=False)
    metadata = prepare_dataset(source, tmp_path / 'prepared.csv', source_data_kind='log_return')
    panel = load_panel(Path(metadata['dataset']))
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'features.csv').write_text('frozen fixture features\n')
    config = dict(source_data_kind='log_return', models=['MASTER'],
                  train_percentages=[80], lookbacks=[32], horizons=[5], trade_windows=[1],
                  cross_losses=['ranknet', 'ranknet_proximity'], lambdas=[.5], deltas=[0],
                  var_lags=32,
                  model_defaults={'MASTER': dict(master_bundle_dir=str(bundle),
                      master_returns_path=str(source), master_asset_blocks=['retornos', 'ohlcv', 'atividade'],
                      master_use_market=True, batch_size=32, norm=True, seed=2026)},
                  training=dict(num_epochs=100, patience=10, lr=.0001),
                  loss=dict(loss_temporal='mse', loss_cross_mask=['gaussian', .025, 'natural']),
                  strategy=dict(train_ratio_in_tv=.875, stride=1, num_rollings=48000, seed=2026))
    root = tmp_path / 'predictions'
    state = dict(root=root, config=config, metadata=metadata, groups={}, paths=[])
    for mode in ('temporal', 'composto'):
        plan = dict(config=copy.deepcopy(config), metadata=metadata,
                    git_commit='original-pilot-fixture',
                    design_sha256=runner.digest(runner.provenance(config, metadata)))
        tasks = runner.make_tasks(config, mode)
        group = root / runner.group_path(tasks[0])
        group.mkdir(parents=True)
        (group / 'run_config.json').write_text(json.dumps(plan))
        design = [runner.audit_row(config, task, metadata, root, plan['design_sha256'])
                  for task in tasks]
        for name in ('design_composite_trading.csv', 'desing_composite_trading.csv'):
            (group / name).write_text(runner.csv_text(design))
        completed = []
        for task, row in zip(tasks, design):
            output = root / runner.output_path(task)
            split = runner.split_counts(config, task, len(panel))
            cut, horizon = split['n_pretest'], task['pred_len']
            actual = np.stack([panel.iloc[i:i+horizon].to_numpy()
                               for i in range(cut, len(panel)-horizon+1)])
            exported = export_prediction_windows(actual, actual + .001, panel, cut,
                                                 horizon, output, step_offset=1)
            completed.append(dict(row, slurm_job_id='original',
                                  parquet_sha256=exported['parquet_sha256']))
            windows = dict(train=split['n_train']-32-5+1,
                           validation=split['n_validation']-5+1)
            hyper = runner.model_spec(config, task)[1]
            audit = dict(status='TRAINED', limited_run=False, target_kind='economic_log_return',
                         config=hyper, split=dict(n_dates=len(panel), **{key: split[key]
                             for key in ('n_pretest', 'n_train', 'n_validation', 'n_test')}),
                         window_counts=windows,
                         feature_normalization=dict(train_end=split['n_train'],
                             schema=dict(assets=metadata['targets'])),
                         target_normalization=dict(n_samples_seen=split['n_train']),
                         bundle_report=dict(assets=metadata['targets']),
                         best_epoch=1, best_validation_loss=.5,
                         history=[dict(epoch=epoch, lr=.0001,
                             train=dict(loss=.6, n_windows=windows['train']),
                             validation=dict(loss=.5 if epoch == 1 else .6,
                                             n_windows=windows['validation']))
                             for epoch in range(1, 12)])
            (output / 'master_training.json').write_text(json.dumps(audit))
            (output / 'master_checkpoint.pt').write_bytes(b'inference fixture; never loaded')
            state['paths'].append(output)
        (group / 'completed_tasks.csv').write_text(runner.csv_text(completed))
        state['groups'][mode] = group
    return state


def validate(state):
    return validator.validate_pilot(state['root'], expected_assets=2, expected_rows=80)


def mutate_training(state, mutate, index=0):
    path = state['paths'][index] / 'master_training.json'
    data = json.loads(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data))


def test_three_verified_forecasts_and_inputs_unchanged(pilot):
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
              for p in pilot['root'].rglob('*') if p.is_file()}
    report = validate(pilot)
    assert report['status'] == 'PASSED', report.get('error')
    assert len(report['verified_paths']) == 3
    assert [task['loss'] for task in report['tasks']] == ['mse', 'ranknet', 'ranknet_proximity']
    assert all(task['n_origins'] == 12 and task['n_rows'] == 60 for task in report['tasks'])
    after = {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
             for p in pilot['root'].rglob('*') if p.is_file()}
    assert before == after


@pytest.mark.parametrize('damage', [
    'limited', 'limit_config', 'partial_train', 'partial_validation', 'best_epoch',
    'best_loss', 'nonfinite', 'gap_epoch', 'truncated', 'wrong_split',
    'normalization', 'assets', 'hyperparameters', 'target_units', 'window_counts',
])
def test_training_report_damage_blocks_all_outputs(pilot, damage):
    def change(data):
        if damage == 'limited': data['limited_run'] = True
        elif damage == 'limit_config': data['config']['master_max_train_batches'] = 2
        elif damage == 'partial_train': data['history'][0]['train']['n_windows'] -= 1
        elif damage == 'partial_validation': data['history'][0]['validation']['n_windows'] -= 1
        elif damage == 'best_epoch': data['best_epoch'] = 2
        elif damage == 'best_loss': data['best_validation_loss'] = .1
        elif damage == 'nonfinite': data['history'][0]['train']['loss'] = float('nan')
        elif damage == 'gap_epoch': data['history'][1]['epoch'] = 3
        elif damage == 'truncated': data['history'].pop()
        elif damage == 'wrong_split': data['split']['n_train'] += 1
        elif damage == 'normalization': data['target_normalization']['n_samples_seen'] += 1
        elif damage == 'assets': data['bundle_report']['assets'].reverse()
        elif damage == 'hyperparameters': data['config']['seq_len'] = 128
        elif damage == 'target_units': data['target_kind'] = 'standardized_return'
        elif damage == 'window_counts': data['window_counts']['train'] -= 1
    mutate_training(pilot, change, index=2)
    report = validate(pilot)
    assert report['status'] == 'FAILED'
    assert report['verified_paths'] == report['tasks'] == []


@pytest.mark.parametrize('damage', ['empty_checkpoint', 'missing_checkpoint', 'parquet', 'completion'])
def test_missing_or_changed_artifact_is_never_accepted(pilot, damage):
    path = pilot['paths'][0]
    if damage == 'empty_checkpoint': (path / 'master_checkpoint.pt').write_bytes(b'')
    elif damage == 'missing_checkpoint': (path / 'master_checkpoint.pt').unlink()
    elif damage == 'parquet': (path / 'part.parquet').write_bytes(b'corrupt')
    else:
        completed = pilot['groups']['temporal'] / 'completed_tasks.csv'
        completed.write_text(completed.read_text().splitlines()[0] + '\n')
    report = validate(pilot)
    assert report['status'] == 'FAILED'
    assert report['verified_paths'] == []


def test_other_grid_is_rejected_before_accepting_artifacts(pilot):
    plan_path = pilot['groups']['composto'] / 'run_config.json'
    plan = json.loads(plan_path.read_text())
    plan['config']['loss']['loss_cross_mask'][-1] = 'complementary'
    plan_path.write_text(json.dumps(plan))
    assert 'natural gaussian' in validate(pilot)['error']


def test_wrong_expected_dimensions_are_rejected(pilot):
    assert validator.validate_pilot(pilot['root'])['status'] == 'FAILED'


def test_cli_failure_writes_report_and_returns_nonzero(tmp_path, capsys):
    path = tmp_path / 'report.json'
    assert validator.main(['--predictions', str(tmp_path / 'missing'), '--report', str(path)]) == 1
    assert json.loads(path.read_text())['status'] == 'FAILED'
    assert 'FAILED' in capsys.readouterr().out


def test_cli_refuses_report_within_original_predictions(pilot):
    with pytest.raises(SystemExit) as exc:
        validator.main(['--predictions', str(pilot['root']), '--report',
                        str(pilot['paths'][0] / 'master_training.json')])
    assert exc.value.code == 2


def test_cli_does_not_process_on_login_host(tmp_path, monkeypatch):
    monkeypatch.setattr(validator.socket, 'gethostname', lambda: 'phocus4.ufmg.br')
    with pytest.raises(SystemExit) as exc:
        validator.main(['--predictions', str(tmp_path), '--report', str(tmp_path / 'report.json')])
    assert exc.value.code == 2


def test_real_master_reports_and_exports_pass_without_mocked_producers(tmp_path):
    """Use the real adapter to guard against false blocking of cluster reports.

    The R-shaped fixture has 48 dates, so its lookback is 4 instead of 32.
    All other validation logic, report serialization and completion evidence
    use the same producers as the pilot. No audit or model is stubbed.
    """
    torch = pytest.importorskip('torch')
    pytest.importorskip('pyarrow')
    from test_master_data import ExportFixture
    from ts_benchmark.baselines.master import MASTER

    fixture = ExportFixture(tmp_path / 'input')
    metadata = prepare_dataset(fixture.returns, tmp_path / 'prepared.csv',
                               source_data_kind='log_return')
    panel = load_panel(Path(metadata['dataset']))
    config = dict(source_data_kind='log_return', models=['MASTER'],
                  train_percentages=[80], lookbacks=[4], horizons=[5], trade_windows=[1],
                  cross_losses=['ranknet', 'ranknet_proximity'], lambdas=[.5], deltas=[0],
                  var_lags=32, model_defaults={'MASTER': dict(
                      master_bundle_dir=str(fixture.directory), master_returns_path=str(fixture.returns),
                      master_expected_assets=3, master_expected_dates=48,
                      master_asset_blocks=['retornos', 'ohlcv', 'atividade'],
                      master_use_market=True, batch_size=32, d_model=8, t_heads=2,
                      s_heads=2, dropout=.1, gate_beta=1, norm=True,
                      lradj='type1', device='cpu', seed=2026)},
                  training=dict(num_epochs=100, patience=10, lr=.0001),
                  loss=dict(loss_temporal='mse', loss_data_kind='log_return',
                      loss_score_kind='simple_return', loss_cross_score_normalization='zscore',
                      loss_inverse_norm=True, loss_track_components=True,
                      loss_cross_mask=['gaussian', .025, 'natural']),
                  strategy=dict(train_ratio_in_tv=.875, stride=1, num_rollings=48000, seed=2026))
    output_root = tmp_path / 'predictions'
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for mode in ('temporal', 'composto'):
            tasks = runner.make_tasks(config, mode)
            signature = runner.digest(runner.provenance(config, metadata))
            plan = dict(config=config, metadata=metadata,
                        git_commit='real-adapter-fixture', design_sha256=signature)
            group = output_root / runner.group_path(tasks[0])
            group.mkdir(parents=True)
            (group / 'run_config.json').write_text(json.dumps(plan))
            design = [runner.audit_row(config, task, metadata, output_root, signature)
                      for task in tasks]
            for name in ('design_composite_trading.csv', 'desing_composite_trading.csv'):
                (group / name).write_text(runner.csv_text(design))
            completed = []
            for task, row in zip(tasks, design):
                output = output_root / runner.output_path(task)
                # benchmark_command is the real producer of model kwargs,
                # including persistent audit/checkpoint locations.
                command = runner.benchmark_command(config, task, metadata['dataset'],
                    len(panel), tmp_path / 'temporary-results', model_artifact_dir=output)
                hyper = json.loads(command[command.index('--model-hyper-params') + 1])
                model = MASTER(**hyper)
                cut = runner.split_counts(config, task, len(panel))['n_pretest']
                model.forecast_fit(panel.iloc[:cut], train_ratio_in_tv=.875)
                origins = list(range(cut, len(panel) - task['pred_len'] + 1))
                actual = np.stack([panel.iloc[i:i+5].to_numpy() for i in origins])
                forecast = np.stack([model.forecast(5, panel.iloc[:i]) for i in origins])
                exported = export_prediction_windows(actual, forecast, panel, cut, 5,
                                                     output, step_offset=1)
                completed.append(dict(row, slurm_job_id='real-adapter',
                                      parquet_sha256=exported['parquet_sha256']))
                verified = validator._training_report(output, config, task, metadata)
                assert verified['n_origins'] == 6
                assert verified['n_rows'] == 30
                assert verified['epochs'] >= 11
            (group / 'completed_tasks.csv').write_text(runner.csv_text(completed))
            evidence = validator.audit_run(runner, config, tasks, output_root, plan)
            assert evidence['counts'] == dict(tasks=len(tasks), complete=len(tasks),
                                               missing=0, blocked=0), evidence
    finally:
        torch.set_num_threads(old_threads)
