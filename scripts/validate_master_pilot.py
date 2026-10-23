#!/usr/bin/env python3
"""Read-only validation of the three completed MASTER pilot forecasts.

Never trains, resumes, rewrites manifests or loads serialized model weights.
Only the explicitly requested report is written by the command-line entrypoint.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import socket

import b3_experiment as runner
from b3_recovery_audit import audit_run


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fixed_design(config, mode):
    expected = dict(models=['MASTER'], train_percentages=[80], lookbacks=[32],
                    horizons=[5], trade_windows=[1], source_data_kind='log_return')
    for key, value in expected.items():
        _require(config.get(key) == value, f'Pilot configuration differs: {key}')
    _require(config['loss']['loss_temporal'] == 'mse', 'Pilot requires temporal MSE')
    _require(config['strategy']['train_ratio_in_tv'] == .875
             and config['strategy']['stride'] == 1, 'Pilot split/stride differs')
    _require(config['training']['num_epochs'] == 100
             and config['training']['patience'] == 10, 'Pilot training limits differ')
    if mode == 'composto':
        _require(config['cross_losses'] == ['ranknet', 'ranknet_proximity']
                 and config['lambdas'] == [.5] and config['deltas'] == [0],
                 'Pilot composite objectives differ')
        mask = config['loss']['loss_cross_mask']
        _require(mask in (['gaussian', .025], ['gaussian', .025, 'natural']),
                 'Pilot requires natural gaussian mask a=0.025')


def _training_report(path, config, task, metadata):
    audit = json.loads((path / 'master_training.json').read_text())
    _require(audit.get('status') == 'TRAINED' and audit.get('limited_run') is False,
             f'Incomplete or limited training: {path}')
    _require(audit.get('target_kind') == 'economic_log_return',
             f'Target units differ: {path}')
    checkpoint = path / 'master_checkpoint.pt'
    _require(checkpoint.is_file() and checkpoint.stat().st_size > 0,
             f'Missing or empty checkpoint: {checkpoint}')
    hyper = runner.model_spec(config, task)[1]
    trained_config = audit['config']
    for key, value in hyper.items():
        # The adapter resolves its two input paths to absolute locations.
        if key in {'master_bundle_dir', 'master_returns_path'}:
            value = str(Path(value).expanduser().resolve())
        _require(trained_config.get(key) == value,
                 f'Training configuration differs from manifest: {key}: {path}')
    for key in ('master_max_train_batches', 'master_max_validation_batches'):
        _require(trained_config.get(key) is None, f'Limited batches: {key}: {path}')

    split = runner.split_counts(config, task, metadata['n_rows'])
    expected_split = dict(n_dates=split['n_rows'], **{key: split[key] for key in
                         ('n_pretest', 'n_train', 'n_validation', 'n_test')})
    _require(audit.get('split') == expected_split, f'Training split differs: {path}')
    windows = dict(train=split['n_train'] - task['lookback'] - task['pred_len'] + 1,
                   validation=split['n_validation'] - task['pred_len'] + 1)
    _require(min(windows.values()) > 0 and audit.get('window_counts') == windows,
             f'Training window counts differ: {path}')
    _require(audit['feature_normalization']['train_end'] == split['n_train']
             and audit['target_normalization']['n_samples_seen'] == split['n_train'],
             f'Normalization does not use exactly the training prefix: {path}')
    assets = metadata['targets']
    _require(audit['bundle_report']['assets'] == assets
             and audit['feature_normalization']['schema']['assets'] == assets,
             f'Training asset order differs: {path}')

    history = audit.get('history')
    _require(isinstance(history, list) and 1 <= len(history) <= hyper['num_epochs'],
             f'Invalid training history length: {path}')
    for epoch, record in enumerate(history, 1):
        _require(type(record.get('epoch')) is int and record['epoch'] == epoch,
                 f'Training epochs are not consecutive: {path}')
        _require(isinstance(record.get('lr'), (int, float))
                 and math.isfinite(record['lr']) and record['lr'] > 0,
                 f'Invalid learning rate: {path}')
        for partition, count in windows.items():
            metrics = record.get(partition, {})
            _require(type(metrics.get('n_windows')) is int and metrics['n_windows'] == count,
                     f'Incomplete {partition} windows at epoch {epoch}: {path}')
            _require('loss' in metrics and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) for value in metrics.values()),
                f'Invalid {partition} metrics at epoch {epoch}: {path}')
    best = min(history, key=lambda record: record['validation']['loss'])
    _require(type(audit.get('best_epoch')) is int and audit['best_epoch'] == best['epoch']
             and audit.get('best_validation_loss') == best['validation']['loss'],
             f'Best epoch/loss differs from training history: {path}')
    # Early stopping may only end before the epoch budget after patience
    # consecutive non-improving epochs. This excludes truncated training logs.
    best_loss, stale = math.inf, 0
    for position, record in enumerate(history, 1):
        loss = record['validation']['loss']
        if loss < best_loss:
            best_loss, stale = loss, 0
        else:
            stale += 1
        _require(stale < hyper['patience'] or position == len(history),
                 f'History continued after early stopping: {path}')
    _require(len(history) == hyper['num_epochs'] or stale >= hyper['patience'],
             f'History ended before epoch budget or early stopping: {path}')
    return dict(experiment=task['experimento'], loss=task['loss'],
                parquet=str(path / 'part.parquet'), n_origins=split['n_windows'],
                n_rows=split['n_windows'] * task['pred_len'], n_assets=len(assets),
                first_origin_step=split['n_pretest'],
                last_origin_step=split['n_pretest'] + split['n_windows'] - 1,
                epochs=len(history), best_epoch=audit['best_epoch'],
                best_validation_loss=audit['best_validation_loss'])


def validate_pilot(predictions_root, expected_assets=55, expected_rows=3664):
    """Return verified paths only when all three saved pilot outputs pass."""
    root = Path(predictions_root).expanduser().resolve()
    report = dict(status='FAILED', predictions_root=str(root), verified_paths=[], tasks=[])
    try:
        _require(type(expected_assets) is int and expected_assets > 0
                 and type(expected_rows) is int and expected_rows > 0,
                 'Expected dimensions must be positive integers')
        common = None
        verified = []
        for mode, count in (('temporal', 1), ('composto', 2)):
            group = root / 'treino=80' / f'experimento={mode}'
            plan = json.loads((group / 'run_config.json').read_text())
            config, metadata = plan['config'], plan['metadata']
            _fixed_design(config, mode)
            _require(metadata['n_assets'] == expected_assets
                     and metadata['n_rows'] == expected_rows and metadata['step_offset'] == 1,
                     f'Unexpected data dimensions or step offset: {mode}')
            identity = {key: metadata[key] for key in (
                'source_sha256', 'dataset_sha256', 'calendar_sha256', 'targets')}
            _require(common is None or identity == common,
                     'Pilot objectives do not share the same data/calendar/asset order')
            common = identity
            tasks = runner.make_tasks(config, mode)
            _require(len(tasks) == count, f'Unexpected number of {mode} tasks')
            evidence = audit_run(runner, config, tasks, root, plan)
            if evidence['counts'] != dict(tasks=count, complete=count, missing=0, blocked=0):
                reason = (evidence['blocked'][0]['reason'] if evidence['blocked']
                          else f"Missing tasks: {evidence['missing_indices']}")
                raise ValueError(f'{mode}: forecast audit failed: {reason}')
            for task in tasks:
                verified.append(_training_report(root / runner.output_path(task),
                                                 config, task, metadata))
        report.update(status='PASSED', tasks=verified,
                      verified_paths=[task['parquet'] for task in verified],
                      n_assets=expected_assets, n_dates=expected_rows)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    if socket.gethostname().split('.')[0] == 'phocus4':
        parser.error('Submeta a validação pelo Slurm; phocus4 é somente acesso ao cluster.')
    report_path = args.report.expanduser().resolve()
    predictions = args.predictions.expanduser().resolve()
    if report_path == predictions or predictions in report_path.parents:
        parser.error('Salve o relatório fora da pasta das previsões originais.')
    report = validate_pilot(predictions)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    if report['status'] == 'PASSED':
        print(f"MASTER piloto: PASSED; 3 previsões verificadas; relatório: {report_path}", flush=True)
    else:
        print(f"MASTER piloto: FAILED; {report['error']}", flush=True)
        print(f'Relatório: {report_path}', flush=True)
    return 0 if report['status'] == 'PASSED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
