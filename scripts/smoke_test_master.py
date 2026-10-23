#!/usr/bin/env python3
"""Bounded integration test on the exported panel; never an experiment result.

Six fits (three loss families × H=1/H=5), two train/validation batches and two
forecast origins per fit. Run through run_master_experiment.sh on the cluster.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir', type=Path, required=True)
    parser.add_argument('--returns-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-assets', type=int, default=55)
    parser.add_argument('--expected-dates', type=int, default=3664)
    parser.add_argument('--lookback', type=int, default=32)
    args = parser.parse_args()
    if socket.gethostname().split('.')[0] == 'phocus4':
        parser.error('Use run_master_experiment.sh: phocus4 é somente acesso ao cluster.')

    import numpy as np
    import pandas as pd
    import torch
    from ts_benchmark.baselines.master import MASTER
    from ts_benchmark.data.master import load_master_bundle

    torch.set_num_threads(1)
    started = time.monotonic()
    report = dict(status='RUNNING', purpose='bounded_smoke_not_experiment',
                  limits=dict(epochs=1, train_batches=2, validation_batches=2,
                              forecast_origins=2), cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        bundle = load_master_bundle(args.bundle_dir, args.returns_path,
                                    expected_assets=args.expected_assets,
                                    expected_dates=args.expected_dates)
        # Same index used by b3_experiment_io.prepare_dataset; real dates remain
        # in the feature bundle and are aligned by the adapter's explicit map.
        panel = pd.DataFrame(bundle.targets, columns=bundle.assets,
                             index=pd.date_range('2000-01-02', periods=bundle.n_dates))
        cut = len(panel) * 80 // 100
        report.update(n_assets=bundle.n_assets, n_dates=bundle.n_dates,
                      provenance=bundle.provenance)
        for horizon in (1, 5):
            for family, cross in [('temporal', 'mse'), ('composto', 'ranknet'),
                                   ('composto_proximity', 'ranknet_proximity')]:
                config = dict(master_bundle_dir=str(args.bundle_dir.resolve()),
                    master_returns_path=str(args.returns_path.resolve()),
                    master_expected_assets=args.expected_assets,
                    master_expected_dates=args.expected_dates,
                    seq_len=args.lookback, horizon=horizon, pred_len=horizon,
                    d_model=16, t_heads=2, s_heads=2, dropout=0.0,
                    batch_size=2, num_epochs=1, patience=1, lr=0.0001,
                    norm=True, seed=2026, device='cpu',
                    master_max_train_batches=2, master_max_validation_batches=2,
                    loss='mse' if family == 'temporal' else 'composite_trading',
                    loss_temporal='mse', loss_cross=cross, loss_trade_window=horizon,
                    loss_cross_lambda=0.5, loss_data_kind='log_return',
                    loss_score_kind='simple_return', loss_cross_score_normalization='zscore',
                    loss_inverse_norm=True, loss_track_components=True,
                    loss_cross_mask=['gaussian', 0.025, 'natural'])
                model = MASTER(**config)
                model.forecast_fit(panel.iloc[:cut], train_ratio_in_tv=0.875)
                forecasts = np.stack([model.forecast(horizon, panel.iloc[:origin])
                                      for origin in (cut, cut + 1)])
                if forecasts.shape != (2, horizon, bundle.n_assets) or not np.isfinite(forecasts).all():
                    raise AssertionError('Previsões com forma incorreta ou não finitas')
                report['cases'].append(dict(family=family, horizon=horizon,
                    forecast_shape=list(forecasts.shape), finite=True,
                    best_epoch=model.audit_['best_epoch'],
                    validation_loss=model.audit_['best_validation_loss'],
                    limited_run=model.audit_['limited_run']))
                print(f'MASTER SMOKE: {family}; H={horizon}; shape={forecasts.shape}; OK', flush=True)
        report['status'] = 'PASSED'
    except Exception as exc:
        report.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(f'Relatório: {args.output}', flush=True)


if __name__ == '__main__':
    main()
