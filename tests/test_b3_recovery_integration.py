"""Real frozen-runner subprocess audit and resume; Slurm alone is simulated."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import b3_experiment as runner
from b3_experiment_io import export_prediction_windows, load_panel, prepare_dataset


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def script(path, body):
    path.write_text('#!/usr/bin/env bash\nset -euo pipefail\n' + body)
    path.chmod(0o755)


def test_real_snapshot_audits_completed_output_and_resubmits_only_missing(tmp_path):
    pytest.importorskip('torch')
    pytest.importorskip('pyarrow')
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    source = tmp_path / 'economic_returns.csv'
    pd.DataFrame({'date': range(1, 61), 'Z3': np.arange(60) / 1000,
                  'A4': -np.arange(60) / 2000}).to_csv(source, index=False)
    prepared = tmp_path / 'prepared' / 'fixture_returns.csv'
    metadata = prepare_dataset(source, prepared, source_data_kind='log_return')
    runner.register_dataset(prepared, metadata)
    config = dict(train_percentages=[80], models=['simpleMLP'], lookbacks=[4],
                  horizons=[1, 2], trade_windows=[1], cross_losses=['mse'], lambdas=[.5],
                  deltas=[0], var_lags=1, model_defaults={'simpleMLP': {}},
                  training={'num_epochs': 1, 'patience': 1, 'lr': .001},
                  loss={'loss_temporal': 'mse'},
                  strategy={'train_ratio_in_tv': .8, 'seed': 2026, 'stride': 1},
                  source_data_kind='log_return')
    tasks = runner.make_tasks(config, 'composto')
    assert len(tasks) == 2
    out_base = tmp_path / 'predictions'
    out_root = out_base / 'fixture'
    group = out_root / runner.group_path(tasks[0])
    group.mkdir(parents=True)
    signature = runner.digest(runner.provenance(config, metadata))
    saved = dict(config=config, metadata=metadata, git_commit=commit,
                 design_sha256=signature, config_path='/original/frozen/launcher.sh')
    (group / 'run_config.json').write_text(json.dumps(saved, indent=2))
    design = [runner.audit_row(config, task, metadata, out_root, signature) for task in tasks]
    for name in ('design_composite_trading.csv', 'desing_composite_trading.csv'):
        (group / name).write_text(runner.csv_text(design))

    # Produce task 0 as an original completed output without fitting a model.
    panel = load_panel(prepared)
    cut = runner.split_counts(config, tasks[0], len(panel))['n_pretest']
    actual = np.stack([panel.iloc[i:i+1].to_numpy() for i in range(cut, len(panel))])
    output = out_root / runner.output_path(tasks[0])
    exported = export_prediction_windows(actual, actual + .001, panel, cut, 1,
                                          output, step_offset=1)
    completed = dict(design[0], slurm_job_id='original_0', parquet_sha256=exported['parquet_sha256'])
    with (group / 'completed_tasks.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(completed))
        writer.writeheader()
        writer.writerow(completed)
    original_paths = [source, prepared, Path(metadata['calendar']),
                      prepared.parent / 'FORECAST_META.csv', output / 'part.parquet',
                      *[group / name for name in ('run_config.json', 'completed_tasks.csv',
                         'design_composite_trading.csv', 'desing_composite_trading.csv')]]
    originals = {str(path): (path.read_bytes(), path.stat().st_mtime_ns) for path in original_paths}

    binary = tmp_path / 'fake_slurm'
    binary.mkdir()
    script(binary / 'squeue', 'exit 0\n')
    script(binary / 'scontrol', "printf 'MaxArraySize = 1000\\n'\n")
    script(binary / 'sbatch', 'printf "%s\\n" "$@" > "$RECOVERY_TEST_CAPTURE"\nprintf "1234;fixture\\n"\n')
    capture = tmp_path / 'sbatch_args'
    state = tmp_path / 'recovery_state'
    result_base = tmp_path / 'raw_results'
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('SBATCH_', 'RECOVERY_WORKER_', 'RECOVERY_MAX_'))}
    environment.update(PATH=str(binary) + os.pathsep + os.environ['PATH'],
                       SLURM_JOB_ID='999999', PYTHONDONTWRITEBYTECODE='1',
                       RECOVERY_TEST_CAPTURE=str(capture), RECOVERY_MAX_CONCURRENT='1')
    base = [sys.executable, str(ROOT / 'scripts/b3_recovery.py')]
    arguments = ['--repo', str(ROOT), '--state-dir', str(state), '--out-base', str(out_base),
                 '--result-base', str(result_base), '--run-tag', 'fixture']
    audited = subprocess.run([*base, 'audit', *arguments], env=environment,
                             text=True, capture_output=True, timeout=120)
    assert audited.returncode == 0, audited.stdout + audited.stderr
    report = json.loads((state / 'report.json').read_text())
    assert report['status'] == 'READY'
    run = report['runs'][0]
    assert run['audit']['complete_indices'] == [0]
    assert run['audit']['missing_indices'] == [1]
    assert run['audit']['blocked'] == []
    assert run['audit']['design_sha256'] == signature
    assert run['audit']['original_commit'] == commit
    assert not capture.exists()
    snapshot = Path(run['snapshot'])
    assert ROOT not in snapshot.parents
    identity = json.loads((snapshot / '.b3_recovery/snapshot.json').read_text())
    assert identity['original_commit'] == commit
    assert digest_file(snapshot / 'scripts/b3_experiment.py') == identity['files']['scripts/b3_experiment.py']['sha256']
    assert (snapshot / 'dataset/forecasting' / prepared.name).read_bytes() == prepared.read_bytes()
    assert (snapshot / 'scripts/b3_experiment.py').read_bytes() == subprocess.check_output(
        ['git', '-C', str(ROOT), 'show', commit + ':scripts/b3_experiment.py'])

    resumed = subprocess.run([*base, 'resume', *arguments], env=environment,
                             text=True, capture_output=True, timeout=120)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    report = json.loads((state / 'report.json').read_text())
    assert report['status'] == 'SUBMITTED'
    assert report['runs'][0]['jobs'][0]['job_id'] == '1234'
    assert report['runs'][0]['jobs'][0]['count'] == 1
    command = capture.read_text().splitlines()
    assert '--array=0-0%1' in command
    assert '--partition=gorgonas' in command
    assert not any(value.startswith(('--gres=', '--nodelist=')) for value in command)
    indices_path = Path(command[command.index('--indices') + 1])
    assert json.loads(indices_path.read_text()) == [1]
    assert command[command.index('--indices-sha') + 1] == digest_file(indices_path)
    execution_path = Path(command[command.index('--plan') + 1])
    execution = json.loads(execution_path.read_text())
    assert execution['missing_indices'] == [1]
    assert execution['snapshot'] == str(snapshot)
    assert command[command.index('--plan-sha') + 1] == digest_file(execution_path)
    assert execution['run_config_sha256'] == digest_file(group / 'run_config.json')
    assert execution['design_sha256'] == digest_file(group / 'design_composite_trading.csv')
    assert not (out_root / runner.output_path(tasks[1]) / 'part.parquet').exists()
    assert {str(path): (path.read_bytes(), path.stat().st_mtime_ns) for path in original_paths} == originals
