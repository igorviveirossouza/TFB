#!/usr/bin/env python3
"""One reproducible design shared by the temporal, composite and VAR launchers.

No heavy dependency or filesystem mutation is needed for --dry-run / --task.
Workers consume the prepared design, verify its provenance, then run TFB v3.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = {'temporal': 'run_temporal_experiment.sh',
             'composto': 'run_composite_trading_experiment.sh',
             'var': 'run_var_experiment.sh'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_config():
    if os.environ.get('B3_INLINE_CONFIG'):
        path = Path(os.environ['B3_CONFIG_SOURCE']).resolve()
        config = json.loads(os.environ['B3_INLINE_CONFIG'])
    elif os.environ.get('B3_CONFIG'):
        # Explicit JSON inputs remain available for programmatic callers.
        path = Path(os.environ['B3_CONFIG']).resolve()
        config = json.loads(path.read_text())
    else:
        raise ValueError('Execute um dos três run*.sh; os hiperparâmetros ficam no próprio script.')
    for key in ('train_percentages', 'models', 'lookbacks', 'horizons',
                'trade_windows', 'cross_losses', 'lambdas', 'deltas'):
        values = config[key]
        if not values or len(values) != len(set(values)):
            raise ValueError(f'{key}: vetor vazio ou duplicado')
    for key in ('lookbacks', 'horizons', 'trade_windows'):
        if any(type(x) is not int or x < 1 for x in config[key]):
            raise ValueError(f'{key}: use inteiros positivos')
    if any(type(x) is not int or not 0 < x < 100 for x in config['train_percentages']):
        raise ValueError('train_percentages: inteiros entre 1 e 99')
    if set(config['models']) - set(config['model_defaults']):
        raise ValueError('Modelo sem defaults de arquitetura')
    if set(config['cross_losses']) - {'ranknet', 'hinge', 'listnet', 'pairwise_mse'}:
        raise ValueError('Loss cross não suportada nesta grade')
    if any(not 0 < x < 1 for x in config['lambdas']):
        raise ValueError('Lambda composto deve estar em (0,1)')
    if any(not math.isfinite(x) or x < 0 for x in config['deltas']):
        raise ValueError('Delta deve ser finito e não negativo')
    if config['strategy']['stride'] != 1:
        raise ValueError('Este experimento requer stride=1')
    if not 0 < config['strategy']['train_ratio_in_tv'] < 1:
        raise ValueError('A validação interna deve ser preservada')
    for key in ('num_epochs', 'patience', 'lr'):
        if not math.isfinite(config['training'][key]) or config['training'][key] <= 0:
            raise ValueError(f'{key} deve ser positivo e finito')
    return path, config


def make_tasks(config, mode):
    tasks = []
    for pct in config['train_percentages']:
        losses = config['cross_losses'] if mode == 'composto' else ['mse']
        lambdas = config['lambdas'] if mode == 'composto' else [0]
        deltas = config['deltas'] if mode == 'composto' else [0]
        models = config['models'] if mode != 'var' else ['VAR']
        lookbacks = config['lookbacks'] if mode != 'var' else [config['var_lags']]
        horizons = config['horizons'] if mode != 'var' else [max(config['horizons'])]
        for loss, lam, delta, model, lb, horizon in itertools.product(
                losses, lambdas, deltas, models, lookbacks, horizons):
            ks = [k for k in config['trade_windows'] if k <= horizon] if mode == 'composto' else [None]
            for k in ks:
                tasks.append(dict(array_index=len(tasks), treino=pct, experimento=mode,
                                  loss=loss, cross_lambda=lam, cross_delta=delta,
                                  modelo=model, lookback=lb, pred_len=horizon, k=k))
    return tasks


def group_path(task):
    return Path(f"treino={task['treino']}") / f"experimento={task['experimento']}"


def output_path(task):
    lam = format(task['cross_lambda'], 'g') if task['experimento'] == 'composto' else '00'
    path = group_path(task) / f"loss={task['loss']}" / f'lambda={lam}'
    if task['experimento'] == 'composto':
        path /= f"delta={task['cross_delta']:g}"
    path = path / f"modelo={task['modelo']}" / f"lookback={task['lookback']}" / f"pred_len={task['pred_len']}"
    if task['k'] is not None:
        path /= f"k_dir={task['k']}"
    return path


def task_output_paths(config, task, out_root):
    horizons = config['horizons'] if task['modelo'] == 'VAR' else [task['pred_len']]
    return {str(h): out_root / output_path(dict(task, pred_len=h)) for h in horizons}


def model_spec(config, task):
    name = task['modelo']
    if name == 'VAR':
        return 'self_impl.VAR_model', {'lags': config['var_lags']}, None, 'full'
    hyper = dict(config['model_defaults'][name], **config['training'])
    hyper.update(seq_len=task['lookback'], pred_len=task['pred_len'], horizon=task['pred_len'])
    # Ordinary MSE is exactly the composite's temporal term. Do not evaluate
    # unused cross scores at lambda=0 (0 * NaN would still contaminate the loss).
    hyper.update(config['loss'])
    hyper.update(loss='composite_trading' if task['experimento'] == 'composto' else config['loss']['loss_temporal'], loss_cross=task['loss'],
                 loss_trade_window=task['k'] or 1,
                 loss_cross_lambda=task['cross_lambda'], loss_cross_delta=task['cross_delta'])
    qualified = 'duet.DUET' if name == 'DUET' else f'time_series_library.{name}'
    adapter = None if name == 'DUET' else 'transformer_adapter'
    return qualified, hyper, adapter, 'efficient' if name == 'TimesNet' else 'full'


def split_counts(config, task, n_rows):
    # Integer arithmetic implements floor(N*pct/100), avoiding floating-point edges.
    cut = n_rows * task['treino'] // 100
    fit = cut if task['modelo'] == 'VAR' else int(cut * config['strategy']['train_ratio_in_tv'])
    return dict(n_rows=n_rows, n_pretest=cut, n_train=fit, n_validation=cut-fit,
                n_test=n_rows-cut, n_windows=n_rows-cut-task['pred_len']+1)


def strategy_args(config, task, n_rows):
    strategy = dict(config['strategy'])
    cut = split_counts(config, task, n_rows)['n_pretest']
    ratio = task['treino'] / 100
    # TFB uses int(tv_ratio*N); choose an interior point only on a roundoff edge.
    if int(ratio * n_rows) != cut:
        ratio = (cut + 0.5) / n_rows
    strategy.update(horizon=task['pred_len'], tv_ratio=ratio)
    if task['modelo'] == 'VAR':
        strategy['train_ratio_in_tv'] = 1.0
    return strategy


def atomic_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False, encoding='utf-8') as f:
        temporary = Path(f.name)
        f.write(content)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def csv_text(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def audit_row(config, task, metadata, out_root, signature):
    _, hyper, _, deterministic = model_spec(config, task)
    paths = task_output_paths(config, task, out_root)
    cut = split_counts(config, task, metadata['n_rows'])['n_pretest']
    cross_blocks = task['pred_len'] // task['k'] if task['k'] is not None else ''
    return dict(task, **split_counts(config, task, metadata['n_rows']),
                dataset='log_retornos', data_source=metadata['source'],
                source_sha256=metadata['source_sha256'], dataset_sha256=metadata['dataset_sha256'],
                n_assets=metadata['n_assets'], seed=config['strategy']['seed'],
                temporal_loss=config['loss']['loss_temporal'],
                delta_effective=task['experimento'] == 'composto' and task['loss'] != 'listnet',
                cross_blocks=cross_blocks,
                cross_steps=cross_blocks * task['k'] if task['k'] is not None else '',
                cross_tail_steps=task['pred_len'] % task['k'] if task['k'] is not None else '',
                forecast_horizons=json.dumps([int(h) for h in paths]),
                n_windows_by_horizon=json.dumps({h:metadata['n_rows']-cut-int(h)+1 for h in paths}),
                output_dirs=json.dumps({h:str(path) for h, path in paths.items()}),
                model_hyper_params=json.dumps(hyper, sort_keys=True),
                strategy_args=json.dumps(strategy_args(config, task, metadata['n_rows']), sort_keys=True),
                deterministic=deterministic, design_sha256=signature,
                output_dir=str(out_root / output_path(task)))


def validate_lengths(config, tasks, metadata):
    for task in tasks:
        split = split_counts(config, task, metadata['n_rows'])
        max_windows = metadata['n_rows'] - split['n_pretest'] - min(config['horizons']) + 1 if task['modelo'] == 'VAR' else split['n_windows']
        if split['n_windows'] < 1 or max_windows > config['strategy']['num_rollings']:
            raise ValueError('Teste sem janelas ou NUM_ROLLINGS insuficiente para cobrir todo o teste')
        if split['n_train'] < task['lookback'] + task['pred_len'] + 31:
            raise ValueError('Treino insuficiente para contexto, horizonte e batch de 32')
        if task['modelo'] != 'VAR' and split['n_validation'] < task['pred_len']:
            raise ValueError('Validação menor que o horizonte')
        if task['modelo'] == 'VAR':
            regressors = 1 + config['var_lags'] * metadata['n_assets']
            if split['n_train'] - config['var_lags'] <= regressors:
                raise ValueError(f"VAR({config['var_lags']}) não identificado: {split['n_train']-config['var_lags']} observações para {regressors} regressores por equação")


def register_dataset(dataset, metadata):
    # Register once, under the preparation lock; workers never race to add this file.
    import pandas as pd
    sys.path.insert(0, str(ROOT))
    from ts_benchmark.data.utils import load_series_info
    path = dataset.parent / 'FORECAST_META.csv'
    row = load_series_info(str(dataset))
    previous = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=row.keys())
    rows = [row]
    source = Path(metadata['source'])
    if source.parent == dataset.parent:
        # Source may be wide or use integer trading labels. It is not loaded by
        # this experiment, but registering its dimensions avoids TFB auto-scans.
        rows.append(dict(row, file_name=source.name, length=metadata['n_prices'],
                         freq='other', if_univariate=metadata['n_assets'] == 1))
    previous = previous.loc[~previous.file_name.isin([r['file_name'] for r in rows])]
    updated = pd.concat([previous, pd.DataFrame(rows)], ignore_index=True)
    atomic_text(path, updated.to_csv(index=False))


def provenance(config, metadata):
    code_files = ['scripts/b3_experiment.py', 'scripts/b3_experiment_io.py',
                  'scripts/run_benchmark_composite_trading_loss_v3.py',
                  'ts_benchmark/baselines/composite_trading_loss_v2.py',
                  'ts_benchmark/baselines/composite_trading_loss_v3.py',
                  'ts_benchmark/baselines/self_impl/VAR/VAR.py',
                  'ts_benchmark/baselines/deep_forecasting_model_base.py']
    return dict(config=config, source_sha256=metadata['source_sha256'],
                dataset_sha256=metadata['dataset_sha256'],
                code_sha256={p:file_hash(ROOT / p) for p in code_files})


def prepare(config_path, config, tasks, out_root):
    from b3_experiment_io import prepare_dataset
    source = Path(os.environ.get('DATASET_PATH', ROOT / 'dataset/forecasting/b3_daily_tfb_new.csv')).resolve()
    dataset = ROOT / 'dataset/forecasting/b3_log_returns_new.csv'
    with locked(dataset.parent / '.b3_new_prepare.lock'):
        metadata = prepare_dataset(source, dataset)
        validate_lengths(config, tasks, metadata)
        signature = digest(provenance(config, metadata))
        groups = sorted({group_path(t) for t in tasks})
        # Check every destination before writing any manifest.
        for group in groups:
            plan_path = out_root / group / 'run_config.json'
            if plan_path.exists() and json.loads(plan_path.read_text())['design_sha256'] != signature:
                raise ValueError(f'Desenho/dados/código diferentes em {plan_path}; use outro OUT_ROOT')
        register_dataset(dataset, metadata)
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        for group in groups:
            group_tasks = [t for t in tasks if group_path(t) == group]
            rows = [audit_row(config, t, metadata, out_root, signature) for t in group_tasks]
            directory = out_root / group
            manifest = csv_text(rows)
            for name in ('design_composite_trading.csv', 'desing_composite_trading.csv'):
                atomic_text(directory / name, manifest)
            if not (directory / 'completed_tasks.csv').exists():
                atomic_text(directory / 'completed_tasks.csv', csv_text([dict(rows[0], slurm_job_id='', parquet_sha256='')]).splitlines()[0] + '\n')
            plan = dict(design_sha256=signature, config=config, config_path=str(config_path),
                        metadata=metadata, git_commit=commit)
            atomic_text(directory / 'run_config.json', json.dumps(plan, indent=2) + '\n')
    return metadata


def benchmark_command(config, task, dataset, n_rows, result_dir):
    model, hyper, adapter, deterministic = model_spec(config, task)
    command = [sys.executable, str(ROOT / 'scripts/run_benchmark_composite_trading_loss_v3.py'),
               '--config-path', 'rolling_forecast_config.json', '--data-name-list', Path(dataset).name,
               '--strategy-args', json.dumps(strategy_args(config, task, n_rows)),
               '--model-name', model, '--model-hyper-params', json.dumps(hyper),
               '--deterministic', deterministic, '--num-workers', '1', '--timeout', '60000',
               '--save-path', str(result_dir), '--save-true-pred', 'True']
    if adapter:
        command.extend(['--adapter', adapter])
    # Sequential TFB inherits Slurm's CUDA_VISIBLE_DEVICES. Passing --gpus 0
    # would overwrite an allocation on physical GPU 1 (or a GPU UUID).
    return command


def record_completed(directory, row):
    path = directory / 'completed_tasks.csv'
    with locked(directory / '.completed_tasks.lock'):
        with path.open(newline='') as stream:
            rows = [r for r in csv.DictReader(stream) if int(r['array_index']) != row['array_index']]
        rows.append(row)
        rows.sort(key=lambda r:int(r['array_index']))
        atomic_text(path, csv_text(rows))


def var_forecasts(config, panel, cut):
    """Fit once before the test and forecast all origins with fixed coefficients."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from statsmodels.tsa.tsatools import lagmat
    sys.path.insert(0, str(ROOT))
    from ts_benchmark.baselines.self_impl.VAR.VAR import VAR

    lags = config['var_lags']
    train = panel.iloc[:cut]
    values = StandardScaler().fit_transform(train.to_numpy())
    z = lagmat(values, maxlag=lags, trim='both', original='ex')
    z = np.column_stack([np.ones(len(z)), z])
    if len(z) <= z.shape[1] or np.linalg.matrix_rank(z) < z.shape[1]:
        raise ValueError(f'VAR({lags}): regressores sem posto completo ou graus de liberdade insuficientes')
    model = VAR(lags=lags)
    model.forecast_fit(train, train_ratio_in_tv=1.0)
    maximum = max(config['horizons'])
    forecasts = np.empty((len(panel)-cut, maximum, panel.shape[1]))
    for i, origin in enumerate(range(cut, len(panel))):
        # Only already-observed lags are inputs; the model is never refitted.
        forecasts[i] = model.forecast(maximum, panel.iloc[origin-lags:origin])
    return forecasts


def worker(config, tasks, out_root, index):
    from b3_experiment_io import export_predictions, export_prediction_windows, load_panel
    if index < 0 or index >= len(tasks):
        raise ValueError(f'Índice fora da grade: {index}')
    task = tasks[index]
    directory = out_root / group_path(task)
    plan = json.loads((directory / 'run_config.json').read_text())
    metadata = plan['metadata']
    signature = digest(provenance(config, metadata))
    if plan['design_sha256'] != signature:
        raise ValueError('Configuração/código alterados após preparar a grade')
    for key, hash_key in [('dataset','dataset_sha256'), ('calendar','calendar_sha256')]:
        if file_hash(metadata[key]) != metadata[hash_key]:
            raise ValueError(f'{key} alterado após preparar a grade')
    output = out_root / output_path(task)
    paths = task_output_paths(config, task, out_root)
    with locked(output / '.task.lock'):
        parquet = output / 'part.parquet'
        with locked(directory / '.completed_tasks.lock'):
            with (directory / 'completed_tasks.csv').open(newline='') as f:
                prior = next((r for r in csv.DictReader(f) if int(r['array_index']) == index), None)
        if task['modelo'] == 'VAR':
            try:
                prior_hashes = json.loads(prior['parquet_sha256']) if prior else {}
            except (ValueError, TypeError):
                prior_hashes = {}
            finished = bool(prior) and all((path/'part.parquet').exists() and
                       file_hash(path/'part.parquet') == prior_hashes.get(h) for h, path in paths.items())
        else:
            finished = bool(prior) and parquet.exists() and file_hash(parquet) == prior['parquet_sha256']
        if finished:
            print(f'Já concluído: {output}', flush=True)
            return
        panel = load_panel(Path(metadata['dataset']))
        split = split_counts(config, task, len(panel))
        if task['modelo'] == 'VAR':
            import numpy as np
            cut = split['n_pretest']
            print(f'TASK={index}: VAR({config["var_lags"]}), um ajuste; horizontes={config["horizons"]}', flush=True)
            forecasts = var_forecasts(config, panel, cut)
            for h, path in paths.items():
                horizon = int(h)
                count = len(panel) - cut - horizon + 1
                actual = np.stack([panel.iloc[origin:origin+horizon].to_numpy() for origin in range(cut, cut+count)])
                export_prediction_windows(actual, forecasts[:count, :horizon], panel, cut, horizon, path,
                                          keep_csv=os.environ.get('MANTER_CSV','FALSE').upper() == 'TRUE')
            parquet_hash = json.dumps({h:file_hash(path/'part.parquet') for h, path in paths.items()}, sort_keys=True)
        else:
            result_root = Path(os.environ.get('RESULT_ROOT', '/snfs2/igor.viveiros/result/b3_new'))
            result_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='task_', dir=result_root) as temporary:
                command = benchmark_command(config, task, metadata['dataset'], len(panel), Path(temporary))
                print(f'TASK={index}: {output}', flush=True)
                subprocess.run(command, cwd=ROOT, check=True)
                export_predictions(Path(temporary), panel, split['n_pretest'], task['pred_len'], output,
                                   keep_csv=os.environ.get('MANTER_CSV','FALSE').upper() == 'TRUE')
            parquet_hash = file_hash(parquet)
        row = audit_row(config, task, metadata, out_root, signature)
        row.update(slurm_job_id=os.environ.get('SLURM_JOB_ID',''), parquet_sha256=parquet_hash)
        record_completed(directory, row)
        print(f'OK: {len(paths)} arquivo(s) Parquet em {directory}', flush=True)


def array_chunks(n_tasks, chunk_size=1000):
    if chunk_size < 1:
        raise ValueError('ARRAY_CHUNK_SIZE deve ser positivo')
    return [(offset, min(chunk_size, n_tasks-offset)) for offset in range(0,n_tasks,chunk_size)]


def submit(mode, tasks, out_root, config_path):
    if shutil.which('sbatch') is None:
        raise RuntimeError('sbatch não encontrado; submeta os scripts no cluster')
    chunk_size = int(os.environ.get('ARRAY_CHUNK_SIZE','1000'))
    if shutil.which('scontrol'):
        result = subprocess.run(['scontrol','show','config'], capture_output=True, text=True, check=True)
        match = re.search(r'MaxArraySize\s*=\s*(\d+)', result.stdout)
        if match:
            chunk_size = min(chunk_size, int(match.group(1)))
    max_jobs = int(os.environ.get('MAX_GPU_JOBS','6'))
    if max_jobs < 1:
        raise ValueError('MAX_GPU_JOBS deve ser positivo')
    logs = Path(os.environ.get('LOG_ROOT', ROOT / 'logs/b3_new'))
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TFB_ROOT=str(ROOT), B3_CONFIG=str(config_path), OUT_ROOT=str(out_root))
    previous = None
    for offset, count in array_chunks(len(tasks), chunk_size):
        command = ['sbatch','--parsable','--export=ALL', '-p',os.environ.get('GPU_PARTITION','medusas_dev'),
                   '--time='+os.environ.get('GPU_TIME','08:00:00'),
                   f'--array=0-{count-1}%{max_jobs}', f'--job-name=b3_{mode}',
                   f'--output={logs}/{mode}_%A_%a.out', f'--error={logs}/{mode}_%A_%a.err']
        for variable, flag in [('NODELIST', '--nodelist'), ('CPUS_PER_TASK', '--cpus-per-task'), ('MEMORY', '--mem')]:
            if os.environ.get(variable):
                command.append(f'{flag}={os.environ[variable]}')
        if mode != 'var':
            command += ['--gres=gpu:1']
        if previous:
            command += [f'--dependency=afterany:{previous}']
        command += [str(ROOT/'scripts'/LAUNCHERS[mode]),'worker','--offset',str(offset)]
        result = subprocess.run(command, env=env, check=True, capture_output=True, text=True)
        previous = result.stdout.strip().split(';')[0]
        if not previous.isdigit():
            raise RuntimeError(f'Resposta inesperada de sbatch: {result.stdout}')
        print(f'Job {previous}: array local 0-{count-1}; tarefas globais {offset}-{offset+count-1}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=LAUNCHERS)
    parser.add_argument('action', nargs='?', choices=['submit','worker'], default='submit')
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--dry-run', action='store_true')
    choice.add_argument('--task', type=int, help='Inspeciona uma tarefa; não treina nem grava')
    choice.add_argument('--prepare', action='store_true', help='Prepara dados e auditoria sem submeter')
    choice.add_argument('--print-config', action='store_true', help='Exibe a configuração definida no .sh')
    parser.add_argument('--offset', type=int, default=0)
    args = parser.parse_args()
    config_path, config = load_config()
    if args.print_config:
        print(json.dumps(config, indent=2))
        return
    tasks = make_tasks(config, args.mode)
    out_root = Path(os.environ.get('OUT_ROOT','/snfs2/igor.viveiros/previsoes/novas_previsoes')).resolve()
    if args.task is not None:
        if not 0 <= args.task < len(tasks):
            parser.error('Tarefa fora da grade')
        task = tasks[args.task]
        print(json.dumps(dict(task, output_dir=str(out_root/output_path(task)),
                              forecast_horizons=config['horizons'] if task['modelo'] == 'VAR' else [task['pred_len']],
                              model_spec=model_spec(config,task)), indent=2))
        return
    if args.action == 'worker':
        index = args.offset + int(os.environ['SLURM_ARRAY_TASK_ID'])
        worker(config,tasks,out_root,index)
        return
    print(f'{args.mode}: {len(tasks)} tarefas; índices globais 0-{len(tasks)-1}', flush=True)
    print(f'Saída: {out_root/output_path(tasks[0])}', flush=True)
    cuts = ', '.join(f'{pct/100:g}' for pct in config['train_percentages'])
    print(f'Cortes: floor(N × p), p em {{{cuts}}}; N = pregões dos log-retornos.', flush=True)
    validation = 100 * (1 - config['strategy']['train_ratio_in_tv'])
    print('VAR usa todo o pré-teste.' if args.mode == 'var' else f'Validação interna: {validation:g}% do bloco pré-teste.', flush=True)
    if args.mode == 'var':
        print(f'Um ajuste por corte; os horizontes saem do mesmo VAR: {config["horizons"]}.', flush=True)
    if args.mode == 'composto':
        pairs = sum(k <= h for h in config['horizons'] for k in config['trade_windows'])
        factors = [len(config[key]) for key in ('train_percentages', 'models', 'lookbacks')]
        factors += [pairs] + [len(config[key]) for key in ('cross_losses', 'lambdas', 'deltas')]
        print('Grade: ' + ' × '.join(map(str, factors)) + f' = {len(tasks)} treinamentos.', flush=True)
        print(f'{pairs} pares H/K válidos; delta não altera ListNet.', flush=True)
    if args.dry_run:
        chunks = array_chunks(len(tasks), int(os.environ.get('ARRAY_CHUNK_SIZE','1000')))
        print(f'{len(chunks)} array(s), tamanho máximo {max(c for _,c in chunks)}; último: 0-{chunks[-1][1]-1}')
        return
    if not args.prepare and shutil.which('sbatch') is None:
        raise RuntimeError('sbatch não encontrado; use --dry-run ou --prepare fora do cluster')
    prepare(config_path,config,tasks,out_root)
    if not args.prepare:
        submit(args.mode,tasks,out_root,config_path)


if __name__ == '__main__':
    main()
