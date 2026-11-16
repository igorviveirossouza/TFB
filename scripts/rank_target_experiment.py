#!/usr/bin/env python3
"""Isolated rank-target experiments; legacy launchers and datasets stay intact.

Plans require only the standard library. A prepared design freezes its inputs,
configuration and code hashes under its own prediction root. Workers never read
configuration from the submitting shell and refuse a changed checkout.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import b3_experiment as b3
from scripts import ordinal_experiment as ordinal
from scripts.ordinal_model_registry import discover_models, resolve_model

SCHEMA_VERSION = 'rank_target_v1'
ORDINAL_LOSSES = ('ordinal_ce', 'bucket_ranknet', 'bucket_hinge', 'bucket_listnet')
RANKING_LOSSES = ('ranknet', 'hinge', 'listnet', 'listmle', 'approxndcg', 'spearman', 'pairwise_mse')
SEMANTICS = dict(prediction_kind='rank_position', target_kind='rank',
                 score_direction='lower_is_better', rank_aggregation='mean')


def resolve_rank_model(requested, registry=None):
    """Also expose TSL classes present on disk but absent from __init__ exports.

    The legacy registry is deliberately unchanged. A fallback still resolves an
    actual class in its matching module; it never invents an architecture.
    """
    try:
        return resolve_model(requested, ROOT, registry)
    except ValueError:
        if not isinstance(requested, str) or not requested.isidentifier():
            raise
        source = ROOT/'ts_benchmark'/'baselines'/'time_series_library'/'models'/(requested+'.py')
        if not source.is_file() or not any(isinstance(node, ast.ClassDef) and node.name == requested
                for node in ast.parse(source.read_text()).body):
            raise
        _, spec = resolve_model(f'time_series_library.models.{requested}.{requested}', ROOT, registry)
        return requested, spec


def settings(environ=None):
    env = os.environ if environ is None else environ
    supplied = dict(env)
    supplied.setdefault('EXPERIMENT_TAG', 'rank_targets/rank_ordinal_t80_v1')
    family = supplied.setdefault('LOSS_FAMILY', 'ordinal')
    supplied.setdefault('INPUT_KIND', 'rank')
    supplied.setdefault('INPUT_NORMALIZATION', 'none')
    supplied.setdefault('CROSS_LOSSES_VALUES', ' '.join(ORDINAL_LOSSES if family == 'ordinal' else ('ranknet', 'hinge', 'listnet', 'listmle')))
    supplied.setdefault('K_EVAL_VALUES', supplied.get('RANK_WINDOWS_VALUES', '1 5 10 15 20 25'))
    out = ordinal.settings(supplied)
    out.update({key: supplied[key] for key in ('LOSS_FAMILY', 'INPUT_KIND', 'INPUT_NORMALIZATION')})
    for key in ('NS_RANKNET_ALPHA', 'NS_LISTNET_TAU', 'NS_LISTMLE_TAU', 'NS_APPROXNDCG_TAU', 'NS_SPEARMAN_TAU'):
        out[key] = supplied.get(key, '1')
    # These transformations are forbidden for rank targets, regardless of old
    # environment variables left in the shell by another experiment.
    out.update(SOURCE_DATA_KIND='log_return', NS_SCORE_KIND='rank_position',
               NS_CROSS_SCORE_NORMALIZATION='none', NS_LOSS_INVERSE_NORM='false')
    return out


def build_branches(s):
    family, input_kind, normalization = (s[k] for k in ('LOSS_FAMILY', 'INPUT_KIND', 'INPUT_NORMALIZATION'))
    if family not in {'ordinal', 'ranking'}:
        raise ValueError('LOSS_FAMILY: use ordinal ou ranking')
    if input_kind not in {'rank', 'log_return'}:
        raise ValueError('INPUT_KIND: use rank ou log_return; o alvo e sempre rank')
    if normalization not in {'none', 'standard', 'window', 'both'}:
        raise ValueError('INPUT_NORMALIZATION: use none, standard, window ou both')
    if s['NS_RANKING_PROTOCOL'] != 'blocks':
        raise ValueError('Este launcher usa NS_RANKING_PROTOCOL=blocks; escolha K_LOSS_VALUES explicitamente para treino em K dias. RANK_WINDOWS_VALUES e somente avaliacao.')
    tokens, number = ordinal.tokens, ordinal.number
    registry, specs = discover_models(ROOT), {}
    for requested in tokens(s['MODEL'], str, 'MODEL'):
        label, spec = resolve_rank_model(requested, registry)
        specs.setdefault(label, spec)
    architecture = {name: dict(ordinal.DEFAULT_MODELS.get(name, {})) for name in specs}
    overrides = json.loads(s['NS_MODEL_DEFAULTS_JSON'].strip() or '{}')
    if not isinstance(overrides, dict):
        raise ValueError('NS_MODEL_DEFAULTS_JSON deve ser objeto por modelo')
    reserved = {'batch_size', 'seq_len', 'pred_len', 'horizon', 'lr', 'num_epochs', 'patience',
                'validation_mode', 'rank_target_mode', 'input_kind', 'input_normalization'}
    for requested, values in overrides.items():
        label, _ = resolve_rank_model(requested, registry)
        if label not in architecture:
            continue
        if not isinstance(values, dict) or any(key in reserved or key.startswith(('loss', 'momentum')) for key in values):
            raise ValueError('Override de arquitetura invalido; treino e loss usam exports especificos: '+requested)
        architecture[label].update(values)
    for values in architecture.values():
        values.update(norm=False, batch_size=number(s, 'NS_BATCH_SIZE', int), parallel_strategy=None)
    temporal = tokens(s['TEMPORAL_LOSSES_VALUES'], str, 'TEMPORAL_LOSSES_VALUES')
    if set(temporal) - {'mse', 'mae', 'huber'}:
        raise ValueError('TEMPORAL_LOSSES_VALUES: use mse, mae ou huber')
    cross = tokens(s['CROSS_LOSSES_VALUES'], str, 'CROSS_LOSSES_VALUES')
    allowed = ORDINAL_LOSSES if family == 'ordinal' else RANKING_LOSSES
    if set(cross) - set(allowed):
        raise ValueError('CROSS_LOSSES_VALUES para '+family+': '+', '.join(allowed))
    buckets = tokens(s['BUCKET_COUNTS_VALUES'], int, 'BUCKET_COUNTS_VALUES', 2) if family == 'ordinal' else [None]
    lambdas = tokens(s['LAMBDAS_VALUES'], float, 'LAMBDAS_VALUES', 0, 1)
    horizons = tokens(s['HORIZON_VALUES'], int, 'HORIZON_VALUES', 1)
    windows = tokens(s['K_LOSS_VALUES'], int, 'K_LOSS_VALUES', 1)
    tokens(s['K_EVAL_VALUES'], int, 'K_EVAL_VALUES', 1)
    if any(not any(k <= h and h % k == 0 for k in windows) for h in horizons):
        raise ValueError('Cada H precisa de pelo menos um K_loss que divida H')
    for key in ('NS_MAX_CONCURRENT_JOBS', 'NS_ARRAY_CHUNK_SIZE', 'NS_CPUS_PER_TASK'):
        number(s, key, int)
    for key in ('NS_PARALLEL_BRANCHES', 'NS_RESUME_CHAIN_OLD_JOBS'):
        ordinal.boolean(s, key)
    if s['NS_AFTER_JOB'] and not s['NS_AFTER_JOB'].isdigit():
        raise ValueError('NS_AFTER_JOB deve ser ID Slurm numerico ou vazio')
    if s['NS_STRIDE'] != '1':
        raise ValueError('NS_STRIDE=1 e obrigatorio para preservar o contrato de origens das analises')
    ratio = number(s, 'NS_TRAIN_RATIO_IN_TV')
    if not 0 < ratio < 1:
        raise ValueError('NS_TRAIN_RATIO_IN_TV deve estar entre zero e um')
    margin = number(s, 'NS_HINGE_MARGIN', positive=False)
    if margin < 0:
        raise ValueError('NS_HINGE_MARGIN deve ser nao negativa')
    loss = dict(loss_temporal=temporal[0], loss_data_kind='rank', loss_score_kind='rank_position',
        loss_cross_score_normalization='none', loss_inverse_norm=False,
        loss_cross_construction='blocks', loss_rank_windows=[],
        loss_cross_scale=number(s, 'NS_CROSS_SCALE'), loss_huber_delta=number(s, 'NS_HUBER_DELTA'),
        loss_track_components=ordinal.boolean(s, 'NS_LOSS_TRACK_COMPONENTS'))
    training = dict(num_epochs=number(s, 'NS_NUM_EPOCHS', int), patience=number(s, 'NS_PATIENCE', int),
                    lr=number(s, 'NS_LEARNING_RATE'), validation_mode=s['NS_VALIDATION_MODE'])
    if s['NS_LRADJ']:
        training['lradj'] = s['NS_LRADJ']
    config = dict(schema_version=SCHEMA_VERSION, **SEMANTICS, loss_family=family,
        input_kind=input_kind, input_normalization=normalization, source_data_kind='log_return',
        train_percentages=tokens(s['TRAIN_VALUES'], int, 'TRAIN_VALUES', 1, 99),
        models=list(specs), model_specs=specs, model_defaults=architecture, training=training, loss=loss,
        lookbacks=tokens(s['LOOKBACKS_VALUES'], int, 'LOOKBACKS_VALUES', 1), horizons=horizons,
        trade_windows=windows,
        cross_losses=cross, lambdas=lambdas, deltas=[0],
        strategy=dict(train_ratio_in_tv=ratio, stride=1, num_rollings=number(s, 'NS_NUM_ROLLINGS', int), seed=0))
    options = dict(loss_ordinal_tau=number(s, 'NS_ORDINAL_TAU'), loss_bucket_tau=number(s, 'NS_BUCKET_TAU'),
                   loss_bucket_target_tau=number(s, 'NS_BUCKET_TARGET_TAU'), loss_hinge_margin=margin,
                   loss_ranknet_alpha=number(s, 'NS_RANKNET_ALPHA'), loss_listnet_tau=number(s, 'NS_LISTNET_TAU'),
                   loss_listmle_tau=number(s, 'NS_LISTMLE_TAU'), loss_approxndcg_tau=number(s, 'NS_APPROXNDCG_TAU'),
                   loss_spearman_tau=number(s, 'NS_SPEARMAN_TAU'))
    active = {'ordinal_ce': ['loss_ordinal_tau'], 'bucket_ranknet': ['loss_bucket_tau'],
              'bucket_hinge': ['loss_hinge_margin'], 'bucket_listnet': ['loss_bucket_tau', 'loss_bucket_target_tau'],
              'hinge': ['loss_hinge_margin'], 'ranknet': ['loss_ranknet_alpha'],
              'listnet': ['loss_listnet_tau'], 'listmle': ['loss_listmle_tau'],
              'approxndcg': ['loss_approxndcg_tau'], 'spearman': ['loss_spearman_tau'], 'pairwise_mse': []}
    branches = []
    for seed in tokens(s['SEEDS_VALUES'], int, 'SEEDS_VALUES', 0):
        for temporal_index, name in enumerate(temporal):
            base = copy.deepcopy(config)
            base['strategy']['seed'], base['loss']['loss_temporal'] = seed, name
            prefix = f'input={input_kind}/normalization={normalization}/family={family}/protocolo=blocks/seed={seed}/temporal_loss={name}'
            if 0 in lambdas:
                cfg = copy.deepcopy(base)
                cfg.update(cross_losses=[], lambdas=[0])
                branches.append(dict(mode='temporal', relative=prefix+'/temporal', config=cfg))
            positive = [value for value in lambdas if value > 0 and (value < 1 or temporal_index == 0)]
            for count, cross_name in itertools.product(buckets, cross):
                if not positive:
                    continue
                cfg = copy.deepcopy(base)
                cfg.update(cross_losses=[cross_name], lambdas=positive)
                cfg['loss'].update({key: options[key] for key in active[cross_name]})
                suffix = '' if count is None else f'_T{count}'
                if count is not None:
                    cfg['loss']['loss_bucket_count'] = count
                slug = '_'.join(f'{key.removeprefix("loss_")}={options[key]!r}' for key in active[cross_name])
                branches.append(dict(mode='composto', relative=f'{prefix}/variant={cross_name}{suffix}'+('_'+slug if slug else ''), config=cfg))
    for branch in branches:
        cfg, tasks = branch['config'], []
        for pct, model, lb, horizon, lam in itertools.product(cfg['train_percentages'], cfg['models'], cfg['lookbacks'], cfg['horizons'], cfg['lambdas']):
            cross_name = cfg['loss']['loss_temporal'] if branch['mode'] == 'temporal' else cfg['cross_losses'][0]
            ks = [None] if branch['mode'] == 'temporal' else [k for k in windows if k <= horizon and horizon % k == 0]
            for k in ks:
                task = dict(array_index=len(tasks), treino=pct, experimento=branch['mode'], loss=cross_name,
                            cross_lambda=lam, cross_delta=0, modelo=model, lookback=lb, pred_len=horizon, k=k)
                task.update(b3.ordinal_options(cfg, cross_name))
                tasks.append(task)
        branch.update(tasks=tasks, signature=b3.digest(cfg))
    return branches


def task_hyper(config, task):
    hyper = dict(config['model_defaults'][task['modelo']], **config['training'])
    hyper.update(config['loss'])
    hyper.update(seq_len=task['lookback'], horizon=task['pred_len'], pred_len=task['pred_len'],
        seed=config['strategy']['seed'],
        loss_cross=task['loss'] if task['cross_lambda'] else 'ranknet',
        loss_cross_lambda=task['cross_lambda'], loss_trade_window=task['k'] or 1)
    return hyper


def model_instance(config, task):
    from ts_benchmark.baselines.rank_target_model import RankTargetForecaster
    spec = config['model_specs'][task['modelo']]
    return RankTargetForecaster(model_name=spec['model_name'], model_adapter=spec['adapter'],
        model_label=task['modelo'], input_kind=config['input_kind'],
        input_normalization=config['input_normalization'], **task_hyper(config, task))


def code_hashes():
    files = list((ROOT/'ts_benchmark').rglob('*.py'))
    files.extend(ROOT/'scripts'/name for name in ('rank_target_experiment.py', 'run_rank_target_experiment.sh',
        'b3_experiment.py', 'b3_experiment_io.py', 'ordinal_experiment.py', 'ordinal_model_registry.py'))
    return {path.relative_to(ROOT).as_posix(): b3.file_hash(path) for path in sorted(files)}


def semantic_settings(s):
    return {key: s[key] for key in ('EXPERIMENT_TAG', 'PRED_ROOT', 'NS_RETURNS_FILE', 'NS_RESULT_ROOT', 'NS_LOG_ROOT', 'NS_MANTER_CSV')}


def plan(s, branches):
    tasks = [task for branch in branches for task in branch['tasks']]
    print(f"Rank target | {s['INPUT_KIND']} -> rank | familia={s['LOSS_FAMILY']} | normalizacao da entrada={s['INPUT_NORMALIZATION']}")
    print(f"Modelos={s['MODEL']} | treino={s['TRAIN_VALUES']} | LB={s['LOOKBACKS_VALUES']} | H={s['HORIZON_VALUES']}")
    print(f"Treino: K_loss={s['K_LOSS_VALUES']}; blocos disjuntos, todos os H passos. K=1 e ranking diario.")
    print(f"Avaliacao: K={s['K_EVAL_VALUES']}; RANK_WINDOWS_VALUES nao altera o treino.")
    print('Saida: posicao continua (menor = melhor); sem expm1, zscore transversal ou denormalizacao da saida.')
    for key, label in (('PRED_ROOT', 'Previsoes'), ('NS_RESULT_ROOT', 'Resultados'), ('NS_LOG_ROOT', 'Logs')):
        print(f'{label}: {s[key]}')
    for branch in branches:
        print(f"{len(branch['tasks']):6d}\t{branch['relative']}")
    print(f'TOTAL: {len(tasks)} treinos unicos')
    print('Controles lambda=0: '+str(sum(t['cross_lambda'] == 0 for t in tasks))+
          '; mistos: '+str(sum(0 < t['cross_lambda'] < 1 for t in tasks))+
          '; ranking puro lambda=1: '+str(sum(t['cross_lambda'] == 1 for t in tasks)))
    suffix = 'por ramo' if ordinal.boolean(s, 'NS_PARALLEL_BRANCHES') else 'no experimento inteiro'
    print(f"Concorrencia: ate {s['NS_MAX_CONCURRENT_JOBS']} tarefas {suffix}; maximo {s['NS_ARRAY_CHUNK_SIZE']} tarefas por array.")


def _state_root(s):
    return Path(s['PRED_ROOT'])/'.rank_target'


def prepare(s, branches):
    from scripts.b3_experiment_io import prepare_dataset
    state = _state_root(s)
    with b3.locked(state/'.prepare.lock'):
        # Construction checks the native model contract before any Slurm job.
        checked = set()
        for branch in branches:
            for task in branch['tasks']:
                key = task['modelo'], task['lookback'], task['pred_len']
                if key not in checked:
                    model_instance(branch['config'], task)
                    checked.add(key)
        metadata = prepare_dataset(Path(s['NS_RETURNS_FILE']), state/'input'/'log_returns.csv', source_data_kind='log_return')
        metadata.update(SEMANTICS, input_kind=s['INPUT_KIND'], input_normalization=s['INPUT_NORMALIZATION'])
        for branch in branches:
            cfg = branch['config']
            for task in branch['tasks']:
                split = b3.split_counts(cfg, task, metadata['n_rows'])
                if split['n_train'] < task['lookback']+task['pred_len'] or split['n_validation'] < task['pred_len'] or split['n_windows'] < 1:
                    raise ValueError('Treino/validacao/teste insuficientes para '+str(task))
                if split['n_windows'] > cfg['strategy']['num_rollings']:
                    raise ValueError('NS_NUM_ROLLINGS deve cobrir todas as origens do teste: '+str(split['n_windows']))
                if cfg['loss'].get('loss_bucket_count', 2) > metadata['n_assets']:
                    raise ValueError('BUCKET_COUNTS_VALUES nao pode exceder o numero de ativos')
        code = code_hashes()
        design = dict(schema_version=SCHEMA_VERSION, branches=branches, settings=semantic_settings(s),
                      metadata=metadata, code_sha256=code)
        signature = b3.digest(design)
        design['design_sha256'] = signature
        snapshot = state/'design.json'
        if snapshot.exists() and json.loads(snapshot.read_text())['design_sha256'] != signature:
            raise ValueError('Configuracao, dados ou codigo alterados no mesmo PRED_ROOT; use outro destino. Consulte --submission-status.')
        # Validate every owned manifest before writing any of them.
        manifests = []
        for branch in branches:
            for group in sorted({b3.group_path(task) for task in branch['tasks']}):
                path = Path(s['PRED_ROOT'])/branch['relative']/group/'run_config.json'
                if path.exists() and json.loads(path.read_text())['design_sha256'] != signature:
                    raise ValueError('Outro desenho ocupa '+str(path))
                manifests.append((path, branch['config']))
        b3.atomic_text(snapshot, json.dumps(design, indent=2)+'\n')
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        for path, cfg in manifests:
            run = dict(schema_version=SCHEMA_VERSION, design_sha256=signature, config=cfg,
                       config_path=str(snapshot), metadata=metadata, git_commit=commit)
            b3.atomic_text(path, json.dumps(run, indent=2)+'\n')
    print(f'Desenho preparado: {snapshot}', flush=True)
    return snapshot, design


def load_design(snapshot):
    path = Path(snapshot).resolve()
    design = json.loads(path.read_text())
    if design.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Snapshot nao pertence ao experimento rank target')
    expected = design['design_sha256']
    if b3.digest({key: value for key, value in design.items() if key != 'design_sha256'}) != expected:
        raise ValueError('Snapshot alterado')
    if design['code_sha256'] != code_hashes():
        raise ValueError('Codigo alterado depois da preparacao; worker nao executado')
    meta = design['metadata']
    for key in ('dataset', 'calendar'):
        if b3.file_hash(Path(meta[key])) != meta[key+'_sha256']:
            raise ValueError('Entrada congelada alterada: '+key)
    return design


def worker(snapshot, branch_index, task_index):
    import numpy as np
    import pandas as pd
    import torch
    from scripts.b3_experiment_io import load_panel, export_prediction_windows
    from ts_benchmark.baselines.rank_target_data import stable_descending_ranks_numpy
    design = load_design(snapshot)
    if not 0 <= branch_index < len(design['branches']):
        raise ValueError('Ramo fora da grade')
    branch = design['branches'][branch_index]
    if not 0 <= task_index < len(branch['tasks']):
        raise ValueError('Tarefa fora da grade')
    task, cfg = branch['tasks'][task_index], branch['config']
    output = Path(design['settings']['PRED_ROOT'])/branch['relative']/b3.output_path(task)
    signature = design['design_sha256']
    with b3.locked(output/'.task.lock'):
        parquet, completed = output/'part.parquet', output/'execution_metadata.json'
        if completed.exists():
            prior = json.loads(completed.read_text())
            if prior.get('design_sha256') == signature and parquet.exists() and prior.get('parquet_sha256') == b3.file_hash(parquet):
                print('Concluido anteriormente: '+str(output), flush=True)
                return prior
            raise ValueError('Previsao concluida foi alterada: '+str(output))
        seed = cfg['strategy']['seed']
        import random
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        panel = load_panel(Path(design['metadata']['dataset']))
        # Keep original calendar features when the source supplied dates. TFB's
        # synthetic daily labels are only necessary for integer step sources.
        calendar = pd.read_csv(design['metadata']['calendar'])
        original_dates = calendar['source_target']
        if pd.to_numeric(original_dates, errors='coerce').isna().all():
            panel.index = pd.DatetimeIndex(pd.to_datetime(original_dates, errors='raise'), name='date')
        split = b3.split_counts(cfg, task, len(panel))
        cut, horizon = split['n_pretest'], task['pred_len']
        model = model_instance(cfg, task)
        model.forecast_fit(panel.iloc[:cut], train_ratio_in_tv=cfg['strategy']['train_ratio_in_tv'])
        origins = np.arange(cut-1, len(panel)-horizon, dtype=np.int64)
        prediction = model.predict_origins(panel, origins)
        ranks = pd.DataFrame(stable_descending_ranks_numpy(panel.to_numpy()), index=panel.index, columns=panel.columns)
        actual = np.stack([ranks.iloc[origin+1:origin+1+horizon].to_numpy() for origin in origins])
        result = export_prediction_windows(actual, prediction, ranks, cut, horizon, output,
            keep_csv=design['settings']['NS_MANTER_CSV'].lower() in {'true', '1'}, step_offset=1)
        audit = dict(schema_version=1, design_sha256=signature, parquet_sha256=result['parquet_sha256'],
            **SEMANTICS, input_kind=cfg['input_kind'], input_normalization=cfg['input_normalization'],
            hostname=socket.gethostname(), completed_at_utc=datetime.now(timezone.utc).isoformat(),
            split=split, task=task, fit_metadata=getattr(model, 'fit_metadata_', {}),
            slurm_job_id=os.environ.get('SLURM_JOB_ID', ''), slurm_array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID', ''),
            slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID', ''))
        result_output = Path(design['settings']['NS_RESULT_ROOT'])/branch['relative']/b3.output_path(task)
        result_output.mkdir(parents=True, exist_ok=True)
        b3.atomic_text(result_output/'training_history.json', json.dumps(getattr(model, 'history_', []), indent=2, allow_nan=False)+'\n')
        # Continuous positions are the primary artifact; the checkpoint also
        # retains ordinal cutpoints for reproducible probability/class decoding.
        checkpoint = result_output/'.model_checkpoint.pt.tmp'
        torch.save(dict(model=model.model.state_dict(), criterion=model.criterion.state_dict(),
                        task=task, config=cfg, fit_metadata=audit['fit_metadata']), checkpoint)
        checkpoint.replace(result_output/'model_checkpoint.pt')
        audit['checkpoint'] = str(result_output/'model_checkpoint.pt')
        audit['checkpoint_sha256'] = b3.file_hash(result_output/'model_checkpoint.pt')
        b3.atomic_text(completed, json.dumps(audit, indent=2, allow_nan=False)+'\n')
        print('Previsoes: '+str(parquet), flush=True)
        return audit


def array_commands(s, branches, snapshot):
    commands = []
    chunk = int(s['NS_ARRAY_CHUNK_SIZE'])
    for branch_index, branch in enumerate(branches):
        for start in range(0, len(branch['tasks']), chunk):
            stop = min(start+chunk, len(branch['tasks']))-1
            log = Path(s['NS_LOG_ROOT'])/f'branch_{branch_index:03d}'
            args = ['sbatch', '--parsable', '--partition='+s['NS_PARTITION'], '--time='+s['NS_TIME_LIMIT'],
                '--cpus-per-task='+s['NS_CPUS_PER_TASK'], '--mem='+s['NS_MEMORY'],
                f'--array=0-{stop-start}%{s["NS_MAX_CONCURRENT_JOBS"]}',
                f'--job-name=rank_b{branch_index:03d}', '--output='+str(log/'%A_%a.out'), '--error='+str(log/'%A_%a.err')]
            if s['NS_GRES']:
                args.append('--gres='+s['NS_GRES'])
            if s['NS_NODELIST']:
                args.append('--nodelist='+s['NS_NODELIST'])
            # A frozen worker script supplies the array id at execution time.
            script = _state_root(s)/f'worker_{branch_index:03d}_{start:06d}.sh'
            commands.append(dict(branch=branch_index, start=start, stop=stop, args=args, script=str(script), log=str(log)))
    return commands


def submission_status(s):
    path = _state_root(s)/'submission.json'
    return json.loads(path.read_text()) if path.exists() else dict(status='not_submitted', path=str(path))


def submit(s, branches):
    if shutil.which('sbatch') is None:
        raise RuntimeError('sbatch nao encontrado; use --plan ou --dry-run fora do cluster')
    if not Path(s['NS_PYTHON_BIN']).is_file() and shutil.which(s['NS_PYTHON_BIN']) is None:
        raise ValueError('NS_PYTHON_BIN nao encontrado')
    snapshot, design = prepare(s, branches)
    state = _state_root(s)
    with b3.locked(state/'.submission.lock'):
        path = state/'submission.json'
        if path.exists():
            previous = json.loads(path.read_text())
            if previous.get('design_sha256') != design['design_sha256']:
                raise ValueError('Outro desenho ja foi submetido neste PRED_ROOT')
            if previous.get('status') == 'submitted':
                print('Desenho ja submetido: '+','.join(previous['job_ids']))
                return previous
            raise ValueError('Submissao parcial ou inconclusiva; confira --submission-status e o scheduler antes de reutilizar o destino')
        record = dict(status='submitting', design_sha256=design['design_sha256'], pred_root=s['PRED_ROOT'], job_ids=[], arrays=[])
        b3.atomic_text(path, json.dumps(record, indent=2)+'\n')
        previous_job, branch_tail = s['NS_AFTER_JOB'], {}
        try:
            for command in array_commands(s, branches, snapshot):
                branch_index = command['branch']
                worker_command = [str(ROOT/'scripts/run_rank_target_experiment.sh'), '--on-node', '--worker', str(snapshot), '--branch', str(branch_index)]
                script = ('#!/usr/bin/env bash\nset -Eeuo pipefail\n'+
                    'export TFB_ROOT='+shlex.quote(str(ROOT))+'\n'+
                    'export PYTHON_BIN='+shlex.quote(s['NS_PYTHON_BIN'])+'\n'+
                    'export VENV_PATH='+shlex.quote(s['NS_VENV_PATH'])+'\n'+
                    'exec bash '+shlex.join(worker_command)+f' --index "$(( ${{SLURM_ARRAY_TASK_ID:?}} + {command["start"]} ))"\n')
                b3.atomic_text(Path(command['script']), script)
                Path(command['log']).mkdir(parents=True, exist_ok=True)
                args = list(command['args'])
                dependency = branch_tail.get(branch_index, s['NS_AFTER_JOB']) if ordinal.boolean(s, 'NS_PARALLEL_BRANCHES') else previous_job
                if dependency:
                    args.append('--dependency=afterany:'+dependency)
                args.append(command['script'])
                result = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                job_id = result.stdout.strip().split(';')[0]
                if result.returncode or not job_id.isdigit():
                    raise RuntimeError('sbatch falhou: '+result.stderr.strip()+' '+result.stdout.strip())
                previous_job = branch_tail[branch_index] = job_id
                record['job_ids'].append(job_id)
                record['arrays'].append(dict(command, job_id=job_id, dependency=dependency))
                b3.atomic_text(path, json.dumps(record, indent=2)+'\n')
                print(f'Job {job_id}: ramo {branch_index}, tarefas {command["start"]}-{command["stop"]}', flush=True)
        except BaseException:
            record['status'] = 'partial'
            b3.atomic_text(path, json.dumps(record, indent=2)+'\n')
            raise
        record['status'] = 'submitted'
        b3.atomic_text(path, json.dumps(record, indent=2)+'\n')
        return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--plan', '--trainplan', action='store_true')
    action.add_argument('--dry-run', action='store_true')
    action.add_argument('--print-config', action='store_true')
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--train', action='store_true')
    action.add_argument('--local', action='store_true', help='Treina toda a grade no processo local, sem Slurm')
    action.add_argument('--submission-status', action='store_true')
    action.add_argument('--worker', metavar='SNAPSHOT')
    parser.add_argument('--branch', type=int, default=0)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            worker(args.worker, args.branch, args.index)
            return
        s = settings()
        if args.submission_status:
            print(json.dumps(submission_status(s), indent=2))
            return
        branches = build_branches(s)
        if args.print_config:
            print(json.dumps(dict(settings=s, branches=branches), indent=2))
        elif args.prepare:
            prepare(s, branches)
        elif args.train:
            plan(s, branches)
            submit(s, branches)
        elif args.local:
            snapshot, _ = prepare(s, branches)
            for bi, branch in enumerate(branches):
                for ti in range(len(branch['tasks'])):
                    worker(snapshot, bi, ti)
        else:
            plan(s, branches)
            if args.dry_run:
                for command in array_commands(s, branches, _state_root(s)/'design.json'):
                    print(shlex.join(command['args']+[command['script']]))
    except (ValueError, OSError, RuntimeError) as error:
        print('ERRO: '+str(error), file=sys.stderr)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
