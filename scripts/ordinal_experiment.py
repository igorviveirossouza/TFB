#!/usr/bin/env python3
"""Isolated ordinal grid using the native B3 producer and frozen Slurm workers."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import b3_experiment as b3

DEFAULT_MODELS = {
    'Nonstationary_Transformer': dict(d_model=128, d_ff=256, e_layers=2, d_layers=1,
        n_heads=8, dropout=.1, factor=3, norm=True, p_hidden_dims=[32, 32],
        p_hidden_layers=2, lradj='type1'),
    'iTransformer': dict(d_model=128, d_ff=256, e_layers=2, n_heads=8,
        dropout=.1, norm=True, lradj='type1'),
    'simpleFORMER': dict(d_model=512, d_ff=2048, e_layers=2, n_heads=8,
        temporal_hidden_dim=0, dropout=.1, norm=True, use_norm=True,
        norm_eps=1e-5, lradj='type1'),
    'simpleMLP': dict(d_model=512, d_ff=2048, e_layers=2, n_heads=8,
        temporal_hidden_dim=0, dropout=.1, norm=True, use_norm=True,
        norm_eps=1e-5, lradj='type1'),
    'TimesNet': dict(d_ff=512, d_model=256, factor=3, norm=True, top_k=5),
    'FEDformer': dict(d_ff=512, d_model=256, factor=3, norm=True),
    'DUET': dict(CI=1, d_ff=32, d_model=32, dropout=.5, e_layers=1, factor=3,
        fc_dropout=.2, hidden_size=256, k=1, lradj='type1', n_heads=2,
        norm=True, num_experts=4, patch_len=48),
}
DEFAULTS = dict(
    EXPERIMENT_TAG='isolados/ordinal_buckets_t80_v1', MODEL='Nonstationary_Transformer',
    TRAIN_VALUES='80', LOOKBACKS_VALUES='32 256 512', HORIZON_VALUES='1 25',
    K_LOSS_VALUES='1', LAMBDAS_VALUES='0 0.1 0.5 0.8 1', SEEDS_VALUES='2026',
    TEMPORAL_LOSSES_VALUES='mse', BUCKET_COUNTS_VALUES='3',
    CROSS_LOSSES_VALUES='ordinal_ce bucket_ranknet bucket_hinge bucket_listnet',
    NS_RANKING_PROTOCOL='origin_anchored_windows', RANK_WINDOWS_VALUES='1 5 10 15 20 25',
    K_EVAL_VALUES='1 5 10 15 20 25', NS_ORDINAL_TAU='1', NS_BUCKET_TAU='1',
    NS_BUCKET_TARGET_TAU='1', NS_HINGE_MARGIN='0.1',
    NS_BATCH_SIZE='16', NS_NUM_EPOCHS='100', NS_PATIENCE='10', NS_LEARNING_RATE='0.0001',
    NS_LRADJ='', NS_TRAIN_RATIO_IN_TV='0.875', NS_VALIDATION_MODE='ordered_sample_mean',
    NS_STRIDE='1', NS_NUM_ROLLINGS='48000', NS_CROSS_SCALE='1', NS_HUBER_DELTA='0.5',
    NS_SCORE_KIND='simple_return', NS_CROSS_SCORE_NORMALIZATION='zscore',
    NS_LOSS_INVERSE_NORM='true', NS_LOSS_TRACK_COMPONENTS='true',
    NS_PARTITION='gorgonas', NS_GRES='gpu:1', NS_NODELIST='', NS_TIME_LIMIT='01:30:00',
    NS_CPUS_PER_TASK='3', NS_MEMORY='32G', NS_MAX_CONCURRENT_JOBS='2',
    NS_ARRAY_CHUNK_SIZE='1000', NS_PARALLEL_BRANCHES='0', NS_AFTER_JOB='',
    NS_RESUME_CHAIN_OLD_JOBS='0', NS_MANTER_CSV='FALSE',
    NS_PYTHON_BIN='/sonic_home/igor.viveiros/py310/bin/python',
    NS_VENV_PATH='/sonic_home/igor.viveiros/py310/bin/activate', SOURCE_DATA_KIND='log_return',
)


def settings(environ=None):
    env = os.environ if environ is None else environ
    out = {key: env.get(key, value) for key, value in DEFAULTS.items()}
    out['TFB_ROOT'] = str(ROOT)
    tag = out['EXPERIMENT_TAG']
    if not tag or Path(tag).is_absolute() or '..' in Path(tag).parts:
        raise ValueError('EXPERIMENT_TAG deve ser caminho relativo sem ..')
    derived = dict(PRED_ROOT=f'/snfs2/igor.viveiros/previsoes/{tag}',
        NS_RESULT_ROOT=f'/snfs2/igor.viveiros/result/{tag}', NS_LOG_ROOT=str(ROOT/'logs'/tag),
        NS_RETURNS_FILE=str(ROOT/'dataset/forecasting/b3_daily_tfb_new.csv'),
        NS_MODEL_DEFAULTS_JSON=json.dumps(DEFAULT_MODELS, separators=(',', ':')))
    out.update({key: env.get(key, value) for key, value in derived.items()})
    return out


def tokens(value, cast, label, minimum=None, maximum=None):
    result = []
    for item in value.split():
        try:
            x = cast(item)
        except ValueError as exc:
            raise ValueError(f'{label}: valor invalido {item!r}') from exc
        if isinstance(x, float) and not math.isfinite(x):
            raise ValueError(label + ': valores devem ser finitos')
        if minimum is not None and x < minimum or maximum is not None and x > maximum:
            raise ValueError(label + ': valor fora do intervalo')
        if x not in result:
            result.append(x)
    if not result:
        raise ValueError(label + ': lista vazia')
    return result


def number(s, key, cast=float, positive=True):
    values = tokens(s[key], cast, key)
    if len(values) != 1 or positive and values[0] <= 0:
        raise ValueError(key + ': exige um numero positivo')
    return values[0]


def boolean(s, key):
    value = s[key].lower()
    if value not in {'true', 'false', '0', '1'}:
        raise ValueError(key + ': use true/false ou 1/0')
    return value in {'true', '1'}


def build_branches(s):
    models = tokens(s['MODEL'], str, 'MODEL')
    models = list(dict.fromkeys('Nonstationary_Transformer' if x == 'NonStationary_Transformer' else x for x in models))
    if set(models) - set(DEFAULT_MODELS):
        raise ValueError('Modelo sem integracao ordinal nativa: ' + ' '.join(sorted(set(models)-set(DEFAULT_MODELS))))
    model_defaults = json.loads(s['NS_MODEL_DEFAULTS_JSON'])
    if not isinstance(model_defaults, dict):
        raise ValueError('NS_MODEL_DEFAULTS_JSON deve ser objeto por modelo')
    reserved = {'batch_size', 'seq_len', 'pred_len', 'horizon', 'task_head_mode', 'rank_windows',
                'ranking_construction', 'parallel_strategy', 'lr', 'num_epochs', 'patience', 'validation_mode'}
    for name in models:
        opts = model_defaults.get(name)
        if not isinstance(opts, dict):
            raise ValueError('Modelo sem arquitetura em NS_MODEL_DEFAULTS_JSON: ' + name)
        if any(k in reserved or k.startswith(('loss', 'momentum')) for k in opts):
            raise ValueError('JSON de arquitetura contem campos de treino/loss; use exports especificos')
    batch = number(s, 'NS_BATCH_SIZE', int)
    architecture = {name: dict(model_defaults[name], batch_size=batch, parallel_strategy=None) for name in models}
    temporal = tokens(s['TEMPORAL_LOSSES_VALUES'], str, 'TEMPORAL_LOSSES_VALUES')
    if set(temporal) - {'mse', 'mae', 'huber'}:
        raise ValueError('TEMPORAL_LOSSES_VALUES: use mse, mae ou huber')
    cross = tokens(s['CROSS_LOSSES_VALUES'], str, 'CROSS_LOSSES_VALUES')
    if set(cross) - b3.ORDINAL_LOSSES:
        raise ValueError('CROSS_LOSSES_VALUES: use somente as quatro losses ordinais')
    buckets = tokens(s['BUCKET_COUNTS_VALUES'], int, 'BUCKET_COUNTS_VALUES', 2)
    lambdas = tokens(s['LAMBDAS_VALUES'], float, 'LAMBDAS_VALUES', 0, 1)
    seeds = tokens(s['SEEDS_VALUES'], int, 'SEEDS_VALUES', 0)
    windows = tokens(s['K_LOSS_VALUES'], int, 'K_LOSS_VALUES', 1)
    horizons = tokens(s['HORIZON_VALUES'], int, 'HORIZON_VALUES', 1)
    protocol = s['NS_RANKING_PROTOCOL']
    if protocol not in {'blocks', 'origin_anchored_windows'}:
        raise ValueError('NS_RANKING_PROTOCOL deve ser blocks ou origin_anchored_windows')
    rank_windows = tokens(s['RANK_WINDOWS_VALUES'], int, 'RANK_WINDOWS_VALUES', 1) if protocol == 'origin_anchored_windows' else []
    if protocol == 'origin_anchored_windows' and (windows != [1] or any(not any(k <= h for k in rank_windows) for h in horizons)):
        raise ValueError('origin_anchored_windows exige K_LOSS_VALUES=1 e ao menos uma janela <= cada H')
    tokens(s['K_EVAL_VALUES'], int, 'K_EVAL_VALUES', 1)
    if protocol == 'blocks' and any(not any(k <= h and h % k == 0 for k in windows) for h in horizons):
        raise ValueError('Cada H precisa de um K_loss que divida H')
    for key in ('NS_MAX_CONCURRENT_JOBS', 'NS_ARRAY_CHUNK_SIZE', 'NS_CPUS_PER_TASK'):
        number(s, key, int)
    boolean(s, 'NS_PARALLEL_BRANCHES'); boolean(s, 'NS_RESUME_CHAIN_OLD_JOBS')
    if s['NS_AFTER_JOB'] and not re.fullmatch('[0-9]+', s['NS_AFTER_JOB']):
        raise ValueError('NS_AFTER_JOB deve ser ID Slurm numerico ou vazio')
    if s['NS_SCORE_KIND'] not in {'simple_return', 'log_return'}:
        raise ValueError('NS_SCORE_KIND deve ser simple_return ou log_return')
    if s['NS_CROSS_SCORE_NORMALIZATION'] not in {'zscore', 'none'}:
        raise ValueError('NS_CROSS_SCORE_NORMALIZATION deve ser zscore ou none')
    loss = dict(loss_temporal=temporal[0], loss_data_kind='log_return',
        loss_score_kind=s['NS_SCORE_KIND'], loss_cross_score_normalization=s['NS_CROSS_SCORE_NORMALIZATION'],
        loss_cross_scale=number(s, 'NS_CROSS_SCALE'), loss_huber_delta=number(s, 'NS_HUBER_DELTA'),
        loss_inverse_norm=boolean(s, 'NS_LOSS_INVERSE_NORM'),
        loss_track_components=boolean(s, 'NS_LOSS_TRACK_COMPONENTS'))
    training = dict(num_epochs=number(s, 'NS_NUM_EPOCHS', int), patience=number(s, 'NS_PATIENCE', int),
                    lr=number(s, 'NS_LEARNING_RATE'))
    if s['NS_LRADJ']:
        training['lradj'] = s['NS_LRADJ']
    if s['NS_VALIDATION_MODE']:
        training['validation_mode'] = s['NS_VALIDATION_MODE']
    config = dict(source_data_kind=s['SOURCE_DATA_KIND'],
        train_percentages=tokens(s['TRAIN_VALUES'], int, 'TRAIN_VALUES', 1, 99),
        models=models, model_defaults=architecture, training=training, loss=loss,
        lookbacks=tokens(s['LOOKBACKS_VALUES'], int, 'LOOKBACKS_VALUES', 1), horizons=horizons,
        trade_windows=windows, cross_losses=['ranknet'], lambdas=[.5], deltas=[0], var_lags=13,
        strategy=dict(train_ratio_in_tv=number(s, 'NS_TRAIN_RATIO_IN_TV'),
            stride=number(s, 'NS_STRIDE', int), num_rollings=number(s, 'NS_NUM_ROLLINGS', int), seed=seeds[0]))
    ordinal = dict(loss_ordinal_tau=number(s, 'NS_ORDINAL_TAU'), loss_bucket_tau=number(s, 'NS_BUCKET_TAU'),
                   loss_bucket_target_tau=number(s, 'NS_BUCKET_TARGET_TAU'),
                   loss_hinge_margin=number(s, 'NS_HINGE_MARGIN', positive=False))
    if ordinal['loss_hinge_margin'] < 0:
        raise ValueError('NS_HINGE_MARGIN deve ser nao negativa')
    active = {'ordinal_ce': ['loss_ordinal_tau'], 'bucket_ranknet': ['loss_bucket_tau'],
              'bucket_hinge': ['loss_hinge_margin'], 'bucket_listnet': ['loss_bucket_tau', 'loss_bucket_target_tau']}
    branches = []
    for seed in seeds:
        for ti, temporal_loss in enumerate(temporal):
            base = copy.deepcopy(config)
            base['strategy']['seed'] = seed
            base['loss']['loss_temporal'] = temporal_loss
            prefix = f'protocolo={protocol}/seed={seed}/temporal_loss={temporal_loss}'
            if 0 in lambdas:
                branches.append(dict(mode='temporal', relative=prefix+'/temporal', config=copy.deepcopy(base)))
            positive = [v for v in lambdas if v > 0 and (v < 1 or ti == 0)]
            if not positive:
                continue
            for count in buckets:
                for name in cross:
                    cfg = copy.deepcopy(base)
                    cfg.update(cross_losses=[name], lambdas=positive)
                    cfg['loss'].update(loss_bucket_count=count, loss_cross_construction=protocol,
                                       loss_rank_windows=rank_windows)
                    cfg['loss'].update({key: ordinal[key] for key in active[name]})
                    slug = '_'.join(f'{key.removeprefix("loss_")}={ordinal[key]!r}' for key in active[name])
                    relative = f'{prefix}/variant={name}_T{count}_{slug}'
                    branches.append(dict(mode='composto', relative=relative, config=cfg))
    if not branches:
        raise ValueError('Grade vazia')
    previous = {key: os.environ.get(key) for key in ('B3_INLINE_CONFIG', 'B3_CONFIG_SOURCE')}
    try:
        for branch in branches:
            os.environ['B3_INLINE_CONFIG'] = json.dumps(branch['config'], allow_nan=False)
            os.environ['B3_CONFIG_SOURCE'] = str(ROOT/'scripts/run_ordinal_experiment.sh')
            _, cfg = b3.load_config()
            branch['config'] = cfg
            branch['tasks'] = b3.make_tasks(cfg, branch['mode'])
            if not branch['tasks']:
                raise ValueError('Ramo sem tarefas: ' + branch['relative'])
            branch['signature'] = b3.digest(cfg)
    finally:
        for key, value in previous.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value
    return branches


def branch_environment(s, branch, dependency=''):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith('B3_'):
            env.pop(key)
    env.update(TFB_ROOT=str(ROOT), PYTHON_BIN=s['NS_PYTHON_BIN'], VENV_PATH=s['NS_VENV_PATH'],
        DATASET_PATH=s['NS_RETURNS_FILE'], OUT_ROOT=str(Path(s['PRED_ROOT'])/branch['relative']),
        RESULT_ROOT=str(Path(s['NS_RESULT_ROOT'])/branch['relative']),
        LOG_ROOT=str(Path(s['NS_LOG_ROOT'])/branch['relative']),
        RUN_TAG=s['EXPERIMENT_TAG']+'/'+branch['relative'], GPU_PARTITION=s['NS_PARTITION'],
        GPU_TIME=s['NS_TIME_LIMIT'], GRES=s['NS_GRES'], NODELIST=s['NS_NODELIST'],
        CPUS_PER_TASK=s['NS_CPUS_PER_TASK'], MEMORY=s['NS_MEMORY'],
        MAX_GPU_JOBS=s['NS_MAX_CONCURRENT_JOBS'], ARRAY_CHUNK_SIZE=s['NS_ARRAY_CHUNK_SIZE'],
        MANTER_CSV=s['NS_MANTER_CSV'], B3_AFTER_JOB=dependency,
        B3_CONFIG_SOURCE=str(ROOT/'scripts/run_ordinal_experiment.sh'),
        B3_WORKER_LAUNCHER=str(ROOT/'scripts/run_ordinal_experiment.sh'),
        B3_INLINE_CONFIG=json.dumps(branch['config'], allow_nan=False))
    return env


def run_producer(s, branch, action, dependency=''):
    # Inspection requires no training venv; the frozen workers retain NS_PYTHON_BIN.
    python = sys.executable if action in {'--dry-run', '--print-config'} else s['NS_PYTHON_BIN']
    command = [python, str(ROOT/'scripts/b3_experiment.py'), branch['mode'], action]
    result = subprocess.run(command, env=branch_environment(s, branch, dependency),
                            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end='', flush=True)
    jobs = re.findall(r'^Job ([0-9]+):', result.stdout, flags=re.MULTILINE)
    if result.returncode:
        error = RuntimeError(f'Produtor falhou ({result.returncode}): {branch["relative"]}')
        error.job_ids = jobs
        raise error
    return jobs


def plan(s, branches):
    tasks = [t for b in branches for t in b['tasks']]
    counts = [sum(t['cross_lambda'] == 0 for t in tasks),
              sum(0 < t['cross_lambda'] < 1 for t in tasks), sum(t['cross_lambda'] == 1 for t in tasks)]
    print(f"Ordinal | modelos={s['MODEL']} | cortes={s['TRAIN_VALUES']} | LB={s['LOOKBACKS_VALUES']} | H={s['HORIZON_VALUES']}")
    print(f"Protocolo={s['NS_RANKING_PROTOCOL']}; T={s['BUCKET_COUNTS_VALUES']}; seeds={s['SEEDS_VALUES']}")
    print(f"Previsoes: {s['PRED_ROOT']}")
    for b in branches:
        print(f"{len(b['tasks']):6d}\t{b['relative']}")
    print(f'Controles lambda=0: {counts[0]}; mistos: {counts[1]}; ordinal puro lambda=1: {counts[2]}')
    print(f'TOTAL: {len(tasks)} treinos unicos')
    limit = s['NS_MAX_CONCURRENT_JOBS']
    print(f'Concorrencia: ate {limit} tarefas por experimento' if not boolean(s, 'NS_PARALLEL_BRANCHES') else
          f'Concorrencia: ate {limit} tarefas POR RAMO ({len(branches)} ramos)')


def train(s, branches):
    if shutil.which('sbatch') is None:
        raise RuntimeError('sbatch nao encontrado; use --plan ou --dry-run fora do cluster')
    if not Path(s['NS_PYTHON_BIN']).is_file() and shutil.which(s['NS_PYTHON_BIN']) is None:
        raise RuntimeError('NS_PYTHON_BIN nao encontrado')
    state = Path(s['NS_LOG_ROOT'])/'ordinal_submissions'
    state.mkdir(parents=True, exist_ok=True)
    with (state/'.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        markers, existing = {}, {}
        for branch in branches:
            marker = state/(hashlib.sha256(branch['relative'].encode()).hexdigest()+'.json')
            markers[branch['relative']] = marker
            if marker.exists():
                prior = json.loads(marker.read_text())
                if prior['signature'] != branch['signature']:
                    raise ValueError('Configuracao alterada em ramo submetido; use outra EXPERIMENT_TAG')
                if prior.get('status') == 'partial':
                    raise ValueError('Submissao parcial registrada: confira os IDs Slurm em '+str(marker)+
                                     ' antes de qualquer nova submissao')
                existing[branch['relative']] = prior
        if (existing and len(existing) < len(branches) and not boolean(s, 'NS_PARALLEL_BRANCHES')
                and not boolean(s, 'NS_RESUME_CHAIN_OLD_JOBS')):
            raise ValueError('Ha ramos ja submetidos e ramos pendentes. Para manter o limite global, '
                             'use NS_RESUME_CHAIN_OLD_JOBS=1 para encadear aos IDs registrados.')
        # All branches are prepared/validated before the first scheduler mutation.
        for branch in branches:
            run_producer(s, branch, '--prepare')
        previous = s['NS_AFTER_JOB']
        for branch in branches:
            marker = markers[branch['relative']]
            if branch['relative'] in existing:
                prior = existing[branch['relative']]
                print('Ja submetido: '+branch['relative'], flush=True)
                if boolean(s, 'NS_RESUME_CHAIN_OLD_JOBS'):
                    previous = prior['job_ids'][-1]
                continue
            dependency = s['NS_AFTER_JOB'] if boolean(s, 'NS_PARALLEL_BRANCHES') else previous
            try:
                jobs = run_producer(s, branch, 'submit', dependency)
            except RuntimeError as exc:
                if getattr(exc, 'job_ids', None):
                    b3.atomic_text(marker, json.dumps(dict(signature=branch['signature'],
                        job_ids=exc.job_ids, branch=branch['relative'], status='partial'), indent=2)+'\n')
                raise
            if not jobs:
                raise RuntimeError('Submissao sem IDs Slurm; interrompido para evitar repeticao')
            b3.atomic_text(marker, json.dumps(dict(signature=branch['signature'], job_ids=jobs,
                branch=branch['relative'], status='submitted'), indent=2)+'\n')
            previous = jobs[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    for flag in ('plan', 'dry-run', 'train', 'submit', 'prepare', 'print-config', 'print-exports'):
        actions.add_argument('--'+flag, action='store_true')
    args = parser.parse_args()
    s = settings()
    if args.print_exports:
        for key, value in s.items():
            print('export '+key+'='+shlex.quote(value))
        return
    branches = build_branches(s)
    if args.print_config:
        print(json.dumps([{k:v for k,v in b.items() if k != 'tasks'} for b in branches], indent=2))
        return
    plan(s, branches)
    if args.train or args.submit:
        train(s, branches)
    elif args.dry_run or args.prepare:
        for branch in branches:
            run_producer(s, branch, '--prepare' if args.prepare else '--dry-run')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit('ERRO: '+str(exc))
