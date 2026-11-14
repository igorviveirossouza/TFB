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
from scripts.ordinal_model_registry import discover_models, resolve_model

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
    for key in ('PRED_ROOT', 'NS_LOG_ROOT', 'NS_RESULT_ROOT', 'NS_RETURNS_FILE'):
        path = Path(out[key])
        out[key] = str((path if path.is_absolute() else ROOT/path).resolve())
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
    registry = discover_models(ROOT)
    model_specs = {}
    for requested in tokens(s['MODEL'], str, 'MODEL'):
        label, spec = resolve_model(requested, ROOT, registry)
        model_specs.setdefault(label, spec)
    models = list(model_specs)
    overrides = json.loads(s['NS_MODEL_DEFAULTS_JSON'].strip() or '{}')
    if not isinstance(overrides, dict):
        raise ValueError('NS_MODEL_DEFAULTS_JSON deve ser objeto por modelo')
    model_defaults = {name: {'norm': True, **DEFAULT_MODELS.get(name, {})} for name in models}
    reserved = {'batch_size', 'seq_len', 'pred_len', 'horizon', 'task_head_mode', 'rank_windows',
                'ranking_construction', 'parallel_strategy', 'lr', 'num_epochs', 'patience', 'validation_mode'}
    for requested, opts in overrides.items():
        if not isinstance(opts, dict):
            raise ValueError('Override de arquitetura deve ser objeto: '+requested)
        name, _ = resolve_model(requested, ROOT, registry)
        if name not in model_defaults:
            continue
        if any(k in reserved or k.startswith(('loss', 'momentum')) for k in opts):
            raise ValueError('JSON de arquitetura contem campos de treino/loss; use exports especificos')
        model_defaults[name].update(opts)
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
        models=models, model_defaults=architecture, model_specs=model_specs,
        training=training, loss=loss,
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
    print(f"Resultados: {s['NS_RESULT_ROOT']}")
    print(f"Logs: {s['NS_LOG_ROOT']}")
    print(f"Estado de submissao: {submission_state(s)}")
    for b in branches:
        print(f"{len(b['tasks']):6d}\t{b['relative']}")
    print(f'Controles lambda=0: {counts[0]}; mistos: {counts[1]}; ordinal puro lambda=1: {counts[2]}')
    print(f'TOTAL: {len(tasks)} treinos unicos')
    limit = s['NS_MAX_CONCURRENT_JOBS']
    print(f'Concorrencia: ate {limit} tarefas por experimento' if not boolean(s, 'NS_PARALLEL_BRANCHES') else
          f'Concorrencia: ate {limit} tarefas POR RAMO ({len(branches)} ramos)')


def submission_state(s):
    """The prediction destination owns its jobs, independently of the log path."""
    return Path(s['PRED_ROOT']).resolve()/'.ordinal_submissions'


def marker_name(branch):
    return hashlib.sha256(branch['relative'].encode()).hexdigest()+'.json'


def marker_record(s, branch, status, job_ids, *, signature=None, config=None, dependency=None):
    return dict(schema_version=2, pred_root=str(Path(s['PRED_ROOT']).resolve()),
        experiment_tag=s['EXPERIMENT_TAG'], log_root=str(Path(s['NS_LOG_ROOT']).resolve()),
        result_root=str(Path(s['NS_RESULT_ROOT']).resolve()), branch=branch['relative'],
        signature=branch['signature'] if signature is None else signature,
        config=branch['config'] if config is None else config, status=status,
        parallel_branches=boolean(s, 'NS_PARALLEL_BRANCHES'), dependency=dependency,
        job_ids=list(job_ids))


def read_marker(path):
    try:
        prior = json.loads(path.read_text())
    except (ValueError, OSError) as exc:
        raise ValueError('Registro de submissao ilegivel: '+str(path)) from exc
    if (not isinstance(prior, dict) or not isinstance(prior.get('signature'), str)
            or not isinstance(prior.get('job_ids'), list)
            or any(not isinstance(job, str) or not job.isdigit() for job in prior['job_ids'])):
        raise ValueError('Registro de submissao invalido: '+str(path))
    return prior


def changed_fields(old, new, prefix=''):
    if isinstance(old, dict) and isinstance(new, dict):
        result = []
        for key in sorted(set(old)|set(new)):
            name = prefix+'.'+key if prefix else key
            if key not in old or key not in new:
                result.append(dict(field=name, before=old.get(key), after=new.get(key)))
            else:
                result.extend(changed_fields(old[key], new[key], name))
        return result
    return [] if old == new else [dict(field=prefix, before=old, after=new)]


def legacy_evidence(s, branch, prior):
    """Find a matching frozen config for diagnostics, not proof of submission.

    The same design may have been prepared in multiple destinations; a matching
    run_config alone never binds legacy Slurm IDs to this prediction root.
    """
    base = Path(s['PRED_ROOT'])/branch['relative']
    for path in sorted(base.glob('treino=*/experimento=*/run_config.json')):
        try:
            config = json.loads(path.read_text())['config']
            if b3.digest(config) == prior['signature']:
                return dict(path=str(path), config=config)
        except (ValueError, OSError, KeyError, TypeError):
            continue
    return None


def inspect_submissions(s, branches):
    """Read both generations of records without creating files or scheduler calls."""
    result = []
    for branch in branches:
        canonical = submission_state(s)/marker_name(branch)
        legacy = Path(s['NS_LOG_ROOT'])/'ordinal_submissions'/marker_name(branch)
        row = dict(branch=branch['relative'], marker=str(canonical), legacy_marker=str(legacy),
                   current_signature=branch['signature'], status='not_submitted',
                   prior=None, changed_fields=[], legacy_disposition='absent', legacy_job_ids=[])
        if canonical.exists():
            prior = read_marker(canonical)
            if (prior.get('schema_version') != 2 or prior.get('branch') != branch['relative']
                    or prior.get('pred_root') != str(Path(s['PRED_ROOT']).resolve())):
                raise ValueError('Identidade do registro nao corresponde ao destino: '+str(canonical))
            row.update(prior=prior, status=prior.get('status', 'submitted'))
            if legacy.exists():
                row['legacy_disposition'] = 'superseded_by_canonical'
                try:
                    row['legacy_job_ids'] = read_marker(legacy)['job_ids']
                except ValueError as exc:
                    row['legacy_error'] = str(exc)
        elif legacy.exists():
            prior = read_marker(legacy)
            row['legacy_job_ids'] = prior['job_ids']
            recorded_root = prior.get('pred_root')
            row['legacy_pred_root'] = recorded_root
            if recorded_root is not None and (not isinstance(recorded_root, str) or not Path(recorded_root).is_absolute()):
                raise ValueError('Destino invalido no registro legado: '+str(legacy))
            if recorded_root is not None and Path(recorded_root).resolve() != Path(s['PRED_ROOT']).resolve():
                row['legacy_disposition'] = 'known_other_destination'
            else:
                evidence = legacy_evidence(s, branch, prior)
                row.update(prior=prior, status=prior.get('status', 'submitted'))
                if recorded_root is None:
                    row['legacy_disposition'] = 'destination_unidentified'
                    row['status'] = 'legacy_ambiguous'
                    if evidence is not None:
                        row['evidence'] = evidence['path']
                        row['evidence_note'] = 'Config equivalente preparada; nao comprova destino dos jobs legados.'
                        row['prior'] = dict(prior, config=evidence['config'])
                else:
                    if prior.get('branch') != branch['relative']:
                        raise ValueError('Ramo do registro legado nao corresponde ao destino: '+str(legacy))
                    snapshot = prior.get('config')
                    if not isinstance(snapshot, dict):
                        snapshot = evidence['config'] if evidence is not None else None
                    row.update(legacy_disposition='migration_verified')
                    row['prior'] = marker_record(s, branch, prior.get('status', 'submitted'),
                        prior['job_ids'], signature=prior['signature'], config=snapshot or {})
                    row['prior']['config'] = snapshot
                    row['prior']['migrated_from'] = str(legacy)
                    row['prior']['parallel_branches'] = prior.get('parallel_branches')
                    row['prior']['dependency'] = prior.get('dependency')
        if row['prior'] is not None:
            row['config_changed'] = row['prior']['signature'] != branch['signature']
            if isinstance(row['prior'].get('config'), dict):
                row['changed_fields'] = changed_fields(row['prior']['config'], branch['config'])
        else:
            row['config_changed'] = False
        result.append(row)
    return result


def submission_status(s, branches):
    rows = inspect_submissions(s, branches)
    # Keep the diagnostic compact; fields/hashes/IDs describe the exact conflict.
    for row in rows:
        prior = row.pop('prior')
        row['prior_job_ids'] = prior.get('job_ids', []) if prior else []
        row['prior_signature'] = prior.get('signature') if prior else None
    return dict(pred_root=s['PRED_ROOT'], log_root=s['NS_LOG_ROOT'],
        result_root=s['NS_RESULT_ROOT'], state_root=str(submission_state(s)), branches=rows,
        unselected_history=unselected_history(s, branches))


def unselected_history(s, branches):
    selected = {marker_name(branch) for branch in branches}
    result = []
    canonical_paths = sorted(submission_state(s).glob('*.json'))
    canonical_names = {path.name for path in canonical_paths}
    for path in canonical_paths:
        if path.name in selected:
            continue
        try:
            prior = read_marker(path)
            result.append(dict(marker=str(path), branch=prior.get('branch'),
                status=prior.get('status'), job_ids=prior['job_ids'],
                pred_root=prior.get('pred_root'), storage='canonical'))
        except ValueError as exc:
            result.append(dict(marker=str(path), status='invalid', job_ids=[], error=str(exc)))
    for path in sorted((Path(s['NS_LOG_ROOT'])/'ordinal_submissions').glob('*.json')):
        if path.name in selected or path.name in canonical_names:
            continue
        try:
            prior = read_marker(path)
            recorded_root = prior.get('pred_root')
            if (isinstance(recorded_root, str) and Path(recorded_root).is_absolute()
                    and Path(recorded_root).resolve() != Path(s['PRED_ROOT']).resolve()):
                continue
            result.append(dict(marker=str(path), branch=prior.get('branch'),
                status='legacy_ambiguous' if recorded_root is None else prior.get('status'),
                job_ids=prior['job_ids'], pred_root=recorded_root, storage='legacy'))
        except ValueError as exc:
            result.append(dict(marker=str(path), status='legacy_invalid', job_ids=[], error=str(exc)))
    return result


def validate_submission_record(row):
    prior = row['prior']
    if prior is None:
        return
    jobs = ','.join(prior['job_ids']) or 'sem IDs confirmados'
    if row['legacy_disposition'] == 'destination_unidentified':
        raise ValueError('Destino do registro legado nao identificado: '+row['legacy_marker']+
            '; IDs Slurm: '+jobs+'. Nenhuma submissao nova foi feita. '
            'Use --submission-status; para um experimento independente, defina nova EXPERIMENT_TAG '
            'e PRED_ROOT/NS_LOG_ROOT/NS_RESULT_ROOT correspondentes. O registro antigo foi preservado.')
    if row['config_changed']:
        fields = ', '.join(change['field'] for change in row['changed_fields'][:12]) or 'configuracao sem snapshot comparavel'
        raise ValueError('Configuracao alterada no mesmo destino de previsoes. Registro: '+row['marker']+
            '; IDs Slurm: '+jobs+'; campos alterados: '+fields+'. '
            'Use --submission-status para os valores; uma nova configuracao exige outro PRED_ROOT '
            '(normalmente nova EXPERIMENT_TAG com os tres caminhos atualizados).')
    if row['status'] in {'partial', 'submitting'}:
        label = 'Submissao parcial' if row['status'] == 'partial' else 'Submissao sem confirmacao final'
        raise ValueError(label+' registrada: '+row['marker']+'; IDs Slurm: '+jobs+
                         '. Confira o scheduler antes de qualquer nova submissao.')
    if row['status'] != 'submitted' or not prior['job_ids']:
        raise ValueError('Estado de submissao inconclusivo: '+row['marker'])


def train(s, branches):
    if shutil.which('sbatch') is None:
        raise RuntimeError('sbatch nao encontrado; use --plan ou --dry-run fora do cluster')
    if not Path(s['NS_PYTHON_BIN']).is_file() and shutil.which(s['NS_PYTHON_BIN']) is None:
        raise RuntimeError('NS_PYTHON_BIN nao encontrado')
    state = submission_state(s)
    state.mkdir(parents=True, exist_ok=True)
    with (state/'.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        rows = inspect_submissions(s, branches)
        markers = {row['branch']: Path(row['marker']) for row in rows}
        # Preserve proven historical ownership even when the requested config
        # differs. Changing the log directory later cannot bypass this guard.
        for row in rows:
            if row['legacy_disposition'] == 'migration_verified':
                b3.atomic_text(Path(row['marker']), json.dumps(row['prior'], indent=2)+'\n')
        for row in rows:
            validate_submission_record(row)
        existing = {row['branch']:row['prior'] for row in rows if row['prior'] is not None}
        history = unselected_history(s, branches)
        if history and len(existing) < len(branches) and not boolean(s, 'NS_PARALLEL_BRANCHES'):
            ids = ','.join(job for row in history for job in row['job_ids']) or 'sem IDs confirmados'
            raise ValueError('Ha historico de ramos nao selecionados no mesmo PRED_ROOT; IDs Slurm: '+ids+
                '. A submissao serial de novos ramos foi interrompida. Consulte --submission-status '
                'e preserve o desenho anterior, ou use outro destino para um experimento independente.')
        if (existing and len(existing) < len(branches) and not boolean(s, 'NS_PARALLEL_BRANCHES')
                and not boolean(s, 'NS_RESUME_CHAIN_OLD_JOBS')):
            raise ValueError('Ha ramos ja submetidos e ramos pendentes. Para manter o limite global, '
                             'use NS_RESUME_CHAIN_OLD_JOBS=1 para encadear aos IDs registrados.')
        if existing and len(existing) < len(branches) and not boolean(s, 'NS_PARALLEL_BRANCHES'):
            submitted = [branch['relative'] in existing for branch in branches]
            existing_ids = ','.join(job for prior in existing.values() for job in prior['job_ids'])
            if any(submitted[submitted.index(False):]):
                raise ValueError('Retomada serial com ramos submetidos intercalados aos pendentes. '
                                 'A retomada automatica aceita somente um prefixo ja submetido; '
                                 'IDs Slurm: '+existing_ids+'. Confira --submission-status antes de reorganizar a grade.')
            if any(prior.get('parallel_branches') is not False for prior in existing.values()):
                raise ValueError('Historico de submissao paralelo ou indefinido; nao e seguro assumir '
                                 'um unico encadeamento serial; IDs Slurm: '+existing_ids+'. Confira --submission-status.')
            prefix = [existing[branch['relative']] for branch in branches if branch['relative'] in existing]
            if any(current.get('dependency') != previous['job_ids'][-1]
                   for previous, current in zip(prefix, prefix[1:])):
                raise ValueError('Ordem dos ramos anteriores nao corresponde ao encadeamento registrado; '
                                 'retomada serial interrompida; IDs Slurm: '+existing_ids+'. Consulte --submission-status.')
            known_chain_ids = {job for prior in prefix for job in prior['job_ids']}
            known_chain_ids.add(prefix[0].get('dependency'))
            if s['NS_AFTER_JOB'] and s['NS_AFTER_JOB'] not in known_chain_ids:
                raise ValueError('NS_AFTER_JOB nao pertence ao encadeamento anterior; nao e possivel '
                                 'combinar automaticamente as duas dependencias nesta retomada. '
                                 'Solicitado: '+s['NS_AFTER_JOB']+'; IDs anteriores: '+existing_ids+'.')
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
            # Persist intent before contacting Slurm. A crash or lost stdout
            # must not allow a second submission with unknown live job IDs.
            b3.atomic_text(marker, json.dumps(marker_record(s, branch, 'submitting', [], dependency=dependency), indent=2)+'\n')
            try:
                jobs = run_producer(s, branch, 'submit', dependency)
            except RuntimeError as exc:
                if getattr(exc, 'job_ids', None):
                    b3.atomic_text(marker, json.dumps(marker_record(s, branch, 'partial', exc.job_ids, dependency=dependency), indent=2)+'\n')
                raise
            if not jobs:
                raise RuntimeError('Submissao sem IDs Slurm; estado pendente em '+str(marker)+'; interrompido para evitar repeticao')
            b3.atomic_text(marker, json.dumps(marker_record(s, branch, 'submitted', jobs, dependency=dependency), indent=2)+'\n')
            previous = jobs[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    for flag in ('plan', 'dry-run', 'train', 'submit', 'prepare', 'print-config', 'print-exports', 'submission-status'):
        actions.add_argument('--'+flag, action='store_true')
    args = parser.parse_args()
    s = settings()
    if args.print_exports:
        for key, value in s.items():
            print('export '+key+'='+shlex.quote(value))
        return
    branches = build_branches(s)
    if args.submission_status:
        print(json.dumps(submission_status(s, branches), indent=2, ensure_ascii=False))
        return
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
