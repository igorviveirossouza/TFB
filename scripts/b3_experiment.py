#!/usr/bin/env python3
"""One reproducible design shared by the temporal, composite and VAR launchers.

No heavy dependency or filesystem mutation is needed for --dry-run / --task.
Workers consume the prepared design, verify its provenance, then run TFB v3.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ts_benchmark.baselines.proximity_losses import PROXIMITY_LOSSES, validate_cross_mask
from ts_benchmark.baselines.ranking_mixture import (
    RANK_MIX_COMPONENTS, validate_rank_mix_config,
    validate_rank_mix_proximity_components,
)

LAUNCHERS = {'temporal': 'run_temporal_experiment.sh',
             'composto': 'run_composite_trading_experiment.sh',
             'var': 'run_var_experiment.sh'}
MOMENTUM_KEYS = ('momentum_mode', 'momentum_lookback', 'momentum_eta',
                 'momentum_gain_init', 'momentum_diagnostics')
MOMENTUM_INPUT_KEYS = ('momentum_input_mode', 'momentum_input_lookback')
LR_SCHEDULES = {'type1', 'type2', 'type3', 'constant'}
MOMENTUM_MODELS = {'Nonstationary_Transformer', 'DUET', 'TimesNet', 'FEDformer'}
RANK_MIX_LOSSES = {'rankmix', 'rankmix_proximity'}
RANK_MIX_FIELDS = ('weights', 'mode', 'regions', 'fraction')
RANK_MIX_PROXIMITY_KEY = 'loss_rank_mix_proximity_components'
RANKING_PROTOCOL_FIELDS = ('loss_cross_construction', 'loss_rank_windows')
ORDINAL_LOSSES = {'ordinal_ce', 'bucket_ranknet', 'bucket_hinge', 'bucket_listnet'}
ORDINAL_FIELDS = ('bucket_count', 'ordinal_tau', 'bucket_tau', 'bucket_target_tau',
                  'bucket_margin', 'bucket_tie_policy')


def ordinal_options(config, name):
    """Active ordinal settings only; safe to inspect without Torch installed."""
    if name not in ORDINAL_LOSSES:
        return {}
    loss = config['loss']
    count = loss.get('loss_bucket_count', 3)
    if type(count) is not int or count < 2:
        raise ValueError('loss_bucket_count deve ser inteiro >= 2')
    result = dict(bucket_count=count, bucket_tie_policy='stable_channel_order')
    keys = {'ordinal_ce': [('ordinal_tau', 'loss_ordinal_tau', 1.0)],
            'bucket_ranknet': [('bucket_tau', 'loss_bucket_tau', 1.0)],
            'bucket_hinge': [('bucket_margin', 'loss_hinge_margin', .1)],
            'bucket_listnet': [('bucket_tau', 'loss_bucket_tau', 1.0),
                               ('bucket_target_tau', 'loss_bucket_target_tau', 1.0)]}
    for short, key, default in keys[name]:
        value = loss.get(key, default)
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < 0 or value == 0 and short != 'bucket_margin'):
            raise ValueError(key + ' deve ser finito e positivo (margem pode ser zero)')
        result[short] = float(value)
    return result


def ranking_protocol_settings(config, task=None):
    """Validate the opt-in shared-output objective, without importing Torch.

    Missing protocol fields retain the legacy configuration and task identities.
    Origin windows are prefix sums, not the native disjoint K-day blocks.
    """
    loss = config['loss']
    if not any(key in loss for key in RANKING_PROTOCOL_FIELDS):
        return None
    construction = loss.get('loss_cross_construction')
    if not isinstance(construction, str) or construction not in {'blocks', 'origin_anchored_windows'}:
        raise ValueError('CROSS_CONSTRUCTION deve ser blocks ou origin_anchored_windows')
    windows = loss.get('loss_rank_windows', [])
    if (not isinstance(windows, list) or any(type(k) is not int or k < 1 for k in windows)
            or len(windows) != len(set(windows))):
        raise ValueError('RANK_WINDOWS exige inteiros positivos unicos')
    if construction == 'blocks':
        if windows:
            raise ValueError('RANK_WINDOWS nao se aplica a blocks; use TRADE_WINDOWS/K_LOSS_VALUES')
    else:
        if not windows or any(not any(k <= h for k in windows) for h in config['horizons']):
            raise ValueError('Cada H precisa de ao menos uma janela de RANK_WINDOWS <= H')
        if any(k != 1 for k in config['trade_windows']):
            raise ValueError('origin_anchored_windows exige K_LOSS_VALUES=1; use RANK_WINDOWS')
        if (config.get('source_data_kind', 'log_return') != 'log_return'
                or loss.get('loss_data_kind', 'log_return') != 'log_return'):
            raise ValueError('origin_anchored_windows exige dados log_return')
        if (not loss.get('loss_inverse_norm', True)
                and any(config['model_defaults'].get(model, {}).get('norm', True)
                        for model in config['models'])):
            raise ValueError('origin_anchored_windows com norm=True exige loss_inverse_norm=True '
                             'para supervisionar retornos economicos')
        if loss.get('momentum_mode', 'off') != 'off' or loss.get('momentum_input_mode', 'off') != 'off':
            raise ValueError('origin_anchored_windows exige momentum somente como benchmark externo')
        if any('task_head_mode' in config['model_defaults'].get(model, {})
               for model in config['models']):
            raise ValueError('origin_anchored_windows usa adaptadores nativos; nao combine task_head_mode')
    result = dict(cross_construction=construction, rank_windows=list(windows))
    if task is not None:
        active = task['experimento'] == 'composto' and task['cross_lambda'] > 0
        result['ranking_supervision_active'] = active
        result['rank_windows'] = ([k for k in windows if k <= task['pred_len']]
                                  if active else [])
    return result


def task_head_settings(config, task=None):
    from scripts.itransformer_task_heads import settings
    return settings(config, task)


def rank_mix_options(config):
    """Canonical mixture settings, without importing training/GPU dependencies."""
    loss = config['loss']
    return validate_rank_mix_config(
        weights=loss.get('loss_rank_mix_weights'),
        mode=loss.get('loss_rank_mix_mode', 'global'),
        regions=loss.get('loss_rank_mix_regions'),
        fraction=loss.get('loss_rank_mix_fraction', 1 / 3),
    )


def rank_mix_proximity_options(config):
    """Return only an opt-in partial selection; all retains the legacy identity."""
    if 'rankmix_proximity' not in config['cross_losses']:
        return None
    selected = validate_rank_mix_proximity_components(
        config['loss'].get(RANK_MIX_PROXIMITY_KEY))
    return None if selected == list(RANK_MIX_COMPONENTS) else selected


def momentum_options(config):
    """Validate optional scientific parameters without importing Torch."""
    loss = config['loss']
    unknown = {key for key in loss if key.startswith('momentum_')} - set(MOMENTUM_KEYS + MOMENTUM_INPUT_KEYS)
    if unknown:
        raise ValueError(f'Parâmetro momentum desconhecido: {sorted(unknown)}')
    mode = loss.get('momentum_mode', 'off')
    if not isinstance(mode, str) or mode not in {'off', 'residual_output'}:
        raise ValueError('MOMENTUM_MODE deve ser off ou residual_output')
    lookback = loss.get('momentum_lookback')
    if lookback is not None and (type(lookback) is not int or lookback < 1):
        raise ValueError('MOMENTUM_LOOKBACK deve ser um inteiro positivo ou ficar vazio (= LB do modelo)')
    eta = loss.get('momentum_eta', 0.0)
    if type(eta) not in (int, float) or not math.isfinite(eta) or eta < 0:
        raise ValueError('MOMENTUM_ETA deve ser finito e não negativo')
    gain_init = loss.get('momentum_gain_init', 0.0)
    if type(gain_init) not in (int, float) or not math.isfinite(gain_init):
        raise ValueError('MOMENTUM_GAIN_INIT deve ser um numero finito')
    if type(loss.get('momentum_diagnostics', False)) is not bool:
        raise ValueError('MOMENTUM_DIAGNOSTICS deve ser booleano')
    return mode, lookback, float(eta)


def momentum_input_options(config):
    """Explicit momentum feature, independently switchable from residual output."""
    loss = config['loss']
    mode = loss.get('momentum_input_mode', 'off')
    if not isinstance(mode, str) or mode not in {'off', 'zero', 'feature'}:
        raise ValueError('MOMENTUM_INPUT_MODE deve ser off, zero ou feature')
    lookback = loss.get('momentum_input_lookback')
    if lookback is not None and (type(lookback) is not int or lookback < 1):
        raise ValueError('MOMENTUM_INPUT_LOOKBACK deve ser inteiro positivo ou vazio (= LB do modelo)')
    if mode != 'off':
        if loss.get('momentum_mode', 'off') != 'off':
            raise ValueError('Momentum na entrada nao pode ser combinado com momentum residual_output')
        unsupported = set(config['models']) - {'Nonstationary_Transformer'}
        if unsupported:
            raise ValueError(f'Momentum na entrada ainda nao validado para: {sorted(unsupported)}')
        if (config.get('source_data_kind', 'log_return') != 'log_return'
                or loss.get('loss_data_kind', 'log_return') != 'log_return'):
            raise ValueError('Momentum na entrada requer SOURCE_DATA_KIND=log_return e loss_data_kind=log_return')
        defaults = config['model_defaults']['Nonstationary_Transformer']
        if defaults.get('target_mode', 'path') == 'cumulative':
            raise ValueError('Momentum na entrada requer target_mode=path')
        if defaults.get('task_name', 'short_term_forecast') not in {'short_term_forecast', 'long_term_forecast'}:
            raise ValueError('Momentum na entrada suporta somente tarefas de forecast')
    return mode, lookback


def validate_momentum_experiment(config):
    momentum_input_options(config)
    if momentum_options(config)[0] != 'off':
        unsupported = set(config['models']) - MOMENTUM_MODELS
        if unsupported:
            raise ValueError(f'Momentum residual ainda não validado para: {sorted(unsupported)}')
        if config.get('source_data_kind', 'log_return') != 'log_return':
            raise ValueError('Momentum residual requer SOURCE_DATA_KIND=log_return (step_offset=1)')
        loss = config['loss']
        normalization = loss.get('loss_cross_score_normalization',
                                 loss.get('loss_rank_score_normalization', 'zscore'))
        value_losses = {'mse', 'pairwise_mse', 'pairwise_mse_proximity'}
        if str(normalization).lower() == 'none' and value_losses.intersection(config['cross_losses']):
            raise ValueError('Momentum residual exige zscore para mse e pairwise_mse (inclusive proximity)')


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
    config.setdefault('source_data_kind', 'log_return')
    if config['source_data_kind'] not in {'price', 'log_return'}:
        raise ValueError('SOURCE_DATA_KIND deve ser price ou log_return')
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
    if set(config['cross_losses']) - ({'mse', 'bpr', 'ranknet', 'hinge', 'listnet', 'pairwise_mse', 'spearman', 'listmle', 'approxndcg'} | RANK_MIX_LOSSES | ORDINAL_LOSSES | set(PROXIMITY_LOSSES)):
        raise ValueError('Loss cross não suportada nesta grade')
    for name in set(config['cross_losses']) & ORDINAL_LOSSES:
        ordinal_options(config, name)
    if set(config['cross_losses']) & RANK_MIX_LOSSES:
        config['loss'].update({'loss_rank_mix_' + key: value
                               for key, value in rank_mix_options(config).items()})
    else:
        for key in RANK_MIX_FIELDS:
            config['loss'].pop('loss_rank_mix_' + key, None)
    proximity_components = rank_mix_proximity_options(config)
    if proximity_components is None:
        config['loss'].pop(RANK_MIX_PROXIMITY_KEY, None)
    else:
        config['loss'][RANK_MIX_PROXIMITY_KEY] = proximity_components
    if any(loss in PROXIMITY_LOSSES for loss in config['cross_losses']):
        config['loss']['loss_cross_mask'] = list(validate_cross_mask(
            config['loss'].get('loss_cross_mask')))
    for key in ('loss_huber_delta', 'loss_spearman_tau', 'loss_listnet_tau',
                'loss_listmle_tau', 'loss_approxndcg_tau'):
        value = config['loss'].get(key, 0.5 if key == 'loss_huber_delta' else 1.0)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} deve ser positivo e finito')
    if any(not 0 < x <= 1 for x in config['lambdas']):
        raise ValueError('Lambda composto deve estar em (0,1]')
    if any(not math.isfinite(x) or x < 0 for x in config['deltas']):
        raise ValueError('Delta deve ser finito e não negativo')
    if config['strategy']['stride'] != 1:
        raise ValueError('Este experimento requer stride=1')
    if not 0 < config['strategy']['train_ratio_in_tv'] < 1:
        raise ValueError('A validação interna deve ser preservada')
    for key in ('num_epochs', 'patience', 'lr'):
        if not math.isfinite(config['training'][key]) or config['training'][key] <= 0:
            raise ValueError(f'{key} deve ser positivo e finito')
    if 'lradj' in config['training'] and (not isinstance(config['training']['lradj'], str)
                                           or config['training']['lradj'] not in LR_SCHEDULES):
        raise ValueError('LRADJ deve ser type1, type2, type3 ou constant')
    if ('validation_mode' in config['training']
            and config['training']['validation_mode'] != 'ordered_sample_mean'):
        raise ValueError('VALIDATION_MODE deve ser vazio ou ordered_sample_mean')
    input_mode, _ = momentum_input_options(config)
    if input_mode == 'off':
        for key in MOMENTUM_INPUT_KEYS:
            config['loss'].pop(key, None)
    mode, _, _ = momentum_options(config)
    if mode == 'off':
        # The opt-out keeps historical config identities and cache keys intact.
        for key in MOMENTUM_KEYS:
            config['loss'].pop(key, None)
    else:
        # Neutral new options must not change old frozen design identities.
        if config['loss'].get('momentum_gain_init', 0.0) == 0.0:
            config['loss'].pop('momentum_gain_init', None)
        if not config['loss'].get('momentum_diagnostics', False):
            config['loss'].pop('momentum_diagnostics', None)
    validate_momentum_experiment(config)
    ranking_protocol_settings(config)
    from scripts.itransformer_task_heads import validate_config as validate_task_heads
    validate_task_heads(config)
    return path, config


def cross_deltas(config, loss):
    # ListNet and proximity losses have no pairwise dead zone. Keep one inactive delta
    # for paths/audits, independent of the editable DELTAS vector.
    return [0] if loss in {'mse', 'listnet', 'listmle', 'approxndcg', 'spearman', 'rankmix'} or loss in PROXIMITY_LOSSES or loss in ORDINAL_LOSSES else config['deltas']


def make_tasks(config, mode):
    ranking_protocol_settings(config)
    proximity_components = rank_mix_proximity_options(config)
    momentum_mode, momentum_lb, momentum_eta = momentum_options(config)
    input_mode, input_lb = momentum_input_options(config)
    if mode == 'composto':
        validate_momentum_experiment(config)
    tasks = []
    for pct in config['train_percentages']:
        losses = (config['cross_losses'] if mode == 'composto' else
                  [config['loss']['loss_temporal'] if mode == 'temporal' else 'mse'])
        lambdas = config['lambdas'] if mode == 'composto' else [0]
        models = config['models'] if mode != 'var' else ['VAR']
        lookbacks = config['lookbacks'] if mode != 'var' else [config['var_lags']]
        horizons = config['horizons'] if mode != 'var' else [max(config['horizons'])]
        for loss in losses:
            mask = (validate_cross_mask(config['loss'].get('loss_cross_mask'))
                    if loss in PROXIMITY_LOSSES else None)
            deltas = cross_deltas(config, loss) if mode == 'composto' else [0]
            for lam, delta, model, lb, horizon in itertools.product(
                    lambdas, deltas, models, lookbacks, horizons):
                ks = [k for k in config['trade_windows'] if k <= horizon and horizon % k == 0] if mode == 'composto' else [None]
                for k in ks:
                    task = dict(array_index=len(tasks), treino=pct, experimento=mode,
                                loss=loss, cross_lambda=lam, cross_delta=delta,
                                modelo=model, lookback=lb, pred_len=horizon, k=k)
                    task.update(ordinal_options(config, loss))
                    if loss in RANK_MIX_LOSSES:
                        task.update({'rank_mix_' + key: value
                                     for key, value in rank_mix_options(config).items()})
                    if loss == 'rankmix_proximity' and proximity_components is not None:
                        task['rank_mix_proximity_components'] = list(proximity_components)
                    if mode == 'composto' and momentum_mode != 'off':
                        if not 0 < lam <= 1:
                            raise ValueError('Momentum residual exige lambda composto em (0,1]')
                        task.update(momentum_mode=momentum_mode,
                                    momentum_lookback=momentum_lb or lb,
                                    momentum_eta=momentum_eta)
                        if config['loss'].get('momentum_gain_init', 0.0) != 0.0:
                            task['momentum_gain_init'] = float(config['loss']['momentum_gain_init'])
                        if config['loss'].get('momentum_diagnostics', False):
                            task['momentum_diagnostics'] = True
                    if mode == 'composto' and input_mode != 'off':
                        task.update(momentum_input_mode=input_mode,
                                    momentum_input_lookback=input_lb or lb)
                    if mask is not None:
                        task.update(mask_type=mask[0], mask_a=mask[1])
                        if len(mask) >= 3 and mask[2] != 'natural':
                            task['mask_mode'] = mask[2]
                        if len(mask) > 3:
                            task['mask_shape'] = list(mask[3:])
                    protocol = ranking_protocol_settings(config, task)
                    if protocol is not None:
                        task.update(protocol)
                    tasks.append(task)
    return tasks


def group_path(task):
    return Path(f"treino={task['treino']}") / f"experimento={task['experimento']}"


def output_path(task):
    lam = format(task['cross_lambda'], 'g') if task['experimento'] == 'composto' else '00'
    path = group_path(task) / f"loss={task['loss']}" / f'lambda={lam}'
    if task['experimento'] == 'composto':
        path /= f"delta={task['cross_delta']:g}"
        if task['loss'] in ORDINAL_LOSSES:
            path /= f"buckets={task['bucket_count']}"
            for key in ('ordinal_tau', 'bucket_tau', 'bucket_target_tau', 'bucket_margin'):
                if key in task:
                    path /= f"{key}={float(task[key])!r}"
        if task['loss'] in RANK_MIX_LOSSES:
            path = (path / f"mix={task['rank_mix_mode']}"
                    / ('weights=' + '_'.join(repr(float(x)) for x in task['rank_mix_weights']))
                    / ('regions=' + '_'.join(task['rank_mix_regions']))
                    / f"fraction={float(task['rank_mix_fraction'])!r}")
            if task['loss'] == 'rankmix_proximity' and task.get('rank_mix_proximity_components'):
                path /= 'proximity_on=' + '_'.join(task['rank_mix_proximity_components'])
        if task['loss'] in PROXIMITY_LOSSES:
            # repr preserves distinct float widths; :g would round to six digits.
            path = path / f"mask={task['mask_type']}" / f"a={float(task['mask_a'])!r}"
            if task.get('mask_mode', 'natural') != 'natural':
                path /= f"mask_mode={task['mask_mode']}"
            if task.get('mask_shape'):
                path /= 'shape=' + '_'.join(repr(float(x)) for x in task['mask_shape'])
        if task.get('momentum_mode') == 'residual_output':
            path = (path / 'momentum=residual_output' /
                    f"momentum_lb={task['momentum_lookback']}" /
                    f"momentum_eta={float(task['momentum_eta'])!r}")
            if task.get('momentum_gain_init', 0.0) != 0.0:
                path /= f"momentum_gain_init={float(task['momentum_gain_init'])!r}"
        if task.get('momentum_input_mode', 'off') != 'off':
            path = (path / f"momentum_input={task['momentum_input_mode']}" /
                    f"momentum_input_lb={task['momentum_input_lookback']}")
    path = path / f"modelo={task['modelo']}" / f"lookback={task['lookback']}" / f"pred_len={task['pred_len']}"
    if task['k'] is not None:
        path /= f"k_dir={task['k']}"
    return path


def task_output_paths(config, task, out_root):
    horizons = config['horizons'] if task['modelo'] == 'VAR' else [task['pred_len']]
    return {str(h): out_root / output_path(dict(task, pred_len=h)) for h in horizons}


def model_spec(config, task):
    name = task['modelo']
    selected = config.get('model_specs', {}).get(name)
    if name == 'VAR' and selected is None:
        return 'self_impl.VAR_model', {'lags': config['var_lags']}, None, 'full'
    hyper = dict(config['model_defaults'][name], **config['training'])
    hyper.update(seq_len=task['lookback'], pred_len=task['pred_len'], horizon=task['pred_len'])
    # Ordinary MSE is exactly the composite's temporal term. Do not evaluate
    # unused cross scores at lambda=0 (0 * NaN would still contaminate the loss).
    hyper.update(config['loss'])
    for key in RANK_MIX_FIELDS:
        hyper.pop('loss_rank_mix_' + key, None)
    hyper.pop(RANK_MIX_PROXIMITY_KEY, None)
    if task['loss'] in RANK_MIX_LOSSES:
        hyper.update({'loss_rank_mix_' + key: task['rank_mix_' + key]
                      for key in RANK_MIX_FIELDS})
    if task['loss'] == 'rankmix_proximity' and task.get('rank_mix_proximity_components'):
        hyper[RANK_MIX_PROXIMITY_KEY] = list(task['rank_mix_proximity_components'])
    # Temporal Huber uses the native criterion's key; its scale must match the
    # composite temporal term exactly in the paired control.
    if task['experimento'] == 'temporal' and config['loss']['loss_temporal'] == 'huber':
        hyper['huber_delta'] = config['loss'].get('loss_huber_delta', 0.5)
    # Controls always stay purely temporal, even in a momentum-enabled design.
    for key in MOMENTUM_KEYS + MOMENTUM_INPUT_KEYS:
        hyper.pop(key, None)
    if task.get('momentum_mode') == 'residual_output':
        hyper.update({key: task[key] for key in MOMENTUM_KEYS if key in task})
    if task.get('momentum_input_mode', 'off') != 'off':
        hyper.update({key: task[key] for key in MOMENTUM_INPUT_KEYS})
    # Inactive masks must not change the original models' hyperparameters/cache keys.
    hyper.pop('loss_cross_mask', None)
    if task['loss'] in PROXIMITY_LOSSES:
        hyper['loss_cross_mask'] = list(validate_cross_mask(
            [task['mask_type'], task['mask_a'], task.get('mask_mode', 'natural'),
             *task.get('mask_shape', [])]))
    hyper.update(loss='composite_trading' if task['experimento'] == 'composto' else config['loss']['loss_temporal'], loss_cross=task['loss'],
                 loss_trade_window=task['k'] or 1,
                 loss_cross_lambda=task['cross_lambda'], loss_cross_delta=task['cross_delta'])
    if selected is not None:
        hyper.update(_ordinal_model_name=selected['model_name'],
                     _ordinal_model_adapter=selected['adapter'],
                     _ordinal_model_label=name)
        return 'ordinal_model_adapter.OrdinalModel', hyper, None, selected['deterministic']
    heads = task_head_settings(config, task)
    if heads is not None:
        hyper.update(task_head_mode=heads['task_head_mode'], rank_windows=heads['rank_windows'],
                     ranking_construction=heads['ranking_construction'])
        return 'itransformer_task_heads.ITransformerRestoredTaskAdapter', hyper, None, 'full'
    if name == 'MASTER':
        hyper['seed'] = config['strategy']['seed']
        return 'master.MASTER', hyper, None, 'full'
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
    row = dict(task, **split_counts(config, task, metadata['n_rows']),
                dataset='log_retornos', data_source=metadata['source'],
                source_data_kind=metadata.get('source_data_kind', 'price'),
                step_offset=metadata['step_offset'],
                source_sha256=metadata['source_sha256'], dataset_sha256=metadata['dataset_sha256'],
                n_assets=metadata['n_assets'], seed=config['strategy']['seed'],
                temporal_loss=config['loss']['loss_temporal'],
                delta_effective=task['experimento'] == 'composto' and task['loss'] not in {'mse', 'listnet', 'listmle', 'approxndcg', 'spearman', 'rankmix'} and task['loss'] not in PROXIMITY_LOSSES and task['loss'] not in ORDINAL_LOSSES,
                mask_type=task.get('mask_type', ''), mask_a=task.get('mask_a', ''),
                mask_mode=task.get('mask_mode', 'natural') if task['loss'] in PROXIMITY_LOSSES else '',
                mask_effective=task['loss'] in PROXIMITY_LOSSES,
                mask_shape=json.dumps(task['mask_shape'], separators=(',', ':')) if task.get('mask_shape') else '',
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
    if set(config['cross_losses']) & ORDINAL_LOSSES:
        row.update({key: task.get(key, '') for key in ORDINAL_FIELDS})
    if set(config['cross_losses']) & RANK_MIX_LOSSES:
        # Every row in a mixed registry has the same columns. Arrays use JSON,
        # not Python list repr, so R and CPU consumers can recover the contract.
        for key in RANK_MIX_FIELDS:
            name = 'rank_mix_' + key
            value = task.get(name, '')
            row[name] = json.dumps(value, separators=(',', ':')) if isinstance(value, list) else value
    if rank_mix_proximity_options(config) is not None:
        selected = task.get('rank_mix_proximity_components')
        row['rank_mix_proximity_components'] = (
            json.dumps(selected, separators=(',', ':')) if selected is not None else '')
    protocol = ranking_protocol_settings(config, task)
    if protocol is not None:
        row.update(protocol, rank_windows=json.dumps(protocol['rank_windows']))
        if protocol['cross_construction'] == 'origin_anchored_windows':
            row.update(cross_blocks='', cross_steps='', cross_tail_steps='')
    heads = task_head_settings(config, task)
    if heads is not None:
        row.update(heads, rank_windows=json.dumps(heads['rank_windows']),
                   cross_construction='origin_anchored_windows',
                   cross_blocks='', cross_steps='', cross_tail_steps='')
    return row


def validate_lengths(config, tasks, metadata):
    for task in tasks:
        if 'bucket_count' in task and task['bucket_count'] > metadata['n_assets']:
            raise ValueError('Numero de buckets maior que o numero de ativos')
        split = split_counts(config, task, metadata['n_rows'])
        max_windows = metadata['n_rows'] - split['n_pretest'] - min(config['horizons']) + 1 if task['modelo'] == 'VAR' else split['n_windows']
        if split['n_windows'] < 1 or max_windows > config['strategy']['num_rollings']:
            raise ValueError('Teste sem janelas ou NUM_ROLLINGS insuficiente para cobrir todo o teste')
        # MASTER keeps partial batches (a sample contains the whole universe).
        history = max(task['lookback'], task.get('momentum_lookback', task['lookback']),
                      task.get('momentum_input_lookback', task['lookback']))
        minimum = history + task['pred_len'] + (0 if task['modelo'] == 'MASTER' else 31)
        if split['n_train'] < minimum:
            raise ValueError('Treino insuficiente para contexto, horizonte e batch do modelo')
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
        rows.append(dict(row, file_name=source.name, length=metadata.get('n_source_rows', metadata['n_prices']),
                         freq='other', if_univariate=metadata['n_assets'] == 1))
    previous = previous.loc[~previous.file_name.isin([r['file_name'] for r in rows])]
    updated = pd.concat([previous, pd.DataFrame(rows)], ignore_index=True)
    atomic_text(path, updated.to_csv(index=False))


def provenance(config, metadata):
    code_files = ['scripts/b3_experiment.py', 'scripts/b3_experiment_io.py',
                  'scripts/run_benchmark_composite_trading_loss_v3.py',
                  'ts_benchmark/baselines/composite_trading_loss_v2.py',
                  'ts_benchmark/baselines/composite_trading_loss_v3.py',
                  'ts_benchmark/baselines/proximity_losses.py',
                  'ts_benchmark/baselines/global_ranking_losses.py',
                  'ts_benchmark/baselines/listwise_ranking_losses.py',
                  'ts_benchmark/baselines/self_impl/VAR/VAR.py',
                  'ts_benchmark/baselines/deep_forecasting_model_base.py']
    if set(config['cross_losses']) & RANK_MIX_LOSSES:
        code_files += ['ts_benchmark/baselines/ranking_mixture.py',
                       'ts_benchmark/baselines/hybrid_ranking_losses.py']
    if set(config['cross_losses']) & ORDINAL_LOSSES:
        code_files += ['ts_benchmark/baselines/ordinal_losses.py',
                       'ts_benchmark/baselines/utils.py',
                       'scripts/ordinal_experiment.py', 'scripts/run_ordinal_experiment.sh']
    if config.get('model_specs'):
        code_files += ['scripts/ordinal_model_registry.py',
                       'ts_benchmark/baselines/ordinal_model_adapter.py',
                       'ts_benchmark/models/model_loader.py']
        # Native defaults and architecture dependencies belong to the frozen design.
        for spec in config['model_specs'].values():
            source = (ROOT / spec['source_path']).resolve()
            source.relative_to(ROOT.resolve())
            code_files.append(source.relative_to(ROOT).as_posix())
            family = ROOT / 'ts_benchmark' / 'baselines' / spec['family']
            family.resolve().relative_to((ROOT / 'ts_benchmark' / 'baselines').resolve())
            if family.is_dir():
                code_files.extend(p.relative_to(ROOT).as_posix() for p in sorted(family.rglob('*.py')))
        code_files = list(dict.fromkeys(code_files))
    if task_head_settings(config) is not None:
        code_files += ['scripts/itransformer_task_heads.py',
                       'ts_benchmark/baselines/itransformer_task_heads.py',
                       'ts_benchmark/baselines/ns_task_heads.py',
                       'ts_benchmark/baselines/ns_restored_task_heads.py',
                       'ts_benchmark/baselines/time_series_library/models/iTransformer.py',
                       'ts_benchmark/baselines/time_series_library/adapters_for_transformers.py',
                       'ts_benchmark/baselines/time_series_library/layers/Embed.py',
                       'ts_benchmark/baselines/time_series_library/layers/Transformer_EncDec.py',
                       'ts_benchmark/baselines/time_series_library/layers/SelfAttention_Family.py',
                       'ts_benchmark/utils/random_utils.py']
    if momentum_options(config)[0] != 'off':
        code_files += ['ts_benchmark/baselines/momentum_score.py',
                       'ts_benchmark/baselines/momentum_score_io.py']
    if momentum_input_options(config)[0] != 'off':
        code_files += ['ts_benchmark/baselines/momentum_input.py',
                       'ts_benchmark/baselines/momentum_score.py',
                       'ts_benchmark/baselines/time_series_library/adapters_for_transformers.py',
                       'ts_benchmark/baselines/time_series_library/models/Nonstationary_Transformer.py']
    master_data = None
    if 'MASTER' in config['models']:
        hyper = config['model_defaults']['MASTER']
        source = Path(hyper['master_returns_path']).resolve()
        if source != Path(metadata['source']).resolve():
            raise ValueError('MASTER deve usar o mesmo painel bruto do desenho B3')
        if config['source_data_kind'] != 'log_return':
            raise ValueError('MASTER requer os log-retornos econômicos reconstruídos')
        bundle = Path(hyper['master_bundle_dir']).resolve()
        # Hash all exported files, not only the manifest: mutation of a feature
        # file must invalidate a prepared design before a task can be skipped.
        master_data = dict(returns_sha256=file_hash(source),
                           bundle_sha256={p.name:file_hash(p) for p in sorted(bundle.iterdir()) if p.is_file()})
        code_files += ['ts_benchmark/data/master.py', 'ts_benchmark/data/master_dataset.py',
                       'scripts/run_master_experiment.sh',
                       'ts_benchmark/baselines/custom_losses.py',
                       'ts_benchmark/baselines/composite_trading_loss.py',
                       'ts_benchmark/baselines/master/__init__.py',
                       'ts_benchmark/baselines/master/network.py',
                       'ts_benchmark/baselines/master/master.py']
    result = dict(config=config, source_sha256=metadata['source_sha256'],
                dataset_sha256=metadata['dataset_sha256'],
                source_data_kind=metadata.get('source_data_kind', 'price'),
                step_offset=metadata['step_offset'],
                code_sha256={p:file_hash(ROOT / p) for p in code_files})
    if master_data is not None:
        result['master_data'] = master_data
    return result


def prepare(config_path, config, tasks, out_root):
    from b3_experiment_io import prepare_dataset
    if config.get('model_specs'):
        from ts_benchmark.baselines.ordinal_model_adapter import validate_model
        checked = set()
        for task in tasks:
            if task['modelo'] not in checked:
                _, hyper, _, _ = model_spec(config, task)
                validate_model(**hyper)
                checked.add(task['modelo'])
    source = Path(os.environ.get('DATASET_PATH', ROOT / 'dataset/forecasting/b3_daily_tfb_new.csv')).resolve()
    dataset = ROOT / 'dataset/forecasting/b3_log_returns_new.csv'
    with locked(dataset.parent / '.b3_new_prepare.lock'):
        metadata = prepare_dataset(source, dataset, source_data_kind=config['source_data_kind'])
        print(f"Entrada: {config['source_data_kind']}; {metadata['n_rows']} pregões de log-retornos, "
              f"{metadata['n_assets']} ativos; step inicial={metadata['step_offset']}.", flush=True)
        validate_lengths(config, tasks, metadata)
        if 'MASTER' in config['models']:
            from ts_benchmark.data.master import load_master_bundle
            hyper = config['model_defaults']['MASTER']
            load_master_bundle(hyper['master_bundle_dir'], hyper['master_returns_path'],
                               asset_blocks=hyper.get('master_asset_blocks', ('retornos', 'ohlcv', 'atividade')),
                               use_market=hyper.get('master_use_market', True),
                               expected_assets=hyper.get('master_expected_assets'),
                               expected_dates=hyper.get('master_expected_dates'))
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


def benchmark_command(config, task, dataset, n_rows, result_dir, model_artifact_dir=None):
    model, hyper, adapter, deterministic = model_spec(config, task)
    if task.get('momentum_mode') == 'residual_output':
        if model_artifact_dir is None:
            raise ValueError('Momentum residual requer diretório para momentum_state.json')
        hyper['momentum_state_path'] = str(model_artifact_dir / 'momentum_state.json')
    if task['modelo'] == 'MASTER' and model_artifact_dir is not None:
        hyper.update(master_audit_path=str(model_artifact_dir / 'master_training.json'),
                     master_checkpoint_path=str(model_artifact_dir / 'master_checkpoint.pt'))
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


def record_execution_metadata(paths, parquet_hash, signature, extra=None, task=None):
    """Record the successful worker's host, never the submission/requested node.

    Sidecars travel with forecasts copied locally, without changing Parquet or
    completed_tasks.csv schemas. The hash binds each record to its prediction.
    Call only after export succeeds; already-completed tasks keep their origin.
    """
    hashes = json.loads(parquet_hash) if isinstance(parquet_hash, str) and parquet_hash.startswith('{') else None
    execution = dict(schema_version=1, hostname=socket.gethostname(),
                     completed_at_utc=datetime.now(timezone.utc).isoformat(),
                     design_sha256=signature)
    for variable in ('SLURM_NODENAME', 'SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID',
                     'SLURM_ARRAY_TASK_ID', 'SLURM_JOB_PARTITION', 'SLURM_JOB_NODELIST'):
        key = 'slurm_node_name' if variable == 'SLURM_NODENAME' else variable.lower()
        execution[key] = os.environ.get(variable, '')
    for horizon, path in paths.items():
        row = dict(execution, parquet_sha256=hashes[horizon] if hashes is not None else parquet_hash)
        if extra is not None:
            row.update(extra)
        if (task is not None and task['loss'] == 'rankmix_proximity'
                and task.get('rank_mix_proximity_components')):
            row['rank_mix_proximity_components'] = list(task['rank_mix_proximity_components'])
        atomic_text(path / 'execution_metadata.json', json.dumps(row, indent=2) + '\n')


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
    heads = task_head_settings(config, task)
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
        parquet = output / (heads['primary_file'] if heads is not None else 'part.parquet')
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
            if finished and heads is not None:
                from scripts.itransformer_task_heads import validate_completed
                finished = validate_completed(output, heads, signature, prior['parquet_sha256'])
        if finished:
            if task.get('momentum_mode') == 'residual_output':
                from b3_experiment_io import validate_momentum_export
                validate_momentum_export(output)
            print(f'Já concluído: {output}', flush=True)
            return
        panel = load_panel(Path(metadata['dataset']))
        split = split_counts(config, task, len(panel))
        execution_extra = ranking_protocol_settings(config, task)
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
                                          keep_csv=os.environ.get('MANTER_CSV','FALSE').upper() == 'TRUE',
                                          step_offset=metadata['step_offset'])
            parquet_hash = json.dumps({h:file_hash(path/'part.parquet') for h, path in paths.items()}, sort_keys=True)
        elif heads is not None:
            from scripts.itransformer_task_heads import fit_and_export
            _, hyper, _, _ = model_spec(config, task)
            print(f'TASK={index}: iTransformer {heads["task_head_mode"]}, janelas={heads["rank_windows"]}: {output}', flush=True)
            execution_extra = fit_and_export(config, task, hyper, panel, split['n_pretest'], output,
                                            step_offset=metadata['step_offset'])
            parquet_hash = execution_extra['files_sha256'][heads['primary_file']]
        else:
            result_root = Path(os.environ.get('RESULT_ROOT', '/snfs2/igor.viveiros/result/b3_new'))
            result_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='task_', dir=result_root) as temporary:
                command = benchmark_command(config, task, metadata['dataset'], len(panel), Path(temporary), output)
                print(f'TASK={index}: {output}', flush=True)
                subprocess.run(command, cwd=ROOT, check=True)
                export_predictions(Path(temporary), panel, split['n_pretest'], task['pred_len'], output,
                                   keep_csv=os.environ.get('MANTER_CSV','FALSE').upper() == 'TRUE',
                                   step_offset=metadata['step_offset'])
                if task.get('momentum_mode') == 'residual_output':
                    from b3_experiment_io import export_momentum_scores
                    export_momentum_scores(panel, split['n_pretest'], task['pred_len'], output,
                                           step_offset=metadata['step_offset'])
            parquet_hash = file_hash(parquet)
        row = audit_row(config, task, metadata, out_root, signature)
        row.update(slurm_job_id=os.environ.get('SLURM_JOB_ID',''), parquet_sha256=parquet_hash)
        record_execution_metadata(paths, parquet_hash, signature, extra=execution_extra, task=task)
        record_completed(directory, row)
        print(f'OK: {len(paths)} arquivo(s) Parquet em {directory}', flush=True)


def array_chunks(n_tasks, chunk_size=1000):
    if chunk_size < 1:
        raise ValueError('ARRAY_CHUNK_SIZE deve ser positivo')
    return [(offset, min(chunk_size, n_tasks-offset)) for offset in range(0,n_tasks,chunk_size)]


def sbatch_command(mode, offset, count, previous=None):
    """Build the same resource request for submission and dry-run, without I/O."""
    max_jobs = int(os.environ.get('MAX_GPU_JOBS','6'))
    if max_jobs < 1:
        raise ValueError('MAX_GPU_JOBS deve ser positivo')
    logs = Path(os.environ.get('LOG_ROOT', ROOT / 'logs/b3_new'))
    command = ['sbatch','--parsable','--export=ALL', '-p',os.environ.get('GPU_PARTITION','medusas_dev'),
               '--time='+os.environ.get('GPU_TIME','08:00:00'),
               f'--array=0-{count-1}%{max_jobs}', f'--job-name=b3_{mode}',
               f'--output={logs}/{mode}_%A_%a.out', f'--error={logs}/{mode}_%A_%a.err']
    for variable, flag in [('NODELIST', '--nodelist'), ('CPUS_PER_TASK', '--cpus-per-task'), ('MEMORY', '--mem')]:
        if os.environ.get(variable):
            command.append(f'{flag}={os.environ[variable]}')
    # Keep defaults for direct Python callers; an explicitly empty GRES
    # must stay empty, including in the temporal/composite launchers.
    gres = os.environ.get('GRES', '' if mode == 'var' else 'gpu:1')
    if gres:
        command.append(f'--gres={gres}')
    # Optional dependency chains independent wrapper branches without changing
    # the existing within-branch array chaining or legacy submissions.
    dependency = previous if previous is not None else os.environ.get('B3_AFTER_JOB', '')
    if dependency:
        if not isinstance(dependency, str) or not re.fullmatch(r'[0-9]+', dependency):
            raise ValueError('B3_AFTER_JOB/dependencia deve ser um ID Slurm numerico.')
        command += [f'--dependency=afterany:{dependency}']
    launcher = os.environ.get('B3_WORKER_LAUNCHER')
    if launcher:
        command += [launcher, '--on-node', mode, 'worker', '--offset', str(offset)]
    else:
        command += [str(ROOT/'scripts'/LAUNCHERS[mode]),'worker','--offset',str(offset)]
    return command


def submit(mode, tasks, out_root, config_path):
    if shutil.which('sbatch') is None:
        raise RuntimeError('sbatch não encontrado; submeta os scripts no cluster')
    chunk_size = int(os.environ.get('ARRAY_CHUNK_SIZE','1000'))
    if shutil.which('scontrol'):
        result = subprocess.run(['scontrol','show','config'], capture_output=True, text=True, check=True)
        match = re.search(r'MaxArraySize\s*=\s*(\d+)', result.stdout)
        if match:
            chunk_size = min(chunk_size, int(match.group(1)))
    logs = Path(os.environ.get('LOG_ROOT', ROOT / 'logs/b3_new'))
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TFB_ROOT=str(ROOT), B3_CONFIG=str(config_path), OUT_ROOT=str(out_root))
    previous = None
    for offset, count in array_chunks(len(tasks), chunk_size):
        command = sbatch_command(mode, offset, count, previous)
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
        protocol = ranking_protocol_settings(config)
        if protocol is not None:
            if protocol['cross_construction'] == 'origin_anchored_windows':
                print('Ranking: origin_anchored_windows; somas desde a origem; '
                      f"janelas={protocol['rank_windows']} filtradas por K<=H no mesmo treino; "
                      'K_loss=1 e identificador neutro.', flush=True)
            else:
                print('Ranking: blocks; blocos futuros disjuntos de K_loss pregoes.', flush=True)
        if momentum_options(config)[0] != 'off':
            _, lookback, eta = momentum_options(config)
            print(f'Momentum residual: LB={lookback or "LB do modelo"}; eta={eta:g}; '
                  f"ganho inicial={config['loss'].get('momentum_gain_init', 0.0):g}; "
                  f"diagnostico={config['loss'].get('momentum_diagnostics', False)}; "
                  'controle temporal inalterado; scores em ranking_scores.parquet.', flush=True)
            print(f"Taxa de aprendizado: LR={config['training']['lr']:g}; "
                  f"regra={config['training'].get('lradj', 'default do modelo')}.", flush=True)
        input_mode, input_lb = momentum_input_options(config)
        if input_mode != 'off':
            print(f'Momentum na entrada: modo={input_mode}; LB={input_lb or "LB do modelo"}; '
                  'projecao aprendida no encoder; sem soma fixa na saida; previsoes em part.parquet.', flush=True)
        if config['training'].get('validation_mode') == 'ordered_sample_mean':
            print('Validacao: ordem fixa e media ponderada pelo numero de amostras.', flush=True)
        if set(config['cross_losses']) & RANK_MIX_LOSSES:
            mixture = rank_mix_options(config)
            print('Mistura ListNet/ListMLE/ApproxNDCG: '
                  f"pesos={mixture['weights']}; modo={mixture['mode']}; "
                  f"regioes={mixture['regions']}; fracao={mixture['fraction']:g}.", flush=True)
            selected = rank_mix_proximity_options(config)
            if selected is not None:
                print('Proximity seletiva na mistura: ' + ', '.join(selected)
                      + '; demais componentes usam a loss original.', flush=True)
        pairs = sum(k <= h and h % k == 0 for h in config['horizons'] for k in config['trade_windows'])
        factors = [len(config[key]) for key in ('train_percentages', 'models', 'lookbacks')]
        variants = sum(len(cross_deltas(config, loss)) for loss in config['cross_losses'])
        factors += [pairs, len(config['lambdas']), variants]
        print('Grade: ' + ' × '.join(map(str, factors)) + f' = {len(tasks)} treinamentos.', flush=True)
        counts = {loss: sum(t['loss'] == loss for t in tasks) for loss in config['cross_losses']}
        print('Por loss: ' + '; '.join(f'{loss}={count}' for loss, count in counts.items()) + '.', flush=True)
        print(f'{pairs} pares H/K válidos; {variants} combinações efetivas de loss/delta.', flush=True)
        if 'listnet' in config['cross_losses']:
            print('ListNet: delta não se aplica; uma execução por combinação, registrada como delta=0.', flush=True)
        other_global = [loss for loss in config['cross_losses']
                        if loss in {'listmle', 'approxndcg', 'spearman'}]
        if other_global:
            print(', '.join(other_global) + ': delta não se aplica; uma execução por combinação.', flush=True)
        if any(loss in PROXIMITY_LOSSES for loss in config['cross_losses']):
            mask = validate_cross_mask(config['loss']['loss_cross_mask'])
            mask_type, mask_a = mask[:2]
            mask_mode = mask[2] if len(mask) >= 3 else 'natural'
            ties = 'empates com peso zero' if mask_mode == 'complementary' or mask_type == 'intermediate_band' else 'inclui empates'
            shape_note = f', formato={list(mask[3:])}' if len(mask) > 3 else ''
            print(f'Proximity: máscara={mask_type}, a={mask_a!r}, modo={mask_mode}{shape_note}; {ties}; '
                  'delta não se aplica, sem repetir DELTAS.', flush=True)
    if args.dry_run:
        chunks = array_chunks(len(tasks), int(os.environ.get('ARRAY_CHUNK_SIZE','1000')))
        print(f'{len(chunks)} array(s), tamanho máximo {max(c for _,c in chunks)}; último: 0-{chunks[-1][1]-1}')
        print('sbatch do primeiro array (prévia; não consulta MaxArraySize do cluster):')
        print(shlex.join(sbatch_command(args.mode, *chunks[0])))
        return
    if not args.prepare and shutil.which('sbatch') is None:
        raise RuntimeError('sbatch não encontrado; use --dry-run ou --prepare fora do cluster')
    prepare(config_path,config,tasks,out_root)
    if not args.prepare:
        submit(args.mode,tasks,out_root,config_path)


if __name__ == '__main__':
    main()

