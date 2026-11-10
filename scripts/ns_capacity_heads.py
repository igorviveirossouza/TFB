#!/usr/bin/env python3
"""Frozen, resumable Slurm experiments using native TFB fitting and inference.

The plan is stdlib-only. Workers load balanced economic log returns, fit once
on the pre-test block with internal validation, and forecast all test origins.
"""
import argparse
import contextlib
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, allow_nan=False)
        name = f.name
    os.replace(name, path)


def words(name, default, cast=str):
    result = [cast(x) for x in os.environ.get(name, default).split()]
    if not result or len(result) != len(set(result)):
        raise ValueError(f'{name}: lista vazia ou repetida')
    return result


def positive(name, default, cast=float):
    value = cast(os.environ.get(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} deve ser positivo e finito')
    return value


def configuration(study):
    if study not in {'capacity', 'heads', 'heads_restored'}:
        raise ValueError('Estudo invalido')
    if os.environ.get('MODEL', 'Nonstationary_Transformer') != 'Nonstationary_Transformer':
        raise ValueError('Este estudo arquitetural e exclusivo do Nonstationary_Transformer')
    if any(os.environ.get(k, 'off') != 'off' for k in ['MOMENTUM_MODE', 'MOMENTUM_INPUT_MODE']):
        raise ValueError('Momentum deve permanecer off: benchmark externo')
    base = dict(d_model=128, d_ff=256, e_layers=2, d_layers=1, n_heads=8,
                p_hidden_dims=[32, 32], p_hidden_layers=2, dropout=.1, factor=3)
    expanded = dict(base, d_model=256, d_ff=512, e_layers=3, d_layers=2,
                    p_hidden_dims=[64, 64])
    profiles = json.loads(os.environ.get('NS_CAPACITY_SPECS', json.dumps(
        {'base': base, 'expanded': expanded} if study == 'capacity' else {'base': base})))
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError('NS_CAPACITY_SPECS exige objeto JSON nao vazio')
    allowed = set(base) | {'rank_head_width'}
    for label, profile in profiles.items():
        if not re.fullmatch('[A-Za-z0-9_-]+', label) or not isinstance(profile, dict) or set(profile) - allowed:
            raise ValueError(f'Perfil de capacidade invalido: {label}')
        profiles[label] = profile = dict(base, **profile)
        for key in allowed - {'p_hidden_dims', 'dropout', 'rank_head_width'}:
            if type(profile[key]) is not int or profile[key] <= 0:
                raise ValueError(f'{key} exige inteiro positivo')
        dims = profile['p_hidden_dims']
        if not isinstance(dims, list) or len(dims) != profile['p_hidden_layers'] or any(type(x) is not int or x <= 0 for x in dims):
            raise ValueError('p_hidden_dims deve corresponder a p_hidden_layers')
        if profile['d_model'] % profile['n_heads'] or not 0 <= profile['dropout'] < 1:
            raise ValueError('d_model deve ser divisivel por n_heads; dropout em [0,1)')
        if 'rank_head_width' in profile and (type(profile['rank_head_width']) is not int or profile['rank_head_width'] <= 0):
            raise ValueError('rank_head_width deve ser inteiro positivo')
        if study == 'heads_restored' and 'rank_head_width' in profile:
            raise ValueError('heads_restored usa projecoes lineares equivalentes; remova rank_head_width')
    if len({signature(x) for x in profiles.values()}) != len(profiles):
        raise ValueError('Perfis de capacidade duplicados')
    weights = [float(x) for x in os.environ.get('RANK_MIX_WEIGHTS', '.5 .5 0').split()]
    if len(weights) != 3 or any(not math.isfinite(w) or w < 0 for w in weights) or abs(sum(weights) - 1) > 1e-8:
        raise ValueError('RANK_MIX_WEIGHTS exige tres pesos nao negativos somando 1')
    modes = words('RANK_MIX_MODES', 'global')
    if set(modes) - {'global', 'regional'}:
        raise ValueError('RANK_MIX_MODES: global/regional')
    regions = os.environ.get('RANK_MIX_REGIONS', 'top bottom top').split()
    fraction = positive('RANK_MIX_FRACTION', '0.3333333333333333')
    if len(regions) != 3 or set(regions) - {'top', 'bottom'} or fraction > .5:
        raise ValueError('Regioes/fracao invalidas')
    losses = words('CROSS_LOSSES_VALUES', 'rankmix rankmix_proximity')
    accepted = {'rankmix', 'ranknet', 'hinge', 'listnet', 'listmle', 'approxndcg', 'spearman'}
    if any(x.removesuffix('_proximity') not in accepted for x in losses):
        raise ValueError('Loss de ranking nao suportada neste estudo')
    if any(x.removesuffix('_proximity') + ('_proximity' if not x.endswith('_proximity') else '') not in losses for x in losses):
        raise ValueError('Inclua cada loss original e sua versao Proximity')
    mask = [os.environ.get('NS_MASK_TYPE', 'cauchy'), positive('NS_MASK_A', '.025'), os.environ.get('NS_MASK_MODE', 'natural')]
    # Keep kernel families/shapes consistent with the existing validator.
    from ts_benchmark.baselines.proximity_losses import validate_cross_mask
    mask += [float(x) for x in os.environ.get('NS_MASK_SHAPE', '').split()]
    mask = list(validate_cross_mask(mask))
    axes = dict(lookbacks=words('LOOKBACKS_VALUES', '32 256 512', int),
                horizons=words('HORIZON_VALUES', '1 25', int),
                k_loss=words('K_LOSS_VALUES', '1', int),
                lambdas=words('LAMBDAS_VALUES', '0 .5 1', float),
                seeds=words('SEEDS_VALUES', os.environ.get('NS_SEED', '2026'), int),
                temporals=words('TEMPORAL_LOSSES_VALUES', 'mse'),
                train=words('TRAIN_VALUES', '80', int),
                rank_windows=words('RANK_WINDOWS_VALUES', '1 5 10 15 20 25', int))
    for key in ['lookbacks', 'horizons', 'k_loss', 'rank_windows']:
        if min(axes[key]) < 1:
            raise ValueError(f'{key} exige inteiros positivos')
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in axes['lambdas']) or not {0, 1}.issubset(axes['lambdas']):
        raise ValueError('LAMBDAS_VALUES deve incluir 0 e 1, dentro de [0,1]')
    if set(axes['temporals']) - {'mse', 'huber', 'mae'} or not 0 < min(axes['train']) <= max(axes['train']) < 100:
        raise ValueError('Loss temporal ou corte invalido')
    if any(not 0 <= s <= 4294967295 for s in axes['seeds']):
        raise ValueError('Seed fora do intervalo uint32')
    if study in {'heads', 'heads_restored'} and axes['k_loss'] != [1]:
        raise ValueError('Em heads use RANK_WINDOWS_VALUES; K_LOSS_VALUES deve permanecer 1')
    head_modes = words('TASK_HEAD_MODES', 'shared separate') if study in {'heads', 'heads_restored'} else ['shared']
    if set(head_modes) - {'shared', 'separate'}:
        raise ValueError('TASK_HEAD_MODES: shared/separate')
    training = dict(batch_size=positive('NS_BATCH_SIZE', '16', int), num_epochs=positive('NS_NUM_EPOCHS', '100', int),
                    patience=positive('NS_PATIENCE', '10', int), lr=positive('NS_LEARNING_RATE', '.0001'),
                    lradj=os.environ.get('NS_LRADJ') or 'type1',
                    validation_mode='ordered_sample_mean', norm=True, parallel_strategy=None)
    ratio = positive('NS_TRAIN_RATIO_IN_TV', '.875')
    if ratio >= 1 or training['lradj'] not in {'type1', 'type2', 'type3', 'constant'}:
        raise ValueError('Validacao/lradj invalido')
    loss_args = dict(loss='composite_trading', loss_data_kind='log_return', loss_score_kind=os.environ.get('NS_SCORE_KIND', 'simple_return'),
                     loss_inverse_norm=True, loss_cross_scale=positive('NS_CROSS_SCALE', '1'),
                     loss_cross_score_normalization=os.environ.get('NS_CROSS_SCORE_NORMALIZATION', 'zscore'),
                     loss_cross_mask=mask, loss_rank_mix_weights=weights, loss_rank_mix_regions=regions,
                     loss_rank_mix_fraction=fraction, loss_huber_delta=positive('NS_HUBER_DELTA', '.5'),
                     loss_ranknet_alpha=positive('NS_RANKNET_ALPHA', '1'),
                     loss_listnet_tau=positive('NS_LISTNET_TAU', '1'), loss_listmle_tau=positive('NS_LISTMLE_TAU', '1'),
                     loss_approxndcg_tau=positive('NS_APPROXNDCG_TAU', '1'), loss_spearman_tau=positive('NS_SPEARMAN_TAU', '1'),
                     loss_hinge_margin=positive('NS_HINGE_MARGIN', '.1'), momentum_mode='off', momentum_input_mode='off')
    if loss_args['loss_score_kind'] not in {'simple_return', 'log_return'} or loss_args['loss_cross_score_normalization'] not in {'zscore', 'none'}:
        raise ValueError('Score kind/normalization invalido')
    config = dict(schema_version=1, study=study, profiles=profiles, **axes, cross_losses=losses,
                mix_modes=modes, head_modes=head_modes, training=training, loss_args=loss_args, validation_ratio=ratio)
    if study == 'heads_restored':
        config.update(schema_version=2, ranking_construction='restored_daily_projection_v1')
    return config


def tasks(config):
    out = []
    for capacity, seed, train, lb, h, lam in itertools.product(config['profiles'], config['seeds'], config['train'], config['lookbacks'], config['horizons'], config['lambdas']):
        ks = [k for k in config['k_loss'] if k <= h and h % k == 0] if config['study'] == 'capacity' else [1]
        windows = [k for k in config['rank_windows'] if k <= h]
        if not ks or not windows:
            raise ValueError(f'H={h} sem janela valida')
        for temporal in (config['temporals'] if lam < 1 else ['none']):
            for loss, mode, k, head in ([('none', 'none', 0, 'shared')] if lam == 0 else
                    [(loss, mode, k, head) for loss in config['cross_losses']
                     for mode in (config['mix_modes'] if loss.startswith('rankmix') else ['global'])
                     for k in ks for head in config['head_modes']]):
                row = dict(capacity=capacity, seed=seed, treino=train, lookback=lb, pred_len=h,
                           lambda_value=lam, temporal_loss=temporal, loss=loss, mix_mode=mode, k_loss=k,
                           head_mode=head, rank_windows=windows, architecture=config['profiles'][capacity])
                if config['study'] == 'heads_restored':
                    row['ranking_construction'] = config['ranking_construction']
                row['fit_id'] = signature(row)[:20]
                out.append(row)
    if len({r['fit_id'] for r in out}) != len(out):
        raise ValueError('Grade duplicada')
    return out


def model_options(config, task):
    return dict(config['training'], **task['architecture'], **config['loss_args'],
                seq_len=task['lookback'], horizon=task['pred_len'], pred_len=task['pred_len'],
                loss_temporal='mse' if task['temporal_loss'] == 'none' else task['temporal_loss'],
                loss_cross='rankmix' if task['loss'] == 'none' else task['loss'],
                loss_cross_lambda=task['lambda_value'], loss_trade_window=max(1, task['k_loss']),
                loss_rank_mix_mode='global' if task['mix_mode'] == 'none' else task['mix_mode'])


def code_files():
    return [ROOT / 'scripts/ns_capacity_heads.py', ROOT / 'scripts/b3_experiment_io.py',
            *sorted((ROOT / 'ts_benchmark').rglob('*.py'))]


@contextlib.contextmanager
def lock(path):
    import fcntl
    with Path(path).open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def submit(config, root):
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / '.submission.lock'):
        source = Path(os.environ.get('NS_RETURNS_FILE', ROOT / 'dataset/forecasting/b3_daily_tfb_new.csv')).resolve()
        from b3_experiment_io import load_panel
        panel = load_panel(source)
        if list(panel.index) != list(range(1, len(panel) + 1)) or panel.shape[1] < 3:
            raise ValueError('Fonte deve conter log-retornos com date=1..T e ao menos 3 ativos')
        grid = tasks(config)
        for t in grid:
            cut = len(panel) * t['treino'] // 100
            fit = int(cut * config['validation_ratio'])
            if fit - t['lookback'] - t['pred_len'] + 1 < config['training']['batch_size'] or cut - fit < t['pred_len'] or len(panel) - cut < t['pred_len']:
                raise ValueError('Dados insuficientes para treino, validacao ou teste')
        files = code_files()
        hashes = {str(p.relative_to(ROOT)): sha(p) for p in files}
        contract = dict(config=config, tasks=grid, source_sha256=sha(source), code_sha256=hashes,
                        n_dates=len(panel), assets=list(panel.columns))
        design_sha = signature(contract)
        plan_path = root / 'experiment_config.json'
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            if plan['design_sha256'] != design_sha:
                raise ValueError('Configuracao/dados/codigo alterados: use outra EXPERIMENT_TAG')
        else:
            frozen = root / 'frozen_code'
            for p in files:
                dest = frozen / p.relative_to(ROOT)
                dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
            dataset = root / 'returns.csv'; shutil.copyfile(source, dataset)
            if sha(dataset) != contract['source_sha256'] or any(sha(frozen / p) != v for p, v in hashes.items()):
                raise ValueError('Fonte/codigo mudou durante o congelamento')
            plan = dict(contract, design_sha256=design_sha, original_source=str(source),
                        source_file='returns.csv', code_dir='frozen_code')
            atomic_json(plan_path, plan)
        chunk = positive('NS_ARRAY_CHUNK_SIZE', '1000', int)
        cfg = subprocess.run(['scontrol', 'show', 'config'], capture_output=True, text=True)
        match = re.search(r'MaxArraySize\s*=\s*(\d+)', cfg.stdout)
        if cfg.returncode or not match:
            raise RuntimeError('Nao foi possivel consultar MaxArraySize: ' + cfg.stderr.strip())
        chunk = min(chunk, int(match.group(1)))
        max_jobs = positive('NS_MAX_CONCURRENT_JOBS', '3', int)
        root.joinpath('logs').mkdir(exist_ok=True)
        previous = os.environ.get('NS_AFTER_JOB', '')
        parallel = os.environ.get('NS_PARALLEL_BRANCHES', '0')
        if parallel not in {'0', '1'} or (previous and not previous.isdigit()):
            raise ValueError('NS_PARALLEL_BRANCHES ou NS_AFTER_JOB invalido')
        for offset in range(0, len(grid), chunk):
            mark = root / f'submission_{offset}.json'
            if mark.exists():
                old = json.loads(mark.read_text())
                if not old.get('job_id'):
                    raise RuntimeError(f'Submissao incerta: confira {mark} e o Slurm')
                print(f'Array ja submetido: {old["job_id"]}; nao significa concluido', flush=True)
                continue
            command = ['sbatch', '--parsable', '--export=ALL', '-p', os.environ.get('NS_PARTITION', 'gorgonas_dev'),
                       '--time=' + os.environ.get('NS_TIME_LIMIT', '02:00:00'),
                       '--cpus-per-task=' + str(positive('NS_CPUS_PER_TASK', '3', int)),
                       f'--array=0-{min(chunk, len(grid)-offset)-1}%{max_jobs}', '--job-name=ns_' + config['study'],
                       '--output=' + str(root / 'logs/%A_%a.out'), '--error=' + str(root / 'logs/%A_%a.err')]
            for env, flag in [('NS_GRES', '--gres='), ('NS_NODELIST', '--nodelist='), ('NS_MEMORY', '--mem=')]:
                if os.environ.get(env): command.append(flag + os.environ[env])
            if previous: command.append('--dependency=afterany:' + previous)
            # Invoke a frozen shell worker; paths are passed as arguments, never interpolated into --wrap.
            worker_sh = root / 'worker.sh'
            if not worker_sh.exists():
                worker_sh.write_text('#!/usr/bin/env bash\nset -Eeuo pipefail\nexec "$1" "$2" --worker "$3" --offset "$4"\n')
            command += [str(worker_sh), sys.executable, str(root / 'frozen_code/scripts/ns_capacity_heads.py'), str(plan_path), str(offset)]
            atomic_json(mark, dict(command=command, job_id=None))
            result = subprocess.run(command, capture_output=True, text=True)
            job = result.stdout.strip().split(';')[0]
            if result.returncode or not job.isdigit():
                raise RuntimeError(f'Submissao falhou/incerta; consulte {mark}: {result.stderr} {result.stdout}')
            atomic_json(mark, dict(command=command, job_id=job))
            print(f'Job {job}: tarefas globais {offset}..{min(offset+chunk,len(grid))-1}', flush=True)
            if parallel == '0': previous = job


def run_worker(plan_path, index):
    import numpy as np
    import pandas as pd
    import torch
    from b3_experiment_io import load_panel
    from ts_benchmark.baselines.ns_task_heads import NSTaskAdapter
    plan_path = Path(plan_path).resolve(); root = plan_path.parent
    plan = json.loads(plan_path.read_text()); cfg = plan['config']; task = plan['tasks'][index]
    if sha(root / plan['source_file']) != plan['source_sha256']:
        raise ValueError('Fonte congelada alterada')
    if any(sha(root / plan['code_dir'] / p) != value for p, value in plan['code_sha256'].items()):
        raise ValueError('Codigo congelado alterado')
    dest = root / ('fit=' + task['fit_id']); dest.mkdir(exist_ok=True)
    with lock(dest / '.worker.lock'):
        record_path = dest / 'result.json'
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record['design_sha256'] != plan['design_sha256'] or any(sha(dest / p) != value for p,value in record['files_sha256'].items()):
                raise ValueError('Resultado existente inconsistente')
            print('Concluido anteriormente:', task['fit_id']); return
        from ts_benchmark.utils.random_utils import fix_all_random_seed
        fix_all_random_seed(task['seed'])
        torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
        panel = load_panel(root / plan['source_file'])
        if list(panel.columns) != plan['assets'] or len(panel) != plan['n_dates']:
            raise ValueError('Painel divergente do contrato')
        # Match b3_experiment_io.prepare_dataset's synthetic calendar exactly.
        panel.index = pd.date_range('2000-01-02', periods=len(panel), freq='D', name='date')
        cut = len(panel) * task['treino'] // 100
        options = model_options(cfg, task)
        adapter_class = NSTaskAdapter
        if cfg['study'] == 'heads_restored':
            from ts_benchmark.baselines.ns_restored_task_heads import NSRestoredTaskAdapter
            adapter_class = NSRestoredTaskAdapter
        model = adapter_class(study=cfg['study'], head_mode=task['head_mode'], rank_windows=task['rank_windows'], **options)
        model.forecast_fit(panel.iloc[:cut], train_ratio_in_tv=cfg['validation_ratio'])
        h, lb = task['pred_len'], task['lookback']
        valid_returns = task['head_mode'] != 'separate' or task['lambda_value'] < 1
        origins = np.arange(cut, len(panel)-h+1)
        forecasts, rankings = [], []
        batch_size = cfg['training']['batch_size']
        class Maker:
            def __init__(self, selected): self.selected = selected
            def make_batch(self, batch_size, win_size):
                return dict(input=np.stack([panel.iloc[o-win_size:o].values for o in self.selected]),
                            time_stamps=np.stack([panel.index[o-win_size:o].values for o in self.selected]), covariates=None)
        for start in range(0, len(origins), batch_size):
            selected = origins[start:start+batch_size]
            pred = model.batch_forecast(h, Maker(selected))
            if pred.shape != (len(selected), h, panel.shape[1]) or (valid_returns and not np.isfinite(pred).all()):
                raise ValueError('Previsoes invalidas')
            forecasts.append(pred)
            if task['head_mode'] == 'separate':
                score = model.export_rank_scores
                if score.shape != pred.shape or not np.isfinite(score).all(): raise ValueError('Scores invalidos')
                rankings.append(score)
        pred = np.concatenate(forecasts)
        files = {}
        # At lambda=1 a separate temporal head is untrained: do not publish its values.
        def save_frame(frame, name):
            temporary = dest / (name + '.tmp'); frame.to_parquet(temporary, index=False)
            os.replace(temporary, dest / name); files[name] = sha(dest / name)
        if valid_returns:
            frame = pd.DataFrame(pred.reshape(-1, panel.shape[1]), columns=panel.columns)
            frame.insert(0, 'h', np.tile(np.arange(1,h+1), len(origins)))
            frame.insert(0, 'origin_step', np.repeat(origins, h)); frame.insert(0, 'step', frame.origin_step+frame.h)
            save_frame(frame, 'part.parquet')
        if rankings:
            rank = np.concatenate(rankings); ks = task['rank_windows']
            frame = pd.DataFrame(rank[:, np.array(ks)-1].reshape(-1, panel.shape[1]), columns=panel.columns)
            frame.insert(0, 'k_eval', np.tile(ks, len(origins))); frame.insert(0, 'origin_step', np.repeat(origins, len(ks)))
            save_frame(frame, 'ranking_scores.parquet')
        scaler = dict(mean=model.scaler.mean_.tolist(), scale=model.scaler.scale_.tolist())
        checkpoint_path = dest / 'checkpoint.pt.tmp'
        torch.save(dict(state_dict=model.check_point, scaler=scaler, options=options, task=task), checkpoint_path)
        os.replace(checkpoint_path, dest / 'checkpoint.pt'); files['checkpoint.pt'] = sha(dest / 'checkpoint.pt')
        semantics = 'direct_window_score' if rankings else 'cumulative_log_return'
        if rankings and cfg['study'] == 'heads_restored':
            semantics = 'cumulative_' + cfg['loss_args']['loss_score_kind']
        atomic_json(record_path, dict(task=task, design_sha256=plan['design_sha256'], files_sha256=files,
            parameter_count=sum(p.numel() for p in model.model.parameters()), return_forecast_valid=valid_returns,
            score_semantics=semantics,
            host=socket.gethostname(), slurm_job_id=os.environ.get('SLURM_JOB_ID'), slurm_node=os.environ.get('SLURMD_NODENAME'),
            torch_version=torch.__version__, device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
            scaler=scaler, effective_options=options))
        print('Concluido:', dest, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=['capacity', 'heads', 'heads_restored'])
    parser.add_argument('--plan', action='store_true'); parser.add_argument('--train', action='store_true')
    parser.add_argument('--worker'); parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.worker:
        index = args.index if args.index is not None else args.offset + int(os.environ['SLURM_ARRAY_TASK_ID'])
        run_worker(args.worker, index); return
    if not args.study or args.plan == args.train: parser.error('Use --study ... e exatamente --plan ou --train')
    config = configuration(args.study); grid = tasks(config)
    print(f'Estudo {args.study}: {len(grid)} treinos; {len(config["profiles"])} capacidade(s); lambdas={config["lambdas"]}', flush=True)
    print(f'LB={config["lookbacks"]}; H={config["horizons"]}; K_loss={config["k_loss"]}; treino={config["train"]}; seeds={config["seeds"]}', flush=True)
    print(f'Losses={config["cross_losses"]}; mistura={config["mix_modes"]}; pesos={config["loss_args"]["loss_rank_mix_weights"]}', flush=True)
    print('Momentum externo; lambda=0 compartilhado entre losses/modos; lambda=1 compartilhado entre losses temporais.', flush=True)
    print('Arquiteturas:', json.dumps(config['profiles']), flush=True)
    if args.study == 'heads': print('Janelas supervisionadas por origem:', config['rank_windows'], '(filtradas por H); escores diretos, sem soma de escores.', flush=True)
    if args.study == 'heads_restored':
        print('Projecoes lineares com mesma inicializacao; media/desvio por ativo restaurados nas duas saidas.', flush=True)
        print('Janelas supervisionadas por origem:', config['rank_windows'], '(filtradas por H); acumulacao apos restauracao das duas normalizacoes.', flush=True)
    tag = os.environ.get('EXPERIMENT_TAG', f'isolados/ns_{args.study}_t80_v1')
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_./=-]*', tag) or '..' in tag or '//' in tag or tag.endswith('/'):
        raise ValueError('EXPERIMENT_TAG invalida')
    root = Path(os.environ.get('PRED_ROOT', '/snfs2/igor.viveiros/previsoes/' + tag)).resolve()
    print('Saida:', root, flush=True)
    if args.train: submit(config, root)


if __name__ == '__main__':
    main()
