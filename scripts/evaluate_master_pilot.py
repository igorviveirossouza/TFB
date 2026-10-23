#!/usr/bin/env python3
"""Audit saved MASTER predictions and evaluate them with the existing CPU engines.

This entry point never trains, prepares training data or changes predictions.
The two engines keep their own immutable plans and resume completed backtests.
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

from backtest_cpu import pipeline as temporal
from backtest_composto import pipeline as composite
from backtest_cpu.io import read_json, read_table
from validate_master_pilot import validate_pilot

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ROOT / 'config/estrategias_master_piloto.json'


def validate_plans(plans):
    """Both families must evaluate the same origins, assets and economic protocol."""
    tasks = [task for plan in plans.values() for task in plan['tasks']]
    if (len(plans['temporal']['tasks']) != 1 or len(plans['composto']['tasks']) != 2
            or {t['loss'] for t in tasks} != {'mse', 'ranknet', 'ranknet_proximity'}
            or any(t['modelo'] != 'MASTER' or t['treino'] != 80
                   or t['lookback'] != 32 or t['pred_len'] != 5 for t in tasks)):
        raise ValueError('Planos devem conter exatamente as tres previsoes do piloto MASTER.')
    if len({(t['corte'], t['common_last_origin']) for t in tasks}) != 1:
        raise ValueError('Familias possuem origens/amostras comuns diferentes.')
    if plans['temporal']['inputs'] != plans['composto']['inputs']:
        raise ValueError('Familias possuem insumos/CDI diferentes.')
    for plan in plans.values():
        if (plan['k'] != [1, 5] or plan['slippage_bps'] != 0 or plan['cost_bps'] != 0
                or plan['strategies'] != read_json(STRATEGIES)):
            raise ValueError('Protocolo deve usar K=1/5, Top-9, duas compras e custos zero.')
    discovery = plans['composto']['discovery']
    if discovery['awaiting'] or not discovery['expected_complete']:
        raise ValueError('Grade composta incompleta; nenhuma avaliacao parcial sera iniciada.')


def export_comparison(run):
    """Plain tables for comparing objectives on the same test sample."""
    outputs = {}
    for source, name in [('ic_resumo', 'comparacao_ic_diario.csv'),
                         ('ic_acumulado_resumo', 'comparacao_ic_acumulado.csv'),
                         ('metricas', 'comparacao_carteiras.csv')]:
        frames = []
        for mode in ('temporal', 'composto'):
            frame = read_table(run / mode / 'resumos' / (source + '.parquet'))
            frame['familia'] = mode
            # IDs are local to each plan; this key is unique across families.
            frame['avaliacao_id'] = mode + ':' + frame.task_id.astype(str)
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True)
        if source == 'metricas' and (len(combined) != 12 or
                combined.duplicated(['avaliacao_id', 'k', 'estrategia']).any()):
            raise ValueError('Esperadas exatamente 12 carteiras distintas no piloto.')
        temporary = run / ('.' + name + '.tmp')
        combined.to_csv(temporary, index=False)
        os.replace(temporary, run / name)
        outputs[name] = str(run / name)
    return outputs


def _evaluate(args, run):
    report = validate_pilot(args.predictions)
    temporal.atomic_json(run / 'validacao_previsoes.json', report)
    if report['status'] != 'PASSED':
        raise ValueError('Auditoria das previsoes falhou: ' + report.get('error', str(report)))
    print('MASTER: tres previsoes auditadas. Iniciando ICs e backtests existentes.', flush=True)

    plans = {}
    for mode, engine in [('temporal', temporal), ('composto', composite)]:
        prep = argparse.Namespace(
            predictions=str(Path(args.predictions).resolve()), market=str(Path(args.market).resolve()),
            returns_file=str(Path(args.returns_file).resolve()), strategies=str(STRATEGIES),
            run_dir=str(run / mode), train=[80], k=[1, 5], slippage_bps=0., cost_bps=0.,
            dry_run=False, allow_local=args.allow_local)
        engine.prepare(prep)
        plans[mode] = read_json(run / mode / 'plano.json')
    validate_plans(plans)

    groups = {}
    for mode, entry in [('temporal', 'run_backtests.py'), ('composto', 'run_backtests_composto.py')]:
        directory = run / mode
        # The engines run their frozen copies, never a changing shared checkout.
        command = [sys.executable, '-B', str(directory / 'codigo' / entry), 'run',
                   '--run-dir', str(directory), '--workers', str(args.workers)]
        result = subprocess.run(command, check=False)
        if result.returncode:
            raise ValueError(f'Avaliacao {mode} falhou (exit={result.returncode}); consulte o log do job e {directory}/resumos/falhas.parquet, se existente.')
        state = read_json(directory / 'resumos/estado_execucao.json')
        if not state['completa']:
            raise ValueError(f'Avaliacao {mode} incompleta; consulte {directory}/resumos/falhas.parquet.')
        groups[mode] = state
    outputs = export_comparison(run)
    return {'status': 'PASSED', 'n_predictions': 3, 'n_portfolios': 12,
            'k_evaluation': [1, 5], 'k_loss': 1, 'groups': groups, 'tables': outputs}


def evaluate(args):
    run = temporal.storage_path(args.run_dir, args.allow_local)
    for raw in (args.predictions, args.market, args.returns_file):
        path = Path(raw).resolve()
        if run == path or run.is_relative_to(path) or path.is_relative_to(run):
            raise ValueError('Avaliacao deve ficar separada das previsoes e dos insumos.')
    if args.workers < 1 or args.workers > int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1)):
        raise ValueError('Numero de processos excede CPUs disponiveis ou e invalido.')
    run.mkdir(parents=True, exist_ok=True)
    with (run / '.evaluation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Ja existe uma avaliacao deste piloto em execucao.') from error
        temporal.atomic_json(run / 'avaliacao.json', {'status': 'RUNNING'})
        try:
            report = _evaluate(args, run)
        except Exception as error:
            temporal.atomic_json(run / 'avaliacao.json', {'status': 'FAILED', 'error': str(error)})
            raise
        temporal.atomic_json(run / 'avaliacao.json', report)
    print(f'MASTER: PASSED; 3 previsoes; 12 carteiras; K=1/5. Resultados: {run}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('predictions', 'market', 'returns-file', 'run-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--allow-local', action='store_true', help='Somente fixtures/testes.')
    args = parser.parse_args()
    try:
        evaluate(args)
    except Exception as error:
        print('ERRO MASTER: ' + str(error), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
