#!/usr/bin/env python3
"""Benchmark momentum separado, usando o motor congelado do backtest de referencia.

Nao relê previsoes nem modifica a execucao original. O sinal observado no
fechamento t e sum(r_corrigido[t-L+1:t]), isto e, log(P*_t/P*_(t-L)), com P*
um indice de retorno total. O sinal ordinal nao e uma previsao diaria nem
depende de H. Cada tarefa treino/L grava IC por K e carteiras em tasks/;
resumos/ reune IC acumulado resumido, metricas, falhas e estado da execucao.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
import uuid

# Inclusive durante os imports do snapshot: a referencia deve ser somente leitura.
sys.dont_write_bytecode = True
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


VERSION = "2026-10-16-momentum-v1"
FORMULA = "sum(log_retorno_corrigido[t-L+1:t])=log(P*_t/P*_(t-L));P*=indice_retorno_total"
SUMMARY_TABLES = ("ic_acumulado_resumo", "metricas", "falhas")
_WORKER_CONTEXT = None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_signature(plan):
    unsigned = dict(plan)
    signature = unsigned.pop("signature", None)
    if signature != json_digest(unsigned):
        raise ValueError("Assinatura do plano de referencia divergente.")


def import_frozen_engine(reference, plan):
    directory = reference / "codigo" / "backtest_cpu"
    expected = plan["code"]
    if not {"__init__.py", "core.py", "io.py", "pipeline.py"} <= set(expected):
        raise ValueError("Snapshot de referencia sem motor completo.")
    if {p.name for p in directory.glob("*.py")} != set(expected):
        raise ValueError("Arquivos do snapshot de referencia divergem do plano.")
    for name, sha in expected.items():
        if Path(name).name != name or not name.endswith(".py") or digest(directory/name) != sha:
            raise ValueError(f"Snapshot de referencia alterado: {name}")
    name = "_momentum_reference_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, directory/"__init__.py",
                                                 submodule_search_locations=[str(directory)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return SimpleNamespace(**{part: importlib.import_module(f"{name}.{part}")
                              for part in ("core", "io", "pipeline")})


def derive_tasks(reference_plan, n):
    temporal = [t for t in reference_plan["tasks"] if t["experimento"] == "temporal"]
    if not temporal:
        raise ValueError("Referencia sem tarefas temporais para definir a grade momentum.")
    ks = reference_plan["k"]
    if (not ks or any(type(k) is not int or k < 1 for k in ks)
            or len(set(ks)) != len(ks)):
        raise ValueError("Grade K invalida no plano de referencia.")
    result = []
    for train in sorted({t["treino"] for t in temporal}):
        if type(train) is not int or not 0 < train < 100:
            raise ValueError("Corte de treino invalido.")
        all_tasks = [t for t in reference_plan["tasks"] if t["treino"] == train]
        hmax = max(t["pred_len"] for t in all_tasks)
        cut, common = n * train // 100, n - hmax
        if (cut < 1 or common < cut or max(ks) > hmax
                or any(t["corte"] != cut or t["common_last_origin"] != common for t in all_tasks)):
            raise ValueError("Cortes/origens comuns/K inconsistentes na referencia.")
        group = [t for t in temporal if t["treino"] == train]
        for lookback in sorted({t["lookback"] for t in group}):
            if type(lookback) is not int or not 1 <= lookback <= cut:
                raise ValueError("Lookback sem historico completo no corte; inicio nao sera truncado.")
            horizons = sorted({t["pred_len"] for t in group if t["lookback"] == lookback})
            if max(ks) > max(horizons):
                raise ValueError("K sem horizonte temporal correspondente para este lookback.")
            result.append({"id": len(result), "treino": train, "lookback": lookback,
                           "backtest_source_id": len(result),
                           "experimento": "momentum", "modelo": "MOMENTUM",
                           "pred_len": None, "loss": "momentum", "lambda": "00",
                           "corte": cut, "common_last_origin": common,
                           "horizontes_referencia": horizons,
                           "relpath": f"treino={train}/experimento=momentum/modelo=MOMENTUM/lookback={lookback}"})
    return result


def momentum_scores(log_returns, origins, lookback):
    """t usa exclusivamente os L retornos terminados em t (indices 1-based)."""
    returns = np.asarray(log_returns, dtype=float)
    origins_raw = np.asarray(origins)
    if (returns.ndim != 2 or not np.isfinite(returns).all()
            or origins_raw.ndim != 1 or not len(origins_raw)
            or not np.isfinite(origins_raw).all()
            or np.any(origins_raw != np.floor(origins_raw))
            or type(lookback) is not int or lookback < 1):
        raise ValueError("Retornos/origens/lookback invalidos para momentum.")
    origins = origins_raw.astype(np.int64)
    if origins.min() < lookback or origins.max() > len(returns):
        raise ValueError("Historico insuficiente para momentum.")
    prefix = np.vstack([np.zeros(returns.shape[1]), np.cumsum(returns, axis=0)])
    return prefix[origins] - prefix[origins-lookback]


def validate_output_location(run, reference, plan, allow_local):
    if not allow_local and not run.is_relative_to(Path("/snfs2")):
        raise ValueError(f"Toda saida deve ficar no SNFS2: {run}")
    protected = [reference, Path(plan["inputs"]["base"]).resolve(),
                 Path(plan["inputs"]["retornos"]).resolve()]
    # Protege a arvore de previsoes sem abrir nenhum de seus arquivos.
    for task in plan["tasks"]:
        if "path" not in task:
            continue
        path = Path(task["path"]).resolve()
        roots = [p.parent for p in path.parents if p.name.startswith("treino=")]
        protected.append(roots[-1] if roots else path.parent)
    for path in protected:
        if run == path or run.is_relative_to(path) or path.is_relative_to(run):
            raise ValueError("Saida momentum deve estar separada da referencia, insumos e previsoes.")


def load_reference(args):
    reference = Path(args.reference_run).expanduser().resolve()
    run = Path(args.run_dir).expanduser().resolve()
    plan = read_json(reference/"plano.json")
    verify_signature(plan)
    validate_output_location(run, reference, plan, args.allow_local)
    state = read_json(reference/"resumos"/"estado_execucao.json")
    if (state.get("signature") != plan["signature"] or state.get("completa") is not True
            or state.get("total") != len(plan["tasks"])
            or state.get("COMPLETA") != len(plan["tasks"])
            or any(state.get(k) != 0 for k in ("PARCIAL", "ERRO", "PENDENTE"))):
        raise ValueError("Execucao de referencia nao esta completa ou resumo/plano divergem.")
    engine = import_frozen_engine(reference, plan)
    if engine.pipeline.runtime_versions() != plan["runtime"]:
        raise ValueError("Use o mesmo ambiente Python/bibliotecas da execucao de referencia.")
    market, actual = engine.io.load_bundle(plan["inputs"]["base"], plan["inputs"]["retornos"])
    if actual != plan["inputs"]:
        raise ValueError("Insumos de mercado/retornos/CDI mudaram desde a execucao de referencia.")
    tasks = derive_tasks(plan, market.n)
    strategies = plan["strategies"]
    if not strategies or len({s["id"] for s in strategies}) != len(strategies):
        raise ValueError("Estrategias ausentes/duplicadas na referencia.")
    for definition in strategies:
        engine.core.Strategy(**definition)
    for key in ("slippage_bps", "cost_bps"):
        if not np.isfinite(plan[key]) or not 0 <= plan[key] < 10000:
            raise ValueError("Custos/slippage invalidos na referencia.")
    result = {"schema_version": VERSION, "run_dir": str(run), "allow_local": args.allow_local,
              "reference_run": str(reference), "reference_signature": plan["signature"],
              "inputs": actual, "tasks": tasks, "k": sorted(plan["k"]),
              "strategies": strategies, "slippage_bps": plan["slippage_bps"],
              "cost_bps": plan["cost_bps"], "engine_code": plan["code"],
              "driver_sha256": digest(__file__), "runtime": engine.pipeline.runtime_versions(),
              "sinal": FORMULA, "pred_len": "nao_aplicavel_sem_duplicacao_por_H",
              "amostra_completa": "origens_diarias_corte_ate_N-K",
              "amostra_comum": "mesmas_origens_da_referencia_corte_ate_N-Hmax",
              "carteiras": "mesmas_origens_da_referencia_corte_ate_N-Hmax_passo_K",
              "regime": plan["regime"], "rf": plan["rf"], "rf_convencao": plan["rf_convencao"],
              "dividendos": plan["dividendos"], "distribuicao_ativos": plan["distribuicao_ativos"],
              "short": plan["short"]}
    result["signature"] = json_digest(result)
    return run, reference, result, market, engine


def add_meta(frame, task, plan):
    frame = frame.copy()
    for key in ("treino", "experimento", "modelo", "lookback", "loss", "lambda"):
        frame[key] = task[key]
    frame["pred_len"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame["task_id"] = task["id"]
    frame["slippage_bps"] = plan["slippage_bps"]
    frame["cost_bps"] = plan["cost_bps"]
    return frame


def valid_task(directory, signature):
    try:
        state = read_json(directory/"conclusao.json")
        required = {"ic_acumulado.parquet", "ic_acumulado_resumo.parquet", "falhas.parquet"}
        if state.get("signature") != signature or state.get("status") != "COMPLETA":
            return False
        required |= {"carteira_diaria.parquet", "carteira_periodos.parquet", "posicoes.parquet", "metricas.parquet"}
        return required <= set(state["files"]) and all(
            Path(name).name == name and digest(directory/name) == sha for name, sha in state["files"].items())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def execute_task(task, plan, market, engine, run):
    destination = run/"tasks"/task["relpath"]
    status_path = run/"status"/f"{task['id']:04d}.json"
    if valid_task(destination, plan["signature"]):
        state = read_json(destination/"conclusao.json")
        engine.pipeline.atomic_json(status_path, state)
        print(f"task={task['id']} REUTILIZADA treino={task['treino']} lookback={task['lookback']}", flush=True)
        return state
    start = time.perf_counter()
    stage = None
    try:
        if destination.exists():
            marker = destination/"conclusao.json"
            if marker.exists():
                try:
                    previous = read_json(marker)
                except ValueError:
                    previous = {}  # Publicacao interrompida desta pasta/plano.
                if previous.get("signature", plan["signature"]) != plan["signature"]:
                    raise ValueError("Tarefa existente pertence a outra execucao; nao sera substituida.")
        origins = np.arange(task["corte"], market.n)
        scores = momentum_scores(market.arrays["log_retorno_corrigido"], origins, task["lookback"])
        ics, outputs, failures = [], {}, []
        for k in plan["k"]:
            daily = origins[origins+k <= market.n]
            ics.append(engine.core.ic_frame(scores[:len(daily)], daily, k, market, task["common_last_origin"]))
            chosen = np.arange(task["corte"], task["common_last_origin"]+1, k)
            signal = scores[chosen-task["corte"]]
            for definition in plan["strategies"]:
                try:
                    result = engine.core.backtest(signal, chosen, k, market, engine.core.Strategy(**definition),
                                                  plan["slippage_bps"], plan["cost_bps"])
                    for name, frame in result.items():
                        outputs.setdefault(name, []).append(frame)
                except ValueError as error:
                    failures.append({"k": k, "estrategia": definition["id"], "erro": str(error)})
        ic = pd.concat(ics, ignore_index=True)
        tables = {name: pd.concat(frames, ignore_index=True) for name, frames in outputs.items()}
        tables.update(ic_acumulado=ic, ic_acumulado_resumo=engine.core.ic_summary(ic),
                      falhas=pd.DataFrame(failures, columns=["k", "estrategia", "erro"]))
        # Arquivo de metricas existe mesmo se todas as estrategias estiverem bloqueadas.
        tables.setdefault("metricas", pd.DataFrame(columns=["k", "estrategia"]))
        stage = Path(tempfile.mkdtemp(prefix=f"task_{task['id']:04d}_", dir=run/"tmp"))
        for name, frame in tables.items():
            add_meta(frame, task, plan).to_parquet(stage/f"{name}.parquet", index=False, compression="snappy")
        state = {"task_id": task["id"], "signature": plan["signature"],
                 "status": "PARCIAL" if failures else "COMPLETA", "n_falhas": len(failures),
                 "falhas": failures, "seconds": time.perf_counter()-start,
                 "finished_utc": datetime.now(timezone.utc).isoformat(),
                 "files": {p.name: digest(p) for p in sorted(stage.glob("*.parquet"))}}
        engine.pipeline.atomic_json(stage/"conclusao.json", state)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            # Marcador por ultimo: uma interrupcao nunca torna arquivos parciais validos.
            (destination/"conclusao.json").unlink(missing_ok=True)
            for path in stage.glob("*.parquet"):
                os.replace(path, destination/path.name)
            for path in destination.glob("*.parquet"):
                if path.name not in state["files"]:
                    path.unlink()
            os.replace(stage/"conclusao.json", destination/"conclusao.json")
        else:
            os.replace(stage, destination)
    except Exception as error:
        state = {"task_id": task["id"], "signature": plan["signature"], "status": "ERRO",
                 "erro": str(error), "seconds": time.perf_counter()-start}
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
    engine.pipeline.atomic_json(status_path, state)
    print(f"task={task['id']} {state['status']} treino={task['treino']} lookback={task['lookback']} "
          f"{state['seconds']:.2f}s {state.get('erro', '')}", flush=True)
    for failure in state.get("falhas", []):
        print(f"  k={failure['k']} estrategia={failure['estrategia']}: {failure['erro']}", flush=True)
    return state


def publish_summaries(run, plan, states, engine):
    stage = Path(tempfile.mkdtemp(prefix="resumos_", dir=run/"tmp"))
    try:
        for name in SUMMARY_TABLES:
            frames = []
            for task, state in zip(plan["tasks"], states):
                if state["status"] not in ("COMPLETA", "PARCIAL"):
                    continue
                path = run/"tasks"/task["relpath"]/f"{name}.parquet"
                if digest(path) != state["files"][path.name]:
                    raise ValueError(f"Resultado momentum alterado: {path}")
                frames.append(pq.ParquetFile(path).read(use_threads=False).to_pandas())
            # Filtrar vazios evita mudanca de dtype na concatenacao de falhas/metricas.
            populated = [f for f in frames if len(f)]
            table = (pd.concat(populated, ignore_index=True) if populated else frames[0].iloc[:0]
                     if frames else pd.DataFrame(columns=["task_id", "k", "estrategia", "erro"]))
            table.to_parquet(stage/f"{name}.parquet", index=False, compression="snappy")
        counts = {key: sum(s["status"] == key for s in states)
                  for key in ("COMPLETA", "PARCIAL", "ERRO", "PENDENTE")}
        state = {"signature": plan["signature"], "reference_signature": plan["reference_signature"],
                 "total": len(plan["tasks"]), **counts, "completa": counts["COMPLETA"] == len(plan["tasks"]),
                 "updated_utc": datetime.now(timezone.utc).isoformat()}
        pd.DataFrame([{k: s.get(k) for k in ("task_id", "status", "seconds", "erro")} for s in states]).to_csv(
            stage/"status_tarefas.csv", index=False)
        engine.pipeline.atomic_json(stage/"estado_execucao.json", state)
        target = run/"resumos"
        if target.exists():
            (target/"estado_execucao.json").unlink(missing_ok=True)
            for path in stage.iterdir():
                if path.name != "estado_execucao.json":
                    os.replace(path, target/path.name)
            os.replace(stage/"estado_execucao.json", target/"estado_execucao.json")
        else:
            os.replace(stage, target)
        print(f"Consolidacao momentum: {counts}", flush=True)
        return 0 if state["completa"] else 1
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def validate_existing_output(run_dir, plan):
    marker = run_dir/"plano.json"
    if marker.exists():
        if read_json(marker) != plan:
            raise ValueError("Execucao momentum distinta nesta pasta; use outro RUN_ID.")
    elif run_dir.exists():
        allowed = {"locks", "logs", "tmp", "codigo"}
        if any(p.name not in allowed and not re.fullmatch(r"worker_momentum_[A-Za-z0-9]{6}\.sh", p.name)
               for p in run_dir.iterdir()):
            raise ValueError("Pasta de saida ocupada sem plano momentum; use uma pasta separada.")
    driver = run_dir/"codigo"/"run_momentum_reference.py"
    if driver.exists() and digest(driver) != plan["driver_sha256"]:
        raise ValueError("Script congelado momentum diferente; use outro RUN_ID.")


def initialize_worker(args):
    global _WORKER_CONTEXT
    run_dir, _, plan, market, engine = load_reference(args)
    engine.pipeline.setup_runtime(run_dir)
    _WORKER_CONTEXT = (run_dir, plan, market, engine)


def execute_worker(task):
    run_dir, plan, market, engine = _WORKER_CONTEXT
    return execute_task(task, plan, market, engine, run_dir)


def run(args):
    run_dir, reference, plan, market, engine = load_reference(args)
    combinations = len(plan["tasks"])*len(plan["k"])
    print(f"Momentum: {len(plan['tasks'])} tarefas treino/lookback, {combinations} ICs acumulados, "
          f"{combinations*len(plan['strategies'])} carteiras; sem duplicacao por H.", flush=True)
    print(f"Sinal: {FORMULA}\nReferencia: {reference}\nSaida: {run_dir}", flush=True)
    print(f"K={plan['k']}; slippage={plan['slippage_bps']:g} bps; custos={plan['cost_bps']:g} bps por ponta.", flush=True)
    marker = run_dir/"plano.json"
    validate_existing_output(run_dir, plan)
    requested = getattr(args, "workers", 1)
    if type(requested) is not int or not 1 <= requested <= 16:
        raise ValueError("Workers deve ser um inteiro entre 1 e 16.")
    if args.dry_run:
        print("Plano conferido; nenhum arquivo criado e nenhum calculo executado.", flush=True)
        return 0
    allocation = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    if requested > allocation:
        raise ValueError("Numero de processos excede as CPUs disponiveis/alocadas.")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir/"locks").mkdir(exist_ok=True)
    with engine.pipeline.locked(run_dir/"locks"/"momentum.lock"):
        validate_existing_output(run_dir, plan)
        if not marker.exists():
            code = run_dir/"codigo"
            code.mkdir(exist_ok=True)
            driver = code/"run_momentum_reference.py"
            if not driver.exists():
                shutil.copyfile(__file__, driver)
            engine.pipeline.atomic_json(marker, plan)
        engine.pipeline.setup_runtime(run_dir)
        for name in ("tasks", "status", "logs"):
            (run_dir/name).mkdir(exist_ok=True)
        # Um resumo antigo nao pode continuar completo se uma retomada for interrompida.
        if (run_dir/"resumos"/"estado_execucao.json").exists():
            (run_dir/"resumos"/"estado_execucao.json").unlink()
        workers = min(requested, len(plan["tasks"]))
        print(f"Momentum: {workers} processos, 1 thread por processo.", flush=True)
        if workers == 1:
            states = [execute_task(task, plan, market, engine, run_dir) for task in plan["tasks"]]
        else:
            states = [None]*len(plan["tasks"])
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                     initializer=initialize_worker, initargs=(args,)) as pool:
                futures = [pool.submit(execute_worker, task) for task in plan["tasks"]]
                for future in as_completed(futures):
                    state = future.result()
                    states[state["task_id"]] = state
        return publish_summaries(run_dir, plan, states, engine)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-run", required=True, help="Execucao temporal/VAR completa, com plano e snapshot.")
    p.add_argument("--run-dir", required=True, help="Pasta momentum separada da referencia e dos insumos.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-local", action="store_true", help="Fixtures/testes fora do SNFS2.")
    p.add_argument("--workers", type=int, default=1, help="Processos CPU independentes; uma thread por processo.")
    return p


def main():
    try:
        status = run(parser().parse_args())
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"ERRO: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
    raise SystemExit(status)


if __name__ == "__main__":
    main()
