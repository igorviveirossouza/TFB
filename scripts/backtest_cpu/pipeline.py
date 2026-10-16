"""Preparacao auditavel, tarefas independentes, retomada atomica e consolidação."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import shutil
import tempfile
import time
import traceback
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy

from . import SCHEMA_VERSION
from .core import Strategy, backtest, ic_daily_frame, ic_frame, ic_summary
from .io import (digest, discover_tasks, json_digest, load_bundle, load_predictions,
                 load_strategies, read_json, read_table)

_MARKET = None
_PLAN = None


def storage_path(path, allow_local=False):
    p = Path(path).expanduser().resolve()
    if not allow_local and not p.is_relative_to(Path("/snfs2")):
        raise ValueError(f"Toda saida deve ficar no SNFS2: {p}")
    return p


@contextmanager
def locked(path):
    with Path(path).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def atomic_json(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=".json_", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def setup_runtime(run):
    # Executado tambem quando a CLI e chamada diretamente, fora do launcher SH.
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"]:
        os.environ[key] = "1"
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    for env, suffix in [("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "cache"),
                        ("MPLCONFIGDIR", "cache/matplotlib"), ("PYTHONPYCACHEPREFIX", "cache/pycache")]:
        p = run/suffix
        p.mkdir(parents=True, exist_ok=True)
        os.environ[env] = str(p)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    tempfile.tempdir = str(run/"tmp")


def code_files():
    return sorted(Path(__file__).parent.glob("*.py"))


def runtime_versions():
    return {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "pyarrow": pa.__version__}


def prepare(args):
    run = storage_path(args.run_dir, args.allow_local)
    predictions = Path(args.predictions).resolve()
    if run == predictions or run.is_relative_to(predictions) or predictions.is_relative_to(run):
        raise ValueError("Saida e previsoes devem estar em arvores separadas.")
    if (len(set(args.train)) != len(args.train) or any(not 0 < x < 100 for x in args.train)
            or not args.k or len(set(args.k)) != len(args.k) or min(args.k) < 1):
        raise ValueError("Cortes/K invalidos ou duplicados.")
    for value in (args.slippage_bps, args.cost_bps):
        if not np.isfinite(value) or not 0 <= value < 10000:
            raise ValueError("Slippage/custos devem estar em [0,10000) bps por ponta.")
    strategies = load_strategies(args.strategies)
    market, identity = load_bundle(args.market, args.returns_file)
    tasks, provenance = discover_tasks(predictions, sorted(args.train), market, identity["retornos_sha256"])
    if any(not any(k <= t["pred_len"] for k in args.k) for t in tasks):
        raise ValueError("Cada horizonte deve ter ao menos um K elegivel. Inclua K=1.")
    if max(args.k) > max(t["pred_len"] for t in tasks):
        raise ValueError("K solicitado sem horizonte de previsao correspondente.")
    source_files = code_files()
    plan = {"schema_version": SCHEMA_VERSION, "run_dir": str(run), "allow_local": args.allow_local,
            "inputs": identity, "prediction_manifests": provenance, "tasks": tasks,
            "strategies": strategies, "k": sorted(args.k), "slippage_bps": args.slippage_bps,
            "cost_bps": args.cost_bps, "code": {p.name: digest(p) for p in source_files},
            "runtime": runtime_versions(),
            "regime": "abertura_t+1_fechamento_t+k", "rf": identity["rf_fonte"], "rf_convencao": identity["rf_convencao"],
            "dividendos": "direito_na_data_ex_sem_reinvestir_no_bloco_disponivel_no_rebalanceamento",
            "distribuicao_ativos": "quantidades_recebidas_marcadas_diariamente_liquidadas_com_origem;sem_trajetoria_bloqueia",
            "short": "exposicao_bruta_ate_1x_sem_aluguel_margem_ou_remuneracao_de_garantias",
            "comparabilidade": "origens_comuns_a_todos_modelos_L_H_por_corte; fases_ancoradas_no_corte"}
    plan["signature"] = json_digest(plan)
    combinations = sum(sum(k <= t["pred_len"] for k in plan["k"]) for t in tasks)
    portfolios = sum(sum(k <= t["pred_len"] for k in plan["k"]) for t in tasks
                     if t["id"] == t["backtest_source_id"])*len(strategies)
    print(f"{len(tasks)} Parquets: temporal={sum(t['experimento']=='temporal' for t in tasks)}, "
          f"VAR={sum(t['experimento']=='var' for t in tasks)}; "
          f"{sum(t['pred_len'] for t in tasks)} series de IC diario por h; {combinations} ICs acumulados; "
          f"{portfolios} carteiras (VAR sem duplicacao por H).", flush=True)
    print(f"{market.n} pregoes, {len(market.tickers)} ativos. Slippage={args.slippage_bps:g} bps; "
          f"custos={args.cost_bps:g} bps por ponta. RF={plan['rf']}. Saida: {run}", flush=True)
    for (ex, _), receipts in sorted(market.received.items()):
        for receipt in receipts:
            first = market.calendar.date_pregao.iloc[ex].date()
            last = market.calendar.date_pregao.iloc[ex+len(receipt['close'])-1].date()
            print(f"Ativo recebido: {receipt['ticker']}, {first} a {last}, "
                  f"{len(receipt['close'])} pregoes; fora do ranking.", flush=True)
    if args.dry_run:
        print("Plano conferido; nenhum arquivo criado e nenhum job submetido.")
        return
    run.mkdir(parents=True, exist_ok=True)
    setup_runtime(run)
    for sub in ("tasks", "status", "locks", "logs"):
        (run/sub).mkdir(exist_ok=True)
    with locked(run/"locks/prepare.lock"):
        target = run/"plano.json"
        if target.exists():
            old = read_json(target)
            if old != plan:
                raise ValueError("Dados/codigo/estrategias/parametros diferentes nesta execucao. Use outro RUN_ID.")
            verify_snapshot(run, plan)
            print("Plano identico preservado; tarefas completas serao verificadas e reutilizadas.")
            return
        code = run/"codigo"
        code.mkdir(exist_ok=True)
        (code/"backtest_cpu").mkdir(exist_ok=True)
        for p in source_files:
            shutil.copyfile(p, code/"backtest_cpu"/p.name)
        # Entry point autocontido: independente do repositorio e do spool do Slurm.
        (code/"run_backtests.py").write_text(
            "import os, sys\nsys.dont_write_bytecode = True\n"
            "for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',"
            "'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):\n    os.environ[k] = '1'\n"
            "from backtest_cpu.cli import main\nif __name__ == '__main__':\n    main()\n", encoding="utf-8")
        atomic_json(run/"estrategias.json", strategies)
        atomic_json(target, plan)


def verify_snapshot(run, plan):
    unsigned = dict(plan)
    signature = unsigned.pop("signature")
    if json_digest(unsigned) != signature:
        raise ValueError("Assinatura do plano divergente.")
    for name, sha in plan["code"].items():
        if digest(run/"codigo/backtest_cpu"/name) != sha:
            raise ValueError(f"Snapshot de codigo alterado: {name}")


def initialize(plan):
    global _MARKET, _PLAN
    _PLAN = plan
    run = storage_path(plan["run_dir"], plan["allow_local"])
    setup_runtime(run)
    if runtime_versions() != plan["runtime"]:
        raise ValueError("Versoes Python/bibliotecas mudaram desde o planejamento. Use o mesmo ambiente ou novo RUN_ID.")
    inp = plan["inputs"]
    _MARKET, actual = load_bundle(inp["base"], inp["retornos"])
    if actual != inp:
        raise ValueError("Insumos de mercado/retornos/CDI mudaram apos preparar o plano.")


def valid_result(directory, signature, require_complete=True):
    try:
        state = read_json(directory/"conclusao.json")
        if state["signature"] != signature or (require_complete and state["status"] != "COMPLETA"):
            return False
        expected = {"ic.parquet", "ic_resumo.parquet", "ic_acumulado.parquet", "ic_acumulado_resumo.parquet", "falhas.parquet"}
        if state["status"] == "COMPLETA" and state["carteiras_proprias"]:
            expected.update({"carteira_diaria.parquet", "carteira_periodos.parquet", "posicoes.parquet", "metricas.parquet"})
        elif not state["carteiras_proprias"]:
            expected.add("referencia_carteiras.parquet")
        if not expected <= set(state["files"]):
            return False
        return all(Path(name).name == name and digest(directory/name) == sha
                   for name, sha in state["files"].items())
    except (OSError, ValueError, KeyError):
        return False


def with_meta(frame, task, plan):
    frame = frame.copy()
    for key in ("treino", "experimento", "modelo", "lookback", "pred_len", "loss", "lambda"):
        frame[key] = task[key]
    frame["task_id"] = task["id"]
    frame["slippage_bps"] = plan["slippage_bps"]
    frame["cost_bps"] = plan["cost_bps"]
    return frame


def execute_task(task):
    plan, market = _PLAN, _MARKET
    run = Path(plan["run_dir"])
    destination = run/"tasks"/task["relpath"]
    status_path = run/"status"/f"{task['id']:04d}.json"
    start = time.perf_counter()
    with locked(run/"locks"/f"task_{task['id']:04d}.lock"):
        try:
            # Mesmo a retomada verifica a origem; nunca reutiliza previsao reescrita.
            if digest(task["path"]) != task["sha256"]:
                raise ValueError(f"Previsao alterada: {task['path']}")
            if valid_result(destination, plan["signature"]):
                atomic_json(status_path, read_json(destination/"conclusao.json"))
                return {"task_id": task["id"], "status": "REUTILIZADA"}
            origins, predictions, cumulative = load_predictions(task, market, verify_hash=False)
            daily_ic = ic_daily_frame(predictions, origins, market, task["common_last_origin"])
            ics, outputs, failures = [], {}, []
            for k in plan["k"]:
                if k > task["pred_len"]:
                    continue
                scores = cumulative[:, k-1, :]
                ics.append(ic_frame(scores, origins, k, market, task["common_last_origin"]))
                if task["backtest_source_id"] != task["id"]:
                    continue
                selected_origins = np.arange(task["corte"], task["common_last_origin"]+1, k)
                selected_scores = scores[selected_origins-task["corte"]]
                for definition in plan["strategies"]:
                    strategy = Strategy(**definition)
                    try:
                        result = backtest(selected_scores, selected_origins, k, market, strategy,
                                          plan["slippage_bps"], plan["cost_bps"])
                        for name, frame in result.items():
                            outputs.setdefault(name, []).append(frame)
                    except ValueError as error:
                        failures.append({"k": k, "estrategia": strategy.id, "erro": str(error)})
            ic = pd.concat(ics, ignore_index=True)
            tables = {name: pd.concat(frames, ignore_index=True) for name, frames in outputs.items()}
            tables.update(ic=daily_ic, ic_resumo=ic_summary(daily_ic, "h"),
                          ic_acumulado=ic, ic_acumulado_resumo=ic_summary(ic),
                          falhas=pd.DataFrame(failures, columns=["k", "estrategia", "erro"]))
            own = task["backtest_source_id"] == task["id"]
            if not own:
                source = plan["tasks"][task["backtest_source_id"]]
                tables["referencia_carteiras"] = pd.DataFrame([{"task_id_fonte": source["id"],
                    "pred_len_fonte": source["pred_len"], "caminho_relativo": source["relpath"]}])
            stage = Path(tempfile.mkdtemp(prefix=f"task_{task['id']:04d}_", dir=run/"tmp"))
            try:
                for name, frame in tables.items():
                    with_meta(frame, task, plan).to_parquet(stage/f"{name}.parquet", index=False, compression="snappy")
                state = {"task_id": task["id"], "signature": plan["signature"],
                         "status": "PARCIAL" if failures else "COMPLETA", "n_falhas": len(failures),
                         "falhas": failures,
                         "carteiras_proprias": own,
                         "seconds": time.perf_counter()-start,
                         "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                         "pid": os.getpid(), "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                         "finished_utc": datetime.now(timezone.utc).isoformat(),
                         "files": {p.name: digest(p) for p in sorted(stage.glob("*.parquet"))}}
                atomic_json(stage/"conclusao.json", state)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    # Somente resultados desta assinatura: retomar nao apaga dados alheios.
                    try:
                        prior = read_json(destination/"conclusao.json")
                    except (OSError, ValueError):
                        # O plano da execucao foi conferido; marcador ausente/corrompido
                        # identifica uma publicacao incompleta desta tarefa.
                        prior = {}
                    if prior and prior["signature"] != plan["signature"]:
                        raise ValueError("Destino pertence a outro plano; escolha outro RUN_ID.")
                    # Publicacao por arquivo; conclusao e substituida por ultimo.
                    (destination/"conclusao.json").unlink(missing_ok=True)
                    for old in destination.glob("*.parquet"):
                        old.unlink()
                    for p in stage.glob("*.parquet"):
                        os.replace(p, destination/p.name)
                    os.replace(stage/"conclusao.json", destination/"conclusao.json")
                else:
                    os.replace(stage, destination)
                atomic_json(status_path, state)
                return state
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
        except Exception as error:
            state = {"task_id": task["id"], "signature": plan["signature"], "status": "ERRO",
                     "erro": str(error), "traceback": traceback.format_exc(),
                     "seconds": time.perf_counter()-start}
            atomic_json(status_path, state)
            return state


def consolidate(run, plan):
    """Reune resumos e IC diario ja salvo, sem recalcular IC ou carteiras.

    A serie temporal e copiada por tarefa para limitar memoria. O plano e os
    resultados individuais permanecem intactos, inclusive em execucoes antigas.
    """
    with locked(run/"locks/consolidar.lock"):
        tables = {"metricas": [], "ic_resumo": [], "ic_acumulado_resumo": [], "falhas": []}
        statuses = []
        daily_tmp = run/"tmp/ic_diario_consolidado.parquet"
        daily_rows = daily_tasks = 0
        daily_writer = None
        try:
            with ExitStack() as streams:
                for task in plan["tasks"]:
                    directory = run/"tasks"/task["relpath"]
                    # O lock cobre a verificacao e a leitura, evitando republicacoes parciais.
                    with locked(run/"locks"/f"task_{task['id']:04d}.lock"):
                        status = run/"status"/f"{task['id']:04d}.json"
                        last = read_json(status) if status.exists() else {}
                        if last.get("status") != "ERRO" and valid_result(directory, plan["signature"], require_complete=False):
                            state = read_json(directory/"conclusao.json")
                            for name in tables:
                                if f"{name}.parquet" in state["files"]:
                                    tables[name].append(read_table(directory/f"{name}.parquet"))
                            # ParquetFile nao infere particoes Hive a partir do caminho.
                            daily = pq.ParquetFile(directory/"ic.parquet").read(use_threads=False)
                            if daily_writer is None:
                                daily_writer = streams.enter_context(
                                    pq.ParquetWriter(daily_tmp, daily.schema, compression="snappy"))
                            daily_writer.write_table(daily)
                            daily_rows += daily.num_rows
                            daily_tasks += 1
                            statuses.append({"task_id": task["id"], "status": state["status"],
                                             "seconds": state["seconds"], "max_rss_mib": state["max_rss_mib"]})
                        else:
                            statuses.append({"task_id": task["id"], "status": "ERRO" if last.get("status") == "ERRO" else "PENDENTE",
                                             "erro": last.get("erro", "")})
        except BaseException:
            daily_tmp.unlink(missing_ok=True)
            raise
        summary = run/"resumos"
        summary.mkdir(exist_ok=True)
        if daily_tasks:
            os.replace(daily_tmp, summary/"ic_diario.parquet")
        else:
            daily_tmp.unlink(missing_ok=True)
            (summary/"ic_diario.parquet").unlink(missing_ok=True)
        # O consumidor R exige o estado da execucao; nunca mistura tabelas antigas.
        for name, frames in tables.items():
            if frames:
                tmp = run/"tmp"/f"{name}_consolidado.parquet"
                pd.concat(frames, ignore_index=True).to_parquet(tmp, index=False, compression="snappy")
                os.replace(tmp, summary/f"{name}.parquet")
            elif (summary/f"{name}.parquet").exists():
                (summary/f"{name}.parquet").unlink()
        pd.DataFrame(statuses).to_csv(run/"tmp/status_tarefas.csv", index=False)
        os.replace(run/"tmp/status_tarefas.csv", summary/"status_tarefas.csv")
        counts = {s: sum(row["status"] == s for row in statuses) for s in ("COMPLETA", "PARCIAL", "ERRO", "PENDENTE")}
        state = {"signature": plan["signature"], "total": len(statuses), **counts,
                 "completa": counts["COMPLETA"] == len(statuses),
                 "ic_diario": {"arquivo": "ic_diario.parquet" if daily_tasks else None,
                               "n_linhas": daily_rows, "n_tarefas": daily_tasks},
                 "updated_utc": datetime.now(timezone.utc).isoformat()}
        atomic_json(summary/"estado_execucao.json", state)
        print(f"Consolidacao: {counts}", flush=True)
        print(f"IC diario: {daily_rows} linhas, {daily_tasks} tarefas; "
              "valores existentes preservados, sem recalculo.", flush=True)
        return state


def run_tasks(args):
    run = Path(args.run_dir).resolve()
    plan = read_json(run/"plano.json")
    storage_path(run, plan["allow_local"])
    if str(run) != plan["run_dir"]:
        raise ValueError("Execucao movida. Copias locais servem a consulta R; retome no caminho original.")
    setup_runtime(run)
    verify_snapshot(run, plan)
    if args.command == "consolidate":
        state = consolidate(run, plan)
        return 0 if state["completa"] else 1
    if args.workers < 1 or args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Workers/lotes invalidos.")
    allocation = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    if args.workers > allocation:
        raise ValueError(f"workers={args.workers} excede CPUs disponiveis/alocadas={allocation}.")
    tasks = plan["tasks"][args.shard::args.shards]
    if args.pilot:
        chosen = set()
        for mode in ("temporal", "var"):
            candidates = [t for t in plan["tasks"] if t["experimento"] == mode]
            for select in (min, max):
                chosen.add(select(candidates, key=lambda t: t["pred_len"])["id"])
        tasks = [t for t in tasks if t["id"] in chosen]
    if not tasks:
        raise ValueError("Lote sem tarefas.")
    # Validacao uma vez no coordenador garante erro legivel antes do pool.
    initialize(plan)
    print(f"Lote {args.shard+1}/{args.shards}: {len(tasks)} tarefas, "
          f"{min(args.workers, len(tasks))} processos, 1 thread por processo.", flush=True)
    failures = 0
    def report(result):
        nonlocal failures
        failures += result["status"] in {"ERRO", "PARCIAL"}
        print(f"task={result['task_id']} {result['status']} "
              f"{result.get('seconds', 0):.2f}s {result.get('erro', '')}", flush=True)
        for failure in result.get("falhas", []):
            print(f"  k={failure['k']} estrategia={failure['estrategia']}: {failure['erro']}", flush=True)
    if args.workers == 1:
        for task in tasks:
            report(execute_task(task))
    else:
        # Spawn evita herdar pools internos Arrow/BLAS ou estados de bibliotecas.
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks)), mp_context=mp.get_context("spawn"),
                                 initializer=initialize, initargs=(plan,)) as pool:
            pending = {pool.submit(execute_task, task): task for task in tasks}
            for future in as_completed(pending):
                report(future.result())
    consolidate(run, plan)
    return 1 if failures else 0
