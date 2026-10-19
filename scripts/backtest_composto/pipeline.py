"""Backtest composto incremental: contrato fixo, catalogo extensivel, tarefas imutaveis."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
import multiprocessing as mp
import os
from pathlib import Path
import resource
import shutil
import tempfile
import time
import traceback

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from backtest_cpu.core import Strategy, backtest, ic_daily_frame, ic_frame, ic_summary
from backtest_cpu.io import digest, json_digest, load_bundle, load_strategies, read_json, read_table
from backtest_cpu.pipeline import atomic_json, locked, runtime_versions, setup_runtime, storage_path, valid_result
from .io import discover_composite_tasks, load_composite_predictions

VERSION = "2026-10-19-composto-proximity-v2"
_PLAN = None
_MARKET = None
SUMMARY_NAMES = ("metricas", "ic_resumo", "ic_acumulado_resumo", "falhas")
SERIES = {"ic_diario": "ic", "ic_acumulado": "ic_acumulado"}


def verify_signature(value, label):
    unsigned = dict(value)
    signature = unsigned.pop("signature", None)
    if signature != json_digest(unsigned):
        raise ValueError(f"Assinatura divergente: {label}.")


def source_files():
    scripts = Path(__file__).resolve().parent.parent
    files = [scripts/"run_backtests_composto.py"]
    for package in ("backtest_cpu", "backtest_composto"):
        files.extend(sorted((scripts/package).glob("*.py")))
    return scripts, files


def verify_snapshot(run, contract):
    verify_signature(contract, "contrato")
    for name, sha in contract["code"].items():
        p = Path(name)
        if p.is_absolute() or ".." in p.parts or digest(run/"codigo"/p) != sha:
            raise ValueError(f"Codigo congelado alterado: {name}")


def validate_locations(run, predictions, market, returns):
    for path in (predictions, market, returns):
        if run == path or run.is_relative_to(path) or path.is_relative_to(run):
            raise ValueError("Resultados devem ficar separados das previsoes e dos insumos.")


def task_signature(task, contract_signature):
    clean = {k: v for k, v in task.items() if k != "task_signature"}
    return json_digest({"contract_signature": contract_signature, "task": clean})


def validate_plan(plan, contract):
    verify_signature(plan, "catalogo")
    if plan["contract_signature"] != contract["signature"]:
        raise ValueError("Catalogo pertence a outro contrato.")
    if [t["id"] for t in plan["tasks"]] != list(range(len(plan["tasks"]))):
        raise ValueError("IDs de tarefas invalidos.")
    if len({t["relpath"] for t in plan["tasks"]}) != len(plan["tasks"]):
        raise ValueError("Tarefas duplicadas no catalogo.")
    for t in plan["tasks"]:
        if t["task_signature"] != task_signature(t, contract["signature"]):
            raise ValueError("Assinatura individual divergente.")


def build_plan(contract, old, candidates, discovery):
    if old:
        validate_plan(old, contract)
        # A amostra comum/desenho de um treino registrado nao pode mudar depois.
        old_groups = {g["treino"]: g for g in old["discovery"]["groups"] if g.get("design_sha256")}
        for g in discovery["groups"]:
            previous = old_groups.get(g["treino"])
            if previous and any(g.get(k) != previous.get(k) for k in
                                ("design_sha256", "hmax", "common_last_origin")):
                raise ValueError("Desenho/amostra comum de treino existente mudou. Use outro RUN_ID.")
    tasks = list(old["tasks"]) if old else []
    found = {t["relpath"]: t for t in candidates}
    if len(found) != len(candidates):
        raise ValueError("Previsoes compostas duplicadas.")
    for prior in tasks:
        candidate = found.pop(prior["relpath"], None)
        if candidate is None:
            raise ValueError(f"Previsao anteriormente registrada desapareceu: {prior['relpath']}")
        original = {k: v for k, v in prior.items() if k not in {"id", "backtest_source_id", "task_signature"}}
        if candidate != original:
            raise ValueError(f"Previsao/configuracao anteriormente registrada mudou: {prior['relpath']}. Use outro RUN_ID.")
    for candidate in sorted(found.values(), key=lambda t: t["relpath"]):
        task = dict(candidate, id=len(tasks), backtest_source_id=len(tasks))
        if not any(k <= task["pred_len"] for k in contract["k"]):
            raise ValueError("Previsao sem K de avaliacao elegivel; inclua K=1.")
        task["task_signature"] = task_signature(task, contract["signature"])
        tasks.append(task)
    for group in discovery["groups"]:
        if group.get("hmax") is not None and max(contract["k"]) > group["hmax"]:
            raise ValueError("K de avaliacao excede os horizontes declarados de um treino.")
    plan = {"schema_version": VERSION, "contract_signature": contract["signature"],
            "run_dir": contract["run_dir"], "allow_local": contract["allow_local"],
            "inputs": contract["inputs"], "k": contract["k"], "strategies": contract["strategies"],
            "slippage_bps": contract["slippage_bps"], "cost_bps": contract["cost_bps"],
            "runtime": contract["runtime"], "tasks": tasks, "discovery": discovery}
    plan["signature"] = json_digest(plan)
    return plan


def prepare(args):
    run = storage_path(args.run_dir, args.allow_local)
    predictions, market_path, returns = (Path(p).expanduser().resolve() for p in
                                        (args.predictions, args.market, args.returns_file))
    validate_locations(run, predictions, market_path, returns)
    if (not args.train or len(set(args.train)) != len(args.train) or any(not 0 < t < 100 for t in args.train)
            or not args.k or min(args.k) < 1 or len(set(args.k)) != len(args.k)):
        raise ValueError("Cortes/K invalidos ou duplicados.")
    for value in (args.slippage_bps, args.cost_bps):
        if not np.isfinite(value) or not 0 <= value < 10000:
            raise ValueError("Custos/slippage devem estar em [0,10000) bps por ponta.")
    market, identity = load_bundle(market_path, returns)
    scripts, files = source_files()
    contract = {"schema_version": VERSION, "run_dir": str(run), "allow_local": args.allow_local,
                "predictions_root": str(predictions), "inputs": identity, "k": sorted(args.k),
                "strategies": load_strategies(args.strategies), "slippage_bps": args.slippage_bps,
                "cost_bps": args.cost_bps, "runtime": runtime_versions(),
                "code": {str(p.relative_to(scripts)): digest(p) for p in files},
                "regime": "abertura_t+1_fechamento_t+k",
                "amostra_comum": "N-max_horizonte_declarado_por_treino;origens_ancoradas_no_corte",
                "k_dir": "janela_da_loss;independente_do_k_de_avaliacao",
                "dividendos": "direito_na_data_ex_sem_reinvestir_no_bloco_disponivel_no_rebalanceamento",
                "distribuicao_ativos": "quantidades_recebidas_marcadas_diariamente_liquidadas_com_origem;sem_trajetoria_bloqueia",
                "short": "exposicao_bruta_ate_1x_sem_aluguel_margem_ou_remuneracao_de_garantias"}
    contract["signature"] = json_digest(contract)

    def discover(old):
        trains = sorted(set(args.train) | {g["treino"] for g in old["discovery"]["groups"]}) if old else sorted(args.train)
        candidates, discovery = discover_composite_tasks(predictions, trains, market, identity["retornos_sha256"])
        return build_plan(contract, old, candidates, discovery)

    def check_contract():
        path = run/"contrato.json"
        if path.exists():
            old_contract = read_json(path)
            if old_contract != contract:
                raise ValueError("Insumos/codigo/parametros mudaram. Use outro RUN_ID composto.")
            verify_snapshot(run, old_contract)
        elif (run/"plano.json").exists():
            raise ValueError("Pasta pertence a outra execucao sem contrato composto.")

    check_contract()
    old = read_json(run/"plano.json") if (run/"plano.json").exists() else None
    plan = discover(old)
    new = len(plan["tasks"]) - (len(old["tasks"]) if old else 0)
    print(f"Composto: {len(plan['tasks'])} previsoes disponiveis registradas; {new} novas; "
          f"{plan['discovery']['awaiting']} estimativas ainda aguardadas na grade conhecida.", flush=True)
    print(f"Treinos={plan['discovery']['requested_train']}; K={plan['k']}; "
          f"{market.n} pregoes, {len(market.tickers)} ativos. Saida: {run}", flush=True)
    if plan["discovery"]["missing_groups"]:
        print(f"Treinos ainda sem cadastro: {plan['discovery']['missing_groups']}", flush=True)
    if args.dry_run:
        print("Plano conferido; nenhum arquivo criado e nenhum job submetido.", flush=True)
        return 0
    if run.exists() and not (run/"contrato.json").exists() and any(run.iterdir()):
        # A pasta de uma preparacao interrompida pode conter somente o snapshot/locks.
        if any(p.name not in {"codigo", "locks", "planos"} for p in run.iterdir()):
            raise ValueError("Pasta ocupada sem contrato composto; escolha outro RUN_ID.")
    run.mkdir(parents=True, exist_ok=True)
    (run/"locks").mkdir(exist_ok=True)
    with locked(run/"locks/catalogo.lock"):
        check_contract()
        latest = read_json(run/"plano.json") if (run/"plano.json").exists() else None
        if latest != old:
            plan = discover(latest)
        if not (run/"contrato.json").exists():
            for source in files:
                target = run/"codigo"/source.relative_to(scripts)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            atomic_json(run/"contrato.json", contract)
        (run/"planos").mkdir(exist_ok=True)
        revision = run/"planos"/f"{plan['signature']}.json"
        if revision.exists() and read_json(revision) != plan:
            raise ValueError("Revisao de catalogo alterada.")
        if not revision.exists():
            atomic_json(revision, plan)
        atomic_json(run/"plano.json", plan)
    setup_runtime(run)
    for name in ("tasks", "status", "logs"):
        (run/name).mkdir(exist_ok=True)
    print(f"Catalogo congelado: {revision}", flush=True)
    return 0


def initialize(plan):
    global _PLAN, _MARKET
    _PLAN = plan
    if runtime_versions() != plan["runtime"]:
        raise ValueError("Ambiente Python mudou desde a preparacao.")
    setup_runtime(Path(plan["run_dir"]))
    inp = plan["inputs"]
    _MARKET, actual = load_bundle(inp["base"], inp["retornos"])
    if actual != inp:
        raise ValueError("Insumos de mercado/retornos/CDI mudaram desde a preparacao.")


def with_meta(frame, task, plan):
    table = pa.Table.from_pandas(frame, preserve_index=False)
    values = {k: (task[k], pa.int64()) for k in ("treino", "lookback", "pred_len", "k_dir")}
    values.update({k: (str(task[k]), pa.string()) for k in ("experimento", "modelo", "loss", "lambda")})
    values.update({"delta": (float(task["delta"]), pa.float64()),
                   "cross_lambda": (task["cross_lambda"], pa.float64()),
                   "cross_delta": (task["cross_delta"], pa.float64()),
                   "delta_effective": (task["delta_effective"], pa.bool_()),
                   "mask_type": (task.get("mask_type"), pa.string()),
                   "mask_a": (task.get("mask_a"), pa.float64()),
                   "mask_effective": ("mask_type" in task, pa.bool_()),
                   "k_loss": (task["k_dir"], pa.int64()), "task_id": (task["id"], pa.int64()),
                   "seed": (task.get("seed"), pa.int64()),
                   "training_array_index": (task["training_array_index"], pa.int64()),
                   "temporal_loss": (task["temporal_loss"], pa.string()),
                   "dataset": ("log_retornos", pa.string()),
                   "slippage_bps": (plan["slippage_bps"], pa.float64()),
                   "cost_bps": (plan["cost_bps"], pa.float64()),
                   "task_signature": (task["task_signature"], pa.string())})
    for key, (value, dtype) in values.items():
        array = pa.array([value]*len(frame), type=dtype)
        if key in table.column_names:
            table = table.set_column(table.column_names.index(key), key, array)
        else:
            table = table.append_column(key, array)
    return table


def execute_task(task):
    plan, market = _PLAN, _MARKET
    run = Path(plan["run_dir"])
    directory = run/"tasks"/task["relpath"]
    status_path = run/"status"/f"{task['id']:06d}.json"
    start = time.perf_counter()
    stage = None
    with locked(run/"locks"/f"task_{task['id']:06d}.lock"):
        try:
            if digest(task["path"]) != task["sha256"]:
                raise ValueError(f"Previsao alterada: {task['path']}")
            if valid_result(directory, task["task_signature"]):
                state = read_json(directory/"conclusao.json")
                atomic_json(status_path, state)
                return dict(state, status="REUTILIZADA")
            origins, predictions, cumulative = load_composite_predictions(task, market, verify_hash=False)
            daily = ic_daily_frame(predictions, origins, market, task["common_last_origin"])
            ics, outputs, failures = [], {}, []
            for k in plan["k"]:
                if k > task["pred_len"]:
                    continue
                scores = cumulative[:, k-1, :]
                ics.append(ic_frame(scores, origins, k, market, task["common_last_origin"]))
                chosen = np.arange(task["corte"], task["common_last_origin"]+1, k)
                signal = scores[chosen-task["corte"]]
                for definition in plan["strategies"]:
                    try:
                        result = backtest(signal, chosen, k, market, Strategy(**definition),
                                          plan["slippage_bps"], plan["cost_bps"])
                        for name, frame in result.items():
                            outputs.setdefault(name, []).append(frame)
                    except ValueError as error:
                        failures.append({"k": k, "estrategia": definition["id"], "erro": str(error)})
            accumulated = pd.concat(ics, ignore_index=True)
            tables = {name: pd.concat(frames, ignore_index=True) for name, frames in outputs.items()}
            tables.update(ic=daily, ic_resumo=ic_summary(daily, "h"),
                          ic_acumulado=accumulated, ic_acumulado_resumo=ic_summary(accumulated),
                          falhas=pd.DataFrame(failures, columns=["k", "estrategia", "erro"]))
            stage = Path(tempfile.mkdtemp(prefix=f"task_{task['id']:06d}_", dir=run/"tmp"))
            for name, frame in tables.items():
                table = with_meta(frame, task, plan)
                if name == "falhas":
                    for col, dtype in (("k", pa.int64()), ("estrategia", pa.string()), ("erro", pa.string())):
                        table = table.set_column(table.column_names.index(col), col, table[col].cast(dtype))
                pq.write_table(table, stage/f"{name}.parquet", compression="snappy")
            state = {"task_id": task["id"], "signature": task["task_signature"],
                     "task_signature": task["task_signature"], "contract_signature": plan["contract_signature"],
                     "status": "PARCIAL" if failures else "COMPLETA", "n_falhas": len(failures),
                     "falhas": failures, "carteiras_proprias": True,
                     "seconds": time.perf_counter()-start,
                     "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                     "finished_utc": datetime.now(timezone.utc).isoformat(),
                     "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                     "files": {p.name: digest(p) for p in stage.glob("*.parquet")}}
            atomic_json(stage/"conclusao.json", state)
            directory.parent.mkdir(parents=True, exist_ok=True)
            if directory.exists():
                try:
                    previous = read_json(directory/"conclusao.json")
                except (OSError, ValueError):
                    previous = {}
                if previous and previous.get("signature") != task["task_signature"]:
                    raise ValueError("Destino pertence a outra tarefa/assinatura.")
                (directory/"conclusao.json").unlink(missing_ok=True)
                for path in stage.glob("*.parquet"):
                    os.replace(path, directory/path.name)
                for path in directory.glob("*.parquet"):
                    if path.name not in state["files"]:
                        path.unlink()
                os.replace(stage/"conclusao.json", directory/"conclusao.json")
            else:
                os.replace(stage, directory)
            atomic_json(status_path, state)
            return state
        except Exception as error:
            state = {"task_id": task["id"], "signature": task["task_signature"], "status": "ERRO",
                     "erro": str(error), "traceback": traceback.format_exc(),
                     "seconds": time.perf_counter()-start}
            atomic_json(status_path, state)
            return state
        finally:
            if stage is not None and stage.exists():
                shutil.rmtree(stage)


def consolidate(run):
    # Congela o catalogo atual durante a publicacao; um job antigo nunca publica
    # um resumo rotulado com uma revisao antiga por cima da revisao mais recente.
    with locked(run/"locks/consolidar.lock"), locked(run/"locks/catalogo.lock"):
        plan, contract = read_json(run/"plano.json"), read_json(run/"contrato.json")
        validate_plan(plan, contract)
        verify_snapshot(run, contract)
        stage = Path(tempfile.mkdtemp(prefix="resumos_", dir=run/"tmp"))
        statuses, frames = [], {name: [] for name in SUMMARY_NAMES}
        writers, rows = {}, {name: 0 for name in SERIES}
        try:
            with ExitStack() as streams:
                for task in plan["tasks"]:
                    directory = run/"tasks"/task["relpath"]
                    with locked(run/"locks"/f"task_{task['id']:06d}.lock"):
                        status_path = run/"status"/f"{task['id']:06d}.json"
                        try:
                            last = read_json(status_path)
                        except (OSError, ValueError):
                            last = {}
                        valid = last.get("status") != "ERRO" and valid_result(
                            directory, task["task_signature"], require_complete=False)
                        if valid:
                            result = read_json(directory/"conclusao.json")
                            for name in SUMMARY_NAMES:
                                if f"{name}.parquet" in result["files"]:
                                    frames[name].append(read_table(directory/f"{name}.parquet"))
                            for name, source in SERIES.items():
                                parquet = pq.ParquetFile(directory/f"{source}.parquet")
                                for batch in parquet.iter_batches(batch_size=65536, use_threads=False):
                                    table = pa.Table.from_batches([batch])
                                    if name not in writers:
                                        writers[name] = streams.enter_context(pq.ParquetWriter(
                                            stage/f"{name}.parquet", table.schema, compression="snappy"))
                                    writers[name].write_table(table)
                                    rows[name] += len(table)
                            label = result["status"]
                        else:
                            result = last
                            label = "ERRO" if last.get("status") == "ERRO" else "PENDENTE"
                        statuses.append({"task_id": task["id"], "status": label,
                                         **{k: task[k] for k in ("treino", "modelo", "lookback", "pred_len", "loss", "lambda", "delta", "k_dir")},
                                         "mask_type": task.get("mask_type"), "mask_a": task.get("mask_a"),
                                         "mask_effective": "mask_type" in task,
                                         "seconds": result.get("seconds"), "erro": result.get("erro", "")})
            for name, parts in frames.items():
                populated = [p for p in parts if len(p)]
                if parts:
                    table = pd.concat(populated, ignore_index=True) if populated else parts[0].iloc[:0]
                    table.to_parquet(stage/f"{name}.parquet", index=False, compression="snappy")
            pd.DataFrame(statuses, columns=["task_id", "status", "treino", "modelo", "lookback", "pred_len", "loss", "lambda", "delta", "k_dir", "mask_type", "mask_a", "mask_effective", "seconds", "erro"]).to_csv(stage/"status_tarefas.csv", index=False)
            counts = {s: sum(x["status"] == s for x in statuses) for s in ("COMPLETA", "PARCIAL", "ERRO", "PENDENTE")}
            state = {"signature": plan["signature"], "contract_signature": plan["contract_signature"],
                     "total": len(statuses), **counts,
                     "completa": counts["COMPLETA"] == len(statuses) and bool(statuses),
                     "completa_grade": counts["COMPLETA"] == len(statuses) and bool(statuses) and plan["discovery"]["expected_complete"],
                     "discovery": plan["discovery"],
                     "series": {name: {"n_linhas": count, "arquivo": f"{name}.parquet" if count else None} for name, count in rows.items()},
                     "updated_utc": datetime.now(timezone.utc).isoformat()}
            atomic_json(stage/"descoberta_previsoes.json", plan["discovery"])
            atomic_json(stage/"estado_execucao.json", state)
            target = run/"resumos"
            target.mkdir(exist_ok=True)
            (target/"estado_execucao.json").unlink(missing_ok=True)
            for path in stage.iterdir():
                if path.name != "estado_execucao.json":
                    os.replace(path, target/path.name)
            for name in (*SUMMARY_NAMES, *SERIES):
                # Evita reter tabela antiga quando nenhuma tarefa correspondente e valida.
                present = bool(frames[name]) if name in frames else rows[name] > 0
                if not present:
                    (target/f"{name}.parquet").unlink(missing_ok=True)
            os.replace(stage/"estado_execucao.json", target/"estado_execucao.json")
            print(f"Consolidacao composta: {counts}; estimativas aguardadas: {plan['discovery']['awaiting']}.", flush=True)
            return state
        finally:
            shutil.rmtree(stage, ignore_errors=True)


def run_tasks(args):
    run = Path(args.run_dir).expanduser().resolve()
    contract = read_json(run/"contrato.json")
    storage_path(run, contract["allow_local"])
    if str(run) != contract["run_dir"]:
        raise ValueError("Retome a execucao no caminho original.")
    verify_snapshot(run, contract)
    setup_runtime(run)
    if args.command == "consolidate":
        state = consolidate(run)
        return int(any(state[x] for x in ("PARCIAL", "ERRO", "PENDENTE")))
    scripts, files = source_files()
    if {str(p.relative_to(scripts)): digest(p) for p in files} != contract["code"]:
        raise ValueError("Codigo em uso difere do contrato; execute codigo/run_backtests_composto.py congelado.")
    path = Path(args.plan_file).resolve() if args.plan_file else run/"plano.json"
    if args.plan_file and (path.parent != run/"planos" or path.suffix != ".json"):
        raise ValueError("Use uma revisao congelada em run-dir/planos.")
    plan = read_json(path)
    validate_plan(plan, contract)
    if args.plan_file and path.stem != plan["signature"]:
        raise ValueError("Nome da revisao nao corresponde a assinatura.")
    latest = read_json(run/"plano.json")
    validate_plan(latest, contract)
    for task in plan["tasks"]:
        if task["id"] >= len(latest["tasks"]) or task != latest["tasks"][task["id"]]:
            raise ValueError("Revisao nao e um prefixo valido do catalogo atual.")
    if args.workers < 1 or args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Workers/lotes invalidos.")
    allocation = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    if args.workers > allocation:
        raise ValueError("Workers excedem as CPUs alocadas.")
    tasks = plan["tasks"][args.shard::args.shards]
    if args.pilot:
        selected = set()
        for loss in sorted({t["loss"] for t in tasks}):
            group = [t for t in tasks if t["loss"] == loss]
            for fn in (min, max):
                selected.add(fn(group, key=lambda t: (t["pred_len"], t["k_dir"], t["id"]))["id"])
        tasks = [t for t in tasks if t["id"] in selected]
    initialize(plan)
    print(f"Lote {args.shard+1}/{args.shards}: {len(tasks)} tarefas; "
          f"{min(args.workers, len(tasks))} processos. Revisao {plan['signature'][:12]}.", flush=True)
    failures = 0
    def report(result):
        nonlocal failures
        failures += result["status"] in {"ERRO", "PARCIAL"}
        print(f"task={result['task_id']} {result['status']} {result.get('seconds', 0):.2f}s {result.get('erro', '')}", flush=True)
        for failure in result.get("falhas", []):
            print(f"  k={failure['k']} estrategia={failure['estrategia']}: {failure['erro']}", flush=True)
    if tasks and args.workers == 1:
        for task in tasks:
            report(execute_task(task))
    elif tasks:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks)), mp_context=mp.get_context("spawn"),
                                 initializer=initialize, initargs=(plan,)) as pool:
            for future in as_completed([pool.submit(execute_task, task) for task in tasks]):
                report(future.result())
    consolidate(run)
    return int(bool(failures))
