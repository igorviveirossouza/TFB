"""Contratos incrementais: novos Parquets nunca invalidam resultados existentes.

Fixtures pequenas reproduzem os manifests/publicacao de b3_experiment. Os
oraculos economicos usam o motor CPU existente; nenhum dado real e acessado.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pyarrow.parquet as pq
import pytest

from test_backtest_composto_io import make_composite_fixture, publish_composite_task

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from backtest_cpu.core import ic_daily_frame, ic_frame, ic_summary
from backtest_cpu.io import digest, load_predictions, read_json


def cli(*args, check=True, entrypoint=None, cwd=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [sys.executable, "-B", str(entrypoint or ROOT / "scripts/run_backtests_composto.py"),
               *map(str, args)]
    result = subprocess.run(command, text=True, capture_output=True, env=env,
                            cwd=cwd or ROOT, timeout=90)
    if check and result.returncode:
        pytest.fail(result.stdout + result.stderr)
    return result


def prepare_args(fixture, run, percentages=(50,)):
    return ["prepare", "--market", fixture.base, "--returns-file", fixture.returns,
            "--predictions", fixture.root, "--strategies", ROOT / "config/estrategias_backtest.json",
            "--run-dir", run, "--train", *map(str, percentages), "--k", "1", "3", "--allow-local"]


def read_parquet(path):
    # Nao inferir particoes Hive externas ao arquivo.
    return pq.ParquetFile(path).read(use_threads=False).to_pandas()


def choose(fixture, train=50, horizon=3, k_loss=1):
    return next(i for i, task in enumerate(fixture.producer_tasks)
                if task["treino"] == train and task["pred_len"] == horizon and task["k"] == k_loss)


def fingerprint(run, tasks):
    return {str(path.relative_to(run)): (digest(path), path.stat().st_mtime_ns)
            for task in tasks
            for path in (run / "tasks" / task["relpath"]).iterdir() if path.is_file()}


def assert_previous_tasks_preserved(previous, current):
    current_by_id = {task["id"]: task for task in current["tasks"]}
    for task in previous["tasks"]:
        assert current_by_id[task["id"]]["task_signature"] == task["task_signature"]
        assert current_by_id[task["id"]]["relpath"] == task["relpath"]


def assert_consolidated_matches_tasks(run, plan):
    for source, target in (("ic", "ic_diario"), ("ic_acumulado", "ic_acumulado"),
                           ("ic_resumo", "ic_resumo"),
                           ("ic_acumulado_resumo", "ic_acumulado_resumo"),
                           ("metricas", "metricas")):
        expected = pd.concat([read_parquet(run / "tasks" / t["relpath"] / f"{source}.parquet")
                              for t in plan["tasks"]], ignore_index=True)
        actual = read_parquet(run / "resumos" / f"{target}.parquet")
        pd.testing.assert_frame_equal(actual, expected)
    daily = read_parquet(run / "resumos/ic_diario.parquet")
    cumulative = read_parquet(run / "resumos/ic_acumulado.parquet")
    assert not daily.duplicated(["task_id", "origin_step", "h"]).any()
    assert not cumulative.duplicated(["task_id", "origin_step", "k"]).any()
    assert (daily.target_step == daily.origin_step + daily.h).all()
    assert (cumulative.target_step == cumulative.origin_step + cumulative.k).all()


def test_incremental_append_train_shards_and_old_submission_preserve_tasks(tmp_path):
    fixture = make_composite_fixture(tmp_path, percentages=(50, 70), horizons=(1, 3),
                                     deltas=(0,), losses=("ranknet",), available_indices=())
    # Chega primeiro H=3/K_loss=1, antes de H=1 na ordenacao do desenho.
    publish_composite_task(fixture, choose(fixture))
    run = tmp_path / "run"
    args = prepare_args(fixture, run)
    cli(*args, "--dry-run")
    assert not run.exists()
    cli(*args)
    first = read_json(run / "plano.json")
    assert len(first["tasks"]) == 1
    old_submission = run / "planos" / f"{first['signature']}.json"
    assert read_json(old_submission) == first
    frozen = run / "codigo/run_backtests_composto.py"
    cli("run", "--run-dir", run, "--plan-file", old_submission, "--workers", "2",
        entrypoint=frozen, cwd=tmp_path)
    before = fingerprint(run, first["tasks"])
    task = first["tasks"][0]
    assert task["common_last_origin"] == fixture.market.n - 3
    assert task["backtest_source_id"] == task["id"]

    # Os ICs tem os mesmos valores/datas do motor congelado, inclusive por h.
    origins, predicted, accumulated = load_predictions(task, fixture.market)
    expected_daily = ic_daily_frame(predicted, origins, fixture.market, task["common_last_origin"])
    got_daily = read_parquet(run / "tasks" / task["relpath"] / "ic.parquet")
    pd.testing.assert_frame_equal(got_daily[expected_daily.columns], expected_daily)
    expected_cumulative = pd.concat([
        ic_frame(accumulated[:, k - 1], origins, k, fixture.market, task["common_last_origin"])
        for k in (1, 3)], ignore_index=True)
    got_cumulative = read_parquet(run / "tasks" / task["relpath"] / "ic_acumulado.parquet")
    pd.testing.assert_frame_equal(got_cumulative[expected_cumulative.columns], expected_cumulative)
    expected_summary = ic_summary(expected_cumulative)
    got_summary = read_parquet(run / "tasks" / task["relpath"] / "ic_acumulado_resumo.parquet")
    pd.testing.assert_frame_equal(got_summary[expected_summary.columns], expected_summary)

    publish_composite_task(fixture, choose(fixture, horizon=1))
    publish_composite_task(fixture, choose(fixture, k_loss=3))
    cli(*args)
    second = read_json(run / "plano.json")
    assert len(second["tasks"]) == 3
    assert second["signature"] != first["signature"]
    assert second["contract_signature"] == first["contract_signature"]
    assert_previous_tasks_preserved(first, second)
    assert fingerprint(run, first["tasks"]) == before
    # Um job submetido antes do append so trabalha sua lista congelada, mas
    # deve consolidar o catalogo novo: nao pode esconder as novas pendencias.
    cli("run", "--run-dir", run, "--plan-file", old_submission, "--workers", "1",
        entrypoint=frozen, cwd=tmp_path)
    state = read_json(run / "resumos/estado_execucao.json")
    assert state["signature"] == second["signature"]
    assert state["total"] == 3 and state["COMPLETA"] == 1 and state["PENDENTE"] == 2
    assert fingerprint(run, first["tasks"]) == before

    current_submission = run / "planos" / f"{second['signature']}.json"
    # Dois coordenadores em paralelo, cada um com seu shard e pool de workers.
    def shard(index):
        return cli("run", "--run-dir", run, "--plan-file", current_submission,
                   "--workers", "2", "--shard", index, "--shards", "2",
                   entrypoint=frozen, cwd=tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(shard, (0, 1)))
    assert fingerprint(run, first["tasks"]) == before
    assert_consolidated_matches_tasks(run, second)
    metadata = read_parquet(run / "resumos/ic_acumulado_resumo.parquet")
    same_h = metadata[metadata.pred_len == 3]
    assert set(same_h.k_loss) == {1, 3}
    assert (metadata.k_loss == metadata.k_dir).all()
    assert set(same_h.k) == {1, 3}
    assert same_h.groupby("k_loss").task_id.nunique().eq(1).all()
    second_before = fingerprint(run, second["tasks"])

    # Outro corte de treino e registrado depois, sem alterar tarefas existentes.
    publish_composite_task(fixture, choose(fixture, train=70))
    cli(*prepare_args(fixture, run, (50, 70)))
    third = read_json(run / "plano.json")
    assert len(third["tasks"]) == 4
    assert_previous_tasks_preserved(second, third)
    assert third["contract_signature"] == first["contract_signature"]
    cli("run", "--run-dir", run, "--workers", "2", entrypoint=frozen)
    assert fingerprint(run, second["tasks"]) == second_before
    state = read_json(run / "resumos/estado_execucao.json")
    assert state["total"] == state["COMPLETA"] == 4
    assert state["completa"] and not state["completa_grade"]
    assert_consolidated_matches_tasks(run, third)


def test_consolidate_both_ic_series_without_inputs_and_reject_corruption(tmp_path):
    fixture = make_composite_fixture(tmp_path, horizons=(1, 3), deltas=(0,),
                                     losses=("ranknet",), available_indices=())
    for index in (choose(fixture, horizon=1), choose(fixture)):
        publish_composite_task(fixture, index)
    run = tmp_path / "run"
    cli(*prepare_args(fixture, run))
    cli("run", "--run-dir", run, "--workers", "2")
    plan = read_json(run / "plano.json")
    originals = fingerprint(run, plan["tasks"])
    values = {name: read_parquet(run / "resumos" / f"{name}.parquet")
              for name in ("ic_diario", "ic_acumulado")}
    for path in (fixture.base, fixture.root, fixture.returns):
        path.rename(path.with_name("indisponivel_" + path.name))
    for name in values:
        (run / "resumos" / f"{name}.parquet").unlink()
    cli("consolidate", "--run-dir", run,
        entrypoint=run / "codigo/run_backtests_composto.py", cwd=tmp_path)
    for name, expected in values.items():
        pd.testing.assert_frame_equal(read_parquet(run / "resumos" / f"{name}.parquet"), expected)
    assert fingerprint(run, plan["tasks"]) == originals
    task = plan["tasks"][0]
    (run / "tasks" / task["relpath"] / "ic_acumulado.parquet").write_bytes(b"corrompido")
    result = cli("consolidate", "--run-dir", run, check=False)
    assert result.returncode == 1
    for name, expected in values.items():
        expected = expected[expected.task_id != task["id"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(read_parquet(run / "resumos" / f"{name}.parquet"), expected)
    assert read_json(run / "resumos/estado_execucao.json")["PENDENTE"] == 1


def test_contract_inputs_and_committed_prediction_cannot_change_in_place(tmp_path):
    fixture = make_composite_fixture(tmp_path, horizons=(1, 3), deltas=(0,),
                                     losses=("ranknet",), available_indices=())
    publish_composite_task(fixture, choose(fixture))
    run = tmp_path / "run"
    args = prepare_args(fixture, run)
    cli(*args)
    cli("run", "--run-dir", run, "--workers", "1")
    plan = read_json(run / "plano.json")
    original_plan = (run / "plano.json").read_bytes()
    outputs = fingerprint(run, plan["tasks"])
    # Cost scenarios, source configuration and base cannot mutate the contract.
    assert cli(*args, "--cost-bps", "5", check=False).returncode == 1
    config_path = fixture.root / "treino=50/experimento=composto/run_config.json"
    original_config = config_path.read_bytes()
    config = json.loads(original_config)
    config["config"]["strategy"]["seed"] += 1
    config_path.write_text(json.dumps(config))
    assert cli(*args, check=False).returncode == 1
    config_path.write_bytes(original_config)
    original_returns = fixture.returns.read_bytes()
    fixture.returns.write_bytes(original_returns + b"\n")
    assert cli(*args, check=False).returncode == 1
    fixture.returns.write_bytes(original_returns)
    assert (run / "plano.json").read_bytes() == original_plan
    assert fingerprint(run, plan["tasks"]) == outputs

    frozen_core = next((run / "codigo").rglob("core.py"))
    original_core = frozen_core.read_bytes()
    frozen_core.write_bytes(original_core + b"\n# snapshot modificado\n")
    assert cli("run", "--run-dir", run, "--workers", "1", check=False).returncode == 1
    assert cli("consolidate", "--run-dir", run, check=False).returncode == 1
    frozen_core.write_bytes(original_core)
    assert fingerprint(run, plan["tasks"]) == outputs

    # Atualizar o hash no registro do produtor nao autoriza substituir a mesma
    # tarefa que o consumidor ja aceitou no catalogo incremental.
    task = plan["tasks"][0]
    parquet = Path(task["path"])
    frame = read_parquet(parquet)
    ticker = fixture.market.tickers[0]
    frame[ticker] *= -1
    frame.to_parquet(parquet, index=False)
    manifest = config_path.with_name("completed_tasks.csv")
    completed = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    completed.loc[:, "parquet_sha256"] = digest(parquet)
    completed.to_csv(manifest, index=False)
    assert cli(*args, check=False).returncode == 1
    assert (run / "plano.json").read_bytes() == original_plan
    assert fingerprint(run, plan["tasks"]) == outputs
