"""Momentum: ausencia de vazamento, comparabilidade e preservacao da referencia."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import run_momentum_reference as momentum
from test_backtest_cpu import (cli as baseline_cli, prepare_args, event_market,
                              read_json, digest, load_bundle, backtest, Strategy)


def cli(reference, output, *extra, check=True):
    result = subprocess.run([sys.executable, "-B", str(ROOT/"scripts/run_momentum_reference.py"),
                             "--reference-run", str(reference), "--run-dir", str(output),
                             "--allow-local", *extra], text=True, capture_output=True, env=os.environ.copy())
    if check and result.returncode:
        pytest.fail(result.stdout+result.stderr)
    return result


@pytest.fixture
def completed_reference(tmp_path):
    args = prepare_args(tmp_path)
    baseline_cli(*args)
    reference = tmp_path/"run"
    baseline_cli("run", "--run-dir", reference, "--workers", "1")
    assert read_json(reference/"resumos/estado_execucao.json")["completa"]
    return reference


def test_signal_is_adjusted_price_ratio_known_at_origin():
    market = event_market()
    r = market.arrays["log_retorno_corrigido"]
    # O split/dividendo da segunda data nao cria o falso momentum de 45/100-1.
    score = momentum.momentum_scores(r, np.array([2, 3]), 1)
    assert score[:, 0] == pytest.approx([0., np.log(46/45)])
    assert score[0, 0] != pytest.approx(np.log(45/100))
    totals = momentum.momentum_scores(r, np.array([3]), 2)
    assert np.expm1(totals[0, 0]) == pytest.approx(46/45-1)
    changed = r.copy()
    changed[3:] = 1000.  # retorno futuro nao entra no score da origem 3
    assert momentum.momentum_scores(changed, np.array([3]), 2) == pytest.approx(totals)
    with pytest.raises(ValueError, match="Historico insuficiente"):
        momentum.momentum_scores(r, np.array([1]), 2)


def test_derive_grid_deduplicates_models_horizons_and_keeps_cuts():
    tasks = [{"treino": p, "lookback": l, "pred_len": h, "experimento": "temporal",
              "corte": 100*p//100, "common_last_origin": 75}
             for p in (50, 70) for l in (2, 4) for h in (5, 25) for _model in range(2)]
    plan = {"k": [1, 5, 25], "tasks": tasks}
    result = momentum.derive_tasks(plan, 100)
    assert [(t["treino"], t["lookback"]) for t in result] == [(50, 2), (50, 4), (70, 2), (70, 4)]
    assert all(t["pred_len"] is None and t["id"] == t["backtest_source_id"] for t in result)
    assert [t["corte"] for t in result] == [50, 50, 70, 70]
    tasks[0]["common_last_origin"] = 74
    with pytest.raises(ValueError, match="inconsistentes"):
        momentum.derive_tasks(plan, 100)


def test_reference_snapshot_no_predictions_no_mutations_and_exact_evaluation(completed_reference):
    reference = completed_reference
    root = reference.parent
    plan = read_json(reference/"plano.json")
    # Momento nao necessita dos Parquets de previsoes nem dos manifests de treino.
    (root/"previsoes").rename(root/"previsoes_indisponiveis")
    files = [p for p in reference.rglob("*") if p.is_file()]
    before = {p: (digest(p), p.stat().st_mtime_ns) for p in files}
    output = root/"momentum"
    result = cli(reference, output, "--dry-run")
    assert "1 tarefas" in result.stdout and not output.exists()
    # Imita a preparacao que o launcher Slurm faz antes de iniciar o Python.
    (output/"codigo").mkdir(parents=True)
    (output/"logs").mkdir(); (output/"tmp").mkdir()
    shutil.copyfile(ROOT/"scripts/run_momentum_reference.py", output/"codigo/run_momentum_reference.py")
    (output/"worker_momentum_aBc123.sh").write_text("#!/bin/bash\n")
    cli(reference, output)
    assert before == {p: (digest(p), p.stat().st_mtime_ns) for p in files}
    assert set(files) == {p for p in reference.rglob("*") if p.is_file()}
    mp = read_json(output/"plano.json")
    assert len(mp["tasks"]) == 1 and mp["reference_signature"] == plan["signature"]
    task = mp["tasks"][0]
    dest = output/"tasks"/task["relpath"]
    ic = pd.read_parquet(dest/"ic_acumulado.parquet")
    summary = pd.read_parquet(output/"resumos/ic_acumulado_resumo.parquet")
    metrics = pd.read_parquet(output/"resumos/metricas.parquet")
    assert not ic.duplicated(["treino", "lookback", "k", "origin_step"]).any()
    assert ic.pred_len.isna().all() and metrics.pred_len.isna().all()
    assert set(ic.k) == {1, 3} and len(summary) == 4 and len(metrics) == 8
    assert summary[summary.amostra == "comum"].n_origens.tolist() == [6, 6]
    assert summary[summary.amostra == "completa"].n_origens.tolist() == [8, 6]
    assert (ic.target_step == ic.origin_step+ic.k).all()
    assert ic[ic.k == 1].origin_step.tolist() == list(range(8, 16))
    assert ic[ic.k == 3].origin_step.tolist() == list(range(8, 14))
    market, _ = load_bundle(plan["inputs"]["base"], plan["inputs"]["retornos"])
    r = market.arrays["log_retorno_corrigido"]
    for row in ic.itertuples():
        manual_past = r[row.origin_step-2:row.origin_step].sum(axis=0)
        manual_future = r[row.origin_step:row.origin_step+row.k].sum(axis=0)
        assert row.spearman == pytest.approx(spearmanr(manual_past, manual_future).statistic)
    periods = pd.read_parquet(dest/"carteira_periodos.parquet")
    for k in (1, 3):
        origins = np.arange(8, 14, k)
        assert periods[periods.k == k].origin_step.unique().tolist() == origins.tolist()
        scores = np.stack([r[o-2:o].sum(axis=0) for o in origins])
        for definition in plan["strategies"]:
            expected = backtest(scores, origins, k, market, Strategy(**definition),
                                plan["slippage_bps"], plan["cost_bps"])["metricas"]
            actual = metrics[(metrics.k == k) & (metrics.estrategia == definition["id"])]
            pd.testing.assert_frame_equal(actual[expected.columns].reset_index(drop=True), expected,
                                          check_exact=False, rtol=1e-12, atol=1e-12)
    old_time = (dest/"ic_acumulado.parquet").stat().st_mtime_ns
    assert "REUTILIZADA" in cli(reference, output).stdout
    assert (dest/"ic_acumulado.parquet").stat().st_mtime_ns == old_time
    # Corrupcao recuperavel recompõe a tarefa, nunca a referencia.
    (dest/"ic_acumulado.parquet").write_text("quebrado")
    cli(reference, output)
    assert read_json(output/"resumos/estado_execucao.json")["completa"]
    (dest/"conclusao.json").write_text("{interrompido")
    cli(reference, output)
    assert read_json(output/"resumos/estado_execucao.json")["completa"]
    assert before == {p: (digest(p), p.stat().st_mtime_ns) for p in files}
    mp["cost_bps"] = 100
    (output/"plano.json").write_text(json.dumps(mp))
    before_output = {p: digest(p) for p in output.rglob("*") if p.is_file()}
    assert "distinta" in cli(reference, output, check=False).stderr
    assert before_output == {p: digest(p) for p in output.rglob("*") if p.is_file()}


def test_guards_reference_inputs_snapshot_and_partial_state(completed_reference):
    reference, root = completed_reference, completed_reference.parent
    output = root/"momentum"
    for forbidden in (reference/"momentum", root/"base"/"momentum", root,
                      root/"previsoes"/"momentum"):
        assert cli(reference, forbidden, "--dry-run", check=False).returncode == 1
    occupied = root/"ocupado"
    occupied.mkdir(); (occupied/"nao_apagar.txt").write_text("preservar")
    assert "ocupada" in cli(reference, occupied, "--dry-run", check=False).stderr
    assert {p.name for p in occupied.iterdir()} == {"nao_apagar.txt"}
    # Dry-run tambem valida que dados sao identicos aos da referencia.
    path = root/"base/mercado_diario.csv"
    original = path.read_bytes(); path.write_bytes(original+b"\n")
    assert "Checksum" in cli(reference, output, check=False).stderr
    assert not output.exists()
    path.write_bytes(original)
    code = reference/"codigo/backtest_cpu/core.py"
    original = code.read_bytes(); code.write_bytes(original+b"\n")
    assert "Snapshot" in cli(reference, output, check=False).stderr
    assert not output.exists()
    code.write_bytes(original)
    state_path = reference/"resumos/estado_execucao.json"
    state = read_json(state_path); state["completa"] = False
    state_path.write_text(json.dumps(state))
    assert "nao esta completa" in cli(reference, output, check=False).stderr
    assert not output.exists()


def test_parallel_workers_keep_task_results_and_reference_unchanged(completed_reference):
    reference = completed_reference
    parent = read_json(reference/"plano.json")
    # Adiciona um lookback ao contrato da referencia sintetica para exercitar
    # duas tarefas simultaneas; o benchmark nao consulta previsoes de redes.
    additions = []
    for original in parent["tasks"]:
        if original["experimento"] == "temporal":
            task = dict(original)
            task["id"] = len(parent["tasks"])+len(additions)
            task["lookback"] = 3
            task["backtest_source_id"] = task["id"]
            additions.append(task)
    parent["tasks"].extend(additions)
    del parent["signature"]
    parent["signature"] = momentum.json_digest(parent)
    (reference/"plano.json").write_text(json.dumps(parent))
    status_path = reference/"resumos/estado_execucao.json"
    status = read_json(status_path)
    status.update(signature=parent["signature"], total=len(parent["tasks"]), COMPLETA=len(parent["tasks"]))
    status_path.write_text(json.dumps(status))
    files = [p for p in reference.rglob("*") if p.is_file()]
    before = {p: (digest(p), p.stat().st_mtime_ns) for p in files}
    (reference.parent/"previsoes").rename(reference.parent/"previsoes_indisponiveis")
    output = reference.parent/"momentum"
    result = cli(reference, output, "--workers", "2")
    assert "2 processos" in result.stdout
    state = read_json(output/"resumos/estado_execucao.json")
    assert state["completa"] and state["COMPLETA"] == 2
    met = pd.read_parquet(output/"resumos/metricas.parquet")
    assert len(met) == 16 and set(met.lookback) == {2, 3}
    plan = read_json(output/"plano.json")
    market, _ = load_bundle(parent["inputs"]["base"], parent["inputs"]["retornos"])
    r = market.arrays["log_retorno_corrigido"]
    for task in plan["tasks"]:
        rows = pd.read_parquet(output/"tasks"/task["relpath"]/"ic_acumulado.parquet")
        for row in rows.itertuples():
            score = r[row.origin_step-task["lookback"]:row.origin_step].sum(axis=0)
            target = r[row.origin_step:row.origin_step+row.k].sum(axis=0)
            assert row.spearman == pytest.approx(spearmanr(score, target).statistic)
    timestamps = {p: p.stat().st_mtime_ns for p in (output/"tasks").rglob("*.parquet")}
    assert cli(reference, output, "--workers", "2").stdout.count("REUTILIZADA") == 2
    assert timestamps == {p: p.stat().st_mtime_ns for p in timestamps}
    assert before == {p: (digest(p), p.stat().st_mtime_ns) for p in files}


def test_strategy_failure_is_partial_and_never_reported_as_complete(completed_reference, monkeypatch):
    args = argparse.Namespace(reference_run=str(completed_reference),
                              run_dir=str(completed_reference.parent/"momentum"),
                              dry_run=False, allow_local=True)
    real_load = momentum.load_reference
    def load_with_blocked_strategy(options):
        values = real_load(options)
        engine = values[-1]
        original = engine.core.backtest
        def blocked(scores, origins, k, market, strategy, *costs):
            if strategy.id == "compra_ranking":
                raise ValueError("FECHAMENTO_INEXECUTAVEL: fixture; sem substituir ativo")
            return original(scores, origins, k, market, strategy, *costs)
        engine.core.backtest = blocked
        return values
    monkeypatch.setattr(momentum, "load_reference", load_with_blocked_strategy)
    old_tmp = tempfile.tempdir
    try:
        assert momentum.run(args) == 1
    finally:
        tempfile.tempdir = old_tmp
    out = Path(args.run_dir)
    state = read_json(out/"resumos/estado_execucao.json")
    assert not state["completa"] and state["PARCIAL"] == 1 and state["ERRO"] == 0
    failures = pd.read_parquet(out/"resumos/falhas.parquet")
    assert failures.k.tolist() == [1, 3]
    assert failures.estrategia.eq("compra_ranking").all()
    assert len(pd.read_parquet(out/"resumos/metricas.parquet")) == 6
    assert len(pd.read_parquet(out/"resumos/ic_acumulado_resumo.parquet")) == 4
