"""Small end-to-end MASTER evaluation using the production exporters/CPU engines.

Only the fixed-size audit is replaced in orchestration tests; the market bundle,
completion records, Parquets, portfolio calculations and summaries are real.
"""
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import b3_experiment as producer
import b3_experiment_io as producer_io
from backtest_cpu.io import digest, load_bundle, RF_SOURCE, RF_CONVENTION
from test_backtest_cpu import market_frame


def master_inputs(root):
    """Mirror the three completed pilot fits without performing any training."""
    n, assets = 80, 12
    frame, calendar = market_frame(n, assets)
    base = root / "market"
    base.mkdir()
    returns = root / "returns.csv"
    frame[["date", "log_retorno_corrigido", "cols"]].rename(
        columns={"log_retorno_corrigido": "data"}).to_csv(returns, index=False)
    frame.to_csv(base / "mercado_diario.csv", index=False)
    calendar["retorno_cdi"] = .0002 + .000001 * np.arange(n)
    calendar["log_retorno_cdi"] = np.log1p(calendar.retorno_cdi)
    calendar["n_observacoes_cdi"] = 1
    calendar["rf_fonte"] = RF_SOURCE
    calendar.to_csv(base / "calendario_tfb.csv", index=False)
    pd.DataFrame([dict(versao="2026-10-15-v2-cdi", arquivo_base="mercado_diario.csv",
                       rf_fonte=RF_SOURCE, rf_convencao=RF_CONVENTION,
                       n_linhas=len(frame), n_ativos=assets, n_pregoes=n)]).to_csv(
                           base / "manifesto.csv", index=False)
    pd.DataFrame([
        dict(papel="RETORNOS_USADOS_NO_TREINO", arquivo=str(returns), md5=digest(returns, "md5")),
        dict(papel="cdi_tfb.csv", arquivo="fixture", md5="0" * 32),
    ]).to_csv(base / "fontes.csv", index=False)
    pd.DataFrame([dict(arquivo=p.name, md5=digest(p, "md5")) for p in sorted(base.iterdir())]).to_csv(
        base / "checksums_saidas.csv", index=False)
    market, identity = load_bundle(base, returns)
    metadata = producer_io.prepare_dataset(returns, root / "prepared.csv", source_data_kind="log_return")
    config = dict(
        source_data_kind="log_return", train_percentages=[80], models=["MASTER"],
        lookbacks=[32], horizons=[5], trade_windows=[1],
        cross_losses=["ranknet", "ranknet_proximity"], lambdas=[.5], deltas=[0], var_lags=32,
        strategy=dict(stride=1, train_ratio_in_tv=.875, seed=2026, num_rollings=48000),
        model_defaults={"MASTER": dict(batch_size=32, norm=True, seed=2026)},
        training=dict(num_epochs=100, patience=10, lr=.0001),
        loss=dict(loss_temporal="mse", loss_inverse_norm=True, loss_cross_score_normalization="zscore",
                  loss_listnet_tau=1., loss_cross_mask=["gaussian", .025, "natural"]),
    )
    predictions = root / "predictions"
    panel = producer_io.load_panel(Path(metadata["dataset"]))
    all_tasks = []
    for mode in ("temporal", "composto"):
        tasks = producer.make_tasks(config, mode)
        group = predictions / producer.group_path(tasks[0])
        group.mkdir(parents=True)
        signature = ("a" if mode == "temporal" else "b") * 64
        (group / "run_config.json").write_text(json.dumps(dict(
            config=config, metadata=metadata, design_sha256=signature, git_commit="fixture")))
        rows = []
        for task in tasks:
            cut = n * task["treino"] // 100
            h = task["pred_len"]
            actual = np.stack([panel.iloc[origin:origin + h].to_numpy()
                               for origin in range(cut, n - h + 1)])
            prediction = np.random.default_rng(2026 + len(all_tasks)).normal(0, .01, actual.shape)
            output = predictions / producer.output_path(task)
            result = producer_io.export_prediction_windows(
                actual, prediction, panel, cut, h, output, step_offset=1)
            (output / "master_checkpoint.pt").write_bytes(b"completed-checkpoint-must-not-change")
            row = producer.audit_row(config, task, metadata, predictions, signature)
            row.update(slurm_job_id="fixture", parquet_sha256=result["parquet_sha256"])
            rows.append(row)
            all_tasks.append(task)
        with (group / "completed_tasks.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return SimpleNamespace(base=base, returns=returns, root=predictions, config=config,
                           metadata=metadata, market=market, identity=identity, tasks=all_tasks)


def read_parquet(path):
    return pq.ParquetFile(path).read(use_threads=False).to_pandas()


def fingerprints(root):
    return {str(p.relative_to(root)): (digest(p), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def test_master_completion_contracts_are_readable_without_var(tmp_path):
    from backtest_cpu.io import discover_tasks, load_predictions
    from backtest_composto.io import discover_composite_tasks, load_composite_predictions

    fixture = master_inputs(tmp_path)
    temporal, _ = discover_tasks(fixture.root, [80], fixture.market,
                                fixture.identity["retornos_sha256"])
    composite, discovery = discover_composite_tasks(fixture.root, [80], fixture.market,
                                                   fixture.identity["retornos_sha256"])
    assert len(temporal) == 1 and len(composite) == 2
    assert discovery["expected_complete"]
    assert not (fixture.root / "treino=80/experimento=var").exists()
    expected_origins = np.arange(64, 76)
    for tasks, loader in ((temporal, load_predictions), (composite, load_composite_predictions)):
        for task in tasks:
            assert task["modelo"] == "MASTER"
            assert task["common_last_origin"] == 75
            origins, predictions, accumulated = loader(task, fixture.market)
            np.testing.assert_array_equal(origins, expected_origins)
            assert predictions.shape == (12, 5, 12)
            np.testing.assert_allclose(accumulated, predictions.cumsum(axis=1))


def evaluation_args(fixture, run):
    return SimpleNamespace(predictions=str(fixture.root), market=str(fixture.base),
                           returns_file=str(fixture.returns), run_dir=str(run),
                           workers=1, allow_local=True)


def test_both_families_use_same_protocol_and_reuse_results(tmp_path, monkeypatch):
    import evaluate_master_pilot as evaluation

    fixture = master_inputs(tmp_path)
    run = tmp_path / "evaluation"
    audit = {"status": "PASSED", "verified_paths": [str(p) for p in fixture.root.rglob("part.parquet")],
             "tasks": fixture.tasks}
    monkeypatch.setattr(evaluation, "validate_pilot", lambda *a, **k: audit)
    before = fingerprints(fixture.root)
    args = evaluation_args(fixture, run)
    report = evaluation.evaluate(args)
    assert report["status"] == "PASSED"
    assert report["n_predictions"] == 3 and report["n_portfolios"] == 12
    all_metrics, all_ic = [], []
    for mode, count in (("temporal", 1), ("composto", 2)):
        plan = json.loads((run / mode / "plano.json").read_text())
        assert len(plan["tasks"]) == count
        assert plan["k"] == [1, 5]
        assert plan["slippage_bps"] == plan["cost_bps"] == 0
        protocol = (json.loads((run / mode / "contrato.json").read_text())
                    if mode == "composto" else plan)
        assert protocol["regime"] == "abertura_t+1_fechamento_t+k"
        assert plan["inputs"]["rf_fonte"] == "BCB_SGS12"
        assert plan["strategies"] == [
            dict(id="compra_positivos", lado="compra", n_ativos=9, filtrar_sinal=True),
            dict(id="compra_ranking", lado="compra", n_ativos=9, filtrar_sinal=False),
        ]
        state = report["groups"][mode]
        assert state["completa"] and state["COMPLETA"] == count
        all_metrics.append(read_parquet(run / mode / "resumos/metricas.parquet"))
        all_ic.append(read_parquet(run / mode / "resumos/ic_diario.parquet"))
    metrics = pd.concat(all_metrics, ignore_index=True)
    assert len(metrics) == 12
    assert set(metrics.estrategia) == {"compra_positivos", "compra_ranking"}
    assert set(metrics.k) == {1, 5}
    assert metrics.status_rf.eq("BCB_SGS12").all()
    assert metrics.n_ativos_max.eq(9).all()
    # Both engines compare every loss at the exact same origins and leads.
    daily = pd.concat(all_ic, ignore_index=True)
    schedules = [x[["origin_step", "h", "target_step"]].reset_index(drop=True)
                 for _, x in daily.groupby(["experimento", "task_id"], sort=False)]
    assert len(schedules) == 3
    for schedule in schedules[1:]:
        pd.testing.assert_frame_equal(schedule, schedules[0])
    assert daily.amostra_comum.all()
    for name in ("comparacao_ic_diario.csv", "comparacao_ic_acumulado.csv", "comparacao_carteiras.csv"):
        comparison = pd.read_csv(run / name)
        assert set(comparison.experimento) == {"temporal", "composto"}
        assert comparison.groupby(["experimento", "task_id"]).ngroups == 3
    assert fingerprints(fixture.root) == before
    completed = {mode: fingerprints(run / mode / "tasks") for mode in ("temporal", "composto")}
    assert evaluation.evaluate(args)["status"] == "PASSED"
    assert fingerprints(fixture.root) == before
    assert {mode: fingerprints(run / mode / "tasks") for mode in completed} == completed


def test_failed_prediction_audit_never_starts_backtests(tmp_path, monkeypatch):
    import evaluate_master_pilot as evaluation

    fixture = master_inputs(tmp_path)
    run = tmp_path / "evaluation"
    before = fingerprints(fixture.root)
    monkeypatch.setattr(evaluation, "validate_pilot", lambda *a, **k: {"status": "FAILED"})
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Backtest started after failed audit"))
    with pytest.raises(ValueError):
        evaluation.evaluate(evaluation_args(fixture, run))
    assert not (run / "temporal/plano.json").exists()
    assert not (run / "composto/plano.json").exists()
    assert fingerprints(fixture.root) == before


def test_cross_family_calendar_mismatch_is_rejected(tmp_path):
    import evaluate_master_pilot as evaluation
    from backtest_cpu.io import discover_tasks
    from backtest_composto.io import discover_composite_tasks

    fixture = master_inputs(tmp_path)
    temporal, _ = discover_tasks(fixture.root, [80], fixture.market, fixture.identity["retornos_sha256"])
    composite, discovery = discover_composite_tasks(
        fixture.root, [80], fixture.market, fixture.identity["retornos_sha256"])
    protocol = dict(inputs=fixture.identity, k=[1, 5], slippage_bps=0, cost_bps=0,
                    strategies=json.loads(evaluation.STRATEGIES.read_text()))
    plans = {"temporal": dict(protocol, tasks=temporal),
             "composto": dict(protocol, tasks=composite, discovery=discovery)}
    evaluation.validate_plans(plans)
    composite[0]["common_last_origin"] -= 1
    with pytest.raises(ValueError, match="origens/amostras"):
        evaluation.validate_plans(plans)


def test_failed_engine_is_reported_without_publishing_comparisons(tmp_path, monkeypatch):
    import evaluate_master_pilot as evaluation

    fixture = master_inputs(tmp_path)
    run = tmp_path / "evaluation"
    monkeypatch.setattr(evaluation, "validate_pilot", lambda *a, **k: {"status": "PASSED"})
    calls = []

    def failed_command(command, **kwargs):
        calls.append(command)
        assert Path(command[2]).name == "run_backtests.py"
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(evaluation.subprocess, "run", failed_command)
    with pytest.raises((ValueError, OSError)):
        evaluation.evaluate(evaluation_args(fixture, run))
    assert len(calls) == 1
    assert json.loads((run / "avaliacao.json").read_text())["status"] == "FAILED"
    assert not list(run.glob("comparacao_*.csv"))
