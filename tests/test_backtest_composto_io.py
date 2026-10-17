"""Contratos reais do produtor: publicacao incremental sem ler matrizes no plano."""
import copy
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import b3_experiment as producer
import b3_experiment_io as producer_io
from backtest_cpu.io import digest, load_bundle
from backtest_composto.io import discover_composite_tasks, load_composite_predictions
from test_backtest_cpu import make_inputs


def make_composite_fixture(tmp_path, percentages=(50,), horizons=(1, 3), deltas=(0, .025),
                           losses=("ranknet", "listnet"), available_indices=(0,)):
    """Pequena grade usando audit_row/export_prediction_windows do produtor real."""
    base, returns, root = make_inputs(tmp_path)
    market, identity = load_bundle(base, returns)
    prepared = tmp_path/"prepared_returns.csv"
    metadata = producer_io.prepare_dataset(returns, prepared, source_data_kind="log_return")
    config = {"source_data_kind": "log_return", "train_percentages": list(percentages),
              "models": ["DUET"], "lookbacks": [2], "horizons": list(horizons),
              "trade_windows": list(horizons), "cross_losses": list(losses),
              "lambdas": [.1, .5], "deltas": list(deltas), "var_lags": 2,
              "strategy": {"stride": 1, "train_ratio_in_tv": .875, "seed": 2026, "num_rollings": 100},
              "model_defaults": {"DUET": {"batch_size": 32, "norm": True}},
              "training": {"num_epochs": 2, "patience": 1, "lr": .001},
              "loss": {"loss_temporal": "mse", "loss_inverse_norm": True,
                       "loss_cross_score_normalization": "zscore", "loss_listnet_tau": 1.}}
    tasks = producer.make_tasks(config, "composto")
    signature = producer.digest(producer.provenance(config, metadata))
    fixture = SimpleNamespace(base=base, returns=returns, root=root, market=market, identity=identity,
                              config=config, producer_tasks=tasks, metadata=metadata, design_sha256=signature)
    for pct in percentages:
        group = root/f"treino={pct}"/"experimento=composto"
        group.mkdir(parents=True)
        plan = {"design_sha256": signature, "config": config, "metadata": metadata,
                "config_path": "fixture", "git_commit": "fixture"}
        (group/"run_config.json").write_text(json.dumps(plan))
        first = next(t for t in tasks if t["treino"] == pct)
        row = producer.audit_row(config, first, metadata, root, signature)
        row.update(slurm_job_id="", parquet_sha256="")
        with (group/"completed_tasks.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader()
    for index in available_indices:
        publish_composite_task(fixture, index)
    return fixture


def publish_composite_task(fixture, index, register=True):
    """Escreve um Parquet completo; register=False simula intervalo antes do commit."""
    task = fixture.producer_tasks[index]
    panel = producer_io.load_panel(Path(fixture.metadata["dataset"]))
    cut = len(panel)*task["treino"]//100
    h = task["pred_len"]
    actual = np.stack([panel.iloc[origin:origin+h].to_numpy() for origin in range(cut, len(panel)-h+1)])
    prediction = np.random.default_rng(index+2026).normal(0, .01, actual.shape)
    output = fixture.root/producer.output_path(task)
    result = producer_io.export_prediction_windows(actual, prediction, panel, cut, h, output, step_offset=1)
    row = producer.audit_row(fixture.config, task, fixture.metadata, fixture.root, fixture.design_sha256)
    row.update(slurm_job_id="fixture", parquet_sha256=result["parquet_sha256"])
    if register:
        producer.record_completed(fixture.root/producer.group_path(task), row)
    return row


def discover(fixture, percentages=None):
    selected = fixture.config["train_percentages"] if percentages is None else percentages
    return discover_composite_tasks(fixture.root, selected, fixture.market, fixture.identity["retornos_sha256"])


def rewrite_registry(fixture, mutate, pct=None):
    pct = fixture.config["train_percentages"][0] if pct is None else pct
    path = fixture.root/f"treino={pct}"/"experimento=composto"/"completed_tasks.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame = mutate(frame)
    frame.to_csv(path, index=False)
    return path


def test_partial_discovery_ignores_orphan_and_fixes_common_horizon(tmp_path, monkeypatch):
    f = make_composite_fixture(tmp_path)
    orphan = publish_composite_task(f, 1, register=False)
    # O planejamento nao materializa matrizes nem calcula SHA de todos os Parquets.
    import backtest_cpu.io as common_io
    monkeypatch.setattr(common_io, "read_table", lambda *a, **k: pytest.fail("leitura de previsoes no plano"))
    monkeypatch.setattr(common_io, "digest", lambda *a, **k: pytest.fail("hash integral no plano"))
    tasks, info = discover(f, [50, 70, 90])
    assert len(tasks) == 1 and "id" not in tasks[0] and "k" not in tasks[0]
    task = tasks[0]
    assert task["pred_len"] == 1 and task["k_dir"] == 1
    assert task["common_last_origin"] == f.market.n-3  # H=3 ainda nao foi registrado.
    assert task["lambda"] == "0.1" and task["delta"] == "0"
    group = info["groups"][0]
    assert group["available"] == group["registered"] == 1
    assert group["expected"] == len(f.producer_tasks) == 18
    assert group["awaiting_registration"] == group["awaiting"] == 17
    assert info["missing_groups"] == [70, 90] and not info["expected_complete"]
    assert len(info["manifests"]) == 2
    assert all(t["path"] != str(Path(orphan["output_dir"])/"part.parquet") for t in tasks)


def test_complete_rows_match_official_grid_listnet_and_load_exact_windows(tmp_path):
    f = make_composite_fixture(tmp_path, horizons=(1, 2, 3), available_indices=())
    for i in range(len(f.producer_tasks)):
        publish_composite_task(f, i)
    tasks, info = discover(f)
    assert info["available"] == info["expected"] == len(tasks) and info["awaiting"] == 0
    assert info["expected_complete"] and info["expected_count_known"]
    assert len({t["relpath"] for t in tasks}) == len(tasks)
    assert not any(t["pred_len"] == 3 and t["k_dir"] == 2 for t in tasks)
    for task, original in zip(tasks, f.producer_tasks):
        assert task["relpath"] == str(producer.output_path(original))
        assert task["training_array_index"] == original["array_index"]
        assert task["k_dir"] == original["k"] and task["pred_len"] % task["k_dir"] == 0
        if task["loss"] == "listnet":
            assert task["delta"] == "0" and task["cross_delta"] == 0 and not task["delta_effective"]
    task = next(t for t in tasks if t["pred_len"] == 3 and t["k_dir"] == 3)
    origins, predictions, accumulated = load_composite_predictions(task, f.market)
    expected = pd.read_parquet(task["path"]).sort_values(["origin_step", "h"])
    assert origins.tolist() == list(range(8, 14))
    np.testing.assert_array_equal(predictions, expected[f.market.tickers.tolist()].to_numpy().reshape(6, 3, 4))
    np.testing.assert_allclose(accumulated, predictions.cumsum(axis=1))


def test_registering_more_predictions_preserves_existing_identity_and_common_origins(tmp_path):
    f = make_composite_fixture(tmp_path, percentages=(50, 75))
    old, before = discover(f)
    assert before["expected_count_known"] and not before["expected_complete"]
    publish_composite_task(f, 2)
    second_cut = next(i for i, t in enumerate(f.producer_tasks) if t["treino"] == 75)
    publish_composite_task(f, second_cut)
    new, after = discover(f)
    assert new[0] == old[0]
    assert [t["treino"] for t in new] == [50, 50, 75]
    assert all(t["common_last_origin"] == 13 for t in new)
    assert new[-1]["corte"] == 12
    assert after["available"] == before["available"]+2


def test_registered_missing_and_absent_registry_are_explicit_pending(tmp_path):
    f = make_composite_fixture(tmp_path)
    tasks, _ = discover(f)
    Path(tasks[0]["path"]).unlink()
    tasks, info = discover(f)
    assert not tasks
    assert info["registered_missing"] == 1 and info["awaiting_registration"] == 17
    assert info["awaiting"] == 18
    registry = f.root/"treino=50/experimento=composto/completed_tasks.csv"
    registry.unlink()
    tasks, info = discover(f)
    assert not tasks and info["groups"][0]["status"] == "MISSING_REGISTRY"
    assert info["awaiting_registration"] == info["awaiting"] == 18


@pytest.mark.parametrize("field,value", [("source_sha256", "a"*64), ("design_sha256", "b"*64),
                                         ("cross_delta", ".2"), ("k", "2"), ("n_pretest", "9"),
                                         ("model_hyper_params", "{}"), ("parquet_sha256", "")])
def test_inconsistent_registered_provenance_is_rejected(tmp_path, field, value):
    f = make_composite_fixture(tmp_path)
    def change(frame):
        frame.loc[0, field] = value
        return frame
    rewrite_registry(f, change)
    with pytest.raises(ValueError):
        discover(f)


def test_listnet_delta_and_duplicate_registration_rejected(tmp_path):
    f = make_composite_fixture(tmp_path, losses=("listnet",), available_indices=(0,))
    rewrite_registry(f, lambda frame: pd.concat([frame, frame], ignore_index=True))
    with pytest.raises(ValueError, match="duplicada"):
        discover(f)
    publish_composite_task(f, 0)  # O produtor deduplica pelo array_index.
    def change(frame):
        frame.loc[0, "cross_delta"] = ".025"
        return frame
    rewrite_registry(f, change)
    with pytest.raises(ValueError, match="delta"):
        discover(f)


def test_schema_count_hash_and_prediction_alignment_are_separate_gates(tmp_path):
    f = make_composite_fixture(tmp_path, available_indices=(1,))
    tasks, _ = discover(f)
    task = tasks[0]
    frame = pd.read_parquet(task["path"])
    frame.loc[0, f.market.tickers[0]] += .1
    frame.to_parquet(task["path"], index=False)
    # Rodape ainda valido: SHA integral sera conferido apenas na tarefa.
    discover(f)
    with pytest.raises(ValueError, match="Checksum"):
        load_composite_predictions(task, f.market)
    task["sha256"] = digest(task["path"])
    frame.loc[0, "h"] = 2
    frame.to_parquet(task["path"], index=False)
    task["sha256"] = digest(task["path"])
    with pytest.raises(ValueError, match="desalinhado"):
        load_composite_predictions(task, f.market)
    frame.iloc[:-1].to_parquet(task["path"], index=False)
    with pytest.raises(ValueError, match="linhas/janelas"):
        discover(f)
    frame.drop(columns=f.market.tickers[0]).to_parquet(task["path"], index=False)
    with pytest.raises(ValueError, match="Schema/universo"):
        discover(f)
