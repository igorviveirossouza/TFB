"""Producer/consumer contracts for global rankings and non-MSE controls."""
import csv
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import b3_experiment as producer
from backtest_cpu.io import discover_tasks, load_predictions
from test_backtest_composto_io import (
    discover, make_composite_fixture, publish_composite_task, rewrite_registry,
)

GLOBAL_LOSSES = [name + suffix for name in ("spearman", "listnet", "listmle", "approxndcg")
                 for suffix in ("", "_proximity")]
TEMPERATURES = {"loss_spearman_tau": .3, "loss_listnet_tau": .4,
                "loss_listmle_tau": .6, "loss_approxndcg_tau": .8}


def publish_contract(fixture, mode):
    """Publish actual producer audits and Parquets for a modified tiny design."""
    fixture.producer_tasks = producer.make_tasks(fixture.config, mode)
    fixture.design_sha256 = producer.digest(producer.provenance(fixture.config, fixture.metadata))
    for pct in fixture.config["train_percentages"]:
        group = fixture.root / f"treino={pct}" / f"experimento={mode}"
        if group.exists():
            shutil.rmtree(group)
        group.mkdir(parents=True)
        plan = {"design_sha256": fixture.design_sha256, "config": fixture.config,
                "metadata": fixture.metadata, "config_path": "fixture", "git_commit": "fixture"}
        (group / "run_config.json").write_text(json.dumps(plan))
        first = next(task for task in fixture.producer_tasks if task["treino"] == pct)
        row = producer.audit_row(fixture.config, first, fixture.metadata, fixture.root,
                                 fixture.design_sha256)
        row.update(slurm_job_id="", parquet_sha256="")
        with (group / "completed_tasks.csv").open("w", newline="") as stream:
            csv.DictWriter(stream, fieldnames=list(row)).writeheader()
    for index in range(len(fixture.producer_tasks)):
        publish_composite_task(fixture, index)


@pytest.mark.parametrize("temporal_loss", ["mse", "mae", "huber"])
def test_temporal_producer_paths_match_backtest_and_keep_var_mse(tmp_path, temporal_loss):
    f = make_composite_fixture(tmp_path, available_indices=())
    f.config["loss"].update(loss_temporal=temporal_loss, loss_huber_delta=.7)
    publish_contract(f, "temporal")
    tasks, _ = discover_tasks(f.root, [50], f.market, f.identity["retornos_sha256"])
    temporal = [task for task in tasks if task["experimento"] == "temporal"]
    assert len(temporal) == len(f.producer_tasks)
    for task, original in zip(temporal, f.producer_tasks):
        assert task["loss"] == temporal_loss
        assert task["relpath"] == str(producer.output_path(original))
        assert load_predictions(task, f.market)[0].size > 0
    assert all(task["loss"] == "mse" for task in tasks if task["experimento"] == "var")


def test_temporal_registry_loss_must_match_declared_control(tmp_path):
    f = make_composite_fixture(tmp_path, available_indices=())
    f.config["loss"]["loss_temporal"] = "huber"
    publish_contract(f, "temporal")
    import pandas as pd
    registry = f.root / "treino=50/experimento=temporal/completed_tasks.csv"
    rows = pd.read_csv(registry)
    rows.loc[0, "loss"] = "mse"
    rows.to_csv(registry, index=False)
    with pytest.raises(ValueError, match="Proveniencia inconsistente"):
        discover_tasks(f.root, [50], f.market, f.identity["retornos_sha256"])


@pytest.mark.parametrize("temporal_loss", ["mse", "huber"])
def test_global_rankings_producer_and_backtest_share_grid_identity(tmp_path, temporal_loss):
    f = make_composite_fixture(tmp_path, losses=("ranknet",), available_indices=())
    f.config.update(cross_losses=GLOBAL_LOSSES,
                    deltas=[0., .25, .5], lambdas=[.5, 1.])
    f.config["loss"].update(loss_temporal=temporal_loss, loss_huber_delta=.7,
                           loss_cross_mask=["cauchy", .025], **TEMPERATURES)
    publish_contract(f, "composto")
    tasks, info = discover(f)
    assert info["expected_complete"]
    assert len(tasks) == len(f.producer_tasks) == 48
    assert {task["loss"] for task in tasks} == set(GLOBAL_LOSSES)
    for task, original in zip(tasks, f.producer_tasks):
        assert task["relpath"] == str(producer.output_path(original))
        assert task["temporal_loss"] == temporal_loss
        assert not task["delta_effective"] and task["cross_delta"] == 0
        assert task["training_array_index"] == original["array_index"]
        _, hyper, _, _ = producer.model_spec(f.config, original)
        assert all(hyper[key] == value for key, value in TEMPERATURES.items())
        assert hyper["loss_huber_delta"] == .7
        assert load_predictions(task, f.market)[0].size > 0


@pytest.mark.parametrize("temperature", ["loss_listmle_tau", "loss_approxndcg_tau"])
def test_registered_listwise_temperature_must_match_design(tmp_path, temperature):
    f = make_composite_fixture(tmp_path, losses=("ranknet",), available_indices=())
    f.config.update(cross_losses=["listmle_proximity", "approxndcg_proximity"],
                    deltas=[0., .25], lambdas=[1.])
    f.config["loss"].update(loss_cross_mask=["cauchy", .025], **TEMPERATURES)
    publish_contract(f, "composto")
    def change(frame):
        hyper = json.loads(frame.loc[0, "model_hyper_params"])
        hyper[temperature] = 99.0
        frame.loc[0, "model_hyper_params"] = json.dumps(hyper)
        return frame
    rewrite_registry(f, change)
    with pytest.raises(ValueError, match="Hiperparametros"):
        discover(f)
