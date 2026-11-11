"""Frozen opt-in ranking supervision contracts, without Torch or Slurm."""
from __future__ import annotations

import copy
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import b3_experiment as b3


def launcher_config(**changes):
    env = dict(os.environ)
    for key in ("CROSS_CONSTRUCTION", "RANK_WINDOWS", "B3_INLINE_CONFIG", "B3_CONFIG"):
        env.pop(key, None)
    env.update(TFB_ROOT=str(ROOT), PYTHON_BIN=sys.executable, VENV_PATH="/nonexistent",
               MODELS="Nonstationary_Transformer", TRAIN_PERCENTAGES="60 70 80 90",
               LOOKBACKS="32", HORIZONS="1 25", TRADE_WINDOWS="1",
               CROSS_LOSSES="rankmix rankmix_proximity", LAMBDAS="0.5 1", DELTAS="0",
               CROSS_MASK="cauchy 0.025 natural", SOURCE_DATA_KIND="log_return")
    env.update(changes)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_composite_trading_experiment.sh"), "--print-config"],
        env=env, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


@pytest.fixture
def config():
    return launcher_config(CROSS_CONSTRUCTION="origin_anchored_windows",
                           RANK_WINDOWS="1 5 10 15 20 25")


def test_absent_protocol_retains_legacy_configuration_and_tasks():
    config = launcher_config()
    assert not set(b3.RANKING_PROTOCOL_FIELDS).intersection(config["loss"])
    assert b3.ranking_protocol_settings(config) is None
    for task in b3.make_tasks(config, "composto"):
        assert "cross_construction" not in task
        assert "rank_windows" not in task
        assert "ranking_supervision_active" not in task
    explicit = launcher_config(CROSS_CONSTRUCTION="blocks")
    comparison = copy.deepcopy(explicit)
    comparison["loss"].pop("loss_cross_construction")
    assert comparison == config


def test_environment_freezes_integer_windows_and_worker_uses_frozen_config(config, monkeypatch):
    assert config["loss"]["loss_cross_construction"] == "origin_anchored_windows"
    assert config["loss"]["loss_rank_windows"] == [1, 5, 10, 15, 20, 25]
    monkeypatch.setenv("B3_INLINE_CONFIG", json.dumps(config))
    monkeypatch.setenv("B3_CONFIG_SOURCE", str(ROOT / "scripts/run_composite_trading_experiment.sh"))
    monkeypatch.setenv("CROSS_CONSTRUCTION", "blocks")
    monkeypatch.setenv("RANK_WINDOWS", "2 3")
    _, loaded = b3.load_config()
    assert loaded == config


@pytest.mark.parametrize("model", ["Nonstationary_Transformer", "simpleFORMER", "iTransformer"])
def test_all_architectures_use_native_adapter_and_same_effective_windows(config, model):
    config["models"] = [model]
    config["model_defaults"][model] = dict(norm=True, batch_size=16, d_model=128,
                                           d_ff=256, parallel_strategy=None)
    tasks = b3.make_tasks(config, "composto")
    assert len(tasks) == 4 * 1 * 2 * 2 * 2
    for task in tasks:
        name, hyper, adapter, _ = b3.model_spec(config, task)
        assert name == f"time_series_library.{model}"
        assert adapter == "transformer_adapter"
        assert "task_head_mode" not in hyper
        assert hyper["loss_cross_construction"] == "origin_anchored_windows"
        assert hyper["loss_rank_windows"] == [1, 5, 10, 15, 20, 25]
        assert task["rank_windows"] == ([1] if task["pred_len"] == 1 else [1, 5, 10, 15, 20, 25])
        assert task["ranking_supervision_active"] is True
    for task in b3.make_tasks(config, "temporal"):
        _, hyper, adapter, _ = b3.model_spec(config, task)
        assert hyper["loss"] == "mse" and hyper["loss_cross_lambda"] == 0
        assert adapter == "transformer_adapter"
        assert task["rank_windows"] == []
        assert task["ranking_supervision_active"] is False


@pytest.mark.parametrize("windows", [[], [5], [1, 1], [0, 1], [-1, 1], [True, 1], [1.0], "1 5"])
def test_rejects_invalid_or_unobservable_origin_windows(config, windows):
    config["loss"]["loss_rank_windows"] = windows
    with pytest.raises(ValueError):
        b3.make_tasks(config, "composto")


@pytest.mark.parametrize("change", ["missing_protocol", "bad_protocol", "blocks_with_windows",
                                    "k_not_one", "source_price", "loss_price",
                                    "momentum_output", "momentum_input", "task_heads"])
def test_rejects_ambiguous_or_incompatible_protocol(config, change):
    if change == "missing_protocol":
        config["loss"].pop("loss_cross_construction")
    elif change == "bad_protocol":
        config["loss"]["loss_cross_construction"] = "origin"
    elif change == "blocks_with_windows":
        config["loss"]["loss_cross_construction"] = "blocks"
    elif change == "k_not_one":
        config["trade_windows"] = [1, 5]
    elif change == "source_price":
        config["source_data_kind"] = "price"
    elif change == "loss_price":
        config["loss"]["loss_data_kind"] = "price"
    elif change == "momentum_output":
        config["loss"]["momentum_mode"] = "residual_output"
    elif change == "momentum_input":
        config["loss"]["momentum_input_mode"] = "feature"
    else:
        config["model_defaults"]["Nonstationary_Transformer"]["task_head_mode"] = "shared"
    with pytest.raises(ValueError):
        b3.ranking_protocol_settings(config)


def test_explicit_blocks_accepts_multiple_k_without_prefix_windows(config):
    config["loss"]["loss_cross_construction"] = "blocks"
    config["loss"].pop("loss_rank_windows")
    config["trade_windows"] = [1, 5, 25]
    tasks = b3.make_tasks(config, "composto")
    assert {(t["pred_len"], t["k"]) for t in tasks} == {(1, 1), (25, 1), (25, 5), (25, 25)}
    assert all(t["rank_windows"] == [] for t in tasks)


def test_origin_requires_economic_units_if_any_model_normalizes(config):
    config["loss"]["loss_inverse_norm"] = False
    config["models"] = ["Nonstationary_Transformer", "simpleFORMER"]
    config["model_defaults"]["Nonstationary_Transformer"]["norm"] = False
    config["model_defaults"]["simpleFORMER"]["norm"] = True
    with pytest.raises(ValueError, match="loss_inverse_norm=True"):
        b3.ranking_protocol_settings(config)
    config["model_defaults"]["simpleFORMER"]["norm"] = False
    assert b3.ranking_protocol_settings(config)["cross_construction"] == "origin_anchored_windows"
    # Explicit legacy block supervision keeps its prior normalization options.
    config["model_defaults"]["simpleFORMER"]["norm"] = True
    config["loss"]["loss_cross_construction"] = "blocks"
    config["loss"].pop("loss_rank_windows")
    assert b3.ranking_protocol_settings(config)["cross_construction"] == "blocks"


def test_registry_and_execution_metadata_identify_protocol_without_false_blocks(config, tmp_path):
    metadata = dict(n_rows=3664, n_assets=55, source="returns.csv", step_offset=1,
                    source_sha256="source", dataset_sha256="dataset", source_data_kind="log_return")
    tasks = b3.make_tasks(config, "composto")
    rows = [b3.audit_row(config, t, metadata, tmp_path, "design") for t in tasks]
    decoded = list(csv.DictReader(io.StringIO(b3.csv_text(rows))))
    for task, row in zip(tasks, decoded):
        assert row["cross_construction"] == "origin_anchored_windows"
        assert json.loads(row["rank_windows"]) == task["rank_windows"]
        assert row["cross_blocks"] == row["cross_steps"] == row["cross_tail_steps"] == ""
    task = next(t for t in tasks if t["pred_len"] == 25)
    destination = tmp_path / "forecast"
    b3.record_execution_metadata({"25": destination}, "parquet", "design",
                                 extra=b3.ranking_protocol_settings(config, task))
    saved = json.loads((destination / "execution_metadata.json").read_text())
    assert saved["rank_windows"] == [1, 5, 10, 15, 20, 25]
    assert saved["cross_construction"] == "origin_anchored_windows"


def test_frozen_provenance_covers_v3_and_protocol_parameters(config, monkeypatch):
    monkeypatch.setattr(b3, "file_hash", lambda path: str(path))
    metadata = dict(source_sha256="source", dataset_sha256="dataset", step_offset=1,
                    source_data_kind="log_return")
    first = b3.provenance(config, metadata)
    assert "ts_benchmark/baselines/composite_trading_loss_v3.py" in first["code_sha256"]
    original_signature = b3.digest(first)
    changed = copy.deepcopy(config)
    changed["loss"]["loss_rank_windows"] = [1, 25]
    assert b3.digest(b3.provenance(changed, metadata)) != original_signature
