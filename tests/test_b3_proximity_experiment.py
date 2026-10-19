"""Proximity objectives survive launcher, grid, audit and model-factory boundaries."""
from __future__ import annotations

import copy
from collections import Counter
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
import b3_experiment as experiment


PROXIMITY = ("hinge_proximity", "ranknet_proximity", "pairwise_mse_proximity")
ORIGINAL = ("hinge", "ranknet", "pairwise_mse", "listnet")


@pytest.fixture(scope="module")
def shell_config():
    result = subprocess.run(
        ["bash", str(REPO / "scripts/run_composite_trading_experiment.sh"), "--print-config"],
        env=dict(os.environ, TFB_ROOT=str(REPO), PYTHON_BIN=sys.executable),
        check=True, text=True, capture_output=True,
    )
    return json.loads(result.stdout)


@pytest.fixture
def config(shell_config):
    result = copy.deepcopy(shell_config)
    result.update(train_percentages=[80], models=["simpleMLP"], lookbacks=[32],
                  horizons=[5], trade_windows=[1, 5], lambdas=[.5],
                  deltas=[0, .025, .1], cross_losses=list(ORIGINAL + PROXIMITY))
    result["loss"]["loss_cross_mask"] = ["uniform", .025]
    return result


def load_inline(config, monkeypatch):
    monkeypatch.setenv("B3_INLINE_CONFIG", json.dumps(config))
    monkeypatch.setenv("B3_CONFIG_SOURCE", str(REPO / "scripts/run_composite_trading_experiment.sh"))
    return experiment.load_config()[1]


def test_mixed_grid_does_not_repeat_proximity_for_inactive_deltas(config):
    tasks = experiment.make_tasks(config, "composto")
    assert Counter(task["loss"] for task in tasks) == {
        "hinge": 6, "ranknet": 6, "pairwise_mse": 6, "listnet": 2,
        "hinge_proximity": 2, "ranknet_proximity": 2, "pairwise_mse_proximity": 2,
    }
    assert [task["array_index"] for task in tasks] == list(range(26))
    assert len({experiment.output_path(task) for task in tasks}) == len(tasks)
    new_tasks = [task for task in tasks if task["loss"] in PROXIMITY]
    assert all(task["cross_delta"] == 0 for task in new_tasks)
    assert all((task["mask_type"], task["mask_a"]) == ("uniform", .025) for task in new_tasks)
    config["deltas"] = [.2, .3, .4, .5]
    new_after = [task for task in experiment.make_tasks(config, "composto") if task["loss"] in PROXIMITY]
    without_index = lambda rows: [{key: value for key, value in row.items() if key != "array_index"} for row in rows]
    assert without_index(new_tasks) == without_index(new_after)


@pytest.mark.parametrize("mode", ["composto", "temporal", "var"])
def test_legacy_config_preserves_task_ids_paths_and_hyperparameters(config, monkeypatch, mode):
    config["cross_losses"] = list(ORIGINAL)
    config["loss"].pop("loss_cross_mask")
    legacy = load_inline(config, monkeypatch)
    before = experiment.make_tasks(legacy, mode)
    changed = copy.deepcopy(legacy)
    changed["loss"]["loss_cross_mask"] = ["gaussian", .123456789]
    changed = load_inline(changed, monkeypatch)
    after = experiment.make_tasks(changed, mode)
    assert before == after
    for old, new in zip(before, after):
        assert experiment.output_path(old) == experiment.output_path(new)
        assert experiment.model_spec(legacy, old) == experiment.model_spec(changed, new)
        assert "mask_type" not in new and "mask_a" not in new
        assert "loss_cross_mask" not in experiment.model_spec(changed, new)[1]


def test_default_does_not_expand_existing_grid(shell_config):
    assert shell_config["cross_losses"] == ["ranknet", "hinge", "listnet", "pairwise_mse"]
    assert len(experiment.make_tasks(shell_config, "composto")) == 33696


@pytest.mark.parametrize("mask", [
    None, [], ["uniform"], ["uniform", .1, .2], "uniform 0.1",
    {"type": "uniform", "a": .1}, ["triangular", .1], ["uniform", -.1],
    ["gaussian", 0], ["gaussian", -.1], ["uniform", float("nan")],
    ["uniform", float("inf")], ["gaussian", float("-inf")],
    ["uniform", True], ["uniform", "abc"],
])
def test_bad_mask_fails_before_any_training(config, monkeypatch, mask):
    config["loss"]["loss_cross_mask"] = mask
    with pytest.raises(ValueError):
        load_inline(config, monkeypatch)


def test_missing_mask_for_new_loss_is_rejected(config, monkeypatch):
    config["loss"].pop("loss_cross_mask")
    with pytest.raises(ValueError):
        load_inline(config, monkeypatch)


@pytest.mark.parametrize("mask", [["uniform", 0], ["uniform", .025], ["gaussian", .025]])
def test_valid_masks_are_accepted(config, monkeypatch, mask):
    config["loss"]["loss_cross_mask"] = mask
    loaded = load_inline(config, monkeypatch)
    assert loaded["loss"]["loss_cross_mask"] == mask


def test_paths_preserve_mask_identity_even_for_adjacent_float_values(config):
    config["cross_losses"] = ["hinge_proximity"]
    paths = []
    for mode in ("uniform", "gaussian"):
        for radius in (.025, math.nextafter(.025, math.inf)):
            config["loss"]["loss_cross_mask"] = [mode, radius]
            task = experiment.make_tasks(config, "composto")[0]
            path = experiment.output_path(task)
            assert "delta=0" in path.parts
            assert f"mask={mode}" in path.parts
            assert float(next(p[2:] for p in path.parts if p.startswith("a="))) == radius
            paths.append(path)
    assert len(set(paths)) == 4


def test_mixed_audit_rows_have_consistent_mask_columns(config, tmp_path):
    tasks = experiment.make_tasks(config, "composto")
    metadata = dict(n_rows=1000, source="synthetic.csv", source_data_kind="log_return",
                    step_offset=0, source_sha256="source", dataset_sha256="dataset", n_assets=3)
    rows = [experiment.audit_row(config, task, metadata, tmp_path, "signature") for task in tasks]
    assert all(set(row) == set(rows[0]) for row in rows)
    serialized = list(csv.DictReader(io.StringIO(experiment.csv_text(rows))))
    assert len(serialized) == len(tasks)
    for task, row in zip(tasks, serialized):
        hyper = json.loads(row["model_hyper_params"])
        if task["loss"] in PROXIMITY:
            assert row["mask_type"] == "uniform" and float(row["mask_a"]) == .025
            assert row["mask_effective"] == "True" and row["delta_effective"] == "False"
            assert float(row["cross_delta"]) == 0
            assert hyper["loss_cross_mask"] == ["uniform", .025]
        else:
            assert row["mask_type"] == row["mask_a"] == ""
            assert row["mask_effective"] == "False"
            assert "loss_cross_mask" not in hyper


def editable_launcher(tmp_path, mask):
    content = (REPO / "scripts/run_composite_trading_experiment.sh").read_text()
    arrays = dict(TRAIN_PERCENTAGES="80", MODELS="simpleMLP", LOOKBACKS="32",
                  HORIZONS="5", TRADE_WINDOWS="5", LAMBDAS="0.5",
                  CROSS_LOSSES=" ".join(PROXIMITY), CROSS_MASK=mask)
    for name, value in arrays.items():
        content, count = re.subn(rf"^{name}=\([^\n]*\)", f"{name}=({value})", content, flags=re.MULTILINE)
        assert count == 1, f"editable {name} vector is missing or ambiguous"
    script = tmp_path / "composite.sh"
    script.write_text(content)
    marker = tmp_path / "sbatch_was_called"
    executable = tmp_path / "sbatch"
    executable.write_text('#!/usr/bin/env python3\nimport pathlib\npathlib.Path(' + repr(str(marker)) + ').touch()\nraise SystemExit(90)\n')
    executable.chmod(0o755)
    env = dict(os.environ, TFB_ROOT=str(REPO), PYTHON_BIN=sys.executable,
               VENV_PATH=str(tmp_path / "no-venv"), DATASET_PATH=str(tmp_path / "no-data.csv"),
               OUT_ROOT=str(tmp_path / "out"), RESULT_ROOT=str(tmp_path / "results"),
               LOG_ROOT=str(tmp_path / "logs"), PATH=str(tmp_path) + os.pathsep + os.environ["PATH"])
    return script, env, marker


@pytest.mark.parametrize("mode", ["uniform", "gaussian"])
def test_shell_vector_reaches_inspection_and_dry_run_without_submission(tmp_path, mode):
    script, env, marker = editable_launcher(tmp_path, f"{mode} 0.01234567890123456")
    command = ["bash", str(script)]
    printed = subprocess.run(command + ["--print-config"], env=env, check=True, capture_output=True, text=True)
    assert json.loads(printed.stdout)["loss"]["loss_cross_mask"] == [mode, .01234567890123456]
    for index, loss in enumerate(PROXIMITY):
        shown = subprocess.run(command + ["--task", str(index)], env=env, check=True, capture_output=True, text=True)
        task = json.loads(shown.stdout)
        assert task["loss"] == loss and task["cross_delta"] == 0
        assert task["model_spec"][1]["loss_cross_mask"] == [mode, .01234567890123456]
        assert f"mask={mode}" in Path(task["output_dir"]).parts
    preview = subprocess.run(command + ["--dry-run"], env=env, check=True, capture_output=True, text=True)
    assert "3 tarefas" in preview.stdout
    assert not marker.exists()
    assert not any((tmp_path / name).exists() for name in ("out", "results", "logs"))


@pytest.mark.parametrize("mask", ["uniform", "uniform 0.1 0.2", "gaussian 0", "uniform nan"])
def test_invalid_shell_vector_fails_without_submission_or_outputs(tmp_path, mask):
    script, env, marker = editable_launcher(tmp_path, mask)
    result = subprocess.run(["bash", str(script), "--dry-run"], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert not marker.exists() and not (tmp_path / "out").exists()


@pytest.mark.parametrize("mode", ["uniform", "gaussian"])
@pytest.mark.parametrize("loss_name", PROXIMITY)
def test_serialized_benchmark_hyperparameters_build_the_intended_loss(config, tmp_path, mode, loss_name):
    torch = pytest.importorskip("torch")
    from ts_benchmark.baselines.composite_trading_loss_v3 import build_loss

    config.update(cross_losses=[loss_name], horizons=[1], trade_windows=[1])
    config["loss"].update(loss_cross_mask=[mode, .025], loss_cross_score_normalization="none")
    task = experiment.make_tasks(config, "composto")[0]
    command = experiment.benchmark_command(config, task, "synthetic.csv", 1000, tmp_path)
    hyper = json.loads(command[command.index("--model-hyper-params") + 1])
    assert hyper["loss_cross"] == loss_name and hyper["loss_cross_mask"] == [mode, .025]
    criterion = build_loss(SimpleNamespace(**hyper), normalizer_mean=[0]*3, normalizer_scale=[1]*3)
    pred_returns, true_returns = [.03, -.01, .1], [.01, .01, .04]
    pred = torch.tensor([[[math.log1p(x) for x in pred_returns]]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[[math.log1p(x) for x in true_returns]]], dtype=torch.float64)
    weighted, total_weight = 0., 0.
    for i in range(3):
        for j in range(i + 1, 3):
            predicted_gap = pred_returns[i] - pred_returns[j]
            gap = true_returns[i] - true_returns[j]
            weight = float(abs(gap) <= .025) if mode == "uniform" else math.exp(-gap**2 / (2*.025**2))
            sign = 0 if gap == 0 else (1 if gap > 0 else -1)
            if loss_name == "hinge_proximity":
                value = abs(predicted_gap) if sign == 0 else max(0., hyper["loss_hinge_margin"] - sign * predicted_gap)
            elif loss_name == "pairwise_mse_proximity":
                value = (predicted_gap - gap)**2
            else:
                logit = hyper["loss_ranknet_alpha"] * predicted_gap
                value = max(logit, 0.) + math.log1p(math.exp(-abs(logit))) - ((1+sign)/2) * logit
            weighted += weight * value
            total_weight += weight
    temporal = torch.nn.functional.mse_loss(pred, target)
    expected = (1-hyper["loss_cross_lambda"]) * temporal + hyper["loss_cross_lambda"] * hyper["loss_cross_scale"] * weighted / total_weight
    actual = criterion(pred, target)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    actual.backward()
    assert torch.isfinite(pred.grad).all()
