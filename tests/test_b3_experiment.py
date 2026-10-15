"""Experiment design, split, scheduler and audit contracts without GPU training."""
from __future__ import annotations

import copy
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
import b3_experiment as experiment
import b3_experiment_io as experiment_io


@pytest.fixture
def config():
    result = subprocess.run(
        ["bash", str(REPO / "scripts/run_composite_trading_experiment.sh"), "--print-config"],
        env=dict(__import__("os").environ, TFB_ROOT=str(REPO), PYTHON_BIN=sys.executable),
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("mode,count", [("temporal", 324), ("composto", 33696), ("var", 3)])
def test_full_design_has_unique_destinations_and_contiguous_global_ids(config, mode, count):
    tasks = experiment.make_tasks(config, mode)
    paths = [experiment.output_path(task) for task in tasks]
    assert len(tasks) == len(set(paths)) == count
    assert [task["array_index"] for task in tasks] == list(range(count))
    for task, path in zip(tasks, paths):
        parts = path.parts
        assert parts[0] == f"treino={task['treino']}"
        assert parts[1] == f"experimento={mode}"
        assert not any("seed" in part or "job" in part for part in parts)
        assert any(part.startswith("k_dir=") for part in parts) == (mode == "composto")
        assert any(part.startswith("delta=") for part in parts) == (mode == "composto")
        if mode != "composto":
            assert "lambda=00" in parts


def test_composite_grid_has_no_effective_duplicates(config):
    tasks = experiment.make_tasks(config, 'composto')
    assert Counter(t['loss'] for t in tasks) == {
        'ranknet': 10368, 'hinge': 10368, 'pairwise_mse': 10368, 'listnet': 2592,
    }
    # Ignore delta for ListNet exactly as its objective does. Unique directory
    # names alone cannot detect duplicate training objectives.
    active_keys = [tuple(t[key] for key in ('treino', 'modelo', 'lookback', 'pred_len',
                                          'k', 'loss', 'cross_lambda')) +
                   (None if t['loss'] == 'listnet' else t['cross_delta'],)
                   for t in tasks]
    assert len(set(active_keys)) == len(tasks)
    for loss in config['cross_losses']:
        actual_deltas = {t['cross_delta'] for t in tasks if t['loss'] == loss}
        assert actual_deltas == ({0} if loss == 'listnet' else set(config['deltas']))


@pytest.mark.parametrize('deltas', [[0.05], [0.1, 0.025], [0.1, 0.05, 0.025, 0], [0, 0.025, 0.05, 0.1, 0.2]])
def test_editing_delta_vector_does_not_repeat_remove_or_change_listnet(config, deltas):
    def listnet_design(tasks):
        return [{key: value for key, value in task.items() if key != 'array_index'}
                for task in tasks if task['loss'] == 'listnet']
    before = listnet_design(experiment.make_tasks(config, 'composto'))
    config['deltas'] = deltas
    tasks = experiment.make_tasks(config, 'composto')
    assert listnet_design(tasks) == before
    assert len(before) == 2592
    for loss in ('ranknet', 'hinge', 'pairwise_mse'):
        assert sum(t['loss'] == loss for t in tasks) == 2592 * len(deltas)
        assert {t['cross_delta'] for t in tasks if t['loss'] == loss} == set(deltas)


def test_full_grid_preserves_training_parameters_and_valid_hk_pairs(config):
    expected_pairs = {
        1: {1}, 5: {1, 5}, 10: {1, 5, 10},
        15: {1, 5, 15}, 20: {1, 5, 10, 20}, 25: {1, 5, 25},
    }
    observed_pairs = {h: set() for h in expected_pairs}
    observed_models = set()
    for mode in ("temporal", "composto"):
        for task in experiment.make_tasks(config, mode):
            name, hyper, adapter, deterministic = experiment.model_spec(config, task)
            observed_models.add(task["modelo"])
            assert hyper["num_epochs"] == 100
            assert hyper["patience"] == 10
            assert hyper["lr"] == 1e-4
            assert hyper["seq_len"] == task["lookback"]
            assert hyper["pred_len"] == hyper["horizon"] == task["pred_len"]
            assert hyper["norm"] is True
            if task["modelo"] == "DUET":
                assert name == "duet.DUET" and adapter is None
            else:
                assert adapter == "transformer_adapter"
            assert deterministic == ("efficient" if task["modelo"] == "TimesNet" else "full")
            if mode == "composto":
                observed_pairs[task["pred_len"]].add(task["k"])
                assert hyper["loss_trade_window"] == task["k"]
                assert hyper["loss_cross_lambda"] in {0.1, 0.5, 0.8}
                assert hyper["loss_cross_delta"] in {0, 0.025, 0.05, 0.1}
                assert hyper["loss_ranknet_alpha"] == hyper["loss_listnet_tau"] == hyper["loss_cross_scale"] == 1
                assert hyper["loss_hinge_margin"] == 0.1
            else:
                assert task["k"] is None
                assert hyper["loss_cross_lambda"] == 0
    assert observed_models == set(config["models"])
    assert observed_pairs == expected_pairs


@pytest.mark.parametrize("prediction_scale", [0.02, 1000.0])
def test_temporal_criterion_is_exact_mse_in_value_and_gradient(config, prediction_scale):
    torch = pytest.importorskip("torch", reason="PyTorch is needed to verify the actual training criterion")
    from ts_benchmark.baselines.composite_trading_loss_v3 import build_loss

    task = experiment.make_tasks(config, "temporal")[0]
    _, hyper, _, _ = experiment.model_spec(config, task)
    criterion = build_loss(SimpleNamespace(**hyper), normalizer_mean=[0, 0], normalizer_scale=[1, 1])
    prediction = torch.tensor([[[prediction_scale, -prediction_scale]]], requires_grad=True)
    target = torch.tensor([[[0.01, -0.03]]])
    result = criterion(prediction, target)
    expected = torch.nn.functional.mse_loss(prediction, target)
    actual_gradient = torch.autograd.grad(result, prediction, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, prediction)[0]
    assert torch.isfinite(result)
    assert torch.isfinite(actual_gradient).all()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


def test_cutoffs_match_integer_floor_even_at_float_roundoff_edges(config):
    task = experiment.make_tasks(config, "temporal")[0]
    edge_count = 0
    for n_rows in range(1, 10001):
        for percent in (70, 80, 90):
            task["treino"] = percent
            expected = n_rows * percent // 100
            split = experiment.split_counts(config, task, n_rows)
            strategy = experiment.strategy_args(config, task, n_rows)
            assert split["n_pretest"] == expected
            assert int(strategy["tv_ratio"] * n_rows) == expected
            assert split["n_train"] + split["n_validation"] == expected
            assert split["n_test"] + expected == n_rows
            assert split["n_train"] == int(expected * 0.875)
            edge_count += int((percent / 100) * n_rows) != expected
    assert edge_count > 0  # The test actually exercises the float-cutoff defect.
    task["modelo"] = "VAR"
    split = experiment.split_counts(config, task, 1000)
    assert split["n_validation"] == 0
    assert experiment.strategy_args(config, task, 1000)["train_ratio_in_tv"] == 1


def test_composite_chunks_cover_each_task_once():
    chunks = experiment.array_chunks(33696)
    assert len(chunks) == 34
    assert chunks[-1] == (33000, 696)
    assert all(count == 1000 for _, count in chunks[:-1])
    assert [offset + local for offset, count in chunks for local in range(count)] == list(range(33696))
    with pytest.raises(ValueError):
        experiment.array_chunks(33696, 0)


@pytest.mark.parametrize("mode,count,expected_chunks", [("temporal", 324, 1), ("composto", 33696, 34), ("var", 3, 1)])
def test_submission_resources_offsets_dependencies_and_export(monkeypatch, tmp_path, mode, count, expected_chunks):
    monkeypatch.setattr(experiment.shutil, "which", lambda name: "/fake/sbatch" if name == "sbatch" else None)
    for name, value in {"LOG_ROOT": str(tmp_path / "logs"), "ARRAY_CHUNK_SIZE": "1000", "MAX_GPU_JOBS": "6", "GPU_PARTITION": "medusas_dev", "GPU_TIME": "08:00:00", "VENV_PATH": "/env/activate", "PYTHON_BIN": "/env/python"}.items():
        monkeypatch.setenv(name, value)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=f"{9000 + len(calls)};cluster\n")

    monkeypatch.setattr(experiment.subprocess, "run", fake_run)
    out_root, config_path = tmp_path / "results", tmp_path / "config.json"
    experiment.submit(mode, [None] * count, out_root, config_path)
    assert len(calls) == expected_chunks
    for number, (command, kwargs) in enumerate(calls):
        offset = number * 1000
        size = min(1000, count - offset)
        assert f"--array=0-{size-1}%6" in command
        assert "--time=08:00:00" in command
        assert command[command.index("-p") + 1] == "medusas_dev"
        assert ("--gres=gpu:1" in command) == (mode != "var")
        assert "--export=ALL" in command
        assert command[-3:] == ["worker", "--offset", str(offset)]
        dependencies = [part for part in command if part.startswith("--dependency=")]
        assert dependencies == ([] if number == 0 else [f"--dependency=afterany:{9000+number}"])
        assert kwargs["check"] is True
        assert kwargs["env"]["OUT_ROOT"] == str(out_root)
        assert kwargs["env"]["B3_CONFIG"] == str(config_path)
        assert kwargs["env"]["VENV_PATH"] == "/env/activate"
        assert kwargs["env"]["PYTHON_BIN"] == "/env/python"


def test_submission_respects_smaller_discovered_array_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(experiment.shutil, "which", lambda name: f"/fake/{name}")
    monkeypatch.setenv("LOG_ROOT", str(tmp_path))
    monkeypatch.setenv("ARRAY_CHUNK_SIZE", "1000")
    commands = []

    def fake_run(command, **kwargs):
        if command[0] == "scontrol":
            return SimpleNamespace(stdout="MaxArraySize = 41\n")
        commands.append(command)
        return SimpleNamespace(stdout=str(100 + len(commands)))

    monkeypatch.setattr(experiment.subprocess, "run", fake_run)
    experiment.submit("temporal", [None] * 95, tmp_path / "out", tmp_path / "config.json")
    assert [command[-1] for command in commands] == ["0", "41", "82"]
    assert [next(part for part in command if part.startswith("--array=")) for command in commands] == ["--array=0-40%6", "--array=0-40%6", "--array=0-12%6"]


@pytest.fixture
def prepared_design(config, monkeypatch, tmp_path, request):
    # Isolate generated inputs and audit files from the actual repository.
    fake_root = tmp_path / "checkout"
    for relative in ("scripts/b3_experiment.py", "scripts/b3_experiment_io.py", "scripts/run_benchmark_composite_trading_loss_v3.py", "ts_benchmark/baselines/composite_trading_loss_v2.py", "ts_benchmark/baselines/composite_trading_loss_v3.py", "ts_benchmark/baselines/self_impl/VAR/VAR.py", "ts_benchmark/baselines/deep_forecasting_model_base.py"):
        destination = fake_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, destination)
    source_kind = getattr(request, 'param', 'price')
    source = tmp_path / "source.csv"
    rows = 1001 if source_kind == 'price' else 1000
    if source_kind == 'price':
        data = pd.DataFrame({"date": np.arange(1, rows + 1), "A3": 100 * np.exp(np.arange(rows) * 0.0001), "B4": 50 * np.exp(np.arange(rows) * -0.0002)})
    else:
        data = pd.DataFrame(np.random.default_rng(2026).normal(0, 0.01, (rows, 2)), columns=['A3', 'B4'])
        data.iloc[0, 0] = 0
        data.insert(0, 'date', np.arange(1, rows + 1))
    data.to_csv(source, index=False)
    monkeypatch.setattr(experiment, "ROOT", fake_root)
    monkeypatch.setenv("DATASET_PATH", str(source))
    monkeypatch.setenv("RESULT_ROOT", str(tmp_path / "raw_results"))
    monkeypatch.setattr(experiment, "register_dataset", lambda dataset, metadata: None)
    monkeypatch.setattr(experiment.subprocess, "check_output", lambda *args, **kwargs: "test-commit\n")
    config["models"] = ["DUET"]
    config["source_data_kind"] = source_kind
    config["lookbacks"] = [32]
    config["horizons"] = [5]
    tasks = experiment.make_tasks(config, "temporal")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    out_root = tmp_path / "out"
    metadata = experiment.prepare(config_path, config, tasks, out_root)
    return SimpleNamespace(config=config, config_path=config_path, tasks=tasks, out=out_root, metadata=metadata)


@pytest.mark.parametrize('prepared_design', ['log_return'], indirect=True)
@pytest.mark.parametrize('mode', ['temporal', 'composto', 'var'])
def test_log_return_workers_preserve_cutoffs_and_export_source_steps(prepared_design, monkeypatch, mode):
    pytest.importorskip('pyarrow')
    design = prepared_design
    config = design.config
    tasks = experiment.make_tasks(config, mode)
    metadata = experiment.prepare(design.config_path, config, tasks, design.out)
    assert metadata['n_source_rows'] == metadata['n_rows'] == 1000
    assert metadata['source_data_kind'] == 'log_return'
    panel = experiment_io.load_panel(Path(metadata['dataset']))
    source = experiment_io.load_panel(Path(metadata['source']))
    np.testing.assert_array_equal(panel.to_numpy(), source.to_numpy())
    # Simulate predictions only; preparation, worker routing, validation,
    # real Parquet writes and completion manifests run without a GPU/Slurm.
    monkeypatch.setattr(experiment.subprocess, 'run', lambda *args, **kwargs: None)
    for pct in (70, 80, 90):
        task = next(t for t in tasks if t['treino'] == pct)
        cut = 1000 * pct // 100
        h = task['pred_len']
        actual = np.stack([panel.iloc[i:i+h].to_numpy() for i in range(cut, len(panel)-h+1)])
        monkeypatch.setattr(experiment_io, '_read_payload', lambda _, actual=actual: (actual, actual.copy()))
        forecasts = np.zeros((len(panel)-cut, h, panel.shape[1]))
        forecasts[:len(actual)] = actual
        monkeypatch.setattr(experiment, 'var_forecasts', lambda *args, forecasts=forecasts: forecasts)
        experiment.worker(config, tasks, design.out, task['array_index'])
        frame = pd.read_parquet(design.out / experiment.output_path(task) / 'part.parquet')
        assert frame.iloc[0]['origin_step'] == cut
        assert frame.iloc[0]['step'] == cut + 1
        assert frame.iloc[-1]['step'] == 1000
        assert (frame['step'] - frame['origin_step'] == frame['h']).all()
        # Join exported targets directly to the original return input, so an
        # otherwise self-consistent off-by-one shift cannot pass this check.
        np.testing.assert_array_equal(frame[panel.columns].to_numpy(), source.loc[frame['step']].to_numpy())
        completed = design.out / experiment.group_path(task) / 'completed_tasks.csv'
        row = next(csv.DictReader(completed.open()))
        assert int(row['n_pretest']) == cut
        assert row['source_data_kind'] == 'log_return'
        assert int(row['step_offset']) == 1


def test_prepare_is_repeatable_and_audits_all_three_cuts(prepared_design):
    design = prepared_design
    dataset = Path(design.metadata["dataset"])
    before = dataset.stat().st_mtime_ns
    metadata = experiment.prepare(design.config_path, design.config, design.tasks, design.out)
    assert metadata == design.metadata
    assert dataset.stat().st_mtime_ns == before
    for task in design.tasks:
        group = design.out / experiment.group_path(task)
        manifest = group / "design_composite_trading.csv"
        assert manifest.read_bytes() == (group / "desing_composite_trading.csv").read_bytes()
        row = list(csv.DictReader(manifest.open()))
        assert len(row) == 1
        assert int(row[0]["n_pretest"]) == 1000 * task["treino"] // 100
        assert row[0]["dataset_sha256"] == experiment.file_hash(dataset)
        assert list(csv.DictReader((group / "completed_tasks.csv").open())) == []


def test_composite_audit_records_listnet_once_with_inactive_delta(prepared_design):
    design = prepared_design
    tasks = experiment.make_tasks(design.config, 'composto')
    experiment.prepare(design.config_path, design.config, tasks, design.out)
    for pct in (70, 80, 90):
        group = design.out / f'treino={pct}' / 'experimento=composto'
        manifest = group / 'design_composite_trading.csv'
        assert manifest.read_bytes() == (group / 'desing_composite_trading.csv').read_bytes()
        rows = list(csv.DictReader(manifest.open()))
        assert Counter(row['loss'] for row in rows) == {
            'ranknet': 24, 'hinge': 24, 'pairwise_mse': 24, 'listnet': 6,
        }
        for row in rows:
            assert row['delta_effective'] == ('False' if row['loss'] == 'listnet' else 'True')
            if row['loss'] == 'listnet':
                assert float(row['cross_delta']) == 0
                assert json.loads(row['model_hyper_params'])['loss_cross_delta'] == 0
                assert 'delta=0' in Path(row['output_dir']).parts
        assert list(csv.DictReader((group / 'completed_tasks.csv').open())) == []


def test_prepare_rejects_changed_config_before_replacing_any_audit(prepared_design):
    design = prepared_design
    manifests = sorted(design.out.glob("treino=*/experimento=*/design_composite_trading.csv"))
    before = {path: path.read_bytes() for path in manifests}
    altered = copy.deepcopy(design.config)
    altered["training"]["lr"] = 0.0002
    with pytest.raises(ValueError, match="Desenho/dados/código diferentes"):
        experiment.prepare(design.config_path, altered, design.tasks, design.out)
    assert all(path.read_bytes() == contents for path, contents in before.items())


def test_failed_training_does_not_record_success_or_export(prepared_design, monkeypatch):
    design = prepared_design
    exported = []
    monkeypatch.setattr(experiment_io, "export_predictions", lambda *args, **kwargs: exported.append(args))

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(experiment.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        experiment.worker(design.config, design.tasks, design.out, 0)
    assert not exported
    completed = design.out / experiment.group_path(design.tasks[0]) / "completed_tasks.csv"
    assert list(csv.DictReader(completed.open())) == []
    assert not list(design.out.rglob("part.parquet"))


def test_worker_rejects_mutated_prepared_data_before_training(prepared_design, monkeypatch):
    design = prepared_design
    path = Path(design.metadata["dataset"])
    path.write_text(path.read_text() + "\n")
    trained = []
    monkeypatch.setattr(experiment.subprocess, "run", lambda *args, **kwargs: trained.append(args))
    with pytest.raises(ValueError, match="dataset alterado"):
        experiment.worker(design.config, design.tasks, design.out, 0)
    assert not trained


def test_completed_task_is_replaced_by_global_id_without_duplicate_rows(prepared_design):
    design = prepared_design
    task = design.tasks[0]
    group = design.out / experiment.group_path(task)
    signature = json.loads((group / "run_config.json").read_text())["design_sha256"]
    row = experiment.audit_row(design.config, task, design.metadata, design.out, signature)
    row.update(slurm_job_id="123", parquet_sha256="abc")
    experiment.record_completed(group, row)
    row.update(slurm_job_id="456", parquet_sha256="def")
    experiment.record_completed(group, row)
    records = list(csv.DictReader((group / "completed_tasks.csv").open()))
    assert len(records) == 1
    assert records[0]["slurm_job_id"] == "456"
    assert records[0]["parquet_sha256"] == "def"


def test_dry_run_and_task_inspection_make_no_output(tmp_path):
    import os

    env = dict(os.environ, OUT_ROOT=str(tmp_path / "absent"), DATASET_PATH=str(tmp_path / "missing.csv"))
    for options in (["--dry-run"], ["--task", "33695"]):
        result = subprocess.run(["bash", str(REPO / "scripts/run_composite_trading_experiment.sh"), *options], env=env, check=True, text=True, capture_output=True)
        assert result.stdout.strip()
        if options == ['--dry-run']:
            for expected in ('33696 tarefas', 'ranknet=10368', 'hinge=10368',
                             'pairwise_mse=10368', 'listnet=2592', '34 array(s)',
                             'último: 0-695', 'uma execução por combinação'):
                assert expected in result.stdout
        assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize('source_kind', ['price', 'log_return'])
def test_actual_tfb_data_source_loads_prepared_returns_and_does_not_scan_sidecars(monkeypatch, tmp_path, source_kind):
    from ts_benchmark.data import data_source

    source = tmp_path / "b3_daily_tfb_new.csv"
    destination = tmp_path / "b3_log_returns_new.csv"
    prices = pd.DataFrame({
        "date": [1, 2, 3, 5, 8],
        "Z3": [100.0, 101.0, 103.0, 102.0, 105.0],
        "A4": [30.0, 31.0, 30.5, 32.0, 33.0],
    })
    prices.to_csv(source, index=False)
    metadata = experiment_io.prepare_dataset(source, destination, source_data_kind=source_kind)
    experiment.register_dataset(destination, metadata)
    meta_path = tmp_path / "FORECAST_META.csv"
    registered_bytes = meta_path.read_bytes()
    monkeypatch.setattr(data_source, "FORECASTING_DATASET_PATH", str(tmp_path))
    loader = data_source.LocalForecastingDataSource()
    loader.load_series_list([destination.name])
    loaded = loader.dataset.get_series(destination.name)
    expected = experiment_io.load_panel(destination)
    pd.testing.assert_frame_equal(loaded, expected, check_names=False)
    assert loaded.columns.tolist() == ["Z3", "A4"]
    assert pd.infer_freq(loaded.index) == "D"
    assert loader.dataset.get_series_meta_info(destination.name)["length"] == (4 if source_kind == 'price' else 5)
    assert pd.read_csv(meta_path).set_index('file_name').loc[source.name, 'length'] == 5
    assert meta_path.read_bytes() == registered_bytes
    assert {destination.name, source.name} == set(loader.dataset.metadata.index)


@pytest.mark.parametrize('launcher', experiment.LAUNCHERS.values())
def test_shell_source_kind_default_override_and_invalid_value(launcher):
    import os

    env = dict(os.environ, TFB_ROOT=str(REPO), PYTHON_BIN=sys.executable)
    env.pop('SOURCE_DATA_KIND', None)
    command = ['bash', str(REPO / 'scripts' / launcher), '--print-config']
    default = subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    assert json.loads(default.stdout)['source_data_kind'] == 'log_return'
    prices = subprocess.run(command, env=dict(env, SOURCE_DATA_KIND='price'), check=True, capture_output=True, text=True)
    assert json.loads(prices.stdout)['source_data_kind'] == 'price'
    invalid = subprocess.run(command, env=dict(env, SOURCE_DATA_KIND='auto'), capture_output=True, text=True)
    assert invalid.returncode != 0
    assert 'SOURCE_DATA_KIND' in invalid.stderr


def test_editable_shell_parameters_reach_python_without_json(tmp_path):
    import os

    script = REPO / 'scripts/run_composite_trading_experiment.sh'
    edited = tmp_path / script.name
    content = script.read_text()
    substitutions = {
        'TRAIN_PERCENTAGES=(70 80 90)': 'TRAIN_PERCENTAGES=(80)',
        'MODELS=(DUET Nonstationary_Transformer FEDformer TimesNet simpleFORMER simpleMLP)': 'MODELS=(simpleMLP)',
        'LOOKBACKS=(32 128 256)': 'LOOKBACKS=(64)',
        'HORIZONS=(1 5 10 15 20 25)': 'HORIZONS=(5)',
        'TRADE_WINDOWS=(1 5 10 15 20 25)': 'TRADE_WINDOWS=(5)',
        'CROSS_LOSSES=(ranknet hinge listnet pairwise_mse)': 'CROSS_LOSSES=(hinge)',
        'LAMBDAS=(0.1 0.5 0.8)': 'LAMBDAS=(0.8)',
        'DELTAS=(0 0.025 0.05 0.1)': 'DELTAS=(0.05)',
        'NUM_EPOCHS=100': 'NUM_EPOCHS=77',
        'PATIENCE=10': 'PATIENCE=9',
        'LEARNING_RATE=0.0001': 'LEARNING_RATE=0.0002',
        'CROSS_SCALE=1': 'CROSS_SCALE=2',
        'RANKNET_ALPHA=1': 'RANKNET_ALPHA=3',
        'LISTNET_TAU=1': 'LISTNET_TAU=2',
        'HINGE_MARGIN=0.1': 'HINGE_MARGIN=0.2',
    }
    for before, after in substitutions.items():
        assert before in content
        content = content.replace(before, after)
    edited.write_text(content)
    env = dict(os.environ, TFB_ROOT=str(REPO), PYTHON_BIN=sys.executable,
               OUT_ROOT=str(tmp_path / 'out'), DATASET_PATH=str(tmp_path / 'missing.csv'))
    result = subprocess.run(['bash', str(edited), '--task', '0'], env=env,
                            check=True, text=True, capture_output=True)
    task = json.loads(result.stdout)
    hyper = task['model_spec'][1]
    assert (task['treino'], task['modelo'], task['lookback'], task['pred_len'], task['k']) == (80, 'simpleMLP', 64, 5, 5)
    assert (task['cross_lambda'], task['cross_delta']) == (0.8, 0.05)
    assert (hyper['num_epochs'], hyper['patience'], hyper['lr']) == (77, 9, 0.0002)
    assert (hyper['loss_cross_scale'], hyper['loss_ranknet_alpha'], hyper['loss_listnet_tau'], hyper['loss_hinge_margin']) == (2, 3, 2, 0.2)
    preview = subprocess.run(['bash', str(edited), '--dry-run'], env=env,
                             check=True, text=True, capture_output=True)
    assert '1 tarefas' in preview.stdout
    assert not (tmp_path / 'out').exists()
    original = subprocess.run(['bash', str(script), '--dry-run'], env=env,
                              check=True, text=True, capture_output=True)
    assert '33696 tarefas' in original.stdout


def test_cluster_node_cpu_and_memory_fields_reach_sbatch(monkeypatch, tmp_path):
    monkeypatch.setattr(experiment.shutil, 'which', lambda name: '/fake/sbatch' if name == 'sbatch' else None)
    monkeypatch.setenv('LOG_ROOT', str(tmp_path / 'logs'))
    monkeypatch.setenv('GPU_PARTITION', 'medusas_dev')
    monkeypatch.setenv('GPU_TIME', '08:00:00')
    monkeypatch.setenv('NODELIST', 'medusa3')
    monkeypatch.setenv('CPUS_PER_TASK', '4')
    monkeypatch.setenv('MEMORY', '16G')
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout='12345')
    monkeypatch.setattr(experiment.subprocess, 'run', fake_run)
    experiment.submit('temporal', [None], tmp_path / 'out', REPO / 'scripts/run_temporal_experiment.sh')
    assert '--nodelist=medusa3' in calls[0]
    assert '--cpus-per-task=4' in calls[0]
    assert '--mem=16G' in calls[0]
