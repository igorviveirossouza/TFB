"""Dynamic model identity, worker routing, provenance and preflight ordering."""
from pathlib import Path
import json
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from scripts import b3_experiment as b3
from scripts import ordinal_experiment as ordinal
from scripts.ordinal_model_registry import resolve_model


def configuration(*requested):
    # Start from the existing producer's full scientific defaults, then select
    # arbitrary resolved models independently of the launcher's CLI parser.
    branch = ordinal.build_branches(ordinal.settings({
        "LOOKBACKS_VALUES": "32", "HORIZON_VALUES": "1 25",
        "LAMBDAS_VALUES": "0 .5 1", "CROSS_LOSSES_VALUES": "ordinal_ce",
    }))[1]
    config = branch["config"]
    names, specs = [], {}
    for requested_name in requested:
        name, spec = resolve_model(requested_name)
        names.append(name)
        specs[name] = spec
    config["models"] = names
    config["model_specs"] = specs
    config["model_defaults"] = {name: {"norm": True, "batch_size": 16,
        "parallel_strategy": None} for name in names}
    return config


@pytest.mark.parametrize("requested,qualified,adapter", [
    ("PatchTST", "time_series_library.PatchTST", "transformer_adapter"),
    ("FITS", "fits.FITS", None),
    ("Amplifier", "amplifier.Amplifier", None),
    ("TimeBase", "timebase.TimeBase", None),
    ("ts_benchmark.baselines.time_series_library.models.TimesNet.TimesNet",
     "time_series_library.TimesNet", "transformer_adapter"),
])
def test_resolved_models_and_controls_use_same_ordinal_worker(requested, qualified, adapter):
    config = configuration(requested)
    for mode in ("temporal", "composto"):
        for task in b3.make_tasks(config, mode):
            model, hyper, worker_adapter, deterministic = b3.model_spec(config, task)
            assert model == "ordinal_model_adapter.OrdinalModel"
            assert worker_adapter is None
            assert hyper["_ordinal_model_name"] == qualified
            assert hyper["_ordinal_model_adapter"] == adapter
            assert hyper["_ordinal_model_label"] == task["modelo"]
            assert hyper["seq_len"] == task["lookback"]
            assert hyper["pred_len"] == task["pred_len"]
            assert hyper["loss_cross_lambda"] == task["cross_lambda"]
            assert hyper["loss"] == ("mse" if mode == "temporal" else "composite_trading")
            assert deterministic == ("efficient" if task["modelo"] == "TimesNet" else "full")


def test_worker_command_freezes_resolved_target_and_keeps_architecture_overrides(tmp_path):
    config = configuration("PatchTST")
    config["model_defaults"]["PatchTST"].update(d_model=48, n_heads=4, patch_len=8)
    task = b3.make_tasks(config, "composto")[0]
    command = b3.benchmark_command(config, task, tmp_path / "returns.csv", 3664,
                                   tmp_path / "result")
    assert command[command.index("--model-name") + 1] == "ordinal_model_adapter.OrdinalModel"
    assert "--adapter" not in command
    hyper = json.loads(command[command.index("--model-hyper-params") + 1])
    assert hyper["_ordinal_model_name"] == "time_series_library.PatchTST"
    assert hyper["_ordinal_model_adapter"] == "transformer_adapter"
    assert (hyper["d_model"], hyper["n_heads"], hyper["patch_len"]) == (48, 4, 8)
    assert hyper["loss_cross"] == "ordinal_ce"
    assert hyper["loss_bucket_count"] == 3


def test_legacy_experiments_without_resolution_keep_native_routing():
    config = configuration("Nonstationary_Transformer", "DUET")
    config.pop("model_specs")
    for task in b3.make_tasks(config, "composto"):
        model, hyper, adapter, _ = b3.model_spec(config, task)
        if task["modelo"] == "DUET":
            assert model == "duet.DUET" and adapter is None
        else:
            assert model == "time_series_library.Nonstationary_Transformer"
            assert adapter == "transformer_adapter"
        assert not any(key.startswith("_ordinal_model_") for key in hyper)


def test_provenance_covers_selected_family_defaults_dependencies_and_resolver():
    config = configuration("FITS")
    metadata = {"source_sha256": "source", "dataset_sha256": "dataset",
                "source_data_kind": "log_return", "step_offset": 1}
    frozen = b3.provenance(config, metadata)
    hashes = frozen["code_sha256"]
    for file in ("scripts/ordinal_model_registry.py",
                 "ts_benchmark/baselines/ordinal_model_adapter.py",
                 "ts_benchmark/models/model_loader.py",
                 "ts_benchmark/baselines/fits/fits.py",
                 "ts_benchmark/baselines/fits/fits_model.py",
                 "ts_benchmark/baselines/utils.py"):
        assert hashes[file] == b3.file_hash(ROOT / file)
    assert frozen["config"]["model_specs"]["FITS"]["family"] == "fits"


def test_incompatible_model_preflight_stops_before_dataset_mutation(monkeypatch, tmp_path):
    config = configuration("FITS")
    calls = []
    module = types.ModuleType("ts_benchmark.baselines.ordinal_model_adapter")
    def reject(**hyper):
        calls.append(("validate", hyper["_ordinal_model_label"]))
        raise ValueError("ordinal training capability absent")
    module.validate_model = reject
    monkeypatch.setitem(sys.modules, module.__name__, module)
    import b3_experiment_io as data_io
    def mutate(*args, **kwargs):
        pytest.fail("dataset preparation must not run before model validation")
    monkeypatch.setattr(data_io, "prepare_dataset", mutate)
    monkeypatch.setattr(b3, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="capability absent"):
        b3.prepare(tmp_path / "config.json", config, b3.make_tasks(config, "composto"),
                   tmp_path / "predictions")
    assert calls == [("validate", "FITS")]
    assert not (tmp_path / "dataset").exists()
    assert not (tmp_path / "predictions").exists()


def test_preflight_checks_each_selected_model_once_before_data(monkeypatch, tmp_path):
    config = configuration("PatchTST", "FITS")
    tasks = b3.make_tasks(config, "composto")
    assert len(tasks) > len(config["models"])
    calls = []
    module = types.ModuleType("ts_benchmark.baselines.ordinal_model_adapter")
    module.validate_model = lambda **hyper: calls.append(hyper["_ordinal_model_label"])
    monkeypatch.setitem(sys.modules, module.__name__, module)
    import b3_experiment_io as data_io
    def data_boundary(*args, **kwargs):
        assert calls == ["PatchTST", "FITS"]
        raise RuntimeError("verified before data boundary")
    monkeypatch.setattr(data_io, "prepare_dataset", data_boundary)
    monkeypatch.setattr(b3, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="before data boundary"):
        b3.prepare(tmp_path / "config.json", config, tasks, tmp_path / "predictions")
    assert calls == ["PatchTST", "FITS"]
