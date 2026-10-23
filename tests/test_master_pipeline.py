"""Real MASTER -> TFB rolling forecast -> B3 Parquet export on audited fixtures."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import b3_experiment as experiment
from b3_experiment_io import load_panel
from test_master_data import ExportFixture


def test_master_real_pipeline_exports_all_horizons_and_audit(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("pyarrow")
    # These are imports of the upstream TFB command, even with sequential CPU
    # execution and CSV reporting. Keep the test usable in data-only installs.
    pytest.importorskip("ray")
    pytest.importorskip("dash")
    fixture = ExportFixture(tmp_path / "input")
    checkout = tmp_path / "checkout"
    # Isolate TFB's fixed dataset path without substituting its model, loss,
    # benchmark command, rolling strategy, serialization or exporter.
    for directory in ("scripts", "ts_benchmark", "config"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".sh", ".json"}:
                destination = checkout / path.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
    config = {
        "source_data_kind": "log_return", "models": ["MASTER"], "var_lags": 32,
        "train_percentages": [80], "lookbacks": [4], "horizons": [5],
        "trade_windows": [5], "cross_losses": ["ranknet_proximity"],
        "lambdas": [0.5], "deltas": [0],
        "model_defaults": {"MASTER": {
            "master_bundle_dir": str(fixture.directory),
            "master_returns_path": str(fixture.returns),
            "master_expected_assets": 3, "master_expected_dates": 48,
            "asset_blocks": ["retornos", "ohlcv", "atividade"], "use_market": True,
            "batch_size": 4, "d_model": 8, "t_heads": 2, "s_heads": 2,
            "dropout": 0, "gate_beta": 1, "norm": True, "lradj": "constant",
            "device": "cpu", "seed": 2026,
        }},
        "training": {"num_epochs": 1, "patience": 1, "lr": 0.001},
        "loss": {
            "loss_temporal": "mse", "loss_data_kind": "log_return",
            "loss_score_kind": "simple_return", "loss_cross_score_normalization": "zscore",
            "loss_cross_mask": ["gaussian", 0.025, "natural"],
            "loss_inverse_norm": True, "loss_track_components": True,
        },
        "strategy": {"train_ratio_in_tv": 0.875, "stride": 1,
                     "num_rollings": 48000, "seed": 2026},
    }
    monkeypatch.setattr(experiment, "ROOT", checkout)
    # Repository provenance is the only mocked external operation; the copied
    # checkout deliberately contains no .git directory.
    monkeypatch.setattr(experiment.subprocess, "check_output", lambda *args, **kwargs: "fixture-commit\n")
    monkeypatch.setenv("DATASET_PATH", str(fixture.returns))
    monkeypatch.setenv("RESULT_ROOT", str(tmp_path / "result"))
    monkeypatch.setenv("MANTER_CSV", "FALSE")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    tasks = experiment.make_tasks(config, "composto")
    output_root = tmp_path / "predictions"
    metadata = experiment.prepare(checkout / "scripts/run_master_experiment.sh", config, tasks, output_root)
    experiment.worker(config, tasks, output_root, 0)
    output = output_root / experiment.output_path(tasks[0])
    frame = pd.read_parquet(output / "part.parquet")
    cut, horizon = 48 * 80 // 100, 5
    count = 48 - cut - horizon + 1
    assert frame.columns.tolist() == ["arquivo", *fixture.assets, "h", "origin_step", "step"]
    assert frame.shape == (count * horizon, 7)
    assert frame["arquivo"].nunique() == count
    assert frame["h"].tolist() == list(range(1, horizon + 1)) * count
    assert frame["origin_step"].min() == cut
    assert frame["step"].max() == fixture.n
    assert (frame["step"] - frame["origin_step"] == frame["h"]).all()
    assert np.isfinite(frame[list(fixture.assets)].to_numpy()).all()
    audit = json.loads((output / "master_training.json").read_text())
    assert audit["status"] == "TRAINED"
    assert audit["target_kind"] == "economic_log_return"
    assert audit["config"]["pred_len"] == horizon
    assert audit["config"]["loss_trade_window"] == 5
    assert audit["split"]["n_pretest"] == cut
    assert audit["target_normalization"]["n_samples_seen"] == int(cut * 0.875)
    assert len(audit["history"]) == 1
    assert (output / "master_checkpoint.pt").is_file()
    # Reload the exact serialized model and verify exported ticker order and
    # the first rolling origin against an independently requested forecast.
    from ts_benchmark.baselines.master import MASTER
    model = MASTER.load_checkpoint(output / "master_checkpoint.pt", device="cpu")
    panel = load_panel(Path(metadata["dataset"]))
    first = model.forecast(horizon, panel.iloc[:cut])
    np.testing.assert_allclose(frame.loc[:horizon - 1, list(fixture.assets)], first,
                               rtol=1e-6, atol=1e-8)
    completed_path = output_root / experiment.group_path(tasks[0]) / "completed_tasks.csv"
    with completed_path.open() as stream:
        completed = list(csv.DictReader(stream))
    assert len(completed) == 1
    assert completed[0]["parquet_sha256"] == experiment.file_hash(output / "part.parquet")
