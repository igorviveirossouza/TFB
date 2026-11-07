"""Final score ordering, raw-return eligibility, immutable CPU-side contracts."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"scripts"))
import b3_experiment as producer
import b3_experiment_io as producer_io
from backtest_composto.io import (discover_composite_tasks, load_composite_predictions,
                                  load_composite_ranking_scores, momentum_snapshot)
from backtest_cpu.core import Market, Strategy, backtest, ic_frame
from backtest_cpu.io import digest, read_json
from test_backtest_cpu import market_frame
from test_backtest_composto_io import make_composite_fixture, publish_composite_task, rewrite_registry
from test_backtest_composto_pipeline import cli, prepare_args, read_parquet


def momentum_fixture(tmp_path, *, gain_init=0.0, diagnostics=False, lradj=None):
    f = make_composite_fixture(tmp_path, losses=("ranknet",), deltas=(0,), available_indices=())
    f.config["loss"].update(momentum_mode="residual_output", momentum_lookback=3, momentum_eta=.1)
    if gain_init != 0.0:
        f.config["loss"]["momentum_gain_init"] = gain_init
    if diagnostics:
        f.config["loss"]["momentum_diagnostics"] = True
    if lradj is not None:
        f.config["training"]["lradj"] = lradj
    f.producer_tasks = producer.make_tasks(f.config, "composto")
    f.design_sha256 = producer.digest(producer.provenance(f.config, f.metadata))
    group = f.root/"treino=50"/"experimento=composto"
    (group/"run_config.json").write_text(json.dumps({"design_sha256": f.design_sha256,
        "config": f.config, "metadata": f.metadata}))
    index = next(i for i, task in enumerate(f.producer_tasks) if task["pred_len"] == 3)
    row = publish_composite_task(f, index, register=False)
    output = Path(row["output_dir"])
    state = {"version": "momentum_residual_v1", "mode": "residual_output", "lookback": 3,
             "eta": .1, "gain": 0., "score_kind": "simple_return", "eps": 1e-8,
             "normalization": "cross_sectional_zscore_ddof0", "asset_names": f.market.tickers.tolist()}
    if gain_init != 0.0:
        state["gain_init"] = gain_init
    (output/"momentum_state.json").write_text(json.dumps(state))
    panel = producer_io.load_panel(Path(f.metadata["dataset"]))
    producer_io.export_momentum_scores(panel, len(panel)//2, 3, output)
    producer.record_completed(group, row)
    return f, output


@pytest.mark.parametrize("gain_init,lradj", [(0.0, "type1"), (.1, "type1"), (.1, "constant")])
def test_diagnostic_conditions_keep_producer_reader_paths_and_hyperparameters(tmp_path, gain_init, lradj):
    f, output = momentum_fixture(tmp_path, gain_init=gain_init, diagnostics=True, lradj=lradj)
    tasks, _ = discover_composite_tasks(f.root, [50], f.market, f.identity["retornos_sha256"])
    task = tasks[0]
    original = f.producer_tasks[task["training_array_index"]]
    assert task["relpath"] == str(producer.output_path(original))
    assert Path(task["path"]).parent == output
    assert task["momentum_diagnostics"] is True
    assert task.get("momentum_gain_init", 0.0) == gain_init
    assert task["momentum_state"].get("gain_init", 0.0) == gain_init
    # Learned gain remains distinct from its initialization.
    assert task["momentum_state"]["gain"] == 0.0


@pytest.mark.parametrize("state_gain_init", [None, .2])
def test_state_initialization_must_match_design_even_with_valid_sidecar_hashes(tmp_path, state_gain_init):
    f, output = momentum_fixture(tmp_path, gain_init=.1, diagnostics=True)
    state = read_json(output/"momentum_state.json")
    if state_gain_init is None:
        state.pop("gain_init")
    else:
        state["gain_init"] = state_gain_init
    (output/"momentum_state.json").write_text(json.dumps(state))
    panel = producer_io.load_panel(Path(f.metadata["dataset"]))
    producer_io.export_momentum_scores(panel, len(panel)//2, 3, output)
    with pytest.raises(ValueError, match="Estado momentum incompativel"):
        discover_composite_tasks(f.root, [50], f.market, f.identity["retornos_sha256"])


def test_discovery_binds_resolved_momentum_and_exact_export(tmp_path):
    f, output = momentum_fixture(tmp_path)
    tasks, _ = discover_composite_tasks(f.root, [50], f.market, f.identity["retornos_sha256"])
    task = tasks[0]
    assert "momentum=residual_output/momentum_lb=3/momentum_eta=0.1/modelo=" in task["relpath"]
    assert task["momentum_state"]["gain"] == 0
    assert set(task["momentum_files"]) == {"manifest", "predictions", "state", "scores"}
    origins, _, raw = load_composite_predictions(task, f.market)
    scores = load_composite_ranking_scores(task, f.market, origins)
    history = np.stack([f.market.arrays["log_retorno_corrigido"][o-3:o].sum(axis=0) for o in origins])
    expected = (history-history.mean(axis=1, keepdims=True))/history.std(axis=1, keepdims=True)
    np.testing.assert_allclose(scores, np.repeat(expected[:, None], 3, axis=1), atol=1e-12)
    assert not np.allclose(scores, raw)
    # Any post-plan update, even a harmless JSON formatting change, invalidates reuse.
    manifest = output/"momentum_scores.json"
    manifest.write_text(manifest.read_text()+"\n")
    with pytest.raises(ValueError, match="apos o planejamento"):
        momentum_snapshot(task, f.market, verify_files=True)


def test_signed_eligibility_uses_return_but_selection_uses_final_score():
    frame, calendar = market_frame()
    market = Market.from_frame(frame, calendar)
    scores = np.array([[9., -2., -3., 8.]])
    raw = np.array([[-.1, .2, .1, -.2]])
    origins = np.array([3])
    buy = backtest(scores, origins, 1, market, Strategy("buy", "compra", 1, True), eligibility_signal=raw)
    sell = backtest(scores, origins, 1, market, Strategy("sell", "venda", 1, True), eligibility_signal=raw)
    assert buy["posicoes"].ticker.tolist() == ["AAA1"]  # negative score, positive return
    assert sell["posicoes"].ticker.tolist() == ["AAA3"]  # positive score, negative return
    assert buy["posicoes"].score_log_acumulado.iloc[0] == .2
    assert buy["posicoes"].ranking_score.iloc[0] == -2.
    original = backtest(raw, origins, 1, market, Strategy("buy", "compra", 1, True))
    explicit_off = backtest(raw, origins, 1, market, Strategy("buy", "compra", 1, True), eligibility_signal=None)
    for name in original:
        pd.testing.assert_frame_equal(original[name], explicit_off[name])
    with pytest.raises(ValueError, match="elegibilidade"):
        backtest(scores, origins, 1, market, Strategy("buy", "compra", 1, True),
                 eligibility_signal=np.array([[np.nan]*4]))


def test_unsorted_producer_assets_align_to_market_universe(tmp_path):
    f, output = momentum_fixture(tmp_path)
    assets = f.market.tickers.tolist()[::-1]
    raw = read_parquet(output/"part.parquet")
    raw[["arquivo", *assets, "h", "origin_step", "step"]].to_parquet(output/"part.parquet", index=False)
    state = read_json(output/"momentum_state.json")
    state["asset_names"] = assets
    (output/"momentum_state.json").write_text(json.dumps(state))
    panel = producer_io.load_panel(Path(f.metadata["dataset"]))[assets]
    producer_io.export_momentum_scores(panel, len(panel)//2, 3, output)
    rewrite_registry(f, lambda rows: rows.assign(parquet_sha256=digest(output/"part.parquet")))
    tasks, _ = discover_composite_tasks(f.root, [50], f.market, f.identity["retornos_sha256"])
    origins, _, _ = load_composite_predictions(tasks[0], f.market)
    scores = load_composite_ranking_scores(tasks[0], f.market, origins)
    history = np.stack([f.market.arrays["log_retorno_corrigido"][o-3:o].sum(axis=0) for o in origins])
    expected = (history-history.mean(axis=1, keepdims=True))/history.std(axis=1, keepdims=True)
    np.testing.assert_allclose(scores[:, 0], expected, atol=1e-12)


@pytest.mark.parametrize("gain_init,diagnostics,lradj", [(0.0, False, None), (.1, True, "constant")])
def test_frozen_cpu_pipeline_uses_final_ic_and_refuses_changed_sidecar(tmp_path, gain_init, diagnostics, lradj):
    f, output = momentum_fixture(tmp_path, gain_init=gain_init, diagnostics=diagnostics, lradj=lradj)
    run = tmp_path/"run"
    cli(*prepare_args(f, run))
    plan = read_json(run/"plano.json")
    task = plan["tasks"][0]
    cli("run", "--run-dir", run, "--workers", "1", cwd=tmp_path)
    saved = run/"tasks"/task["relpath"]
    actual = read_parquet(saved/"ic_acumulado.parquet")
    origins, _, _ = load_composite_predictions(task, f.market)
    scores = load_composite_ranking_scores(task, f.market, origins)
    for k in (1, 3):
        expected = ic_frame(scores[:, k-1], origins, k, f.market, task["common_last_origin"])
        pd.testing.assert_frame_equal(actual.loc[actual.k == k, expected.columns].reset_index(drop=True), expected)
    assert actual.momentum_mode.unique().tolist() == ["residual_output"]
    assert actual.eligibility_signal_kind.unique().tolist() == ["raw_cumulative_log_return"]
    assert read_json(saved/"conclusao.json")["status"] == "COMPLETA"
    # The frozen CPU snapshot has no package dependency on training/model code.
    for source in (run/"codigo").rglob("*.py"):
        assert "from ts_benchmark" not in source.read_text()
        assert "import torch" not in source.read_text()
    path = output/"ranking_scores.parquet"
    frame = read_parquet(path)
    frame[f.market.tickers[0]] += .01
    frame.to_parquet(path, index=False)
    cli("run", "--run-dir", run, "--workers", "1", check=False)
    state = read_json(run/"status/000000.json")
    assert state["status"] == "ERRO" and "momentum alterado" in state["erro"]


def test_missing_active_sidecar_never_falls_back_to_forecasts(tmp_path):
    f, output = momentum_fixture(tmp_path)
    (output/"ranking_scores.parquet").unlink()
    with pytest.raises(ValueError, match="momentum ausente"):
        discover_composite_tasks(f.root, [50], f.market, f.identity["retornos_sha256"])
