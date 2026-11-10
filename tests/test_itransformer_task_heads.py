"""CPU checks of the actual iTransformer head objectives and B3 exports."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import b3_experiment as b3
from scripts import itransformer_task_heads as worker
from ts_benchmark.baselines.itransformer_task_heads import (
    ITransformerRestoredTaskAdapter, ITransformerRestoredTaskNetwork,
)
from ts_benchmark.baselines.time_series_library.adapters_for_transformers import TransformerAdapter
from ts_benchmark.baselines.time_series_library.models.iTransformer import iTransformer


@pytest.fixture(autouse=True)
def single_thread(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")


def data():
    return pd.DataFrame(np.random.default_rng(491).normal(0, .02, (120, 5)),
                        columns=list("ABCDE"),
                        index=pd.date_range("2000-01-02", periods=120, name="date"))


def options(lam=.5, **kw):
    result = dict(seq_len=8, horizon=2, norm=True, batch_size=8, num_epochs=1,
                  patience=1, lr=.001, lradj="constant", d_model=8, d_ff=16,
                  e_layers=1, d_layers=1, n_heads=2, dropout=0.,
                  loss_temporal="mse", loss_cross="rankmix_proximity", loss_cross_lambda=lam,
                  loss_trade_window=1, loss_data_kind="log_return", loss_inverse_norm=True,
                  loss_score_kind="simple_return", loss_cross_mask=["cauchy", .025, "natural"],
                  loss_rank_mix_weights=[.5, .5, 0], loss_rank_mix_mode="global",
                  loss_rank_mix_regions=["top", "bottom", "top"], loss_rank_mix_fraction=1/3,
                  validation_mode="ordered_sample_mean")
    result.update(kw)
    return result


def adapter(mode="separate", lam=.5, **kw):
    return ITransformerRestoredTaskAdapter(task_head_mode=mode, rank_windows=[1, 2, 5, 25],
                                           **options(lam, **kw))


def test_matched_initialization_restoration_and_native_forecast():
    cfg = adapter().config
    cfg.enc_in = cfg.dec_in = cfg.c_out = 5
    cfg.label_len, cfg.pred_len, cfg.freq = 4, 2, "d"
    torch.manual_seed(91)
    native = iTransformer(cfg)
    rng = torch.get_rng_state()
    torch.manual_seed(91)
    dual = ITransformerRestoredTaskNetwork(cfg)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in native.state_dict().items():
        torch.testing.assert_close(value, dual.state_dict()[name], rtol=0, atol=0)
    assert dual.ranking_projection.weight.data_ptr() != dual.projection.weight.data_ptr()
    torch.testing.assert_close(dual.ranking_projection.weight, dual.projection.weight, rtol=0, atol=0)
    native.eval(); dual.eval()
    x = torch.randn(2, 8, 5) * torch.tensor([.1, .3, 1., 2., 4.]) + torch.arange(5.)
    marks = torch.zeros(2, 8, 3); ym = torch.zeros(2, 6, 3); dec = torch.randn(2, 6, 5)
    with torch.no_grad():
        a = native(x, marks, dec, ym); b = dual(x, marks, dec, ym)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(b, dual.ranking_returns, rtol=0, atol=0)
    mean = x.mean(1, keepdim=True)
    std = torch.sqrt((x - mean).var(1, keepdim=True, unbiased=False) + 1e-5)
    expected = dual.ranking_projection(dual._encoder_hidden).permute(0, 2, 1)[:, :, :5]
    torch.testing.assert_close(expected * std + mean, dual.ranking_returns, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["log_return", "simple_return"])
@pytest.mark.parametrize("lam", [.5, 1.])
@pytest.mark.parametrize("mix", ["global", "regional"])
def test_fit_gradient_routing_inverse_scaler_windows_and_future_isolation(kind, lam, mix):
    torch.manual_seed(2026)
    m = adapter(lam=lam, loss_score_kind=kind, loss_rank_mix_mode=mix)
    assert m.rank_windows == (1, 2)
    m.forecast_fit(data(), train_ratio_in_tv=.75)
    assert m.model.ranking_projection.weight.grad.abs().sum() > 0
    assert m.model.encoder.attn_layers[0].attention.query_projection.weight.grad.abs().sum() > 0
    temporal_grad = m.model.projection.weight.grad
    assert temporal_grad is None if lam == 1 else temporal_grad.abs().sum() > 0
    m.model.eval()
    x = torch.randn(2, 8, 5); y = torch.randn(2, 6, 5)
    marks = torch.zeros(2, 8, 3); ym = torch.zeros(2, 6, 3)
    with torch.no_grad():
        pred = m._process(x, y, marks, ym)["output"].clone()
        score = m.latest_rank_scores.clone()
        raw = m.model.ranking_returns * torch.tensor(m.scaler.scale_, dtype=pred.dtype)
        raw = raw + torch.tensor(m.scaler.mean_, dtype=pred.dtype)
        expected = raw.cumsum(1)
        if kind == "simple_return": expected = torch.expm1(expected)
        torch.testing.assert_close(score, expected, rtol=1e-6, atol=1e-8)
        y[:, -2:] = 10000
        torch.testing.assert_close(pred, m._process(x, y, marks, ym)["output"], rtol=0, atol=0)
        torch.testing.assert_close(score, m.latest_rank_scores, rtol=0, atol=0)
    assert m.forecast(2, data()).shape == (2, 5)
    assert m.export_rank_scores.shape == (2, 5)


def test_shared_zero_is_native_mse_and_pure_rank_endpoint_matches():
    predictions = []
    for kind in ["native", "shared"]:
        torch.manual_seed(17)
        if kind == "native":
            cfg = options(0, loss="mse", parallel_strategy=None)
            m = TransformerAdapter("iTransformer", iTransformer, **cfg)
        else:
            m = adapter(mode="shared", lam=0)
        m.forecast_fit(data(), train_ratio_in_tv=.75)
        predictions.append(m.forecast(2, data()))
    np.testing.assert_array_equal(*predictions)
    predictions = []
    for mode in ["shared", "separate"]:
        torch.manual_seed(2026)
        m = adapter(mode=mode, lam=1, loss_score_kind="log_return", dropout=.1)
        m.forecast_fit(data(), train_ratio_in_tv=.75)
        pred = m.forecast(2, data())
        predictions.append(pred.cumsum(0) if mode == "shared" else m.export_rank_scores)
    np.testing.assert_allclose(*predictions, rtol=2e-5, atol=2e-7)


def config(mode="separate", h=2):
    hyper = options()
    loss = {k: v for k, v in hyper.items() if k.startswith("loss_")}
    architecture = {k: hyper[k] for k in ["norm", "batch_size", "d_model", "d_ff", "e_layers", "d_layers", "n_heads", "dropout"]}
    architecture.update(task_head_mode=mode, rank_windows=[1, 2, 5, 25])
    return dict(source_data_kind="log_return", models=["iTransformer"], lookbacks=[8],
                horizons=[h], train_percentages=[80], trade_windows=[1],
                cross_losses=["rankmix_proximity"], lambdas=[.5, 1.], deltas=[0],
                model_defaults={"iTransformer": architecture},
                training={k: hyper[k] for k in ["num_epochs", "patience", "lr", "lradj", "validation_mode"]},
                strategy=dict(seed=2026, stride=1, train_ratio_in_tv=.75, num_rollings=48000), loss=loss)


@pytest.mark.parametrize("mode,lam", [("shared", 0), ("shared", 1.), ("separate", .5), ("separate", 1.)])
@pytest.mark.parametrize("h", [1, 2])
def test_b3_real_exports_completion_and_checkpoint(tmp_path, mode, lam, h):
    cfg = config(mode, h)
    worker.validate_config(cfg)
    task = b3.make_tasks(cfg, "temporal" if lam == 0 else "composto")[0]
    task["cross_lambda"] = lam
    qualified, hyper, model_adapter, _ = b3.model_spec(cfg, task)
    assert qualified == "itransformer_task_heads.ITransformerRestoredTaskAdapter" and model_adapter is None
    assert hyper["rank_windows"] == ([1] if h == 1 else [1, 2])
    panel = data(); out = tmp_path / "fit"
    result = worker.fit_and_export(cfg, task, hyper, panel, 96, out)
    contract = worker.settings(cfg, task)
    assert all(result[k] == v for k, v in contract.items())
    assert (out / "part.parquet").exists() == (mode != "separate" or lam < 1)
    assert (out / "ranking_scores.parquet").exists() == (mode == "separate" and lam > 0)
    if result["ranking_scores_valid"]:
        scores = pd.read_parquet(out / "ranking_scores.parquet")
        assert set(scores.k_eval) == set(hyper["rank_windows"])
        assert scores.origin_step.min() == 96 and scores.origin_step.max() == len(panel) - h
        assert len(scores) == (len(panel) - 96 - h + 1) * len(hyper["rank_windows"])
        assert not scores.duplicated(["origin_step", "k_eval"]).any()
    if result["return_forecast_valid"]:
        forecasts = pd.read_parquet(out / "part.parquet")
        assert (forecasts.step == forecasts.origin_step + forecasts.h).all()
    primary_hash = result["files_sha256"][result["primary_file"]]
    b3.record_execution_metadata({str(h): out}, primary_hash, "test-signature", extra=result)
    assert worker.validate_completed(out, contract, "test-signature", primary_hash)
    checkpoint = torch.load(out / "task_head_checkpoint.pt", weights_only=False)
    assert ("ranking_projection.weight" in checkpoint["state_dict"]) == (mode == "separate")
    # The persisted pipeline must detect changed scores, including lambda=1
    # fits that intentionally have no temporal part.parquet.
    with (out / result["primary_file"]).open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="artifact"):
        worker.validate_completed(out, contract, "test-signature", primary_hash)


def test_contract_rejects_silent_ignored_parameters():
    cfg = config()
    cfg["trade_windows"] = [5]
    with pytest.raises(ValueError, match="K_LOSS_VALUES"):
        worker.validate_config(cfg)
    with pytest.raises(ValueError, match="lambda"):
        adapter(mode="separate", lam=0)
    with pytest.raises(ValueError, match="norm=True"):
        adapter(norm=False)
    with pytest.raises(ValueError, match="unique positive"):
        ITransformerRestoredTaskAdapter(task_head_mode="shared", rank_windows=[1, 1], **options())
    bad = config()
    bad["model_defaults"]["iTransformer"]["rank_windows"] = [5, 25]
    with pytest.raises(ValueError, match="Each horizon"):
        worker.validate_config(bad)


def test_frozen_b3_worker_end_to_end_and_idempotence(tmp_path, monkeypatch):
    """Real run_config -> worker -> Parquets -> checked completion metadata."""
    from b3_experiment_io import prepare_dataset
    root = Path(os.environ.get("ITRANSFORMER_HEAD_QA_DIR", str(tmp_path)))
    root.mkdir(parents=True, exist_ok=True)
    source = root / "returns.csv"
    raw = data().copy()
    raw.index = np.arange(1, len(raw) + 1)
    raw.index.name = "date"
    long = raw.reset_index().melt(id_vars="date", var_name="cols", value_name="data")
    long[["date", "data", "cols"]].to_csv(source, index=False)
    metadata = prepare_dataset(source, root / "dataset.csv", source_data_kind="log_return")
    predictions = root / "predictions"
    total = 0
    for head, experiment in [("shared", "temporal"), ("shared", "composto"), ("separate", "composto")]:
        cfg = config(head)
        cfg["horizons"] = [1, 2]
        worker.validate_config(cfg)
        tasks = b3.make_tasks(cfg, experiment)
        out = predictions / ("head=" + head) / "seed=2026" / "temporal_loss=mse" / experiment
        signature = b3.digest(b3.provenance(cfg, metadata))
        for group in sorted({b3.group_path(t) for t in tasks}):
            folder = out / group
            folder.mkdir(parents=True, exist_ok=True)
            plan = dict(config=cfg, config_path=str(ROOT / "scripts/run_itransformer_rankmix_piloto.sh"),
                        metadata=metadata, design_sha256=signature)
            (folder / "run_config.json").write_text(json.dumps(plan, indent=2))
            row = b3.audit_row(cfg, tasks[0], metadata, out, signature)
            row.update(slurm_job_id="", parquet_sha256="")
            (folder / "completed_tasks.csv").write_text(b3.csv_text([row]).splitlines()[0] + "\n")
        for index in range(len(tasks)):
            b3.worker(cfg, tasks, out, index)
            total += 1
        # All files, including ranking-only endpoints, validate without refit.
        with monkeypatch.context() as context:
            def fail_refit(*args, **kwargs):
                raise AssertionError("A completed, hash-checked fit must not train again")
            context.setattr(worker, "fit_and_export", fail_refit)
            for index in range(len(tasks)):
                b3.worker(cfg, tasks, out, index)
    assert total == 10
    assert len(list(predictions.rglob("execution_metadata.json"))) == 10
    assert len(list(predictions.rglob("part.parquet"))) == 8
    assert len(list(predictions.rglob("ranking_scores.parquet"))) == 4
