"""Matched projection, restoration, gradient routing and frozen export tests."""
import json

import numpy as np
import pytest
import torch

from test_ns_capacity_heads import adapter, clean_env, create_fixture, data, runner
from ts_benchmark.baselines.ns_restored_task_heads import NSRestoredTaskAdapter, RestoredTaskNetwork
from ts_benchmark.baselines.time_series_library.models.Nonstationary_Transformer import Nonstationary_Transformer


def restored(mode="separate", lam=.5, loss="rankmix_proximity", **kw):
    options = vars(adapter(mode=mode, lam=lam, loss=loss, **kw).config).copy()
    return NSRestoredTaskAdapter(head_mode=mode, rank_windows=[1, 2], **options)


def test_distinct_contract_default_counts_and_overrides(monkeypatch):
    cfg = runner.configuration("heads_restored")
    grid = runner.tasks(cfg)
    assert len(grid) == 54 and cfg["schema_version"] == 2
    assert len([t for t in grid if t["lambda_value"] == 0]) == 6
    assert all(t["ranking_construction"] == "restored_daily_projection_v1" for t in grid)
    old = runner.tasks(runner.configuration("heads"))
    assert not set(t["fit_id"] for t in old) & set(t["fit_id"] for t in grid)
    monkeypatch.setenv("TEMPORAL_LOSSES_VALUES", "mse huber")
    assert len(runner.tasks(runner.configuration("heads_restored"))) == 84
    monkeypatch.setenv("LOOKBACKS_VALUES", "32")
    monkeypatch.setenv("HORIZON_VALUES", "1 5 25")
    monkeypatch.setenv("RANK_WINDOWS_VALUES", "1 5")
    monkeypatch.setenv("LAMBDAS_VALUES", "0 .25 .5 .8 1")
    assert len(runner.tasks(runner.configuration("heads_restored"))) == 90
    monkeypatch.setenv("K_LOSS_VALUES", "5")
    with pytest.raises(ValueError, match="RANK_WINDOWS_VALUES"):
        runner.configuration("heads_restored")


def test_no_ignored_mlp_width_or_disabled_restoration(monkeypatch):
    monkeypatch.setenv("NS_CAPACITY_SPECS", '{"base":{"rank_head_width":64}}')
    with pytest.raises(ValueError, match="rank_head_width"):
        runner.configuration("heads_restored")
    with pytest.raises(ValueError, match="norm=True"):
        restored(norm=False)
    with pytest.raises(ValueError, match="loss_inverse_norm=True"):
        restored(loss_inverse_norm=False)


def test_identical_initialization_and_observed_statistic_restoration():
    cfg = restored().config
    cfg.enc_in = cfg.dec_in = cfg.c_out = 5
    cfg.label_len, cfg.pred_len = 4, 2
    cfg.freq = "d"
    torch.manual_seed(91)
    native = Nonstationary_Transformer(cfg)
    rng = torch.get_rng_state()
    torch.manual_seed(91)
    dual = RestoredTaskNetwork(cfg)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in native.state_dict().items():
        torch.testing.assert_close(value, dual.state_dict()[name], rtol=0, atol=0)
    assert dual.ranking_projection.weight.data_ptr() != dual.decoder.projection.weight.data_ptr()
    torch.testing.assert_close(dual.ranking_projection.weight, dual.decoder.projection.weight, rtol=0, atol=0)
    native.eval(); dual.eval()
    # Heterogeneous location/scale makes restoration materially affect ranking.
    x = torch.randn(2, 8, 5) * torch.tensor([.1, .3, 1., 2., 4.]) + torch.arange(5.)
    marks = torch.zeros(2, 8, 3); ym = torch.zeros(2, 6, 3); dec = torch.randn(2, 6, 5)
    with torch.no_grad():
        a = native(x, marks, dec, ym); b = dual(x, marks, dec, ym)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(b, dual.ranking_returns, rtol=0, atol=0)
    mean = x.mean(1, keepdim=True)
    std = torch.sqrt((x - mean).var(1, keepdim=True, unbiased=False) + 1e-5)
    expected = (dual.ranking_projection(dual._decoder_hidden) * std + mean)[:, -2:]
    torch.testing.assert_close(expected, dual.ranking_returns, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["log_return", "simple_return"])
@pytest.mark.parametrize("lam", [.5, 1.])
def test_fit_inverse_scaler_cumulative_scores_gradients_and_future_isolation(kind, lam):
    m = restored(lam=lam, loss_score_kind=kind)
    m.forecast_fit(data(), train_ratio_in_tv=.75)
    assert m.model.ranking_projection.weight.grad.abs().sum() > 0
    assert m.model.encoder.attn_layers[0].attention.query_projection.weight.grad.abs().sum() > 0
    temporal_grad = m.model.decoder.projection.weight.grad
    assert temporal_grad is None if lam == 1 else temporal_grad.abs().sum() > 0
    m.model.eval()
    x = torch.randn(2, 8, 5); y = torch.randn(2, 6, 5)
    xm = torch.zeros(2, 8, 3); ym = torch.zeros(2, 6, 3)
    with torch.no_grad():
        a = m._process(x, y, xm, ym)["output"].clone()
        score = m.latest_rank_scores.clone()
        mean = torch.tensor(m.scaler.mean_, dtype=a.dtype)
        scale = torch.tensor(m.scaler.scale_, dtype=a.dtype)
        raw = m.model.ranking_returns * scale + mean
        expected = raw.cumsum(1)
        if kind == "simple_return": expected = torch.expm1(expected)
        torch.testing.assert_close(score, expected, rtol=1e-6, atol=1e-8)
        y[:, -2:] = 10000
        b = m._process(x, y, xm, ym)["output"]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(score, m.latest_rank_scores, rtol=0, atol=0)
    pred = m.forecast(2, data())
    assert pred.shape == (2, 5) and m.export_rank_scores.shape == (2, 5)


@pytest.mark.parametrize("loss", ["rankmix", "rankmix_proximity"])
@pytest.mark.parametrize("horizon", [2, 25])
def test_pure_ranking_endpoint_matches_shared_training(loss, horizon):
    # With no temporal objective, copying the projection should reproduce the
    # shared model, including backbone updates and restored cumulative scores.
    outputs = []
    samples = data()
    if horizon == 25:
        import pandas as pd
        samples = pd.DataFrame(np.random.default_rng(491).normal(0, .02, (180, 5)),
                               columns=list('ABCDE'), index=pd.date_range('2000-01-01', periods=180, name='date'))
    for mode in ["shared", "separate"]:
        torch.manual_seed(2026)
        m = restored(mode=mode, lam=1, loss=loss, loss_score_kind="log_return", horizon=horizon, dropout=.1)
        if horizon == 25: m.rank_windows = (1, 5, 10, 15, 20, 25)
        m.forecast_fit(samples, train_ratio_in_tv=.75)
        pred = m.forecast(horizon, samples)
        outputs.append(pred.cumsum(0) if mode == "shared" else m.export_rank_scores)
    np.testing.assert_allclose(*outputs, rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("lam", [0., .5, 1.])
def test_shared_control_preserves_previous_experiment(lam):
    outputs = []
    for make in [lambda: adapter(mode="shared", lam=lam), lambda: restored(mode="shared", lam=lam)]:
        torch.manual_seed(17); m = make()
        m.forecast_fit(data(), train_ratio_in_tv=.75)
        outputs.append(m.forecast(2, data()))
    np.testing.assert_array_equal(*outputs)


def test_restored_frozen_worker_and_checkpoint_roundtrip(tmp_path, monkeypatch):
    plan, cfg = create_fixture(tmp_path, monkeypatch, study="heads_restored")
    saved = json.loads(plan.read_text())
    ix = next(i for i, t in enumerate(saved["tasks"]) if t["head_mode"] == "separate" and
              t["pred_len"] == 2 and t["lambda_value"] == 1)
    runner.run_worker(plan, ix)
    task = saved["tasks"][ix]; folder = plan.parent / ("fit=" + task["fit_id"])
    record = json.loads((folder / "result.json").read_text())
    assert record["score_semantics"] == "cumulative_simple_return"
    assert not record["return_forecast_valid"] and not (folder / "part.parquet").exists()
    checkpoint = torch.load(folder / "checkpoint.pt", weights_only=False)
    state = checkpoint["state_dict"]
    assert "ranking_projection.weight" in state and "decoder.projection.weight" in state
    assert all(runner.sha(folder / p) == v for p, v in record["files_sha256"].items())
    runner.run_worker(plan, ix)  # Verify idempotent, hash-checked completion.
