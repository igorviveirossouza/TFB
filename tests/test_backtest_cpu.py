"""Oraculos economicos e integracao com contratos reais de arquivo (sem cluster/R)."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from backtest_cpu.core import (Market, Strategy, backtest, cumulative_targets, ic_frame, ic_daily_frame,
                               metrics, spearman_rows)
from backtest_cpu.io import digest, load_bundle, load_predictions, load_rf, read_json, RF_SOURCE, RF_CONVENTION
from backtest_cpu.pipeline import valid_result


def market_frame(n=12, assets=4):
    days = pd.bdate_range("2024-01-02", periods=n)
    tickers = [f"AAA{i}" for i in range(assets)]
    opening = 100 + np.arange(n)[:, None]*.2 + np.arange(assets)[None, :]
    closing = opening*(1 + .003*np.cos(np.arange(n)[:, None]+np.arange(assets)))
    previous = np.vstack([opening[:1], closing[:-1]])
    cal = pd.DataFrame({"date": np.arange(1, n+1), "date_pregao": days,
                        "data_anterior": [pd.Timestamp("2023-12-29"), *days[:-1]]})
    frame = pd.DataFrame({"date": np.repeat(cal.date, assets), "cols": np.tile(tickers, n),
                          "date_pregao": np.repeat(days, assets),
                          "data_anterior": np.repeat(cal.data_anterior, assets),
                          "abertura": opening.ravel(), "fechamento": closing.ravel(),
                          "fechamento_execucao": closing.ravel(), "fechamento_anterior": previous.ravel(),
                          "fator_quantidade_evento": 1., "provento_por_acao_com": 0.,
                          "valor_ativos_por_acao_com": 0., "log_retorno_corrigido": np.log(closing/previous).ravel(),
                          "pode_executar_abertura": True, "pode_executar_fechamento": True,
                          "preco_preenchido": False})
    return frame, cal


def event_market(distribution=False):
    f, cal = market_frame(4, 1)
    # Origem=date1, entrada=date2: split 2 e D=10 pertencem ao titular anterior.
    f["abertura"] = [100, 44, 43, 440]
    f["fechamento"] = f["fechamento_execucao"] = [100, 45, 44, 450]
    f["fechamento_anterior"] = [100, 100, 45, 44]
    f["fator_quantidade_evento"] = [1, 2, 1, .1]
    f["provento_por_acao_com"] = [0, 10, 2, 0]
    f["valor_ativos_por_acao_com"] = [0, 0, 3 if distribution else 0, 0]
    f["log_retorno_corrigido"] = np.log((f.fator_quantidade_evento*f.fechamento+
                                        f.provento_por_acao_com+f.valor_ativos_por_acao_com)/f.fechamento_anterior)
    return Market.from_frame(f, cal)


def test_ic_ties_missing_and_constant_match_r_convention():
    x = np.array([[1, 1, 3, 4], [4, 3, 2, 1], [1, 1, 1, 1], [1, 2, np.nan, np.nan],
                  [1, 2, 3, np.nan], [np.nan]*4])
    y = np.array([[2, 1, 3, 3], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                  [2, 3, 1, 4], [1, 2, 3, 4]])
    result, count, status = spearman_rows(x, y)
    for i in [0, 1, 4]:
        ok = np.isfinite(x[i]) & np.isfinite(y[i])
        assert result[i] == pytest.approx(spearmanr(x[i, ok], y[i, ok]).statistic)
    assert result[0] == pytest.approx(8/9)
    assert np.isnan(result[[2, 3, 5]]).all()
    assert count.tolist() == [4, 4, 4, 2, 3, 0]
    assert status[2] == "RANK_CONSTANTE"


def test_ic_targets_use_corrected_returns_and_exact_origin():
    m = event_market()
    target = cumulative_targets(m, np.array([1]), 3)[0, 0]
    assert target == pytest.approx(np.log(1)*1 + np.log(46/45)+np.log(45/44))
    assert target != pytest.approx(np.log(450/100))
    f, c = market_frame()
    market = Market.from_frame(f, c)
    origins = np.array([6, 7, 8])
    targets = cumulative_targets(market, origins, 3)
    ic = ic_frame(targets, origins, 3, market, 7)
    assert ic.spearman.tolist() == pytest.approx([1, 1, 1])
    assert ic.data_alvo.tolist() == c.date_pregao.iloc[[8, 9, 10]].tolist()
    assert ic.amostra_comum.tolist() == [True, True, False]


def test_daily_ic_distinguishes_leads_and_origins_for_same_target():
    f, c = market_frame()
    m = Market.from_frame(f, c)
    origins = np.array([5, 6])
    actual = m.arrays["log_retorno_corrigido"][origins[:, None]+np.arange(2)]
    pred = actual.copy()
    pred[0, 1] *= -1
    ic = ic_daily_frame(pred, origins, m, 5)
    assert ic.spearman.tolist() == pytest.approx([1, -1, 1, 1])
    same_target = ic[ic.target_step == 7]
    assert same_target.origin_step.tolist() == [5, 6]
    assert same_target.h.tolist() == [2, 1]
    assert ic.amostra_comum.tolist() == [True, True, False, False]


@pytest.mark.parametrize("side,expected", [("compra", 47/44), ("venda", 41/44)])
def test_events_fixed_quantities_and_short_pnl(side, expected):
    m = event_market()
    result = backtest(np.array([[1.]]), np.array([1]), 3, m, Strategy("s", side, 1, False))
    assert result["carteira_diaria"].patrimonio.iloc[-1] == pytest.approx(expected)
    assert abs(result["posicoes"].quantidade_saida_por_capital.iloc[0]) == pytest.approx(.1/44)
    assert abs(result["posicoes"].direitos_por_capital.iloc[0]) == pytest.approx(2/44)
    assert result["carteira_periodos"].data_entrada.iloc[0] == m.calendar.date_pregao.iloc[1]
    assert len(result["carteira_diaria"]) == 3


def test_entry_on_ex_date_excludes_distributed_asset():
    f, c = market_frame(2, 1)
    f.loc[1, ["abertura", "fechamento", "fechamento_execucao", "fechamento_anterior", "valor_ativos_por_acao_com"]] = [33.90, 35.65, 35.65, 41.11, 8.51]
    f.loc[0, ["fechamento", "fechamento_execucao"]] = 41.11
    f["log_retorno_corrigido"] = np.log((f.fechamento+f.valor_ativos_por_acao_com)/f.fechamento_anterior)
    m = Market.from_frame(f, c)
    result = backtest(np.ones((1, 1)), np.array([1]), 1, m, Strategy("s", "compra", 1, False))
    assert result["carteira_periodos"].retorno.iloc[0] == pytest.approx(35.65/33.90-1)
    assert result["posicoes"].direitos_por_capital.iloc[0] == 0
    with pytest.raises(ValueError, match="ATIVO_DISTRIBUIDO"):
        backtest(np.ones((1, 1)), np.array([1]), 3, event_market(True), Strategy("s", "compra", 1, False))


def test_slippage_costs_both_legs_and_capital_constraint():
    f, c = market_frame(2, 1)
    f["abertura"] = f["fechamento"] = f["fechamento_execucao"] = f["fechamento_anterior"] = 100.
    f["log_retorno_corrigido"] = 0.
    m = Market.from_frame(f, c)
    for side, sign in [("compra", 1), ("venda", -1)]:
        out = backtest(np.ones((1, 1)), np.array([1]), 1, m, Strategy("s", side, 1, False), 10, 5)
        entry = 100*(1+sign*.001)
        exit_ = 100*(1-sign*.001)
        q = 1/(entry*1.0005)
        expected = sign*q*(exit_-entry)-.0005*q*(entry+exit_)
        assert out["carteira_periodos"].retorno.iloc[0] == pytest.approx(expected)
        assert abs(out["posicoes"].quantidade_entrada_por_capital.iloc[0])*entry*1.0005 == pytest.approx(1.)


def test_four_rankings_cash_ties_and_no_future_filtering():
    f, c = market_frame(4, 4)
    m = Market.from_frame(f, c)
    scores = np.array([[.2, .2, 0., -.3]])
    strategies = [("compra", True, ["AAA0", "AAA1"]), ("compra", False, ["AAA0", "AAA1", "AAA2"]),
                  ("venda", True, ["AAA3"]), ("venda", False, ["AAA3", "AAA2", "AAA0"])]
    for side, filt, expected in strategies:
        out = backtest(scores, np.array([1]), 1, m, Strategy("s", side, 3, filt))
        assert out["posicoes"].ticker.tolist() == expected
        assert out["posicoes"].peso_entrada.sum() == pytest.approx(1.)
    cash = backtest(-np.ones((1, 4)), np.array([1]), 1, m, Strategy("s", "compra", 3, True))
    assert cash["carteira_diaria"].retorno.tolist() == [0.]
    assert cash["posicoes"].empty
    f.loc[(f.date == 2) & (f.cols == "AAA0"), "pode_executar_abertura"] = False
    with pytest.raises(ValueError, match="ABERTURA_INEXECUTAVEL.*AAA0"):
        backtest(scores, np.array([1]), 1, Market.from_frame(f, c), Strategy("s", "compra", 1, False))
    # Mesmo com saida indisponivel, a escolha nao passa retrospectivamente ao segundo ativo.
    f.loc[(f.date == 2) & (f.cols == "AAA0"), "pode_executar_abertura"] = True
    f.loc[(f.date == 3) & (f.cols == "AAA0"), "pode_executar_fechamento"] = False
    with pytest.raises(ValueError, match="FECHAMENTO_INEXECUTAVEL.*AAA0"):
        backtest(scores, np.array([1]), 2, Market.from_frame(f, c), Strategy("s", "compra", 1, False))


def test_equal_entry_weights_drift_without_daily_rebalancing():
    f, c = market_frame(3, 2)
    f["abertura"] = [100, 50, 100, 50, 110, 45]
    f["fechamento"] = f["fechamento_execucao"] = [100, 50, 110, 45, 120, 40]
    f["fechamento_anterior"] = [100, 50, 100, 50, 110, 45]
    f["log_retorno_corrigido"] = np.log(f.fechamento/f.fechamento_anterior)
    result = backtest(np.ones((1, 2)), np.array([1]), 2, Market.from_frame(f, c), Strategy("s", "compra", 2, False))
    # Q=(.005,.01): primeiro fechamento .55+.45=1; ultimo .60+.40=1.
    assert result["posicoes"].quantidade_entrada_por_capital.tolist() == pytest.approx([.005, .01])
    assert result["carteira_diaria"].patrimonio.tolist() == pytest.approx([1, 1])


def test_daily_drawdown_includes_initial_capital_and_short_insolvency():
    f, c = market_frame(2, 1)
    f.loc[1, ["abertura", "fechamento", "fechamento_execucao"]] = [100, 80, 80]
    f["log_retorno_corrigido"] = np.log(f.fechamento/f.fechamento_anterior)
    m = Market.from_frame(f, c)
    result = backtest(np.ones((1, 1)), np.array([1]), 1, m, Strategy("s", "compra", 1, False))
    assert result["metricas"].max_drawdown.iloc[0] == pytest.approx(-.2)
    f.loc[1, ["fechamento", "fechamento_execucao"]] = 210
    f["log_retorno_corrigido"] = np.log(f.fechamento/f.fechamento_anterior)
    with pytest.raises(ValueError, match="PATRIMONIO_NAO_POSITIVO"):
        backtest(np.ones((1, 1)), np.array([1]), 1, Market.from_frame(f, c), Strategy("s", "venda", 1, False))


def test_metrics_cdi_and_missing_not_silently_zero():
    r = np.array([-.1, .2, .03])
    rf = np.array([.001, .002, .003])
    eq = np.cumprod(1+r)
    d = pd.DataFrame({"data": pd.date_range("2024-01-01", periods=3), "retorno": r, "rf": rf,
                      "patrimonio": eq, "drawdown": eq/np.maximum.accumulate(np.r_[1., eq])[1:]-1})
    periods = pd.DataFrame({"n_ativos": [2]})
    met = metrics(d, periods).iloc[0]
    assert met.sharpe_cdi == pytest.approx(np.sqrt(252)*np.mean(r-rf)/np.std(r-rf, ddof=1))
    assert met.cdi_acumulado == pytest.approx(np.prod(1+rf)-1)
    assert met.status_rf == RF_SOURCE
    assert met.retorno_acumulado == pytest.approx(eq[-1]-1)
    d["rf"] = np.nan
    met = metrics(d, periods).iloc[0]
    assert np.isnan(met.sharpe_cdi) and np.isfinite(met.sharpe_rf_zero)


def test_rf_preserves_r_compounding_and_checks_missing_units_source_and_dates():
    cal = pd.DataFrame({"date": [1, 2], "date_pregao": ["2024-01-03", "2024-01-05"],
                        "data_anterior": ["2024-01-02", "2024-01-03"],
                        "retorno_cdi": [.01, 1.02*1.03-1],
                        "n_observacoes_cdi": [1, 2], "rf_fonte": RF_SOURCE})
    cal["log_retorno_cdi"] = np.log1p(cal.retorno_cdi)
    assert load_rf(cal.iloc[::-1]) == pytest.approx([.01, .0506])
    for column, value, error in [("retorno_cdi", np.nan, "invalido/incompleto"),
                                  ("retorno_cdi", 1., "simples/log"),
                                  ("rf_fonte", "SELIC_SGS11", "invalido/incompleto"),
                                  ("n_observacoes_cdi", 0, "invalido/incompleto"),
                                  ("data_anterior", "2024-01-04", "Datas"),
                                  ("date", 2, "duplicado/desalinhado")]:
        broken = cal.copy(); broken.loc[0, column] = value
        with pytest.raises(ValueError, match=error):
            load_rf(broken)
    with pytest.raises(ValueError, match="CDI ausente"):
        load_rf(cal.drop(columns="retorno_cdi"))


def make_inputs(root):
    f, cal = market_frame(16, 4)
    base = root/"base"
    base.mkdir()
    returns = root/"retornos.csv"
    f[["date", "log_retorno_corrigido", "cols"]].rename(columns={"log_retorno_corrigido": "data"}).to_csv(returns, index=False)
    f.to_csv(base/"mercado_diario.csv", index=False)
    cal["retorno_cdi"] = .0002 + .00001*np.arange(len(cal))
    cal["log_retorno_cdi"] = np.log1p(cal.retorno_cdi)
    cal["n_observacoes_cdi"] = 1
    cal["rf_fonte"] = RF_SOURCE
    cal.to_csv(base/"calendario_tfb.csv", index=False)
    pd.DataFrame([{"versao": "2026-10-15-v2-cdi", "arquivo_base": "mercado_diario.csv",
                   "rf_fonte": RF_SOURCE, "rf_convencao": RF_CONVENTION,
                   "n_linhas": len(f), "n_ativos": 4, "n_pregoes": 16}]).to_csv(base/"manifesto.csv", index=False)
    pd.DataFrame([{"papel": "RETORNOS_USADOS_NO_TREINO", "arquivo": str(returns), "md5": digest(returns, "md5")},
                  {"papel": "cdi_tfb.csv", "arquivo": "auditoria/cdi_tfb.csv", "md5": "0"*32}]).to_csv(base/"fontes.csv", index=False)
    pd.DataFrame([{"arquivo": p.name, "md5": digest(p, "md5")} for p in base.iterdir()]).to_csv(base/"checksums_saidas.csv", index=False)
    pred_root = root/"previsoes"
    for mode in ("temporal", "var"):
        group = pred_root/"treino=50"/f"experimento={mode}"
        group.mkdir(parents=True)
        models = ["DUET", "TimesNet"] if mode == "temporal" else ["VAR"]
        cfg = {"models": models, "lookbacks": [2], "horizons": [1, 3], "var_lags": 2,
               "train_percentages": [50], "strategy": {"stride": 1}}
        meta = {"source_data_kind": "log_return", "step_offset": 1, "n_rows": 16,
                "n_assets": 4, "targets": sorted(f.cols.unique()), "source_sha256": digest(returns)}
        (group/"run_config.json").write_text(json.dumps({"metadata": meta, "config": cfg, "design_sha256": "fixture"}))
        rows = []
        for model in models:
            hashes = {}
            for h in [1, 3]:
                out = group/"loss=mse"/"lambda=00"/f"modelo={model}"/"lookback=2"/f"pred_len={h}"
                out.mkdir(parents=True)
                origins = np.arange(8, 16-h+1)
                records = pd.DataFrame({"arquivo": np.repeat([f"janela_{o:06d}.csv" for o in origins], h),
                                         "h": np.tile(np.arange(1, h+1), len(origins)),
                                         "origin_step": np.repeat(origins, h)})
                records["step"] = records.origin_step+records.h
                for j, ticker in enumerate(meta["targets"]):
                    records[ticker] = (.2-j*.15)*np.cos(records.origin_step)
                records.to_parquet(out/"part.parquet", index=False)
                hashes[str(h)] = digest(out/"part.parquet")
                if mode == "temporal":
                    rows.append({"modelo": model, "lookback": 2, "pred_len": h, "parquet_sha256": hashes[str(h)]})
            if mode == "var":
                rows.append({"modelo": model, "lookback": 2, "pred_len": 3, "parquet_sha256": json.dumps(hashes)})
        for row in rows:
            row.update(source_sha256=digest(returns), design_sha256="fixture", source_data_kind="log_return",
                       step_offset=1, loss="mse", cross_lambda=0, experimento=mode, treino=50, n_pretest=8)
        pd.DataFrame(rows).to_csv(group/"completed_tasks.csv", index=False)
    return base, returns, pred_root


def cli(*args, check=True):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-B", str(ROOT/"scripts/run_backtests.py"), *map(str, args)],
                            text=True, capture_output=True, env=env)
    if check and result.returncode:
        pytest.fail(result.stdout+result.stderr)
    return result


def prepare_args(root, local=True):
    base, returns, predictions = make_inputs(root)
    args = ["prepare", "--market", base, "--returns-file", returns, "--predictions", predictions,
            "--strategies", ROOT/"config/estrategias_backtest.json", "--run-dir", root/"run",
            "--train", "50", "--k", "1", "3"]
    return args+["--allow-local"] if local else args


def test_end_to_end_parallel_resume_parquet_calendar_and_corruption(tmp_path):
    args = prepare_args(tmp_path)
    cli(*args, "--dry-run")
    assert not (tmp_path/"run").exists()
    cli(*args)
    run = tmp_path/"run"
    plan = read_json(run/"plano.json")
    assert len(plan["tasks"]) == 6
    assert plan["rf"] == RF_SOURCE and plan["rf_convencao"] == RF_CONVENTION
    # Codigo copiado funciona a partir de /var/spool equivalente, sem imports relativos ao repo.
    result = subprocess.run([sys.executable, "-B", str(run/"codigo/run_backtests.py"), "run", "--run-dir", str(run), "--workers", "2"],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout+result.stderr
    state = read_json(run/"resumos/estado_execucao.json")
    assert state["completa"] and state["COMPLETA"] == 6
    metrics_all = pd.read_parquet(run/"resumos/metricas.parquet")
    assert len(metrics_all) == 32  # Redes: 2 x (1+2 K) x 4; VAR: 2 K x 4, sem repetir H.
    assert metrics_all.sharpe_cdi.notna().all()
    assert metrics_all.status_rf.eq(RF_SOURCE).all()
    assert metrics_all[metrics_all.modelo == "VAR"].pred_len.unique().tolist() == [3]
    for k in [1, 3]:
        group = metrics_all[metrics_all.k == k]
        assert group.inicio.nunique() == group.fim.nunique() == 1
    task = plan["tasks"][0]
    dest = run/"tasks"/task["relpath"]
    d = pd.read_parquet(dest/"carteira_diaria.parquet")
    assert d.rf.to_numpy() == pytest.approx(.0002 + .00001*(d.date.to_numpy()-1))
    periods = pd.read_parquet(dest/"carteira_periodos.parquet")
    assert periods.retorno_cdi.to_numpy() == pytest.approx(.0002 + .00001*(periods.entry_step.to_numpy()-1))
    before = (dest/"ic.parquet").stat().st_mtime_ns
    result = cli("run", "--run-dir", run, "--workers", "1")
    assert result.stdout.count("REUTILIZADA") == 6
    assert (dest/"ic.parquet").stat().st_mtime_ns == before
    # Corrupcao de uma saida refaz apenas a tarefa afetada.
    (dest/"ic.parquet").write_bytes(b"corrupted")
    result = cli("run", "--run-dir", run, "--workers", "1")
    assert result.stdout.count("REUTILIZADA") == 5
    assert valid_result(dest, plan["signature"])
    # Interrupcao entre arquivos e marcador final tambem pode ser retomada.
    (dest/"conclusao.json").unlink()
    result = cli("run", "--run-dir", run, "--workers", "1")
    assert result.stdout.count("REUTILIZADA") == 5
    assert valid_result(dest, plan["signature"])
    (dest/"conclusao.json").write_text("{incompleto")
    result = cli("run", "--run-dir", run, "--workers", "1")
    assert result.stdout.count("REUTILIZADA") == 5
    assert valid_result(dest, plan["signature"])
    # Codigo/parametros diferentes exigem outra pasta; nunca mistura cenarios.
    assert cli(*args, "--cost-bps", "5", check=False).returncode == 1
    # Uma previsao reescrita nao reutiliza resultados antigos nem aparece como completa.
    Path(task["path"]).write_bytes(b"alterada")
    result = cli("run", "--run-dir", run, "--workers", "1", check=False)
    assert result.returncode == 1
    assert read_json(run/"resumos/estado_execucao.json")["ERRO"] == 1


def test_incomplete_grid_source_offset_and_snfs2_guard(tmp_path):
    args = prepare_args(tmp_path, local=False)
    result = cli(*args, check=False)
    assert result.returncode == 1 and "SNFS2" in result.stderr
    assert not (tmp_path/"run").exists()
    group = tmp_path/"previsoes/treino=50/experimento=temporal"
    done = pd.read_csv(group/"completed_tasks.csv")
    done.iloc[:-1].to_csv(group/"completed_tasks.csv", index=False)
    result = cli(*args, "--allow-local", check=False)
    assert result.returncode == 1 and "Grade incompleta" in result.stderr
    done.to_csv(group/"completed_tasks.csv", index=False)
    config = read_json(group/"run_config.json")
    config["metadata"]["step_offset"] = 2
    (group/"run_config.json").write_text(json.dumps(config))
    result = cli(*args, "--allow-local", check=False)
    assert result.returncode == 1 and "offset" in result.stderr


def test_prediction_alignment_and_bundle_checksums(tmp_path):
    args = prepare_args(tmp_path)
    cli(*args)
    plan = read_json(tmp_path/"run/plano.json")
    m, _ = load_bundle(tmp_path/"base", tmp_path/"retornos.csv")
    task = dict(plan["tasks"][1])
    path = Path(task["path"])
    frame = pd.read_parquet(path)
    frame.loc[0, "step"] += 1
    frame.to_parquet(path, index=False)
    task["sha256"] = digest(path)
    with pytest.raises(ValueError, match="step desalinhado"):
        load_predictions(task, m)
    with (tmp_path/"base/mercado_diario.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Checksum divergente"):
        load_bundle(tmp_path/"base", tmp_path/"retornos.csv")


def test_prepare_requires_embedded_cdi_before_creating_run(tmp_path):
    args = prepare_args(tmp_path)
    base = tmp_path/"base"
    path = base/"calendario_tfb.csv"
    cal = pd.read_csv(path)
    cal.loc[3, "retorno_cdi"] = np.nan
    cal.to_csv(path, index=False)
    checks = pd.read_csv(base/"checksums_saidas.csv")
    checks.loc[checks.arquivo == path.name, "md5"] = digest(path, "md5")
    checks.to_csv(base/"checksums_saidas.csv", index=False)
    result = cli(*args, check=False)
    assert result.returncode == 1 and "CDI invalido/incompleto" in result.stderr
    assert not (tmp_path/"run").exists()
    manifest = base/"manifesto.csv"
    meta = pd.read_csv(manifest); meta["versao"] = "2026-10-15-v1"
    meta.to_csv(manifest, index=False)
    checks.loc[checks.arquivo == manifest.name, "md5"] = digest(manifest, "md5")
    checks.to_csv(base/"checksums_saidas.csv", index=False)
    result = cli(*args, check=False)
    assert result.returncode == 1 and "atualizar_cdi_base_backtest" in result.stderr
    assert not (tmp_path/"run").exists()


def test_partial_portfolio_failure_preserves_ic_and_reports_nonzero(tmp_path):
    args = prepare_args(tmp_path)
    # Altera mercado antes do plano para forcar falha de execucao em todas as entradas.
    path = tmp_path/"base/mercado_diario.csv"
    f = pd.read_csv(path)
    f["pode_executar_abertura"] = False
    f["abertura"] = np.nan
    f.to_csv(path, index=False)
    cp = tmp_path/"base/checksums_saidas.csv"
    check = pd.read_csv(cp)
    check.loc[check.arquivo == path.name, "md5"] = digest(path, "md5")
    check.to_csv(cp, index=False)
    cli(*args)
    result = cli("run", "--run-dir", tmp_path/"run", "--workers", "1", check=False)
    assert result.returncode == 1
    state = read_json(tmp_path/"run/resumos/estado_execucao.json")
    assert not state["completa"] and state["PARCIAL"] == 5 and state["COMPLETA"] == 1
    assert len(pd.read_parquet(tmp_path/"run/resumos/ic_resumo.parquet")) == 24
    assert len(pd.read_parquet(tmp_path/"run/resumos/ic_acumulado_resumo.parquet")) == 18
    assert len(pd.read_parquet(tmp_path/"run/resumos/falhas.parquet")) == 32
