"""Contratos da base R e dos Parquets publicados por TFB/scripts/b3_experiment.py."""
import hashlib
import itertools
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .core import Market, Strategy, integers


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_table(path):
    path = Path(path)
    # ParquetFile evita inferencia Hive de diretorios treino=... ao ler um arquivo.
    # Inferir tipos no CSV inteiro: colunas de auditoria podem misturar IDs e texto.
    return (pq.ParquetFile(path).read(use_threads=False).to_pandas()
            if path.suffix == ".parquet" else pd.read_csv(path, keep_default_na=True, low_memory=False))


def load_strategies(path):
    raw = read_json(path)
    if not isinstance(raw, list) or not raw:
        raise ValueError("Estrategias: esperada lista JSON nao vazia.")
    strategies = [Strategy(**s) for s in raw]
    if len({s.id for s in strategies}) != len(strategies):
        raise ValueError("IDs de estrategias duplicados.")
    return raw


RF_SOURCE = "BCB_SGS12"
RF_CONVENTION = "CDI_ACUMULADO_DATA_ANTERIOR_INCLUSIVE_PREGAO_EXCLUSIVE"


def load_rf(calendar):
    """CDI ja alinhado no R: [data_anterior, date_pregao), simples decimal.

    Cada linha pode acumular mais de um dia ANBIMA. Nao deslocar, interpolar,
    anualizar ou converter novamente a serie recebida.
    """
    required = {"date", "date_pregao", "data_anterior", "retorno_cdi",
                "log_retorno_cdi", "n_observacoes_cdi", "rf_fonte"}
    if not required <= set(calendar):
        raise ValueError("CDI ausente no calendario; atualize a base no R com atualizar_cdi_base_backtest().")
    frame = calendar.sort_values("date")
    dates = integers(frame.date, "CDI/date")
    if not np.array_equal(dates, np.arange(1, len(frame)+1)) or not len(frame):
        raise ValueError("Calendario CDI duplicado/desalinhado.")
    current = pd.to_datetime(frame.date_pregao, format="%Y-%m-%d", errors="raise")
    previous = pd.to_datetime(frame.data_anterior, format="%Y-%m-%d", errors="raise")
    if (current.isna().any() or previous.isna().any() or not current.is_monotonic_increasing
            or current.duplicated().any() or (previous >= current).any()
            or not np.array_equal(previous.values[1:], current.values[:-1])):
        raise ValueError("Datas do calendario CDI invalidas.")
    rates = pd.to_numeric(frame.retorno_cdi, errors="raise").to_numpy(dtype=float)
    logs = pd.to_numeric(frame.log_retorno_cdi, errors="raise").to_numpy(dtype=float)
    counts = integers(frame.n_observacoes_cdi, "CDI/n_observacoes")
    if (frame.rf_fonte.isna().any() or not frame.rf_fonte.eq(RF_SOURCE).all()
            or not np.isfinite(rates).all() or not np.isfinite(logs).all()
            or np.any(rates <= -1) or np.any(counts < 1)
            or np.any(counts > (current-previous).dt.days.to_numpy())):
        raise ValueError("CDI invalido/incompleto; nao substituir taxas por zero ou outra fonte.")
    if not np.allclose(np.log1p(rates), logs, rtol=1e-10, atol=1e-12):
        raise ValueError("CDI simples/log divergentes; confira unidades e calendario no R.")
    return rates


def load_bundle(directory, returns_file):
    directory, returns_file = Path(directory).resolve(), Path(returns_file).resolve()
    checks = pd.read_csv(directory/"checksums_saidas.csv", dtype=str, keep_default_na=False)
    if set(checks.columns) != {"arquivo", "md5"} or checks.arquivo.duplicated().any():
        raise ValueError("Checksums da base de mercado invalidos.")
    names = set(checks.arquivo)
    if not {"manifesto.csv", "calendario_tfb.csv", "fontes.csv"} <= names:
        raise ValueError("Checksums incompletos na base de mercado.")
    files = {"checksums_saidas.csv": digest(directory/"checksums_saidas.csv")}
    for row in checks.itertuples():
        if Path(row.arquivo).name != row.arquivo or row.arquivo in {"", ".", ".."}:
            raise ValueError("Caminho invalido nos checksums da base.")
        path = directory/row.arquivo
        if digest(path, "md5") != row.md5.lower():
            raise ValueError(f"Checksum divergente na base: {path}")
        files[row.arquivo] = digest(path)
    manifest = pd.read_csv(directory/"manifesto.csv")
    if (len(manifest) != 1 or manifest.versao.iloc[0] not in {"2026-10-15-v2-cdi", "2026-10-15-v3-recebidos"}
            or manifest.arquivo_base.iloc[0] not in names
            or manifest.arquivo_base.iloc[0] not in {"mercado_diario.csv", "mercado_diario.parquet"}):
        raise ValueError("Base sem contrato CDI atual; atualize no R com atualizar_cdi_base_backtest().")
    if (not {"rf_fonte", "rf_convencao"} <= set(manifest)
            or manifest.rf_fonte.iloc[0] != RF_SOURCE or manifest.rf_convencao.iloc[0] != RF_CONVENTION):
        raise ValueError("Fonte/convencao CDI desconhecida no manifesto.")
    sources = pd.read_csv(directory/"fontes.csv", dtype=str)
    source = sources[sources.papel == "RETORNOS_USADOS_NO_TREINO"]
    if len(source) != 1 or digest(returns_file, "md5") != source.md5.iloc[0].lower():
        raise ValueError("Retornos de treino diferem do arquivo registrado no bundle de mercado.")
    cdi_source = sources[sources.papel == "cdi_tfb.csv"]
    if len(cdi_source) != 1 or not re.fullmatch(r"[0-9a-fA-F]{32}", str(cdi_source.md5.iloc[0])):
        raise ValueError("Proveniencia do CDI auditado ausente/invalida em fontes.csv.")
    calendar = pd.read_csv(directory/"calendario_tfb.csv")
    frame = read_table(directory/manifest.arquivo_base.iloc[0])
    received = None
    if manifest.versao.iloc[0] == "2026-10-15-v3-recebidos":
        if ("tipo_registro" not in frame or frame.tipo_registro.isna().any()
                or not frame.tipo_registro.isin(["MODELO", "RECEBIDO"]).all()):
            raise ValueError("Tipos de registro invalidos na base com ativos recebidos.")
        received = frame[frame.tipo_registro == "RECEBIDO"].copy()
        frame = frame[frame.tipo_registro == "MODELO"].copy()
        if (not {"n_linhas_recebidos", "n_eventos_recebidos"} <= set(manifest)
                or len(received) != int(manifest.n_linhas_recebidos.iloc[0])
                or "id_distribuicao" not in received
                or received.id_distribuicao.nunique() != int(manifest.n_eventos_recebidos.iloc[0])):
            raise ValueError("Dimensoes de ativos recebidos divergentes do manifesto.")
        for role in ("CADASTRO_RECEBIDOS", "COTACOES_RECEBIDAS", "auditoria_distribuicoes_em_ativos.csv"):
            origin = sources[sources.papel == role]
            if len(origin) != 1 or not re.fullmatch(r"[0-9a-fA-F]{32}", str(origin.md5.iloc[0])):
                raise ValueError(f"Proveniencia de ativos recebidos ausente: {role}")
    elif "tipo_registro" in frame:
        raise ValueError("Registros auxiliares exigem contrato v3-recebidos.")
    market = Market.from_frame(frame, calendar, load_rf(calendar), received)
    dims = (len(frame), len(market.tickers), market.n)
    if dims != tuple(int(manifest[c].iloc[0]) for c in ["n_linhas", "n_ativos", "n_pregoes"]):
        raise ValueError("Dimensoes do manifesto de mercado divergentes.")
    returns = pd.read_csv(returns_file)
    if set(returns.columns) != {"date", "data", "cols"} or returns.duplicated(["date", "cols"]).any():
        raise ValueError("CSV de treino deve conter date,data,cols sem duplicatas.")
    returns["date"] = integers(returns.date, "Retornos/date")
    returns = returns.sort_values(["date", "cols"])
    if (len(returns) != len(frame) or not np.array_equal(returns.date, np.repeat(np.arange(1, market.n+1), len(market.tickers)))
            or not np.array_equal(returns.cols, np.tile(market.tickers, market.n))
            or not np.allclose(returns.data.to_numpy(float).reshape(market.n, -1),
                               market.arrays["log_retorno_corrigido"], rtol=1e-10, atol=1e-12)):
        raise ValueError("Base de mercado nao reproduz exatamente o painel de treino informado.")
    identity = {"base": str(directory), "arquivos": files, "retornos": str(returns_file),
                "retornos_sha256": digest(returns_file),
                "rf_fonte": RF_SOURCE, "rf_convencao": RF_CONVENTION,
                "n_pregoes": market.n, "tickers": market.tickers.tolist()}
    return market, identity


def discover_tasks(root, percentages, market, source_sha256):
    """Grade completa dos run_config; nunca aceita apenas o subconjunto que terminou."""
    root = Path(root).resolve()
    tasks, provenance = [], {}
    for pct, mode in itertools.product(percentages, ("temporal", "var")):
        group = root/f"treino={pct}"/f"experimento={mode}"
        plan = read_json(group/"run_config.json")
        meta, cfg = plan["metadata"], plan["config"]
        if (meta.get("source_data_kind") != "log_return" or meta["step_offset"] != 1
                or meta["n_rows"] != market.n or meta["n_assets"] != len(market.tickers)
                or sorted(meta["targets"]) != market.tickers.tolist()
                or meta["source_sha256"] != source_sha256):
            raise ValueError(f"Fonte, universo ou offset incompatibilidade em {group}/run_config.json")
        if pct not in cfg["train_percentages"] or cfg["strategy"]["stride"] != 1:
            raise ValueError(f"Corte/stride inesperado em {group}")
        models = cfg["models"] if mode == "temporal" else ["VAR"]
        lookbacks = cfg["lookbacks"] if mode == "temporal" else [cfg["var_lags"]]
        horizons = cfg["horizons"]
        if (not models or not lookbacks or not horizons
                or any(not isinstance(m, str) or not re.fullmatch(r"[A-Za-z0-9_]+", m) for m in models)
                or len(set(models)) != len(models)
                or any(type(h) is not int or h < 1 for h in [*lookbacks, *horizons])
                or len(set(horizons)) != len(horizons) or len(set(lookbacks)) != len(lookbacks)):
            raise ValueError(f"Modelos/lookbacks/horizontes invalidos em {group}")
        done_path = group/"completed_tasks.csv"
        done = pd.read_csv(done_path, dtype=str, keep_default_na=False)
        if done.duplicated(["modelo", "lookback", "pred_len"]).any():
            raise ValueError(f"Conclusoes duplicadas em {done_path}")
        expected_keys = set(itertools.product(models, map(str, lookbacks),
                            map(str, horizons if mode == "temporal" else [max(horizons)])))
        actual_keys = set(done[["modelo", "lookback", "pred_len"]].itertuples(index=False, name=None))
        if expected_keys != actual_keys:
            raise ValueError(f"Grade incompleta/divergente em {done_path}: "
                             f"faltam={sorted(expected_keys-actual_keys)[:8]}, extras={sorted(actual_keys-expected_keys)[:8]}")
        provenance[str(group/"run_config.json")] = digest(group/"run_config.json")
        provenance[str(done_path)] = digest(done_path)
        for model, lb, h in itertools.product(models, lookbacks, horizons):
            registered_h = h if mode == "temporal" else max(horizons)
            row = done[(done.modelo == model) & (done.lookback == str(lb)) & (done.pred_len == str(registered_h))].iloc[0]
            if (row.source_sha256 != source_sha256 or row.design_sha256 != plan["design_sha256"]
                    or row.source_data_kind != "log_return" or int(row.step_offset) != 1
                    or row.loss != "mse" or float(row.cross_lambda) != 0
                    or row.experimento != mode or int(row.treino) != pct
                    or int(row.n_pretest) != market.n*pct//100):
                raise ValueError(f"Proveniencia inconsistente em {done_path}: {model}/{lb}/{h}")
            expected_hash = row.parquet_sha256 if mode == "temporal" else json.loads(row.parquet_sha256)[str(h)]
            rel = (Path(f"treino={pct}")/f"experimento={mode}"/"loss=mse"/"lambda=00"
                   /f"modelo={model}"/f"lookback={lb}"/f"pred_len={h}")
            path = root/rel/"part.parquet"
            schema = pq.ParquetFile(path).schema_arrow.names
            if set(schema) != {"arquivo", "h", "origin_step", "step", *market.tickers}:
                raise ValueError(f"Schema/universo inesperado em {path}")
            nrows = pq.ParquetFile(path).metadata.num_rows
            cut = market.n*pct//100
            if cut < 1 or market.n-cut < h or nrows != (market.n-cut-h+1)*h:
                raise ValueError(f"Numero de linhas/janelas incorreto em {path}")
            tasks.append({"id": len(tasks), "treino": pct, "experimento": mode, "modelo": model,
                          "lookback": lb, "pred_len": h, "loss": "mse", "lambda": "00",
                          "corte": cut, "relpath": str(rel), "path": str(path),
                          "sha256": expected_hash, "size": path.stat().st_size})
    for pct in percentages:
        # Todos os modelos/L/H do corte usam o mesmo limite de ORIGENS.
        hmax = max(t["pred_len"] for t in tasks if t["treino"] == pct)
        for t in tasks:
            if t["treino"] == pct:
                t["common_last_origin"] = market.n-hmax
        var = max((t for t in tasks if t["treino"] == pct and t["experimento"] == "var"),
                  key=lambda t: t["pred_len"])
        for t in tasks:
            if t["treino"] == pct:
                # As seis saidas VAR sao prefixos do mesmo ajuste/previsao.
                # Carteiras usam o maior H uma unica vez; IC preserva cada cobertura.
                source = var if t["experimento"] == "var" else t
                t["backtest_source_id"] = source["id"]
    return tasks, provenance


def load_predictions(task, market, verify_hash=True):
    path = Path(task["path"])
    if verify_hash and digest(path) != task["sha256"]:
        raise ValueError(f"Checksum de previsoes divergente: {path}")
    frame = read_table(path).sort_values(["origin_step", "h"])
    h = task["pred_len"]
    origins = np.arange(task["corte"], market.n-h+1)
    if len(frame) != len(origins)*h:
        raise ValueError(f"Janelas incompletas em {path}")
    expected_origin = np.repeat(origins, h)
    expected_h = np.tile(np.arange(1, h+1), len(origins))
    for col, expected in [("origin_step", expected_origin), ("h", expected_h),
                          ("step", expected_origin+expected_h)]:
        if not np.array_equal(integers(frame[col], col), expected):
            raise ValueError(f"{col} desalinhado/duplicado/ausente em {path}")
    if frame.arquivo.isna().any():
        raise ValueError(f"arquivo ausente em {path}")
    names = frame.arquivo.to_numpy().reshape(-1, h)
    if not (names == names[:, :1]).all() or len(np.unique(names[:, 0])) != len(origins):
        raise ValueError(f"Identificadores de janela duplicados/inconsistentes em {path}")
    values = frame[market.tickers.tolist()].to_numpy(dtype=float).reshape(len(origins), h, -1)
    # O exportador TFB exige finitos; nao aceitar silenciosamente uma exportacao corrompida.
    if not np.isfinite(values).all():
        raise ValueError(f"Previsoes nao finitas em {path}")
    return origins, values, np.cumsum(values, axis=1)
