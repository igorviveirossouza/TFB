"""Descoberta incremental, somente leitura, do contrato B3 composto.

O registro de conclusoes e o ponto de publicacao: um part.parquet isolado
nunca entra no plano. O planejamento le CSV/JSON e rodapes Parquet, sem
materializar previsoes nem reler todos os seus bytes. O SHA256 registrado
e conferido integralmente quando a tarefa carrega suas previsoes.
"""
import csv
import hashlib
import io
import itertools
import json
import math
from numbers import Real
from pathlib import Path
import re

import pyarrow as pa
import pyarrow.parquet as pq

from backtest_cpu.io import load_predictions


_HASH = re.compile(r"[0-9a-fA-F]{64}")
_NAME = re.compile(r"[A-Za-z0-9_]+")
_PROXIMITY = {"hinge_proximity", "ranknet_proximity", "pairwise_mse_proximity"}
_LOSSES = {"ranknet", "hinge", "listnet", "pairwise_mse"} | _PROXIMITY


def _cross_mask(cfg):
    """Mirror the producer's mask contract in the self-contained CPU snapshot."""
    value = cfg["loss"].get("loss_cross_mask")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("loss_cross_mask deve ser [tipo, a].")
    name, bandwidth = value
    if not isinstance(name, str) or name.strip().lower() not in {"uniform", "gaussian"}:
        raise ValueError("Tipo de loss_cross_mask invalido.")
    name = name.strip().lower()
    if isinstance(bandwidth, bool) or not isinstance(bandwidth, Real):
        raise ValueError("a da mascara deve ser numerico e finito.")
    bandwidth = _number(bandwidth, "mask_a")
    if bandwidth < 0 or (name == "gaussian" and bandwidth == 0):
        raise ValueError("Mascara exige a >= 0 (uniform) ou a > 0 (gaussian).")
    return name, bandwidth


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_snapshot(path):
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8-sig")), _sha(raw)


def _csv_snapshot(path):
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    fields = reader.fieldnames
    if not fields or len(fields) != len(set(fields)):
        raise ValueError(f"Cabecalho ausente/duplicado: {path}")
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Linha incompleta ou colunas extras: {path}")
    return fields, rows, _sha(raw)


def _number(value, label):
    if isinstance(value, bool):
        raise ValueError(f"Numero invalido em {label}")
    try:
        number = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Numero invalido em {label}") from error
    if not math.isfinite(number):
        raise ValueError(f"Numero nao finito em {label}")
    return number


def _int(value, label):
    if isinstance(value, bool):
        raise ValueError(f"Inteiro invalido em {label}")
    try:
        integer = int(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Inteiro invalido em {label}") from error
    if integer != _number(value, label):
        raise ValueError(f"Inteiro invalido em {label}")
    return integer


def _vector(config, name):
    values = config.get(name)
    if not isinstance(values, list) or not values:
        raise ValueError(f"Configuracao: {name} vazio/ausente.")
    try:
        unique = len(set(values)) == len(values)
    except TypeError as error:
        raise ValueError(f"Configuracao: {name} invalido.") from error
    if not unique:
        raise ValueError(f"Configuracao: {name} duplicado.")
    return values


def _validate_config(plan, pct, market, source_sha256, path):
    cfg, meta = plan["config"], plan["metadata"]
    if (cfg.get("source_data_kind") != "log_return"
            or meta.get("source_data_kind") != "log_return" or meta.get("step_offset") != 1
            or meta.get("n_rows") != market.n or meta.get("n_assets") != len(market.tickers)
            or not isinstance(meta.get("targets"), list)
            or sorted(meta["targets"]) != market.tickers.tolist()
            or meta.get("source_sha256") != source_sha256):
        raise ValueError(f"Fonte, universo ou offset incompatibilidade em {path}")
    for field, value in (("design_sha256", plan.get("design_sha256")),
                         ("dataset_sha256", meta.get("dataset_sha256"))):
        if not isinstance(value, str) or not _HASH.fullmatch(value):
            raise ValueError(f"Hash de proveniencia invalido: {field} em {path}")
    percentages = _vector(cfg, "train_percentages")
    if (any(type(p) is not int or not 0 < p < 100 for p in percentages) or pct not in percentages
            or cfg["strategy"].get("stride") != 1):
        raise ValueError(f"Corte/stride invalido em {path}")
    ratio = _number(cfg["strategy"].get("train_ratio_in_tv"), "train_ratio_in_tv")
    if not 0 < ratio < 1:
        raise ValueError(f"Validacao interna invalida em {path}")
    for name in ("lookbacks", "horizons", "trade_windows"):
        if any(type(x) is not int or x < 1 for x in _vector(cfg, name)):
            raise ValueError(f"Configuracao {name}: inteiros positivos exigidos.")
    if any(not isinstance(m, str) or not _NAME.fullmatch(m) for m in _vector(cfg, "models")):
        raise ValueError("Nomes de modelos invalidos.")
    if set(_vector(cfg, "cross_losses")) - _LOSSES:
        raise ValueError("Loss composta nao suportada no contrato B3.")
    lambdas = [_number(x, "lambdas") for x in _vector(cfg, "lambdas")]
    deltas = [_number(x, "deltas") for x in _vector(cfg, "deltas")]
    if any(not 0 < x < 1 for x in lambdas) or any(x < 0 for x in deltas):
        raise ValueError("Lambda/delta fora do dominio composto.")
    # O produtor usa :g nos caminhos. Valores distintos que colidam ali sao ambiguos.
    for name, values in (("lambda", lambdas), ("delta", deltas)):
        if len({format(x, "g") for x in values}) != len(values):
            raise ValueError(f"Valores de {name} colidem no caminho canonico.")
    if not all(m in cfg["model_defaults"] for m in cfg["models"]):
        raise ValueError("Modelo sem arquitetura registrada.")
    for required in ("training", "loss"):
        if not isinstance(cfg.get(required), dict):
            raise ValueError(f"Configuracao {required} ausente.")
    if set(cfg["cross_losses"]) & _PROXIMITY:
        _cross_mask(cfg)
    cut = market.n*pct//100
    hmax = max(cfg["horizons"])
    if cut < 1 or cut+hmax > market.n:
        raise ValueError("Corte sem observacoes suficientes para o maior H declarado.")
    return cfg, meta, cut, hmax


def _expected_tasks(cfg, pct):
    """Mesma ordem/identidade de b3_experiment.make_tasks; nenhuma dependencia GPU."""
    definitions = []
    for loss in cfg["cross_losses"]:
        mask = _cross_mask(cfg) if loss in _PROXIMITY else None
        deltas = [0] if loss == "listnet" or mask is not None else cfg["deltas"]
        for lam, delta, model, lookback, horizon in itertools.product(
                cfg["lambdas"], deltas, cfg["models"], cfg["lookbacks"], cfg["horizons"]):
            for k_dir in cfg["trade_windows"]:
                if k_dir <= horizon and horizon % k_dir == 0:
                    definitions.append({"treino": pct, "experimento": "composto", "modelo": model,
                                        "lookback": lookback, "pred_len": horizon, "loss": loss,
                                        "cross_lambda": float(lam), "cross_delta": float(delta),
                                        "lambda": format(lam, "g"), "delta": format(delta, "g"),
                                        "k_dir": k_dir, "delta_effective": loss != "listnet" and mask is None})
                    if mask is not None:
                        definitions[-1].update(mask_type=mask[0], mask_a=mask[1])
    if not definitions:
        raise ValueError("Grade composta sem pares H/K de treino elegiveis.")
    offset = cfg["train_percentages"].index(pct)*len(definitions)
    for i, task in enumerate(definitions):
        task["training_array_index"] = offset+i
        path = (Path(f"treino={pct}")/"experimento=composto"/f"loss={task['loss']}"
                /f"lambda={task['lambda']}"/f"delta={task['delta']}")
        if task["loss"] in _PROXIMITY:
            path = path/f"mask={task['mask_type']}"/f"a={task['mask_a']!r}"
        task["relpath"] = str(path/f"modelo={task['modelo']}"/f"lookback={task['lookback']}"
                              /f"pred_len={task['pred_len']}"/f"k_dir={task['k_dir']}")
    if len({t["relpath"] for t in definitions}) != len(definitions):
        raise ValueError("Grade composta produz caminhos duplicados.")
    return definitions


_REQUIRED_ROW = {"array_index", "treino", "experimento", "loss", "cross_lambda", "cross_delta",
                 "modelo", "lookback", "pred_len", "k", "n_rows", "n_pretest", "n_train",
                 "n_validation", "n_test", "n_windows", "dataset", "data_source", "source_data_kind",
                 "step_offset", "source_sha256", "dataset_sha256", "n_assets", "seed", "temporal_loss",
                 "delta_effective", "cross_blocks", "cross_steps", "cross_tail_steps", "forecast_horizons",
                 "n_windows_by_horizon", "output_dirs", "model_hyper_params", "strategy_args",
                 "deterministic", "design_sha256", "output_dir", "parquet_sha256"}


def _validate_row(row, task, cfg, meta, plan, market, cut):
    identity = task["relpath"]
    expected_int = {"array_index": task["training_array_index"], "treino": task["treino"],
                    "lookback": task["lookback"], "pred_len": task["pred_len"], "k": task["k_dir"],
                    "n_rows": market.n, "n_pretest": cut, "n_train": int(cut*cfg["strategy"]["train_ratio_in_tv"]),
                    "n_test": market.n-cut, "n_windows": market.n-cut-task["pred_len"]+1,
                    "step_offset": 1, "n_assets": len(market.tickers), "seed": cfg["strategy"]["seed"],
                    "cross_blocks": task["pred_len"]//task["k_dir"], "cross_steps": task["pred_len"],
                    "cross_tail_steps": 0}
    expected_int["n_validation"] = cut-expected_int["n_train"]
    if any(_int(row[key], key) != value for key, value in expected_int.items()):
        raise ValueError(f"Dimensoes/corte/identidade inconsistentes: {identity}")
    expected_text = {"experimento": "composto", "dataset": "log_retornos", "source_data_kind": "log_return",
                     "modelo": task["modelo"], "loss": task["loss"], "data_source": meta["source"],
                     "source_sha256": meta["source_sha256"], "dataset_sha256": meta["dataset_sha256"],
                     "design_sha256": plan["design_sha256"], "temporal_loss": cfg["loss"]["loss_temporal"],
                     "deterministic": "efficient" if task["modelo"] == "TimesNet" else "full"}
    if any(row[key] != value for key, value in expected_text.items()):
        raise ValueError(f"Proveniencia inconsistente: {identity}")
    for key in ("cross_lambda", "cross_delta"):
        if _number(row[key], key) != task[key]:
            raise ValueError(f"Lambda/delta divergente: {identity}")
    if row["delta_effective"].lower() != str(task["delta_effective"]).lower():
        raise ValueError(f"delta_effective inconsistente: {identity}")
    if task["loss"] in _PROXIMITY:
        if (row.get("mask_type") != task["mask_type"]
                or _number(row.get("mask_a"), "mask_a") != task["mask_a"]
                or row.get("mask_effective", "").lower() != "true"):
            raise ValueError(f"Mascara divergente na auditoria: {identity}")
    elif (row.get("mask_type", "") or row.get("mask_a", "")
          or row.get("mask_effective", "false").lower() != "false"):
        raise ValueError(f"Mascara ativa para loss original: {identity}")
    horizon = task["pred_len"]
    if (json.loads(row["forecast_horizons"]) != [horizon]
            or json.loads(row["n_windows_by_horizon"]) != {str(horizon): expected_int["n_windows"]}):
        raise ValueError(f"Horizontes/janelas divergentes na auditoria: {identity}")
    # O caminho efetivamente aberto e construido da grade; nunca vem do CSV.
    # A raiz pode ter sido movida, mas o sufixo completo deve continuar canonico.
    declared = Path(row["output_dir"])
    suffix = Path(identity).parts
    if (tuple(declared.parts[-len(suffix):]) != suffix
            or json.loads(row["output_dirs"]) != {str(horizon): str(declared)}):
        raise ValueError(f"Caminho declarado divergente da identidade: {identity}")
    hyper = dict(cfg["model_defaults"][task["modelo"]], **cfg["training"])
    hyper.update(seq_len=task["lookback"], pred_len=horizon, horizon=horizon)
    hyper.update(cfg["loss"])
    hyper.pop("loss_cross_mask", None)
    if task["loss"] in _PROXIMITY:
        hyper["loss_cross_mask"] = [task["mask_type"], task["mask_a"]]
    hyper.update(loss="composite_trading", loss_cross=task["loss"], loss_trade_window=task["k_dir"],
                 loss_cross_lambda=task["cross_lambda"], loss_cross_delta=task["cross_delta"])
    strategy = dict(cfg["strategy"])
    ratio = task["treino"]/100
    if int(ratio*market.n) != cut:
        ratio = (cut+.5)/market.n
    strategy.update(horizon=horizon, tv_ratio=ratio)
    if json.loads(row["model_hyper_params"]) != hyper or json.loads(row["strategy_args"]) != strategy:
        raise ValueError(f"Hiperparametros/estrategia divergentes do desenho: {identity}")
    if not _HASH.fullmatch(row["parquet_sha256"]):
        raise ValueError(f"SHA256 de previsao ausente/invalido: {identity}")


def _validate_footer(path, task, market):
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    names = {"arquivo", "h", "origin_step", "step", *market.tickers}
    if len(schema.names) != len(names) or set(schema.names) != names:
        raise ValueError(f"Schema/universo inesperado: {path}")
    for name in ("h", "origin_step", "step"):
        if not pa.types.is_integer(schema.field(name).type):
            raise ValueError(f"Indice nao inteiro no schema: {name} em {path}")
    if not (pa.types.is_string(schema.field("arquivo").type) or pa.types.is_large_string(schema.field("arquivo").type)):
        raise ValueError(f"Identificador de janela nao textual: {path}")
    for ticker in market.tickers:
        kind = schema.field(ticker).type
        if not (pa.types.is_integer(kind) or pa.types.is_floating(kind)):
            raise ValueError(f"Previsoes nao numericas no schema: {ticker} em {path}")
    h = task["pred_len"]
    expected = (market.n-task["corte"]-h+1)*h
    if expected < 1 or parquet.metadata.num_rows != expected:
        raise ValueError(f"Numero de linhas/janelas incorreto: {path}")


def discover_composite_tasks(root, percentages, market, source_sha256):
    """Retorna (tarefas sem id, metadados da descoberta); aceita grade parcial.

    K de treinamento e ``k_dir``. O K de avaliacao pertence ao pipeline e
    nao e inferido desta coluna. ``common_last_origin`` usa TODOS os horizontes
    declarados, inclusive os ainda nao concluidos. Grupos sem configuracao
    ficam explicitamente desconhecidos; nunca recebem contagens inventadas.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Raiz de previsoes ausente: {root}")
    if not isinstance(source_sha256, str) or not _HASH.fullmatch(source_sha256):
        raise ValueError("SHA256 do painel de retornos invalido.")
    if percentages is None:
        percentages = sorted(int(m.group(1)) for p in root.iterdir()
                             if p.is_dir() and (m := re.fullmatch(r"treino=([1-9][0-9]?)", p.name)))
    percentages = list(percentages)
    if (not percentages or any(type(p) is not int or not 0 < p < 100 for p in percentages)
            or len(set(percentages)) != len(percentages)):
        raise ValueError("Cortes de treino vazios/invalidos/duplicados.")
    discovery = {"root": str(root), "requested_train": sorted(percentages), "manifests": {}, "groups": []}
    tasks = []
    for pct in sorted(percentages):
        group = root/f"treino={pct}"/"experimento=composto"
        config_path, done_path = group/"run_config.json", group/"completed_tasks.csv"
        info = {"treino": pct, "status": "MISSING_CONFIG", "design_sha256": None,
                "hmax": None, "common_last_origin": None, "expected": None,
                "registered": 0, "available": 0, "awaiting_registration": None,
                "registered_missing": 0, "awaiting": None, "missing_paths": []}
        discovery["groups"].append(info)
        if not config_path.is_file():
            continue
        plan, sha = _json_snapshot(config_path)
        discovery["manifests"][str(config_path)] = sha
        cfg, meta, cut, hmax = _validate_config(plan, pct, market, source_sha256, config_path)
        expected = _expected_tasks(cfg, pct)
        indexed = {t["training_array_index"]: t for t in expected}
        info.update(status="MISSING_REGISTRY", design_sha256=plan["design_sha256"], hmax=hmax,
                    common_last_origin=market.n-hmax, expected=len(expected),
                    awaiting_registration=len(expected), awaiting=len(expected))
        if not done_path.is_file():
            continue
        fields, rows, sha = _csv_snapshot(done_path)
        discovery["manifests"][str(done_path)] = sha
        if not _REQUIRED_ROW <= set(fields):
            raise ValueError(f"Registro composto sem colunas obrigatorias: {sorted(_REQUIRED_ROW-set(fields))}")
        seen = set()
        for row in rows:
            array_index = _int(row["array_index"], "array_index")
            if array_index in seen:
                raise ValueError(f"Conclusao duplicada: array_index={array_index} em {done_path}")
            seen.add(array_index)
            if array_index not in indexed:
                raise ValueError(f"Conclusao fora da grade/corte: array_index={array_index} em {done_path}")
            task = dict(indexed[array_index])
            _validate_row(row, task, cfg, meta, plan, market, cut)
            path = root/task["relpath"]/"part.parquet"
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"Previsao resolve fora da raiz declarada: {path}")
            if not path.is_file():
                info["missing_paths"].append(str(path))
                continue
            task.update(corte=cut, common_last_origin=market.n-hmax, path=str(path),
                        sha256=row["parquet_sha256"].lower(), size=path.stat().st_size,
                        design_sha256=plan["design_sha256"], source_sha256=source_sha256,
                        dataset_sha256=meta["dataset_sha256"], seed=cfg["strategy"]["seed"],
                        temporal_loss=cfg["loss"]["loss_temporal"])
            _validate_footer(path, task, market)
            tasks.append(task)
            info["available"] += 1
        info.update(status="READY", registered=len(rows), registered_missing=len(info["missing_paths"]),
                    awaiting_registration=len(expected)-len(rows), awaiting=len(expected)-info["available"])
    for name in ("expected", "registered", "available", "awaiting_registration", "registered_missing", "awaiting"):
        discovery[name] = sum(g[name] or 0 for g in discovery["groups"])
    discovery["missing_groups"] = [g["treino"] for g in discovery["groups"] if g["expected"] is None]
    discovery["expected_count_known"] = not discovery["missing_groups"]
    discovery["expected_complete"] = discovery["expected_count_known"] and discovery["awaiting"] == 0
    tasks.sort(key=lambda t: (t["treino"], t["training_array_index"]))
    return tasks, discovery


def load_composite_predictions(task, market, verify_hash=True):
    """Materializa apenas uma tarefa; hash real/alinhamento/finitude pelo motor comum."""
    if (task.get("experimento") != "composto" or type(task.get("k_dir")) is not int
            or task["k_dir"] < 1 or task["pred_len"] % task["k_dir"] != 0
            or not 1 <= task["corte"] <= task["common_last_origin"] <= market.n-task["pred_len"]):
        raise ValueError("Identidade/origens/K de treino invalidos na tarefa composta.")
    return load_predictions(task, market, verify_hash=verify_hash)
