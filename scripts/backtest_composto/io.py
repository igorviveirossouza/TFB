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

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from backtest_cpu.io import digest, load_predictions


_HASH = re.compile(r"[0-9a-fA-F]{64}")
_NAME = re.compile(r"[A-Za-z0-9_]+")
_PROXIMITY = {"hinge_proximity", "ranknet_proximity", "pairwise_mse_proximity"}
_LOSSES = {"ranknet", "hinge", "listnet", "pairwise_mse"} | _PROXIMITY
_MOMENTUM_KEYS = ("momentum_mode", "momentum_lookback", "momentum_eta")


def _momentum_options(cfg):
    loss = cfg["loss"]
    if {key for key in loss if key.startswith("momentum_")} - set(_MOMENTUM_KEYS):
        raise ValueError("Parametro momentum desconhecido no desenho.")
    mode, lb, eta = (loss.get("momentum_mode", "off"), loss.get("momentum_lookback"),
                     loss.get("momentum_eta", 0.0))
    if mode not in {"off", "residual_output"}:
        raise ValueError("momentum_mode invalido.")
    if lb is not None and (type(lb) is not int or lb < 1):
        raise ValueError("momentum_lookback deve ser inteiro positivo.")
    if type(eta) not in (int, float) or not math.isfinite(eta) or eta < 0:
        raise ValueError("momentum_eta deve ser finito e nao negativo.")
    return mode, lb, float(eta)


# Mirror kept self-contained because CPU snapshots do not import training code.
def _validate_cross_mask(value) -> tuple:
    """Validate aliases or [family, A, mode, *shape] without torch/numpy.

    Legacy natural aliases retain two fields, preserving old hyperparameters,
    paths and cache keys. Parameterized families always retain explicit mode.
    A is measured in original realized financial-score units, before z-scoring.
    """
    aliases = {"uniform", "gaussian", "cauchy", "cosine", "laplace",
               "triangular", "epanechnikov", "biweight", "triweight"}
    arity = {"student": 1, "generalized_gaussian": 1,
             "compact": 2, "intermediate_band": 1}
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError("loss_cross_mask exige [tipo, a, modo opcional, parametros de forma].")
    name = value[0]
    if not isinstance(name, str) or name.strip().lower() not in aliases | set(arity):
        raise ValueError("Tipo de loss_cross_mask invalido.")
    name = name.strip().lower()
    required = 3 + arity[name] if name in arity else None
    if (required is not None and len(value) != required) or (required is None and len(value) not in (2, 3)):
        raise ValueError("Numero de parametros de loss_cross_mask invalido para " + name)
    mode = value[2] if len(value) >= 3 else "natural"
    if not isinstance(mode, str) or mode.strip().lower() not in {"natural", "complementary"}:
        raise ValueError("O modo de loss_cross_mask deve ser 'natural' ou 'complementary'.")
    mode = mode.strip().lower()

    def number(raw, label):
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise ValueError(label + " de loss_cross_mask deve ser numerico e finito.")
        try:
            result = float(raw)
        except (ValueError, OverflowError) as exc:
            raise ValueError(label + " de loss_cross_mask deve ser finito.") from exc
        if not math.isfinite(result):
            raise ValueError(label + " de loss_cross_mask deve ser finito.")
        return result

    bandwidth = number(value[1], "Parametro a")
    shape = tuple(number(raw, "Parametro de forma") for raw in value[3:])
    if shape:
        if shape[0] <= 0 or (name == "compact" and shape[1] < 0):
            raise ValueError("Parametros de forma de loss_cross_mask exigem nu/p > 0 e q >= 0.")
    zero_ok = name == "uniform" or (name == "compact" and shape[1] == 0)
    if bandwidth < 0 or (bandwidth == 0 and not zero_ok):
        raise ValueError("loss_cross_mask exige a > 0, salvo uniforme (a >= 0).")
    if name in arity:
        return (name, bandwidth, mode, *shape)
    return (name, bandwidth) if mode == "natural" else (name, bandwidth, mode)

def _cross_mask(cfg):
    return _validate_cross_mask(cfg["loss"].get("loss_cross_mask"))


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
    if any(not 0 < x <= 1 for x in lambdas) or any(x < 0 for x in deltas):
        raise ValueError("Lambda deve estar em (0,1] e delta deve ser nao negativo.")
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
    _momentum_options(cfg)
    cut = market.n*pct//100
    hmax = max(cfg["horizons"])
    if cut < 1 or cut+hmax > market.n:
        raise ValueError("Corte sem observacoes suficientes para o maior H declarado.")
    return cfg, meta, cut, hmax


def _expected_tasks(cfg, pct):
    """Mesma ordem/identidade de b3_experiment.make_tasks; nenhuma dependencia GPU."""
    definitions = []
    momentum_mode, momentum_lb, momentum_eta = _momentum_options(cfg)
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
                    if momentum_mode != "off":
                        definitions[-1].update(momentum_mode=momentum_mode,
                                               momentum_lookback=momentum_lb or lookback,
                                               momentum_eta=momentum_eta)
                    if mask is not None:
                        definitions[-1].update(mask_type=mask[0], mask_a=mask[1])
                        if len(mask) >= 3 and mask[2] != "natural":
                            definitions[-1]["mask_mode"] = mask[2]
                        if len(mask) > 3:
                            definitions[-1]["mask_shape"] = list(mask[3:])
    if not definitions:
        raise ValueError("Grade composta sem pares H/K de treino elegiveis.")
    offset = cfg["train_percentages"].index(pct)*len(definitions)
    for i, task in enumerate(definitions):
        task["training_array_index"] = offset+i
        path = (Path(f"treino={pct}")/"experimento=composto"/f"loss={task['loss']}"
                /f"lambda={task['lambda']}"/f"delta={task['delta']}")
        if task["loss"] in _PROXIMITY:
            path = path/f"mask={task['mask_type']}"/f"a={task['mask_a']!r}"
            if "mask_mode" in task:
                path = path/f"mask_mode={task['mask_mode']}"
            if task.get("mask_shape"):
                path = path/("shape=" + "_".join(repr(float(x)) for x in task["mask_shape"]))
        if task.get("momentum_mode") == "residual_output":
            path = (path/"momentum=residual_output"/f"momentum_lb={task['momentum_lookback']}"
                    /f"momentum_eta={float(task['momentum_eta'])!r}")
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
    if task.get("momentum_mode") == "residual_output":
        if (row.get("momentum_mode") != task["momentum_mode"]
                or _int(row.get("momentum_lookback"), "momentum_lookback") != task["momentum_lookback"]
                or _number(row.get("momentum_eta"), "momentum_eta") != task["momentum_eta"]):
            raise ValueError(f"Momentum divergente na auditoria: {identity}")
    elif any(row.get(key, "") for key in _MOMENTUM_KEYS):
        raise ValueError(f"Momentum ativo fora do desenho: {identity}")
    if row["delta_effective"].lower() != str(task["delta_effective"]).lower():
        raise ValueError(f"delta_effective inconsistente: {identity}")
    if task["loss"] in _PROXIMITY:
        if (row.get("mask_type") != task["mask_type"]
                or _number(row.get("mask_a"), "mask_a") != task["mask_a"]
                or row.get("mask_mode", "natural") != task.get("mask_mode", "natural")
                or row.get("mask_shape", "") != (json.dumps(task["mask_shape"], separators=(",", ":")) if task.get("mask_shape") else "")
                or row.get("mask_effective", "").lower() != "true"):
            raise ValueError(f"Mascara divergente na auditoria: {identity}")
    elif (row.get("mask_type", "") or row.get("mask_a", "") or row.get("mask_mode", "") or row.get("mask_shape", "")
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
    for key in _MOMENTUM_KEYS:
        hyper.pop(key, None)
    if task.get("momentum_mode") == "residual_output":
        hyper.update({key: task[key] for key in _MOMENTUM_KEYS})
    hyper.pop("loss_cross_mask", None)
    if task["loss"] in _PROXIMITY:
        hyper["loss_cross_mask"] = [task["mask_type"], task["mask_a"]]
        if "mask_mode" in task or task.get("mask_shape"):
            hyper["loss_cross_mask"].append(task.get("mask_mode", "natural"))
        hyper["loss_cross_mask"].extend(task.get("mask_shape", []))
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
    unpublished_candidates = {}
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
        unpublished_candidates[pct] = {t["training_array_index"]: t["relpath"] for t in expected}
        info.update(losses=list(cfg["cross_losses"]),
                    cross_mask=cfg["loss"].get("loss_cross_mask")
                    if set(cfg["cross_losses"]) & _PROXIMITY else None)
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
            unpublished_candidates[pct].pop(array_index, None)
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
            if task.get("momentum_mode") == "residual_output":
                task.update(momentum_snapshot(task, market))
            tasks.append(task)
            info["available"] += 1
        info.update(status="READY", registered=len(rows), registered_missing=len(info["missing_paths"]),
                    awaiting_registration=len(expected)-len(rows), awaiting=len(expected)-info["available"])
    # Only an empty group needs this extra metadata scan. Do not load matrices,
    # hash files, or publish an orphan: registration remains the commit point.
    for info in discovery["groups"]:
        if info["available"] == 0 and info["expected"] is not None:
            orphans = [str(root/path/"part.parquet")
                       for path in unpublished_candidates[info["treino"]].values()
                       if (root/path/"part.parquet").is_file()]
            info["unregistered_parquets"] = len(orphans)
            info["unregistered_examples"] = orphans[:3]
    for name in ("expected", "registered", "available", "awaiting_registration", "registered_missing", "awaiting"):
        discovery[name] = sum(g[name] or 0 for g in discovery["groups"])
    discovery["missing_groups"] = [g["treino"] for g in discovery["groups"] if g["expected"] is None]
    discovery["expected_count_known"] = not discovery["missing_groups"]
    discovery["expected_complete"] = discovery["expected_count_known"] and discovery["awaiting"] == 0
    tasks.sort(key=lambda t: (t["treino"], t["training_array_index"]))
    return tasks, discovery


def momentum_snapshot(task, market, verify_files=False):
    """Validate/bind optional sidecars without importing the training package.

    Planning reads JSON and Parquet footers only. Workers verify every file hash
    before using scores or reusing an existing result.
    """
    if task.get("momentum_mode", "off") == "off":
        return {}
    if task.get("momentum_mode") != "residual_output":
        raise ValueError("Modo momentum desconhecido na tarefa.")
    folder = Path(task["path"]).parent
    try:
        manifest, manifest_hash = _json_snapshot(folder/"momentum_scores.json")
        state, state_hash = _json_snapshot(folder/"momentum_state.json")
    except (OSError, ValueError) as error:
        raise ValueError(f"Sidecars momentum ausentes/invalidos em {folder}: {error}") from error
    h, cut = task["pred_len"], task["corte"]
    n_windows = market.n-cut-h+1
    fields = {"schema_version": 1, "version": "momentum_residual_v1", "mode": "residual_output",
              "n_windows": n_windows, "n_rows": n_windows*h, "n_assets": len(market.tickers),
              "horizon": h, "k_values": list(range(1, h+1)), "first_origin_step": cut,
              "last_origin_step": market.n-h, "step_offset": 1, "state": state}
    if any(manifest.get(key) != value for key, value in fields.items()):
        raise ValueError("Manifest momentum incompativel com janelas/estado da tarefa.")
    if not isinstance(manifest.get("input_panel_sha256"), str) or not _HASH.fullmatch(manifest["input_panel_sha256"]):
        raise ValueError("Hash do painel momentum ausente/invalido.")
    if (state.get("version") != "momentum_residual_v1" or state.get("mode") != "residual_output"
            or state.get("normalization") != "cross_sectional_zscore_ddof0"
            or type(state.get("lookback")) is not int or state["lookback"] != task["momentum_lookback"]
            or not 1 <= state["lookback"] <= cut
            or _number(state.get("eta"), "momentum eta") != task["momentum_eta"]
            or state.get("score_kind") not in {"log_return", "simple_return"}
            or _number(state.get("eps"), "momentum eps") <= 0
            or not isinstance(state.get("asset_names"), list)
            or sorted(state["asset_names"]) != market.tickers.tolist()):
        raise ValueError("Estado momentum incompativel com desenho/universo.")
    _number(state.get("gain"), "momentum gain")
    files = manifest.get("files", {})
    if set(files) != {"predictions", "state", "scores"}:
        raise ValueError("Manifest momentum exige hashes de previsoes, estado e scores.")
    expected_names = {"predictions": "part.parquet", "state": "momentum_state.json",
                      "scores": "ranking_scores.parquet"}
    frozen = {"manifest": {"path": str(folder/"momentum_scores.json"), "sha256": manifest_hash}}
    for name, filename in expected_names.items():
        item = files[name]
        if (not isinstance(item, dict) or item.get("name") != filename
                or not isinstance(item.get("sha256"), str) or not _HASH.fullmatch(item["sha256"])):
            raise ValueError(f"Arquivo/hash momentum invalido: {name}.")
        path = folder/filename
        if not path.is_file():
            raise ValueError(f"Arquivo momentum ausente: {path}")
        sha = item["sha256"].lower()
        if name == "predictions" and sha != task["sha256"]:
            raise ValueError("Manifest momentum refere-se a outra previsao.")
        if name == "state" and sha != state_hash:
            raise ValueError("Estado momentum alterado.")
        if verify_files and digest(path) != sha:
            raise ValueError(f"Arquivo momentum alterado: {path}")
        frozen[name] = {"path": str(path), "sha256": sha}
    parquet = pq.ParquetFile(folder/"ranking_scores.parquet")
    schema = parquet.schema_arrow
    if (schema.names != state["asset_names"]+["origin_step", "k_eval"]
            or parquet.metadata.num_rows != n_windows*h
            or any(not pa.types.is_integer(schema.field(x).type) for x in ("origin_step", "k_eval"))
            or any(not (pa.types.is_integer(schema.field(x).type) or pa.types.is_floating(schema.field(x).type))
                   for x in state["asset_names"])):
        raise ValueError("Schema/dimensoes de ranking_scores momentum invalidos.")
    result = {"momentum_files": frozen, "momentum_state": state}
    if "momentum_files" in task and any(task.get(key) != value for key, value in result.items()):
        raise ValueError("Sidecars momentum mudaram apos o planejamento.")
    return result


def load_composite_ranking_scores(task, market, origins, verify_hash=True):
    """Final scores [origins, K, assets], or None for the exact legacy path."""
    if task.get("momentum_mode", "off") == "off":
        return None
    momentum_snapshot(task, market, verify_files=verify_hash)
    table = pq.ParquetFile(Path(task["path"]).parent/"ranking_scores.parquet").read().to_pandas()
    h = task["pred_len"]
    if (not np.array_equal(table["origin_step"], np.repeat(origins, h))
            or not np.array_equal(table["k_eval"], np.tile(np.arange(1, h+1), len(origins)))):
        raise ValueError("Origens/K dos scores momentum desalinhados das previsoes.")
    values = table.loc[:, market.tickers.tolist()].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Scores momentum nao finitos.")
    return values.reshape(len(origins), h, len(market.tickers))


def load_composite_predictions(task, market, verify_hash=True):
    """Materializa apenas uma tarefa; hash real/alinhamento/finitude pelo motor comum."""
    if (task.get("experimento") != "composto" or type(task.get("k_dir")) is not int
            or task["k_dir"] < 1 or task["pred_len"] % task["k_dir"] != 0
            or not 1 <= task["corte"] <= task["common_last_origin"] <= market.n-task["pred_len"]):
        raise ValueError("Identidade/origens/K de treino invalidos na tarefa composta.")
    return load_predictions(task, market, verify_hash=verify_hash)
