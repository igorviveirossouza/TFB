#!/usr/bin/env python3
"""Converte resultados TFB do experimento de alvo acumulado para janelas do clean.

As previsões deste experimento já são trajetórias acumuladas [H,N]. Portanto a
conversão apenas preserva os valores e a indexação temporal; NÃO acumula novamente
os horizontes. No backtest, a janela K usa diretamente a linha h=K.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

CUMULATIVE_LOSSES = [
    "mse_accum",
    "mse_score",
    "rank_hinge",
    "rank_margin",
    "rank_bpr",
    "ranknet",
    "whr1",
    "whr2",
    "listnet",
    "fingat",
    "huber",
    "mae",
    "mse",
]


def load_clean_converter(clean_root: Path):
    script_path = clean_root / "utils" / "convert_tfb_custom_loss_results.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Conversor do clean não encontrado: {script_path}")

    spec = importlib.util.spec_from_file_location(
        "clean_tfb_converter_cumulative_base", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Não foi possível carregar {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    loss_pattern = "|".join(
        re.escape(x) for x in sorted(CUMULATIVE_LOSSES, key=len, reverse=True)
    )
    module.LOSS_NAMES = list(CUMULATIVE_LOSSES)
    module.LOSS_PATTERN = loss_pattern
    module.RUN_RE = re.compile(
        rf"^(?P<model>.+?)_(?P<loss>{loss_pattern})_"
        rf"(?P<data_kind>.+?)_lb(?P<seq_len>\d+)_h(?P<horizon>\d+)_"
        rf"k(?P<loss_k>\d+)_seed(?P<seed>\d+)$"
    )
    return module


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def enrich_converted_metadata(run_dir: Path, converted_dir: Path) -> None:
    source_manifest = run_dir / "run_manifest.json"
    target_metadata = converted_dir / "metadata.json"
    if not target_metadata.exists():
        return

    metadata = read_json(target_metadata)
    run_manifest = read_json(source_manifest) if source_manifest.exists() else {}
    metadata["target_mode"] = "cumulative"
    metadata["prediction_semantics"] = "cumulative_path"
    metadata["backtest_rule"] = "use_h_equals_k_directly_do_not_accumulate_again"
    metadata["returns_mode"] = "cumulative"
    metadata["training_manifest"] = run_manifest

    for key in (
        "experiment_id",
        "git_branch",
        "git_commit",
        "model",
        "loss",
        "dataset_label",
        "data_name",
        "target_data_kind",
        "target_score_kind",
        "seq_len",
        "trained_pred_len",
        "loss_k",
        "seed",
    ):
        if key in run_manifest:
            metadata[key] = run_manifest[key]

    # O output econômico deste experimento é retorno acumulado, inclusive quando
    # a entrada do modelo é preço. Isto evita que o backtest interprete a saída
    # como nível de preço.
    score_kind = str(run_manifest.get("target_score_kind", "simple_return"))
    metadata["backtest_model_output"] = (
        "log_returns" if score_kind == "log_return" else "returns"
    )

    with target_metadata.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def shard_items(items: list[Path], shard_index: int, num_shards: int) -> list[Path]:
    if num_shards < 1:
        raise ValueError("num_shards deve ser >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"shard_index inválido: {shard_index}; esperado 0 <= índice < {num_shards}"
        )
    return [item for index, item in enumerate(items) if index % num_shards == shard_index]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converte previsões TFB do experimento de alvo acumulado."
    )
    parser.add_argument("--clean-root", required=True, type=Path)
    parser.add_argument("--tfb-result-root", required=True, type=Path)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--dataset-map-json", default=None)
    parser.add_argument("--calendar-path", type=Path, default=None)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    converter = load_clean_converter(args.clean_root)
    if not args.tfb_result_root.exists():
        raise FileNotFoundError(f"Raiz de resultados ausente: {args.tfb_result_root}")

    dataset_map = converter.load_dataset_map(args.dataset_path, args.dataset_map_json)
    calendar_wide = (
        converter.read_tfb_wide(args.calendar_path)
        if args.calendar_path is not None
        else None
    )
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_run_dirs = sorted(
        path
        for path in args.tfb_result_root.iterdir()
        if path.is_dir() and path.name not in {"logs", "manifests"}
    )
    run_dirs = shard_items(all_run_dirs, args.shard_index, args.num_shards)
    dataset_cache: dict[Path, pd.DataFrame] = {}
    manifest: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    print(f"Total de runs disponíveis: {len(all_run_dirs)}")
    print(f"Shard de conversão: {args.shard_index}/{args.num_shards}")
    print(f"Runs neste shard: {len(run_dirs)}")

    for position, run_dir in enumerate(run_dirs, 1):
        try:
            info = converter.convert_one_run(
                run_dir,
                dataset_cache,
                dataset_map,
                calendar_wide,
                args.output_root,
                args.overwrite,
                require_full=not args.allow_partial,
            )
            converted_dir = Path(info["converted_dir"])
            enrich_converted_metadata(run_dir, converted_dir)
            info["target_mode"] = "cumulative"
            info["prediction_semantics"] = "cumulative_path"
            info["shard_index"] = args.shard_index
            info["num_shards"] = args.num_shards
            manifest.append(info)
            print(f"[{position}/{len(run_dirs)}] OK {run_dir.name}")
        except Exception as exc:
            errors.append(
                {
                    "run_name": run_dir.name,
                    "source_dir": str(run_dir),
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "error": repr(exc),
                }
            )
            print(f"[{position}/{len(run_dirs)}] ERRO {run_dir.name}: {exc}")

    suffix = f"shard_{args.shard_index:03d}"
    pd.DataFrame(manifest).to_csv(
        args.output_root / f"conversion_manifest_{suffix}.csv", index=False
    )
    pd.DataFrame(errors).to_csv(
        args.output_root / f"conversion_errors_{suffix}.csv", index=False
    )

    print(f"Convertidos neste shard: {len(manifest)}")
    print(f"Erros neste shard: {len(errors)}")
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
