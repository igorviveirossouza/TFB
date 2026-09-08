#!/usr/bin/env python3
"""Convert decoded TFB forecasts for the composite-trading experiment.

This converter preserves the temporal convention established in tfb_check:
each output window receives the final columns h, origin_step, step, with
(step - origin_step) == h. Both return datasets use step_offset=2 because
their rows represent the same price-to-price trading intervals despite
different original date labels.

The preferred output is a Hive-partitioned Parquet dataset:
  dataset/model/lookback/pred_len/k_dir
CSV windows can optionally be retained for audit/debug purposes.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

import pandas as pd


RX = re.compile(r"csv_sample_(\d+)_inference_data\.csv$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--decoded-dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--pred-len", required=True, type=int)
    p.add_argument("--lookback", required=True, type=int)
    p.add_argument("--step-offset", required=True, type=int)
    p.add_argument("--tv-ratio", required=True, type=float)

    # Parquet output (preferred path).
    p.add_argument("--parquet-root")
    p.add_argument("--dataset-label")
    p.add_argument("--model")
    p.add_argument("--k", type=int)

    # Optional legacy/audit CSV output.
    p.add_argument("--output-dir")
    p.add_argument("--keep-csv", action="store_true")
    return p.parse_args()


def infer_dataset_shape(path: Path) -> tuple[int, list[str]]:
    orig = pd.read_csv(path)
    reserved = {"step", "date"}

    if "cols" in orig.columns:
        work = orig[orig["cols"].astype(str).str.lower() != "label"].copy()
        if "date" in work.columns:
            original_len = int(work.groupby("cols", sort=False)["date"].nunique().max())
        else:
            original_len = int(work["cols"].value_counts(sort=False).max())

        target_cols: list[str] = []
        seen: set[str] = set()
        for x in work["cols"].tolist():
            s = str(x).strip()
            if s and s.lower() != "label" and s not in reserved and s not in seen:
                seen.add(s)
                target_cols.append(s)
    else:
        original_len = len(orig)
        target_cols = [
            str(c)
            for c in orig.columns
            if str(c) not in reserved and str(c).lower() != "label"
        ]

    return original_len, target_cols


def parquet_partition_dir(args: argparse.Namespace) -> Path:
    required = {
        "dataset_label": args.dataset_label,
        "model": args.model,
        "k": args.k,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Para gerar Parquet faltam argumentos: " + ", ".join(missing)
        )

    values = [str(args.dataset_label), str(args.model)]
    if any("/" in x for x in values):
        raise ValueError("dataset-label/model não podem conter '/'.")

    return (
        Path(args.parquet_root)
        / f"dataset={args.dataset_label}"
        / f"modelo={args.model}"
        / f"lookback=lookback_{args.lookback}"
        / f"pred_len=pred_len_{args.pred_len}"
        / f"k_dir=k_{args.k}"
    )


def write_parquet_atomic(df: pd.DataFrame, partition_dir: Path) -> Path:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow não está instalado. Instale-o no ambiente Python para gerar Parquet."
        ) from exc

    partition_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = partition_dir.parent / (
        f".{partition_dir.name}.pid{os.getpid()}.tmp.parquet"
    )

    try:
        df.to_parquet(
            tmp_file,
            index=False,
            engine="pyarrow",
            compression="snappy",
        )

        if partition_dir.exists():
            shutil.rmtree(partition_dir)
        partition_dir.mkdir(parents=True, exist_ok=True)

        final_file = partition_dir / "part.parquet"
        tmp_file.replace(final_file)
        return final_file
    finally:
        if tmp_file.exists():
            tmp_file.unlink()


def main() -> None:
    args = parse_args()

    if args.parquet_root is None and args.output_dir is None:
        raise ValueError("Informe --parquet-root ou --output-dir.")

    # Backward compatibility: an old caller that only supplies --output-dir
    # still receives the original janela_*.csv files.
    keep_csv = bool(args.keep_csv or (args.output_dir and not args.parquet_root))
    if keep_csv and not args.output_dir:
        raise ValueError("--keep-csv exige --output-dir.")

    pred_dir = Path(args.decoded_dir)
    original_path = Path(args.dataset)

    csv_dir = Path(args.output_dir) if args.output_dir else None
    if keep_csv and csv_dir is not None:
        if csv_dir.exists():
            shutil.rmtree(csv_dir)
        csv_dir.mkdir(parents=True, exist_ok=True)

    files: list[tuple[int, Path]] = []
    for p in pred_dir.glob("csv_sample_*_inference_data.csv"):
        m = RX.search(p.name)
        if m:
            files.append((int(m.group(1)), p))
    files.sort()

    if not files:
        raise RuntimeError(f"Nenhum csv_sample em {pred_dir}")

    indexes = [i for i, _ in files]
    if indexes != list(range(len(indexes))):
        raise RuntimeError(
            f"sample_idx não contíguos: início={indexes[:5]} fim={indexes[-5:]}"
        )

    original_len, target_cols = infer_dataset_shape(original_path)
    n_files = len(files)

    first_zero = original_len - args.pred_len - (n_files - 1)
    expected_first_zero = int(args.tv_ratio * original_len)

    if first_zero != expected_first_zero:
        raise RuntimeError(
            "Fronteira temporal inconsistente: "
            f"first_zero={first_zero}, esperado={expected_first_zero}, "
            f"N={original_len}, h={args.pred_len}, n_janelas={n_files}."
        )

    if first_zero - args.lookback < 0:
        raise RuntimeError("Lookback invade o início da série.")

    reserved = {"step", "date"}
    parquet_frames: list[pd.DataFrame] = []
    first_origin: int | None = None
    first_step: int | None = None

    for sample_idx, path in files:
        df = pd.read_csv(path)

        drop_cols = [c for c in df.columns if str(c) in reserved]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        if len(df) != args.pred_len:
            raise RuntimeError(
                f"{path.name}: esperado {args.pred_len} linhas, obtido {len(df)}"
            )
        if len(df.columns) != len(target_cols):
            raise RuntimeError(
                f"{path.name}: {len(df.columns)} colunas previstas versus "
                f"{len(target_cols)} ativos no dataset"
            )

        df.columns = target_cols

        start_zero = first_zero + sample_idx
        zero_steps = list(range(start_zero, start_zero + args.pred_len))
        visible_steps = [z + args.step_offset for z in zero_steps]
        origin_step = visible_steps[0] - 1

        df["h"] = range(1, args.pred_len + 1)
        df["origin_step"] = origin_step
        df["step"] = visible_steps

        if not ((df["step"] - df["origin_step"]) == df["h"]).all():
            raise RuntimeError(f"Invariante temporal falhou em {path.name}")

        janela_name = f"janela_{sample_idx:06d}.csv"

        if keep_csv and csv_dir is not None:
            df.to_csv(csv_dir / janela_name, index=False)

        if args.parquet_root is not None:
            frame = df.copy()
            frame.insert(0, "arquivo", janela_name)
            parquet_frames.append(frame)

        if sample_idx == 0:
            first_origin = int(df["origin_step"].iloc[0])
            first_step = int(df["step"].iloc[0])

    parquet_file: Path | None = None
    if args.parquet_root is not None:
        if not parquet_frames:
            raise RuntimeError("Nenhuma janela disponível para gerar o Parquet.")
        parquet_df = pd.concat(parquet_frames, ignore_index=True)
        partition_dir = parquet_partition_dir(args)
        parquet_file = write_parquet_atomic(parquet_df, partition_dir)

    print(
        f"Conversão OK: N={original_len}; janelas={n_files}; "
        f"origem_inicial={first_origin}; alvo_h1={first_step}; "
        f"h={args.pred_len}; offset={args.step_offset}"
    )
    if parquet_file is not None:
        print(f"Parquet OK: {parquet_file}")
    if keep_csv and csv_dir is not None:
        print(f"CSVs mantidos: {csv_dir}")


if __name__ == "__main__":
    main()
