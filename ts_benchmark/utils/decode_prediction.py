#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import pickle
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.recording import read_record_file


def decode_cell(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return pickle.loads(base64.b64decode(text))


def iter_samples(obj: Any) -> Iterable[Any]:
    if isinstance(obj, list):
        yield from obj
        return
    if isinstance(obj, tuple):
        yield from obj
        return
    if isinstance(obj, pd.DataFrame):
        yield obj
        return
    if isinstance(obj, pd.Series):
        yield obj.to_frame()
        return

    arr = np.asarray(obj)
    if arr.ndim == 3:
        for i in range(arr.shape[0]):
            yield arr[i]
    elif arr.ndim == 2:
        yield arr
    elif arr.ndim == 1:
        yield arr.reshape(-1, 1)
    else:
        raise ValueError(f"Formato de previsão não suportado: shape={arr.shape}")


def to_frame(sample: Any) -> pd.DataFrame:
    if isinstance(sample, pd.DataFrame):
        return sample.reset_index(drop=True)
    if isinstance(sample, pd.Series):
        return sample.reset_index(drop=True).to_frame()
    return pd.DataFrame(np.asarray(sample))


def decode_prediction_file(
    record_file: str | Path,
    output_dir: str | Path | None = None,
    columns: list[str] | None = None,
    sample_offset: int = 0,
    clean_output: bool = False,
) -> int:
    record_file = Path(record_file)
    if output_dir is None:
        stem = record_file.name
        if stem.endswith(".tar.gz"):
            stem = stem[:-7]
        output_dir = record_file.parent / f"{stem}_decoded"

    output_dir = Path(output_dir)
    if clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if columns is None:
        columns = [FieldNames.INFERENCE_DATA]

    records = read_record_file(str(record_file))
    decoded_count = 0

    for row_idx, row in records.iterrows():
        if row.get(FieldNames.LOG_INFO):
            print(f"[WARN] row={row_idx} tem log_info: {row[FieldNames.LOG_INFO]}")

        for column in columns:
            if column not in records.columns:
                raise ValueError(f"Coluna ausente no record: {column}")

            decoded = decode_cell(row[column])
            if decoded is None:
                continue

            samples = list(iter_samples(decoded))
            for local_idx, sample in enumerate(samples):
                sample_idx = sample_offset + decoded_count + local_idx
                out = to_frame(sample)
                out.to_csv(output_dir / f"csv_sample_{sample_idx}_{column}.csv", index=False)

            decoded_count += len(samples)

    return decoded_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decodifica actual_data/inference_data salvos pelo TFB com --save-true-pred."
    )
    parser.add_argument("record_file", help="Arquivo .csv.tar.gz gerado pelo TFB.")
    parser.add_argument("--output-dir", default=None, help="Diretório para CSVs decodificados.")
    parser.add_argument(
        "--columns",
        nargs="+",
        default=[FieldNames.INFERENCE_DATA],
        choices=[FieldNames.ACTUAL_DATA, FieldNames.INFERENCE_DATA],
        help="Colunas codificadas a extrair.",
    )
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--clean-output", action="store_true")
    args = parser.parse_args()

    n = decode_prediction_file(
        args.record_file,
        output_dir=args.output_dir,
        columns=args.columns,
        sample_offset=args.sample_offset,
        clean_output=args.clean_output,
    )
    print(f"Arquivos decodificados: {n}")


if __name__ == "__main__":
    main()
