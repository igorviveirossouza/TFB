import base64
import json
import os
import pickle
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


PREDICTION_COLUMNS = {"actual_data", "inference_data"}


def to_csv(data, save_dir: str, save_name: str):
    """
    Save the input data into CSV file(s).

    - DataFrame: saves directly in save_dir/save_name
    - ndarray shape (num, time, dim): saves one CSV per sample in sample_i/
    - list: saves one CSV per sample in sample_i/
    """
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(data, pd.DataFrame):
        data.to_csv(os.path.join(save_dir, save_name), index=False)

    elif isinstance(data, np.ndarray):
        if data.ndim == 2:
            pd.DataFrame(data).to_csv(os.path.join(save_dir, save_name), index=False)
        elif data.ndim >= 3:
            num = data.shape[0]
            for i in range(num):
                sample_dir = os.path.join(save_dir, f"sample_{i}")
                os.makedirs(sample_dir, exist_ok=True)
                pd.DataFrame(data[i]).to_csv(os.path.join(sample_dir, save_name), index=False)
        else:
            pd.DataFrame(data).to_csv(os.path.join(save_dir, save_name), index=False)

    elif isinstance(data, list):
        for i, item in enumerate(data):
            sample_dir = os.path.join(save_dir, f"sample_{i}")
            os.makedirs(sample_dir, exist_ok=True)
            if isinstance(item, pd.DataFrame):
                item.to_csv(os.path.join(sample_dir, save_name), index=False)
            else:
                pd.DataFrame(item).to_csv(os.path.join(sample_dir, save_name), index=False)

    else:
        raise TypeError("Unsupported type for data. Must be pd.DataFrame, np.ndarray, or list.")


def safe_decode_column(row, column_name: str, source_name: str, row_idx: int):
    """
    Decodes a base64 + pickle column safely.
    Returns None if the column does not exist or is empty.
    """
    if column_name not in row.index:
        return None

    value = row[column_name]
    if pd.isna(value):
        return None

    if not isinstance(value, str) or not value.strip():
        return None

    try:
        decoded_bytes = base64.b64decode(value)
        return pickle.loads(decoded_bytes)
    except Exception as e:
        raise ValueError(
            f"[{source_name}] row={row_idx} column={column_name}: erro ao decodificar: {e}"
        ) from e


def build_output_dir(base_output_dir: str, source_name: str) -> str:
    """
    Builds a short, predictable output directory name.

    Examples:
    - BandWiseAdapter.1774720475.gorgona6.184230.csv
      -> decoded_BandWiseAdapter.1774720475.gorgona6.184230
    - BandWiseAdapter.1774720475.gorgona6.184230.csv.tar.gz
      -> decoded_BandWiseAdapter.1774720475.gorgona6.184230
    """
    name = source_name
    if name.endswith(".csv.tar.gz"):
        stem = name[:-11]  # remove .csv.tar.gz
    elif name.endswith(".csv"):
        stem = name[:-4]
    else:
        stem = Path(name).stem

    return os.path.join(base_output_dir, f"decoded_{stem}")


def save_metadata(save_dir: str, source_name: str, row_idx: int, row):
    metadata = {
        "source_name": source_name,
        "row_index": int(row_idx),
        "model_name": row.get("model_name", None),
        "file_name": row.get("file_name", None),
        "fit_time": row.get("fit_time", None),
        "inference_time": row.get("inference_time", None),
    }

    with open(os.path.join(save_dir, "decode_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)


def decode_result_dataframe(data: pd.DataFrame, base_output_dir: str, source_name: str):
    """
    Decodes only raw prediction result CSVs.
    Skips files without actual_data/inference_data columns.
    """
    if not any(col in data.columns for col in PREDICTION_COLUMNS):
        print(f"[SKIP] {source_name}: arquivo sem colunas de predição.")
        return

    save_dir = build_output_dir(base_output_dir, source_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[INFO] Salvando dados decodificados em: {save_dir}")

    for index, row in data.iterrows():
        sample_dir = os.path.join(save_dir, f"sample_{index}")
        os.makedirs(sample_dir, exist_ok=True)

        inference_data = safe_decode_column(row, "inference_data", source_name, index)
        actual_data = safe_decode_column(row, "actual_data", source_name, index)

        if inference_data is None and actual_data is None:
            print(f"[SKIP] {source_name}: linha {index} sem dados decodificáveis.")
            continue

        if inference_data is not None:
            to_csv(inference_data, sample_dir, "inference_data.csv")

        if actual_data is not None:
            to_csv(actual_data, sample_dir, "actual_data.csv")

        save_metadata(sample_dir, source_name, index, row)

        metrics = row.drop(labels=[c for c in ["actual_data", "inference_data"] if c in row.index])
        pd.DataFrame([metrics]).to_csv(os.path.join(sample_dir, "metrics.csv"), index=False)


def decode_csv_file(filepath: str):
    """
    Processes one raw .csv result file.
    """
    data = pd.read_csv(filepath)
    base_output_dir = os.path.dirname(filepath)
    source_name = os.path.basename(filepath)
    print(f"[INFO] Decodificando CSV: {filepath}")
    decode_result_dataframe(data, base_output_dir, source_name)


def decode_tar_gz_file(filepath: str):
    """
    Processes one .csv.tar.gz file:
    - extracts the main CSV into <archive>_extracted/
    - decodes prediction columns into a short output folder
    """
    archive_path = Path(filepath)
    source_name = archive_path.name

    extract_dir = archive_path.parent / source_name.replace(".tar.gz", "_extracted")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(filepath, "r:gz") as tar:
        csv_members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".csv")]
        if not csv_members:
            raise FileNotFoundError(f"Nenhum CSV encontrado em {filepath}")

        expected_name = source_name.replace(".tar.gz", "")
        selected_member = None
        for member in csv_members:
            if Path(member.name).name == expected_name:
                selected_member = member
                break

        if selected_member is None:
            selected_member = csv_members[0]

        extracted_file = tar.extractfile(selected_member)
        if extracted_file is None:
            raise RuntimeError(f"Não foi possível extrair {selected_member.name} de {filepath}")

        data = pd.read_csv(extracted_file)

    raw_csv_path = extract_dir / Path(selected_member.name).name
    data.to_csv(raw_csv_path, index=False)

    print(f"[INFO] Decodificando archive: {filepath}")
    decode_result_dataframe(data, str(extract_dir), source_name)


def process_directory(dirpath: str):
    """
    Processes only .csv.tar.gz files in the top-level directory.

    Important:
    - does not recurse
    - does not process *_extracted
    - does not process sample_*/actual_data.csv
    - does not process sample_*/inference_data.csv
    - does not process test_report.*.csv
    """
    base_dir = Path(dirpath)

    archives = sorted(
        [
            p for p in base_dir.iterdir()
            if p.is_file() and p.name.endswith(".csv.tar.gz")
        ]
    )

    if not archives:
        print(f"[INFO] Nenhum arquivo .csv.tar.gz encontrado em {dirpath}")
        return

    for archive in archives:
        try:
            decode_tar_gz_file(str(archive))
        except Exception as e:
            print(f"[ERROR] Falha ao processar {archive.name}: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise ValueError(
            "Uso: python decode_prediction.py <arquivo.csv | arquivo.csv.tar.gz | diretorio>"
        )

    input_path = sys.argv[1]

    if os.path.isdir(input_path):
        process_directory(input_path)
    elif os.path.isfile(input_path):
        if input_path.endswith(".csv.tar.gz"):
            decode_tar_gz_file(input_path)
        elif input_path.endswith(".csv"):
            decode_csv_file(input_path)
        else:
            raise ValueError(
                "Arquivo não suportado. Use .csv, .csv.tar.gz, ou informe um diretório."
            )
    else:
        raise FileNotFoundError(f"Caminho não encontrado: {input_path}")
