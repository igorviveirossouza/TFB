#!/bin/bash
set -euo pipefail

source /sonic_home/igor.viveiros/py310/bin/activate

cd /sonic_home/igor.viveiros/src/TFB || exit 1

RESULT_BASE="/sonic_home/igor.viveiros/src/TFB/result/experimentos/AtencaoSolo/Encoder/Losses"
OUT_BASE="/sonic_home/igor.viveiros/src/TFB/Previsoes"
DECODE_SCRIPT="/sonic_home/igor.viveiros/src/TFB/ts_benchmark/utils/decode_prediction.py"

mkdir -p "$OUT_BASE"

find "$RESULT_BASE" -type f -name "*.tar.gz" | while read -r tarfile; do
    echo "Extraindo: $tarfile"

    # nome relativo a partir de RESULT_BASE
    rel_path="${tarfile#$RESULT_BASE/}"
    rel_dir="$(dirname "$rel_path")"
    base_name="$(basename "${tarfile%.tar.gz}")"

    extract_dir="$OUT_BASE/$rel_dir/${base_name}_extracted"
    mkdir -p "$extract_dir"

    tar -xzf "$tarfile" -C "$extract_dir"

    find "$extract_dir" -type f -name "*.csv" | while read -r csvfile; do
        echo "Decodificando CSV: $csvfile"
        python "$DECODE_SCRIPT" "$csvfile"
    done
done

echo "Decodificação finalizada."