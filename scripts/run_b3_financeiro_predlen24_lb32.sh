#!/bin/bash
set -euo pipefail

MODEL_KEY="${1:?Uso: bash scripts/run_b3_financeiro_predlen24_lb32.sh <DUET|TimesNet|FEDformer|Nonstationary_Transformer>}"

TFB_ROOT="${TFB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_NAME="${DATA_NAME:-b3_daily_return_financeiro.csv}"
SEQ_LEN="${SEQ_LEN:-32}"
PRED_LEN="${PRED_LEN:-24}"
LABEL_LEN="${LABEL_LEN:-18}"

SAVE_SUBDIR="${SAVE_SUBDIR:-paralelo_predlen24_lb32/seq_len_${SEQ_LEN}/pred_len_${PRED_LEN}/${MODEL_KEY}}"
RESULT_DIR="${TFB_ROOT}/result/${SAVE_SUBDIR}"
DECODED_ROOT="${DECODED_ROOT:-${TFB_ROOT}/Previsoes/paralelo_predlen24_lb32}"
DECODED_DIR="${DECODED_ROOT}/seq_len_${SEQ_LEN}/pred_len_${PRED_LEN}/${MODEL_KEY}"
DECODE_SCRIPT="${DECODE_SCRIPT:-${TFB_ROOT}/ts_benchmark/utils/decode_prediction.py}"

mkdir -p "$RESULT_DIR" "$DECODED_DIR"

MODEL_NAME=""
MODEL_HYPER_PARAMS=""
ADAPTER_ARG=()
DETERMINISTIC_MODE="full"

case "$MODEL_KEY" in
  "DUET")
    MODEL_NAME="duet.DUET"
    MODEL_HYPER_PARAMS="{\"CI\": 1, \"batch_size\": 32, \"d_ff\": 32, \"d_model\": 32, \"dropout\": 0.5, \"e_layers\": 1, \"factor\": 3, \"fc_dropout\": 0.2, \"hidden_size\": 256, \"pred_len\": ${PRED_LEN}, \"horizon\": ${PRED_LEN}, \"k\": 1, \"loss\": \"MAE\", \"lr\": 0.01, \"lradj\": \"type1\", \"n_heads\": 2, \"norm\": true, \"num_epochs\": 100, \"num_experts\": 4, \"patch_len\": 48, \"patience\": 10, \"seq_len\": ${SEQ_LEN}}"
    ;;
  "TimesNet")
    MODEL_NAME="time_series_library.TimesNet"
    MODEL_HYPER_PARAMS="{\"batch_size\": 32, \"d_ff\": 512, \"d_model\": 256, \"factor\": 3, \"pred_len\": ${PRED_LEN}, \"horizon\": ${PRED_LEN}, \"norm\": true, \"seq_len\": ${SEQ_LEN}, \"top_k\": 5}"
    ADAPTER_ARG=(--adapter "transformer_adapter")
    DETERMINISTIC_MODE="efficient"
    ;;
  "FEDformer")
    MODEL_NAME="time_series_library.FEDformer"
    MODEL_HYPER_PARAMS="{\"batch_size\": 32, \"d_ff\": 512, \"d_model\": 256, \"factor\": 3, \"pred_len\": ${PRED_LEN}, \"horizon\": ${PRED_LEN}, \"norm\": true, \"seq_len\": ${SEQ_LEN}}"
    ADAPTER_ARG=(--adapter "transformer_adapter")
    ;;
  "Nonstationary_Transformer")
    MODEL_NAME="time_series_library.Nonstationary_Transformer"
    MODEL_HYPER_PARAMS="{\"d_ff\": 256, \"d_model\": 128, \"dropout\": 0.1, \"factor\": 3, \"pred_len\": ${PRED_LEN}, \"horizon\": ${PRED_LEN}, \"norm\": true, \"p_hidden_dims\": [32, 32], \"p_hidden_layers\": 2, \"seq_len\": ${SEQ_LEN}}"
    ADAPTER_ARG=(--adapter "transformer_adapter")
    ;;
  *)
    echo "Modelo não mapeado: ${MODEL_KEY}" >&2
    exit 1
    ;;
esac

echo "=== TFB financeiro predlen24/lb32 ==="
echo "TFB_ROOT=${TFB_ROOT}"
echo "DATA_NAME=${DATA_NAME}"
echo "MODEL_KEY=${MODEL_KEY}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "SEQ_LEN=${SEQ_LEN} | PRED_LEN=${PRED_LEN}"
echo "RESULT_DIR=${RESULT_DIR}"
echo "DECODED_DIR=${DECODED_DIR}"

cd "$TFB_ROOT"

mapfile -t BEFORE_TARS < <(find "$RESULT_DIR" -maxdepth 1 -type f -name "*.csv.tar.gz" -printf '%f\n' 2>/dev/null | sort)

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

"$PYTHON_BIN" ./scripts/run_benchmark.py \
  --config-path "rolling_forecast_config.json" \
  --data-name-list "$DATA_NAME" \
  --strategy-args "{\"horizon\": ${PRED_LEN}}" \
  --model-name "$MODEL_NAME" \
  --model-hyper-params "$MODEL_HYPER_PARAMS" \
  "${ADAPTER_ARG[@]}" \
  --deterministic "$DETERMINISTIC_MODE" \
  --gpus 0 \
  --num-workers 1 \
  --timeout 60000 \
  --save-path "$SAVE_SUBDIR" \
  --save-true-pred True

mapfile -t AFTER_TARS < <(find "$RESULT_DIR" -maxdepth 1 -type f -name "*.csv.tar.gz" -printf '%f\n' | sort)

NEW_TARS=()
for tar_name in "${AFTER_TARS[@]}"; do
  if ! printf '%s\n' "${BEFORE_TARS[@]}" | grep -Fxq "$tar_name"; then
    NEW_TARS+=("${RESULT_DIR}/${tar_name}")
  fi
done

if [[ ${#NEW_TARS[@]} -eq 0 ]]; then
  echo "ERRO: nenhum novo .csv.tar.gz encontrado em ${RESULT_DIR}" >&2
  exit 1
fi

rm -rf "$DECODED_DIR"
mkdir -p "$DECODED_DIR"

for tarfile in "${NEW_TARS[@]}"; do
  sample_offset="$(find "$DECODED_DIR" -maxdepth 1 -type f -name 'csv_sample_*_inference_data.csv' | wc -l | tr -d ' ')"
  "$PYTHON_BIN" "$DECODE_SCRIPT" "$tarfile" \
    --output-dir "$DECODED_DIR" \
    --columns inference_data \
    --sample-offset "$sample_offset"
done

echo "TFB finalizado. Previsões decodificadas: ${DECODED_DIR}"
