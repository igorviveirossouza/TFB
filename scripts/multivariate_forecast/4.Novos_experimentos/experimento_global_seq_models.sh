#!/bin/bash
#SBATCH -p gorgonas
#SBATCH --cpus-per-task=32
#SBATCH --output=/sonic_home/igor.viveiros/logs/experimento-global-%j.out

set -euo pipefail

source /sonic_home/igor.viveiros/py310/bin/activate
PYTHON_BIN=/sonic_home/igor.viveiros/py310/bin/python

export MPLCONFIGDIR=/tmp/$USER-mpl

cd /sonic_home/igor.viveiros/src/TFB || exit 1

DATA_NAME="b3_daily_tfb.csv"
PRED_LEN=24
LABEL_LEN=18
JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_DATE="$(date +%Y%m%d)"
PREDICTION_OP_INDEX=0

RESULT_ROOT="/sonic_home/igor.viveiros/src/TFB/result/experimento_global"
PREV_ROOT="/sonic_home/igor.viveiros/src/TFB/Previsoes"
DECODE_SCRIPT="/sonic_home/igor.viveiros/src/TFB/ts_benchmark/utils/decode_prediction.py"

SEQ_LENS=(18 26 104)
MODELS=(
  "DUET"
  "TimesNet"
  "FEDformer"
  "Nonstationary_Transformer"
  "TIOMS_OHLC_DILATE_g001"
  "TIOMS_OHLC_MSE"
  "TIOMS_AttentionAdapterChannel"
)

mkdir -p "$RESULT_ROOT" "$PREV_ROOT"

decode_predictions() {
  local tarfile="$1"
  local seq_len="$2"
  local model_key="$3"
  local pred_len="$4"
  local op_index="$5"

  local out_dir="${PREV_ROOT}/${seq_len}/${model_key}"
  mkdir -p "$out_dir"

  local tar_dir raw_stem extracted_dir copy_index
  tar_dir="$(dirname "$tarfile")"
  raw_stem="$(basename "${tarfile%.tar.gz}")"
  extracted_dir="${tar_dir}/$(basename "$tarfile" .tar.gz)_extracted"
  copy_index=0

  "$PYTHON_BIN" "$DECODE_SCRIPT" "$tarfile"

  if [[ ! -d "$extracted_dir" ]]; then
    echo "[WARN] Pasta extraída não encontrada para ${tarfile}" >&2
    return
  fi

  find "$extracted_dir" -type f -path "*/decoded_*/*/*.csv" | while read -r decoded_csv; do
    local rows output_name sample_tag data_tag
    rows="$("$PYTHON_BIN" -c "import pandas as pd; print(len(pd.read_csv(r'''$decoded_csv''')))" 2>/dev/null || echo 0)"
    if [[ "$rows" -ne "$pred_len" ]]; then
      echo "CSV ignorado (length=${rows}, esperado=${pred_len}): ${decoded_csv}"
      continue
    fi

    sample_tag="$(basename "$(dirname "$decoded_csv")")"
    data_tag="$(basename "$decoded_csv" .csv)"
    if [[ "$data_tag" != "inference_data" ]]; then
      continue
    fi

    output_name="${model_key}_${RUN_DATE}_job${JOB_ID}_op${op_index}_idx${copy_index}_${raw_stem}_${sample_tag}_${data_tag}.csv"
    cp "$decoded_csv" "${out_dir}/${output_name}"
    echo "CSV decodificado salvo: ${out_dir}/${output_name}"
    copy_index=$((copy_index + 1))
  done
}

for SEQ_LEN in "${SEQ_LENS[@]}"; do
  for MODEL_KEY in "${MODELS[@]}"; do
    echo "=================================================="
    echo "Executando: seq_len=${SEQ_LEN} | modelo=${MODEL_KEY}"
    echo "=================================================="

    RESULT_DIR="${RESULT_ROOT}/seq_len_${SEQ_LEN}/${MODEL_KEY}"
    mkdir -p "$RESULT_DIR"

    MODEL_NAME=""
    MODEL_HYPER_PARAMS=""
    ADAPTER_ARG=()
    DATASET_ARG=()
    DETERMINISTIC_MODE="full"
    EVAL_BACKEND_ARG=()

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
      "TIOMS_OHLC_DILATE_g001")
        MODEL_NAME="TIOMS.AttentionAdapterChannelEnc"
        MODEL_HYPER_PARAMS="{\"loss\": \"DILATE\", \"dilate_alpha\": 0.5, \"dilate_gamma\": 0.01, \"embedding_type\": \"nonlinear\", \"embedding_hidden_dim\": 16, \"lag_size\": 7, \"spectral_num_freqs\": 18, \"causal_att\": \"no_self\", \"channel_agg_type\": \"none\", \"seq_len\": ${SEQ_LEN}, \"pred_len\": ${PRED_LEN}, \"horizon\": ${PRED_LEN}, \"label_len\": ${LABEL_LEN}, \"batch_size\": 32, \"dropout\": 0.1, \"eps\": 1e-5, \"d_model\": 32, \"n_heads\": 4, \"ff_dim\": 128, \"channel_n_heads\": 60, \"patience\": 8, \"num_epochs\": 60, \"temporal_pool_type\": \"last\"}"
        DATASET_ARG=(--data-set-name "user_forecast_ohlcv")
        EVAL_BACKEND_ARG=(--eval-backend sequential)
        ;;
      "TIOMS_OHLC_MSE")
        MODEL_NAME="TIOMS.AttentionAdapterChannelEnc"
        MODEL_HYPER_PARAMS="{\"loss\": \"MSE\", \"embedding_type\": \"nonlinear\", \"embedding_hidden_dim\": 16, \"lag_size\": 7, \"spectral_num_freqs\": 18, \"causal_att\": \"no_self\", \"channel_agg_type\": \"none\", \"seq_len\": ${SEQ_LEN}, \"pred_len\": ${PRED_LEN}, \"horizon\": ${PRED_LEN}, \"label_len\": ${LABEL_LEN}, \"batch_size\": 32, \"dropout\": 0.1, \"eps\": 1e-5, \"d_model\": 32, \"n_heads\": 4, \"ff_dim\": 128, \"channel_n_heads\": 60, \"patience\": 8, \"num_epochs\": 60, \"temporal_pool_type\": \"last\"}"
        DATASET_ARG=(--data-set-name "user_forecast_ohlcv")
        EVAL_BACKEND_ARG=(--eval-backend sequential)
        ;;
      "TIOMS_AttentionAdapterChannel")
        MODEL_NAME="TIOMS.AttentionAdapterChannel"
        MODEL_HYPER_PARAMS="{\"loss\": \"MSE\", \"embedding_type\": \"nonlinear\", \"embedding_hidden_dim\": 16, \"lag_size\": 7, \"spectral_num_freqs\": 18, \"causal_att\": \"no_self\", \"channel_agg_type\": \"none\", \"seq_len\": ${SEQ_LEN}, \"pred_len\": ${PRED_LEN}, \"label_len\": ${LABEL_LEN}, \"horizon\": ${PRED_LEN}, \"batch_size\": 32, \"dropout\": 0.1, \"eps\": 1e-5, \"d_model\": 32, \"n_heads\": 4, \"ff_dim\": 128, \"channel_n_heads\": 60, \"patience\": 8, \"num_epochs\": 60, \"temporal_pool_type\": \"last\"}"
        EVAL_BACKEND_ARG=(--eval-backend sequential)
        ;;
      *)
        echo "Modelo não mapeado: ${MODEL_KEY}" >&2
        exit 1
        ;;
    esac

    mapfile -t BEFORE_TARS < <(find "$RESULT_DIR" -maxdepth 1 -type f -name "*.csv.tar.gz" -printf '%f\n' | sort)

    "$PYTHON_BIN" ./scripts/run_benchmark.py \
      --config-path "rolling_forecast_config.json" \
      --data-name-list "$DATA_NAME" \
      "${DATASET_ARG[@]}" \
      --strategy-args "{\"horizon\": ${PRED_LEN}}" \
      --model-name "$MODEL_NAME" \
      --model-hyper-params "$MODEL_HYPER_PARAMS" \
      "${ADAPTER_ARG[@]}" \
      --deterministic "$DETERMINISTIC_MODE" \
      "${EVAL_BACKEND_ARG[@]}" \
      --gpus 0 \
      --num-workers 1 \
      --timeout 60000 \
      --save-path "experimento_global/seq_len_${SEQ_LEN}/${MODEL_KEY}" \
      --save-true-pred True

    mapfile -t AFTER_TARS < <(find "$RESULT_DIR" -maxdepth 1 -type f -name "*.csv.tar.gz" -printf '%f\n' | sort)

    NEW_TARS=()
    for tar_name in "${AFTER_TARS[@]}"; do
      if ! printf '%s\n' "${BEFORE_TARS[@]}" | grep -Fxq "$tar_name"; then
        NEW_TARS+=("${RESULT_DIR}/${tar_name}")
      fi
    done

    if [[ ${#NEW_TARS[@]} -eq 0 ]]; then
      echo "[WARN] Nenhum novo tar.gz encontrado para seq_len=${SEQ_LEN}, modelo=${MODEL_KEY}"
      continue
    fi

    for tarfile in "${NEW_TARS[@]}"; do
      PREDICTION_OP_INDEX=$((PREDICTION_OP_INDEX + 1))
      decode_predictions "$tarfile" "seq_len_${SEQ_LEN}" "$MODEL_KEY" "$PRED_LEN" "$PREDICTION_OP_INDEX"
    done
  done
done

echo "Todos os experimentos finalizados." 
