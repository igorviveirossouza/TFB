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

rename_tars() {
  local result_dir="$1"
  local model_key="$2"

  find "$result_dir" -maxdepth 1 -type f -name "*.tar.gz" | while read -r tarfile; do
    local old_name new_name
    old_name="$(basename "$tarfile")"
    if [[ "$old_name" == *"${model_key}"* && "$old_name" == *"${JOB_ID}"* ]]; then
      continue
    fi

    new_name="${model_key}_${JOB_ID}_${old_name}"
    mv "$tarfile" "${result_dir}/${new_name}"
    echo "Tar renomeado: ${result_dir}/${new_name}"
  done
}

decode_predictions() {
  local result_dir="$1"
  local seq_len="$2"
  local model_key="$3"

  local out_dir="${PREV_ROOT}/${seq_len}/${model_key}"
  mkdir -p "$out_dir"

  find "$result_dir" -maxdepth 1 -type f -name "*.tar.gz" | while read -r tarfile; do
    local base_name tmpdir
    base_name="$(basename "${tarfile%.tar.gz}")"
    tmpdir="$(mktemp -d)"

    tar -xzf "$tarfile" -C "$tmpdir"

    find "$tmpdir" -type f -name "*.csv" | while read -r csvfile; do
      "$PYTHON_BIN" "$DECODE_SCRIPT" "$csvfile"
    done

    find "$tmpdir" -type f -path "*/decoded_*/*.csv" | while read -r decoded_csv; do
      local output_name
      output_name="${model_key}_${JOB_ID}_${base_name}_$(basename "$decoded_csv")"
      cp "$decoded_csv" "${out_dir}/${output_name}"
      echo "CSV decodificado salvo: ${out_dir}/${output_name}"
    done

    rm -rf "$tmpdir"
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

    rename_tars "$RESULT_DIR" "$MODEL_KEY"
    decode_predictions "$RESULT_DIR" "seq_len_${SEQ_LEN}" "$MODEL_KEY"
  done
done

echo "Todos os experimentos finalizados." 
