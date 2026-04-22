#!/bin/bash
#SBATCH -p gorgonas
#SBATCH --cpus-per-task=32
#SBATCH --output=/sonic_home/igor.viveiros/logs/experimento_global-%j.out

set -euo pipefail

source /sonic_home/igor.viveiros/py310/bin/activate
PYTHON_BIN=/sonic_home/igor.viveiros/py310/bin/python

export MPLCONFIGDIR=/tmp/$USER-mpl

cd /sonic_home/igor.viveiros/src/TFB || exit 1

SEQ_LEN=36
PRED_LEN=24
LABEL_LEN=18
DATA_NAME="b3_daily_tfb.csv"

RESULT_ROOT="/sonic_home/igor.viveiros/src/TFB/result/experimentos/experimento_global"
PREV_ROOT="/sonic_home/igor.viveiros/src/TFB/Previsoes/experimento_global"
DECODE_SCRIPT="/sonic_home/igor.viveiros/src/TFB/ts_benchmark/utils/decode_prediction.py"

mkdir -p "$RESULT_ROOT" "$PREV_ROOT"

extrair_previsoes () {
    MODEL_SUBDIR="$1"
    RESULT_BASE="${RESULT_ROOT}/${MODEL_SUBDIR}"
    OUT_BASE="${PREV_ROOT}/${MODEL_SUBDIR}"

    mkdir -p "$OUT_BASE"

    find "$RESULT_BASE" -type f -name "*.tar.gz" | while read -r tarfile; do
        echo "Extraindo: $tarfile"

        base_name="$(basename "${tarfile%.tar.gz}")"
        extract_dir="$OUT_BASE/${base_name}_extracted"
        mkdir -p "$extract_dir"

        tar -xzf "$tarfile" -C "$extract_dir"

        find "$extract_dir" -type f -name "*.csv" | while read -r csvfile; do
            echo "Decodificando CSV: $csvfile"
            python "$DECODE_SCRIPT" "$csvfile"
        done

        find "$extract_dir" -type f -path "*/decoded_*/*.csv" | while read -r decoded_csv; do
            dest_name="$(basename "$decoded_csv")"
            cp "$decoded_csv" "$OUT_BASE/${base_name}__${dest_name}"
            echo "Copiado para: $OUT_BASE/${base_name}__${dest_name}"
        done
    done
}

# ==================================================
# DUET
# ==================================================
#SALVAR_EM="DUET"
#rm -rf "${RESULT_ROOT}/${SALVAR_EM}" "${PREV_ROOT}/${SALVAR_EM}"
#mkdir -p "${RESULT_ROOT}/${SALVAR_EM}" "${PREV_ROOT}/${SALVAR_EM}"

echo "=================================================="
echo "DUET | dataset=${DATA_NAME}"
echo "=================================================="

#$PYTHON_BIN ./scripts/run_benchmark.py \
#  --config-path "rolling_forecast_config.json" \
#  --data-name-list "${DATA_NAME}" \
#  --strategy-args "{\"horizon\": ${PRED_LEN}, \"train_ratio_in_tv\": 1.0}" \
#  --model-name "duet.DUET" \
#  --model-hyper-params "{\"CI\": 1, \"batch_size\": 32, \"d_ff\": 32, \"d_model\": 32, \"dropout\": 0.5, \"e_layers\": 1, \"factor\": 3, \"fc_dropout\": 0.2, \"hidden_size\": 256, \"pred_len\": ${PRED_LEN}, \"horizon\": ${PRED_LEN}, \"k\": 1, \"loss\": \"MAE\", \"lr\": 0.01, \"lradj\": \"type1\", \"n_heads\": 2, \"norm\": true, \"num_epochs\": 100, \"num_experts\": 4, \"patch_len\": 48, \"patience\": 10, \"seq_len\": ${SEQ_LEN}}" \
#  --deterministic "full" \
#  --gpus 0 \
#  --num-workers 1 \
#  --timeout 60000 \
#  --save-path "experimentos/experimento_global/${SALVAR_EM}" \
#  --save-true-pred True

#extrair_previsoes "${SALVAR_EM}"

# ==================================================
# TIMESNET
# ==================================================
#SALVAR_EM="TimesNet"
#rm -rf "${RESULT_ROOT}/${SALVAR_EM}" "${PREV_ROOT}/${SALVAR_EM}"
#mkdir -p "${RESULT_ROOT}/${SALVAR_EM}" "${PREV_ROOT}/${SALVAR_EM}"

#echo "=================================================="
#echo "TIMESNET | dataset=${DATA_NAME}"
#echo "=================================================="

#$PYTHON_BIN ./scripts/run_benchmark.py \
#  --config-path "rolling_forecast_config.json" \
#  --data-name-list "${DATA_NAME}" \
#  --strategy-args "{\"horizon\": ${PRED_LEN}, \"train_ratio_in_tv\": 1.0}" \
#  --model-name "time_series_library.TimesNet" \
#  --model-hyper-params "{\"d_ff\": 512, \"d_model\": 256, \"factor\": 3, \"pred_len\": ${PRED_LEN}, \"horizon\": ${PRED_LEN}, \"norm\": true, \"seq_len\": ${SEQ_LEN}, \"top_k\": 5}" \
#  --adapter "transformer_adapter" \
#  --gpus 0 \
#  --num-workers 1 \
#  --timeout 60000 \
#  --save-path "experimentos/experimento_global/${SALVAR_EM}" \
#  --save-true-pred True

#extrair_previsoes "${SALVAR_EM}"

# ==================================================
# TIOMS + MSE
# ==================================================
SALVAR_EM="TIOMS_MSE"
rm -rf "${RESULT_ROOT}/${SALVAR_EM}" "${PREV_ROOT}/${SALVAR_EM}"
mkdir -p "${RESULT_ROOT}/${SALVAR_EM}" "${PREV_ROOT}/${SALVAR_EM}"

echo "=================================================="
echo "TIOMS + MSE | dataset=${DATA_NAME}"
echo "=================================================="

$PYTHON_BIN ./scripts/run_benchmark.py \
  --config-path "rolling_forecast_config.json" \
  --data-name-list "${DATA_NAME}" \
  --data-set-name "user_forecast_ohlcv" \
  --strategy-args "{\"horizon\": ${PRED_LEN}}" \
  --model-name "TIOMS.AttentionAdapterChannelEnc" \
  --model-hyper-params "{\"loss\": \"MSE\", \"embedding_type\": \"nonlinear\", \"embedding_hidden_dim\": 16, \"lag_size\": 7, \"spectral_num_freqs\": 18, \"causal_att\": \"no_self\", \"channel_agg_type\": \"none\", \"seq_len\": ${SEQ_LEN}, \"pred_len\": ${PRED_LEN}, \"horizon\": ${PRED_LEN}, \"label_len\": ${LABEL_LEN}, \"batch_size\": 32, \"dropout\": 0.1, \"eps\": 1e-5, \"d_model\": 32, \"n_heads\": 4, \"ff_dim\": 128, \"channel_n_heads\": 60, \"patience\": 8, \"num_epochs\": 60, \"temporal_pool_type\": \"last\"}" \
  --deterministic "full" \
  --eval-backend sequential \
  --num-workers 1 \
  --timeout 60000 \
  --save-path "experimentos/experimento_global/${SALVAR_EM}" \
  --save-true-pred False

extrair_previsoes "${SALVAR_EM}"

# ==================================================
# TIOMS + DILATE
# ==================================================
SALVAR_EM="TIOMS_DILATE"
rm -rf "${RESULT_ROOT}/${SALVAR_EM}" "${PREV_ROOT}/${SALVAR_EM}"
mkdir -p "${RESULT_ROOT}/${SALVAR_EM}" "${PREV_ROOT}/${SALVAR_EM}"

echo "=================================================="
echo "TIOMS + DILATE | dataset=${DATA_NAME}"
echo "=================================================="

$PYTHON_BIN ./scripts/run_benchmark.py \
  --config-path "rolling_forecast_config.json" \
  --data-name-list "${DATA_NAME}" \
  --data-set-name "user_forecast_ohlcv" \
  --strategy-args "{\"horizon\": ${PRED_LEN}}" \
  --model-name "TIOMS.AttentionAdapterChannelEnc" \
  --model-hyper-params "{\"loss\": \"DILATE\", \"dilate_alpha\": 0.5, \"dilate_gamma\": 0.05, \"embedding_type\": \"nonlinear\", \"embedding_hidden_dim\": 16, \"lag_size\": 7, \"spectral_num_freqs\": 18, \"causal_att\": \"no_self\", \"channel_agg_type\": \"none\", \"seq_len\": ${SEQ_LEN}, \"pred_len\": ${PRED_LEN}, \"horizon\": ${PRED_LEN}, \"label_len\": ${LABEL_LEN}, \"batch_size\": 32, \"dropout\": 0.1, \"eps\": 1e-5, \"d_model\": 32, \"n_heads\": 4, \"ff_dim\": 128, \"channel_n_heads\": 60, \"patience\": 8, \"num_epochs\": 60, \"temporal_pool_type\": \"last\"}" \
  --deterministic "full" \
  --eval-backend sequential \
  --num-workers 1 \
  --timeout 60000 \
  --save-path "experimentos/experimento_global/${SALVAR_EM}" \
  --save-true-pred False

extrair_previsoes "${SALVAR_EM}"

echo "Experimento global finalizado."