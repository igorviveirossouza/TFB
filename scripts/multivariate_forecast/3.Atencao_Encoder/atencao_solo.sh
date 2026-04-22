#!/bin/bash
#SBATCH -p gorgonas
#SBATCH --cpus-per-task=32
#SBATCH --output=/sonic_home/igor.viveiros/logs/Losses-%j.out

source /sonic_home/igor.viveiros/py310/bin/activate
PYTHON_BIN=/sonic_home/igor.viveiros/py310/bin/python

export MPLCONFIGDIR=/tmp/$USER-mpl

cd /sonic_home/igor.viveiros/src/TFB || exit 1

SEQ_LEN=104
PRED_LEN=24
LABEL_LEN=18

# DATA_NAME=("b3_daily_tfb.csv" "NASDAQ.csv" "NYSE.csv" "b3_log_returns.csv")
 DATA_NAME=("b3_daily_tfb.csv" )
# MASKS=("non_causal" "causal" "no_self")

MASKS=("no_self")
EMBEDDINGS=("nonlinear")

for DATA in "${DATA_NAME[@]}"; do
  for MASK in "${MASKS[@]}"; do
    for EMB in "${EMBEDDINGS[@]}"; do
      echo "=================================================="
      echo "ATENÇÃO | dataset=${DATA} | máscara=${MASK} | embedding=${EMB}"
      echo "=================================================="

      # SALVAR_EM="AtencaoSolo/Embeddings/${DATA}/${MASK}/${EMB}"
      #SALVAR_EM="AtencaoSolo/Encoder/h24/predicoes"
      SALVAR_EM="experimentos/experimento_global/SemOHLCV"

      $PYTHON_BIN ./scripts/run_benchmark.py \
        --config-path "rolling_forecast_config.json" \
        --data-name-list "${DATA}" \
        --data-set-name "user_forecast_ohlcv" \
        --strategy-args "{\"horizon\": ${PRED_LEN}}" \
        --model-name "TIOMS.AttentionAdapterChannelEnc" \
        --model-hyper-params "{\"loss\": \"DILATE\", \"embedding_type\": \"${EMB}\", \"embedding_hidden_dim\": 16, \"lag_size\": 7, \"spectral_num_freqs\": 18, \"causal_att\": \"${MASK}\", \"channel_agg_type\": \"none\", \"seq_len\": ${SEQ_LEN}, \"pred_len\": ${PRED_LEN}, \"horizon\": ${PRED_LEN}, \"label_len\": ${LABEL_LEN}, \"batch_size\": 32, \"dropout\": 0.1, \"eps\": 1e-5, \"d_model\": 32, \"n_heads\": 4, \"ff_dim\": 128, \"channel_n_heads\": 60, \"patience\": 8, \"num_epochs\": 60, \"temporal_pool_type\": \"last\"}" \        --deterministic "full" \
        --eval-backend sequential \
        --num-workers 1 \
        --timeout 60000 \
        --save-path "experimentos/${SALVAR_EM}" \
        --save-true-pred True
    done
  done
done

echo "Experimentos finalizados"

#echo "Decodificando previsões..."

#RESULT_BASE="result/experimentos/$SALVAR_EM"
#DECODE_SCRIPT="ts_benchmark/utils/decode_prediction.py"

#find "$RESULT_BASE" -type f -name "*.tar.gz" | while read -r tarfile; do
#     echo "Extraindo: $tarfile"

#     extract_dir="${tarfile%.tar.gz}_extracted"
#     mkdir -p "$extract_dir"

#     tar -xzf "$tarfile" -C "$extract_dir"

#     find "$extract_dir" -type f -name "*.csv" | while read -r csvfile; do
#         echo "Decodificando CSV: $csvfile"
#         python "$DECODE_SCRIPT" "$csvfile"
#     done
# done

#echo "Decodificação finalizada"