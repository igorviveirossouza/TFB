#!/bin/bash
#SBATCH -p gorgonas
#SBATCH --cpus-per-task=32
#SBATCH --output=/sonic_home/igor.viveiros/logs/atencaoSolo-%j.out

source /sonic_home/igor.viveiros/py310/bin/activate
PYTHON_BIN=/sonic_home/igor.viveiros/py310/bin/python

export MPLCONFIGDIR=/tmp/$USER-mpl

cd /sonic_home/igor.viveiros/src/TFB || exit 1

SEQ_LEN=104
PRED_LEN=24
LABEL_LEN=18

DATA_NAME=("b3_daily_tfb.csv" "NASDAQ.csv" "NYSE.csv" "b3_log_returns.csv")

MASKS=("non_causal" "causal" "no_self")

for DATA in "${DATA_NAME[@]}"; do
  for MASK in "${MASKS[@]}"; do
    echo "=================================================="
    echo "ATENÇÃO | dataset=${DATA} | máscara=${MASK}"
    echo "=================================================="

    SALVAR_EM="AtencaoSolo/${DATA}/${MASK}"

    $PYTHON_BIN ./scripts/run_benchmark.py \
      --config-path "rolling_forecast_config.json" \
      --data-name-list "${DATA}" \
      --strategy-args "{\"horizon\": ${PRED_LEN}}" \
      --model-name "TIOMS.AttentionAdapterChannel" \
      --model-hyper-params "{\"loss\": \"TimeWeightedMSE\", \"loss_decay_rate\": 0.5, \"causal_att\": \"${MASK}\", \"channel_agg_type\": \"none\", \"seq_len\": ${SEQ_LEN}, \"pred_len\": ${PRED_LEN}, \"label_len\": ${LABEL_LEN}, \"batch_size\": 32, \"dropout\": 0.1, \"eps\": 1e-5, \"d_model\": 64, \"n_heads\": 8, \"ff_dim\": 128, \"channel_n_heads\": 4, \"patience\": 8 ,\"num_epochs\": 60, \"temporal_pool_type\": \"last\"}" \
      --deterministic "full" \
      --eval-backend sequential \
      --num-workers 1 \
      --timeout 60000 \
      --save-path "experimentos/${SALVAR_EM}" \
      --save-true-pred False
  done
done

echo "Experimentos finalizados"