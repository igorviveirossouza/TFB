#!/bin/bash
#SBATCH -p gorgonas
#SBATCH --cpus-per-task=32
#SBATCH --output=/sonic_home/igor.viveiros/logs/TIOMS_band_vs_noband-%j.out

source /sonic_home/igor.viveiros/py310/bin/activate
PYTHON_BIN=/sonic_home/igor.viveiros/py310/bin/python

export MPLCONFIGDIR=/tmp/$USER-mpl

cd /sonic_home/igor.viveiros/src/TFB || exit 1

SEQ_LEN=104
PRED_LEN=24
LABEL_LEN=18
DATA_NAME="ETTh1.csv"

SALVAR_EM="Abblation/ETTh1/last"

AGGREGATORS=(
  "attention"
  "none"
  "linear"
  "residual_gated"
  "linear_residual"
  "linear_lowrank_residual"
  "mlp_mixer"
)

for AGG in "${AGGREGATORS[@]}"; do
  echo "=================================================="
  echo "TIOMS COM BANDAS | channel_agg_type=${AGG}"
  echo "=================================================="

  $PYTHON_BIN ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "${DATA_NAME}" \
    --strategy-args "{\"horizon\": ${PRED_LEN}}" \
    --model-name "TIOMS.LearnableBandWiseAdapterAudit" \
    --model-hyper-params "{\"channel_agg_type\": \"${AGG}\", \"seq_len\": ${SEQ_LEN}, \"pred_len\": ${PRED_LEN}, \"label_len\": ${LABEL_LEN}, \"batch_size\": 32, \"dropout\": 0.1, \"eps\": 1e-5, \"expert_type\": \"attention\", \"expert_d_model\": 64, \"expert_n_heads\": 8, \"expert_ff_dim\": 128, \"channel_n_heads\": 4, \"temporal_pool_type\": \"last\", \"aggregator_type\": \"sum\", \"aggregator_hidden_dim\": 64}" \
    --deterministic "full" \
    --eval-backend sequential \
    --num-workers 1 \
    --timeout 60000 \
    --save-path "experimentos/${SALVAR_EM}" \
    --save-true-pred False

  echo "=================================================="
  echo "TIOMS SEM BANDAS | channel_agg_type=${AGG}"
  echo "=================================================="

  $PYTHON_BIN ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "${DATA_NAME}" \
    --strategy-args "{\"horizon\": ${PRED_LEN}}" \
    --model-name "TIOMS.NoBandWiseAdapterChanel" \
    --model-hyper-params "{\"channel_agg_type\": \"${AGG}\", \"seq_len\": ${SEQ_LEN}, \"pred_len\": ${PRED_LEN}, \"label_len\": ${LABEL_LEN}, \"batch_size\": 32, \"dropout\": 0.1, \"eps\": 1e-5, \"expert_type\": \"attention\", \"expert_d_model\": 64, \"expert_n_heads\": 8, \"expert_ff_dim\": 128, \"channel_n_heads\": 4, \"temporal_pool_type\": \"last\", \"channel_rank\": 8, \"channel_mlp_hidden_mult\": 2}" \
    --deterministic "full" \
    --eval-backend sequential \
    --num-workers 1 \
    --timeout 60000 \
    --save-path "experimentos/${SALVAR_EM}" \
    --save-true-pred False
done

echo "Experimentos finalizados"