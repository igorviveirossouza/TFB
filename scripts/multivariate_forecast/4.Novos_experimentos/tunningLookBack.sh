#!/bin/bash
#SBATCH -p gorgonas
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=32
#SBATCH --output=/sonic_home/igor.viveiros/logs/tioms-context-window-%j.out

set -euo pipefail

source /sonic_home/igor.viveiros/py310/bin/activate
PYTHON_BIN=/sonic_home/igor.viveiros/py310/bin/python

export MPLCONFIGDIR=/tmp/$USER-mpl
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

cd /sonic_home/igor.viveiros/src/TFB || exit 1

setor="financeiro"

DATA_NAME="b3_daily_${setor}.csv"
PRED_LEN=24
LABEL_LEN=18
MODEL_KEY="TIOMS_ENC_DEC_OHLC_TIMEWei_decai_04"
MODEL_NAME="TIOMS.CrossAttentionAdapterChannelEncDec"
RESULT_ROOT="/sonic_home/igor.viveiros/src/TFB/result/experimento_context_window//${setor}"

# Janelas de contexto a avaliar.
SEQ_LENS=(8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256 512)

#SEQ_LENS=(144)

mkdir -p "$RESULT_ROOT"

for SEQ_LEN in "${SEQ_LENS[@]}"; do
  echo "=================================================="
  echo "Executando: seq_len=${SEQ_LEN} | modelo=${MODEL_KEY}"
  echo "=================================================="

  RESULT_DIR="${RESULT_ROOT}/seq_len_${SEQ_LEN}/${MODEL_KEY}"
  mkdir -p "$RESULT_DIR"

  MODEL_HYPER_PARAMS="{\"loss\": \"TimeWeightedMSE\",  \"loss_decay_rate\": 0.4, \"embedding_type\": \"nonlinear\", \"embedding_hidden_dim\": 16, \"lag_size\": 7, \"spectral_num_freqs\": 18, \"causal_att\": \"non_causal\", \"use_b3_cross_mask\": true, \"channel_agg_type\": \"none\", \"seq_len\": ${SEQ_LEN}, \"pred_len\": ${PRED_LEN}, \"horizon\": ${PRED_LEN}, \"label_len\": ${LABEL_LEN}, \"batch_size\": 4, \"dropout\": 0.1, \"eps\": 1e-5, \"d_model\": 32, \"n_heads\": 4, \"ff_dim\": 128, \"channel_n_heads\": 4, \"patience\": 8, \"num_epochs\": 60}"

  "$PYTHON_BIN" ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "$DATA_NAME" \
    --data-set-name "user_forecast_ohlcv" \
    --strategy-args "{\"horizon\": ${PRED_LEN}}" \
    --model-name "$MODEL_NAME" \
    --model-hyper-params "$MODEL_HYPER_PARAMS" \
    --deterministic "full" \
    --eval-backend sequential \
    --gpus 0 \
    --num-workers 1 \
    --timeout 60000 \
    --save-path "experimento_context_window/financeiro/seq_len_${SEQ_LEN}/${MODEL_KEY}" \
    --save-true-pred False

done

echo "Todos os experimentos de janela de contexto finalizaram."
