#!/bin/bash
#SBATCH -p medusas_shr
#SBATCH --array=0-15%8
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name=b3-v2-conv
#SBATCH --output=/sonic_home/igor.viveiros/src/TFB/logs/b3-v2-conv-%A_%a.out
#SBATCH --error=/sonic_home/igor.viveiros/src/TFB/logs/b3-v2-conv-%A_%a.err

set -euo pipefail

TFB_ROOT="${TFB_ROOT:-/sonic_home/igor.viveiros/src/TFB}"
CLEAN_ROOT="${CLEAN_ROOT:-/sonic_home/igor.viveiros/clean}"
PYTHON_BIN="${PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"
EXPERIMENT_ID="${EXPERIMENT_ID:?EXPERIMENT_ID não definido}"
N_CONVERT_SHARDS="${N_CONVERT_SHARDS:-16}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

if (( SHARD_INDEX >= N_CONVERT_SHARDS )); then
  echo "SHARD_INDEX=$SHARD_INDEX fora de N_CONVERT_SHARDS=$N_CONVERT_SHARDS."
  exit 0
fi

cd "$TFB_ROOT"
if [ -f "$TFB_ROOT/dataset/forecasting/b3_returns.csv" ]; then
  SIMPLE_RETURN_PATH="$TFB_ROOT/dataset/forecasting/b3_returns.csv"
elif [ -f "$TFB_ROOT/dataset/forecasting/b3_daily_return.csv" ]; then
  SIMPLE_RETURN_PATH="$TFB_ROOT/dataset/forecasting/b3_daily_return.csv"
else
  echo "Base de retorno simples ausente." >&2
  exit 3
fi

TFB_RESULT_ROOT="$TFB_ROOT/result/b3_custom_losses_v2/$EXPERIMENT_ID"
OUTPUT_ROOT="$CLEAN_ROOT/previsoes/tfb_custom_losses_v2/$EXPERIMENT_ID"
DATASET_MAP=$(cat <<EOF
{"log_return": "$TFB_ROOT/dataset/forecasting/b3_log_returns.csv", "simple_return": "$SIMPLE_RETURN_PATH", "price": "$TFB_ROOT/dataset/forecasting/b3_daily_tfb.csv"}
EOF
)

mkdir -p "$TFB_ROOT/logs" "$OUTPUT_ROOT"
printf 'CONVERSION_SHARD=%s/%s TFB_RESULT_ROOT=%s OUTPUT_ROOT=%s\n' \
  "$SHARD_INDEX" "$N_CONVERT_SHARDS" "$TFB_RESULT_ROOT" "$OUTPUT_ROOT"

"$PYTHON_BIN" scripts/convert_tfb_custom_losses_v2.py \
  --clean-root "$CLEAN_ROOT" \
  --tfb-result-root "$TFB_RESULT_ROOT" \
  --dataset-path "$TFB_ROOT/dataset/forecasting/b3_log_returns.csv" \
  --dataset-map-json "$DATASET_MAP" \
  --calendar-path "$TFB_ROOT/dataset/forecasting/b3_daily_tfb.csv" \
  --output-root "$OUTPUT_ROOT" \
  --overwrite \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$N_CONVERT_SHARDS"
