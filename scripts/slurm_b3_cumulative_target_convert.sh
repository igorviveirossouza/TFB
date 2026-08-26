#!/bin/bash
#SBATCH -p medusas_shr
#SBATCH --array=0-15%8
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name=b3-cum-conv
#SBATCH --output=/sonic_home/igor.viveiros/src/TFB/logs/b3-cum-conv-%A_%a.out
#SBATCH --error=/sonic_home/igor.viveiros/src/TFB/logs/b3-cum-conv-%A_%a.err

set -euo pipefail

TFB_ROOT="${TFB_ROOT:-/sonic_home/igor.viveiros/src/TFB}"
CLEAN_ROOT="${CLEAN_ROOT:-/sonic_home/igor.viveiros/clean}"
PRED_ROOT="${PRED_ROOT:-/snfs2/igor.viveiros/previsoes}"
PYTHON_BIN="${PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"
EXPERIMENT_ID="${EXPERIMENT_ID:-b3_cumulative_target_v1}"
N_CONVERT_SHARDS="${N_CONVERT_SHARDS:-16}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

if (( SHARD_INDEX >= N_CONVERT_SHARDS )); then
  echo "SHARD_INDEX=$SHARD_INDEX fora de N_CONVERT_SHARDS=$N_CONVERT_SHARDS."
  exit 0
fi

cd "$TFB_ROOT"
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$CURRENT_BRANCH" != "experiment/b3-loss-clean-v1" ]; then
  echo "ERRO: execute na branch experiment/b3-loss-clean-v1; atual=$CURRENT_BRANCH" >&2
  exit 2
fi

SIMPLE_RETURN_PATH=""
for candidate in \
  "$TFB_ROOT/dataset/forecasting/b3_daily_return.csv" \
  "$TFB_ROOT/dataset/forecasting/b3_returns.csv"; do
  if [ -f "$candidate" ]; then
    SIMPLE_RETURN_PATH="$candidate"
    break
  fi
done
if [ -z "$SIMPLE_RETURN_PATH" ]; then
  echo "Base de retorno simples ausente." >&2
  exit 3
fi

TFB_RESULT_ROOT="$TFB_ROOT/result/b3_cumulative_target/$EXPERIMENT_ID"
OUTPUT_ROOT="$PRED_ROOT/previsoes/tfb_cumulative_target/$EXPERIMENT_ID"
DATASET_MAP=$(cat <<EOF
{"simple_return": "$SIMPLE_RETURN_PATH", "log_return": "$TFB_ROOT/dataset/forecasting/b3_log_returns.csv", "price": "$TFB_ROOT/dataset/forecasting/b3_daily_tfb.csv"}
EOF
)

mkdir -p "$TFB_ROOT/logs" "$OUTPUT_ROOT"
printf 'PARTITION=%s SHARD=%s/%s RESULT_ROOT=%s OUTPUT_ROOT=%s\n' \
  "${SLURM_JOB_PARTITION:-medusas_shr}" "$SHARD_INDEX" "$N_CONVERT_SHARDS" "$TFB_RESULT_ROOT" "$OUTPUT_ROOT"

"$PYTHON_BIN" scripts/convert_tfb_cumulative_target.py \
  --clean-root "$CLEAN_ROOT" \
  --tfb-result-root "$TFB_RESULT_ROOT" \
  --dataset-path "$TFB_ROOT/dataset/forecasting/b3_log_returns.csv" \
  --dataset-map-json "$DATASET_MAP" \
  --calendar-path "$TFB_ROOT/dataset/forecasting/b3_daily_tfb.csv" \
  --output-root "$OUTPUT_ROOT" \
  --overwrite \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$N_CONVERT_SHARDS"
