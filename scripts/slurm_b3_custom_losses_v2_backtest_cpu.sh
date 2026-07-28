#!/bin/bash
#SBATCH -p medusas_shr
#SBATCH --array=0-15%8
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name=b3-v2-bt
#SBATCH --output=/sonic_home/igor.viveiros/src/TFB/logs/b3-v2-bt-%A_%a.out
#SBATCH --error=/sonic_home/igor.viveiros/src/TFB/logs/b3-v2-bt-%A_%a.err

set -euo pipefail

TFB_ROOT="${TFB_ROOT:-/sonic_home/igor.viveiros/src/TFB}"
CLEAN_ROOT="${CLEAN_ROOT:-/sonic_home/igor.viveiros/clean}"
PYTHON_BIN="${PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"
EXPERIMENT_ID="${EXPERIMENT_ID:?EXPERIMENT_ID não definido}"
N_BACKTEST_SHARDS="${N_BACKTEST_SHARDS:-16}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
MAX_ASSETS="${MAX_ASSETS:-9}"
ONLY_POSITIVE_PRED="${ONLY_POSITIVE_PRED:-true}"
ANNUAL_RF="${ANNUAL_RF:-0.043}"
RETURNS_MODE="${RETURNS_MODE:-step}"

if (( SHARD_INDEX >= N_BACKTEST_SHARDS )); then
  echo "SHARD_INDEX=$SHARD_INDEX fora de N_BACKTEST_SHARDS=$N_BACKTEST_SHARDS."
  exit 0
fi

PRED_ROOT="$CLEAN_ROOT/previsoes/tfb_custom_losses_v2/$EXPERIMENT_ID"
OUTPUT_DIR="$CLEAN_ROOT/simulacoes/tfb_custom_losses_v2/$EXPERIMENT_ID"
PRICE_PATH="$TFB_ROOT/dataset/forecasting/b3_daily_tfb.csv"

mkdir -p "$TFB_ROOT/logs" "$OUTPUT_DIR"
cd "$CLEAN_ROOT"
printf 'BACKTEST_SHARD=%s/%s PRED_ROOT=%s OUTPUT_DIR=%s\n' \
  "$SHARD_INDEX" "$N_BACKTEST_SHARDS" "$PRED_ROOT" "$OUTPUT_DIR"

"$PYTHON_BIN" utils/run_tfb_custom_loss_backtests.py \
  --pred-root "$PRED_ROOT" \
  --price-path "$PRICE_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --model-output auto \
  --returns-mode "$RETURNS_MODE" \
  --max-assets "$MAX_ASSETS" \
  --only-positive-pred "$ONLY_POSITIVE_PRED" \
  --annual-rf "$ANNUAL_RF" \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$N_BACKTEST_SHARDS"
