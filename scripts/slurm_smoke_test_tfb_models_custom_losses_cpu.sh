#!/bin/bash
#SBATCH -p gorgonas_dev
#SBATCH --time=01:30:00
#SBATCH --job-name=tfb-loss-smoke
#SBATCH --output=/sonic_home/igor.viveiros/src/TFB/logs/tfb-loss-smoke-%j.out
#SBATCH --error=/sonic_home/igor.viveiros/src/TFB/logs/tfb-loss-smoke-%j.err

set -euo pipefail

TFB_ROOT="/sonic_home/igor.viveiros/src/TFB"
PYTHON_BIN="/sonic_home/igor.viveiros/py310/bin/python"
LOG_DIR="$TFB_ROOT/logs"

export CUDA_VISIBLE_DEVICES=""
export MPLCONFIGDIR="/tmp/${USER}-mpl"

MODELS="${MODELS:-all}"
LOSSES="${LOSSES:-ranknet}"
LOSS_K="${LOSS_K:-none}"
N_OBS="${N_OBS:-256}"
N_ASSETS="${N_ASSETS:-24}"
SEQ_LEN="${SEQ_LEN:-16}"
HORIZON="${HORIZON:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
N_HEADS="${N_HEADS:-8}"
OUTPUT="${OUTPUT:-artifacts/tfb_models_custom_loss_smoke_results_cpu.csv}"

mkdir -p "$LOG_DIR" "$MPLCONFIGDIR"
cd "$TFB_ROOT"

printf 'HOSTNAME: %s\n' "$(hostname)"
printf 'PYTHON_BIN: %s\n' "$PYTHON_BIN"
printf 'TFB_ROOT: %s\n' "$TFB_ROOT"
printf 'BRANCH: %s\n' "$(git rev-parse --abbrev-ref HEAD)"
printf 'COMMIT: %s\n' "$(git rev-parse HEAD)"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
printf 'MODELS: %s\n' "$MODELS"
printf 'LOSSES: %s\n' "$LOSSES"
printf 'LOSS_K: %s\n' "$LOSS_K"
printf 'N_OBS: %s\n' "$N_OBS"
printf 'N_ASSETS: %s\n' "$N_ASSETS"
printf 'SEQ_LEN: %s\n' "$SEQ_LEN"
printf 'HORIZON: %s\n' "$HORIZON"
printf 'N_HEADS: %s\n' "$N_HEADS"

git status --short

"$PYTHON_BIN" scripts/smoke_test_tfb_models_custom_losses.py \
  --device cpu \
  --models "$MODELS" \
  --losses "$LOSSES" \
  --data-kind log_return \
  --score-kind log_return \
  --loss-k "$LOSS_K" \
  --n-obs "$N_OBS" \
  --n-assets "$N_ASSETS" \
  --seq-len "$SEQ_LEN" \
  --horizon "$HORIZON" \
  --batch-size "$BATCH_SIZE" \
  --num-epochs "$NUM_EPOCHS" \
  --n-heads "$N_HEADS" \
  --output "$OUTPUT"

printf 'Resultado CSV: %s/%s\n' "$TFB_ROOT" "$OUTPUT"
