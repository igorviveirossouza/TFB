#!/bin/bash
#SBATCH -p medusas_shr
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --job-name=loss-smoke
#SBATCH --output=/sonic_home/igor.viveiros/paralelo/logs/loss-smoke-%j.out
#SBATCH --error=/sonic_home/igor.viveiros/paralelo/logs/loss-smoke-%j.err

set -euo pipefail

TFB_ROOT="/sonic_home/igor.viveiros/src/TFB"
PYTHON_BIN="/sonic_home/igor.viveiros/py310/bin/python"
LOG_DIR="/sonic_home/igor.viveiros/paralelo/logs"

mkdir -p "$LOG_DIR"
cd "$TFB_ROOT"

echo "HOSTNAME: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "PYTHON_BIN: $PYTHON_BIN"
echo "TFB_ROOT: $TFB_ROOT"

git status --short

"$PYTHON_BIN" scripts/smoke_test_custom_losses.py \
  --device cuda \
  --batch-size 4 \
  --horizon 5 \
  --n-assets 66 \
  --top-k 9 \
  --data-kind all \
  --output artifacts/custom_loss_smoke_results_gpu.csv

echo "Resultado CSV: $TFB_ROOT/artifacts/custom_loss_smoke_results_gpu.csv"
