#!/bin/bash
#SBATCH -p gorgonas_dev
#SBATCH --time=00:20:00
#SBATCH --job-name=loss-smoke-cpu
#SBATCH --output=/sonic_home/igor.viveiros/paralelo/logs/loss-smoke-cpu-%j.out
#SBATCH --error=/sonic_home/igor.viveiros/paralelo/logs/loss-smoke-cpu-%j.err

set -euo pipefail

TFB_ROOT="/sonic_home/igor.viveiros/src/TFB"
PYTHON_BIN="/sonic_home/igor.viveiros/py310/bin/python"
LOG_DIR="/sonic_home/igor.viveiros/paralelo/logs"

mkdir -p "$LOG_DIR"
cd "$TFB_ROOT"

echo "HOSTNAME: $(hostname)"
echo "PYTHON_BIN: $PYTHON_BIN"
echo "TFB_ROOT: $TFB_ROOT"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-unset}"

git status --short

"$PYTHON_BIN" scripts/smoke_test_custom_losses.py \
  --device cpu \
  --batch-size 4 \
  --horizon 5 \
  --n-assets 66 \
  --top-k 9 \
  --data-kind all \
  --output artifacts/custom_loss_smoke_results_cpu.csv

echo "Resultado CSV: $TFB_ROOT/artifacts/custom_loss_smoke_results_cpu.csv"
