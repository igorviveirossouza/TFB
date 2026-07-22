#!/bin/bash
#SBATCH -p gorgonas_dev
#SBATCH --time=00:20:00
#SBATCH --job-name=duet-loss-smoke
#SBATCH --output=/sonic_home/igor.viveiros/paralelo/logs/duet-loss-smoke-%j.out
#SBATCH --error=/sonic_home/igor.viveiros/paralelo/logs/duet-loss-smoke-%j.err

set -euo pipefail

TFB_ROOT="/sonic_home/igor.viveiros/src/TFB"
PYTHON_BIN="/sonic_home/igor.viveiros/py310/bin/python"
LOG_DIR="/sonic_home/igor.viveiros/paralelo/logs"

mkdir -p "$LOG_DIR"
cd "$TFB_ROOT"

echo "HOSTNAME: $(hostname)"
echo "PYTHON_BIN: $PYTHON_BIN"
echo "TFB_ROOT: $TFB_ROOT"
echo "BRANCH: $(git rev-parse --abbrev-ref HEAD)"
echo "COMMIT: $(git rev-parse HEAD)"

git status --short

"$PYTHON_BIN" scripts/smoke_test_duet_custom_loss.py \
  --device cpu \
  --losses ranknet \
  --data-kind log_return \
  --score-kind log_return \
  --n-obs 96 \
  --n-assets 12 \
  --seq-len 16 \
  --horizon 1 \
  --batch-size 4 \
  --num-epochs 1 \
  --output artifacts/duet_custom_loss_smoke_results_cpu.csv

echo "Resultado CSV: $TFB_ROOT/artifacts/duet_custom_loss_smoke_results_cpu.csv"
