#!/bin/bash
#SBATCH -p gorgonas_dev
#SBATCH --time=00:10:00
#SBATCH --job-name=verify-losses
#SBATCH --output=/sonic_home/igor.viveiros/src/TFB/logs/verify-losses-%j.out
#SBATCH --error=/sonic_home/igor.viveiros/src/TFB/logs/verify-losses-%j.err

set -euo pipefail

TFB_ROOT="${TFB_ROOT:-/sonic_home/igor.viveiros/src/TFB}"
PYTHON_BIN="${PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"
LOG_DIR="$TFB_ROOT/logs"

mkdir -p "$LOG_DIR"
cd "$TFB_ROOT"

printf 'HOSTNAME: %s\n' "$(hostname)"
printf 'PYTHON_BIN: %s\n' "$PYTHON_BIN"
printf 'TFB_ROOT: %s\n' "$TFB_ROOT"
printf 'BRANCH: %s\n' "$(git rev-parse --abbrev-ref HEAD)"
printf 'COMMIT: %s\n' "$(git rev-parse HEAD)"

git status --short

"$PYTHON_BIN" scripts/verify_custom_losses_behavior.py --strict
