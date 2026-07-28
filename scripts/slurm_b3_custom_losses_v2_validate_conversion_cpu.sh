#!/bin/bash
#SBATCH -p medusas_shr
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name=b3-v2-val
#SBATCH --output=/sonic_home/igor.viveiros/src/TFB/logs/b3-v2-val-%j.out
#SBATCH --error=/sonic_home/igor.viveiros/src/TFB/logs/b3-v2-val-%j.err

set -euo pipefail

TFB_ROOT="${TFB_ROOT:-/sonic_home/igor.viveiros/src/TFB}"
CLEAN_ROOT="${CLEAN_ROOT:-/sonic_home/igor.viveiros/clean}"
PYTHON_BIN="${PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"
EXPERIMENT_ID="${EXPERIMENT_ID:?EXPERIMENT_ID não definido}"
EXPECTED_RUNS="${EXPECTED_RUNS:?EXPECTED_RUNS não definido}"
OUTPUT_ROOT="$CLEAN_ROOT/previsoes/tfb_custom_losses_v2/$EXPERIMENT_ID"

cd "$TFB_ROOT"
"$PYTHON_BIN" scripts/validate_tfb_custom_losses_v2_conversion.py \
  --output-root "$OUTPUT_ROOT" \
  --expected-runs "$EXPECTED_RUNS" \
  --strict
