#!/bin/bash
#SBATCH -p medusas_shr
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=b3-v2-stat
#SBATCH --output=/sonic_home/igor.viveiros/src/TFB/logs/b3-v2-stat-%j.out
#SBATCH --error=/sonic_home/igor.viveiros/src/TFB/logs/b3-v2-stat-%j.err

set -euo pipefail

TFB_ROOT="${TFB_ROOT:-/sonic_home/igor.viveiros/src/TFB}"
CLEAN_ROOT="${CLEAN_ROOT:-/sonic_home/igor.viveiros/clean}"
PYTHON_BIN="${PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"
EXPERIMENT_ID="${EXPERIMENT_ID:?EXPERIMENT_ID não definido}"
EXPECTED_RUNS="${EXPECTED_RUNS:?EXPECTED_RUNS não definido}"
BACKTEST_ROOT="$CLEAN_ROOT/simulacoes/tfb_custom_losses_v2/$EXPERIMENT_ID"
STATS_ROOT="$BACKTEST_ROOT/estatisticas"

cd "$TFB_ROOT"
"$PYTHON_BIN" scripts/finalize_tfb_custom_losses_v2.py \
  --backtest-root "$BACKTEST_ROOT" \
  --stats-root "$STATS_ROOT" \
  --expected-runs "$EXPECTED_RUNS" \
  --strict

echo "Estatísticas finais: $STATS_ROOT"
