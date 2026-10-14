#!/usr/bin/env bash
# Edite a grade, os hiperparâmetros e o cluster diretamente neste arquivo.
# Use: bash scripts/run_var_experiment.sh --dry-run
set -euo pipefail

EXPERIMENT="var"

# -----------------------------------------------------------------------------
# GRADE E HIPERPARÂMETROS DO VAR
# -----------------------------------------------------------------------------
TRAIN_PERCENTAGES=(70 80 90)
HORIZONS=(1 5 10 15 20 25)
VAR_LAGS=32
SEED=2026
STRIDE=1
NUM_ROLLINGS=48000

# -----------------------------------------------------------------------------
# CAMINHOS E CLUSTER — edite estes valores neste próprio script
# -----------------------------------------------------------------------------
TFB_ROOT="${TFB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PATH="${VENV_PATH:-/sonic_home/igor.viveiros/py310/bin/activate}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_PATH="${DATASET_PATH:-$TFB_ROOT/dataset/forecasting/b3_daily_tfb_new.csv}"
OUT_ROOT="${OUT_ROOT:-/snfs2/igor.viveiros/previsoes/novas_previsoes}"
RESULT_ROOT="${RESULT_ROOT:-/snfs2/igor.viveiros/result/b3_new}"
LOG_ROOT="${LOG_ROOT:-$TFB_ROOT/logs/b3_new}"

PARTITION="${PARTITION:-medusas_dev}"    # Partição do Slurm
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"     # Tempo máximo por job
NODELIST="${NODELIST:-}"                # Ex.: medusa3; vazio = Slurm escolhe
CPUS_PER_TASK="${CPUS_PER_TASK:-3}"
MEMORY="${MEMORY:-}"                    # Ex.: 16G; vazio = padrão da partição
MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-6}"
ARRAY_CHUNK_SIZE="${ARRAY_CHUNK_SIZE:-1000}"
MANTER_CSV="${MANTER_CSV:-FALSE}"

# A execução abaixo é compartilhada; não é necessário editar arquivos Python.
B3_CONFIG_SOURCE="$TFB_ROOT/scripts/run_var_experiment.sh"
source "$TFB_ROOT/scripts/b3_experiment_env.sh"
exec "$PYTHON_BIN" "$TFB_ROOT/scripts/b3_experiment.py" "$EXPERIMENT" "$@"
