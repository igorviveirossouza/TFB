#!/bin/bash
# Submete o experimento completo: estimação GPU em array + conversão em array.
#
# Uso inicial (MSE, grade idêntica ao primeiro exercício):
#   bash scripts/submit_b3_cumulative_target_pipeline.sh
#
# Futuro, várias losses:
#   LOSSES=mse,ranknet,listnet bash scripts/submit_b3_cumulative_target_pipeline.sh
#
# Outro cluster/partição:
#   PARTITION=minha_particao bash scripts/submit_b3_cumulative_target_pipeline.sh

set -euo pipefail

TFB_ROOT="${TFB_ROOT:-/sonic_home/igor.viveiros/src/TFB}"
PARTITION="${PARTITION:-medusas_shr}"
PRED_ROOT="${PRED_ROOT:-/snfs2/igor.viveiros/previsoes}"
EXPERIMENT_ID="${EXPERIMENT_ID:-b3_cumulative_target_$(date +%Y%m%d_%H%M%S)}"

MODELS="${MODELS:-DUET,TimesNet,FEDformer,Nonstationary_Transformer}"
LOSSES="${LOSSES:-mse}"
DATASETS="${DATASETS:-simple_return,log_return,price}"
SEQ_LENS="${SEQ_LENS:-32,104,246}"
HORIZONS="${HORIZONS:-1,5,10,15,24}"
SEED="${SEED:-2021}"

MAX_PARALLEL="${MAX_PARALLEL:-6}"
MAX_ARRAY_TASKS="${MAX_ARRAY_TASKS:-1000}"
N_CONVERT_SHARDS="${N_CONVERT_SHARDS:-16}"
CONVERT_PARALLEL="${CONVERT_PARALLEL:-8}"
RUN_CONVERSION="${RUN_CONVERSION:-true}"

count_csv() {
  local value="$1"
  awk -F',' '{print NF}' <<< "$value"
}

N_MODELS=$(count_csv "$MODELS")
N_LOSSES=$(count_csv "$LOSSES")
N_DATASETS=$(count_csv "$DATASETS")
N_SEQS=$(count_csv "$SEQ_LENS")
N_HORIZONS=$(count_csv "$HORIZONS")
TOTAL=$((N_MODELS * N_LOSSES * N_DATASETS * N_SEQS * N_HORIZONS))

if (( TOTAL < 1 )); then
  echo "Grade vazia." >&2
  exit 2
fi
if (( MAX_ARRAY_TASKS < 1 || MAX_ARRAY_TASKS > 1000 )); then
  echo "MAX_ARRAY_TASKS deve estar entre 1 e 1000." >&2
  exit 3
fi
if (( N_CONVERT_SHARDS < 1 || N_CONVERT_SHARDS > 1000 )); then
  echo "N_CONVERT_SHARDS deve estar entre 1 e 1000." >&2
  exit 4
fi

cd "$TFB_ROOT"
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$CURRENT_BRANCH" != "experiment/b3-loss-clean-v1" ]; then
  echo "ERRO: branch atual=$CURRENT_BRANCH; esperado experiment/b3-loss-clean-v1" >&2
  exit 5
fi

mkdir -p "$TFB_ROOT/logs"

# As variáveis são herdadas pelos jobs. Evitamos --export com listas contendo vírgulas.
export TFB_ROOT PRED_ROOT EXPERIMENT_ID MODELS LOSSES DATASETS SEQ_LENS HORIZONS SEED
export N_CONVERT_SHARDS

printf '============================================================\n'
printf 'Experimento: %s\n' "$EXPERIMENT_ID"
printf 'Partição:    %s\n' "$PARTITION"
printf 'Modelos:     %s\n' "$MODELS"
printf 'Losses:      %s\n' "$LOSSES"
printf 'Datasets:    %s\n' "$DATASETS"
printf 'Lookbacks:   %s\n' "$SEQ_LENS"
printf 'Horizontes:  %s\n' "$HORIZONS"
printf 'Seed:        %s\n' "$SEED"
printf 'Total runs:  %s\n' "$TOTAL"
printf '============================================================\n'

ESTIMATION_JOB_IDS=()
OFFSET=0
while (( OFFSET < TOTAL )); do
  REMAINING=$((TOTAL - OFFSET))
  CHUNK_SIZE=$MAX_ARRAY_TASKS
  if (( REMAINING < CHUNK_SIZE )); then
    CHUNK_SIZE=$REMAINING
  fi
  LAST_LOCAL=$((CHUNK_SIZE - 1))

  export TASK_OFFSET="$OFFSET"
  JOB_ID=$(sbatch --parsable \
    -p "$PARTITION" \
    --array="0-${LAST_LOCAL}%${MAX_PARALLEL}" \
    scripts/slurm_b3_cumulative_target_gpu.sh)
  JOB_ID="${JOB_ID%%;*}"
  ESTIMATION_JOB_IDS+=("$JOB_ID")
  echo "Estimação: job=$JOB_ID offset=$OFFSET tamanho=$CHUNK_SIZE"
  OFFSET=$((OFFSET + CHUNK_SIZE))
done

if [[ "${RUN_CONVERSION,,}" =~ ^(true|1|yes|y|sim|s)$ ]]; then
  DEPENDENCY=$(IFS=:; echo "${ESTIMATION_JOB_IDS[*]}")
  unset TASK_OFFSET
  CONVERT_LAST=$((N_CONVERT_SHARDS - 1))
  CONVERT_JOB_ID=$(sbatch --parsable \
    -p "$PARTITION" \
    --dependency="afterok:${DEPENDENCY}" \
    --array="0-${CONVERT_LAST}%${CONVERT_PARALLEL}" \
    scripts/slurm_b3_cumulative_target_convert.sh)
  CONVERT_JOB_ID="${CONVERT_JOB_ID%%;*}"
  echo "Conversão: job=$CONVERT_JOB_ID dependency=afterok:${DEPENDENCY}"
else
  CONVERT_JOB_ID="não submetido"
fi

printf '\nRESULTADOS TFB: %s/result/b3_cumulative_target/%s\n' "$TFB_ROOT" "$EXPERIMENT_ID"
printf 'PREVISÕES:      %s/tfb_cumulative_target/%s\n' "$PRED_ROOT" "$EXPERIMENT_ID"
printf 'JOBS ESTIMAÇÃO: %s\n' "${ESTIMATION_JOB_IDS[*]}"
printf 'JOB CONVERSÃO:  %s\n' "$CONVERT_JOB_ID"
