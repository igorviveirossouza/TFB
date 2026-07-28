#!/bin/bash
# Orquestrador do experimento B3 com custom losses v2.
# Este script apenas submete os jobs; todo processamento ocorre na medusas_shr.

set -euo pipefail

TFB_ROOT="${TFB_ROOT:-/sonic_home/igor.viveiros/src/TFB}"
CLEAN_ROOT="${CLEAN_ROOT:-/sonic_home/igor.viveiros/clean}"
EXPERIMENT_ID="${EXPERIMENT_ID:-$(date +%Y%m%d_%H%M%S)}"
MODELS="${MODELS:-duet,timesnet,fedformer,nonstationary}"
LOSSES="${LOSSES:-mse_path_v2,mse_score_v2,ranknet_v2,ranknet_hybrid_v2,listnet_v2,fingat_v2}"
DATASETS="${DATASETS:-log_return,simple_return,price}"
SEQ_LENS="${SEQ_LENS:-32,104,246}"
HORIZONS="${HORIZONS:-1,5,10,20,24}"
SEEDS="${SEEDS:-2021}"
MAX_GPU_PARALLEL="${MAX_GPU_PARALLEL:-4}"
N_CONVERT_SHARDS="${N_CONVERT_SHARDS:-16}"
MAX_CONVERT_PARALLEL="${MAX_CONVERT_PARALLEL:-8}"
N_BACKTEST_SHARDS="${N_BACKTEST_SHARDS:-16}"
MAX_BACKTEST_PARALLEL="${MAX_BACKTEST_PARALLEL:-8}"
ALLOW_EXISTING_EXPERIMENT="${ALLOW_EXISTING_EXPERIMENT:-false}"

count_csv() {
  local value="$1"
  local array
  IFS=',' read -r -a array <<< "$value"
  echo "${#array[@]}"
}

job_id() {
  local raw="$1"
  echo "${raw%%;*}"
}

cd "$TFB_ROOT"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$BRANCH" != "speed" ]; then
  echo "Branch incorreta: $BRANCH. Faça checkout da speed antes de submeter." >&2
  exit 10
fi

for command in sbatch squeue git; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "Comando obrigatório não encontrado: $command" >&2
    exit 11
  }
done

case "${ALLOW_EXISTING_EXPERIMENT,,}" in
  true|1|yes|y|sim|s) ;;
  *)
    if [ -e "$TFB_ROOT/result/b3_custom_losses_v2/$EXPERIMENT_ID" ] || \
       [ -e "$CLEAN_ROOT/previsoes/tfb_custom_losses_v2/$EXPERIMENT_ID" ] || \
       [ -e "$CLEAN_ROOT/simulacoes/tfb_custom_losses_v2/$EXPERIMENT_ID" ]; then
      echo "EXPERIMENT_ID já possui saídas: $EXPERIMENT_ID" >&2
      echo "Use outro identificador ou ALLOW_EXISTING_EXPERIMENT=true." >&2
      exit 12
    fi
    ;;
esac

N_MODELS="$(count_csv "$MODELS")"
N_LOSSES="$(count_csv "$LOSSES")"
N_DATASETS="$(count_csv "$DATASETS")"
N_SEQS="$(count_csv "$SEQ_LENS")"
N_HORIZONS="$(count_csv "$HORIZONS")"
N_SEEDS="$(count_csv "$SEEDS")"
EXPECTED_RUNS=$((N_MODELS * N_LOSSES * N_DATASETS * N_SEQS * N_HORIZONS * N_SEEDS))
GPU_ARRAY_END=$((EXPECTED_RUNS - 1))
CONVERT_ARRAY_END=$((N_CONVERT_SHARDS - 1))
BACKTEST_ARRAY_END=$((N_BACKTEST_SHARDS - 1))

export TFB_ROOT CLEAN_ROOT EXPERIMENT_ID MODELS LOSSES DATASETS SEQ_LENS HORIZONS SEEDS
export EXPECTED_RUNS N_CONVERT_SHARDS N_BACKTEST_SHARDS

MANIFEST_DIR="$TFB_ROOT/manifests/custom_losses_v2_pipeline/$EXPERIMENT_ID"
mkdir -p "$TFB_ROOT/logs" "$MANIFEST_DIR"

VERIFY_RAW="$(sbatch --parsable --export=ALL scripts/slurm_b3_custom_losses_v2_verify_cpu.sh)"
VERIFY_JOB="$(job_id "$VERIFY_RAW")"

TRAIN_RAW="$(sbatch --parsable --export=ALL \
  --dependency="afterok:$VERIFY_JOB" \
  --array="0-${GPU_ARRAY_END}%${MAX_GPU_PARALLEL}" \
  scripts/slurm_b3_custom_losses_v2_grid_gpu.sh)"
TRAIN_JOB="$(job_id "$TRAIN_RAW")"

CONVERT_RAW="$(sbatch --parsable --export=ALL \
  --dependency="afterok:$TRAIN_JOB" \
  --array="0-${CONVERT_ARRAY_END}%${MAX_CONVERT_PARALLEL}" \
  scripts/slurm_b3_custom_losses_v2_convert_cpu.sh)"
CONVERT_JOB="$(job_id "$CONVERT_RAW")"

VALIDATE_RAW="$(sbatch --parsable --export=ALL \
  --dependency="afterok:$CONVERT_JOB" \
  scripts/slurm_b3_custom_losses_v2_validate_conversion_cpu.sh)"
VALIDATE_JOB="$(job_id "$VALIDATE_RAW")"

BACKTEST_RAW="$(sbatch --parsable --export=ALL \
  --dependency="afterok:$VALIDATE_JOB" \
  --array="0-${BACKTEST_ARRAY_END}%${MAX_BACKTEST_PARALLEL}" \
  scripts/slurm_b3_custom_losses_v2_backtest_cpu.sh)"
BACKTEST_JOB="$(job_id "$BACKTEST_RAW")"

FINAL_RAW="$(sbatch --parsable --export=ALL \
  --dependency="afterok:$BACKTEST_JOB" \
  scripts/slurm_b3_custom_losses_v2_finalize_cpu.sh)"
FINAL_JOB="$(job_id "$FINAL_RAW")"

cat > "$MANIFEST_DIR/pipeline_submission.json" <<EOF
{
  "experiment_id": "$EXPERIMENT_ID",
  "submitted_from_branch": "$BRANCH",
  "submitted_from_commit": "$(git rev-parse HEAD)",
  "expected_runs": $EXPECTED_RUNS,
  "grid": {
    "models": "$MODELS",
    "losses": "$LOSSES",
    "datasets": "$DATASETS",
    "seq_lens": "$SEQ_LENS",
    "horizons": "$HORIZONS",
    "seeds": "$SEEDS"
  },
  "parallelism": {
    "max_gpu_parallel": $MAX_GPU_PARALLEL,
    "conversion_shards": $N_CONVERT_SHARDS,
    "max_conversion_parallel": $MAX_CONVERT_PARALLEL,
    "backtest_shards": $N_BACKTEST_SHARDS,
    "max_backtest_parallel": $MAX_BACKTEST_PARALLEL
  },
  "jobs": {
    "verify": "$VERIFY_JOB",
    "train_gpu_array": "$TRAIN_JOB",
    "convert_cpu_array": "$CONVERT_JOB",
    "validate_conversion": "$VALIDATE_JOB",
    "backtest_cpu_array": "$BACKTEST_JOB",
    "finalize_statistics": "$FINAL_JOB"
  },
  "paths": {
    "tfb_results": "$TFB_ROOT/result/b3_custom_losses_v2/$EXPERIMENT_ID",
    "converted_predictions": "$CLEAN_ROOT/previsoes/tfb_custom_losses_v2/$EXPERIMENT_ID",
    "backtests": "$CLEAN_ROOT/simulacoes/tfb_custom_losses_v2/$EXPERIMENT_ID",
    "statistics": "$CLEAN_ROOT/simulacoes/tfb_custom_losses_v2/$EXPERIMENT_ID/estatisticas"
  }
}
EOF

cat <<EOF
Pipeline v2 submetido.
EXPERIMENT_ID: $EXPERIMENT_ID
TOTAL DE MODELOS/CONFIGURAÇÕES: $EXPECTED_RUNS
VERIFY:      $VERIFY_JOB
TRAIN GPU:   $TRAIN_JOB
CONVERT CPU: $CONVERT_JOB
VALIDATE:    $VALIDATE_JOB
BACKTEST:    $BACKTEST_JOB
FINALIZE:    $FINAL_JOB

Acompanhar:
  squeue -j $VERIFY_JOB,$TRAIN_JOB,$CONVERT_JOB,$VALIDATE_JOB,$BACKTEST_JOB,$FINAL_JOB

Manifesto:
  $MANIFEST_DIR/pipeline_submission.json
EOF
