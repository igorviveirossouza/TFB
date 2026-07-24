#!/bin/bash
#SBATCH -p medusas_shr
#SBATCH --gres=gpu:1
#SBATCH --array=0-127%4
#SBATCH --time=24:00:00
#SBATCH --job-name=b3-loss-t3
#SBATCH --output=/sonic_home/igor.viveiros/src/TFB/logs/b3-loss-t3-%A_%a.out
#SBATCH --error=/sonic_home/igor.viveiros/src/TFB/logs/b3-loss-t3-%A_%a.err

set -euo pipefail

TFB_ROOT="${TFB_ROOT:-/sonic_home/igor.viveiros/src/TFB}"
cd "$TFB_ROOT"

# Grade padrão do Teste 3:
# 4 modelos x 8 losses x 4 loss_k = 128 jobs.
MODELS_CSV="${MODELS:-duet,timesnet,fedformer,nonstationary}"
LOSSES_CSV="${LOSSES:-rank_hinge,rank_margin,rank_bpr,ranknet,whr1,whr2,listnet,fingat}"
LOSS_KS_CSV="${LOSS_KS:-1,5,10,24}"

IFS=',' read -r -a MODEL_ARR <<< "$MODELS_CSV"
IFS=',' read -r -a LOSS_ARR <<< "$LOSSES_CSV"
IFS=',' read -r -a LOSS_K_ARR <<< "$LOSS_KS_CSV"

N_MODELS=${#MODEL_ARR[@]}
N_LOSSES=${#LOSS_ARR[@]}
N_KS=${#LOSS_K_ARR[@]}
TOTAL=$((N_MODELS * N_LOSSES * N_KS))
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if (( TASK_ID >= TOTAL )); then
  echo "TASK_ID=$TASK_ID fora da grade TOTAL=$TOTAL. Encerrando."
  exit 0
fi

K_IDX=$((TASK_ID % N_KS))
LOSS_IDX=$(((TASK_ID / N_KS) % N_LOSSES))
MODEL_IDX=$((TASK_ID / (N_KS * N_LOSSES)))

export MODEL="${MODEL_ARR[$MODEL_IDX]}"
export LOSS="${LOSS_ARR[$LOSS_IDX]}"
export LOSS_K="${LOSS_K_ARR[$K_IDX]}"

export DATA_NAME="${DATA_NAME:-b3_log_returns.csv}"
export LOSS_DATA_KIND="${LOSS_DATA_KIND:-log_return}"
export LOSS_SCORE_KIND="${LOSS_SCORE_KIND:-log_return}"

export SEQ_LEN="${SEQ_LEN:-32}"
export HORIZON="${HORIZON:-24}"
export NUM_EPOCHS="${NUM_EPOCHS:-20}"
export NUM_ROLLINGS="${NUM_ROLLINGS:-512}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export PATIENCE="${PATIENCE:-5}"
export LR="${LR:-0.001}"

export LOSS_RANK_LAMBDA="${LOSS_RANK_LAMBDA:-1.0}"
export LOSS_MARGIN="${LOSS_MARGIN:-0.01}"
export LOSS_HINGE_MARGIN="${LOSS_HINGE_MARGIN:-$LOSS_MARGIN}"
export LOSS_WHR_MARGIN="${LOSS_WHR_MARGIN:-$LOSS_MARGIN}"
export LOSS_RANKNET_ALPHA="${LOSS_RANKNET_ALPHA:-1.0}"
export LOSS_LISTNET_TAU="${LOSS_LISTNET_TAU:-0.01}"
export LOSS_FINGAT_DELTA="${LOSS_FINGAT_DELTA:-0.01}"
export LOSS_FINGAT_MARGIN="${LOSS_FINGAT_MARGIN:-0.0}"
export LOSS_FINGAT_MOVE_LOGIT_SCALE="${LOSS_FINGAT_MOVE_LOGIT_SCALE:-0.01}"
export LOSS_INVERSE_NORM="${LOSS_INVERSE_NORM:-true}"
export NORM="${NORM:-true}"

export SAVE_ROOT="${SAVE_ROOT:-b3_custom_loss_test3}"
RUN_NAME="${MODEL}_${LOSS}_${LOSS_DATA_KIND}_lb${SEQ_LEN}_h${HORIZON}_k${LOSS_K}_seed${SEED:-2021}"
export SAVE_PATH="${SAVE_ROOT}/${RUN_NAME}"

printf 'TEST3_TASK_ID: %s/%s\n' "$TASK_ID" "$TOTAL"
printf 'MODEL_IDX=%s LOSS_IDX=%s K_IDX=%s\n' "$MODEL_IDX" "$LOSS_IDX" "$K_IDX"
printf 'MODEL=%s LOSS=%s LOSS_K=%s\n' "$MODEL" "$LOSS" "$LOSS_K"
printf 'SAVE_PATH=%s\n' "$SAVE_PATH"

bash scripts/slurm_b3_custom_loss_pilot_gpu.sh
