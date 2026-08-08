#!/bin/bash
#SBATCH -p medusas_shr
#SBATCH --gres=gpu:1
#SBATCH --array=0-39%4
#SBATCH --time=04:00:00
#SBATCH --job-name=smoke-b3-loss-v1
#SBATCH --output=/sonic_home/igor.viveiros/src/TFB/logs/smoke-b3-loss-v1-%A_%a.out
#SBATCH --error=/sonic_home/igor.viveiros/src/TFB/logs/smoke-b3-loss-v1-%A_%a.err

set -euo pipefail

TFB_ROOT="${TFB_ROOT:-/sonic_home/igor.viveiros/src/TFB}"
cd "$TFB_ROOT"
mkdir -p logs

# Smoke pequeno: 1 dataset, 1 lookback, 1 horizonte, 1 época, poucas janelas.
# Testa as 4 arquiteturas TFB e as losses da rodada A + mse_step_accum diagnóstico.
MODELS_CSV="${MODELS:-duet,timesnet,fedformer,nonstationary}"
LOSSES_CSV="${LOSSES:-mse_accum,mse_step_accum,rank_hinge,rank_margin,rank_bpr,ranknet,whr1,whr2,listnet,fingat}"

IFS=',' read -r -a MODEL_ARR <<< "$MODELS_CSV"
IFS=',' read -r -a LOSS_ARR <<< "$LOSSES_CSV"

N_MODELS=${#MODEL_ARR[@]}
N_LOSSES=${#LOSS_ARR[@]}
TOTAL=$((N_MODELS * N_LOSSES))
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if (( TASK_ID >= TOTAL )); then
  echo "TASK_ID=$TASK_ID fora da grade TOTAL=$TOTAL. Encerrando."
  exit 0
fi

LOSS_IDX=$((TASK_ID % N_LOSSES))
MODEL_IDX=$((TASK_ID / N_LOSSES))

export MODEL="${MODEL_ARR[$MODEL_IDX]}"
export LOSS="${LOSS_ARR[$LOSS_IDX]}"
export DATA_NAME="${DATA_NAME:-b3_log_returns.csv}"
export LOSS_DATA_KIND="${LOSS_DATA_KIND:-log_return}"
export LOSS_SCORE_KIND="${LOSS_SCORE_KIND:-log_return}"
export SEQ_LEN="${SEQ_LEN:-32}"
export HORIZON="${HORIZON:-5}"
export LOSS_K="${LOSS_K:-$HORIZON}"
export SEED="${SEED:-2021}"

export NUM_EPOCHS="${NUM_EPOCHS:-1}"
export NUM_ROLLINGS="${NUM_ROLLINGS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export PATIENCE="${PATIENCE:-1}"
export LR="${LR:-0.001}"

export SAVE_ROOT="${SAVE_ROOT:-b3_loss_clean_v1_smoke}"
export CLEAR_SAVE_PATH="${CLEAR_SAVE_PATH:-true}"

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

cat <<EOF
============================================================
SMOKE TFB b3_loss_clean_v1
TASK_ID:         $TASK_ID / $((TOTAL - 1))
MODEL:           $MODEL
LOSS:            $LOSS
DATA_NAME:       $DATA_NAME
SEQ_LEN:         $SEQ_LEN
HORIZON/LOSS_K:  $HORIZON
NUM_EPOCHS:      $NUM_EPOCHS
NUM_ROLLINGS:    $NUM_ROLLINGS
SAVE_ROOT:       $SAVE_ROOT
============================================================
EOF

bash scripts/slurm_b3_custom_loss_pilot_gpu.sh
