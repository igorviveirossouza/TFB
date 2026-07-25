#!/bin/bash
#SBATCH -p medusas_shr
#SBATCH --gres=gpu:1
#SBATCH --array=0-479%4
#SBATCH --time=24:00:00
#SBATCH --job-name=b3-loss-t3
#SBATCH --output=/sonic_home/igor.viveiros/src/TFB/logs/b3-loss-t3-%A_%a.out
#SBATCH --error=/sonic_home/igor.viveiros/src/TFB/logs/b3-loss-t3-%A_%a.err

set -euo pipefail

TFB_ROOT="${TFB_ROOT:-/sonic_home/igor.viveiros/src/TFB}"
cd "$TFB_ROOT"

# Grade padrão do Teste 3:
# 4 modelos x 8 losses x 3 lookbacks x 5 horizontes = 480 jobs.
# Para comparabilidade com o TFB tradicional, sempre impomos:
#   HORIZON = LOSS_K = K
# com K em {1,5,10,20,24}.
# Os contextos/lookbacks seguem os experimentos anteriores: seq_len em {32,104,246}.
# IMPORTANTE: NUM_ROLLINGS é forçado alto para usar todas as janelas disponíveis.
# IMPORTANTE: CLEAR_SAVE_PATH=true evita misturar arquivos antigos de runs parciais no mesmo diretório.
MODELS_CSV="${MODELS:-duet,timesnet,fedformer,nonstationary}"
LOSSES_CSV="${LOSSES:-rank_hinge,rank_margin,rank_bpr,ranknet,whr1,whr2,listnet,fingat}"
SEQ_LENS_CSV="${SEQ_LENS:-32,104,246}"
HORIZONS_CSV="${HORIZONS:-1,5,10,20,24}"

IFS=',' read -r -a MODEL_ARR <<< "$MODELS_CSV"
IFS=',' read -r -a LOSS_ARR <<< "$LOSSES_CSV"
IFS=',' read -r -a SEQ_LEN_ARR <<< "$SEQ_LENS_CSV"
IFS=',' read -r -a HORIZON_ARR <<< "$HORIZONS_CSV"

N_MODELS=${#MODEL_ARR[@]}
N_LOSSES=${#LOSS_ARR[@]}
N_SEQ_LENS=${#SEQ_LEN_ARR[@]}
N_HORIZONS=${#HORIZON_ARR[@]}
TOTAL=$((N_MODELS * N_LOSSES * N_SEQ_LENS * N_HORIZONS))
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if (( TASK_ID >= TOTAL )); then
  echo "TASK_ID=$TASK_ID fora da grade TOTAL=$TOTAL. Encerrando."
  exit 0
fi

H_IDX=$((TASK_ID % N_HORIZONS))
SEQ_IDX=$(((TASK_ID / N_HORIZONS) % N_SEQ_LENS))
LOSS_IDX=$(((TASK_ID / (N_HORIZONS * N_SEQ_LENS)) % N_LOSSES))
MODEL_IDX=$((TASK_ID / (N_HORIZONS * N_SEQ_LENS * N_LOSSES)))

export MODEL="${MODEL_ARR[$MODEL_IDX]}"
export LOSS="${LOSS_ARR[$LOSS_IDX]}"
export SEQ_LEN="${SEQ_LEN_ARR[$SEQ_IDX]}"
export HORIZON="${HORIZON_ARR[$H_IDX]}"
export LOSS_K="$HORIZON"

export DATA_NAME="${DATA_NAME:-b3_log_returns.csv}"
export LOSS_DATA_KIND="${LOSS_DATA_KIND:-log_return}"
export LOSS_SCORE_KIND="${LOSS_SCORE_KIND:-log_return}"

export NUM_EPOCHS="${NUM_EPOCHS:-20}"
# Não herdar NUM_ROLLINGS=512 do ambiente: este experimento deve usar todas as janelas.
export NUM_ROLLINGS="999999"
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
export CLEAR_SAVE_PATH="${CLEAR_SAVE_PATH:-true}"

export SAVE_ROOT="${SAVE_ROOT:-b3_custom_loss_test3}"
RUN_NAME="${MODEL}_${LOSS}_${LOSS_DATA_KIND}_lb${SEQ_LEN}_h${HORIZON}_k${LOSS_K}_seed${SEED:-2021}"
export SAVE_PATH="${SAVE_ROOT}/${RUN_NAME}"

printf 'TEST3_TASK_ID: %s/%s\n' "$TASK_ID" "$TOTAL"
printf 'MODEL_IDX=%s LOSS_IDX=%s SEQ_IDX=%s H_IDX=%s\n' "$MODEL_IDX" "$LOSS_IDX" "$SEQ_IDX" "$H_IDX"
printf 'MODEL=%s LOSS=%s SEQ_LEN=%s HORIZON=%s LOSS_K=%s\n' "$MODEL" "$LOSS" "$SEQ_LEN" "$HORIZON" "$LOSS_K"
printf 'NUM_ROLLINGS=%s\n' "$NUM_ROLLINGS"
printf 'CLEAR_SAVE_PATH=%s\n' "$CLEAR_SAVE_PATH"
printf 'SAVE_PATH=%s\n' "$SAVE_PATH"

bash scripts/slurm_b3_custom_loss_pilot_gpu.sh
