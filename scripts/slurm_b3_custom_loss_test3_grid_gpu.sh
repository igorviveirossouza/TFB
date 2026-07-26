#!/bin/bash
#SBATCH -p medusas_shr
#SBATCH --gres=gpu:1
#SBATCH --array=0-959%4
#SBATCH --time=24:00:00
#SBATCH --job-name=b3-loss-t3
#SBATCH --output=/sonic_home/igor.viveiros/src/TFB/logs/b3-loss-t3-%A_%a.out
#SBATCH --error=/sonic_home/igor.viveiros/src/TFB/logs/b3-loss-t3-%A_%a.err

set -euo pipefail

TFB_ROOT="${TFB_ROOT:-/sonic_home/igor.viveiros/src/TFB}"
cd "$TFB_ROOT"

# Grade padrão do Teste 3, agora com múltiplas bases.
# Ativo por padrão:
#   2 bases x 4 modelos x 8 losses x 3 lookbacks x 5 horizontes = 960 jobs.
# A base log_return já foi rodada e fica comentada para não gastar GPU de novo.
# Para rodar tudo novamente, descomente a linha log_return abaixo e submeta com:
#   sbatch --array=0-1439%4 scripts/slurm_b3_custom_loss_test3_grid_gpu.sh
#
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

# Formato: dataset_label:data_file_candidates:loss_data_kind:loss_score_kind
# data_file_candidates aceita alternativas separadas por "|"; o script usa a primeira existente.
DATASET_SPECS=(
  # "log_return:b3_log_returns.csv:log_return:log_return"  # já rodado; descomente só se quiser refazer tudo
  "simple_return:b3_returns.csv|b3_daily_return.csv:simple_return:simple_return"
  "price:b3_daily_tfb.csv:price:log_return"
)

IFS=',' read -r -a MODEL_ARR <<< "$MODELS_CSV"
IFS=',' read -r -a LOSS_ARR <<< "$LOSSES_CSV"
IFS=',' read -r -a SEQ_LEN_ARR <<< "$SEQ_LENS_CSV"
IFS=',' read -r -a HORIZON_ARR <<< "$HORIZONS_CSV"

N_DATASETS=${#DATASET_SPECS[@]}
N_MODELS=${#MODEL_ARR[@]}
N_LOSSES=${#LOSS_ARR[@]}
N_SEQ_LENS=${#SEQ_LEN_ARR[@]}
N_HORIZONS=${#HORIZON_ARR[@]}
TOTAL=$((N_DATASETS * N_MODELS * N_LOSSES * N_SEQ_LENS * N_HORIZONS))
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if (( TASK_ID >= TOTAL )); then
  echo "TASK_ID=$TASK_ID fora da grade TOTAL=$TOTAL. Encerrando."
  exit 0
fi

H_IDX=$((TASK_ID % N_HORIZONS))
SEQ_IDX=$(((TASK_ID / N_HORIZONS) % N_SEQ_LENS))
LOSS_IDX=$(((TASK_ID / (N_HORIZONS * N_SEQ_LENS)) % N_LOSSES))
MODEL_IDX=$(((TASK_ID / (N_HORIZONS * N_SEQ_LENS * N_LOSSES)) % N_MODELS))
DATASET_IDX=$((TASK_ID / (N_HORIZONS * N_SEQ_LENS * N_LOSSES * N_MODELS)))

DATASET_SPEC="${DATASET_SPECS[$DATASET_IDX]}"
IFS=':' read -r DATASET_LABEL DATA_FILE_CANDIDATES DATA_KIND SCORE_KIND <<< "$DATASET_SPEC"

DATA_FILE=""
IFS='|' read -r -a DATA_FILE_ARR <<< "$DATA_FILE_CANDIDATES"
for candidate in "${DATA_FILE_ARR[@]}"; do
  if [ -f "dataset/forecasting/$candidate" ]; then
    DATA_FILE="$candidate"
    break
  fi
done

if [ -z "$DATA_FILE" ]; then
  echo "Nenhum arquivo encontrado para dataset=$DATASET_LABEL. Candidatos: $DATA_FILE_CANDIDATES" >&2
  exit 3
fi

export MODEL="${MODEL_ARR[$MODEL_IDX]}"
export LOSS="${LOSS_ARR[$LOSS_IDX]}"
export SEQ_LEN="${SEQ_LEN_ARR[$SEQ_IDX]}"
export HORIZON="${HORIZON_ARR[$H_IDX]}"
export LOSS_K="$HORIZON"

export DATA_NAME="$DATA_FILE"
export LOSS_DATA_KIND="$DATA_KIND"
export LOSS_SCORE_KIND="$SCORE_KIND"

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
printf 'DATASET_IDX=%s MODEL_IDX=%s LOSS_IDX=%s SEQ_IDX=%s H_IDX=%s\n' "$DATASET_IDX" "$MODEL_IDX" "$LOSS_IDX" "$SEQ_IDX" "$H_IDX"
printf 'DATASET_LABEL=%s DATA_NAME=%s LOSS_DATA_KIND=%s LOSS_SCORE_KIND=%s\n' "$DATASET_LABEL" "$DATA_NAME" "$LOSS_DATA_KIND" "$LOSS_SCORE_KIND"
printf 'MODEL=%s LOSS=%s SEQ_LEN=%s HORIZON=%s LOSS_K=%s\n' "$MODEL" "$LOSS" "$SEQ_LEN" "$HORIZON" "$LOSS_K"
printf 'NUM_ROLLINGS=%s\n' "$NUM_ROLLINGS"
printf 'CLEAR_SAVE_PATH=%s\n' "$CLEAR_SAVE_PATH"
printf 'SAVE_PATH=%s\n' "$SAVE_PATH"

bash scripts/slurm_b3_custom_loss_pilot_gpu.sh
