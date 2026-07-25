#!/bin/bash
#SBATCH -p medusas_shr
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=b3-loss-pilot
#SBATCH --output=/sonic_home/igor.viveiros/src/TFB/logs/b3-loss-pilot-%j.out
#SBATCH --error=/sonic_home/igor.viveiros/src/TFB/logs/b3-loss-pilot-%j.err

set -euo pipefail

TFB_ROOT="${TFB_ROOT:-/sonic_home/igor.viveiros/src/TFB}"
PYTHON_BIN="${PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"
LOG_DIR="$TFB_ROOT/logs"

export MPLCONFIGDIR="/tmp/${USER}-mpl"

# Dataset TFB em formato longo dentro de dataset/forecasting.
DATA_NAME="${DATA_NAME:-b3_log_returns.csv}"

# Modelo: duet | timesnet | fedformer | nonstationary
MODEL="${MODEL:-duet}"

# Loss: rank_hinge | rank_margin | rank_bpr | ranknet | whr1 | whr2 | listnet | fingat
LOSS="${LOSS:-ranknet}"

SEQ_LEN="${SEQ_LEN:-32}"
HORIZON="${HORIZON:-24}"
LOSS_K="${LOSS_K:-$HORIZON}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
PATIENCE="${PATIENCE:-2}"
LR="${LR:-0.001}"
D_MODEL="${D_MODEL:-32}"
D_FF="${D_FF:-64}"
HIDDEN_SIZE="${HIDDEN_SIZE:-32}"
N_HEADS="${N_HEADS:-8}"
NUM_WORKERS="${NUM_WORKERS:-1}"
NUM_CPUS="${NUM_CPUS:-4}"
TIMEOUT="${TIMEOUT:-60000}"
NUM_ROLLINGS="${NUM_ROLLINGS:-32}"
TV_RATIO="${TV_RATIO:-0.8}"
TRAIN_RATIO="${TRAIN_RATIO:-0.875}"
STRIDE="${STRIDE:-1}"
SEED="${SEED:-2021}"
SAVE_TRUE_PRED="${SAVE_TRUE_PRED:-true}"
CLEAR_SAVE_PATH="${CLEAR_SAVE_PATH:-false}"
LOSS_DATA_KIND="${LOSS_DATA_KIND:-log_return}"
LOSS_SCORE_KIND="${LOSS_SCORE_KIND:-log_return}"
LOSS_RANK_LAMBDA="${LOSS_RANK_LAMBDA:-1.0}"
LOSS_MARGIN="${LOSS_MARGIN:-0.01}"
LOSS_HINGE_MARGIN="${LOSS_HINGE_MARGIN:-$LOSS_MARGIN}"
LOSS_WHR_MARGIN="${LOSS_WHR_MARGIN:-$LOSS_MARGIN}"
LOSS_RANKNET_ALPHA="${LOSS_RANKNET_ALPHA:-1.0}"
LOSS_LISTNET_TAU="${LOSS_LISTNET_TAU:-0.01}"
LOSS_FINGAT_DELTA="${LOSS_FINGAT_DELTA:-0.01}"
LOSS_FINGAT_MARGIN="${LOSS_FINGAT_MARGIN:-0.0}"
LOSS_FINGAT_MOVE_LOGIT_SCALE="${LOSS_FINGAT_MOVE_LOGIT_SCALE:-0.01}"
LOSS_INVERSE_NORM="${LOSS_INVERSE_NORM:-true}"
NORM="${NORM:-true}"

json_bool() {
  case "${1,,}" in
    true|1|yes|y|sim|s) echo "true" ;;
    false|0|no|n|nao|não) echo "false" ;;
    *) echo "Valor booleano inválido: $1" >&2; exit 4 ;;
  esac
}

SAVE_TRUE_PRED_JSON="$(json_bool "$SAVE_TRUE_PRED")"
LOSS_INVERSE_NORM_JSON="$(json_bool "$LOSS_INVERSE_NORM")"
NORM_JSON="$(json_bool "$NORM")"

mkdir -p "$LOG_DIR" "$MPLCONFIGDIR"
cd "$TFB_ROOT"

case "$MODEL" in
  duet)
    MODEL_NAME="duet.duet.DUET"
    ADAPTER="None"
    EXTRA_HPARAMS=', "num_experts": 2, "noisy_gating": false, "k": 1, "CI": true'
    ;;
  timesnet)
    MODEL_NAME="time_series_library.TimesNet"
    ADAPTER="transformer_adapter"
    EXTRA_HPARAMS=', "top_k": 2, "num_kernels": 2'
    ;;
  fedformer)
    MODEL_NAME="time_series_library.FEDformer"
    ADAPTER="transformer_adapter"
    EXTRA_HPARAMS=', "moving_avg": 3'
    ;;
  nonstationary)
    MODEL_NAME="time_series_library.Nonstationary_Transformer"
    ADAPTER="transformer_adapter"
    EXTRA_HPARAMS=''
    ;;
  *)
    echo "Modelo desconhecido: $MODEL" >&2
    exit 2
    ;;
esac

if [ ! -f "dataset/forecasting/$DATA_NAME" ]; then
  echo "Arquivo não encontrado: $TFB_ROOT/dataset/forecasting/$DATA_NAME" >&2
  exit 3
fi

RUN_NAME="${MODEL}_${LOSS}_${LOSS_DATA_KIND}_lb${SEQ_LEN}_h${HORIZON}_k${LOSS_K}_seed${SEED}"
SAVE_PATH="${SAVE_PATH:-b3_custom_loss_pilot/${RUN_NAME}}"
RESULT_DIR="$TFB_ROOT/result/$SAVE_PATH"

case "${CLEAR_SAVE_PATH,,}" in
  true|1|yes|y|sim|s)
    echo "Limpando resultado anterior: $RESULT_DIR"
    rm -rf "$RESULT_DIR"
    ;;
esac

STRATEGY_ARGS=$(cat <<EOF
{"horizon": $HORIZON, "tv_ratio": $TV_RATIO, "train_ratio_in_tv": {"__default__": $TRAIN_RATIO}, "stride": $STRIDE, "num_rollings": $NUM_ROLLINGS, "seed": $SEED, "save_true_pred": $SAVE_TRUE_PRED_JSON, "target_channel": null}
EOF
)

MODEL_HYPER_PARAMS=$(cat <<EOF
{"batch_size": $BATCH_SIZE, "d_ff": $D_FF, "d_model": $D_MODEL, "hidden_size": $HIDDEN_SIZE, "lr": $LR, "horizon": $HORIZON, "seq_len": $SEQ_LEN, "num_epochs": $NUM_EPOCHS, "patience": $PATIENCE, "n_heads": $N_HEADS, "norm": $NORM_JSON, "loss": "$LOSS", "loss_data_kind": "$LOSS_DATA_KIND", "loss_score_kind": "$LOSS_SCORE_KIND", "loss_k": $LOSS_K, "loss_rank_lambda": $LOSS_RANK_LAMBDA, "loss_margin": $LOSS_MARGIN, "loss_hinge_margin": $LOSS_HINGE_MARGIN, "loss_whr_margin": $LOSS_WHR_MARGIN, "loss_ranknet_alpha": $LOSS_RANKNET_ALPHA, "loss_listnet_tau": $LOSS_LISTNET_TAU, "loss_fingat_delta": $LOSS_FINGAT_DELTA, "loss_fingat_margin": $LOSS_FINGAT_MARGIN, "loss_fingat_move_logit_scale": $LOSS_FINGAT_MOVE_LOGIT_SCALE, "loss_inverse_norm": $LOSS_INVERSE_NORM_JSON, "parallel_strategy": null$EXTRA_HPARAMS}
EOF
)

printf 'HOSTNAME: %s\n' "$(hostname)"
printf 'PYTHON_BIN: %s\n' "$PYTHON_BIN"
printf 'TFB_ROOT: %s\n' "$TFB_ROOT"
printf 'BRANCH: %s\n' "$(git rev-parse --abbrev-ref HEAD)"
printf 'COMMIT: %s\n' "$(git rev-parse HEAD)"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
printf 'DATA_NAME: %s\n' "$DATA_NAME"
printf 'MODEL: %s\n' "$MODEL"
printf 'MODEL_NAME: %s\n' "$MODEL_NAME"
printf 'ADAPTER: %s\n' "$ADAPTER"
printf 'LOSS: %s\n' "$LOSS"
printf 'HORIZON: %s\n' "$HORIZON"
printf 'LOSS_K: %s\n' "$LOSS_K"
printf 'NUM_ROLLINGS: %s\n' "$NUM_ROLLINGS"
printf 'CLEAR_SAVE_PATH: %s\n' "$CLEAR_SAVE_PATH"
printf 'SAVE_PATH: %s\n' "$SAVE_PATH"
printf 'STRATEGY_ARGS: %s\n' "$STRATEGY_ARGS"
printf 'MODEL_HYPER_PARAMS: %s\n' "$MODEL_HYPER_PARAMS"

git status --short

"$PYTHON_BIN" scripts/run_benchmark.py \
  --config-path "rolling_forecast_config.json" \
  --data-name-list "$DATA_NAME" \
  --strategy-args "$STRATEGY_ARGS" \
  --model-name "$MODEL_NAME" \
  --model-hyper-params "$MODEL_HYPER_PARAMS" \
  --adapter "$ADAPTER" \
  --eval-backend sequential \
  --num-workers "$NUM_WORKERS" \
  --num-cpus "$NUM_CPUS" \
  --timeout "$TIMEOUT" \
  --save-true-pred "$SAVE_TRUE_PRED_JSON" \
  --save-path "$SAVE_PATH"

printf 'Resultado em: %s/result/%s\n' "$TFB_ROOT" "$SAVE_PATH"
