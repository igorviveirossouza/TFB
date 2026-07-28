#!/bin/bash
#SBATCH -p gorgonas_dev
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=b3-loss-v2
#SBATCH --output=/sonic_home/igor.viveiros/src/TFB/logs/b3-loss-v2-%j.out
#SBATCH --error=/sonic_home/igor.viveiros/src/TFB/logs/b3-loss-v2-%j.err

set -euo pipefail

TFB_ROOT="${TFB_ROOT:-/sonic_home/igor.viveiros/src/TFB}"
PYTHON_BIN="${PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"
LOG_DIR="$TFB_ROOT/logs"
MANIFEST_DIR="$TFB_ROOT/manifests/custom_losses_v2"
export MPLCONFIGDIR="/tmp/${USER}-mpl"

DATA_NAME="${DATA_NAME:-b3_log_returns.csv}"
MODEL="${MODEL:-duet}"
# mse_path_v2 | mse_score_v2 | ranknet_v2 | ranknet_hybrid_v2 | listnet_v2 | fingat_v2
LOSS="${LOSS:-fingat_v2}"

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
LOSS_DATA_KIND="${LOSS_DATA_KIND:-log_return}"
LOSS_SCORE_KIND="${LOSS_SCORE_KIND:-log_return}"
LOSS_RANK_LAMBDA="${LOSS_RANK_LAMBDA:-0.5}"
LOSS_RANKNET_ALPHA="${LOSS_RANKNET_ALPHA:-1.0}"
LOSS_LISTNET_TAU="${LOSS_LISTNET_TAU:-1.0}"
LOSS_FINGAT_DELTA="${LOSS_FINGAT_DELTA:-0.2}"
LOSS_DIRECTION_SCALE="${LOSS_DIRECTION_SCALE:-0.01}"
LOSS_SCORE_NORMALIZATION="${LOSS_SCORE_NORMALIZATION:-zscore}"
LOSS_HYBRID_POINT_NORMALIZATION="${LOSS_HYBRID_POINT_NORMALIZATION:-target_std}"
LOSS_INVERSE_NORM="${LOSS_INVERSE_NORM:-true}"
NORM="${NORM:-true}"

json_bool() {
  case "${1,,}" in
    true|1|yes|y|sim|s) echo "true" ;;
    false|0|no|n|nao|não) echo "false" ;;
    *) echo "Valor booleano inválido: $1" >&2; exit 4 ;;
  esac
}

case "$LOSS" in
  mse_path_v2|mse_score_v2|ranknet_v2|ranknet_hybrid_v2|listnet_v2|fingat_v2) ;;
  *) echo "Loss v2 desconhecida: $LOSS" >&2; exit 5 ;;
esac

if [ "$HORIZON" -ne "$LOSS_K" ]; then
  echo "Este piloto usa loss_horizon_mode=strict: HORIZON deve ser igual a LOSS_K." >&2
  echo "Recebido HORIZON=$HORIZON e LOSS_K=$LOSS_K." >&2
  exit 6
fi

SAVE_TRUE_PRED_JSON="$(json_bool "$SAVE_TRUE_PRED")"
LOSS_INVERSE_NORM_JSON="$(json_bool "$LOSS_INVERSE_NORM")"
NORM_JSON="$(json_bool "$NORM")"

mkdir -p "$LOG_DIR" "$MANIFEST_DIR" "$MPLCONFIGDIR"
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
  *) echo "Modelo desconhecido: $MODEL" >&2; exit 2 ;;
esac

if [ ! -f "dataset/forecasting/$DATA_NAME" ]; then
  echo "Arquivo não encontrado: $TFB_ROOT/dataset/forecasting/$DATA_NAME" >&2
  exit 3
fi

RUN_NAME="${MODEL}_${LOSS}_${LOSS_DATA_KIND}_lb${SEQ_LEN}_h${HORIZON}_k${LOSS_K}_seed${SEED}"
SAVE_PATH="${SAVE_PATH:-b3_custom_losses_v2_pilot/${RUN_NAME}}"
MANIFEST_PATH="$MANIFEST_DIR/${RUN_NAME}.json"
GIT_COMMIT="$(git rev-parse HEAD)"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"

STRATEGY_ARGS=$(cat <<EOF
{"horizon": $HORIZON, "tv_ratio": $TV_RATIO, "train_ratio_in_tv": {"__default__": $TRAIN_RATIO}, "stride": $STRIDE, "num_rollings": $NUM_ROLLINGS, "seed": $SEED, "save_true_pred": $SAVE_TRUE_PRED_JSON, "target_channel": null}
EOF
)

MODEL_HYPER_PARAMS=$(cat <<EOF
{"batch_size": $BATCH_SIZE, "d_ff": $D_FF, "d_model": $D_MODEL, "hidden_size": $HIDDEN_SIZE, "lr": $LR, "horizon": $HORIZON, "seq_len": $SEQ_LEN, "num_epochs": $NUM_EPOCHS, "patience": $PATIENCE, "n_heads": $N_HEADS, "norm": $NORM_JSON, "loss": "$LOSS", "loss_api_version": "v2", "loss_data_kind": "$LOSS_DATA_KIND", "loss_score_kind": "$LOSS_SCORE_KIND", "loss_k": $LOSS_K, "loss_horizon_mode": "strict", "loss_rank_lambda": $LOSS_RANK_LAMBDA, "loss_ranknet_alpha": $LOSS_RANKNET_ALPHA, "loss_listnet_tau": $LOSS_LISTNET_TAU, "loss_fingat_delta": $LOSS_FINGAT_DELTA, "loss_direction_scale": $LOSS_DIRECTION_SCALE, "loss_score_normalization": "$LOSS_SCORE_NORMALIZATION", "loss_hybrid_point_normalization": "$LOSS_HYBRID_POINT_NORMALIZATION", "loss_inverse_norm": $LOSS_INVERSE_NORM_JSON, "parallel_strategy": null$EXTRA_HPARAMS}
EOF
)

cat > "$MANIFEST_PATH" <<EOF
{
  "loss_api_version": "v2",
  "git_branch": "$GIT_BRANCH",
  "git_commit": "$GIT_COMMIT",
  "data_name": "$DATA_NAME",
  "model": "$MODEL",
  "model_name": "$MODEL_NAME",
  "trained_pred_len": $HORIZON,
  "evaluation_k": $LOSS_K,
  "loss": "$LOSS",
  "seed": $SEED,
  "save_path": "$SAVE_PATH",
  "strategy_args": $STRATEGY_ARGS,
  "model_hyper_params": $MODEL_HYPER_PARAMS
}
EOF

printf 'HOSTNAME: %s\n' "$(hostname)"
printf 'PYTHON_BIN: %s\n' "$PYTHON_BIN"
printf 'TFB_ROOT: %s\n' "$TFB_ROOT"
printf 'BRANCH: %s\n' "$GIT_BRANCH"
printf 'COMMIT: %s\n' "$GIT_COMMIT"
printf 'DATA_NAME: %s\n' "$DATA_NAME"
printf 'MODEL: %s\n' "$MODEL"
printf 'LOSS: %s\n' "$LOSS"
printf 'TRAINED_PRED_LEN: %s\n' "$HORIZON"
printf 'EVALUATION_K: %s\n' "$LOSS_K"
printf 'MANIFEST_PATH: %s\n' "$MANIFEST_PATH"
printf 'SAVE_PATH: %s\n' "$SAVE_PATH"
printf 'MODEL_HYPER_PARAMS: %s\n' "$MODEL_HYPER_PARAMS"

git status --short

"$PYTHON_BIN" scripts/verify_custom_losses_v2.py

"$PYTHON_BIN" scripts/run_benchmark_custom_losses_v2.py \
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
