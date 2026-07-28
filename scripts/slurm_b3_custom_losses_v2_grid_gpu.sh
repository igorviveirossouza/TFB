#!/bin/bash
#SBATCH -p medusas_shr
#SBATCH --gres=gpu:1
#SBATCH --array=0-1079%4
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=b3-loss-v2
#SBATCH --output=/sonic_home/igor.viveiros/src/TFB/logs/b3-loss-v2-%A_%a.out
#SBATCH --error=/sonic_home/igor.viveiros/src/TFB/logs/b3-loss-v2-%A_%a.err

set -euo pipefail

TFB_ROOT="${TFB_ROOT:-/sonic_home/igor.viveiros/src/TFB}"
PYTHON_BIN="${PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"
EXPERIMENT_ID="${EXPERIMENT_ID:?EXPERIMENT_ID não definido}"
MODELS_CSV="${MODELS:-duet,timesnet,fedformer,nonstationary}"
LOSSES_CSV="${LOSSES:-mse_path_v2,mse_score_v2,ranknet_v2,ranknet_hybrid_v2,listnet_v2,fingat_v2}"
DATASETS_CSV="${DATASETS:-log_return,simple_return,price}"
SEQ_LENS_CSV="${SEQ_LENS:-32,104,246}"
HORIZONS_CSV="${HORIZONS:-1,5,10,20,24}"
SEEDS_CSV="${SEEDS:-2021}"

BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
LR="${LR:-0.001}"
D_MODEL="${D_MODEL:-32}"
D_FF="${D_FF:-64}"
HIDDEN_SIZE="${HIDDEN_SIZE:-32}"
N_HEADS="${N_HEADS:-8}"
NUM_WORKERS="${NUM_WORKERS:-1}"
NUM_CPUS="${NUM_CPUS:-4}"
TIMEOUT="${TIMEOUT:-60000}"
NUM_ROLLINGS="${NUM_ROLLINGS:-999999}"
TV_RATIO="${TV_RATIO:-0.8}"
TRAIN_RATIO="${TRAIN_RATIO:-0.875}"
STRIDE="${STRIDE:-1}"
SAVE_TRUE_PRED="${SAVE_TRUE_PRED:-true}"
NORM="${NORM:-true}"
LOSS_INVERSE_NORM="${LOSS_INVERSE_NORM:-true}"
LOSS_RANK_LAMBDA="${LOSS_RANK_LAMBDA:-0.5}"
LOSS_RANKNET_ALPHA="${LOSS_RANKNET_ALPHA:-1.0}"
LOSS_LISTNET_TAU="${LOSS_LISTNET_TAU:-1.0}"
LOSS_FINGAT_DELTA="${LOSS_FINGAT_DELTA:-0.2}"
LOSS_DIRECTION_SCALE="${LOSS_DIRECTION_SCALE:-0.01}"
LOSS_SCORE_NORMALIZATION="${LOSS_SCORE_NORMALIZATION:-zscore}"
LOSS_HYBRID_POINT_NORMALIZATION="${LOSS_HYBRID_POINT_NORMALIZATION:-target_std}"
OVERWRITE_RUNS="${OVERWRITE_RUNS:-true}"

json_bool() {
  case "${1,,}" in
    true|1|yes|y|sim|s) echo "true" ;;
    false|0|no|n|nao|não) echo "false" ;;
    *) echo "Booleano inválido: $1" >&2; exit 4 ;;
  esac
}

IFS=',' read -r -a MODEL_ARR <<< "$MODELS_CSV"
IFS=',' read -r -a LOSS_ARR <<< "$LOSSES_CSV"
IFS=',' read -r -a DATASET_ARR <<< "$DATASETS_CSV"
IFS=',' read -r -a SEQ_ARR <<< "$SEQ_LENS_CSV"
IFS=',' read -r -a HORIZON_ARR <<< "$HORIZONS_CSV"
IFS=',' read -r -a SEED_ARR <<< "$SEEDS_CSV"

N_MODELS=${#MODEL_ARR[@]}
N_LOSSES=${#LOSS_ARR[@]}
N_DATASETS=${#DATASET_ARR[@]}
N_SEQS=${#SEQ_ARR[@]}
N_HORIZONS=${#HORIZON_ARR[@]}
N_SEEDS=${#SEED_ARR[@]}
TOTAL=$((N_MODELS * N_LOSSES * N_DATASETS * N_SEQS * N_HORIZONS * N_SEEDS))
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if (( TASK_ID >= TOTAL )); then
  echo "TASK_ID=$TASK_ID fora da grade TOTAL=$TOTAL."
  exit 0
fi

SEED_IDX=$((TASK_ID % N_SEEDS))
H_IDX=$(((TASK_ID / N_SEEDS) % N_HORIZONS))
SEQ_IDX=$(((TASK_ID / (N_SEEDS * N_HORIZONS)) % N_SEQS))
DATASET_IDX=$(((TASK_ID / (N_SEEDS * N_HORIZONS * N_SEQS)) % N_DATASETS))
LOSS_IDX=$(((TASK_ID / (N_SEEDS * N_HORIZONS * N_SEQS * N_DATASETS)) % N_LOSSES))
MODEL_IDX=$((TASK_ID / (N_SEEDS * N_HORIZONS * N_SEQS * N_DATASETS * N_LOSSES)))

MODEL="${MODEL_ARR[$MODEL_IDX]}"
LOSS="${LOSS_ARR[$LOSS_IDX]}"
DATASET_LABEL="${DATASET_ARR[$DATASET_IDX]}"
SEQ_LEN="${SEQ_ARR[$SEQ_IDX]}"
HORIZON="${HORIZON_ARR[$H_IDX]}"
LOSS_K="$HORIZON"
SEED="${SEED_ARR[$SEED_IDX]}"

case "$LOSS" in
  mse_path_v2|mse_score_v2|ranknet_v2|ranknet_hybrid_v2|listnet_v2|fingat_v2) ;;
  *) echo "Loss fora do experimento v2: $LOSS" >&2; exit 5 ;;
esac

cd "$TFB_ROOT"
if [ "$(git rev-parse --abbrev-ref HEAD)" != "speed" ]; then
  echo "O job deve executar na branch speed." >&2
  exit 6
fi
mkdir -p "$TFB_ROOT/logs" "$TFB_ROOT/manifests/custom_losses_v2_pipeline/$EXPERIMENT_ID"
export MPLCONFIGDIR="/tmp/${USER}-mpl-${SLURM_JOB_ID:-$$}"
mkdir -p "$MPLCONFIGDIR"

case "$DATASET_LABEL" in
  log_return)
    DATA_NAME="b3_log_returns.csv"
    LOSS_DATA_KIND="log_return"
    LOSS_SCORE_KIND="log_return"
    ;;
  simple_return)
    if [ -f "dataset/forecasting/b3_returns.csv" ]; then
      DATA_NAME="b3_returns.csv"
    elif [ -f "dataset/forecasting/b3_daily_return.csv" ]; then
      DATA_NAME="b3_daily_return.csv"
    else
      echo "Dataset de retorno simples não encontrado." >&2
      exit 7
    fi
    LOSS_DATA_KIND="simple_return"
    LOSS_SCORE_KIND="simple_return"
    ;;
  price)
    DATA_NAME="b3_daily_tfb.csv"
    LOSS_DATA_KIND="price"
    LOSS_SCORE_KIND="log_return"
    ;;
  *) echo "Dataset desconhecido: $DATASET_LABEL" >&2; exit 8 ;;
esac

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
  *) echo "Modelo desconhecido: $MODEL" >&2; exit 9 ;;
esac

SAVE_TRUE_PRED_JSON="$(json_bool "$SAVE_TRUE_PRED")"
NORM_JSON="$(json_bool "$NORM")"
LOSS_INVERSE_NORM_JSON="$(json_bool "$LOSS_INVERSE_NORM")"
RUN_NAME="${MODEL}_${LOSS}_${LOSS_DATA_KIND}_lb${SEQ_LEN}_h${HORIZON}_k${LOSS_K}_seed${SEED}"
SAVE_ROOT="b3_custom_losses_v2/${EXPERIMENT_ID}"
SAVE_PATH="${SAVE_ROOT}/${RUN_NAME}"
RESULT_DIR="$TFB_ROOT/result/$SAVE_PATH"

case "${OVERWRITE_RUNS,,}" in
  true|1|yes|y|sim|s) rm -rf "$RESULT_DIR" ;;
esac
mkdir -p "$RESULT_DIR"

STRATEGY_ARGS=$(cat <<EOF
{"horizon": $HORIZON, "tv_ratio": $TV_RATIO, "train_ratio_in_tv": {"__default__": $TRAIN_RATIO}, "stride": $STRIDE, "num_rollings": $NUM_ROLLINGS, "seed": $SEED, "save_true_pred": $SAVE_TRUE_PRED_JSON, "target_channel": null}
EOF
)

MODEL_HYPER_PARAMS=$(cat <<EOF
{"batch_size": $BATCH_SIZE, "d_ff": $D_FF, "d_model": $D_MODEL, "hidden_size": $HIDDEN_SIZE, "lr": $LR, "horizon": $HORIZON, "seq_len": $SEQ_LEN, "num_epochs": $NUM_EPOCHS, "patience": $PATIENCE, "n_heads": $N_HEADS, "norm": $NORM_JSON, "loss": "$LOSS", "loss_api_version": "v2", "loss_data_kind": "$LOSS_DATA_KIND", "loss_score_kind": "$LOSS_SCORE_KIND", "loss_k": $LOSS_K, "loss_horizon_mode": "strict", "loss_rank_lambda": $LOSS_RANK_LAMBDA, "loss_ranknet_alpha": $LOSS_RANKNET_ALPHA, "loss_listnet_tau": $LOSS_LISTNET_TAU, "loss_fingat_delta": $LOSS_FINGAT_DELTA, "loss_direction_scale": $LOSS_DIRECTION_SCALE, "loss_score_normalization": "$LOSS_SCORE_NORMALIZATION", "loss_hybrid_point_normalization": "$LOSS_HYBRID_POINT_NORMALIZATION", "loss_inverse_norm": $LOSS_INVERSE_NORM_JSON, "parallel_strategy": null$EXTRA_HPARAMS}
EOF
)

GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
GIT_COMMIT="$(git rev-parse HEAD)"
cat > "$RESULT_DIR/run_manifest.json" <<EOF
{
  "experiment_id": "$EXPERIMENT_ID",
  "loss_api_version": "v2",
  "git_branch": "$GIT_BRANCH",
  "git_commit": "$GIT_COMMIT",
  "slurm_job_id": "${SLURM_JOB_ID:-}",
  "slurm_array_task_id": "$TASK_ID",
  "model": "$MODEL",
  "model_name": "$MODEL_NAME",
  "loss": "$LOSS",
  "dataset_label": "$DATASET_LABEL",
  "data_name": "$DATA_NAME",
  "data_kind": "$LOSS_DATA_KIND",
  "score_kind": "$LOSS_SCORE_KIND",
  "seq_len": $SEQ_LEN,
  "trained_pred_len": $HORIZON,
  "evaluation_k": $LOSS_K,
  "seed": $SEED,
  "save_path": "$SAVE_PATH",
  "strategy_args": $STRATEGY_ARGS,
  "model_hyper_params": $MODEL_HYPER_PARAMS
}
EOF

printf 'TASK=%s/%s MODEL=%s LOSS=%s DATASET=%s LB=%s H=K=%s SEED=%s\n' \
  "$TASK_ID" "$TOTAL" "$MODEL" "$LOSS" "$DATASET_LABEL" "$SEQ_LEN" "$HORIZON" "$SEED"
printf 'BRANCH=%s COMMIT=%s SAVE_PATH=%s\n' "$GIT_BRANCH" "$GIT_COMMIT" "$SAVE_PATH"

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

touch "$RESULT_DIR/_SUCCESS"
echo "Treinamento e previsão concluídos: $RESULT_DIR"
