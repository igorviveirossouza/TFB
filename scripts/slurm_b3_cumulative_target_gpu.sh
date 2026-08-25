#!/bin/bash
#SBATCH -p medusas_shr
#SBATCH --gres=gpu:1
#SBATCH --array=0-179%6
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=b3-cumtarget
#SBATCH --output=/sonic_home/igor.viveiros/src/TFB/logs/b3-cumtarget-%A_%a.out
#SBATCH --error=/sonic_home/igor.viveiros/src/TFB/logs/b3-cumtarget-%A_%a.err

set -euo pipefail

TFB_ROOT="${TFB_ROOT:-/sonic_home/igor.viveiros/src/TFB}"
PYTHON_BIN="${PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"
EXPERIMENT_ID="${EXPERIMENT_ID:-b3_cumulative_target_v1}"

# Mesma grade do primeiro exercício temporal. A única mudança experimental é
# o alvo: trajetória de retornos acumulados padronizados [B,H,N].
MODELS_CSV="${MODELS:-DUET,TimesNet,FEDformer,Nonstationary_Transformer}"
LOSSES_CSV="${LOSSES:-mse}"
DATASETS_CSV="${DATASETS:-simple_return,log_return,price}"
SEQ_LENS_CSV="${SEQ_LENS:-32,104,246}"
HORIZONS_CSV="${HORIZONS:-1,5,10,15,24}"
TASK_OFFSET="${TASK_OFFSET:-0}"
SEED="${SEED:-2021}"

TARGET_NORM="${TARGET_NORM:-true}"
CUMULATIVE_LOSS_HORIZON_MODE="${CUMULATIVE_LOSS_HORIZON_MODE:-all}"
LOSS_RANK_LAMBDA="${LOSS_RANK_LAMBDA:-1.0}"
LOSS_MARGIN="${LOSS_MARGIN:-0.01}"
LOSS_HINGE_MARGIN="${LOSS_HINGE_MARGIN:-$LOSS_MARGIN}"
LOSS_WHR_MARGIN="${LOSS_WHR_MARGIN:-$LOSS_MARGIN}"
LOSS_RANKNET_ALPHA="${LOSS_RANKNET_ALPHA:-1.0}"
LOSS_LISTNET_TAU="${LOSS_LISTNET_TAU:-0.01}"
LOSS_FINGAT_DELTA="${LOSS_FINGAT_DELTA:-0.01}"
LOSS_FINGAT_MARGIN="${LOSS_FINGAT_MARGIN:-0.0}"
LOSS_FINGAT_MOVE_LOGIT_SCALE="${LOSS_FINGAT_MOVE_LOGIT_SCALE:-0.01}"
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

N_MODELS=${#MODEL_ARR[@]}
N_LOSSES=${#LOSS_ARR[@]}
N_DATASETS=${#DATASET_ARR[@]}
N_SEQS=${#SEQ_ARR[@]}
N_HORIZONS=${#HORIZON_ARR[@]}
TOTAL=$((N_MODELS * N_LOSSES * N_DATASETS * N_SEQS * N_HORIZONS))
RAW_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_ID=$((RAW_TASK_ID + TASK_OFFSET))

if (( TASK_ID >= TOTAL )); then
  echo "TASK_ID=$TASK_ID fora da grade TOTAL=$TOTAL. Encerrando."
  exit 0
fi

H_IDX=$((TASK_ID % N_HORIZONS))
SEQ_IDX=$(((TASK_ID / N_HORIZONS) % N_SEQS))
DATASET_IDX=$(((TASK_ID / (N_HORIZONS * N_SEQS)) % N_DATASETS))
LOSS_IDX=$(((TASK_ID / (N_HORIZONS * N_SEQS * N_DATASETS)) % N_LOSSES))
MODEL_IDX=$((TASK_ID / (N_HORIZONS * N_SEQS * N_DATASETS * N_LOSSES)))

MODEL_KEY="${MODEL_ARR[$MODEL_IDX]}"
LOSS="${LOSS_ARR[$LOSS_IDX]}"
DATASET_LABEL="${DATASET_ARR[$DATASET_IDX]}"
SEQ_LEN="${SEQ_ARR[$SEQ_IDX]}"
HORIZON="${HORIZON_ARR[$H_IDX]}"
LOSS_K="$HORIZON"

case "$LOSS" in
  mse|mae|huber|mse_accum|mse_score|rank_hinge|rank_margin|rank_bpr|ranknet|whr1|whr2|listnet|fingat) ;;
  *) echo "Loss não preparada para o experimento cumulativo: $LOSS" >&2; exit 5 ;;
esac

case "$DATASET_LABEL" in
  simple_return)
    DATA_NAME="b3_daily_return.csv"
    TARGET_DATA_KIND="simple_return"
    TARGET_SCORE_KIND="simple_return"
    ;;
  log_return)
    DATA_NAME="b3_log_returns.csv"
    TARGET_DATA_KIND="log_return"
    TARGET_SCORE_KIND="log_return"
    ;;
  price)
    DATA_NAME="b3_daily_tfb.csv"
    TARGET_DATA_KIND="price"
    TARGET_SCORE_KIND="simple_return"
    ;;
  *) echo "Dataset desconhecido: $DATASET_LABEL" >&2; exit 6 ;;
esac

cd "$TFB_ROOT"
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$CURRENT_BRANCH" != "experiment/b3-loss-clean-v1" ]; then
  echo "ERRO: execute na branch experiment/b3-loss-clean-v1; atual=$CURRENT_BRANCH" >&2
  exit 7
fi

if [ ! -f "dataset/forecasting/$DATA_NAME" ]; then
  echo "Dataset ausente: $TFB_ROOT/dataset/forecasting/$DATA_NAME" >&2
  exit 8
fi

TARGET_NORM_JSON="$(json_bool "$TARGET_NORM")"
MODEL_NAME=""
MODEL_HYPER_PARAMS=""
ADAPTER_ARG=()
DETERMINISTIC_MODE="full"

# Os hiperparâmetros abaixo reproduzem o run_b3_financeiro_predlen24_lb32.sh
# do primeiro exercício. Só acrescentamos target_mode/target_* e LOSS.
case "$MODEL_KEY" in
  DUET)
    MODEL_NAME="duet.DUET"
    MODEL_HYPER_PARAMS="{\"CI\": 1, \"batch_size\": 32, \"d_ff\": 32, \"d_model\": 32, \"dropout\": 0.5, \"e_layers\": 1, \"factor\": 3, \"fc_dropout\": 0.2, \"hidden_size\": 256, \"pred_len\": ${HORIZON}, \"horizon\": ${HORIZON}, \"k\": 1, \"loss\": \"${LOSS}\", \"lr\": 0.01, \"lradj\": \"type1\", \"n_heads\": 2, \"norm\": true, \"num_epochs\": 100, \"num_experts\": 4, \"patch_len\": 48, \"patience\": 10, \"seq_len\": ${SEQ_LEN}, \"target_mode\": \"cumulative\", \"target_data_kind\": \"${TARGET_DATA_KIND}\", \"target_score_kind\": \"${TARGET_SCORE_KIND}\", \"target_norm\": ${TARGET_NORM_JSON}, \"loss_k\": ${LOSS_K}, \"cumulative_loss_horizon_mode\": \"${CUMULATIVE_LOSS_HORIZON_MODE}\", \"loss_rank_lambda\": ${LOSS_RANK_LAMBDA}, \"loss_margin\": ${LOSS_MARGIN}, \"loss_hinge_margin\": ${LOSS_HINGE_MARGIN}, \"loss_whr_margin\": ${LOSS_WHR_MARGIN}, \"loss_ranknet_alpha\": ${LOSS_RANKNET_ALPHA}, \"loss_listnet_tau\": ${LOSS_LISTNET_TAU}, \"loss_fingat_delta\": ${LOSS_FINGAT_DELTA}, \"loss_fingat_margin\": ${LOSS_FINGAT_MARGIN}, \"loss_fingat_move_logit_scale\": ${LOSS_FINGAT_MOVE_LOGIT_SCALE}}"
    ;;
  TimesNet)
    MODEL_NAME="time_series_library.TimesNet"
    MODEL_HYPER_PARAMS="{\"batch_size\": 32, \"d_ff\": 512, \"d_model\": 256, \"factor\": 3, \"pred_len\": ${HORIZON}, \"horizon\": ${HORIZON}, \"norm\": true, \"seq_len\": ${SEQ_LEN}, \"top_k\": 5, \"loss\": \"${LOSS}\", \"target_mode\": \"cumulative\", \"target_data_kind\": \"${TARGET_DATA_KIND}\", \"target_score_kind\": \"${TARGET_SCORE_KIND}\", \"target_norm\": ${TARGET_NORM_JSON}, \"loss_k\": ${LOSS_K}, \"cumulative_loss_horizon_mode\": \"${CUMULATIVE_LOSS_HORIZON_MODE}\", \"loss_rank_lambda\": ${LOSS_RANK_LAMBDA}, \"loss_margin\": ${LOSS_MARGIN}, \"loss_hinge_margin\": ${LOSS_HINGE_MARGIN}, \"loss_whr_margin\": ${LOSS_WHR_MARGIN}, \"loss_ranknet_alpha\": ${LOSS_RANKNET_ALPHA}, \"loss_listnet_tau\": ${LOSS_LISTNET_TAU}, \"loss_fingat_delta\": ${LOSS_FINGAT_DELTA}, \"loss_fingat_margin\": ${LOSS_FINGAT_MARGIN}, \"loss_fingat_move_logit_scale\": ${LOSS_FINGAT_MOVE_LOGIT_SCALE}}"
    ADAPTER_ARG=(--adapter "transformer_adapter")
    DETERMINISTIC_MODE="efficient"
    ;;
  FEDformer)
    MODEL_NAME="time_series_library.FEDformer"
    MODEL_HYPER_PARAMS="{\"batch_size\": 32, \"d_ff\": 512, \"d_model\": 256, \"factor\": 3, \"pred_len\": ${HORIZON}, \"horizon\": ${HORIZON}, \"norm\": true, \"seq_len\": ${SEQ_LEN}, \"loss\": \"${LOSS}\", \"target_mode\": \"cumulative\", \"target_data_kind\": \"${TARGET_DATA_KIND}\", \"target_score_kind\": \"${TARGET_SCORE_KIND}\", \"target_norm\": ${TARGET_NORM_JSON}, \"loss_k\": ${LOSS_K}, \"cumulative_loss_horizon_mode\": \"${CUMULATIVE_LOSS_HORIZON_MODE}\", \"loss_rank_lambda\": ${LOSS_RANK_LAMBDA}, \"loss_margin\": ${LOSS_MARGIN}, \"loss_hinge_margin\": ${LOSS_HINGE_MARGIN}, \"loss_whr_margin\": ${LOSS_WHR_MARGIN}, \"loss_ranknet_alpha\": ${LOSS_RANKNET_ALPHA}, \"loss_listnet_tau\": ${LOSS_LISTNET_TAU}, \"loss_fingat_delta\": ${LOSS_FINGAT_DELTA}, \"loss_fingat_margin\": ${LOSS_FINGAT_MARGIN}, \"loss_fingat_move_logit_scale\": ${LOSS_FINGAT_MOVE_LOGIT_SCALE}}"
    ADAPTER_ARG=(--adapter "transformer_adapter")
    ;;
  Nonstationary_Transformer)
    MODEL_NAME="time_series_library.Nonstationary_Transformer"
    MODEL_HYPER_PARAMS="{\"d_ff\": 256, \"d_model\": 128, \"dropout\": 0.1, \"factor\": 3, \"pred_len\": ${HORIZON}, \"horizon\": ${HORIZON}, \"norm\": true, \"p_hidden_dims\": [32, 32], \"p_hidden_layers\": 2, \"seq_len\": ${SEQ_LEN}, \"loss\": \"${LOSS}\", \"target_mode\": \"cumulative\", \"target_data_kind\": \"${TARGET_DATA_KIND}\", \"target_score_kind\": \"${TARGET_SCORE_KIND}\", \"target_norm\": ${TARGET_NORM_JSON}, \"loss_k\": ${LOSS_K}, \"cumulative_loss_horizon_mode\": \"${CUMULATIVE_LOSS_HORIZON_MODE}\", \"loss_rank_lambda\": ${LOSS_RANK_LAMBDA}, \"loss_margin\": ${LOSS_MARGIN}, \"loss_hinge_margin\": ${LOSS_HINGE_MARGIN}, \"loss_whr_margin\": ${LOSS_WHR_MARGIN}, \"loss_ranknet_alpha\": ${LOSS_RANKNET_ALPHA}, \"loss_listnet_tau\": ${LOSS_LISTNET_TAU}, \"loss_fingat_delta\": ${LOSS_FINGAT_DELTA}, \"loss_fingat_margin\": ${LOSS_FINGAT_MARGIN}, \"loss_fingat_move_logit_scale\": ${LOSS_FINGAT_MOVE_LOGIT_SCALE}}"
    ADAPTER_ARG=(--adapter "transformer_adapter")
    ;;
  *) echo "Modelo desconhecido: $MODEL_KEY" >&2; exit 9 ;;
esac

RUN_NAME="${MODEL_KEY}_${LOSS}_${TARGET_DATA_KIND}_lb${SEQ_LEN}_h${HORIZON}_k${LOSS_K}_seed${SEED}"
SAVE_ROOT="b3_cumulative_target/${EXPERIMENT_ID}"
SAVE_PATH="${SAVE_ROOT}/${RUN_NAME}"
RESULT_DIR="$TFB_ROOT/result/$SAVE_PATH"

case "${OVERWRITE_RUNS,,}" in
  true|1|yes|y|sim|s) rm -rf "$RESULT_DIR" ;;
esac
mkdir -p "$RESULT_DIR" "$TFB_ROOT/logs"

GIT_COMMIT="$(git rev-parse HEAD)"
STRATEGY_ARGS="{\"horizon\": ${HORIZON}, \"seed\": ${SEED}}"
cat > "$RESULT_DIR/run_manifest.json" <<EOF
{
  "experiment_id": "$EXPERIMENT_ID",
  "git_branch": "$CURRENT_BRANCH",
  "git_commit": "$GIT_COMMIT",
  "slurm_job_id": "${SLURM_JOB_ID:-}",
  "slurm_array_task_id": "$RAW_TASK_ID",
  "task_offset": "$TASK_OFFSET",
  "model": "$MODEL_KEY",
  "model_name": "$MODEL_NAME",
  "loss": "$LOSS",
  "dataset_label": "$DATASET_LABEL",
  "data_name": "$DATA_NAME",
  "target_mode": "cumulative",
  "target_data_kind": "$TARGET_DATA_KIND",
  "target_score_kind": "$TARGET_SCORE_KIND",
  "prediction_semantics": "cumulative_path",
  "seq_len": $SEQ_LEN,
  "trained_pred_len": $HORIZON,
  "loss_k": $LOSS_K,
  "seed": $SEED,
  "save_path": "$SAVE_PATH",
  "strategy_args": $STRATEGY_ARGS,
  "model_hyper_params": $MODEL_HYPER_PARAMS
}
EOF

printf 'TASK=%s OFFSET=%s GLOBAL=%s/%s MODEL=%s LOSS=%s DATASET=%s LB=%s H=%s\n' \
  "$RAW_TASK_ID" "$TASK_OFFSET" "$TASK_ID" "$TOTAL" "$MODEL_KEY" "$LOSS" "$DATASET_LABEL" "$SEQ_LEN" "$HORIZON"
printf 'BRANCH=%s COMMIT=%s SAVE_PATH=%s\n' "$CURRENT_BRANCH" "$GIT_COMMIT" "$SAVE_PATH"
printf 'TARGET=%s/%s target_norm=%s\n' "$TARGET_DATA_KIND" "$TARGET_SCORE_KIND" "$TARGET_NORM"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
"$PYTHON_BIN" scripts/run_benchmark_cumulative_target.py \
  --config-path "rolling_forecast_config.json" \
  --data-name-list "$DATA_NAME" \
  --strategy-args "$STRATEGY_ARGS" \
  --model-name "$MODEL_NAME" \
  --model-hyper-params "$MODEL_HYPER_PARAMS" \
  "${ADAPTER_ARG[@]}" \
  --deterministic "$DETERMINISTIC_MODE" \
  --gpus 0 \
  --num-workers 1 \
  --timeout 60000 \
  --save-path "$SAVE_PATH" \
  --save-true-pred True

touch "$RESULT_DIR/_SUCCESS"
echo "Treino/previsão concluído: $RESULT_DIR"
