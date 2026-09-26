#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# simpleMLP / simpleFORMER: same TFB composite-trading v3 pipeline
#
# Edit only the GLOBAL HYPERPARAMETERS section for experimental design.
# Supported datasets: retornos_simples, log_retornos, prices.
# Prices are supported but are NOT included by default.
# Only pairs (H,K) satisfying K <= H and H % K == 0 are run.
#
# Cross-sectional design:
#   1) build financially meaningful score S inside each non-overlapping K block;
#   2) standardize S across assets inside each batch/block;
#   3) apply the selected cross-sectional loss to standardized scores Z.
# ==============================================================================

# ------------------------------------------------------------------------------
# GLOBAL HYPERPARAMETERS OF THE EXPERIMENT
# ------------------------------------------------------------------------------

MODELS=(
  simpleMLP
  simpleFORMER
)

# Options: retornos_simples log_retornos prices
DATASETS=(
  #retornos_simples
  log_retornos
)

read -r -a LOOKBACKS <<< "${LOOKBACKS_OVERRIDE:-32 104 246}"
#LOOKBACKS=(32)

read -r -a HORIZONS <<< "${HORIZONS_OVERRIDE:-1 5 10 15 20 24}"
#HORIZONS=(1 20)

read -r -a TRADE_WINDOWS <<< "${TRADE_WINDOWS_OVERRIDE:-1 5 10 15 20 24}"
#TRADE_WINDOWS=(1 5)

# Composite loss
TEMPORAL_LOSS="${TEMPORAL_LOSS:-mse}"
CROSS_LOSS="${CROSS_LOSS:-hinge}"                    # mse | pairwise_mse | ranknet | listnet | bpr | hinge
CROSS_LAMBDA="${CROSS_LAMBDA:-0.999}"                  # Peso da tarefa cross-sectional
SCORE_KIND="${SCORE_KIND:-simple_return}"          # simple_return | log_return
CROSS_SCORE_NORMALIZATION="${CROSS_SCORE_NORMALIZATION:-zscore}"  # zscore | none
CROSS_SCALE="${CROSS_SCALE:-1}"                     # Controla a escala (impacto) da loss cross-section
#------------------------------------------------------------------------------------------------
RANKNET_ALPHA="${RANKNET_ALPHA:-5}"                 # controla inclinação/intensidade da penalização pairwise. Quando = 1 -> BRP = ranknet
LISTNET_TAU="${LISTNET_TAU:-1.0}"                   # controla a temperatura na listnet
HINGE_MARGIN="${HINGE_MARGIN:-0.1}"                  # margem m da Hinge
#------------------------------------------------------------------------------------------------
CROSS_DELTA="${CROSS_DELTA:-0.0}"                   # zona morta pairwise; ignora pares com |d_ij| <= delta

# Architecture and training (all may be overridden through environment variables).
# LOSS_NAME=mse uses the ordinary TFB MSE. composite_trading uses the v3 loss;
# CROSS_LAMBDA=0 then gives the temporal baseline with component diagnostics.
LOSS_NAME="${LOSS_NAME:-composite_trading}"
D_MODEL="${D_MODEL:-512}"
E_LAYERS="${E_LAYERS:-2}"
D_FF="${D_FF:-2048}"                  # simpleFORMER FFN only
N_HEADS="${N_HEADS:-8}"               # simpleFORMER temporal attention only
TEMPORAL_HIDDEN_DIM="${TEMPORAL_HIDDEN_DIM:-0}"  # simpleMLP: 0 means 2*LB
DROPOUT="${DROPOUT:-0.1}"
NORM="${NORM:-true}"                 # TFB training-set StandardScaler
USE_NORM="${USE_NORM:-true}"         # reversible per-window normalization
NORM_EPS="${NORM_EPS:-0.00001}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
LRADJ="${LRADJ:-type1}"

# Saída
# FALSE (default): salva somente Parquet.
# TRUE: salva Parquet e também os CSVs janela_*.csv para auditoria/debug.
MANTER_CSV="${MANTER_CSV:-FALSE}"

# ------------------------------------------------------------------------------
# OPERATIONAL CONFIGURATION
# ------------------------------------------------------------------------------

TFB_ROOT="${TFB_ROOT:-/sonic_home/igor.viveiros/src/TFB}"
VENV_PATH="${VENV_PATH:-/sonic_home/igor.viveiros/py310/bin/activate}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_FILE="${CONFIG_FILE:-rolling_forecast_config.json}"
OUT_ROOT="${OUT_ROOT:-/snfs2/igor.viveiros/previsoes/parquet/simple_${LOSS_NAME}_${CROSS_LOSS}_lambda${CROSS_LAMBDA}_delta${CROSS_DELTA}}"  # Diretório de saída das previsões
RESULT_ROOT="${RESULT_ROOT:-/snfs2/igor.viveiros/result}"
PARQUET_ROOT="${PARQUET_ROOT:-${OUT_ROOT}/parquet}"
EXPERIMENT_ID="${EXPERIMENT_ID:-$(basename "${OUT_ROOT%/}")}"  # Isola resultados temporários entre experimentos
LOG_ROOT="${LOG_ROOT:-${TFB_ROOT}/logs}"
GPU_PARTITION="${GPU_PARTITION:-medusas_dev}"
GPU_TIME="${GPU_TIME:-08:00:00}"
MAX_GPU_JOBS="${MAX_GPU_JOBS:-6}"
SEED="${SEED:-2026}"
TV_RATIO="${TV_RATIO:-0.8}"
TRAIN_RATIO_IN_TV="${TRAIN_RATIO_IN_TV:-0.875}"
STRIDE="${STRIDE:-1}"
NUM_ROLLINGS="${NUM_ROLLINGS:-48000}"

SCRIPT_PATH="$(readlink -f "$0")"

# label:candidate_files:data_kind:step_offset
DATASET_SPECS_ALL=(
  "retornos_simples:b3_return_tfb.csv|b3_returns.csv|b3_daily_return.csv:simple_return:2"
  "log_retornos:b3_log_returns.csv|b3_log_return_tfb.csv:log_return:2"
  "prices:b3_daily_tfb.csv|b3_prices.csv|b3_price_tfb.csv:price:1"
)

build_dataset_specs() {
  DATASET_SPECS=()
  local requested spec label found
  for requested in "${DATASETS[@]}"; do
    found=0
    for spec in "${DATASET_SPECS_ALL[@]}"; do
      label="${spec%%:*}"
      if [[ "$requested" == "$label" ]]; then
        DATASET_SPECS+=("$spec")
        found=1
        break
      fi
    done
    if (( found == 0 )); then
      echo "ERRO: dataset inválido: ${requested}. Permitidos: retornos_simples log_retornos prices" >&2
      exit 2
    fi
  done
}

validate_models() {
  (( ${#MODELS[@]} > 0 && ${#DATASETS[@]} > 0 && ${#LOOKBACKS[@]} > 0 )) || {
    echo "ERRO: MODELS, DATASETS e LOOKBACKS não podem ser vazios." >&2; exit 2;
  }
  local model
  for model in "${MODELS[@]}"; do
    case "$model" in
      simpleMLP|simpleFORMER) ;;
      *) echo "ERRO: modelo inválido: $model" >&2; exit 2 ;;
    esac
  done
}

validate_loss() {
  case "$LOSS_NAME" in mse|composite_trading) ;; *) echo "ERRO: LOSS_NAME deve ser mse ou composite_trading." >&2; exit 2 ;; esac
  case "$TEMPORAL_LOSS" in mse|mae|huber) ;; *) echo "ERRO: TEMPORAL_LOSS inválida: $TEMPORAL_LOSS" >&2; exit 2 ;; esac
  case "$CROSS_LOSS" in mse|pairwise_mse|ranknet|listnet|bpr|hinge) ;; *) echo "ERRO: CROSS_LOSS inválida: $CROSS_LOSS" >&2; exit 2 ;; esac
  case "$SCORE_KIND" in simple_return|log_return) ;; *) echo "ERRO: SCORE_KIND inválido: $SCORE_KIND" >&2; exit 2 ;; esac
  case "$CROSS_SCORE_NORMALIZATION" in zscore|none) ;; *) echo "ERRO: CROSS_SCORE_NORMALIZATION inválida: $CROSS_SCORE_NORMALIZATION" >&2; exit 2 ;; esac
}

validate_output_config() {
  MANTER_CSV="${MANTER_CSV^^}"
  case "$MANTER_CSV" in
    TRUE|FALSE) ;;
    *) echo "ERRO: MANTER_CSV deve ser TRUE ou FALSE." >&2; exit 2 ;;
  esac
}

build_valid_hk_pairs() {
  HK_PAIRS=()
  local h k lb
  for lb in "${LOOKBACKS[@]}"; do
    [[ "$lb" =~ ^[1-9][0-9]*$ ]] || { echo "ERRO: lookback inválido: $lb" >&2; exit 2; }
  done
  for h in "${HORIZONS[@]}"; do
    (( h > 0 )) || { echo "ERRO: H deve ser positivo: $h" >&2; exit 2; }
    for k in "${TRADE_WINDOWS[@]}"; do
      (( k > 0 )) || { echo "ERRO: K deve ser positivo: $k" >&2; exit 2; }
      if (( k <= h && h % k == 0 )); then
        HK_PAIRS+=("${h}:${k}")
      fi
    done
  done
  (( ${#HK_PAIRS[@]} > 0 )) || { echo "ERRO: nenhum par (H,K) válido." >&2; exit 2; }
}

resolve_data_file() {
  local candidates="$1" candidate
  IFS='|' read -r -a arr <<< "$candidates"
  for candidate in "${arr[@]}"; do
    if [[ -f "${TFB_ROOT}/dataset/forecasting/${candidate}" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  echo "ERRO: nenhum dataset encontrado. Candidatos: ${candidates}" >&2
  return 1
}

prepare_worker() {
  [[ -d "$TFB_ROOT" ]] || { echo "ERRO: TFB_ROOT não existe: $TFB_ROOT" >&2; exit 2; }
  [[ -f "$VENV_PATH" ]] || { echo "ERRO: ambiente Python não encontrado: $VENV_PATH" >&2; exit 2; }
  # shellcheck disable=SC1090
  source "$VENV_PATH"
  cd "$TFB_ROOT"
  mkdir -p "$OUT_ROOT" "$PARQUET_ROOT" "$RESULT_ROOT" "$LOG_ROOT"
  [[ -f "${TFB_ROOT}/config/${CONFIG_FILE}" ]] || { echo "ERRO: config não encontrada: ${TFB_ROOT}/config/${CONFIG_FILE}" >&2; exit 2; }
  [[ -f "scripts/run_benchmark_composite_trading_loss_v3.py" ]] || { echo "ERRO: launcher v3 não encontrado." >&2; exit 2; }
  [[ -f "scripts/convert_composite_trading_predictions.py" ]] || { echo "ERRO: conversor não encontrado." >&2; exit 2; }
  "$PYTHON_BIN" -c 'import pyarrow' >/dev/null 2>&1 || {
    echo "ERRO: pyarrow não está instalado no ambiente Python. Instale pyarrow antes de rodar o experimento." >&2
    exit 2
  }
  echo "TFB commit: $(git rev-parse HEAD 2>/dev/null || echo desconhecido)"
}

model_args() {
  local model_key="$1" lb="$2" h="$3" k="$4" data_kind="$5"
  MODEL_NAME="time_series_library.${model_key}"
  ADAPTER_ARG=(--adapter "transformer_adapter")
  DETERMINISTIC_MODE="full"

  # Serialize typed values instead of interpolating numeric strings into JSON.
  # In particular, a numeric value such as 02 cannot produce invalid JSON.
  MODEL_HYPER_PARAMS=$("$PYTHON_BIN" - "$model_key" \
    seq_len "$lb" horizon "$h" pred_len "$h" \
    d_model "$D_MODEL" e_layers "$E_LAYERS" d_ff "$D_FF" n_heads "$N_HEADS" \
    temporal_hidden_dim "$TEMPORAL_HIDDEN_DIM" dropout "$DROPOUT" \
    norm "$NORM" use_norm "$USE_NORM" norm_eps "$NORM_EPS" \
    batch_size "$BATCH_SIZE" lr "$LEARNING_RATE" num_epochs "$NUM_EPOCHS" \
    patience "$PATIENCE" lradj "$LRADJ" \
    loss "$LOSS_NAME" loss_temporal "$TEMPORAL_LOSS" loss_cross "$CROSS_LOSS" \
    loss_trade_window "$k" loss_cross_lambda "$CROSS_LAMBDA" \
    loss_cross_scale "$CROSS_SCALE" loss_ranknet_alpha "$RANKNET_ALPHA" \
    loss_listnet_tau "$LISTNET_TAU" loss_hinge_margin "$HINGE_MARGIN" \
    loss_cross_delta "$CROSS_DELTA" loss_data_kind "$data_kind" \
    loss_score_kind "$SCORE_KIND" loss_cross_score_normalization "$CROSS_SCORE_NORMALIZATION" <<'PYCONFIG'
import json
import math
import sys

model = sys.argv[1]
raw = dict(zip(sys.argv[2::2], sys.argv[3::2]))
integers = {"seq_len", "horizon", "pred_len", "d_model", "e_layers", "d_ff",
            "n_heads", "temporal_hidden_dim", "batch_size", "num_epochs",
            "patience", "loss_trade_window"}
floats = {"dropout", "norm_eps", "lr", "loss_cross_lambda", "loss_cross_scale",
          "loss_ranknet_alpha", "loss_listnet_tau", "loss_hinge_margin", "loss_cross_delta"}
booleans = {"norm", "use_norm"}
params = {}
for key, value in raw.items():
    if key in integers:
        params[key] = int(value)
    elif key in floats:
        params[key] = float(value)
        if not math.isfinite(params[key]):
            raise ValueError(key + " must be finite")
    elif key in booleans:
        if value.lower() not in {"true", "false", "1", "0"}:
            raise ValueError(key + " must be true or false")
        params[key] = value.lower() in {"true", "1"}
    else:
        params[key] = value
for key in integers - {"temporal_hidden_dim"}:
    if params[key] <= 0:
        raise ValueError(key + " must be positive")
if params["temporal_hidden_dim"] < 0:
    raise ValueError("temporal_hidden_dim must be >= 0")
if model == "simpleFORMER" and params["d_model"] % params["n_heads"]:
    raise ValueError("D_MODEL must be divisible by N_HEADS")
if not 0 <= params["loss_cross_lambda"] <= 1 or not 0 <= params["dropout"] < 1:
    raise ValueError("CROSS_LAMBDA must be in [0,1] and DROPOUT in [0,1)")
for key in ("norm_eps", "lr", "loss_ranknet_alpha", "loss_listnet_tau"):
    if params[key] <= 0:
        raise ValueError(key + " must be positive")
for key in ("loss_cross_delta", "loss_cross_scale", "loss_hinge_margin"):
    if params[key] < 0:
        raise ValueError(key + " must be nonnegative")
params.update(loss_inverse_norm=True, loss_track_components=True)
print(json.dumps(params, allow_nan=False))
PYCONFIG
  )
}

run_tfb() {
  local data_file="$1" model_name="$2" model_hyper="$3" deterministic="$4" h="$5" result_dir="$6"
  shift 6
  local adapter_args=("$@")

  rm -rf "$result_dir"
  mkdir -p "$result_dir"

  "$PYTHON_BIN" ./scripts/run_benchmark_composite_trading_loss_v3.py \
    --config-path "$CONFIG_FILE" \
    --data-name-list "$data_file" \
    --strategy-args "{\"horizon\":${h},\"tv_ratio\":${TV_RATIO},\"train_ratio_in_tv\":${TRAIN_RATIO_IN_TV},\"stride\":${STRIDE},\"num_rollings\":${NUM_ROLLINGS},\"seed\":${SEED}}" \
    --model-name "$model_name" \
    --model-hyper-params "$model_hyper" \
    "${adapter_args[@]}" \
    --deterministic "$deterministic" \
    --gpus 0 \
    --num-workers 1 \
    --timeout 60000 \
    --save-path "$result_dir" \
    --save-true-pred True
}

decode_predictions() {
  local result_dir="$1" h="$2" decoded_dir="$3"
  rm -rf "$decoded_dir"
  mkdir -p "$decoded_dir"
  mapfile -t tars < <(find "$result_dir" -maxdepth 1 -type f -name '*.csv.tar.gz' | sort)
  (( ${#tars[@]} > 0 )) || { echo "ERRO: nenhum .csv.tar.gz em $result_dir" >&2; exit 4; }

  local copy_index=0 tarfile extracted_dir decoded_csv rows
  for tarfile in "${tars[@]}"; do
    "$PYTHON_BIN" ts_benchmark/utils/decode_prediction.py "$tarfile"
    extracted_dir="$(dirname "$tarfile")/$(basename "$tarfile" .tar.gz)_extracted"
    [[ -d "$extracted_dir" ]] || { echo "ERRO: pasta extraída não encontrada: $extracted_dir" >&2; exit 4; }
    while IFS= read -r decoded_csv; do
      rows=$(($(wc -l < "$decoded_csv") - 1))
      [[ "$rows" -eq "$h" ]] || continue
      cp "$decoded_csv" "${decoded_dir}/csv_sample_${copy_index}_inference_data.csv"
      copy_index=$((copy_index + 1))
    done < <(find "$extracted_dir" -type f -name 'inference_data.csv' | sort)
  done
  (( copy_index > 0 )) || { echo "ERRO: nenhuma previsão decodificada com h=$h." >&2; exit 4; }
}

convert_predictions() {
  local decoded_dir="$1" original_dataset="$2" h="$3" lb="$4" step_offset="$5"
  local dataset_label="$6" model_key="$7" k="$8" csv_dir="$9"

  local cmd=(
    "$PYTHON_BIN" scripts/convert_composite_trading_predictions.py
    --decoded-dir "$decoded_dir"
    --dataset "$original_dataset"
    --pred-len "$h"
    --lookback "$lb"
    --step-offset "$step_offset"
    --tv-ratio "$TV_RATIO"
    --parquet-root "$PARQUET_ROOT"
    --dataset-label "$dataset_label"
    --model "$model_key"
    --k "$k"
  )

  if [[ "$MANTER_CSV" == "TRUE" ]]; then
    cmd+=(--output-dir "$csv_dir" --keep-csv)
  fi

  "${cmd[@]}"

  # Em modo Parquet-only, remove eventual CSV antigo dessa configuração.
  if [[ "$MANTER_CSV" == "FALSE" && -d "$csv_dir" ]]; then
    rm -rf "$csv_dir"
  fi
}

record_completed_task() {
  local task_id="$1" dataset_label="$2" model_key="$3" lb="$4" h="$5" k="$6" final_dir="$7"
  local completed_file="${OUT_ROOT}/completed_tasks.csv"
  local lock_file="${OUT_ROOT}/.completed_tasks.lock"
  local header="array_index,dataset,model,lookback,pred_len,k,temporal_loss,cross_loss,cross_lambda,cross_scale,ranknet_alpha,hinge_margin,listnet_tau,cross_delta,score_kind,cross_score_normalization,seed,slurm_job_id,output_dir,loss_name,d_model,e_layers,d_ff,n_heads,temporal_hidden_dim,dropout,norm,use_norm,norm_eps,batch_size,lr,num_epochs,patience,lradj"
  local row="${task_id},${dataset_label},${model_key},${lb},${h},${k},${TEMPORAL_LOSS},${CROSS_LOSS},${CROSS_LAMBDA},${CROSS_SCALE},${RANKNET_ALPHA},${HINGE_MARGIN},${LISTNET_TAU},${CROSS_DELTA},${SCORE_KIND},${CROSS_SCORE_NORMALIZATION},${SEED},${SLURM_JOB_ID:-},${final_dir},${LOSS_NAME},${D_MODEL},${E_LAYERS},${D_FF},${N_HEADS},${TEMPORAL_HIDDEN_DIM},${DROPOUT},${NORM},${USE_NORM},${NORM_EPS},${BATCH_SIZE},${LEARNING_RATE},${NUM_EPOCHS},${PATIENCE},${LRADJ}"

  mkdir -p "$OUT_ROOT"
  (
    flock -x 200

    if [[ ! -s "$completed_file" ]]; then
      printf '%s\n' "$header" > "$completed_file"
    fi

    local tmp_file
    tmp_file="$(mktemp "${OUT_ROOT}/.completed_tasks.XXXXXX")"
    awk -F',' -v idx="$task_id" 'NR == 1 || $1 != idx' "$completed_file" > "$tmp_file"
    printf '%s\n' "$row" >> "$tmp_file"
    mv "$tmp_file" "$completed_file"
  ) 200>"$lock_file"
}

write_manifest() {
  mkdir -p "$OUT_ROOT"
  local manifest="${OUT_ROOT}/design_simple_composite_trading_v3.csv"
  {
    echo "dataset,modelo,lookback,pred_len,k,temporal_loss,cross_loss,cross_lambda,cross_delta,score_kind,cross_score_normalization,seed,loss_name,d_model,e_layers,d_ff,n_heads,temporal_hidden_dim,dropout,norm,use_norm,norm_eps,batch_size,lr,num_epochs,patience,lradj,cross_scale,ranknet_alpha,listnet_tau,hinge_margin"
    local spec label candidates data_kind offset model lb pair h k
    for spec in "${DATASET_SPECS[@]}"; do
      IFS=':' read -r label candidates data_kind offset <<< "$spec"
      for model in "${MODELS[@]}"; do
        for lb in "${LOOKBACKS[@]}"; do
          for pair in "${HK_PAIRS[@]}"; do
            IFS=':' read -r h k <<< "$pair"
            echo "${label},${model},${lb},${h},${k},${TEMPORAL_LOSS},${CROSS_LOSS},${CROSS_LAMBDA},${CROSS_DELTA},${SCORE_KIND},${CROSS_SCORE_NORMALIZATION},${SEED},${LOSS_NAME},${D_MODEL},${E_LAYERS},${D_FF},${N_HEADS},${TEMPORAL_HIDDEN_DIM},${DROPOUT},${NORM},${USE_NORM},${NORM_EPS},${BATCH_SIZE},${LEARNING_RATE},${NUM_EPOCHS},${PATIENCE},${LRADJ},${CROSS_SCALE},${RANKNET_ALPHA},${LISTNET_TAU},${HINGE_MARGIN}"
          done
        done
      done
    done
  } > "$manifest"
  echo "Manifesto: $manifest"
}

run_worker() {
  prepare_worker
  local task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID não definido}"
  (( task_id >= 0 && task_id < N_TASKS )) || { echo "ERRO: task_id fora da grade: $task_id / $N_TASKS" >&2; exit 3; }

  local rem="$task_id"
  local hk_idx=$((rem % N_HK)); rem=$((rem / N_HK))
  local lb_idx=$((rem % N_LOOKBACKS)); rem=$((rem / N_LOOKBACKS))
  local model_idx=$((rem % N_MODELS)); rem=$((rem / N_MODELS))
  local dataset_idx=$((rem % N_DATASETS))

  local spec="${DATASET_SPECS[$dataset_idx]}"
  local dataset_label candidates data_kind step_offset
  IFS=':' read -r dataset_label candidates data_kind step_offset <<< "$spec"

  local model_key="${MODELS[$model_idx]}"
  local lb="${LOOKBACKS[$lb_idx]}"
  local pair="${HK_PAIRS[$hk_idx]}" h k
  IFS=':' read -r h k <<< "$pair"

  local data_file
  data_file="$(resolve_data_file "$candidates")"
  local original_dataset="${TFB_ROOT}/dataset/forecasting/${data_file}"

  model_args "$model_key" "$lb" "$h" "$k" "$data_kind"

  local tag="${dataset_label}_${model_key}_lb${lb}_h${h}_k${k}_task${task_id}"
  local save_subdir="simple_composite_trading_v3/${EXPERIMENT_ID}/${tag}"
  local result_dir="${RESULT_ROOT}/${save_subdir}"
  local decoded_dir="${result_dir}/decoded"
  local csv_dir="${OUT_ROOT}/${dataset_label}/${model_key}/lookback_${lb}/pred_len_${h}/k_${k}"
  local parquet_dir="${PARQUET_ROOT}/dataset=${dataset_label}/modelo=${model_key}/lookback=lookback_${lb}/pred_len=pred_len_${h}/k_dir=k_${k}"

  echo "TASK=$task_id experiment=$EXPERIMENT_ID dataset=$dataset_label model=$model_key lb=$lb H=$h K=$k"
  run_tfb "$data_file" "$MODEL_NAME" "$MODEL_HYPER_PARAMS" "$DETERMINISTIC_MODE" "$h" "$result_dir" "${ADAPTER_ARG[@]}"
  decode_predictions "$result_dir" "$h" "$decoded_dir"
  convert_predictions "$decoded_dir" "$original_dataset" "$h" "$lb" "$step_offset" "$dataset_label" "$model_key" "$k" "$csv_dir"
  record_completed_task "$task_id" "$dataset_label" "$model_key" "$lb" "$h" "$k" "$parquet_dir"
  echo "OK: $parquet_dir"
}

# Importing definitions must never submit a job or modify output directories.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  return 0
fi
case "${1:-}" in
  ""|worker|--dry-run) ;;
  *) echo "Uso: bash $0 [--dry-run|worker]" >&2; exit 2 ;;
esac

validate_models
validate_loss
validate_output_config
build_dataset_specs
build_valid_hk_pairs

N_DATASETS=${#DATASET_SPECS[@]}
N_MODELS=${#MODELS[@]}
N_LOOKBACKS=${#LOOKBACKS[@]}
N_HK=${#HK_PAIRS[@]}
N_TASKS=$((N_DATASETS * N_MODELS * N_LOOKBACKS * N_HK))


if [[ "${1:-}" == "worker" ]]; then
  run_worker
  exit 0
fi

# Validate architecture/numeric parameters before submitting any tasks.
IFS=':' read -r example_h example_k <<< "${HK_PAIRS[0]}"
IFS=':' read -r example_dataset example_candidates example_kind example_offset <<< "${DATASET_SPECS[0]}"
for model in "${MODELS[@]}"; do
  model_args "$model" "${LOOKBACKS[0]}" "$example_h" "$example_k" "$example_kind"
done
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "Modelos: ${MODELS[*]} | Tarefas: ${N_TASKS} | HK: ${HK_PAIRS[*]}"
  echo "Exemplo de configuração: dataset=${example_dataset}, LB=${LOOKBACKS[0]}, H=${example_h}, K=${example_k}"
  for model in "${MODELS[@]}"; do
    model_args "$model" "${LOOKBACKS[0]}" "$example_h" "$example_k" "$example_kind"
    echo "${MODEL_NAME} ${MODEL_HYPER_PARAMS}"
  done
  exit 0
fi

mkdir -p "$LOG_ROOT" "$OUT_ROOT" "$PARQUET_ROOT"
write_manifest

echo "Experimento: ${EXPERIMENT_ID}"
echo "Datasets : ${DATASETS[*]}"
echo "Modelos  : ${MODELS[*]}"
echo "Lookbacks: ${LOOKBACKS[*]}"
echo "H        : ${HORIZONS[*]}"
echo "K        : ${TRADE_WINDOWS[*]}"
echo "Pares HK : ${HK_PAIRS[*]}"
echo "Arquitetura: d=${D_MODEL}, blocos=${E_LAYERS}, heads=${N_HEADS}, d_ff=${D_FF}, temporal_hidden=${TEMPORAL_HIDDEN_DIM}, norm=${NORM}, use_norm=${USE_NORM}"
echo "Treino: batch=${BATCH_SIZE}, lr=${LEARNING_RATE}, epochs=${NUM_EPOCHS}, patience=${PATIENCE}"
echo "Loss     : factory=${LOSS_NAME}, temporal=${TEMPORAL_LOSS}, cross=${CROSS_LOSS}, lambda=${CROSS_LAMBDA}, cross_norm=${CROSS_SCORE_NORMALIZATION}, CS=${CROSS_SCALE}, delta=${CROSS_DELTA}"
echo "Parquet  : ${PARQUET_ROOT}"
echo "Manter CSV: ${MANTER_CSV}"
echo "Tarefas  : ${N_TASKS}"

sbatch \
  -p "$GPU_PARTITION" \
  --gres=gpu:1 \
  --time="$GPU_TIME" \
  --array="0-$((N_TASKS - 1))%${MAX_GPU_JOBS}" \
  --job-name="${EXPERIMENT_ID}" \
  --output="${LOG_ROOT}/${EXPERIMENT_ID}_%A_%a.out" \
  --error="${LOG_ROOT}/${EXPERIMENT_ID}_%A_%a.err" \
  "$SCRIPT_PATH" worker
