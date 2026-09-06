#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Composite trading experiment
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
  DUET
  Nonstationary_Transformer
  FEDformer
  TimesNet
)

# Options: retornos_simples log_retornos prices
DATASETS=(
  #retornos_simples
  log_retornos
)

LOOKBACKS=(32 104 246)
#LOOKBACKS=(32)

HORIZONS=(1 5 10 15 20 24)
#HORIZONS=(1 20)

TRADE_WINDOWS=(1 5 10 15 20 24)
#TRADE_WINDOWS=(1 5)

# Composite loss
TEMPORAL_LOSS="mse"
CROSS_LOSS="mse"                    # mse | ranknet | listnet | bpr | hinge
CROSS_LAMBDA="0.0"                 # Peso da tarefa cross-sectional
SCORE_KIND="simple_return"          # simple_return | log_return
CROSS_SCORE_NORMALIZATION="zscore"  # zscore | none
CROSS_SCALE="1.0"                   # Controla a escala (impacto) da loss cross-section
RANKNET_ALPHA="1.0"                 # controla inclinação/intensidade da penalização pairwise. Quando = 1 -> BRP = ranknet
LISTNET_TAU="1.0"                   # controla a temperatura na listnet 

# ------------------------------------------------------------------------------
# OPERATIONAL CONFIGURATION
# ------------------------------------------------------------------------------

TFB_ROOT="${TFB_ROOT:-/sonic_home/igor.viveiros/src/TFB}"
VENV_PATH="${VENV_PATH:-/sonic_home/igor.viveiros/py310/bin/activate}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_FILE="${CONFIG_FILE:-rolling_forecast_config.json}"
OUT_ROOT="${OUT_ROOT:-/snfs2/igor.viveiros/previsoes/composite_trading_v3/mse_lambda_00}"  # Diretório de saída das previsões
LOG_ROOT="${LOG_ROOT:-${TFB_ROOT}/logs}"
GPU_PARTITION="${GPU_PARTITION:-medusas_shr}"
GPU_TIME="${GPU_TIME:-48:00:00}"
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
  local model
  for model in "${MODELS[@]}"; do
    case "$model" in
      DUET|Nonstationary_Transformer|FEDformer|TimesNet) ;;
      *) echo "ERRO: modelo inválido: $model" >&2; exit 2 ;;
    esac
  done
}

validate_loss() {
  case "$TEMPORAL_LOSS" in mse|mae|huber) ;; *) echo "ERRO: TEMPORAL_LOSS inválida: $TEMPORAL_LOSS" >&2; exit 2 ;; esac
  case "$CROSS_LOSS" in mse|ranknet|listnet|bpr|hinge) ;; *) echo "ERRO: CROSS_LOSS inválida: $CROSS_LOSS" >&2; exit 2 ;; esac
  case "$SCORE_KIND" in simple_return|log_return) ;; *) echo "ERRO: SCORE_KIND inválido: $SCORE_KIND" >&2; exit 2 ;; esac
  case "$CROSS_SCORE_NORMALIZATION" in zscore|none) ;; *) echo "ERRO: CROSS_SCORE_NORMALIZATION inválida: $CROSS_SCORE_NORMALIZATION" >&2; exit 2 ;; esac
}

build_valid_hk_pairs() {
  HK_PAIRS=()
  local h k
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
  mkdir -p "$OUT_ROOT" "$LOG_ROOT"
  [[ -f "${TFB_ROOT}/config/${CONFIG_FILE}" ]] || { echo "ERRO: config não encontrada: ${TFB_ROOT}/config/${CONFIG_FILE}" >&2; exit 2; }
  [[ -f "scripts/run_benchmark_composite_trading_loss_v3.py" ]] || { echo "ERRO: launcher v3 não encontrado." >&2; exit 2; }
  [[ -f "scripts/convert_composite_trading_predictions.py" ]] || { echo "ERRO: conversor não encontrado." >&2; exit 2; }
  echo "TFB commit: $(git rev-parse HEAD 2>/dev/null || echo desconhecido)"
}

model_args() {
  local model_key="$1" lb="$2" h="$3" k="$4" data_kind="$5"
  MODEL_NAME=""
  MODEL_HYPER_PARAMS=""
  ADAPTER_ARG=()
  DETERMINISTIC_MODE="full"

  local loss_fields
  loss_fields="\"loss_cross_scale\":${CROSS_SCALE},\"loss_ranknet_alpha\":${RANKNET_ALPHA},\"loss_listnet_tau\":${LISTNET_TAU},\"loss\":\"composite_trading\",\"loss_temporal\":\"${TEMPORAL_LOSS}\",\"loss_cross\":\"${CROSS_LOSS}\",\"loss_trade_window\":${k},\"loss_cross_lambda\":${CROSS_LAMBDA},\"loss_data_kind\":\"${data_kind}\",\"loss_score_kind\":\"${SCORE_KIND}\",\"loss_cross_score_normalization\":\"${CROSS_SCORE_NORMALIZATION}\",\"loss_inverse_norm\":true,\"loss_track_components\":true"

  case "$model_key" in
    DUET)
      MODEL_NAME="duet.DUET"
      MODEL_HYPER_PARAMS="{\"CI\":1,\"batch_size\":32,\"d_ff\":32,\"d_model\":32,\"dropout\":0.5,\"e_layers\":1,\"factor\":3,\"fc_dropout\":0.2,\"hidden_size\":256,\"pred_len\":${h},\"horizon\":${h},\"k\":1,\"lr\":0.01,\"lradj\":\"type1\",\"n_heads\":2,\"norm\":true,\"num_epochs\":100,\"num_experts\":4,\"patch_len\":48,\"patience\":10,\"seq_len\":${lb},${loss_fields}}"
      ;;
    TimesNet)
      MODEL_NAME="time_series_library.TimesNet"
      MODEL_HYPER_PARAMS="{\"batch_size\":32,\"d_ff\":512,\"d_model\":256,\"factor\":3,\"pred_len\":${h},\"horizon\":${h},\"norm\":true,\"seq_len\":${lb},\"top_k\":5,${loss_fields}}"
      ADAPTER_ARG=(--adapter "transformer_adapter")
      DETERMINISTIC_MODE="efficient"
      ;;
    FEDformer)
      MODEL_NAME="time_series_library.FEDformer"
      MODEL_HYPER_PARAMS="{\"batch_size\":32,\"d_ff\":512,\"d_model\":256,\"factor\":3,\"pred_len\":${h},\"horizon\":${h},\"norm\":true,\"seq_len\":${lb},${loss_fields}}"
      ADAPTER_ARG=(--adapter "transformer_adapter")
      ;;
    Nonstationary_Transformer)
      MODEL_NAME="time_series_library.Nonstationary_Transformer"
      MODEL_HYPER_PARAMS="{\"d_ff\":256,\"d_model\":128,\"dropout\":0.1,\"factor\":3,\"pred_len\":${h},\"horizon\":${h},\"norm\":true,\"p_hidden_dims\":[32,32],\"p_hidden_layers\":2,\"seq_len\":${lb},${loss_fields}}"
      ADAPTER_ARG=(--adapter "transformer_adapter")
      ;;
  esac
}

run_tfb() {
  local data_file="$1" model_name="$2" model_hyper="$3" deterministic="$4" h="$5" save_subdir="$6"
  shift 6
  local adapter_args=("$@")
  local result_dir="${TFB_ROOT}/result/${save_subdir}"
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
    --save-path "$save_subdir" \
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
  local decoded_dir="$1" original_dataset="$2" h="$3" lb="$4" step_offset="$5" final_dir="$6"
  rm -rf "$final_dir"
  mkdir -p "$final_dir"
  "$PYTHON_BIN" scripts/convert_composite_trading_predictions.py \
    --decoded-dir "$decoded_dir" \
    --dataset "$original_dataset" \
    --pred-len "$h" \
    --lookback "$lb" \
    --step-offset "$step_offset" \
    --tv-ratio "$TV_RATIO" \
    --output-dir "$final_dir"
}

record_completed_task() {
  local task_id="$1" dataset_label="$2" model_key="$3" lb="$4" h="$5" k="$6" final_dir="$7"
  local completed_file="${OUT_ROOT}/completed_tasks.csv"
  local lock_file="${OUT_ROOT}/.completed_tasks.lock"
  local header="array_index,dataset,model,lookback,pred_len,k,temporal_loss,cross_loss,cross_lambda,cross_scale,ranknet_alpha,listnet_tau,score_kind,cross_score_normalization,seed,slurm_job_id,output_dir"
  local row="${task_id},${dataset_label},${model_key},${lb},${h},${k},${TEMPORAL_LOSS},${CROSS_LOSS},${CROSS_LAMBDA},${CROSS_SCALE},${RANKNET_ALPHA},${LISTNET_TAU},${SCORE_KIND},${CROSS_SCORE_NORMALIZATION},${SEED},${SLURM_JOB_ID:-},${final_dir}"

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
  local manifest="${OUT_ROOT}/design_composite_trading_v3.csv"
  {
    echo "dataset,modelo,lookback,pred_len,k,temporal_loss,cross_loss,cross_lambda,score_kind,cross_score_normalization,seed"
    local spec label candidates data_kind offset model lb pair h k
    for spec in "${DATASET_SPECS[@]}"; do
      IFS=':' read -r label candidates data_kind offset <<< "$spec"
      for model in "${MODELS[@]}"; do
        for lb in "${LOOKBACKS[@]}"; do
          for pair in "${HK_PAIRS[@]}"; do
            IFS=':' read -r h k <<< "$pair"
            echo "${label},${model},${lb},${h},${k},${TEMPORAL_LOSS},${CROSS_LOSS},${CROSS_LAMBDA},${SCORE_KIND},${CROSS_SCORE_NORMALIZATION},${SEED}"
          done
        done
      done
    done
  } > "$manifest"
  echo "Manifesto: $manifest"
}

validate_models
validate_loss
build_dataset_specs
build_valid_hk_pairs

N_DATASETS=${#DATASET_SPECS[@]}
N_MODELS=${#MODELS[@]}
N_LOOKBACKS=${#LOOKBACKS[@]}
N_HK=${#HK_PAIRS[@]}
N_TASKS=$((N_DATASETS * N_MODELS * N_LOOKBACKS * N_HK))

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
  local save_subdir="composite_trading_v3/${tag}"
  local result_dir="${TFB_ROOT}/result/${save_subdir}"
  local decoded_dir="${result_dir}/decoded"
  local final_dir="${OUT_ROOT}/${dataset_label}/${model_key}/lookback_${lb}/pred_len_${h}/k_${k}"

  echo "TASK=$task_id dataset=$dataset_label model=$model_key lb=$lb H=$h K=$k"
  run_tfb "$data_file" "$MODEL_NAME" "$MODEL_HYPER_PARAMS" "$DETERMINISTIC_MODE" "$h" "$save_subdir" "${ADAPTER_ARG[@]}"
  decode_predictions "$result_dir" "$h" "$decoded_dir"
  convert_predictions "$decoded_dir" "$original_dataset" "$h" "$lb" "$step_offset" "$final_dir"
  record_completed_task "$task_id" "$dataset_label" "$model_key" "$lb" "$h" "$k" "$final_dir"
  echo "OK: $final_dir"
}

if [[ "${1:-}" == "worker" ]]; then
  run_worker
  exit 0
fi

mkdir -p "$LOG_ROOT" "$OUT_ROOT"
write_manifest

echo "Datasets : ${DATASETS[*]}"
echo "Modelos  : ${MODELS[*]}"
echo "Lookbacks: ${LOOKBACKS[*]}"
echo "H        : ${HORIZONS[*]}"
echo "K        : ${TRADE_WINDOWS[*]}"
echo "Pares HK : ${HK_PAIRS[*]}"
echo "Loss     : temporal=${TEMPORAL_LOSS}, cross=${CROSS_LOSS}, lambda=${CROSS_LAMBDA}, cross_norm=${CROSS_SCORE_NORMALIZATION}"
echo "Tarefas  : ${N_TASKS}"

sbatch \
  -p "$GPU_PARTITION" \
  --gres=gpu:1 \
  --time="$GPU_TIME" \
  --array="0-$((N_TASKS - 1))%${MAX_GPU_JOBS}" \
  --job-name="mse_lambda_00" \
  --output="${LOG_ROOT}/mse_lambda_00_%A_%a.out" \
  --error="${LOG_ROOT}/mse_lambda_00_%A_%a.err" \
  "$SCRIPT_PATH" worker