#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Experimento: loss composta temporal + cross-sectional por janelas de trading
#
# Edite somente a seção "HIPERPARÂMETROS GLOBAIS DO EXPERIMENTO".
#
# A loss recebe previsões [B,H,N]:
#   L = (1-lambda) * L_temporal + lambda * L_cross
#
# A parte cross-sectional divide H em blocos não sobrepostos de tamanho K.
# Somente pares (H,K) com K <= H e H % K == 0 são executados.
#
# Uso no login node:
#   bash scripts/run_composite_trading_experiment.sh
#
# Para executar diretamente um único worker Slurm:
#   sbatch ... scripts/run_composite_trading_experiment.sh worker
# ==============================================================================

# ------------------------------------------------------------------------------
# HIPERPARÂMETROS GLOBAIS DO EXPERIMENTO
# ------------------------------------------------------------------------------

MODELS=(
  DUET
  Nonstationary_Transformer
  FEDformer
  TimesNet
)

DATASETS=(
  retornos_simples
  log_retornos
)

LOOKBACKS=(32 104 246)
HORIZONS=(1 5 10 15 20 24)
TRADE_WINDOWS=(1 5 10 15 20 24)

# Loss composta.
TEMPORAL_LOSS="mse"
CROSS_LOSS="mse"          # preparado também para: ranknet, listnet, bpr, hinge
CROSS_LAMBDA="0.50"       # peso da parcela cross-sectional
SCORE_KIND="simple_return"

# ------------------------------------------------------------------------------
# CONFIGURAÇÃO OPERACIONAL
# ------------------------------------------------------------------------------

TFB_ROOT="${TFB_ROOT:-/sonic_home/igor.viveiros/src/TFB}"
VENV_PATH="${VENV_PATH:-/sonic_home/igor.viveiros/py310/bin/activate}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_FILE="${CONFIG_FILE:-rolling_forecast_config.json}"

OUT_ROOT="${OUT_ROOT:-/snfs2/igor.viveiros/previsoes/composite_trading}"
LOG_ROOT="${LOG_ROOT:-${TFB_ROOT}/logs/composite_trading}"

GPU_PARTITION="${GPU_PARTITION:-medusas_shr}"
GPU_TIME="${GPU_TIME:-48:00:00}"
MAX_GPU_JOBS="${MAX_GPU_JOBS:-6}"

SEED="${SEED:-2026}"
TV_RATIO="${TV_RATIO:-0.8}"
TRAIN_RATIO_IN_TV="${TRAIN_RATIO_IN_TV:-0.875}"
STRIDE="${STRIDE:-1}"
NUM_ROLLINGS="${NUM_ROLLINGS:-48000}"

SCRIPT_PATH="$(readlink -f "$0")"

# label:candidatos:data_kind:step_offset
DATASET_SPECS_ALL=(
  "retornos_simples:b3_return_tfb.csv|b3_returns.csv|b3_daily_return.csv:simple_return:2"
  "log_retornos:b3_log_returns.csv|b3_log_return_tfb.csv:log_return:2"
)

# ------------------------------------------------------------------------------
# Funções de desenho experimental
# ------------------------------------------------------------------------------

build_dataset_specs() {
  DATASET_SPECS=()
  local requested spec label
  for requested in "${DATASETS[@]}"; do
    local found=0
    for spec in "${DATASET_SPECS_ALL[@]}"; do
      label="${spec%%:*}"
      if [[ "$requested" == "$label" ]]; then
        DATASET_SPECS+=("$spec")
        found=1
        break
      fi
    done
    if (( found == 0 )); then
      echo "ERRO: dataset inválido: ${requested}. Permitidos: retornos_simples log_retornos" >&2
      exit 2
    fi
  done
}

validate_models() {
  local model
  for model in "${MODELS[@]}"; do
    case "$model" in
      DUET|Nonstationary_Transformer|FEDformer|TimesNet) ;;
      *)
        echo "ERRO: modelo inválido: $model" >&2
        exit 2
        ;;
    esac
  done
}

build_valid_hk_pairs() {
  HK_PAIRS=()
  local h k
  for h in "${HORIZONS[@]}"; do
    if (( h <= 0 )); then
      echo "ERRO: horizonte deve ser positivo: H=$h" >&2
      exit 2
    fi
    for k in "${TRADE_WINDOWS[@]}"; do
      if (( k <= 0 )); then
        echo "ERRO: janela de trade deve ser positiva: K=$k" >&2
        exit 2
      fi
      if (( k <= h && h % k == 0 )); then
        HK_PAIRS+=("${h}:${k}")
      fi
    done
  done

  if (( ${#HK_PAIRS[@]} == 0 )); then
    echo "ERRO: nenhum par (H,K) válido. É necessário K <= H e H divisível por K." >&2
    exit 2
  fi
}

resolve_data_file() {
  local candidates="$1"
  local candidate
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

  [[ -f "${TFB_ROOT}/config/${CONFIG_FILE}" ]] || {
    echo "ERRO: config TFB não encontrada: ${TFB_ROOT}/config/${CONFIG_FILE}" >&2
    exit 2
  }
  [[ -f "scripts/run_benchmark_composite_trading_loss.py" ]] || {
    echo "ERRO: launcher da loss composta não encontrado." >&2
    exit 2
  }
  [[ -f "scripts/convert_composite_trading_predictions.py" ]] || {
    echo "ERRO: conversor composite_trading não encontrado." >&2
    exit 2
  }

  echo "TFB commit: $(git rev-parse HEAD 2>/dev/null || echo desconhecido)"
}

# ------------------------------------------------------------------------------
# Hiperparâmetros ESPECÍFICOS dos modelos.
# Não fazem parte da grade global editável.
# ------------------------------------------------------------------------------

model_args() {
  local model_key="$1"
  local lb="$2"
  local h="$3"
  local k="$4"
  local data_kind="$5"

  MODEL_NAME=""
  MODEL_HYPER_PARAMS=""
  ADAPTER_ARG=()
  DETERMINISTIC_MODE="full"

  local loss_fields
  loss_fields="\"loss\":\"composite_trading\",\"loss_temporal\":\"${TEMPORAL_LOSS}\",\"loss_cross\":\"${CROSS_LOSS}\",\"loss_trade_window\":${k},\"loss_cross_lambda\":${CROSS_LAMBDA},\"loss_data_kind\":\"${data_kind}\",\"loss_score_kind\":\"${SCORE_KIND}\",\"loss_inverse_norm\":true,\"loss_track_components\":true"

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
  local data_file="$1"
  local model_name="$2"
  local model_hyper="$3"
  local deterministic="$4"
  local h="$5"
  local save_subdir="$6"
  shift 6
  local adapter_args=("$@")

  local result_dir="${TFB_ROOT}/result/${save_subdir}"
  rm -rf "$result_dir"
  mkdir -p "$result_dir"

  "$PYTHON_BIN" ./scripts/run_benchmark_composite_trading_loss.py \
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
  local result_dir="$1"
  local h="$2"
  local decoded_dir="$3"

  rm -rf "$decoded_dir"
  mkdir -p "$decoded_dir"

  mapfile -t tars < <(find "$result_dir" -maxdepth 1 -type f -name '*.csv.tar.gz' | sort)
  (( ${#tars[@]} > 0 )) || {
    echo "ERRO: nenhum .csv.tar.gz em $result_dir" >&2
    exit 4
  }

  local copy_index=0
  local tarfile extracted_dir decoded_csv rows
  for tarfile in "${tars[@]}"; do
    "$PYTHON_BIN" ts_benchmark/utils/decode_prediction.py "$tarfile"
    extracted_dir="$(dirname "$tarfile")/$(basename "$tarfile" .tar.gz)_extracted"
    [[ -d "$extracted_dir" ]] || {
      echo "ERRO: pasta extraída não encontrada: $extracted_dir" >&2
      exit 4
    }

    while IFS= read -r decoded_csv; do
      rows=$(($(wc -l < "$decoded_csv") - 1))
      [[ "$rows" -eq "$h" ]] || continue
      cp "$decoded_csv" "${decoded_dir}/csv_sample_${copy_index}_inference_data.csv"
      copy_index=$((copy_index + 1))
    done < <(find "$extracted_dir" -type f -name 'inference_data.csv' | sort)
  done

  (( copy_index > 0 )) || {
    echo "ERRO: nenhuma previsão decodificada com h=$h." >&2
    exit 4
  }
}

convert_predictions() {
  local decoded_dir="$1"
  local original_dataset="$2"
  local h="$3"
  local lb="$4"
  local step_offset="$5"
  local final_dir="$6"

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

write_manifest() {
  mkdir -p "$OUT_ROOT"
  local manifest="${OUT_ROOT}/design_composite_trading.csv"

  {
    echo "dataset,modelo,lookback,pred_len,k,temporal_loss,cross_loss,cross_lambda,score_kind,seed"
    local spec label candidates data_kind offset model lb pair h k
    for spec in "${DATASET_SPECS[@]}"; do
      IFS=':' read -r label candidates data_kind offset <<< "$spec"
      for model in "${MODELS[@]}"; do
        for lb in "${LOOKBACKS[@]}"; do
          for pair in "${HK_PAIRS[@]}"; do
            IFS=':' read -r h k <<< "$pair"
            echo "${label},${model},${lb},${h},${k},${TEMPORAL_LOSS},${CROSS_LOSS},${CROSS_LAMBDA},${SCORE_KIND},${SEED}"
          done
        done
      done
    done
  } > "$manifest"

  echo "Manifesto: $manifest"
}

# ------------------------------------------------------------------------------
# Inicialização do desenho
# ------------------------------------------------------------------------------

validate_models
build_dataset_specs
build_valid_hk_pairs

N_DATASETS=${#DATASET_SPECS[@]}
N_MODELS=${#MODELS[@]}
N_LOOKBACKS=${#LOOKBACKS[@]}
N_HK=${#HK_PAIRS[@]}
N_TASKS=$((N_DATASETS * N_MODELS * N_LOOKBACKS * N_HK))

# ------------------------------------------------------------------------------
# Worker
# ------------------------------------------------------------------------------

run_worker() {
  prepare_worker

  local task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID não definido}"
  if (( task_id < 0 || task_id >= N_TASKS )); then
    echo "ERRO: task_id fora da grade: $task_id / $N_TASKS" >&2
    exit 3
  fi

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
  local pair="${HK_PAIRS[$hk_idx]}"
  local h k
  IFS=':' read -r h k <<< "$pair"

  local data_file
  data_file="$(resolve_data_file "$candidates")"
  local original_dataset="${TFB_ROOT}/dataset/forecasting/${data_file}"

  model_args "$model_key" "$lb" "$h" "$k" "$data_kind"

  local loss_tag="${TEMPORAL_LOSS}_${CROSS_LOSS}_lambda${CROSS_LAMBDA}"
  loss_tag="${loss_tag//./p}"

  local save_subdir="composite_trading/${dataset_label}_${model_key}_lb${lb}_h${h}_k${k}_${loss_tag}_seed${SEED}"
  local result_dir="${TFB_ROOT}/result/${save_subdir}"
  local decoded_dir="${result_dir}/decoded_h${h}"
  local final_dir="${OUT_ROOT}/${dataset_label}/${model_key}/lookback_${lb}/pred_len_${h}/k_${k}/${loss_tag}"

  echo "======================================================================"
  echo "task=$task_id/$((N_TASKS - 1))"
  echo "dataset=$dataset_label ($data_file; data_kind=$data_kind)"
  echo "modelo=$model_key"
  echo "lookback=$lb"
  echo "H=$h"
  echo "K=$k"
  echo "loss temporal=$TEMPORAL_LOSS"
  echo "loss cross=$CROSS_LOSS"
  echo "lambda=$CROSS_LAMBDA"
  echo "saída=$final_dir"
  echo "======================================================================"

  run_tfb \
    "$data_file" \
    "$MODEL_NAME" \
    "$MODEL_HYPER_PARAMS" \
    "$DETERMINISTIC_MODE" \
    "$h" \
    "$save_subdir" \
    "${ADAPTER_ARG[@]}"

  decode_predictions "$result_dir" "$h" "$decoded_dir"

  convert_predictions \
    "$decoded_dir" \
    "$original_dataset" \
    "$h" \
    "$lb" \
    "$step_offset" \
    "$final_dir"

  echo "OK: ${dataset_label} ${model_key} lb=${lb} H=${h} K=${k}"
}

# ------------------------------------------------------------------------------
# Launcher
# ------------------------------------------------------------------------------

launch() {
  mkdir -p "$LOG_ROOT" "$OUT_ROOT"

  echo "Pares (H,K) válidos:"
  printf '  %s\n' "${HK_PAIRS[@]}"
  echo
  echo "Datasets : ${DATASETS[*]}"
  echo "Modelos  : ${MODELS[*]}"
  echo "Lookbacks: ${LOOKBACKS[*]}"
  echo "Loss     : temporal=${TEMPORAL_LOSS}; cross=${CROSS_LOSS}; lambda=${CROSS_LAMBDA}"
  echo "Tarefas  : ${N_TASKS}"

  write_manifest

  local array_spec="0-$((N_TASKS - 1))%${MAX_GPU_JOBS}"

  local job_id
  job_id=$(sbatch --parsable \
    -p "$GPU_PARTITION" \
    --gres=gpu:1 \
    --array="$array_spec" \
    --time="$GPU_TIME" \
    --job-name="tfb_composite" \
    --output="${LOG_ROOT}/composite-%A_%a.out" \
    --error="${LOG_ROOT}/composite-%A_%a.err" \
    "$SCRIPT_PATH" worker)

  echo "Array submetido: ${job_id}"
  echo "squeue -j ${job_id}"
}

case "${1:-launcher}" in
  launcher)
    launch
    ;;
  worker)
    run_worker
    ;;
  *)
    echo "Uso: bash $0 [launcher|worker]" >&2
    exit 2
    ;;
esac
