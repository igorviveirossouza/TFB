#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# tfb_check
# Recria o experimento temporal TFB (estimacao + conversao) e adiciona VAR.
#
# Grade TFB original:
#   datasets : retornos_simples, log_retornos
#   modelos  : DUET, TimesNet, FEDformer, Nonstationary_Transformer
#   lookbacks: 32, 104, 246
#   pred_len : 1, 5, 10, 15, 20, 24
#   trading k: k = pred_len
#   seed     : 2021
#
# VAR:
#   implementacao oficial TFB self_impl.VAR_model
#   lag oficial/default do repositorio: 13
#   VAR nao possui seq_len/lookback 32/104/246; portanto e estimado uma vez por
#   dataset x horizonte e salvo como lookback_13 (lag VAR = 13).
#
# Saida final:
#   /snfs2/igor.viveiros/previsoes/tfb_check/
#
# Uso no login node:
#   bash tfb_check.sh
# ==============================================================================

TFB_ROOT="${TFB_ROOT:-/sonic_home/igor.viveiros/src/TFB}"
VENV_PATH="${VENV_PATH:-/sonic_home/igor.viveiros/py310/bin/activate}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-rolling_forecast_config.json}"
OUT_ROOT="${OUT_ROOT:-/snfs2/igor.viveiros/previsoes/tfb_check}"
LOG_ROOT="${LOG_ROOT:-${TFB_ROOT}/logs/tfb_check}"

GPU_PARTITION="medusas_shr"
CPU_PARTITION="medusas_shr"
GPU_TIME="${GPU_TIME:-48:00:00}"
CPU_TIME="${CPU_TIME:-24:00:00}"
MAX_GPU_JOBS="${MAX_GPU_JOBS:-8}"
MAX_CPU_JOBS="${MAX_CPU_JOBS:-4}"

SEED=2026
TV_RATIO=0.8
TRAIN_RATIO_IN_TV=0.875
STRIDE=1
NUM_ROLLINGS=48000

MODELS=(DUET TimesNet FEDformer Nonstationary_Transformer)
LOOKBACKS=(32 104 246)
HORIZONS=(1 5 10 15 20 24)

# label : candidatos do CSV no TFB : step_offset para alinhamento temporal
# step_offset=2 porque uma linha de retorno representa o movimento entre dois
# instantes consecutivos; assim, o primeiro alvo previsto fica no instante seguinte
# ao ultimo retorno observado na origem da previsao.
DATASET_SPECS=(
  "retornos_simples:b3_return_tfb.csv|b3_returns.csv|b3_daily_return.csv:2"
  "log_retornos:b3_log_returns.csv|b3_log_return_tfb.csv:2"
)

N_DATASETS=${#DATASET_SPECS[@]}
N_MODELS=${#MODELS[@]}
N_LOOKBACKS=${#LOOKBACKS[@]}
N_HORIZONS=${#HORIZONS[@]}
N_DEEP=$((N_DATASETS * N_MODELS * N_LOOKBACKS * N_HORIZONS))
N_VAR=$((N_DATASETS * N_HORIZONS))

SCRIPT_PATH="$(readlink -f "$0")"

# ------------------------------------------------------------------------------
# Funcoes
# ------------------------------------------------------------------------------
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
  if [[ ! -d "$TFB_ROOT" ]]; then
    echo "ERRO: TFB_ROOT nao existe: $TFB_ROOT" >&2
    exit 2
  fi
  if [[ ! -f "$VENV_PATH" ]]; then
    echo "ERRO: ambiente Python nao encontrado: $VENV_PATH" >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "$VENV_PATH"
  cd "$TFB_ROOT"
  mkdir -p "$OUT_ROOT" "$LOG_ROOT"

if [[ ! -f "${TFB_ROOT}/config/${CONFIG_PATH}" ]]; then
    echo "ERRO: config TFB nao encontrada: ${TFB_ROOT}/${CONFIG_PATH}" >&2
    exit 2
  fi
  if [[ ! -f "ts_benchmark/utils/decode_prediction.py" ]]; then
    echo "ERRO: decode_prediction.py nao encontrado." >&2
    exit 2
  fi

  echo "TFB commit: $(git rev-parse HEAD 2>/dev/null || echo desconhecido)"
}

model_args() {
  local model_key="$1"
  local lb="$2"
  local h="$3"

  MODEL_NAME=""
  MODEL_HYPER_PARAMS=""
  ADAPTER_ARG=()
  DETERMINISTIC_MODE="full"

  case "$model_key" in
    DUET)
      MODEL_NAME="duet.DUET"
      MODEL_HYPER_PARAMS="{\"CI\":1,\"batch_size\":32,\"d_ff\":32,\"d_model\":32,\"dropout\":0.5,\"e_layers\":1,\"factor\":3,\"fc_dropout\":0.2,\"hidden_size\":256,\"pred_len\":${h},\"horizon\":${h},\"k\":1,\"loss\":\"MAE\",\"lr\":0.01,\"lradj\":\"type1\",\"n_heads\":2,\"norm\":true,\"num_epochs\":100,\"num_experts\":4,\"patch_len\":48,\"patience\":10,\"seq_len\":${lb}}"
      ;;
    TimesNet)
      MODEL_NAME="time_series_library.TimesNet"
      MODEL_HYPER_PARAMS="{\"batch_size\":32,\"d_ff\":512,\"d_model\":256,\"factor\":3,\"pred_len\":${h},\"horizon\":${h},\"norm\":true,\"seq_len\":${lb},\"top_k\":5}"
      ADAPTER_ARG=(--adapter "transformer_adapter")
      DETERMINISTIC_MODE="efficient"
      ;;
    FEDformer)
      MODEL_NAME="time_series_library.FEDformer"
      MODEL_HYPER_PARAMS="{\"batch_size\":32,\"d_ff\":512,\"d_model\":256,\"factor\":3,\"pred_len\":${h},\"horizon\":${h},\"norm\":true,\"seq_len\":${lb}}"
      ADAPTER_ARG=(--adapter "transformer_adapter")
      ;;
    Nonstationary_Transformer)
      MODEL_NAME="time_series_library.Nonstationary_Transformer"
      MODEL_HYPER_PARAMS="{\"d_ff\":256,\"d_model\":128,\"dropout\":0.1,\"factor\":3,\"pred_len\":${h},\"horizon\":${h},\"norm\":true,\"p_hidden_dims\":[32,32],\"p_hidden_layers\":2,\"seq_len\":${lb}}"
      ADAPTER_ARG=(--adapter "transformer_adapter")
      ;;
    *)
      echo "ERRO: modelo desconhecido: $model_key" >&2
      exit 3
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

  "$PYTHON_BIN" ./scripts/run_benchmark.py \
    --config-path "$CONFIG_PATH" \
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
  if (( ${#tars[@]} == 0 )); then
    echo "ERRO: nenhum .csv.tar.gz em $result_dir" >&2
    exit 4
  fi

  local copy_index=0
  local tarfile extracted_dir decoded_csv rows
  for tarfile in "${tars[@]}"; do
    "$PYTHON_BIN" ts_benchmark/utils/decode_prediction.py "$tarfile"
    extracted_dir="$(dirname "$tarfile")/$(basename "$tarfile" .tar.gz)_extracted"
    if [[ ! -d "$extracted_dir" ]]; then
      echo "ERRO: pasta extraida nao encontrada: $extracted_dir" >&2
      exit 4
    fi

    while IFS= read -r decoded_csv; do
      rows=$(($(wc -l < "$decoded_csv") - 1))
      [[ "$rows" -eq "$h" ]] || continue
      cp "$decoded_csv" "${decoded_dir}/csv_sample_${copy_index}_inference_data.csv"
      copy_index=$((copy_index + 1))
    done < <(find "$extracted_dir" -type f -name 'inference_data.csv' | sort)
  done

  if (( copy_index == 0 )); then
    echo "ERRO: nenhuma previsao decodificada com h=$h." >&2
    exit 4
  fi
}

convert_no_leakage() {
  local decoded_dir="$1"
  local original_dataset="$2"
  local h="$3"
  local lookback="$4"
  local step_offset="$5"
  local final_dir="$6"

  rm -rf "$final_dir"
  mkdir -p "$final_dir"

  "$PYTHON_BIN" - "$decoded_dir" "$original_dataset" "$h" "$lookback" "$step_offset" "$final_dir" "$TV_RATIO" <<'PY'
from pathlib import Path
import re
import sys
import pandas as pd

pred_dir = Path(sys.argv[1])
original_path = Path(sys.argv[2])
pred_len = int(sys.argv[3])
lookback = int(sys.argv[4])
step_offset = int(sys.argv[5])
out_dir = Path(sys.argv[6])
tv_ratio = float(sys.argv[7])

rx = re.compile(r"csv_sample_(\d+)_inference_data\.csv$")
files = []
for p in pred_dir.glob("csv_sample_*_inference_data.csv"):
    m = rx.search(p.name)
    if m:
        files.append((int(m.group(1)), p))
files.sort()

if not files:
    raise RuntimeError(f"Nenhum csv_sample em {pred_dir}")
idx = [i for i, _ in files]
if idx != list(range(len(idx))):
    raise RuntimeError(f"sample_idx nao contiguos: inicio={idx[:5]} fim={idx[-5:]}")

orig = pd.read_csv(original_path)
reserved = {"step", "date"}

if "cols" in orig.columns:
    work = orig[orig["cols"].astype(str).str.lower() != "label"].copy()
    if "date" in work.columns:
        original_len = int(work.groupby("cols", sort=False)["date"].nunique().max())
    else:
        original_len = int(work["cols"].value_counts(sort=False).max())
    target_cols = []
    seen = set()
    for x in work["cols"].tolist():
        s = str(x).strip()
        if s and s.lower() != "label" and s not in reserved and s not in seen:
            seen.add(s)
            target_cols.append(s)
else:
    original_len = len(orig)
    target_cols = [str(c) for c in orig.columns if str(c) not in reserved and str(c).lower() != "label"]

n_files = len(files)
first_zero = original_len - pred_len - (n_files - 1)
expected_first_zero = int(tv_ratio * original_len)

# Guarda central contra desalinhamento/leakage: o primeiro alvo do rolling
# deve ser exatamente o primeiro indice do bloco de teste do TFB.
if first_zero != expected_first_zero:
    raise RuntimeError(
        "Fronteira temporal inconsistente: "
        f"first_zero={first_zero}, esperado={expected_first_zero}, "
        f"N={original_len}, h={pred_len}, n_janelas={n_files}."
    )

if first_zero - lookback < 0:
    raise RuntimeError("Lookback/lag invade o inicio da serie.")

for sample_idx, path in files:
    df = pd.read_csv(path)
    for c in list(df.columns):
        if str(c) in reserved:
            df = df.drop(columns=[c])
    if len(df) != pred_len:
        raise RuntimeError(f"{path.name}: esperado {pred_len} linhas, obtido {len(df)}")
    if len(df.columns) != len(target_cols):
        raise RuntimeError(
            f"{path.name}: {len(df.columns)} colunas previstas versus {len(target_cols)} ativos no dataset"
        )
    df.columns = target_cols

    start_zero = first_zero + sample_idx
    zero_steps = list(range(start_zero, start_zero + pred_len))
    visible_steps = [z + step_offset for z in zero_steps]
    origin_step = visible_steps[0] - 1

    df["h"] = range(1, pred_len + 1)
    df["origin_step"] = origin_step
    df["step"] = visible_steps

    if not ((df["step"] - df["origin_step"]) == df["h"]).all():
        raise RuntimeError(f"Invariante temporal falhou em {path.name}")

    df.to_csv(out_dir / f"janela_{sample_idx:06d}.csv", index=False)

first = pd.read_csv(out_dir / "janela_000000.csv")
print(
    f"Conversao OK: N={original_len}; janelas={n_files}; "
    f"origem_inicial={int(first.origin_step.iloc[0])}; "
    f"alvo_h1={int(first.step.iloc[0])}; h={pred_len}; offset={step_offset}"
)
PY
}

write_manifest() {
  mkdir -p "$OUT_ROOT"
  local manifest="${OUT_ROOT}/design_tfb_check.csv"
  {
    echo "dataset,modelo,lookback,pred_len,k,seed,observacao"
    local spec label candidates offset model lb h
    for spec in "${DATASET_SPECS[@]}"; do
      IFS=':' read -r label candidates offset <<< "$spec"
      for model in "${MODELS[@]}"; do
        for lb in "${LOOKBACKS[@]}"; do
          for h in "${HORIZONS[@]}"; do
            echo "${label},${model},${lb},${h},${h},${SEED},temporal_pointwise"
          done
        done
      done
      for h in "${HORIZONS[@]}"; do
        echo "${label},VAR,13,${h},${h},${SEED},TFB_VAR_lags_13"
      done
    done
  } > "$manifest"
  echo "Design salvo em: $manifest"
}

# ------------------------------------------------------------------------------
# Launcher: um array GPU para redes profundas e um array CPU para VAR.
# ------------------------------------------------------------------------------
MODE="${1:-launcher}"
if [[ "$MODE" == "launcher" ]]; then
  mkdir -p "$LOG_ROOT" "$OUT_ROOT"
  write_manifest

  deep_job=$(sbatch --parsable \
    -p "$GPU_PARTITION" \
    --gres=gpu:1 \
    --array="0-$((N_DEEP - 1))%${MAX_GPU_JOBS}" \
    --time="$GPU_TIME" \
    --job-name="tfb_check_deep" \
    --output="${LOG_ROOT}/deep-%A_%a.out" \
    --error="${LOG_ROOT}/deep-%A_%a.err" \
    "$SCRIPT_PATH" deep)

  var_job=$(sbatch --parsable \
    -p "$CPU_PARTITION" \
    --array="0-$((N_VAR - 1))%${MAX_CPU_JOBS}" \
    --time="$CPU_TIME" \
    --job-name="tfb_check_var" \
    --output="${LOG_ROOT}/var-%A_%a.out" \
    --error="${LOG_ROOT}/var-%A_%a.err" \
    "$SCRIPT_PATH" var)

  echo "tfb_check submetido."
  echo "Deep models job: $deep_job  (${N_DEEP} tarefas)"
  echo "VAR job:         $var_job  (${N_VAR} tarefas)"
  echo "Previsoes:       $OUT_ROOT"
  exit 0
fi

prepare_worker
TASK_ID="${SLURM_ARRAY_TASK_ID:?Este modo deve rodar como array Slurm.}"

# ------------------------------------------------------------------------------
# Redes profundas: 2 datasets x 4 modelos x 3 LB x 6 horizontes = 144.
# ------------------------------------------------------------------------------
if [[ "$MODE" == "deep" ]]; then
  if (( TASK_ID >= N_DEEP )); then
    exit 0
  fi

  H_IDX=$((TASK_ID % N_HORIZONS))
  LB_IDX=$(((TASK_ID / N_HORIZONS) % N_LOOKBACKS))
  MODEL_IDX=$(((TASK_ID / (N_HORIZONS * N_LOOKBACKS)) % N_MODELS))
  DATASET_IDX=$((TASK_ID / (N_HORIZONS * N_LOOKBACKS * N_MODELS)))

  H="${HORIZONS[$H_IDX]}"
  LB="${LOOKBACKS[$LB_IDX]}"
  MODEL_KEY="${MODELS[$MODEL_IDX]}"
  SPEC="${DATASET_SPECS[$DATASET_IDX]}"
  IFS=':' read -r DATASET_LABEL DATA_CANDIDATES STEP_OFFSET <<< "$SPEC"
  DATA_FILE="$(resolve_data_file "$DATA_CANDIDATES")"
  ORIGINAL_DATASET="${TFB_ROOT}/dataset/forecasting/${DATA_FILE}"

  model_args "$MODEL_KEY" "$LB" "$H"

  RUN_NAME="${DATASET_LABEL}_${MODEL_KEY}_lb${LB}_h${H}_k${H}_seed${SEED}"
  SAVE_SUBDIR="tfb_check/${RUN_NAME}"
  RESULT_DIR="${TFB_ROOT}/result/${SAVE_SUBDIR}"
  DECODED_DIR="${OUT_ROOT}/_decoded/${DATASET_LABEL}/${MODEL_KEY}/lookback_${LB}/pred_len_${H}"
  FINAL_DIR="${OUT_ROOT}/${DATASET_LABEL}/${MODEL_KEY}/lookback_${LB}/pred_len_${H}/k_${H}"

  echo "TASK=$TASK_ID DATASET=$DATASET_LABEL DATA=$DATA_FILE MODEL=$MODEL_KEY LB=$LB H=$H K=$H"

  run_tfb "$DATA_FILE" "$MODEL_NAME" "$MODEL_HYPER_PARAMS" "$DETERMINISTIC_MODE" "$H" "$SAVE_SUBDIR" "${ADAPTER_ARG[@]}"
  decode_predictions "$RESULT_DIR" "$H" "$DECODED_DIR"
  convert_no_leakage "$DECODED_DIR" "$ORIGINAL_DATASET" "$H" "$LB" "$STEP_OFFSET" "$FINAL_DIR"

  echo "OK: $FINAL_DIR"
  exit 0
fi

# ------------------------------------------------------------------------------
# VAR oficial TFB: 2 datasets x 6 horizontes = 12.
# O repositorio oficial usa self_impl.VAR_model e o default lags=13.
# Fixamos lags=13 explicitamente para reproducibilidade.
# ------------------------------------------------------------------------------
if [[ "$MODE" == "var" ]]; then
  if (( TASK_ID >= N_VAR )); then
    exit 0
  fi

  H_IDX=$((TASK_ID % N_HORIZONS))
  DATASET_IDX=$((TASK_ID / N_HORIZONS))

  H="${HORIZONS[$H_IDX]}"
  SPEC="${DATASET_SPECS[$DATASET_IDX]}"
  IFS=':' read -r DATASET_LABEL DATA_CANDIDATES STEP_OFFSET <<< "$SPEC"
  DATA_FILE="$(resolve_data_file "$DATA_CANDIDATES")"
  ORIGINAL_DATASET="${TFB_ROOT}/dataset/forecasting/${DATA_FILE}"

  VAR_LAGS=13
  MODEL_NAME="self_impl.VAR_model"
  MODEL_HYPER_PARAMS="{\"lags\":${VAR_LAGS}}"
  DETERMINISTIC_MODE="efficient"
  ADAPTER_ARG=()

  RUN_NAME="${DATASET_LABEL}_VAR_lags${VAR_LAGS}_h${H}_k${H}_seed${SEED}"
  SAVE_SUBDIR="tfb_check/${RUN_NAME}"
  RESULT_DIR="${TFB_ROOT}/result/${SAVE_SUBDIR}"
  DECODED_DIR="${OUT_ROOT}/_decoded/${DATASET_LABEL}/VAR/lookback_${VAR_LAGS}/pred_len_${H}"
  FINAL_DIR="${OUT_ROOT}/${DATASET_LABEL}/VAR/lookback_${VAR_LAGS}/pred_len_${H}/k_${H}"

  echo "TASK=$TASK_ID DATASET=$DATASET_LABEL DATA=$DATA_FILE MODEL=VAR LAGS=$VAR_LAGS H=$H K=$H"

  run_tfb "$DATA_FILE" "$MODEL_NAME" "$MODEL_HYPER_PARAMS" "$DETERMINISTIC_MODE" "$H" "$SAVE_SUBDIR" "${ADAPTER_ARG[@]}"
  decode_predictions "$RESULT_DIR" "$H" "$DECODED_DIR"
  convert_no_leakage "$DECODED_DIR" "$ORIGINAL_DATASET" "$H" "$VAR_LAGS" "$STEP_OFFSET" "$FINAL_DIR"

  echo "OK: $FINAL_DIR"
  exit 0
fi

echo "ERRO: modo invalido: $MODE" >&2
exit 9
