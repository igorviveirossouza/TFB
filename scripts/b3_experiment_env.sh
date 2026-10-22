#!/usr/bin/env bash
# Implementação comum: transforma as variáveis dos três run*.sh em configuração.
# O usuário edita somente o script do experimento que deseja executar.
set -euo pipefail
export TFB_ROOT VENV_PATH PYTHON_BIN DATASET_PATH OUT_ROOT RESULT_ROOT LOG_ROOT
export GRES NODELIST CPUS_PER_TASK MEMORY ARRAY_CHUNK_SIZE MANTER_CSV B3_CONFIG_SOURCE
export GPU_PARTITION="$PARTITION"
export GPU_TIME="$TIME_LIMIT"
export MAX_GPU_JOBS="$MAX_CONCURRENT_JOBS"
if [[ -f "$VENV_PATH" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH"
fi
cd "$TFB_ROOT"

if [[ "${1:-}" == worker && -n "${B3_INLINE_CONFIG:-}" ]]; then
  return
fi

# Campos sem uso em um experimento recebem apenas valores internos neutros.
if [[ "$EXPERIMENT" == "var" ]]; then
  MODELS=(VAR); LOOKBACKS=("$VAR_LAGS")
  MODEL_DEFAULTS_JSON='{"VAR": {}}'
fi
if [[ "$EXPERIMENT" != "composto" ]]; then
  TRADE_WINDOWS=(1); CROSS_LOSSES=(ranknet); LAMBDAS=(0.5); DELTAS=(0)
fi

# Optional for old launcher copies/configs; mandatory only for *_proximity losses.
# Pass each array item independently so malformed vectors cannot be truncated.
mask_arguments=()
if [[ "$EXPERIMENT" == "composto" ]] && declare -p CROSS_MASK &>/dev/null; then
  mask_arguments+=(loss.loss_cross_mask present)
  for mask_item in "${CROSS_MASK[@]}"; do
    mask_arguments+=(loss.loss_cross_mask_item "$mask_item")
  done
fi

# Cada valor chega ao Python como argumento separado, sem interpolação em código.
export B3_INLINE_CONFIG
B3_INLINE_CONFIG=$("$PYTHON_BIN" - \
  source_data_kind "$SOURCE_DATA_KIND" \
  train_percentages "${TRAIN_PERCENTAGES[*]}" models "${MODELS[*]}" \
  lookbacks "${LOOKBACKS[*]}" horizons "${HORIZONS[*]}" \
  trade_windows "${TRADE_WINDOWS[*]}" cross_losses "${CROSS_LOSSES[*]}" \
  lambdas "${LAMBDAS[*]}" deltas "${DELTAS[*]}" \
  model_defaults "$MODEL_DEFAULTS_JSON" var_lags "${VAR_LAGS:-32}" \
  training.num_epochs "${NUM_EPOCHS:-100}" training.patience "${PATIENCE:-10}" \
  training.lr "${LEARNING_RATE:-0.0001}" \
  loss.loss_temporal "${TEMPORAL_LOSS:-mse}" loss.loss_data_kind log_return \
  loss.loss_score_kind "${SCORE_KIND:-simple_return}" \
  loss.loss_cross_score_normalization "${CROSS_SCORE_NORMALIZATION:-zscore}" \
  loss.loss_cross_scale "${CROSS_SCALE:-1}" loss.loss_ranknet_alpha "${RANKNET_ALPHA:-1}" \
  loss.loss_listnet_tau "${LISTNET_TAU:-1}" loss.loss_hinge_margin "${HINGE_MARGIN:-0.1}" \
  loss.loss_inverse_norm "${LOSS_INVERSE_NORM:-true}" \
  loss.loss_track_components "${LOSS_TRACK_COMPONENTS:-true}" \
  strategy.train_ratio_in_tv "${TRAIN_RATIO_IN_TV:-0.875}" \
  strategy.stride "$STRIDE" strategy.num_rollings "$NUM_ROLLINGS" strategy.seed "$SEED" \
  "${mask_arguments[@]}" <<'PY'
import json
import sys

integer_lists = {'train_percentages', 'lookbacks', 'horizons', 'trade_windows'}
float_lists = {'lambdas', 'deltas'}
string_lists = {'models', 'cross_losses'}
integers = {'var_lags', 'training.num_epochs', 'training.patience',
            'strategy.stride', 'strategy.num_rollings', 'strategy.seed'}
floats = {'training.lr', 'strategy.train_ratio_in_tv', 'loss.loss_cross_scale',
          'loss.loss_ranknet_alpha', 'loss.loss_listnet_tau', 'loss.loss_hinge_margin'}
booleans = {'loss.loss_inverse_norm', 'loss.loss_track_components'}
config = {}
mask_items = None
for key, raw in zip(sys.argv[1::2], sys.argv[2::2]):
    if key == 'loss.loss_cross_mask':
        mask_items = []
        continue
    if key == 'loss.loss_cross_mask_item':
        mask_items.append(raw)
        continue
    if key in integer_lists:
        value = [int(x) for x in raw.split()]
    elif key in float_lists:
        value = [float(x) for x in raw.split()]
    elif key in string_lists:
        value = raw.split()
    elif key in integers:
        value = int(raw)
    elif key in floats:
        value = float(raw)
    elif key in booleans:
        if raw.lower() not in {'true', 'false'}:
            raise ValueError(key + ' deve ser true ou false')
        value = raw.lower() == 'true'
    elif key == 'model_defaults':
        value = json.loads(raw)
    else:
        value = raw
    parts = key.split('.')
    node = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
if mask_items is not None:
    if len(mask_items) != 2:
        raise ValueError('CROSS_MASK deve conter exatamente (tipo a)')
    config['loss']['loss_cross_mask'] = [mask_items[0], float(mask_items[1])]
print(json.dumps(config, allow_nan=False))
PY
)
