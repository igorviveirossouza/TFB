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

# Optional knobs stay absent from legacy configurations and cache identities.
extra_loss_arguments=()
[[ -z "${LISTMLE_TAU:-}" ]] || extra_loss_arguments+=(loss.loss_listmle_tau "$LISTMLE_TAU")
[[ -z "${APPROXNDCG_TAU:-}" ]] || extra_loss_arguments+=(loss.loss_approxndcg_tau "$APPROXNDCG_TAU")
[[ -z "${SPEARMAN_TAU:-}" ]] || extra_loss_arguments+=(loss.loss_spearman_tau "$SPEARMAN_TAU")
[[ -z "${HUBER_DELTA:-}" ]] || extra_loss_arguments+=(loss.loss_huber_delta "$HUBER_DELTA")
if [[ "$EXPERIMENT" == composto && " ${CROSS_LOSSES[*]} " =~ [[:space:]]rankmix(_proximity)?[[:space:]] ]]; then
  extra_loss_arguments+=(loss.loss_rank_mix_weights "${RANK_MIX_WEIGHTS:-0.5 0.5 0}"
                        loss.loss_rank_mix_mode "${RANK_MIX_MODE:-global}"
                        loss.loss_rank_mix_regions "${RANK_MIX_REGIONS:-top bottom top}"
                        loss.loss_rank_mix_fraction "${RANK_MIX_FRACTION:-0.3333333333333333}")
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
  training.lr "${LEARNING_RATE:-0.0001}" training.lradj "${LRADJ:-}" \
  training.validation_mode "${VALIDATION_MODE:-}" \
  loss.loss_temporal "${TEMPORAL_LOSS:-mse}" loss.loss_data_kind log_return \
  loss.loss_score_kind "${SCORE_KIND:-simple_return}" \
  loss.loss_cross_score_normalization "${CROSS_SCORE_NORMALIZATION:-zscore}" \
  loss.loss_cross_scale "${CROSS_SCALE:-1}" loss.loss_ranknet_alpha "${RANKNET_ALPHA:-1}" \
  loss.loss_listnet_tau "${LISTNET_TAU:-1}" loss.loss_hinge_margin "${HINGE_MARGIN:-0.1}" \
  loss.loss_inverse_norm "${LOSS_INVERSE_NORM:-true}" \
  loss.loss_track_components "${LOSS_TRACK_COMPONENTS:-true}" \
  loss.momentum_mode "${MOMENTUM_MODE:-off}" \
  loss.momentum_lookback "${MOMENTUM_LOOKBACK:-}" loss.momentum_eta "${MOMENTUM_ETA:-0}" \
  loss.momentum_gain_init "${MOMENTUM_GAIN_INIT:-0}" loss.momentum_diagnostics "${MOMENTUM_DIAGNOSTICS:-0}" \
  loss.momentum_input_mode "${MOMENTUM_INPUT_MODE:-off}" loss.momentum_input_lookback "${MOMENTUM_INPUT_LOOKBACK:-}" \
  momentum_experiment "$EXPERIMENT" \
  strategy.train_ratio_in_tv "${TRAIN_RATIO_IN_TV:-0.875}" \
  strategy.stride "$STRIDE" strategy.num_rollings "$NUM_ROLLINGS" strategy.seed "$SEED" \
  "${mask_arguments[@]}" "${extra_loss_arguments[@]}" <<'PY'
import json
import math
import sys

integer_lists = {'train_percentages', 'lookbacks', 'horizons', 'trade_windows'}
float_lists = {'lambdas', 'deltas', 'loss.loss_rank_mix_weights'}
string_lists = {'models', 'cross_losses', 'loss.loss_rank_mix_regions'}
integers = {'var_lags', 'training.num_epochs', 'training.patience',
            'strategy.stride', 'strategy.num_rollings', 'strategy.seed'}
floats = {'training.lr', 'strategy.train_ratio_in_tv', 'loss.loss_cross_scale',
          'loss.loss_ranknet_alpha', 'loss.loss_listnet_tau', 'loss.loss_hinge_margin',
          'loss.loss_spearman_tau', 'loss.loss_huber_delta',
          'loss.loss_listmle_tau', 'loss.loss_approxndcg_tau', 'loss.loss_rank_mix_fraction'}
booleans = {'loss.loss_inverse_norm', 'loss.loss_track_components'}
config = {}
mask_items = None
momentum_experiment = None
for key, raw in zip(sys.argv[1::2], sys.argv[2::2]):
    if key in {'training.lradj', 'training.validation_mode'} and not raw:
        continue  # Omit the override: retain each adapter's historical schedule.
    if key == 'momentum_experiment':
        momentum_experiment = raw
        continue
    if key == 'loss.loss_cross_mask':
        mask_items = []
        continue
    if key == 'loss.loss_cross_mask_item':
        mask_items.append(raw)
        continue
    if key in {'loss.momentum_lookback', 'loss.momentum_input_lookback'}:
        value = int(raw) if raw else None
    elif key in {'loss.momentum_eta', 'loss.momentum_gain_init'}:
        value = float(raw)
    elif key == 'loss.momentum_diagnostics':
        if raw.lower() not in {'true', 'false', '0', '1'}:
            raise ValueError('MOMENTUM_DIAGNOSTICS deve ser 0, 1, true ou false')
        value = raw.lower() in {'true', '1'}
    elif key in integer_lists:
        value = [int(x) for x in raw.split()]
    elif key in float_lists:
        value = [float(x) for x in raw.split()]
        if key == 'loss.loss_rank_mix_weights' and any(not math.isfinite(x) for x in value):
            raise ValueError('loss_rank_mix_weights exige pesos finitos.')
    elif key in string_lists:
        value = raw.split()
    elif key in integers:
        value = int(raw)
    elif key in floats:
        value = float(raw)
        if key == 'loss.loss_rank_mix_fraction' and not math.isfinite(value):
            raise ValueError('loss_rank_mix_fraction exige numero finito.')
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
    if len(mask_items) not in (2, 3, 4, 5):
        raise ValueError('CROSS_MASK deve conter tipo a [modo [forma...]]; maximo dois parametros de forma')
    mask = [mask_items[0], float(mask_items[1])]
    if len(mask_items) >= 3:
        mask.append(mask_items[2])
        mask.extend(float(item) for item in mask_items[3:])
    config['loss']['loss_cross_mask'] = mask
loss = config['loss']
if loss['momentum_input_mode'] not in {'off', 'zero', 'feature'}:
    raise ValueError('MOMENTUM_INPUT_MODE deve ser off, zero ou feature')
if loss['momentum_input_lookback'] is not None and loss['momentum_input_lookback'] < 1:
    raise ValueError('MOMENTUM_INPUT_LOOKBACK deve ser inteiro positivo ou vazio')
if 'validation_mode' in config['training'] and config['training']['validation_mode'] != 'ordered_sample_mean':
    raise ValueError('VALIDATION_MODE deve ser vazio ou ordered_sample_mean')
if momentum_experiment != 'composto' or loss['momentum_input_mode'] == 'off':
    loss.pop('momentum_input_mode')
    loss.pop('momentum_input_lookback')
elif loss['momentum_mode'] != 'off':
    raise ValueError('Momentum na entrada nao pode ser combinado com momentum residual_output')
if loss['momentum_mode'] not in {'off', 'residual_output'}:
    raise ValueError('MOMENTUM_MODE deve ser off ou residual_output')
if loss['momentum_lookback'] is not None and loss['momentum_lookback'] < 1:
    raise ValueError('MOMENTUM_LOOKBACK deve ser um inteiro positivo ou ficar vazio')
if not math.isfinite(loss['momentum_eta']) or loss['momentum_eta'] < 0:
    raise ValueError('MOMENTUM_ETA deve ser finito e nao negativo')
if not math.isfinite(loss['momentum_gain_init']):
    raise ValueError('MOMENTUM_GAIN_INIT deve ser finito')
if 'lradj' in config['training'] and config['training']['lradj'] not in {'type1', 'type2', 'type3', 'constant'}:
    raise ValueError('LRADJ deve ser type1, type2, type3 ou constant')
if momentum_experiment != 'composto' or loss['momentum_mode'] == 'off':
    for key in ('momentum_mode', 'momentum_lookback', 'momentum_eta',
                'momentum_gain_init', 'momentum_diagnostics'):
        loss.pop(key)
else:
    if loss['momentum_gain_init'] == 0.0:
        loss.pop('momentum_gain_init')
    if not loss['momentum_diagnostics']:
        loss.pop('momentum_diagnostics')
print(json.dumps(config, allow_nan=False))
PY
)

