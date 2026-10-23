#!/usr/bin/env bash
# MASTER usa o desenho/exportador B3 existente. Na máquina de acesso este
# launcher executa somente sbatch; até a preparação ocorre em gorgonas_dev.
set -euo pipefail

ON_NODE=false
if [[ "${1:-}" == --on-node ]]; then ON_NODE=true; shift; fi
MODE="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in smoke|temporal|composto) ;; *)
  echo 'Uso: bash scripts/run_master_experiment.sh [smoke|temporal|composto] [--dry-run|--prepare|--task N]' >&2
  exit 2;;
esac

export TFB_ROOT="${TFB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHON_BIN="${PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"
export MASTER_BUNDLE_DIR="${MASTER_BUNDLE_DIR:-$TFB_ROOT/dataset/master/features_master_v1}"
export DATASET_PATH="${DATASET_PATH:-$TFB_ROOT/dataset/forecasting/b3_daily_tfb_new.csv}"
export RUN_TAG="${RUN_TAG:-master_${MODE}_v1}"
export OUT_ROOT="${OUT_ROOT:-/snfs2/igor.viveiros/previsoes/$RUN_TAG}"
export RESULT_ROOT="${RESULT_ROOT:-$TFB_ROOT/result/master/$RUN_TAG}"
export LOG_ROOT="${LOG_ROOT:-$TFB_ROOT/logs/master/$RUN_TAG}"
export B3_WORKER_LAUNCHER="$TFB_ROOT/scripts/run_master_experiment.sh"

if [[ "$ON_NODE" == false ]]; then
  mkdir -p "$LOG_ROOT"
  exec sbatch --parsable --export=ALL --partition=gorgonas_dev --time=01:30:00 \
    --cpus-per-task=2 --mem=8G --job-name="master_${MODE}_prepare" \
    --output="$LOG_ROOT/prepare_%j.out" --error="$LOG_ROOT/prepare_%j.err" \
    "$B3_WORKER_LAUNCHER" --on-node "$MODE" "$@"
fi
if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname -s)" == phocus4 ]]; then
  echo 'MASTER: processamento permitido somente dentro de job Slurm em nó de computação.' >&2
  exit 2
fi
cd "$TFB_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

if [[ "$MODE" == smoke ]]; then
  exec "$PYTHON_BIN" "$TFB_ROOT/scripts/smoke_test_master.py" \
    --bundle-dir "$MASTER_BUNDLE_DIR" --returns-path "$DATASET_PATH" \
    --output "$TFB_ROOT/result/master/smoke_${SLURM_JOB_ID}.json" "$@"
fi
if [[ "${1:-}" == worker ]]; then
  : "${B3_INLINE_CONFIG:?Configuração congelada ausente no worker}"
  exec "$PYTHON_BIN" "$TFB_ROOT/scripts/b3_experiment.py" "$MODE" "$@"
fi

# Grade: cada H gera um treinamento com saída conjunta [batch,H,ativos].
export TRAIN_PERCENTAGES="${TRAIN_PERCENTAGES:-80}"
export LOOKBACKS="${LOOKBACKS:-32 128 256}"
export HORIZONS="${HORIZONS:-1 5 10 15 20 25}"
export TRADE_WINDOWS="${TRADE_WINDOWS:-1}"
# Todas as losses cross disponíveis no composite v3. Uma máscara por grade.
export CROSS_LOSSES="${CROSS_LOSSES:-mse bpr ranknet hinge listnet pairwise_mse hinge_proximity ranknet_proximity pairwise_mse_proximity}"
export CROSS_MASK="${CROSS_MASK:-gaussian 0.025 natural}"
export LAMBDAS="${LAMBDAS:-0.1 0.5 0.8}"
export DELTAS="${DELTAS:-0}"
export TEMPORAL_LOSS="${TEMPORAL_LOSS:-mse}"  # mse, mae ou huber
export NUM_EPOCHS="${NUM_EPOCHS:-100}" BATCH_SIZE="${BATCH_SIZE:-32}"
export PATIENCE="${PATIENCE:-10}" LEARNING_RATE="${LEARNING_RATE:-0.0001}"
export TRAIN_RATIO_IN_TV="${TRAIN_RATIO_IN_TV:-0.875}" SEED="${SEED:-2026}"
export D_MODEL="${D_MODEL:-64}" T_HEADS="${T_HEADS:-4}" S_HEADS="${S_HEADS:-2}"
export DROPOUT="${DROPOUT:-0.1}" GATE_BETA="${GATE_BETA:-1}"
export ASSET_BLOCKS="${ASSET_BLOCKS:-retornos ohlcv atividade}" USE_MARKET="${USE_MARKET:-true}"
export DEVICE="${DEVICE:-auto}" LRADJ="${LRADJ:-type1}"
# Recursos dos workers. O job de preparação permanece sempre em gorgonas_dev.
export GPU_PARTITION="${PARTITION:-gorgonas_dev}" GPU_TIME="${TIME_LIMIT:-01:30:00}"
export GRES="${GRES-}" CPUS_PER_TASK="${CPUS_PER_TASK:-2}" MEMORY="${MEMORY:-32G}"
export NODELIST="${NODELIST:-}" MAX_GPU_JOBS="${MAX_CONCURRENT_JOBS:-2}"
export ARRAY_CHUNK_SIZE="${ARRAY_CHUNK_SIZE:-1000}" MANTER_CSV="${MANTER_CSV:-FALSE}"
export B3_CONFIG_SOURCE="$B3_WORKER_LAUNCHER"
export B3_INLINE_CONFIG
B3_INLINE_CONFIG=$("$PYTHON_BIN" - <<'PY'
import json
import os
e = os.environ
def vector(key, cast=str):
    return [cast(x) for x in e[key].split()]
def boolean(key):
    if e[key].lower() not in {'true', 'false'}:
        raise ValueError(key + ' deve ser true ou false')
    return e[key].lower() == 'true'
if e['TEMPORAL_LOSS'] not in {'mse', 'mae', 'huber'}:
    raise ValueError('TEMPORAL_LOSS deve ser mse, mae ou huber')
mask = vector('CROSS_MASK')
if len(mask) not in (2, 3):
    raise ValueError('CROSS_MASK deve conter tipo a [modo]')
config = dict(
    source_data_kind='log_return', models=['MASTER'], var_lags=32,
    train_percentages=vector('TRAIN_PERCENTAGES', int),
    lookbacks=vector('LOOKBACKS', int), horizons=vector('HORIZONS', int),
    trade_windows=vector('TRADE_WINDOWS', int), cross_losses=vector('CROSS_LOSSES'),
    lambdas=vector('LAMBDAS', float), deltas=vector('DELTAS', float),
    model_defaults={'MASTER': dict(
        master_bundle_dir=e['MASTER_BUNDLE_DIR'], master_returns_path=e['DATASET_PATH'],
        master_expected_assets=55, master_expected_dates=3664,
        master_asset_blocks=vector('ASSET_BLOCKS'), master_use_market=boolean('USE_MARKET'),
        batch_size=int(e['BATCH_SIZE']), d_model=int(e['D_MODEL']),
        t_heads=int(e['T_HEADS']), s_heads=int(e['S_HEADS']),
        dropout=float(e['DROPOUT']), gate_beta=float(e['GATE_BETA']),
        norm=True, lradj=e['LRADJ'], device=e['DEVICE'], seed=int(e['SEED']))},
    training=dict(num_epochs=int(e['NUM_EPOCHS']), patience=int(e['PATIENCE']), lr=float(e['LEARNING_RATE'])),
    loss=dict(loss_temporal=e['TEMPORAL_LOSS'], loss_data_kind='log_return',
        loss_score_kind='simple_return', loss_cross_score_normalization='zscore',
        loss_cross_scale=1.0, loss_ranknet_alpha=1.0, loss_listnet_tau=1.0,
        loss_hinge_margin=0.1, loss_inverse_norm=True, loss_track_components=True,
        loss_cross_mask=[mask[0], float(mask[1]), *mask[2:]]),
    strategy=dict(train_ratio_in_tv=float(e['TRAIN_RATIO_IN_TV']), stride=1,
        num_rollings=48000, seed=int(e['SEED'])))
print(json.dumps(config, allow_nan=False))
PY
)
exec "$PYTHON_BIN" "$TFB_ROOT/scripts/b3_experiment.py" "$MODE" "$@"
