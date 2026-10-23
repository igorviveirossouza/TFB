#!/usr/bin/env bash
# Três treinamentos completos. Este wrapper só configura e submete jobs;
# nenhum Python é executado na máquina de acesso.
set -euo pipefail

case "${1:-}" in
  ''|--dry-run|--prepare) ;;
  *) echo 'Uso: bash scripts/run_master_pilot.sh [--dry-run|--prepare]' >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then
  echo 'Use no máximo uma opção: --dry-run ou --prepare.' >&2
  exit 2
fi

export TFB_ROOT="${TFB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHON_BIN="${PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"
PILOT_TAG="${MASTER_PILOT_TAG:-master_piloto_l32_h5_k1_t80_v1}"
export RUN_TAG="$PILOT_TAG"
# Nomes próprios evitam reutilizar caminhos exportados por outra grade.
export OUT_ROOT="${MASTER_PILOT_OUT_ROOT:-/snfs2/igor.viveiros/previsoes/$PILOT_TAG}"
PILOT_RESULT_ROOT="${MASTER_PILOT_RESULT_ROOT:-$TFB_ROOT/result/master/$PILOT_TAG}"
PILOT_LOG_ROOT="${MASTER_PILOT_LOG_ROOT:-$TFB_ROOT/logs/master/$PILOT_TAG}"

# Desenho fixo: 1 temporal + 2 compostas. Todas as janelas de treino/validação.
export TRAIN_PERCENTAGES=80 LOOKBACKS=32 HORIZONS=5 TRADE_WINDOWS=1
export TEMPORAL_LOSS=mse CROSS_LOSSES='ranknet ranknet_proximity'
export LAMBDAS=0.5 DELTAS=0 CROSS_MASK='gaussian 0.025 natural'
export NUM_EPOCHS=100 PATIENCE=10 BATCH_SIZE=32 LEARNING_RATE=0.0001
export TRAIN_RATIO_IN_TV=0.875 SEED=2026
export D_MODEL=64 T_HEADS=4 S_HEADS=2 DROPOUT=0.1 GATE_BETA=1 LRADJ=type1
export ASSET_BLOCKS='retornos ohlcv atividade' USE_MARKET=true DEVICE=cpu
export MANTER_CSV=FALSE

# A preparação continua em gorgonas_dev pelo launcher existente.
# Treino completo em gorgonas evita o limite de 1h30 da partição dev.
export PARTITION="${MASTER_PILOT_PARTITION:-gorgonas}"
export TIME_LIMIT="${MASTER_PILOT_TIME_LIMIT:-08:00:00}"
export MEMORY="${MASTER_PILOT_MEMORY:-32G}" CPUS_PER_TASK=2
export MAX_CONCURRENT_JOBS=1 ARRAY_CHUNK_SIZE=1000 NODELIST='' GRES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

for mode in temporal composto; do
  export RESULT_ROOT="$PILOT_RESULT_ROOT/$mode"
  export LOG_ROOT="$PILOT_LOG_ROOT/$mode"
  echo "MASTER piloto: submetendo preparação $mode; logs em $LOG_ROOT"
  bash "$TFB_ROOT/scripts/run_master_experiment.sh" "$mode" "$@"
done
