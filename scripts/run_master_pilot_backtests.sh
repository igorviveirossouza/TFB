#!/usr/bin/env bash
# Um unico job CPU: auditoria, ICs e carteiras; nunca treina modelos.
set -euo pipefail

ON_NODE=false
if [[ "${1:-}" == --on-node ]]; then ON_NODE=true; shift; fi
if [[ $# -ne 0 ]]; then
  echo 'Uso: bash scripts/run_master_pilot_backtests.sh' >&2; exit 2
fi

export TFB_ROOT="${TFB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHON_BIN="${PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"
export MASTER_EVAL_PREDICTIONS="${MASTER_EVAL_PREDICTIONS:-/snfs2/igor.viveiros/previsoes/master_piloto_l32_h5_k1_t80_v1}"
export MASTER_EVAL_MARKET="${MASTER_EVAL_MARKET:-/snfs2/igor.viveiros/analise_resultados/insumos/base_backtest_temporal_001}"
export MASTER_EVAL_RETURNS="${MASTER_EVAL_RETURNS:-$TFB_ROOT/dataset/forecasting/b3_daily_tfb_new.csv}"
export MASTER_EVAL_RUN_DIR="${MASTER_EVAL_RUN_DIR:-/snfs2/igor.viveiros/analise_resultados/backtests/master_piloto_l32_h5_k1_t80_s0_c0_cdi_v1}"
export MASTER_EVAL_LOG_DIR="${MASTER_EVAL_LOG_DIR:-$TFB_ROOT/logs/master/avaliacao_piloto_v1}"
export MASTER_EVAL_WORKERS="${MASTER_EVAL_WORKERS:-2}"
if [[ ! "$MASTER_EVAL_WORKERS" =~ ^([1-9]|1[0-6])$ ]]; then
  echo 'MASTER_EVAL_WORKERS deve ser 1..16.' >&2; exit 2
fi

if [[ "$ON_NODE" == false ]]; then
  # Guarda a versao enviada; atualizacoes posteriores do checkout nao mudam o job.
  export MASTER_EVAL_COMMIT
  MASTER_EVAL_COMMIT=$(git -C "$TFB_ROOT" rev-parse HEAD)
  mkdir -p "$MASTER_EVAL_LOG_DIR"
  JOB=$(sbatch --parsable --export=ALL --partition="${MASTER_EVAL_PARTITION:-gorgonas_dev}" \
    --nodes=1 --ntasks=1 --cpus-per-task="$MASTER_EVAL_WORKERS" \
    --mem="${MASTER_EVAL_MEMORY:-16G}" --time="${MASTER_EVAL_TIME:-01:30:00}" \
    --job-name=master_avaliacao --chdir="$TFB_ROOT" \
    --output="$MASTER_EVAL_LOG_DIR/avaliacao_%j.out" \
    --error="$MASTER_EVAL_LOG_DIR/avaliacao_%j.err" \
    "$TFB_ROOT/scripts/run_master_pilot_backtests.sh" --on-node)
  printf 'Job de avaliacao CPU: %s\nLogs: %s/avaliacao_%s.{out,err}\nResultados: %s\n' \
    "$JOB" "$MASTER_EVAL_LOG_DIR" "${JOB%%;*}" "$MASTER_EVAL_RUN_DIR"
  exit
fi

if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname -s)" == phocus4 ]]; then
  echo 'Processamento permitido somente em job Slurm em no de computacao.' >&2; exit 2
fi
if [[ ! "${MASTER_EVAL_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo 'Commit da submissao ausente/invalido.' >&2; exit 2
fi
case "$MASTER_EVAL_RUN_DIR" in /snfs2/*) ;; *)
  echo 'MASTER_EVAL_RUN_DIR deve ficar no SNFS2.' >&2; exit 2;;
esac

# Extracao e todo Python ocorrem no no alocado. O script de submissao e copiado pelo Slurm.
mkdir -p "$MASTER_EVAL_RUN_DIR/fontes"
SOURCE_DIR=$(mktemp -d "$MASTER_EVAL_RUN_DIR/fontes/job_${SLURM_JOB_ID}_XXXXXX")
git -C "$TFB_ROOT" archive "$MASTER_EVAL_COMMIT" scripts ts_benchmark config | tar -x -C "$SOURCE_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
exec "$PYTHON_BIN" -B -u "$SOURCE_DIR/scripts/evaluate_master_pilot.py" \
  --predictions "$MASTER_EVAL_PREDICTIONS" --market "$MASTER_EVAL_MARKET" \
  --returns-file "$MASTER_EVAL_RETURNS" --run-dir "$MASTER_EVAL_RUN_DIR" \
  --workers "$MASTER_EVAL_WORKERS"
