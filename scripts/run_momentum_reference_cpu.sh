#!/usr/bin/env bash
# Benchmark separado; usa o plano e o motor congelado do backtest de referencia.
# Execute no TFB remoto: planejar | submeter. Nenhuma GPU ou nova previsao.
set -euo pipefail

TFB_ROOT="${TFB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/snfs2/igor.viveiros/analise_resultados/backtests}"
REFERENCE_RUN="${REFERENCE_RUN:-$OUTPUT_ROOT/temporal_var_abertura_fechamento_s0_c0_cdi_v3_rent4}"
RUN_ID="${RUN_ID:-momentum_temporal_var_s0_c0_cdi_v1}"
TIME_LIMIT="${TIME_LIMIT:-01:30:00}"
MEMORY="${MEMORY:-16G}"
WORKERS="${WORKERS:-16}"
ACTION="${1:-planejar}"
if [[ $# -gt 1 || ! "$ACTION" =~ ^(planejar|submeter)$ ]]; then
    echo "Use: bash scripts/run_momentum_reference_cpu.sh planejar|submeter" >&2; exit 2
fi
if [[ ! "$RUN_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
    echo "RUN_ID invalido." >&2; exit 2
fi
if [[ ! "$WORKERS" =~ ^([1-9]|1[0-6])$ ]]; then
    echo "WORKERS deve ser 1..16." >&2; exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
ARGS=(--reference-run "$REFERENCE_RUN" --run-dir "$RUN_DIR" --workers "$WORKERS")
# Valida a referencia, sua grade e os insumos; nao le os Parquets de previsao.
"$PYTHON_BIN" -B "$TFB_ROOT/scripts/run_momentum_reference.py" "${ARGS[@]}" --dry-run
if [[ "$ACTION" == planejar ]]; then
    printf 'Plano conferido; nenhum job submetido. Slurm: gorgonas, ate %s processos, nenhuma GPU.\n' "$WORKERS"
    exit
fi
mkdir -p "$RUN_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd -P)"
REFERENCE_RUN="$(cd "$REFERENCE_RUN" && pwd -P)"
if [[ "$RUN_DIR" != /snfs2/* ]]; then
    echo "Toda saida deve ficar no SNFS2: $RUN_DIR" >&2; exit 2
fi
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/tmp" "$RUN_DIR/codigo"
SNAPSHOT="$RUN_DIR/codigo/run_momentum_reference.py"
if [[ -e "$SNAPSHOT" ]]; then
    if ! cmp -s "$TFB_ROOT/scripts/run_momentum_reference.py" "$SNAPSHOT"; then
        echo "Codigo momentum mudou. Use outro RUN_ID; snapshot existente preservado." >&2; exit 2
    fi
else
    cp "$TFB_ROOT/scripts/run_momentum_reference.py" "$SNAPSHOT"
fi
export TMPDIR="$RUN_DIR/tmp" XDG_CACHE_HOME="$RUN_DIR/cache"
WORKER_SCRIPT="$(mktemp "$RUN_DIR/worker_momentum_XXXXXX.sh")"
{
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    printf 'exec srun --cpu-bind=cores '
    printf '%q ' "$PYTHON_BIN" -B "$SNAPSHOT" --reference-run "$REFERENCE_RUN" --run-dir "$RUN_DIR" --workers "$WORKERS"
    printf '\n'
} > "$WORKER_SCRIPT"
SBATCH=(sbatch --parsable --job-name=momentum --partition=gorgonas_dev
    --nodes=1 --ntasks=1 --cpus-per-task="$WORKERS" --hint=nomultithread
    --mem="$MEMORY" --time="$TIME_LIMIT" --chdir="$RUN_DIR"
    --output="$RUN_DIR/logs/%x_%j.out" --error="$RUN_DIR/logs/%x_%j.err"
    --export=ALL "$WORKER_SCRIPT")
JOB_ID="$("${SBATCH[@]}")"
printf 'Job: %s\nExecucao: %s\nLogs: %s/logs\n' "$JOB_ID" "$RUN_DIR" "$RUN_DIR"
