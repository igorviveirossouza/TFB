#!/usr/bin/env bash
# Execute no TFB, branch experiment/b3-loss-clean-v1 (servidor remoto).
# Edite caminhos e recursos aqui; regras de compra/venda em config/estrategias_backtest.json.
# Use bash scripts/run_backtests_cpu.sh planejar | piloto | submeter | consolidar
set -euo pipefail

TFB_ROOT="${TFB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-/snfs2/igor.viveiros/previsoes/novas_previsoes}"
MARKET_DIR="${MARKET_DIR:-/snfs2/igor.viveiros/analise_resultados/insumos/base_backtest_temporal_001}"
RETURNS_FILE="${RETURNS_FILE:-$TFB_ROOT/dataset/forecasting/b3_daily_tfb_new.csv}"
# CDI obrigatorio, incorporado no calendario_tfb.csv pelo R local.
STRATEGIES_FILE="${STRATEGIES_FILE:-$TFB_ROOT/config/estrategias_backtest.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/snfs2/igor.viveiros/analise_resultados/backtests}"
RUN_ID="${RUN_ID:-temporal_var_abertura_fechamento_s0_c0_cdi_v3}"
TRAIN=(70 80 90)
K_EVAL=(1 5 10 15 20 25)
SLIPPAGE_BPS="${SLIPPAGE_BPS:-0}"  # Por ponta: 5 bps = 0,05%; compra pior e venda pior.
COST_BPS="${COST_BPS:-0}"          # Por ponta, proporcional ao financeiro efetivo.

# Gorgonas tem 16 nucleos fisicos por no e alocacao exclusiva.
WORKERS="${WORKERS:-16}"
N_LOTES="${N_LOTES:-1}"             # 1 inicialmente; ate 3 nos, cada qual com WORKERS processos.
MEMORY="${MEMORY:-56G}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
PARTITION="gorgonas"                # Producao: nunca gorgonas_dev.

ACTION="${1:-planejar}"
if [[ $# -gt 1 || ! "$ACTION" =~ ^(planejar|piloto|submeter|consolidar)$ ]]; then
    echo "Use: bash scripts/run_backtests_cpu.sh planejar|piloto|submeter|consolidar" >&2; exit 2
fi
if [[ ! "$RUN_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ || ! "$WORKERS" =~ ^([1-9]|1[0-6])$ || ! "$N_LOTES" =~ ^[1-3]$ ]]; then
    echo "RUN_ID invalido; WORKERS deve ser 1..16 e N_LOTES 1..3." >&2; exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
PREP=("$PYTHON_BIN" -B "$TFB_ROOT/scripts/run_backtests.py" prepare
    --predictions "$PREDICTIONS_ROOT" --market "$MARKET_DIR" --returns-file "$RETURNS_FILE"
    --strategies "$STRATEGIES_FILE" --run-dir "$RUN_DIR" --train "${TRAIN[@]}" --k "${K_EVAL[@]}"
    --slippage-bps "$SLIPPAGE_BPS" --cost-bps "$COST_BPS")
if [[ "$ACTION" == planejar ]]; then
    "${PREP[@]}" --dry-run
    printf 'Slurm: particao=%s, CPUs/processos=%s por no, memoria=%s, lotes=%s; nenhuma GPU.\n' "$PARTITION" "$WORKERS" "$MEMORY" "$N_LOTES"
    exit
fi
if [[ "$ACTION" == consolidar ]]; then
    # Usa o plano existente e os Parquets prontos; nao prepara nem altera a assinatura.
    if [[ ! -f "$RUN_DIR/plano.json" ]]; then
        echo "Execucao inexistente: $RUN_DIR/plano.json. Use o RUN_ID original." >&2; exit 2
    fi
else
    "${PREP[@]}"
fi
RUN_DIR="$(cd "$RUN_DIR" && pwd -P)"
if [[ "$RUN_DIR" != /snfs2/* ]]; then
    echo "Toda saida deve ficar no SNFS2: $RUN_DIR" >&2; exit 2
fi
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/tmp"
export TMPDIR="$RUN_DIR/tmp" XDG_CACHE_HOME="$RUN_DIR/cache" MPLCONFIGDIR="$RUN_DIR/cache/matplotlib"
export PYTHONPYCACHEPREFIX="$RUN_DIR/cache/pycache"

# Cada submissao recebe seu proprio script, com caminhos absolutos congelados.
# Nenhum source/dirname aponta para /var/spool/slurmd.
WORKER_SCRIPT="$(mktemp "$RUN_DIR/worker_${ACTION}_XXXXXX.sh")"
if [[ "$ACTION" == consolidar ]]; then
    WORKERS=1
    N_LOTES=1
    # Usa o exportador atualizado; o snapshot do calculo original e apenas conferido.
    RUN_COMMAND=("$PYTHON_BIN" -B "$TFB_ROOT/scripts/run_backtests.py" consolidate --run-dir "$RUN_DIR")
else
    RUN_COMMAND=("$PYTHON_BIN" -B "$RUN_DIR/codigo/run_backtests.py" run --run-dir "$RUN_DIR" --workers "$WORKERS")
    if [[ "$ACTION" == piloto ]]; then
        N_LOTES=1
        RUN_COMMAND+=(--pilot)
    fi
    RUN_COMMAND+=(--shards "$N_LOTES")
fi
{
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    printf 'exec srun --cpu-bind=cores '
    printf '%q ' "${RUN_COMMAND[@]}"
    if [[ "$ACTION" == consolidar ]]; then
        printf '\n'
    else
        printf '%s\n' '--shard "${SLURM_ARRAY_TASK_ID:-0}"'
    fi
} > "$WORKER_SCRIPT"
SBATCH=(sbatch --parsable --job-name="ic_bt_${ACTION}" --partition="$PARTITION"
    --nodes=1 --ntasks=1 --cpus-per-task="$WORKERS" --hint=nomultithread --exclusive
    --mem="$MEMORY" --time="$TIME_LIMIT" --array="0-$((N_LOTES-1))%$N_LOTES"
    --chdir="$RUN_DIR" --output="$RUN_DIR/logs/%x_%A_%a.out" --error="$RUN_DIR/logs/%x_%A_%a.err"
    --export=ALL "$WORKER_SCRIPT")
JOB_ID="$("${SBATCH[@]}")"
printf 'Job: %s\nExecucao: %s\nLogs: %s/logs\n' "$JOB_ID" "$RUN_DIR" "$RUN_DIR"
