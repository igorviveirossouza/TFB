#!/usr/bin/env bash
# Composto: execucao independente do temporal/VAR, apenas CPU.
# Use: bash scripts/run_backtests_composto_cpu.sh planejar|piloto|submeter|consolidar
set -euo pipefail

TFB_ROOT="${TFB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-/snfs2/igor.viveiros/previsoes/novas_previsoes}"
MARKET_DIR="${MARKET_DIR:-/snfs2/igor.viveiros/analise_resultados/insumos/base_backtest_temporal_001}"
RETURNS_FILE="${RETURNS_FILE:-$TFB_ROOT/dataset/forecasting/b3_daily_tfb_new.csv}"
STRATEGIES_FILE="${STRATEGIES_FILE:-$TFB_ROOT/config/estrategias_backtest.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/snfs2/igor.viveiros/analise_resultados/backtests}"
# Nao herda RUN_ID: uma sessao usada no temporal nao pode selecionar essa execucao.
COMPOSITE_RUN_ID="${COMPOSITE_RUN_ID:-composto_abertura_fechamento_s0_c0_cdi_v1}"
TRAIN_EXPLICIT="${TRAIN_VALUES+x}"
read -r -a TRAIN <<< "${TRAIN_VALUES:-80}"
read -r -a K_EVAL <<< "${K_EVAL_VALUES:-1 5 10 15 20 25}"
SLIPPAGE_BPS="${SLIPPAGE_BPS:-0}"
COST_BPS="${COST_BPS:-0}"

WORKERS="${WORKERS:-16}"
N_LOTES="${N_LOTES:-1}"  # Ate tres nos; WORKERS processos independentes por no.
MEMORY="${MEMORY:-56G}"
# Destino fixo do backtest; nao herdar particao/tempo de sessoes de treinamento.
TIME_LIMIT="01:30:00"
PARTITION="gorgonas_dev"

ACTION="${1:-planejar}"
if [[ $# -gt 1 || ! "$ACTION" =~ ^(planejar|piloto|submeter|consolidar)$ ]]; then
    echo "Use: bash scripts/run_backtests_composto_cpu.sh planejar|piloto|submeter|consolidar" >&2; exit 2
fi
if [[ ! "$COMPOSITE_RUN_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ || ! "$WORKERS" =~ ^([1-9]|1[0-6])$ || ! "$N_LOTES" =~ ^[1-3]$ ]]; then
    echo "COMPOSITE_RUN_ID invalido; WORKERS deve ser 1..16 e N_LOTES 1..3." >&2; exit 2
fi
if [[ ${#TRAIN[@]} -eq 0 || ${#K_EVAL[@]} -eq 0 ]]; then
    echo "TRAIN_VALUES e K_EVAL_VALUES devem conter inteiros separados por espacos." >&2; exit 2
fi
for value in "${TRAIN[@]}" "${K_EVAL[@]}"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "TRAIN_VALUES e K_EVAL_VALUES devem conter inteiros positivos: $value" >&2; exit 2
    fi
done
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
RUN_DIR="$OUTPUT_ROOT/$COMPOSITE_RUN_ID"
PREP=("$PYTHON_BIN" -B "$TFB_ROOT/scripts/run_backtests_composto.py" prepare
    --predictions "$PREDICTIONS_ROOT" --market "$MARKET_DIR" --returns-file "$RETURNS_FILE"
    --strategies "$STRATEGIES_FILE" --run-dir "$RUN_DIR" --train "${TRAIN[@]}" --k "${K_EVAL[@]}"
    --slippage-bps "$SLIPPAGE_BPS" --cost-bps "$COST_BPS")
if [[ -f "$RUN_DIR/contrato.json" ]]; then
    # O ID seleciona o contrato e seu codigo congelado, inclusive versoes antigas.
    # Nao aplicar defaults/variaveis de outra execucao sobre esse contrato.
    PREP=("$PYTHON_BIN" -B "$TFB_ROOT/scripts/run_backtests_composto.py" prepare --run-dir "$RUN_DIR")
    if [[ -n "$TRAIN_EXPLICIT" ]]; then
        PREP+=(--train "${TRAIN[@]}")
    fi
    echo "Contrato existente: origens, estrategias, Ks e custos serao recuperados de $RUN_DIR/contrato.json."
fi
if [[ "$ACTION" == planejar ]]; then
    "${PREP[@]}" --dry-run
    printf 'Slurm: particao=%s, CPUs/processos=%s por no, memoria=%s, lotes=%s; nenhuma GPU.\n' "$PARTITION" "$WORKERS" "$MEMORY" "$N_LOTES"
    exit
fi
if [[ "$ACTION" == consolidar ]]; then
    if [[ ! -f "$RUN_DIR/plano.json" ]]; then
        echo "Execucao inexistente: $RUN_DIR/plano.json. Use o COMPOSITE_RUN_ID original." >&2; exit 2
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

if [[ "$ACTION" == consolidar ]]; then
    WORKERS=1
    N_LOTES=1
    RUN_COMMAND=("$PYTHON_BIN" -B "$TFB_ROOT/scripts/run_backtests_composto.py" consolidate --run-dir "$RUN_DIR")
else
    # Captura uma revisao imutavel: novas previsoes nao mudam jobs na fila.
    PLAN_FILE="$("$PYTHON_BIN" -B - "$RUN_DIR" <<'PY'
import json
from pathlib import Path
import re
import sys

run = Path(sys.argv[1]).resolve()
with (run / "plano.json").open() as stream:
    signature = json.load(stream)["signature"]
if not isinstance(signature, str) or re.fullmatch(r"[0-9a-f]{64}", signature) is None:
    raise ValueError("Assinatura invalida em plano.json.")
plan_file = run / "planos" / f"{signature}.json"
with plan_file.open() as stream:
    if json.load(stream).get("signature") != signature:
        raise ValueError("A revisao imutavel diverge de plano.json.")
print(plan_file)
PY
    )"
    if [[ ! -f "$RUN_DIR/codigo/run_backtests_composto.py" ]]; then
        echo "Codigo congelado ausente: $RUN_DIR/codigo/run_backtests_composto.py" >&2; exit 2
    fi
    RUN_COMMAND=("$PYTHON_BIN" -B "$RUN_DIR/codigo/run_backtests_composto.py" run
        --run-dir "$RUN_DIR" --plan-file "$PLAN_FILE" --workers "$WORKERS")
    if [[ "$ACTION" == piloto ]]; then
        N_LOTES=1
        RUN_COMMAND+=(--pilot)
    fi
    RUN_COMMAND+=(--shards "$N_LOTES")
fi

# Script proprio por submissao; caminhos absolutos, sem source em /var/spool.
WORKER_SCRIPT="$(mktemp "$RUN_DIR/worker_${ACTION}_XXXXXX.sh")"
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
SBATCH=(sbatch --parsable --job-name="ic_composto_${ACTION}" --partition="$PARTITION"
    --nodes=1 --ntasks=1 --cpus-per-task="$WORKERS" --hint=nomultithread --exclusive
    --mem="$MEMORY" --time="$TIME_LIMIT" --array="0-$((N_LOTES-1))%$N_LOTES"
    --chdir="$RUN_DIR" --output="$RUN_DIR/logs/%x_%A_%a.out" --error="$RUN_DIR/logs/%x_%A_%a.err"
    --export=ALL "$WORKER_SCRIPT")
JOB_ID="$("${SBATCH[@]}")"
printf 'Job: %s\nExecucao: %s\nLogs: %s/logs\n' "$JOB_ID" "$RUN_DIR" "$RUN_DIR"
