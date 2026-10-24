#!/usr/bin/env bash
# O host de acesso apenas copia os auxiliares e submete o coordenador Slurm.
set -euo pipefail

usage() {
  echo 'Uso: bash scripts/recover_composite_experiments.sh [audit|resume] [--run-tag TAG]... [--reject-token TOKEN]...' >&2
}

if [[ "${1:-}" == '--on-node' ]]; then
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo '--on-node requer uma alocação Slurm; não execute Python na phocus4.' >&2
    exit 2
  fi
  shift
  if [[ -z "${RECOVERY_TOOL_DIR:-}" || "$RECOVERY_TOOL_DIR" != /* || ! -f "$RECOVERY_TOOL_DIR/b3_recovery.py" ]]; then
    echo 'RECOVERY_TOOL_DIR deve apontar para os auxiliares congelados desta solicitação.' >&2
    exit 2
  fi
  export PYTHONDONTWRITEBYTECODE=1
  exec "${PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}" \
    "$RECOVERY_TOOL_DIR/b3_recovery.py" "$@"
fi

ACTION="${1:-audit}"
case "$ACTION" in
  audit|resume) [[ $# -eq 0 ]] || shift ;;
  *) usage; exit 2 ;;
esac
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  if [[ $# -lt 2 || -z "$2" || "$2" == -* ]]; then
    usage
    exit 2
  fi
  case "$1" in
    --run-tag) ;;
    --reject-token)
      if [[ "$ACTION" != resume || ! "$2" =~ ^[0-9a-f]{32}$ ]]; then usage; exit 2; fi ;;
    *) usage; exit 2 ;;
  esac
  EXTRA_ARGS+=("$1" "$2")
  shift 2
done

REPO="${TFB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="$(cd "$REPO" && pwd)"
RECOVERY_ROOT="${RECOVERY_ROOT:-/snfs2/igor.viveiros/retomadas_tfb}"
OUT_BASE="${RECOVERY_OUT_BASE:-/snfs2/igor.viveiros/previsoes}"
RESULT_BASE="${RECOVERY_RESULT_BASE:-/snfs2/igor.viveiros/result}"
export PYTHON_BIN="${PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"

# Recursos próprios: não herdar PARTITION, NODELIST ou GRES de uma grade GPU.
export RECOVERY_WORKER_PARTITION="${RECOVERY_WORKER_PARTITION:-gorgonas}"
export RECOVERY_WORKER_TIME="${RECOVERY_WORKER_TIME:-01:30:00}"
export RECOVERY_WORKER_CPUS="${RECOVERY_WORKER_CPUS:-3}"
export RECOVERY_WORKER_MEMORY="${RECOVERY_WORKER_MEMORY-}"
export RECOVERY_WORKER_GRES="${RECOVERY_WORKER_GRES-}"
export RECOVERY_WORKER_NODELIST="${RECOVERY_WORKER_NODELIST-}"
export RECOVERY_MAX_CONCURRENT="${RECOVERY_MAX_CONCURRENT:-2}"

for name in b3_recovery.py b3_recovery_audit.py b3_recovery_snapshot.py recover_composite_experiments.sh; do
  if [[ ! -f "$REPO/scripts/$name" ]]; then
    echo "Auxiliar ausente: $REPO/scripts/$name; nenhuma submissão realizada." >&2
    exit 2
  fi
done
command -v sbatch >/dev/null || { echo 'sbatch não encontrado.' >&2; exit 2; }

mkdir -p "$RECOVERY_ROOT"
STATE_DIR="$(mktemp -d "$RECOVERY_ROOT/solicitacao_XXXXXXXX")"
STATE_DIR="$(cd "$STATE_DIR" && pwd)"
mkdir -p "$STATE_DIR/tools" "$STATE_DIR/logs"
export RECOVERY_TOOL_DIR="$STATE_DIR/tools"
for name in b3_recovery.py b3_recovery_audit.py b3_recovery_snapshot.py recover_composite_experiments.sh; do
  cp -- "$REPO/scripts/$name" "$STATE_DIR/tools/$name"
done
chmod a-w "$STATE_DIR/tools/"*

# Variáveis SBATCH_* têm precedência sobre diretivas e podem trazer GRES/nó
# antigos. Os recursos deste coordenador são definidos somente abaixo.
for variable in "${!SBATCH_@}"; do
  unset "$variable"
done

echo "Retomada: ação=$ACTION; diretório=$STATE_DIR"
echo 'A auditoria será executada em uma alocação Slurm, sem Python no host de acesso.'
JOB_ID="$(sbatch --parsable --export=ALL --no-requeue \
  --partition="${RECOVERY_AUDIT_PARTITION:-gorgonas_dev}" \
  --time="${RECOVERY_AUDIT_TIME:-01:30:00}" \
  --nodes=1 --ntasks=1 \
  --cpus-per-task="${RECOVERY_AUDIT_CPUS:-3}" \
  --mem="${RECOVERY_AUDIT_MEMORY:-8G}" \
  --job-name="b3_recovery_$ACTION" \
  --chdir="$STATE_DIR" \
  --output="$STATE_DIR/logs/coordenador_%j.out" \
  --error="$STATE_DIR/logs/coordenador_%j.err" \
  "$STATE_DIR/tools/recover_composite_experiments.sh" --on-node "$ACTION" \
  --repo "$REPO" --state-dir "$STATE_DIR" \
  --out-base "$OUT_BASE" --result-base "$RESULT_BASE" "${EXTRA_ARGS[@]}")"
echo "Job coordenador: $JOB_ID"
echo "Logs: $STATE_DIR/logs/coordenador_${JOB_ID%%;*}.{out,err}"
echo "Relatórios e planos: $STATE_DIR"
