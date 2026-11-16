#!/usr/bin/env bash
# Launcher isolado: rank->rank ou log-retorno->rank. Legados nao sao alterados.
set -Eeuo pipefail
TFB_ROOT="${TFB_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export TFB_ROOT
if [[ "${1:-}" == --on-node ]]; then
    shift
    if [[ -n "${VENV_PATH:-}" && -f "$VENV_PATH" ]]; then source "$VENV_PATH"; fi
    cd -- "$TFB_ROOT"
    exec "${PYTHON_BIN:?}" "$TFB_ROOT/scripts/rank_target_experiment.py" "$@"
fi
# Preflight/training needs the configured TFB environment; plans stay usable
# with the standard-library Python on a login node without Torch installed.
for argument in "$@"; do
    case "$argument" in
        --prepare|--train|--local|--worker)
            rank_python="${NS_PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"
            if [[ -n "${NS_VENV_PATH:-}" && -f "$NS_VENV_PATH" ]]; then source "$NS_VENV_PATH"; fi
            exec "$rank_python" "$TFB_ROOT/scripts/rank_target_experiment.py" "$@"
            ;;
    esac
done
exec "${RANK_PLAN_PYTHON:-${ORDINAL_PLAN_PYTHON:-python3}}" "$TFB_ROOT/scripts/rank_target_experiment.py" "$@"
