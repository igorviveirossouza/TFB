#!/usr/bin/env bash
# Experimento ordinal isolado. Todos os valores publicos aceitam export.
set -Eeuo pipefail
TFB_ROOT="${TFB_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export TFB_ROOT
if [[ "${1:-}" == --on-node ]]; then
    [[ $# -ge 3 && ( "$2" == temporal || "$2" == composto ) && "$3" == worker ]] || exit 2
    mode=$2; shift 2
    [[ -n "${B3_INLINE_CONFIG:-}" && -n "${B3_CONFIG_SOURCE:-}" ]] || exit 2
    if [[ -n "${VENV_PATH:-}" && -f "$VENV_PATH" ]]; then source "$VENV_PATH"; fi
    cd -- "$TFB_ROOT"
    exec "${PYTHON_BIN:?}" "$TFB_ROOT/scripts/b3_experiment.py" "$mode" "$@"
fi
# O plano e a previa so precisam da biblioteca padrao do Python local.
exec "${ORDINAL_PLAN_PYTHON:-python3}" "$TFB_ROOT/scripts/ordinal_experiment.py" "$@"
