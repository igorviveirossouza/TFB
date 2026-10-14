#!/usr/bin/env bash
# Shared environment. All experimental parameters are in the JSON config.
set -euo pipefail
export TFB_ROOT="${TFB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PATH="${VENV_PATH:-/sonic_home/igor.viveiros/py310/bin/activate}"
if [[ -f "$VENV_PATH" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH"
fi
export PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$TFB_ROOT"
