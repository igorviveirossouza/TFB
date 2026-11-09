#!/usr/bin/env bash
# Shared output versus temporal/ranking heads, with identical supervised windows.
set -Eeuo pipefail
TFB_ROOT="${TFB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NS_PYTHON_BIN="${NS_PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"
exec "$NS_PYTHON_BIN" "$TFB_ROOT/scripts/ns_capacity_heads.py" --study heads "$@"
