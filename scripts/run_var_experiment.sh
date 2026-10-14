#!/usr/bin/env bash
# Edit config/b3_experiments.json; --dry-run inspects without submitting.
set -euo pipefail
TFB_ROOT="${TFB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$TFB_ROOT/scripts/b3_experiment_env.sh"
exec "$PYTHON_BIN" "$TFB_ROOT/scripts/b3_experiment.py" var "$@"
