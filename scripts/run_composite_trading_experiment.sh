#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point.
# The auditable implementation is now in run_composite_trading_experiment_v2.sh,
# which supports retornos_simples, log_retornos and optional prices.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_composite_trading_experiment_v2.sh" "$@"
