#!/bin/bash
#SBATCH -p medusas_shr
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=b3-v2-check
#SBATCH --output=/sonic_home/igor.viveiros/src/TFB/logs/b3-v2-check-%j.out
#SBATCH --error=/sonic_home/igor.viveiros/src/TFB/logs/b3-v2-check-%j.err

set -euo pipefail

TFB_ROOT="${TFB_ROOT:-/sonic_home/igor.viveiros/src/TFB}"
CLEAN_ROOT="${CLEAN_ROOT:-/sonic_home/igor.viveiros/clean}"
PYTHON_BIN="${PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"
EXPERIMENT_ID="${EXPERIMENT_ID:?EXPERIMENT_ID não definido}"

mkdir -p "$TFB_ROOT/logs" "$TFB_ROOT/manifests/custom_losses_v2_pipeline/$EXPERIMENT_ID"
cd "$TFB_ROOT"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$BRANCH" != "speed" ]; then
  echo "Branch incorreta no cluster: $BRANCH. Esperado: speed." >&2
  exit 10
fi

required_files=(
  "$CLEAN_ROOT/utils/convert_tfb_custom_loss_results.py"
  "$CLEAN_ROOT/utils/run_tfb_custom_loss_backtests.py"
  "$CLEAN_ROOT/estrategias/ranking_backtest.py"
  "$TFB_ROOT/dataset/forecasting/b3_log_returns.csv"
  "$TFB_ROOT/dataset/forecasting/b3_daily_tfb.csv"
)
for path in "${required_files[@]}"; do
  if [ ! -f "$path" ]; then
    echo "Arquivo obrigatório ausente: $path" >&2
    exit 11
  fi
done

if [ ! -f "$TFB_ROOT/dataset/forecasting/b3_returns.csv" ] && \
   [ ! -f "$TFB_ROOT/dataset/forecasting/b3_daily_return.csv" ]; then
  echo "Base de retornos simples não encontrada." >&2
  exit 12
fi

"$PYTHON_BIN" scripts/verify_custom_losses_v2.py
"$PYTHON_BIN" -m py_compile \
  scripts/run_benchmark_custom_losses_v2.py \
  scripts/convert_tfb_custom_losses_v2.py \
  scripts/validate_tfb_custom_losses_v2_conversion.py \
  scripts/finalize_tfb_custom_losses_v2.py \
  ts_benchmark/baselines/custom_losses_v2.py

for script in \
  scripts/slurm_b3_custom_losses_v2_grid_gpu.sh \
  scripts/slurm_b3_custom_losses_v2_convert_cpu.sh \
  scripts/slurm_b3_custom_losses_v2_validate_conversion_cpu.sh \
  scripts/slurm_b3_custom_losses_v2_backtest_cpu.sh \
  scripts/slurm_b3_custom_losses_v2_finalize_cpu.sh \
  scripts/submit_b3_custom_losses_v2_pipeline.sh; do
  bash -n "$script"
done

cat > "$TFB_ROOT/manifests/custom_losses_v2_pipeline/$EXPERIMENT_ID/verification.json" <<EOF
{
  "experiment_id": "$EXPERIMENT_ID",
  "branch": "$BRANCH",
  "git_commit": "$(git rev-parse HEAD)",
  "hostname": "$(hostname)",
  "status": "ok"
}
EOF

echo "Validação do pipeline v2 concluída no cluster."
