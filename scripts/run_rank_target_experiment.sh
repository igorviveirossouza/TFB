#!/usr/bin/env bash
# Launcher isolado: rank->rank ou log-retorno->rank. Legados nao sao alterados.
set -Eeuo pipefail
TFB_ROOT="${TFB_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export TFB_ROOT
# Only Slurm dispatch is allowed on the login node. Even plans and preflight
# (CSV reads, model construction and hashes) must execute on a compute node.
# SLURM_JOB_ID alone is not enough: salloc may leave its shell on the login node.
on_compute_node() {
    [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURMD_NODENAME:-}" ]] && [[ "$(hostname -s)" != phocus4 ]]
}
if [[ "${1:-}" == --compute-step || "${1:-}" == --on-node ]]; then
    if ! on_compute_node; then
        echo 'Rank target: processamento permitido somente em no de computacao Slurm.' >&2
        exit 2
    fi
    if [[ "$1" == --compute-step ]]; then shift; fi
fi
if ! on_compute_node; then
    if ! command -v srun >/dev/null 2>&1; then
        echo 'Rank target: srun nao encontrado; nenhum processamento sera executado localmente.' >&2
        exit 2
    fi
    dispatch=(srun --export=ALL --ntasks=1 --chdir="$TFB_ROOT")
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        # An existing salloc allocation needs a remote step, not another job.
        dispatch+=(--jobid="$SLURM_JOB_ID")
    else
        dispatch+=(--partition="${NS_PARTITION:-gorgonas}"
            --time="${NS_TIME_LIMIT:-01:30:00}"
            --cpus-per-task="${NS_CPUS_PER_TASK:-3}"
            --mem="${NS_MEMORY:-32G}" --job-name=rank_dispatch)
        if [[ -n "${NS_NODELIST:-}" ]]; then dispatch+=(--nodelist="$NS_NODELIST"); fi
        # Preparation/plans only need CPU. Training arrays request their own
        # NS_GRES in Python; --local/--worker train inside this allocation.
        dispatch_gres=none
        for argument in "$@"; do
            case "$argument" in
                --local|--worker|--worker=*) dispatch_gres="${NS_GRES-gpu:1}" ;;
            esac
        done
        dispatch+=(--gres="${dispatch_gres:-none}")
    fi
    echo 'Rank target: aguardando no de computacao; processamento e saida via srun.' >&2
    exec "${dispatch[@]}" bash "$TFB_ROOT/scripts/run_rank_target_experiment.sh" --compute-step "$@"
fi
if [[ "${1:-}" == --on-node ]]; then
    shift
    if [[ -n "${VENV_PATH:-}" && -f "$VENV_PATH" ]]; then source "$VENV_PATH"; fi
    cd -- "$TFB_ROOT"
    exec "${PYTHON_BIN:?}" "$TFB_ROOT/scripts/rank_target_experiment.py" "$@"
fi
# Preflight/training needs the configured TFB environment. Plans use the
# standard-library Python, also exclusively on the compute node.
for argument in "$@"; do
    case "$argument" in
        --prepare|--train|--local|--worker|--worker=*)
            rank_python="${NS_PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"
            if [[ -n "${NS_VENV_PATH:-}" && -f "$NS_VENV_PATH" ]]; then source "$NS_VENV_PATH"; fi
            exec "$rank_python" "$TFB_ROOT/scripts/rank_target_experiment.py" "$@"
            ;;
    esac
done
exec "${RANK_PLAN_PYTHON:-${ORDINAL_PLAN_PYTHON:-python3}}" "$TFB_ROOT/scripts/rank_target_experiment.py" "$@"
