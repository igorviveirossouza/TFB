#!/usr/bin/env bash
# Tres condicoes controladas para diagnosticar a aprendizagem da correcao momentum.
# DIAG_* isola a grade de exports cientificos deixados por experimentos anteriores.
set -Eeuo pipefail

TFB_ROOT="${TFB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-momentum/diagnostico_ns_cauchy_a0.025_t80_v1}"
DIAG_MODEL="${DIAG_MODEL:-Nonstationary_Transformer}"
DIAG_TRAIN_VALUES="${DIAG_TRAIN_VALUES:-80}"
DIAG_LOOKBACKS_VALUES="${DIAG_LOOKBACKS_VALUES:-256}"
DIAG_HORIZON_VALUES="${DIAG_HORIZON_VALUES:-25}"
DIAG_K_LOSS_VALUES="${DIAG_K_LOSS_VALUES:-1}"
DIAG_LAMBDAS="${DIAG_LAMBDAS:-1}"
DIAG_CROSS_LOSS="${DIAG_CROSS_LOSS:-ranknet_proximity}"
DIAG_KERNEL_SPEC="${DIAG_KERNEL_SPEC:-cauchy}"
DIAG_MASK_A="${DIAG_MASK_A:-0.025}"
DIAG_MASK_MODE="${DIAG_MASK_MODE:-natural}"
DIAG_MOMENTUM_LOOKBACK="${DIAG_MOMENTUM_LOOKBACK:-512}"
DIAG_MOMENTUM_ETA="${DIAG_MOMENTUM_ETA:-0.01}"
DIAG_GAIN_INIT="${DIAG_GAIN_INIT:-0.1}"
DIAG_CONTROL_GAIN_INIT="${DIAG_CONTROL_GAIN_INIT:-0}"
DIAG_BASE_LRADJ="${DIAG_BASE_LRADJ:-type1}"
DIAG_ALT_LRADJ="${DIAG_ALT_LRADJ:-constant}"
DIAG_CONDITIONS="${DIAG_CONDITIONS:-controle ganho_inicial ganho_lr_constante}"
DIAG_PARALLEL="${DIAG_PARALLEL:-0}"
NS_AFTER_JOB="${NS_AFTER_JOB:-}"
NS_SEED="${NS_SEED:-2026}"
NS_LEARNING_RATE="${NS_LEARNING_RATE:-0.0001}"
NS_BATCH_SIZE="${NS_BATCH_SIZE:-16}"
NS_NUM_EPOCHS="${NS_NUM_EPOCHS:-100}"
NS_PATIENCE="${NS_PATIENCE:-10}"

# These operational knobs keep the same names as the existing launchers.
export NS_SEED NS_LEARNING_RATE NS_BATCH_SIZE NS_NUM_EPOCHS NS_PATIENCE

die() { printf 'ERRO: %s\n' "$*" >&2; exit 2; }
usage() {
    cat <<HELP
Uso: bash scripts/run_ns_momentum_diagnostico.sh --plan | --dry-run | --train

Tres condicoes (defaults):
  controle            ganho inicial 0;   LR type1 (decai por epoca).
  ganho_inicial       ganho inicial 0.1; LR type1.
  ganho_lr_constante  ganho inicial 0.1; LR constant.
Cada condicao: NS, treino80, LB256, H25, K_loss1, lambda1, RankNet proximity,
Cauchy natural A0.025, momentum512 e eta0.01. TOTAL DEFAULT: 3 treinos.
Nao sao treinados controles temporais nem RankNet sem kernel adicionais.
Diagnosticos de ganho/gradientes/perdas por epoca ficam habilitados.

Grade editavel exclusivamente por exports DIAG_* (evita herdar grades anteriores):
DIAG_MODEL, DIAG_TRAIN_VALUES, DIAG_LOOKBACKS_VALUES, DIAG_HORIZON_VALUES,
DIAG_K_LOSS_VALUES, DIAG_LAMBDAS, DIAG_CROSS_LOSS (uma loss *_proximity),
DIAG_KERNEL_SPEC (um kernel, ex.: 'student 10'), DIAG_MASK_A, DIAG_MASK_MODE,
DIAG_MOMENTUM_LOOKBACK, DIAG_MOMENTUM_ETA, DIAG_GAIN_INIT,
DIAG_CONTROL_GAIN_INIT, DIAG_BASE_LRADJ, DIAG_ALT_LRADJ.
DIAG_CONDITIONS seleciona nomes acima; DIAG_PARALLEL=1 remove encadeamento entre condicoes.
EXPERIMENT_TAG define a raiz; cada condicao usa uma subpasta propria.
Exports gerais MODEL/LOOKBACKS_VALUES/HORIZON_VALUES/CROSS_LOSSES_VALUES/KERNEL_SPECS,
NS_ONLY_BRANCHES, NS_LRADJ e MOMENTUM_* nao alteram este desenho; use DIAG_*.

Treino e recursos mantem os exports NS_* habituais: NS_SEED, NS_LEARNING_RATE,
NS_BATCH_SIZE, NS_NUM_EPOCHS, NS_PATIENCE, NS_PARTITION, NS_GRES, NS_NODELIST,
NS_MAX_CONCURRENT_JOBS, NS_TIME_LIMIT, NS_CPUS_PER_TASK, NS_MEMORY,
NS_PYTHON_BIN, NS_RETURNS_FILE. GRES vazio continua aceito.
Por padrao as condicoes sao encadeadas afterany; isso NAO fixa o no/GPU.
Para controlar o hardware, fixe NS_NODELIST e mantenha o mesmo ambiente.
--train valida as tres condicoes antes da primeira submissao e reutiliza as marcas
individuais do launcher de kernels; repetir nao duplica submissoes bem-sucedidas.
--plan apenas conta, --dry-run nao submete. Nao executa backtest nem seleciona
hiperparametros pelo teste. Use nova EXPERIMENT_TAG ao mudar a configuracao.
HELP
}

setup() {
    [[ "$DIAG_PARALLEL" == 0 || "$DIAG_PARALLEL" == 1 ]] || die 'DIAG_PARALLEL deve ser 0 ou 1.'
    [[ -z "$NS_AFTER_JOB" || "$NS_AFTER_JOB" =~ ^[0-9]+$ ]] || die 'NS_AFTER_JOB exige um ID Slurm numerico.'
    [[ "$DIAG_KERNEL_SPEC" != *';'* && "$DIAG_KERNEL_SPEC" != *$'\n'* ]] || die 'DIAG_KERNEL_SPEC aceita um unico kernel.'
    case "$DIAG_CROSS_LOSS" in
        ranknet_proximity|hinge_proximity|pairwise_mse_proximity) ;;
        *) die 'DIAG_CROSS_LOSS deve ser ranknet_proximity, hinge_proximity ou pairwise_mse_proximity.';;
    esac
    read -r -a fields <<< "$DIAG_KERNEL_SPEC"
    ((${#fields[@]})) || die 'DIAG_KERNEL_SPEC vazio.'
    branch=${fields[0]}
    local token
    for token in "${fields[@]:1}"; do
        branch+="_${token//./p}"; branch=${branch//+/p}; branch=${branch//-/m}
    done
    read -r -a conditions <<< "$DIAG_CONDITIONS"
    ((${#conditions[@]})) || die 'DIAG_CONDITIONS vazio.'
    local -A seen=()
    for token in "${conditions[@]}"; do
        case "$token" in controle|ganho_inicial|ganho_lr_constante) ;; *) die "Condicao desconhecida: $token";; esac
        [[ ! ${seen[$token]+yes} ]] || die "Condicao repetida: $token"
        seen[$token]=1
    done
    launcher="$TFB_ROOT/scripts/run_ns_kernels_isolado.sh"
    [[ -f "$launcher" ]] || die "Launcher ausente: $launcher"
}

condition() {
    local name=$1 action=$2 dependency=${3:-} gain=$DIAG_GAIN_INIT schedule=$DIAG_BASE_LRADJ
    [[ "$name" != controle ]] || gain=$DIAG_CONTROL_GAIN_INIT
    [[ "$name" != ganho_lr_constante ]] || schedule=$DIAG_ALT_LRADJ
    printf '\nCondicao: %s; ganho inicial=%s; lradj=%s\n' "$name" "$gain" "$schedule"
    env TFB_ROOT="$TFB_ROOT" EXPERIMENT_TAG="$EXPERIMENT_TAG/$name" \
      MODEL="$DIAG_MODEL" TRAIN_VALUES="$DIAG_TRAIN_VALUES" LOOKBACKS_VALUES="$DIAG_LOOKBACKS_VALUES" \
      HORIZON_VALUES="$DIAG_HORIZON_VALUES" K_LOSS_VALUES="$DIAG_K_LOSS_VALUES" POSITIVE_LAMBDAS="$DIAG_LAMBDAS" \
      CROSS_LOSSES_VALUES="${DIAG_CROSS_LOSS%_proximity} $DIAG_CROSS_LOSS" DELTAS_VALUES=0 \
      KERNEL_SPECS="$DIAG_KERNEL_SPEC" NS_ONLY_BRANCHES="$branch" NS_MASK_A="$DIAG_MASK_A" NS_MASK_MODE="$DIAG_MASK_MODE" \
      MOMENTUM_MODE=residual_output MOMENTUM_LOOKBACK="$DIAG_MOMENTUM_LOOKBACK" MOMENTUM_ETA="$DIAG_MOMENTUM_ETA" \
      MOMENTUM_GAIN_INIT="$gain" MOMENTUM_DIAGNOSTICS=1 NS_LRADJ="$schedule" \
      NS_LOSS_TRACK_COMPONENTS=true NS_PARALLEL_BRANCHES=0 NS_AFTER_JOB="$dependency" \
      bash "$launcher" "$action"
}

plan_all() {
    local name output count total=0
    for name in "${conditions[@]}"; do
        output=$(condition "$name" --plan "$NS_AFTER_JOB") || return $?
        printf '%s\n' "$output"
        count=$(printf '%s\n' "$output" | awk '/^TOTAL: [0-9]+ treinos unicos;/ {print $2}')
        [[ "$count" =~ ^[0-9]+$ ]] || die "Nao foi possivel contar a condicao $name."
        total=$((total+count))
    done
    printf '\nDIAGNOSTICO TOTAL: %d treinos; %d condicoes.\n' "$total" "${#conditions[@]}"
    printf 'Mesmo seed=%s, LR inicial=%s, batch=%s, epocas=%s, patience=%s em todas.\n' "$NS_SEED" "$NS_LEARNING_RATE" "$NS_BATCH_SIZE" "$NS_NUM_EPOCHS" "$NS_PATIENCE"
    if [[ "$DIAG_PARALLEL" == 0 ]]; then printf 'Condicoes encadeadas afterany; no/GPU nao fixados pelo encadeamento.\n';
    else printf 'Condicoes sem dependencia entre si; NS_MAX_CONCURRENT_JOBS vale por array.\n'; fi
}

main() {
    local action=${1:---help} name output last previous=$NS_AFTER_JOB
    [[ $# -le 1 ]] || die 'Argumentos excedentes.'
    case "$action" in --help|-h) usage; return;; --plan|--dry-run|--train|--submit) ;; *) die 'Acao desconhecida; use --help.';; esac
    setup
    plan_all
    [[ "$action" != --plan ]] || return 0
    # All real configurations must validate before any Slurm submission.
    for name in "${conditions[@]}"; do condition "$name" --dry-run "$NS_AFTER_JOB"; done
    [[ "$action" != --dry-run ]] || return 0
    for name in "${conditions[@]}"; do
        if output=$(condition "$name" --train "$previous" 2>&1); then
            printf '%s\n' "$output"
        else
            local code=$?
            printf '%s\n' "$output" >&2
            return "$code"
        fi
        if [[ "$DIAG_PARALLEL" == 0 ]]; then
            last=$(printf '%s\n' "$output" | awk '
              /^Job [0-9]+:/ {gsub(":", "", $2); id=$2}
              /^Ramo .* ja submetido: job [0-9]+;/ {for(i=1;i<NF;i++) if($i=="job") {id=$(i+1); gsub(";", "", id)}}
              END {print id}')
            [[ "$last" =~ ^[0-9]+$ ]] || die "Nao foi possivel identificar o job de $name; confira a marca de submissao antes de retomar."
            previous=$last
        fi
    done
}
main "$@"
