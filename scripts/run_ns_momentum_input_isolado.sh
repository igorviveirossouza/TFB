#!/usr/bin/env bash
# Ablacao 2 x 2 x sementes: momentum explicito na entrada, sem soma na saida.
set -Eeuo pipefail

TFB_ROOT="${TFB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-momentum/input_ns_cauchy_a0.025_t80_h25_k25_v1}"
INPUT_MODEL="${INPUT_MODEL:-Nonstationary_Transformer}"
INPUT_TRAIN_VALUES="${INPUT_TRAIN_VALUES:-80}"
INPUT_LOOKBACKS_VALUES="${INPUT_LOOKBACKS_VALUES:-512}"
INPUT_HORIZON_VALUES="${INPUT_HORIZON_VALUES:-25}"
INPUT_K_LOSS_VALUES="${INPUT_K_LOSS_VALUES:-25}"
INPUT_LAMBDAS="${INPUT_LAMBDAS:-1}"
INPUT_CROSS_LOSSES_VALUES="${INPUT_CROSS_LOSSES_VALUES:-ranknet ranknet_proximity}"
INPUT_KERNEL_SPEC="${INPUT_KERNEL_SPEC:-cauchy}"
INPUT_MASK_A="${INPUT_MASK_A:-0.025}"
INPUT_MASK_MODE="${INPUT_MASK_MODE:-natural}"
INPUT_MOMENTUM_LOOKBACK="${INPUT_MOMENTUM_LOOKBACK:-512}"
INPUT_SEEDS="${INPUT_SEEDS:-2026 2027}"
INPUT_MODES="${INPUT_MODES:-zero feature}"
INPUT_LRADJ="${INPUT_LRADJ:-type1}"
INPUT_PARALLEL="${INPUT_PARALLEL:-0}"
NS_AFTER_JOB="${NS_AFTER_JOB:-}"
NS_LEARNING_RATE="${NS_LEARNING_RATE:-0.0001}"
NS_BATCH_SIZE="${NS_BATCH_SIZE:-16}"
NS_NUM_EPOCHS="${NS_NUM_EPOCHS:-100}"
NS_PATIENCE="${NS_PATIENCE:-10}"
export NS_LEARNING_RATE NS_BATCH_SIZE NS_NUM_EPOCHS NS_PATIENCE

die() { printf 'ERRO: %s\n' "$*" >&2; exit 2; }
usage() {
    cat <<HELP
Uso: bash scripts/run_ns_momentum_input_isolado.sh --plan | --dry-run | --train
     bash scripts/run_ns_momentum_input_isolado.sh --backtest [planejar|submeter|consolidar]

Default: NS; treino80; LB512; momentum512; H25; K_loss25; lambda1.
Entrada zero/feature x RankNet/RankNet proximity x seeds2026/2027 = 8 treinos.
Proximity: Cauchy natural A0.025. Sem controle temporal adicional.
zero: mesma projecao aprendida, mas recebe zeros; feature: recebe momentum real.
Nao ha soma fixa do momentum na saida, ganho residual nem penalidade eta.
Validacao ordenada e ponderada por numero de amostras em TODAS as condicoes.

Parametros cientificos editaveis por export INPUT_* (evita herdar grades antigas):
INPUT_MODEL (nesta versao somente Nonstationary_Transformer), INPUT_TRAIN_VALUES,
INPUT_LOOKBACKS_VALUES, INPUT_HORIZON_VALUES, INPUT_K_LOSS_VALUES, INPUT_LAMBDAS,
INPUT_CROSS_LOSSES_VALUES (um par base/proximity), INPUT_KERNEL_SPEC (um kernel),
INPUT_MASK_A, INPUT_MASK_MODE, INPUT_MOMENTUM_LOOKBACK, INPUT_SEEDS,
INPUT_MODES ('zero feature', ou um deles), INPUT_LRADJ, INPUT_PARALLEL=0|1.
INPUT_LAMBDAS aceita compostos positivos ate 1; este desenho nao adiciona lambda0.
Use LB=momentumLB nas duas condicoes para comparar com a mesma quantidade de passado.
Ao mudar LB/H/K/loss/seeds a contagem e recalculada; use nova EXPERIMENT_TAG.

Recursos/treino seguem NS_*: NS_PARTITION, NS_GRES (vazio=sem solicitar GPU),
NS_NODELIST, NS_MAX_CONCURRENT_JOBS, NS_TIME_LIMIT, NS_CPUS_PER_TASK, NS_MEMORY,
NS_LEARNING_RATE, NS_BATCH_SIZE, NS_NUM_EPOCHS, NS_PATIENCE, NS_RETURNS_FILE,
NS_PYTHON_BIN e os demais parametros gerais do launcher de kernels.
NS_SEED e substituido por cada INPUT_SEEDS; NS_LRADJ por INPUT_LRADJ.
MODEL/LOOKBACKS_VALUES/HORIZON_VALUES/K_LOSS_VALUES/POSITIVE_LAMBDAS,
CROSS_LOSSES_VALUES/KERNEL_SPECS/MOMENTUM_* antigos nao mudam este desenho.
INPUT_PARALLEL=0 encadeia todos os ramos afterany; nao fixa no/GPU.
INPUT_PARALLEL=1 remove dependencias entre ramos/condicoes (exceto NS_AFTER_JOB).
Com 8 ramos de uma tarefa, INPUT_PARALLEL=1 permite ate 8 tarefas simultaneas;
NS_MAX_CONCURRENT_JOBS limita cada array, NAO o total entre arrays.
--train valida todos os grupos antes de submeter; repetir reutiliza marcas existentes.

Saida: /snfs2/igor.viveiros/previsoes/$EXPERIMENT_TAG/seed=<S>/entrada=<zero|feature>/composto/<ramo>
Cada run_config.json registra entrada, janela, seed, loss, validacao e codigo.
part.parquet contem previsoes da rede; nao se produz ranking_scores residual.
Backtest e opcional e deve ser solicitado apos os treinos.
O R de kernels atual NAO aceita a raiz combinando seeds/entradas. Preserve toda
a hierarquia e os run_config.json para a comparacao pareada posterior.
HELP
}

setup() {
    [[ "$INPUT_MODEL" == Nonstationary_Transformer ]] || die 'Entrada momentum implementada somente para Nonstationary_Transformer.'
    [[ "$INPUT_PARALLEL" == 0 || "$INPUT_PARALLEL" == 1 ]] || die 'INPUT_PARALLEL deve ser 0 ou 1.'
    [[ -z "$NS_AFTER_JOB" || "$NS_AFTER_JOB" =~ ^[0-9]+$ ]] || die 'NS_AFTER_JOB exige ID Slurm numerico.'
    [[ "$INPUT_MOMENTUM_LOOKBACK" =~ ^[1-9][0-9]*$ ]] || die 'INPUT_MOMENTUM_LOOKBACK deve ser inteiro positivo.'
    [[ "$INPUT_KERNEL_SPEC" != *';'* && "$INPUT_KERNEL_SPEC" != *$'\n'* ]] || die 'INPUT_KERNEL_SPEC aceita um unico kernel.'
    local token base_loss proximity_loss
    read -r -a fields <<< "$INPUT_KERNEL_SPEC"
    ((${#fields[@]})) || die 'INPUT_KERNEL_SPEC vazio.'
    branch=${fields[0]}
    for token in "${fields[@]:1}"; do
        branch+="_${token//./p}"; branch=${branch//+/p}; branch=${branch//-/m}
    done
    read -r -a losses <<< "$INPUT_CROSS_LOSSES_VALUES"
    ((${#losses[@]} == 2)) || die 'INPUT_CROSS_LOSSES_VALUES exige um par base/proximity.'
    base_loss=; proximity_loss=
    for token in "${losses[@]}"; do
        case "$token" in
            ranknet|hinge|pairwise_mse) base_loss=$token;;
            ranknet_proximity|hinge_proximity|pairwise_mse_proximity) proximity_loss=$token;;
            *) die "Loss nao suportada neste desenho: $token";;
        esac
    done
    [[ -n "$base_loss" && "$proximity_loss" == "${base_loss}_proximity" ]] || die 'As duas losses devem formar o mesmo par base/proximity.'
    read -r -a seeds <<< "$INPUT_SEEDS"
    read -r -a modes <<< "$INPUT_MODES"
    ((${#seeds[@]} && ${#modes[@]})) || die 'INPUT_SEEDS/INPUT_MODES vazios.'
    local -A seen=()
    for token in "${seeds[@]}"; do
        [[ "$token" =~ ^(0|[1-9][0-9]{0,9})$ ]] && ((token <= 4294967295)) || die "Seed invalida: $token"
        [[ ! ${seen[$token]+yes} ]] || die "Seed repetida: $token"
        seen[$token]=1
    done
    seen=()
    for token in "${modes[@]}"; do
        [[ "$token" == zero || "$token" == feature ]] || die "Modo invalido: $token"
        [[ ! ${seen[$token]+yes} ]] || die "Modo repetido: $token"
        seen[$token]=1
    done
    launcher="$TFB_ROOT/scripts/run_ns_kernels_isolado.sh"
    [[ -f "$launcher" ]] || die "Launcher ausente: $launcher"
}

condition() {
    local seed=$1 mode=$2 action=$3 dependency=$4 subaction=${5:-}
    local -a args=("$action")
    [[ -z "$subaction" ]] || args+=("$subaction")
    printf '\nCondicao: seed=%s; entrada=%s; momentumLB=%s; validacao=ordered_sample_mean\n' "$seed" "$mode" "$INPUT_MOMENTUM_LOOKBACK"
    env TFB_ROOT="$TFB_ROOT" EXPERIMENT_TAG="$EXPERIMENT_TAG/seed=$seed/entrada=$mode" \
      MODEL="$INPUT_MODEL" TRAIN_VALUES="$INPUT_TRAIN_VALUES" LOOKBACKS_VALUES="$INPUT_LOOKBACKS_VALUES" \
      HORIZON_VALUES="$INPUT_HORIZON_VALUES" K_LOSS_VALUES="$INPUT_K_LOSS_VALUES" POSITIVE_LAMBDAS="$INPUT_LAMBDAS" \
      CROSS_LOSSES_VALUES="$INPUT_CROSS_LOSSES_VALUES" DELTAS_VALUES=0 \
      KERNEL_SPECS="$INPUT_KERNEL_SPEC" NS_ONLY_BRANCHES="sem_kernel $branch" \
      NS_MASK_A="$INPUT_MASK_A" NS_MASK_MODE="$INPUT_MASK_MODE" \
      MOMENTUM_MODE=off MOMENTUM_LOOKBACK='' MOMENTUM_ETA=0 MOMENTUM_GAIN_INIT=0 MOMENTUM_DIAGNOSTICS=0 \
      MOMENTUM_INPUT_MODE="$mode" MOMENTUM_INPUT_LOOKBACK="$INPUT_MOMENTUM_LOOKBACK" \
      NS_VALIDATION_MODE=ordered_sample_mean NS_SEED="$seed" NS_LRADJ="$INPUT_LRADJ" \
      NS_PARALLEL_BRANCHES="$INPUT_PARALLEL" NS_AFTER_JOB="$dependency" \
      bash "$launcher" "${args[@]}"
}

plan_all() {
    local seed mode output count total=0
    for seed in "${seeds[@]}"; do for mode in "${modes[@]}"; do
        output=$(condition "$seed" "$mode" --plan "$NS_AFTER_JOB") || return $?
        printf '%s\n' "$output"
        count=$(printf '%s\n' "$output" | awk '/^TOTAL: [0-9]+ treinos unicos;/ {print $2}')
        [[ "$count" =~ ^[0-9]+$ ]] || die "Nao foi possivel contar seed=$seed/entrada=$mode."
        total=$((total+count))
    done; done
    printf '\nEXPERIMENTO INPUT TOTAL: %d treinos; %d sementes; %d entradas; 2 losses.\n' "$total" "${#seeds[@]}" "${#modes[@]}"
    printf 'LR=%s (%s); batch=%s; epocas=%s; patience=%s; sem soma fixa na saida.\n' "$NS_LEARNING_RATE" "$INPUT_LRADJ" "$NS_BATCH_SIZE" "$NS_NUM_EPOCHS" "$NS_PATIENCE"
    if [[ "$INPUT_PARALLEL" == 0 ]]; then printf 'Todos os grupos/ramos encadeados afterany; no/GPU nao fixados.\n';
    else printf 'Todos os grupos/ramos independentes; limites por array nao limitam o total global.\n'; fi
}

main() {
    local action=${1:---help} subaction=${2:-planejar} seed mode output code last previous=$NS_AFTER_JOB
    case "$action" in
        --help|-h) usage; return;;
        --plan|--dry-run|--train|--submit) [[ $# -le 1 ]] || die 'Argumentos excedentes.';;
        --backtest) [[ $# -le 2 ]] || die 'Argumentos excedentes.'
            case "$subaction" in planejar|submeter|consolidar) ;; *) die 'Acao de backtest invalida.';; esac;;
        *) die 'Acao desconhecida; use --help.';;
    esac
    setup
    plan_all
    [[ "$action" != --plan ]] || return 0
    if [[ "$action" == --backtest ]]; then
        for seed in "${seeds[@]}"; do for mode in "${modes[@]}"; do
            condition "$seed" "$mode" --backtest "$NS_AFTER_JOB" "$subaction"
        done; done
        return
    fi
    # Validate every actual config before the first Slurm submission.
    for seed in "${seeds[@]}"; do for mode in "${modes[@]}"; do
        condition "$seed" "$mode" --dry-run "$NS_AFTER_JOB"
    done; done
    [[ "$action" != --dry-run ]] || return 0
    for seed in "${seeds[@]}"; do for mode in "${modes[@]}"; do
        if output=$(condition "$seed" "$mode" --train "$previous" 2>&1); then
            printf '%s\n' "$output"
        else
            code=$?; printf '%s\n' "$output" >&2; return "$code"
        fi
        if [[ "$INPUT_PARALLEL" == 0 ]]; then
            last=$(printf '%s\n' "$output" | awk '
              /^Job [0-9]+:/ {gsub(":", "", $2); id=$2}
              /^Ramo .* ja submetido: job [0-9]+;/ {for(i=1;i<NF;i++) if($i=="job") {id=$(i+1); gsub(";", "", id)}}
              END {print id}')
            [[ "$last" =~ ^[0-9]+$ ]] || die "Nao foi possivel identificar o ultimo job; confira a marca de submissao."
            previous=$last
        fi
    done; done
}
main "$@"
