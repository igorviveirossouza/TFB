#!/usr/bin/env bash
# Refinamento em duas etapas: composicao das losses e escala do kernel.
# Reutiliza validacao, launchers, retomada e backtests do experimento rankmix.
set -Eeuo pipefail

EXPERIMENT_TAG="${EXPERIMENT_TAG:-isolados/rankmix_refinamento_t80_k1_h1.25_v1}"
TEMPORAL_LOSSES_VALUES="${TEMPORAL_LOSSES_VALUES:-mse}"
LAMBDAS_VALUES="${LAMBDAS_VALUES:-0 0.1 0.25 0.5 1}"
RANK_MIX_MODES="${RANK_MIX_MODES:-global}"
RANK_MIX_STAGE="${RANK_MIX_STAGE:-composicao}"
# Ordem fixa: ListNet, ListMLE, ApproxNDCG. Separe vetores por ; ou nova linha.
RANK_MIX_WEIGHT_SPECS="${RANK_MIX_WEIGHT_SPECS:-0.5 0.5 0;0.25 0.75 0;0 1 0;0 0.75 0.25}"
RANK_MIX_SELECTED_WEIGHTS="${RANK_MIX_SELECTED_WEIGHTS:-}"
NS_MASK_A_VALUES="${NS_MASK_A_VALUES:-0.01 0.025 0.05}"
# Na etapa seguinte, IDs antigos podem ter sido removidos do historico do Slurm.
NS_RESUME_CHAIN_OLD_JOBS="${NS_RESUME_CHAIN_OLD_JOBS:-0}"

# O caminho do modulo nao depende de TFB_ROOT, que pode apontar para um fixture.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_ns_rankmix_isolado.sh"

# Renomeia apenas funcoes conhecidas do modulo local; nenhum texto de usuario e avaliado.
eval "$(declare -f grid | sed '1s/^grid /rankmix_base_grid /')"
eval "$(declare -f train_launcher | sed '1s/^train_launcher /rankmix_base_train_launcher /')"
eval "$(declare -f branch_fingerprint | sed '1s/^branch_fingerprint /rankmix_base_branch_fingerprint /')"

usage() {
    cat <<HELP
Uso: bash scripts/run_ns_rankmix_refinamento.sh --plan | --dry-run | --train
     bash scripts/run_ns_rankmix_refinamento.sh --backtest [planejar|submeter|consolidar]

Etapa RANK_MIX_STAGE=composicao (padrao):
  RANK_MIX_WEIGHT_SPECS='0.5 0.5 0;0.25 0.75 0;0 1 0;0 0.75 0.25'
  NS_MASK_A=0.025; cada mistura com e sem kernel.
  Defaults: 198 treinos = 6 temporais + 144 mistos + 48 rankings puros.

Etapa RANK_MIX_STAGE=escala:
  Exige RANK_MIX_SELECTED_WEIGHTS com UM vetor escolhido apos a primeira etapa.
  NS_MASK_A_VALUES='0.01 0.025 0.05'; comparador sem kernel treinado uma vez.
  Defaults: grade de 102 treinos. Na mesma EXPERIMENT_TAG, apos composicao
  com parametros iguais e uma mistura ja testada, somente 48 treinos novos;
  os 54 previamente submetidos sao preservados pelas marcas de submissao.
  Nenhuma escolha de mistura e feita automaticamente pelo script.

Defaults comuns: Nonstationary_Transformer; treino80; LB32/256/512; H1/25;
K_loss1; MSE; lambdas 0/.1/.25/.5/1; modo global; Cauchy natural; seed2026.
Ordem dos pesos: ListNet ListMLE ApproxNDCG. Pesos devem somar 1.
Todos os parametros do launcher original continuam editaveis via export:
MODEL, LOOKBACKS_VALUES, HORIZON_VALUES, K_LOSS_VALUES, LAMBDAS_VALUES,
TEMPORAL_LOSSES_VALUES, SEEDS_VALUES, KERNEL_SPECS, NS_MASK_MODE,
RANK_MIX_MODES, RANK_MIX_REGIONS, RANK_MIX_FRACTION, NS_PARTITION,
NS_GRES, NS_NODELIST, NS_MAX_CONCURRENT_JOBS, NS_PARALLEL_BRANCHES,
NS_ONLY_BRANCHES, PRED_ROOT, BACKTEST_ROOT e parametros de treino/recursos.
CROSS_LOSSES_VALUES conserva o par rankmix rankmix_proximity.
Para os vetores de pesos desta grade use RANK_MIX_WEIGHT_SPECS (composicao)
ou RANK_MIX_SELECTED_WEIGHTS (escala), em vez de RANK_MIX_WEIGHTS.

Use a mesma EXPERIMENT_TAG entre etapas para compartilhar controles e A=0.025.
Use outra tag se mudar modelo/grade/hiperparametros de um ramo ja submetido.
--plan conta a grade selecionada, nao consulta Slurm nem presume conclusao.
--train preserva ramos ja submetidos; submissao nao significa treino concluido.
Por padrao, dependencias encadeiam apenas os NOVOS jobs desta invocacao;
IDs antigos das marcas nao sao usados. NS_AFTER_JOB continua sendo respeitado.
NS_ONLY_BRANCHES aceita os nomes completos impressos pelo plano.
Momentum continua benchmark externo, sem entrada nem residual na rede.
Analise local: analise_ic_ns_rankmix.R, usando a raiz completa desta tag.
Previsoes: $PRED_ROOT
HELP
}

grid() {
    local requested_branches=$NS_ONLY_BRANCHES weights_input scales_input spec a canonical_a slug
    local field branch fingerprint i
    local -a specs=() scales=() all_fields=(BRANCHES BRANCH_MODES BRANCH_TEMPORALS
        BRANCH_SEEDS BRANCH_LAMBDAS BRANCH_LOSSES MASKS COUNTS BRANCH_PATHS BRANCH_MIX_MODES)
    local -A seen_weights=() seen_scales=() seen_branches=()

    [[ "$NS_RESUME_CHAIN_OLD_JOBS" == 0 || "$NS_RESUME_CHAIN_OLD_JOBS" == 1 ]] || die 'NS_RESUME_CHAIN_OLD_JOBS deve ser 0 ou 1.'

    case "$RANK_MIX_STAGE" in
      composicao) weights_input=$RANK_MIX_WEIGHT_SPECS; scales_input=$NS_MASK_A;;
      escala)
        [[ -n "$RANK_MIX_SELECTED_WEIGHTS" ]] || die 'Etapa escala exige export RANK_MIX_SELECTED_WEIGHTS="..."; escolha a mistura apos analisar composicao.'
        [[ "$RANK_MIX_SELECTED_WEIGHTS" != *';'* && "$RANK_MIX_SELECTED_WEIGHTS" != *$'\n'* ]] || die 'RANK_MIX_SELECTED_WEIGHTS deve conter somente um vetor de tres pesos.'
        weights_input=$RANK_MIX_SELECTED_WEIGHTS; scales_input=$NS_MASK_A_VALUES;;
      *) die 'RANK_MIX_STAGE deve ser composicao ou escala.';;
    esac
    while IFS= read -r spec; do
        [[ -n "${spec//[[:space:]]/}" ]] && specs+=("$spec")
    done <<< "${weights_input//;/$'\n'}"
    ((${#specs[@]})) || die 'Grade de pesos vazia.'
    # read -a recebe uma unica linha: normaliza novas linhas da lista de A antes.
    read -r -a scales <<< "${scales_input//$'\n'/ }"
    ((${#scales[@]})) || die 'Lista de valores A vazia.'
    for i in "${!scales[@]}"; do
        a=${scales[i]}
        [[ "$a" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] &&
          awk -v x="$a" 'BEGIN {exit !(x>0 && x<1e308)}' || die "A deve ser positivo e finito: $a"
        canonical_a=$(awk -v x="$a" 'BEGIN {printf "%.17g",x}')
        [[ ! ${seen_scales[$canonical_a]+yes} ]] || die "Valor A repetido: $a"
        seen_scales[$canonical_a]=1; scales[i]=$canonical_a
    done
    # O plano base exige A unico; a uniao preserva o metadado de cada ramo.
    for field in "${all_fields[@]}"; do
        declare -g -a "REF_${field}=()"
    done
    REF_BRANCH_WEIGHTS=(); REF_BRANCH_MASK_A=()
    for spec in "${specs[@]}"; do
        local first_scale=1
        for a in "${scales[@]}"; do
            # Escopo dinamico Bash: o grid base enxerga estes valores, sem filtros
            # prematuros sobre nomes aos quais ainda acrescentaremos o parametro A.
            local RANK_MIX_WEIGHTS=$spec NS_MASK_A=$a NS_ONLY_BRANCHES=""
            rankmix_base_grid
            if ((first_scale)); then
                [[ ! ${seen_weights[$RANK_MIX_WEIGHTS]+yes} ]] || die "Vetor de pesos repetido: $spec"
                seen_weights[$RANK_MIX_WEIGHTS]=1; first_scale=0
            fi
            slug=${a//./p}; slug=${slug//-/m}; slug=${slug//+/p}
            for i in "${!BRANCHES[@]}"; do
                if [[ "${BRANCH_LOSSES[i]}" == rankmix_proximity ]]; then
                    BRANCHES[i]+="_a${slug}"
                    BRANCH_PATHS[i]+="_a${slug}"
                fi
                branch=${BRANCHES[i]}
                fingerprint=$(rankmix_base_branch_fingerprint "$i")
                if [[ ${seen_branches[$branch]+yes} ]]; then
                    [[ "${seen_branches[$branch]}" == "$fingerprint" ]] || die "Ramos diferentes com mesmo identificador: $branch"
                    continue
                fi
                seen_branches[$branch]=$fingerprint
                for field in "${all_fields[@]}"; do
                    local -n source_array="$field" destination_array="REF_$field"
                    destination_array+=("${source_array[i]}")
                done
                REF_BRANCH_WEIGHTS+=("$RANK_MIX_WEIGHTS")
                # A nao participa das perdas sem kernel. Mantem tambem esse
                # metadado inerte fixo, independente da ordem das escalas.
                if [[ "${BRANCH_LOSSES[i]}" == rankmix_proximity ]]; then
                    REF_BRANCH_MASK_A+=("$a")
                else
                    REF_BRANCH_MASK_A+=("0.025")
                fi
            done
        done
    done
    for field in "${all_fields[@]}"; do
        local -n source_array="REF_$field" destination_array="$field"
        destination_array=("${source_array[@]}")
    done
    BRANCH_WEIGHTS=("${REF_BRANCH_WEIGHTS[@]}")
    BRANCH_MASK_A=("${REF_BRANCH_MASK_A[@]}")
    # Restaura o filtro solicitado apenas depois de construir toda a uniao.
    NS_ONLY_BRANCHES=$requested_branches
    for branch in $requested_branches; do
        [[ ${seen_branches[$branch]+yes} ]] || die "Ramo inexistente: $branch"
    done
    TOTAL=0; ACTIVE_BRANCHES=0; CONTROL_TOTAL=0; MIXED_TOTAL=0; PURE_TOTAL=0
    for i in "${!BRANCHES[@]}"; do
        selected "${BRANCHES[i]}" || continue
        TOTAL=$((TOTAL+COUNTS[i])); ACTIVE_BRANCHES=$((ACTIVE_BRANCHES+1))
        if [[ "${BRANCH_MODES[i]}" == temporal ]]; then CONTROL_TOTAL=$((CONTROL_TOTAL+COUNTS[i]));
        elif [[ "${BRANCH_LAMBDAS[i]}" == 1 ]]; then PURE_TOTAL=$((PURE_TOTAL+COUNTS[i]));
        else MIXED_TOTAL=$((MIXED_TOTAL+COUNTS[i])); fi
    done
    ((ACTIVE_BRANCHES)) || die 'Nenhum ramo selecionado.'
}

train_launcher() {
    local RANK_MIX_WEIGHTS=${BRANCH_WEIGHTS[$1]} NS_MASK_A=${BRANCH_MASK_A[$1]}
    rankmix_base_train_launcher "$@"
}

branch_fingerprint() {
    local RANK_MIX_WEIGHTS=${BRANCH_WEIGHTS[$1]}
    rankmix_base_branch_fingerprint "$@"
}

plan() {
    grid
    printf 'Etapa: %s; modelos: %s; treino: %s; LB: %s; H: %s; K_loss: %s; seeds: %s\n' "$RANK_MIX_STAGE" "$MODEL" "$TRAIN_VALUES" "$LOOKBACKS_VALUES" "$HORIZON_VALUES" "$K_LOSS_VALUES" "$SEEDS_VALUES"
    printf 'Temporais: %s; rankings: %s; lambdas: %s; modos: %s\n' "$TEMPORAL_LOSSES_VALUES" "$CROSS_LOSSES_VALUES" "$LAMBDAS_VALUES" "$RANK_MIX_MODES"
    printf 'Pesos em ordem [ListNet ListMLE ApproxNDCG]; regioes: [%s]; fracao: %s\n' "$RANK_MIX_REGIONS" "$RANK_MIX_FRACTION"
    for index in "${!BRANCHES[@]}"; do
        selected "${BRANCHES[index]}" || continue
        printf '%-62s %5d treinos  pesos=[%s]  %s\n' "${BRANCHES[index]}" "${COUNTS[index]}" "${BRANCH_WEIGHTS[index]}" "${MASKS[index]}"
    done
    printf 'Controles lambda=0: %d; mistos: %d; ranking puro lambda=1: %d.\n' "$CONTROL_TOTAL" "$MIXED_TOTAL" "$PURE_TOTAL"
    printf 'TOTAL: %d treinos unicos; %d ramos selecionados.\n' "$TOTAL" "$ACTIVE_BRANCHES"
    printf 'Total da grade, incluindo ramos ja submetidos; --train verifica as marcas antes de submeter.\n'
    printf 'Momentum: benchmark externo, 0 treinos; entrada e residual desligados.\n'
    if [[ "$NS_PARALLEL_BRANCHES" == 0 ]]; then
        printf 'Ramos encadeados afterany; ate %s tarefas simultaneas por array ativo.\n' "$NS_MAX_CONCURRENT_JOBS"
    else
        printf 'Ramos paralelos: ate %d tarefas simultaneas.\n' "$((ACTIVE_BRANCHES*NS_MAX_CONCURRENT_JOBS))"
    fi
    printf 'Previsoes: %s\n' "$PRED_ROOT"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
