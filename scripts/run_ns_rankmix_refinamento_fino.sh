#!/usr/bin/env bash
# Tres hipoteses selecionadas, referencias pareadas e controles compartilhados.
set -Eeuo pipefail

EXPERIMENT_TAG="${EXPERIMENT_TAG:-isolados/rankmix_refinamento_fino_t80_k1_h1.25_v1}"
TEMPORAL_LOSSES_VALUES="${TEMPORAL_LOSSES_VALUES:-mse}"
NS_PARTITION="${NS_PARTITION:-gorgonas_dev}"
NS_GRES="${NS_GRES-}"
NS_TIME_LIMIT="${NS_TIME_LIMIT:-01:30:00}"
NS_MAX_CONCURRENT_JOBS="${NS_MAX_CONCURRENT_JOBS:-3}"
NS_PARALLEL_BRANCHES="${NS_PARALLEL_BRANCHES:-0}"
# nome | modo | pesos ListNet ListMLE ApproxNDCG | lambdas mistos | valores A
# Separe exercicios por ; ou nova linha; lambdas e As por espacos.
RANK_MIX_FOCUS_SPECS="${RANK_MIX_FOCUS_SPECS-menos_ranking|global|0.25 0.75 0|0.05|0.01;mais_proximidade|global|0.25 0.75 0|0.1|0.005;pequena_ndcg|regional|0 0.9 0.1|0.5|0.01}"
RANK_MIX_INCLUDE_REFERENCES="${RANK_MIX_INCLUDE_REFERENCES:-1}"
RANK_MIX_REFERENCE_SPECS="${RANK_MIX_REFERENCE_SPECS-referencia_global|global|0.25 0.75 0|0.1|0.01;referencia_regional|regional|0 0.75 0.25|0.5|0.01}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_ns_rankmix_refinamento.sh"
# Renomeia somente uma funcao conhecida do modulo local, nunca texto do usuario.
eval "$(declare -f grid | sed '1s/^grid /rankmix_refinement_grid /')"

usage() {
    cat <<HELP
Uso: bash scripts/run_ns_rankmix_refinamento_fino.sh --plan | --dry-run | --train
     bash scripts/run_ns_rankmix_refinamento_fino.sh --backtest [planejar|submeter|consolidar]

Tres testes, cada um Original/Proximity, mais lambda=0 e lambda=1:
  menos_ranking:    global;   pesos .25 .75 0; lambda .05; A .01
  mais_proximidade: global;   pesos .25 .75 0; lambda .1;  A .005
  pequena_ndcg:     regional; pesos 0 .9 .1;   lambda .5;  A .01
Referencias pareadas (RANK_MIX_INCLUDE_REFERENCES=1):
  global .25 .75 0 / lambda .1 / A .01; regional 0 .75 .25 / lambda .5 / A .01.
Defaults: 102 treinos = 6 temporais + 54 mistos + 42 rankings puros.
RANK_MIX_INCLUDE_REFERENCES=0 reduz a 72, mas retira as referencias pareadas.
Controles e comparadores identicos sao treinados uma unica vez nesta tag.

Grade editavel em RANK_MIX_FOCUS_SPECS e RANK_MIX_REFERENCE_SPECS:
  'nome|modo|w_ListNet w_ListMLE w_ApproxNDCG|lambdas_mistos|As;...'
Cada linha aceita varios lambdas (estritamente entre 0 e 1) e As;
o cruzamento lambda x A ocorre SO DENTRO da linha. Lambda0/1 sao automaticos.
Essas tuplas substituem LAMBDAS_VALUES, POSITIVE_LAMBDAS, NS_MASK_A,
NS_MASK_A_VALUES, RANK_MIX_STAGE, RANK_MIX_MODES, RANK_MIX_WEIGHTS,
RANK_MIX_WEIGHT_SPECS e RANK_MIX_SELECTED_WEIGHTS.
Nao herda a grade antiga desses exports. --plan mostra a grade efetiva.

Demais exports seguem o launcher rankmix: MODEL (um ou varios), TRAIN_VALUES,
LOOKBACKS_VALUES, HORIZON_VALUES, K_LOSS_VALUES, TEMPORAL_LOSSES_VALUES,
SEEDS_VALUES, KERNEL_SPECS, NS_MASK_MODE, RANK_MIX_REGIONS, RANK_MIX_FRACTION,
NS_PARTITION, NS_GRES, NS_NODELIST, NS_TIME_LIMIT, NS_MAX_CONCURRENT_JOBS,
NS_PARALLEL_BRANCHES, NS_ONLY_BRANCHES, PRED_ROOT e BACKTEST_ROOT.
Defaults: NS; treino80; LB32/256/512; H1/25; K_loss1; MSE; seed2026;
Cauchy natural; regioes top bottom top, fracao1/3; momentum externo, sem injecao.
Recursos: gorgonas_dev; sem GRES; 01:30:00; ate 3 tarefas por array;
ramos encadeados. NS_PARALLEL_BRANCHES=1 remove o encadeamento entre ramos,
mas o limite de concorrencia passa a valer por array, nao para o experimento todo.
--train preserva marcas de submissao: elas nao certificam conclusao dos treinos.
Use uma tag nova; nao importa previsoes nem marcas do refinamento anterior.
Analise: analise_ic_ns_rankmix_refinamento_fino.R no repositorio analise_resultados.
Previsoes: $PRED_ROOT
HELP
}

grid() {
    local requested_branches=$NS_ONLY_BRANCHES specs=$RANK_MIX_FOCUS_SPECS
    local row label mode weights lambda_input scales lambda canon slug field i branch fingerprint
    local -a lambdas=() all_fields=(BRANCHES BRANCH_MODES BRANCH_TEMPORALS
        BRANCH_SEEDS BRANCH_LAMBDAS BRANCH_LOSSES MASKS COUNTS BRANCH_PATHS
        BRANCH_MIX_MODES BRANCH_WEIGHTS BRANCH_MASK_A)
    local -A seen_labels=() seen_branches=() seen_lambdas=()
    [[ "$RANK_MIX_INCLUDE_REFERENCES" == 0 || "$RANK_MIX_INCLUDE_REFERENCES" == 1 ]] || die 'RANK_MIX_INCLUDE_REFERENCES deve ser 0 ou 1.'
    [[ -n "${specs//[[:space:]]/}" ]] || die 'RANK_MIX_FOCUS_SPECS vazio.'
    if [[ "$RANK_MIX_INCLUDE_REFERENCES" == 1 ]]; then
        [[ -n "${RANK_MIX_REFERENCE_SPECS//[[:space:]]/}" ]] || die 'RANK_MIX_REFERENCE_SPECS vazio.'
        specs+=";$RANK_MIX_REFERENCE_SPECS"
    fi
    for field in "${all_fields[@]}"; do declare -g -a "FOCUS_${field}=()"; done
    FOCUS_ROWS=()
    while IFS= read -r row; do
        [[ -n "${row//[[:space:]]/}" ]] || continue
        [[ "${row//[^|]/}" == '||||' ]] || die "Tupla exige cinco campos separados por |: $row"
        IFS='|' read -r label mode weights lambda_input scales <<< "$row"
        [[ "$label" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_-]*$ && ! ${seen_labels[$label]+yes} ]] || die "Nome de exercicio invalido/repetido: $label"
        seen_labels[$label]=1
        [[ "$mode" == global || "$mode" == regional ]] || die "Modo invalido: $mode"
        read -r -a lambdas <<< "$lambda_input"
        ((${#lambdas[@]})) || die "Lista de lambdas vazia: $label"
        [[ -n "${scales//[[:space:]]/}" ]] || die "Lista de As vazia: $label"
        seen_lambdas=()
        FOCUS_ROWS+=("$row")
        for lambda in "${lambdas[@]}"; do
            [[ "$lambda" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] &&
              awk -v x="$lambda" 'BEGIN {exit !(x>0 && x<1)}' || die "Lambda misto deve estar em (0,1): $lambda"
            canon=$(awk -v x="$lambda" 'BEGIN {printf "%.15g",x}')
            [[ "$canon" != 0 && "$canon" != 1 && ! ${seen_lambdas[$canon]+yes} ]] || die "Lambda repetido ou indistinguivel de 0/1: $lambda"
            seen_lambdas[$canon]=1
            # Escopo dinamico fornece uma unica tupla ao construtor ja auditado.
            local RANK_MIX_STAGE=escala RANK_MIX_SELECTED_WEIGHTS=$weights
            local RANK_MIX_MODES=$mode NS_MASK_A_VALUES=$scales LAMBDAS_VALUES="0 $canon 1" NS_ONLY_BRANCHES=""
            rankmix_refinement_grid
            for i in "${!BRANCHES[@]}"; do
                # Cada lambda tem JSON proprio; endpoints permanecem reutilizaveis.
                if [[ "${BRANCH_MODES[i]}" == composto ]]; then
                    slug=${BRANCH_LAMBDAS[i]//./p}; slug=${slug//-/m}; slug=${slug//+/p}
                    BRANCHES[i]+="_lambda${slug}"
                    BRANCH_PATHS[i]+="_lambda${slug}"
                fi
                branch=${BRANCHES[i]}; fingerprint=$(branch_fingerprint "$i")
                if [[ ${seen_branches[$branch]+yes} ]]; then
                    [[ "${seen_branches[$branch]}" == "$fingerprint" ]] || die "Ramos diferentes com mesmo identificador: $branch"
                    continue
                fi
                seen_branches[$branch]=$fingerprint
                for field in "${all_fields[@]}"; do
                    local -n source_array="$field" destination_array="FOCUS_$field"
                    destination_array+=("${source_array[i]}")
                done
            done
        done
    done <<< "${specs//;/$'\n'}"
    for field in "${all_fields[@]}"; do
        local -n source_array="FOCUS_$field" destination_array="$field"
        destination_array=("${source_array[@]}")
    done
    NS_ONLY_BRANCHES=$requested_branches
    for branch in $requested_branches; do [[ ${seen_branches[$branch]+yes} ]] || die "Ramo inexistente: $branch"; done
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

plan() {
    grid
    printf 'Refinamento fino; modelos: %s; treino: %s; LB: %s; H: %s; K_loss: %s; seeds: %s\n' "$MODEL" "$TRAIN_VALUES" "$LOOKBACKS_VALUES" "$HORIZON_VALUES" "$K_LOSS_VALUES" "$SEEDS_VALUES"
    printf 'Tuplas efetivas: nome | modo | pesos [ListNet ListMLE ApproxNDCG] | lambdas mistos | As\n'
    printf '  %s\n' "${FOCUS_ROWS[@]}"
    printf 'Lambda0/1 incluidos; controles compartilhados. Temporais: %s; kernels: %s; modo: %s\n' "$TEMPORAL_LOSSES_VALUES" "$KERNEL_SPECS" "$NS_MASK_MODE"
    for index in "${!BRANCHES[@]}"; do
        selected "${BRANCHES[index]}" || continue
        printf '%s  %d treinos  lambda=%s  pesos=[%s]  %s\n' "${BRANCHES[index]}" "${COUNTS[index]}" "${BRANCH_LAMBDAS[index]:-0}" "${BRANCH_WEIGHTS[index]}" "${MASKS[index]}"
    done
    printf 'Controles lambda=0: %d; mistos: %d; ranking puro lambda=1: %d.\n' "$CONTROL_TOTAL" "$MIXED_TOTAL" "$PURE_TOTAL"
    printf 'TOTAL: %d treinos unicos; %d ramos selecionados.\n' "$TOTAL" "$ACTIVE_BRANCHES"
    printf 'Momentum: benchmark externo, 0 treinos; entrada e residual desligados.\n'
    printf 'Slurm: particao=%s; GRES=%s; limite=%s.\n' "$NS_PARTITION" "${NS_GRES:-vazio}" "$NS_TIME_LIMIT"
    if [[ "$NS_PARALLEL_BRANCHES" == 0 ]]; then
        printf 'Ramos encadeados afterany; ate %s tarefas simultaneas por array ativo.\n' "$NS_MAX_CONCURRENT_JOBS"
    else
        printf 'Ramos paralelos: ate %d tarefas simultaneas.\n' "$((ACTIVE_BRANCHES*NS_MAX_CONCURRENT_JOBS))"
    fi
    printf 'Previsoes: %s\nAnalise: analise_ic_ns_rankmix_refinamento_fino.R\n' "$PRED_ROOT"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
