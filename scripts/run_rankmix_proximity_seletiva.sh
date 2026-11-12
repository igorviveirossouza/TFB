#!/usr/bin/env bash
# Mistura global: compara original, Proximity completa e Proximity por componente.
set -Eeuo pipefail
SELECTIVE_SELF="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-isolados/rankmix_proximity_seletiva_t90_v1}"
MODEL="${MODEL:-iTransformer}"
TRAIN_VALUES="${TRAIN_VALUES:-90}"
SEEDS_VALUES="${SEEDS_VALUES:-2028}"
RANK_MIX_WEIGHT_SPECS="${RANK_MIX_WEIGHT_SPECS:-0.5 0.5 0}"
RANK_MIX_MODES="${RANK_MIX_MODES:-global}"
RANK_MIX_INCLUDE_ABLATIONS="${RANK_MIX_INCLUDE_ABLATIONS:-0}"
RANK_MIX_PROXIMITY_SPECS="${RANK_MIX_PROXIMITY_SPECS-all;listnet}"
NS_MASK_A_VALUES="${NS_MASK_A_VALUES:-0.01}"

source "$(dirname -- "$SELECTIVE_SELF")/run_rankmix_robustez.sh"
eval "$(declare -f grid | sed '1s/^grid /selective_parent_grid /')"
eval "$(declare -f branch_fingerprint | sed '1s/^branch_fingerprint /selective_parent_fingerprint /')"
eval "$(declare -f train_launcher | sed '1s/^train_launcher /selective_parent_launcher /')"
eval "$(declare -f print_exports | sed '1s/^print_exports /selective_parent_exports /')"
eval "$(declare -f plan | sed '1s/^plan /selective_parent_plan /')"

usage() {
    cat <<'HELP'
Uso: bash scripts/run_rankmix_proximity_seletiva.sh --plan | --dry-run | --train
     bash scripts/run_rankmix_proximity_seletiva.sh --print-exports | --list-branches
     bash scripts/run_rankmix_proximity_seletiva.sh --backtest [planejar|submeter|consolidar]

Defaults: iTransformer; corte90; seed2028; LB32/256/512; H1/25; MSE.
Lambdas0/.1/.5/.8/1; mistura global [ListNet ListMLE ApproxNDCG]=[.5 .5 0].
Compara rankmix original; Proximity em todos; somente ListNet Proximity.
Kernel Cauchy natural A=.01. Momentum somente benchmark, sem injecao.
78 treinos: 6 temporais + 4 lambdas positivos x 6 pares LB/H x 3 versoes.

RANK_MIX_PROXIMITY_SPECS='all;listnet': grupos separados por ponto e virgula.
Em cada grupo use all ou nomes separados por espaco: listnet listmle approxndcg.
Exemplo: 'all;listnet;listmle'. Componentes ausentes usam a loss original.
Seletores equivalentes nos componentes de peso positivo nao repetem ajustes.
Grupo que nao mascara nenhum componente ativo e ignorado: equivale ao original.
Controles temporais e originais sao compartilhados entre seletores e valores de A.

Todos os parametros do experimento robustez continuam editaveis via export.
--print-exports mostra valores efetivos, caminhos e comandos completos.
NS_MASK_A_VALUES controla A; o launcher nao aprende A nem usa o teste para escolhe-lo.
NS_RANKING_PROTOCOL=origin_anchored_windows: janelas1/5/10/15/20/25, filtradas<=H,
em um unico ajuste. K_LOSS_VALUES=1; blocks habilita o protocolo antigo.
--plan nao submete; --dry-run valida produtores; --train usa JSONs congelados.
Use nova EXPERIMENT_TAG ao mudar parametros de uma grade ja submetida.
HELP
}

canonical_proximity_specs() {
    python3 - "$RANK_MIX_PROXIMITY_SPECS" <<'PY'
import sys
names = ('listnet', 'listmle', 'approxndcg')
result = []
for group in sys.argv[1].split(';'):
    parts = group.split()
    if not parts:
        raise SystemExit('RANK_MIX_PROXIMITY_SPECS contem um grupo vazio.')
    if parts == ['all']:
        value = 'all'
    else:
        if any(x not in names for x in parts):
            raise SystemExit('Seletor Proximity invalido; use all ou listnet/listmle/approxndcg.')
        value = ' '.join(x for x in names if x in parts)
    if value not in result:
        result.append(value)
print('\n'.join(result))
PY
}

grid() {
    local requested_branches=$NS_ONLY_BRANCHES scopes scope effective i field branch
    local NS_ONLY_BRANCHES='' SELECTIVE_BUILDING_PARENT=1
    local -a selective_fields=(BRANCHES BRANCH_MODES BRANCH_TEMPORALS BRANCH_SEEDS BRANCH_LAMBDAS
        BRANCH_LOSSES MASKS COUNTS BRANCH_PATHS BRANCH_MIX_MODES BRANCH_WEIGHTS BRANCH_MASK_A)
    local -A seen=()
    scopes=$(canonical_proximity_specs)
    BRANCH_PROXIMITY_COMPONENTS=()
    selective_parent_grid
    SELECTIVE_BUILDING_PARENT=0
    for field in "${selective_fields[@]}"; do
        declare -g -a "SELECTIVE_INPUT_$field=()"
        local -n src="$field" dst="SELECTIVE_INPUT_$field"
        dst=("${src[@]}"); src=()
    done
    for i in "${!SELECTIVE_INPUT_BRANCHES[@]}"; do
        local branch_scopes='all'
        if [[ "${SELECTIVE_INPUT_BRANCH_LOSSES[i]}" == rankmix_proximity ]]; then branch_scopes=$scopes; fi
        while IFS= read -r scope; do
            effective=''
            if [[ "${SELECTIVE_INPUT_BRANCH_LOSSES[i]}" == rankmix_proximity ]]; then
                effective=$(python3 - "${SELECTIVE_INPUT_BRANCH_WEIGHTS[i]}" "$scope" <<'PY'
import sys
from decimal import Decimal
names = ('listnet', 'listmle', 'approxndcg')
active = [name for name, weight in zip(names, sys.argv[1].split()) if Decimal(weight) > 0]
selected = active if sys.argv[2] == 'all' else [name for name in active if name in sys.argv[2].split()]
print('all' if selected == active else ' '.join(selected))
PY
)
                if [[ -z "$effective" ]]; then
                    printf 'Seletor [%s] sem componente ativo em [%s]: equivalente ao original, omitido.\n' "$scope" "${SELECTIVE_INPUT_BRANCH_WEIGHTS[i]}" >&2
                    continue
                fi
            fi
            [[ ! ${seen["$i|$effective"]+yes} ]] || continue
            seen["$i|$effective"]=1
            for field in "${selective_fields[@]}"; do
                local -n src="SELECTIVE_INPUT_$field" dst="$field"
                dst+=("${src[i]}")
            done
            local last=$((${#BRANCHES[@]}-1))
            if [[ -n "$effective" && "$effective" != all ]]; then
                BRANCHES[last]+="_prox_on_${effective// /_}"
                # Ramos irmaos: a leitura recursiva do ramo completo nao pode
                # incorporar os Parquets da variante seletiva.
                BRANCH_PATHS[last]="${BRANCH_PATHS[last]%/*}/prox_on=${effective// /_}/${BRANCH_PATHS[last]##*/}"
                BRANCH_PROXIMITY_COMPONENTS+=("$effective")
            else BRANCH_PROXIMITY_COMPONENTS+=(''); fi
        done <<< "$branch_scopes"
    done
    NS_ONLY_BRANCHES=$requested_branches
    for branch in $requested_branches; do
        local found=0
        for scope in "${BRANCHES[@]}"; do [[ "$scope" != "$branch" ]] || found=1; done
        ((found)) || die "Ramo inexistente: $branch"
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

branch_fingerprint() {
    local selector=''
    if [[ "${SELECTIVE_BUILDING_PARENT:-0}" != 1 ]]; then selector=${BRANCH_PROXIMITY_COMPONENTS[$1]-}; fi
    if [[ -z "$selector" ]]; then selective_parent_fingerprint "$1";
    else printf '%s\n' "$(selective_parent_fingerprint "$1")" "proximity_components=$selector" | sha256sum | cut -d' ' -f1; fi
}

train_launcher() {
    local SELECTIVE_BRANCH_PROXIMITY_COMPONENTS=${BRANCH_PROXIMITY_COMPONENTS[$1]-}
    selective_parent_launcher "$@"
}

plan() {
    selective_parent_plan
    printf 'Proximity por componente: [%s]. Componentes nao selecionados usam a loss original.\n' "$RANK_MIX_PROXIMITY_SPECS"
}

print_exports() {
    selective_parent_exports | sed '/^bash scripts\/run_rankmix_robustez.sh /d'
    printf 'export RANK_MIX_PROXIMITY_SPECS=%q\n' "$RANK_MIX_PROXIMITY_SPECS"
    # Estes controles sao herdados, embora nao participem das losses default.
    local name
    for name in DELTAS_VALUES NS_SPEARMAN_TAU NS_RANKNET_ALPHA NS_HINGE_MARGIN; do
        printf 'export %s=%q\n' "$name" "${!name}"
    done
    printf '\nbash scripts/run_rankmix_proximity_seletiva.sh --plan\nbash scripts/run_rankmix_proximity_seletiva.sh --train\n'
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    case "${1:---help}" in
      --print-exports) [[ $# == 1 ]] || die 'Argumentos excedentes.'; print_exports;;
      --list-branches) [[ $# == 1 ]] || die 'Argumentos excedentes.'; grid
        for i in "${!BRANCHES[@]}"; do selected "${BRANCHES[i]}" || continue; printf '%s | %s | %s\n' "${BRANCHES[i]}" "${COUNTS[i]}" "${BRANCH_PATHS[i]}"; done;;
      *) main "$@";;
    esac
fi
