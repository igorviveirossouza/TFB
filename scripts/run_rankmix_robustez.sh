#!/usr/bin/env bash
# Losses pareadas em varios cortes; modelos nativos, sem injecao de momentum.
set -Eeuo pipefail
SUPER_SELF="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
TFB_ROOT="${TFB_ROOT:-$(cd -- "$(dirname -- "$SUPER_SELF")/.." && pwd)}"

# O worker usa somente o JSON congelado; exports posteriores nao mudam o treino.
if [[ "${1:-}" == --on-node ]]; then
    [[ $# -ge 3 && ( "$2" == temporal || "$2" == composto ) && "$3" == worker ]] || exit 2
    worker_mode=$2; shift 2
    [[ -n "${B3_INLINE_CONFIG:-}" && -n "${B3_CONFIG_SOURCE:-}" ]] || exit 2
    if [[ -n "${VENV_PATH:-}" && -f "$VENV_PATH" ]]; then source "$VENV_PATH"; fi
    cd -- "$TFB_ROOT"
    exec "${PYTHON_BIN:?}" "$TFB_ROOT/scripts/b3_experiment.py" "$worker_mode" "$@"
fi
if [[ "${1:-}" == --launch-one ]]; then
    [[ $# == 3 && ( "$2" == temporal || "$2" == composto ) ]] || exit 2
    EXPERIMENT=$2; shift 2
    case "$1" in submit|--dry-run|--print-config) ;; *) exit 2;; esac
    read -r -a TRAIN_PERCENTAGES <<< "${TRAIN_PERCENTAGES:?}"
    read -r -a MODELS <<< "${MODELS:?}"
    read -r -a LOOKBACKS <<< "${LOOKBACKS:?}"
    read -r -a HORIZONS <<< "${HORIZONS:?}"
    read -r -a TRADE_WINDOWS <<< "${TRADE_WINDOWS:?}"
    read -r -a CROSS_LOSSES <<< "${CROSS_LOSSES-}"
    read -r -a LAMBDAS <<< "${LAMBDAS-}"
    read -r -a DELTAS <<< "${DELTAS:?}"
    read -r -a CROSS_MASK <<< "${CROSS_MASK:?}"
    MODEL_DEFAULTS_JSON=$("$PYTHON_BIN" - "$SUPER_MODEL_DEFAULTS_JSON" "$BATCH_SIZE" "${MODELS[*]}" <<'PY'
import json, sys
defaults = json.loads(sys.argv[1])
print(json.dumps({name: dict(defaults[name], batch_size=int(sys.argv[2]), parallel_strategy=None)
                  for name in sys.argv[3].split()}))
PY
)
    B3_CONFIG_SOURCE="$SUPER_SELF"
    export B3_WORKER_LAUNCHER="$SUPER_SELF"
    source "$TFB_ROOT/scripts/b3_experiment_env.sh"
    exec "$PYTHON_BIN" "$TFB_ROOT/scripts/b3_experiment.py" "$EXPERIMENT" "$@"
fi

EXPERIMENT_TAG="${EXPERIMENT_TAG:-isolados/rankmix_robustez_t60-90_v1}"
MODEL="${MODEL:-iTransformer}"
TRAIN_VALUES="${TRAIN_VALUES:-60 70 80 90}"
LOOKBACKS_VALUES="${LOOKBACKS_VALUES:-32 256 512}"
HORIZON_VALUES="${HORIZON_VALUES:-1 25}"
K_LOSS_VALUES="${K_LOSS_VALUES:-1}"
LAMBDAS_VALUES="${LAMBDAS_VALUES:-0 0.1 0.5 0.8 1}"
SEEDS_VALUES="${SEEDS_VALUES:-2026 2027 2028}"
TEMPORAL_LOSSES_VALUES="${TEMPORAL_LOSSES_VALUES:-mse}"
RANK_MIX_WEIGHT_SPECS="${RANK_MIX_WEIGHT_SPECS:-0.5 0.5 0;0.25 0.75 0;0 0.75 0.25;0 1 0}"
RANK_MIX_MODES="${RANK_MIX_MODES:-global regional}"
RANK_MIX_INCLUDE_ABLATIONS="${RANK_MIX_INCLUDE_ABLATIONS:-1}"
RANK_MIX_ABLATION_SPECS="${RANK_MIX_ABLATION_SPECS-1 0 0;0 0 1}"
RANK_MIX_ABLATION_LAMBDAS="${RANK_MIX_ABLATION_LAMBDAS:-1}"
NS_MASK_A_VALUES="${NS_MASK_A_VALUES:-0.01 0.025 0.05}"
NS_RANKING_PROTOCOL="${NS_RANKING_PROTOCOL:-origin_anchored_windows}"
RANK_WINDOWS_VALUES="${RANK_WINDOWS_VALUES:-1 5 10 15 20 25}"
NS_MODEL_DEFAULTS_JSON="${NS_MODEL_DEFAULTS_JSON-}"
if [[ -z "$NS_MODEL_DEFAULTS_JSON" ]]; then
    NS_MODEL_DEFAULTS_JSON='{"iTransformer":{"d_model":128,"d_ff":256,"e_layers":2,"n_heads":8,"dropout":0.1,"norm":true,"lradj":"type1"},"Nonstationary_Transformer":{"d_model":128,"d_ff":256,"e_layers":2,"d_layers":1,"n_heads":8,"dropout":0.1,"factor":3,"norm":true,"p_hidden_dims":[32,32],"p_hidden_layers":2,"lradj":"type1"},"simpleFORMER":{"d_model":512,"d_ff":2048,"e_layers":2,"n_heads":8,"temporal_hidden_dim":0,"dropout":0.1,"norm":true,"use_norm":true,"norm_eps":0.00001,"lradj":"type1"}}'
fi
NS_PARTITION="${NS_PARTITION:-gorgonas_dev}"
NS_GRES="${NS_GRES-}"
NS_MAX_CONCURRENT_JOBS="${NS_MAX_CONCURRENT_JOBS:-3}"
NS_PARALLEL_BRANCHES="${NS_PARALLEL_BRANCHES:-0}"
NS_RESUME_CHAIN_OLD_JOBS="${NS_RESUME_CHAIN_OLD_JOBS:-0}"
NS_LOG_ROOT="${NS_LOG_ROOT:-$TFB_ROOT/logs/$EXPERIMENT_TAG}"
NS_RESULT_ROOT="${NS_RESULT_ROOT:-/snfs2/igor.viveiros/result/$EXPERIMENT_TAG}"

source "$(dirname -- "$SUPER_SELF")/run_ns_rankmix_refinamento.sh"
eval "$(declare -f grid | sed '1s/^grid /robust_refinement_grid /')"
eval "$(declare -f branch_fingerprint | sed '1s/^branch_fingerprint /robust_refinement_fingerprint /')"

usage() {
    cat <<'HELP'
Uso: bash scripts/run_rankmix_robustez.sh --plan | --dry-run | --train
     bash scripts/run_rankmix_robustez.sh --list-branches | --print-exports
     bash scripts/run_rankmix_robustez.sh --backtest [planejar|submeter|consolidar]

Defaults: iTransformer; cortes60/70/80/90; LB32/256/512; H1/25; seeds2026/27/28.
MSE; lambdas0/.1/.5/.8/1; quatro vetores em RANK_MIX_WEIGHT_SPECS;
global/regional; original e Proximity Cauchy natural A=.01/.025/.05.
Ablações ListNet e ApproxNDCG isoladas: lambda1, ambos modos e todas versoes.
10440 treinos: 72 controles + 6912 mistos + 3456 rankings puros.
SEEDS_VALUES='2026' reduz a3480; momentum e benchmark externo, zero treinos.

NS_RANKING_PROTOCOL=origin_anchored_windows: media das losses dos retornos
acumulados desde a origem, em RANK_WINDOWS_VALUES, filtradas por janela<=H.
Todas essas janelas pertencem ao MESMO ajuste; K_LOSS_VALUES deve ser1.
NS_RANKING_PROTOCOL=blocks: protocolo antigo, blocos futuros de K_LOSS_VALUES
pregoes; K deve dividir H. Neste modo RANK_WINDOWS_VALUES e inativo.
Os dois protocolos mantem uma unica saida diaria compartilhada, sem cabeca nova.

Todos os exports publicos estao em --print-exports, com valores efetivos.
RANK_MIX_WEIGHT_SPECS cruza integralmente LAMBDAS_VALUES x modos x versoes/A.
RANK_MIX_ABLATION_SPECS cruza RANK_MIX_ABLATION_LAMBDAS x modos x versoes/A.
Pesos: [ListNet ListMLE ApproxNDCG], soma1. Vetores separados por ponto e virgula.
Controles e configuracoes iguais sao treinados uma unica vez por corte/modelo/LB/H/seed.
CROSS_LOSSES_VALUES conserva 'rankmix rankmix_proximity' para comparacao pareada.
NS_MASK_A_VALUES controla a grade de A; NS_MASK_A isolado nao a altera.
MODEL aceita um ou varios nomes com defaults em NS_MODEL_DEFAULTS_JSON.
Arquitetura e fixa dentro de cada modelo, editavel nesse JSON; batch em NS_BATCH_SIZE.

--plan nao submete jobs. --dry-run valida os produtores e caminhos reais.
--train preserva marcas de SUBMISSAO, que nao certificam conclusao.
Use nova EXPERIMENT_TAG ao mudar a grade/hiperparametros ja submetidos.
Padrao encadeia arrays, ate3tarefas simultaneas; com NS_PARALLEL_BRANCHES=1,
NS_MAX_CONCURRENT_JOBS passa a limitar cada array, nao o experimento inteiro.
R e avaliacao economica ficam para a proxima etapa.
HELP
}

validate_protocol() {
    integer_list HORIZON_VALUES "$HORIZON_VALUES"
    case "$NS_RANKING_PROTOCOL" in
      origin_anchored_windows)
        [[ "$K_LOSS_VALUES" == 1 ]] || die 'origin_anchored_windows exige K_LOSS_VALUES=1; edite RANK_WINDOWS_VALUES para as janelas conjuntas.'
        integer_list RANK_WINDOWS_VALUES "$RANK_WINDOWS_VALUES"
        local h k valid
        for h in $HORIZON_VALUES; do
            valid=0
            for k in $RANK_WINDOWS_VALUES; do ((k<=h)) && valid=1; done
            ((valid)) || die "H=$h sem janela supervisionada <=H."
        done;;
      blocks) ;;
      *) die 'NS_RANKING_PROTOCOL deve ser blocks ou origin_anchored_windows.';;
    esac
    [[ "$RANK_MIX_INCLUDE_ABLATIONS" == 0 || "$RANK_MIX_INCLUDE_ABLATIONS" == 1 ]] || die 'RANK_MIX_INCLUDE_ABLATIONS deve ser0ou1.'
    # Somente Python standard library; --plan nao depende do venv de treino.
    python3 - "$NS_MODEL_DEFAULTS_JSON" "$MODEL" <<'PY'
import json, sys
d = json.loads(sys.argv[1])
if not isinstance(d, dict): raise SystemExit('NS_MODEL_DEFAULTS_JSON deve ser objeto por modelo.')
for name in sys.argv[2].split():
    if name not in d or not isinstance(d[name], dict): raise SystemExit('Modelo sem defaults: '+name)
    reserved = {'batch_size','seq_len','pred_len','horizon','task_head_mode','rank_windows',
                'ranking_construction','parallel_strategy','lr','num_epochs','patience','validation_mode'}
    if any(k in reserved or k.startswith(('loss','momentum')) for k in d[name]):
        raise SystemExit('JSON de arquitetura contem campos de treino/loss/cabecas; use os exports especificos.')
PY
}

grid() {
    validate_protocol
    local requested_branches=$NS_ONLY_BRANCHES category specs lambda_input spec lambda slug field i branch fingerprint
    local -a all_fields=(BRANCHES BRANCH_MODES BRANCH_TEMPORALS BRANCH_SEEDS
        BRANCH_LAMBDAS BRANCH_LOSSES MASKS COUNTS BRANCH_PATHS BRANCH_MIX_MODES BRANCH_WEIGHTS BRANCH_MASK_A)
    local -A seen=()
    for field in "${all_fields[@]}"; do declare -g -a "ROBUST_$field=()"; done
    for category in principal ablacao; do
        if [[ "$category" == principal ]]; then specs=$RANK_MIX_WEIGHT_SPECS; lambda_input=$LAMBDAS_VALUES
        else
            [[ "$RANK_MIX_INCLUDE_ABLATIONS" == 1 ]] || continue
            specs=$RANK_MIX_ABLATION_SPECS; lambda_input=$RANK_MIX_ABLATION_LAMBDAS
        fi
        [[ -n "${specs//[[:space:]]/}" && -n "${lambda_input//[[:space:]]/}" ]] || die "Grade vazia: $category"
        # Aliases como .1 e 0.10 sao o mesmo ajuste, inclusive entre categorias.
        lambda_input=$(python3 - "$lambda_input" <<'PY'
import re, sys
from decimal import Decimal
values = []
for token in sys.argv[1].split():
    if not re.fullmatch(r'(\d+(\.\d*)?|\.\d+)', token):
        raise SystemExit('Lambda invalido: '+token)
    x = Decimal(token)
    if not 0 <= x <= 1: raise SystemExit('Lambda fora de [0,1]: '+token)
    value = format(x.normalize(), 'f')
    if value not in values: values.append(value)
print(' '.join(values))
PY
)
        while IFS= read -r spec; do
            [[ -n "${spec//[[:space:]]/}" ]] || continue
            for lambda in $lambda_input; do
                # Cada lambda tem seu JSON; uniao elimina sobreposicoes entre categorias.
                local RANK_MIX_STAGE=escala RANK_MIX_SELECTED_WEIGHTS=$spec
                local LAMBDAS_VALUES=$lambda NS_ONLY_BRANCHES=''
                robust_refinement_grid
                for i in "${!BRANCHES[@]}"; do
                    if [[ "${BRANCH_MODES[i]}" == composto ]]; then
                        slug=${BRANCH_LAMBDAS[i]//./p}
                        BRANCHES[i]+="_lambda${slug}"
                        BRANCH_PATHS[i]+="_lambda${slug}"
                    fi
                    branch=${BRANCHES[i]}; fingerprint=$(branch_fingerprint "$i")
                    if [[ ${seen[$branch]+yes} ]]; then
                        [[ ${seen[$branch]} == "$fingerprint" ]] || die "Colisao de configuracoes: $branch"
                        continue
                    fi
                    seen[$branch]=$fingerprint
                    for field in "${all_fields[@]}"; do
                        local -n src="$field" dst="ROBUST_$field"
                        dst+=("${src[i]}")
                    done
                done
            done
        done <<< "${specs//;/$'\n'}"
    done
    for field in "${all_fields[@]}"; do
        local -n src="ROBUST_$field" dst="$field"
        dst=("${src[@]}")
    done
    NS_ONLY_BRANCHES=$requested_branches
    for branch in $requested_branches; do [[ ${seen[$branch]+yes} ]] || die "Ramo inexistente: $branch"; done
    TOTAL=0; ACTIVE_BRANCHES=0; CONTROL_TOTAL=0; MIXED_TOTAL=0; PURE_TOTAL=0
    for i in "${!BRANCHES[@]}"; do
        selected "${BRANCHES[i]}" || continue
        TOTAL=$((TOTAL+COUNTS[i])); ACTIVE_BRANCHES=$((ACTIVE_BRANCHES+1))
        if [[ "${BRANCH_MODES[i]}" == temporal ]]; then CONTROL_TOTAL=$((CONTROL_TOTAL+COUNTS[i]));
        elif [[ "${BRANCH_LAMBDAS[i]}" == 1 ]]; then PURE_TOTAL=$((PURE_TOTAL+COUNTS[i]));
        else MIXED_TOTAL=$((MIXED_TOTAL+COUNTS[i])); fi
        BRANCH_PATHS[i]="protocolo=$NS_RANKING_PROTOCOL/${BRANCH_PATHS[i]}"
    done
    ((ACTIVE_BRANCHES)) || die 'Nenhum ramo selecionado.'
}

branch_fingerprint() {
    printf '%s\n' "$(robust_refinement_fingerprint "$1")" "$NS_RANKING_PROTOCOL" \
        "$([[ "$NS_RANKING_PROTOCOL" == blocks ]] || printf '%s' "$RANK_WINDOWS_VALUES")" \
        "$NS_MODEL_DEFAULTS_JSON" | sha256sum | cut -d' ' -f1
}

plan() {
    grid
    printf 'Robustez das losses | modelos=%s | cortes=%s | LB=%s | H=%s | seeds=%s\n' "$MODEL" "$TRAIN_VALUES" "$LOOKBACKS_VALUES" "$HORIZON_VALUES" "$SEEDS_VALUES"
    printf 'Protocolo=%s | K_loss=%s | janelas conjuntas=%s\n' "$NS_RANKING_PROTOCOL" "$K_LOSS_VALUES" "$([[ "$NS_RANKING_PROTOCOL" == blocks ]] && printf inativas || printf '%s' "$RANK_WINDOWS_VALUES")"
    printf 'Principal: pesos=[%s]; lambdas=[%s]; modos=[%s]\n' "$RANK_MIX_WEIGHT_SPECS" "$LAMBDAS_VALUES" "$RANK_MIX_MODES"
    printf 'Ablacoes=%s: pesos=[%s]; lambdas=[%s]\n' "$RANK_MIX_INCLUDE_ABLATIONS" "$RANK_MIX_ABLATION_SPECS" "$RANK_MIX_ABLATION_LAMBDAS"
    printf 'Temporais=%s; rankings=%s; kernels=%s; A=[%s]; modo=%s\n' "$TEMPORAL_LOSSES_VALUES" "$CROSS_LOSSES_VALUES" "$KERNEL_SPECS" "$NS_MASK_A_VALUES" "$NS_MASK_MODE"
    printf 'Controles lambda=0: %d; mistos: %d; ranking puro lambda=1: %d.\n' "$CONTROL_TOTAL" "$MIXED_TOTAL" "$PURE_TOTAL"
    printf 'TOTAL: %d treinos unicos; %d ramos selecionados.\n' "$TOTAL" "$ACTIVE_BRANCHES"
    printf 'Momentum externo: 0 treinos; sem entrada/residual. Saida nativa diaria compartilhada.\n'
    printf 'Slurm: particao=%s; GRES=%s; limite=%s; concorrencia por array=%s; arrays paralelos=%s\n' "$NS_PARTITION" "${NS_GRES:-vazio}" "$NS_TIME_LIMIT" "$NS_MAX_CONCURRENT_JOBS" "$NS_PARALLEL_BRANCHES"
    if [[ "$NS_PARALLEL_BRANCHES" == 1 ]]; then printf 'Teto entre ramos: %d tarefas (antes dos limites do cluster).\n' "$((ACTIVE_BRANCHES*NS_MAX_CONCURRENT_JOBS))"; fi
    printf 'Previsoes: %s\nLogs: %s\n' "$PRED_ROOT" "$NS_LOG_ROOT"
}

train_launcher() {
    local index=$1 action=$2 dependency=${3:-} root path protocol=$NS_RANKING_PROTOCOL rw=$RANK_WINDOWS_VALUES
    local -a mix_env=()
    root=$(branch_root "$index"); path=${BRANCH_PATHS[index]}
    if [[ "${BRANCH_MODES[index]}" == composto ]]; then
        mix_env=("RANK_MIX_WEIGHTS=${BRANCH_WEIGHTS[index]}" "RANK_MIX_MODE=${BRANCH_MIX_MODES[index]}" "RANK_MIX_REGIONS=$RANK_MIX_REGIONS" "RANK_MIX_FRACTION=$RANK_MIX_FRACTION")
    fi
    [[ "$protocol" != blocks ]] || rw=''
    env -u RANK_MIX_WEIGHTS -u RANK_MIX_MODE -u RANK_MIX_REGIONS -u RANK_MIX_FRACTION -u B3_INLINE_CONFIG -u B3_CONFIG -u B3_CONFIG_SOURCE -u B3_WORKER_LAUNCHER -u EXPERIMENT \
      TFB_ROOT="$TFB_ROOT" PYTHON_BIN="$NS_PYTHON_BIN" VENV_PATH="$NS_VENV_PATH" \
      SUPER_MODEL_DEFAULTS_JSON="$NS_MODEL_DEFAULTS_JSON" CROSS_CONSTRUCTION="$protocol" RANK_WINDOWS="$rw" \
      DATASET_PATH="$NS_RETURNS_FILE" SOURCE_DATA_KIND=log_return \
      TRAIN_PERCENTAGES="$TRAIN_VALUES" MODELS="$MODEL" LOOKBACKS="$LOOKBACKS_VALUES" HORIZONS="$HORIZON_VALUES" \
      TRADE_WINDOWS="$K_LOSS_VALUES" CROSS_LOSSES="${BRANCH_LOSSES[index]}" LAMBDAS="${BRANCH_LAMBDAS[index]}" DELTAS="$DELTAS_VALUES" \
      CROSS_MASK="${MASKS[index]:-cauchy 0.025 natural}" \
      BATCH_SIZE="$NS_BATCH_SIZE" NUM_EPOCHS="$NS_NUM_EPOCHS" PATIENCE="$NS_PATIENCE" LEARNING_RATE="$NS_LEARNING_RATE" \
      TEMPORAL_LOSS="${BRANCH_TEMPORALS[index]}" HUBER_DELTA="$NS_HUBER_DELTA" SPEARMAN_TAU="$NS_SPEARMAN_TAU" TRAIN_RATIO_IN_TV="$NS_TRAIN_RATIO_IN_TV" SEED="${BRANCH_SEEDS[index]}" STRIDE="$NS_STRIDE" NUM_ROLLINGS="$NS_NUM_ROLLINGS" \
      CROSS_SCALE="$NS_CROSS_SCALE" RANKNET_ALPHA="$NS_RANKNET_ALPHA" LISTNET_TAU="$NS_LISTNET_TAU" LISTMLE_TAU="$NS_LISTMLE_TAU" APPROXNDCG_TAU="$NS_APPROXNDCG_TAU" HINGE_MARGIN="$NS_HINGE_MARGIN" \
      SCORE_KIND="$NS_SCORE_KIND" CROSS_SCORE_NORMALIZATION="$NS_CROSS_SCORE_NORMALIZATION" LOSS_INVERSE_NORM="$NS_LOSS_INVERSE_NORM" LOSS_TRACK_COMPONENTS="$NS_LOSS_TRACK_COMPONENTS" \
      MOMENTUM_MODE=off MOMENTUM_LOOKBACK='' MOMENTUM_ETA=0 MOMENTUM_GAIN_INIT=0 MOMENTUM_DIAGNOSTICS=0 \
      MOMENTUM_INPUT_MODE=off MOMENTUM_INPUT_LOOKBACK='' LRADJ="$NS_LRADJ" VALIDATION_MODE="$NS_VALIDATION_MODE" \
      RUN_TAG="$EXPERIMENT_TAG/$path" OUT_ROOT="$root" RESULT_ROOT="$NS_RESULT_ROOT/$path" LOG_ROOT="$NS_LOG_ROOT/$path" \
      PARTITION="$NS_PARTITION" GRES="$NS_GRES" TIME_LIMIT="$NS_TIME_LIMIT" NODELIST="$NS_NODELIST" CPUS_PER_TASK="$NS_CPUS_PER_TASK" MEMORY="$NS_MEMORY" \
      MAX_CONCURRENT_JOBS="$NS_MAX_CONCURRENT_JOBS" ARRAY_CHUNK_SIZE="$NS_ARRAY_CHUNK_SIZE" MANTER_CSV="$NS_MANTER_CSV" B3_AFTER_JOB="$dependency" \
      "${mix_env[@]}" bash "$SUPER_SELF" --launch-one "${BRANCH_MODES[index]}" "$action"
}

print_exports() {
    local name
    for name in TFB_ROOT EXPERIMENT_TAG MODEL TRAIN_VALUES LOOKBACKS_VALUES HORIZON_VALUES \
        NS_RANKING_PROTOCOL K_LOSS_VALUES RANK_WINDOWS_VALUES LAMBDAS_VALUES SEEDS_VALUES TEMPORAL_LOSSES_VALUES \
        CROSS_LOSSES_VALUES RANK_MIX_WEIGHT_SPECS RANK_MIX_MODES RANK_MIX_REGIONS RANK_MIX_FRACTION \
        RANK_MIX_INCLUDE_ABLATIONS RANK_MIX_ABLATION_SPECS RANK_MIX_ABLATION_LAMBDAS KERNEL_SPECS NS_MASK_A_VALUES NS_MASK_MODE \
        NS_MODEL_DEFAULTS_JSON NS_BATCH_SIZE NS_NUM_EPOCHS NS_PATIENCE NS_LEARNING_RATE NS_LRADJ NS_TRAIN_RATIO_IN_TV \
        NS_VALIDATION_MODE NS_STRIDE NS_NUM_ROLLINGS NS_CROSS_SCALE NS_LISTNET_TAU NS_LISTMLE_TAU NS_APPROXNDCG_TAU \
        NS_HUBER_DELTA NS_SCORE_KIND NS_CROSS_SCORE_NORMALIZATION NS_LOSS_INVERSE_NORM NS_LOSS_TRACK_COMPONENTS \
        NS_PARTITION NS_GRES NS_NODELIST NS_TIME_LIMIT NS_CPUS_PER_TASK NS_MEMORY NS_MAX_CONCURRENT_JOBS \
        NS_ARRAY_CHUNK_SIZE NS_PARALLEL_BRANCHES NS_AFTER_JOB NS_RESUME_CHAIN_OLD_JOBS NS_ONLY_BRANCHES NS_MANTER_CSV \
        NS_PYTHON_BIN NS_VENV_PATH NS_RETURNS_FILE PRED_ROOT NS_LOG_ROOT NS_RESULT_ROOT BACKTEST_ROOT \
        K_EVAL_VALUES NS_MARKET_DIR NS_STRATEGIES_FILE NS_CPU_WORKERS NS_CPU_N_LOTES NS_CPU_MEMORY NS_SLIPPAGE_BPS NS_COST_BPS; do
        printf 'export %s=%q\n' "$name" "${!name}"
    done
    printf '\nbash scripts/run_rankmix_robustez.sh --plan\nbash scripts/run_rankmix_robustez.sh --train\n'
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    case "${1:---help}" in
      --print-exports) [[ $# == 1 ]] || die 'Argumentos excedentes.'; print_exports;;
      --list-branches) [[ $# == 1 ]] || die 'Argumentos excedentes.'; grid
        for i in "${!BRANCHES[@]}"; do selected "${BRANCHES[i]}" || continue; printf '%s | %s | %s\n' "${BRANCHES[i]}" "${COUNTS[i]}" "${BRANCH_PATHS[i]}"; done;;
      *) main "$@";;
    esac
fi
