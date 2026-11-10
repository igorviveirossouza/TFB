#!/usr/bin/env bash
# Copie este arquivo para scripts/ do TFB, branch experiment/b3-loss-clean-v1.
# Grade editavel: iTransformer, cabeças compartilhadas/separadas e RankMix.
# Requer run_ns_rankmix_isolado.sh e o executor B3 com suporte a task heads.
# Evolucao do piloto publicado em 2b94cf9a13ba5ab47c38299f51ac5184c118bd3d.
set -Eeuo pipefail

PILOT_SELF="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
TFB_ROOT="${TFB_ROOT:-$(cd -- "$(dirname -- "$PILOT_SELF")/.." && pwd)}"

# O worker consome o JSON congelado na submissao. Nao recria a grade nem
# substitui os hiperparametros pela configuracao atual do terminal.
if [[ "${1:-}" == --on-node ]]; then
    [[ $# -ge 3 && ( "$2" == temporal || "$2" == composto ) && "$3" == worker ]] || {
        printf 'Invocacao de worker invalida.\n' >&2; exit 2;
    }
    pilot_mode=$2; shift 2
    [[ -n "${B3_INLINE_CONFIG:-}" && -n "${B3_CONFIG_SOURCE:-}" ]] || {
        printf 'Worker sem configuracao congelada.\n' >&2; exit 2;
    }
    if [[ -n "${VENV_PATH:-}" && -f "$VENV_PATH" ]]; then source "$VENV_PATH"; fi
    cd -- "$TFB_ROOT"
    exec "${PYTHON_BIN:?PYTHON_BIN ausente}" "$TFB_ROOT/scripts/b3_experiment.py" "$pilot_mode" "$@"
fi

# Um ramo recebe as variaveis nativas e adiciona somente os defaults da
# arquitetura. O mesmo entrypoint e usado pelo Slurm via B3_WORKER_LAUNCHER.
if [[ "${1:-}" == --launch-one ]]; then
    [[ $# == 3 && ( "$2" == temporal || "$2" == composto ) ]] || {
        printf 'Invocacao de ramo invalida.\n' >&2; exit 2;
    }
    EXPERIMENT=$2; shift 2
    case "$1" in submit|--dry-run|--print-config) ;; *) printf 'Acao interna invalida.\n' >&2; exit 2;; esac
    read -r -a TRAIN_PERCENTAGES <<< "${TRAIN_PERCENTAGES:?}"
    read -r -a MODELS <<< "${MODELS:?}"
    read -r -a LOOKBACKS <<< "${LOOKBACKS:?}"
    read -r -a HORIZONS <<< "${HORIZONS:?}"
    read -r -a TRADE_WINDOWS <<< "${TRADE_WINDOWS:?}"
    read -r -a CROSS_LOSSES <<< "${CROSS_LOSSES-}"
    read -r -a LAMBDAS <<< "${LAMBDAS-}"
    read -r -a DELTAS <<< "${DELTAS:?}"
    read -r -a CROSS_MASK <<< "${CROSS_MASK:?}"
    MODEL_DEFAULTS_JSON=$("$PYTHON_BIN" - "$BATCH_SIZE" "$ITR_TASK_HEAD_MODE" "$ITR_RANK_WINDOWS" <<'PY'
import json, sys
print(json.dumps({"iTransformer": dict(
    batch_size=int(sys.argv[1]), d_model=128, d_ff=256, e_layers=2,
    n_heads=8, dropout=0.1, norm=True, lradj="type1",
    task_head_mode=sys.argv[2], rank_windows=[int(k) for k in sys.argv[3].split()],
    ranking_construction="restored_daily_projection_v1")}))
PY
)
    B3_CONFIG_SOURCE="$PILOT_SELF"
    export B3_WORKER_LAUNCHER="$PILOT_SELF"
    source "$TFB_ROOT/scripts/b3_experiment_env.sh"
    exec "$PYTHON_BIN" "$TFB_ROOT/scripts/b3_experiment.py" "$EXPERIMENT" "$@"
fi

# Defaults editaveis por export. As janelas de ranking sao supervisionadas
# juntas no mesmo treino, por origem, filtradas por K<=H no executor.
MODEL="${MODEL:-iTransformer}"
TRAIN_VALUES="${TRAIN_VALUES:-80}"
LOOKBACKS_VALUES="${LOOKBACKS_VALUES:-32 256 512}"
HORIZON_VALUES="${HORIZON_VALUES:-1 25}"
K_LOSS_VALUES="${K_LOSS_VALUES:-1}"
RANK_WINDOWS_VALUES="${RANK_WINDOWS_VALUES:-1 5 10 15 20 25}"
K_EVAL_VALUES="${K_EVAL_VALUES:-1 5 10 15 20 25}"
SEEDS_VALUES="${SEEDS_VALUES:-${NS_SEED:-2026}}"
TASK_HEAD_MODES="${TASK_HEAD_MODES:-shared separate}"
TEMPORAL_LOSSES_VALUES="${TEMPORAL_LOSSES_VALUES:-mse}"
CROSS_LOSSES_VALUES="${CROSS_LOSSES_VALUES:-rankmix_proximity}"
LAMBDAS_VALUES="${LAMBDAS_VALUES:-0 0.1 0.5 0.8 1}"
RANK_MIX_WEIGHTS="${RANK_MIX_WEIGHTS:-0.5 0.5 0}"
RANK_MIX_MODES="${RANK_MIX_MODES:-global regional}"
RANK_MIX_REGIONS="${RANK_MIX_REGIONS:-top bottom top}"
RANK_MIX_FRACTION="${RANK_MIX_FRACTION:-0.3333333333333333}"
KERNEL_SPECS="${KERNEL_SPECS:-${NS_MASK_TYPE:-cauchy}}"
NS_MASK_A="${NS_MASK_A:-0.025}"
NS_MASK_MODE="${NS_MASK_MODE:-natural}"
NS_BATCH_SIZE="${NS_BATCH_SIZE:-16}"
NS_NUM_EPOCHS="${NS_NUM_EPOCHS:-100}"
NS_PATIENCE="${NS_PATIENCE:-10}"
NS_LEARNING_RATE="${NS_LEARNING_RATE:-0.0001}"
NS_LRADJ="${NS_LRADJ:-type1}"
NS_PARALLEL_BRANCHES="${NS_PARALLEL_BRANCHES:-1}"

# Recursos e destinos podem ser sobrescritos no terminal.
EXPERIMENT_TAG="${EXPERIMENT_TAG:-isolados/itransformer_rankmix_heads_t80_v2}"
NS_PARTITION="${NS_PARTITION:-gorgonas_dev}"
NS_GRES="${NS_GRES-}"
NS_NODELIST="${NS_NODELIST:-}"
NS_TIME_LIMIT="${NS_TIME_LIMIT:-01:30:00}"
NS_MAX_CONCURRENT_JOBS="${NS_MAX_CONCURRENT_JOBS:-2}"
NS_CPUS_PER_TASK="${NS_CPUS_PER_TASK:-3}"
NS_MEMORY="${NS_MEMORY:-}"

[[ -f "$TFB_ROOT/scripts/run_ns_rankmix_isolado.sh" ]] || {
    printf 'Launcher rankmix ausente em %s/scripts. Atualize o checkout TFB.\n' "$TFB_ROOT" >&2; exit 2;
}
source "$TFB_ROOT/scripts/run_ns_rankmix_isolado.sh"
# Renomeia somente funcoes conhecidas do modulo local, nunca texto do usuario.
eval "$(declare -f grid | sed '1s/^grid /itr_rankmix_grid /')"
eval "$(declare -f branch_fingerprint | sed '1s/^branch_fingerprint /itr_rankmix_fingerprint /')"
eval "$(declare -f backtests | sed '1s/^backtests /itr_rankmix_backtests /')"

usage() {
    cat <<'HELP'
Instalacao: copie este arquivo para scripts/ do checkout TFB atualizado.
Uso: bash scripts/run_itransformer_rankmix_piloto.sh --plan
     bash scripts/run_itransformer_rankmix_piloto.sh --dry-run
     bash scripts/run_itransformer_rankmix_piloto.sh --train
     bash scripts/run_itransformer_rankmix_piloto.sh --backtest [planejar|submeter|consolidar]

Defaults: iTransformer, LB32/256/512, H1/25, K_loss1, treino80, seed2026.
Heads shared/separate; RankMix global/regional, pesos [.5 .5 0].
Lambda 0/.1/.5/.8/1; rankmix_proximity; Cauchy natural A.025.
Regional: top/bottom/top, fracao1/3; momentum externo.
102 treinos = 6 controles MSE shared + 72 mistos + 24 rankings puros.
Lambda0 e compartilhado entre heads, modos e losses de ranking;
lambda1 nao repete por loss temporal. Original rankmix e opcional via export.
RANK_WINDOWS_VALUES='1 5 10 15 20 25': todas as janelas no MESMO treino,
ancoradas na origem e filtradas por K<=H, como no estudo NS restaurado.
K_LOSS_VALUES deve ser 1 neste estudo. K_EVAL_VALUES so controla o backtest.
Arquitetura: d_model128, d_ff256, e_layers2, n_heads8, dropout.1.
Batch16; epochs100; patience10; lr1e-4; validacao ordenada; seed2026.

Grade editavel: TRAIN_VALUES, LOOKBACKS_VALUES, HORIZON_VALUES,
RANK_WINDOWS_VALUES, K_EVAL_VALUES, SEEDS_VALUES, TASK_HEAD_MODES,
LAMBDAS_VALUES, TEMPORAL_LOSSES_VALUES, CROSS_LOSSES_VALUES,
RANK_MIX_WEIGHTS, RANK_MIX_MODES, RANK_MIX_REGIONS, RANK_MIX_FRACTION,
KERNEL_SPECS (ou NS_MASK_TYPE), NS_MASK_A, NS_MASK_MODE, NS_ONLY_BRANCHES.
Treino/normalizacao: os parametros NS_* seguem o launcher rankmix nativo.
Use uma EXPERIMENT_TAG nova para esta grade, distinta do piloto fixo anterior.

--plan nao depende de Python/dados/Slurm. --dry-run valida o executor B3.
--train so funciona no cluster e preserva marcas de ramos ja submetidos;
marca de submissao nao comprova conclusao. Confira os logs antes de retomar.
Padrao: gorgonas_dev, GRES vazio, 01:30:00, ate 2 tarefas por array ativo.
Global/regional fazem parte da mesma grade. NS_PARALLEL_BRANCHES=1 (padrao)
permite arrays simultaneos; =0 encadeia. O limite e POR ARRAY, nao global.
NS_GRES vazio nao solicita GPU; para GPU, configure particao e GRES.
Recursos editaveis: NS_PARTITION, NS_GRES, NS_NODELIST, NS_TIME_LIMIT,
NS_MAX_CONCURRENT_JOBS, NS_CPUS_PER_TASK, NS_MEMORY, NS_AFTER_JOB.
Caminhos: TFB_ROOT, NS_PYTHON_BIN, NS_VENV_PATH, NS_RETURNS_FILE,
EXPERIMENT_TAG, PRED_ROOT, BACKTEST_ROOT, NS_MARKET_DIR, NS_STRATEGIES_FILE.
O backtest nativo so e permitido para ramos shared selecionados.
Ramos separate devem ser avaliados pelos ranking_scores no analisador de heads;
suas previsoes temporais auxiliares nao representam o sinal principal.
HELP
}

grid() {
    [[ "$MODEL" == iTransformer ]] || die 'Este launcher suporta MODEL=iTransformer.'
    integer_list RANK_WINDOWS_VALUES "$RANK_WINDOWS_VALUES"
    local requested_branches=$NS_ONLY_BRANCHES requested_losses=$CROSS_LOSSES_VALUES
    local head loss i field value branch h k valid
    local -a heads=() rank_windows=() wanted_losses=() target_heads=()
    local -a itr_fields=(BRANCHES BRANCH_MODES BRANCH_TEMPORALS BRANCH_SEEDS
        BRANCH_LAMBDAS BRANCH_LOSSES MASKS COUNTS BRANCH_PATHS BRANCH_MIX_MODES)
    local -A seen=() loss_requested=() branch_exists=()
    read -r -a heads <<< "$TASK_HEAD_MODES"
    ((${#heads[@]})) || die 'TASK_HEAD_MODES vazio.'
    for head in "${heads[@]}"; do
        [[ ( "$head" == shared || "$head" == separate ) && ! ${seen[$head]+yes} ]] || die "Head invalida/repetida: $head"
        seen[$head]=1
    done
    read -r -a wanted_losses <<< "$requested_losses"
    ((${#wanted_losses[@]})) || die 'CROSS_LOSSES_VALUES vazio.'
    for loss in "${wanted_losses[@]}"; do
        [[ ( "$loss" == rankmix || "$loss" == rankmix_proximity ) && ! ${loss_requested[$loss]+yes} ]] || die "Loss invalida/repetida: $loss"
        loss_requested[$loss]=1
    done
    # O construtor nativo valida ambos os comparadores. Depois conservamos
    # somente as losses solicitadas; nenhum comparador e acrescentado ao treino.
    NS_ONLY_BRANCHES=''; CROSS_LOSSES_VALUES='rankmix rankmix_proximity'
    itr_rankmix_grid
    NS_ONLY_BRANCHES=$requested_branches; CROSS_LOSSES_VALUES=$requested_losses
    [[ ${#windows[@]} == 1 && "${windows[0]}" == 1 ]] || die 'K_LOSS_VALUES deve ser 1; use RANK_WINDOWS_VALUES para as janelas supervisionadas.'
    read -r -a rank_windows <<< "$RANK_WINDOWS_VALUES"
    RANK_WINDOWS_VALUES="${rank_windows[*]}"
    for h in "${horizons[@]}"; do
        valid=0
        for k in "${rank_windows[@]}"; do ((k<=h)) && valid=1; done
        ((valid)) || die "H=$h sem janela em RANK_WINDOWS_VALUES."
    done
    for field in "${itr_fields[@]}"; do declare -g -a "ITR_$field=()"; done
    BRANCH_HEAD_MODES=()
    for i in "${!BRANCHES[@]}"; do
        if [[ "${BRANCH_MODES[i]}" == temporal ]]; then
            target_heads=(shared)
        else
            [[ ${loss_requested[${BRANCH_LOSSES[i]}]+yes} ]] || continue
            target_heads=("${heads[@]}")
        fi
        for head in "${target_heads[@]}"; do
            for field in "${itr_fields[@]}"; do
                local -n src="$field" dst="ITR_$field"
                value=${src[i]}
                [[ "$field" != BRANCHES ]] || value="head_${head}_$value"
                [[ "$field" != BRANCH_PATHS ]] || value="head=$head/$value"
                dst+=("$value")
            done
            BRANCH_HEAD_MODES+=("$head")
        done
    done
    for field in "${itr_fields[@]}"; do
        local -n src="ITR_$field" dst="$field"
        dst=("${src[@]}")
    done
    for branch in "${BRANCHES[@]}"; do
        [[ ! ${branch_exists[$branch]+yes} ]] || die "Ramo duplicado: $branch"
        branch_exists[$branch]=1
    done
    for branch in $requested_branches; do [[ ${branch_exists[$branch]+yes} ]] || die "Ramo inexistente: $branch"; done
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
    printf 'iTransformer: treino=%s; LB=%s; H=%s; K_loss=%s; seeds=%s.\n' "$TRAIN_VALUES" "$LOOKBACKS_VALUES" "$HORIZON_VALUES" "$K_LOSS_VALUES" "$SEEDS_VALUES"
    printf 'Heads: %s; modos: %s; pesos=[%s]; lambdas=%s; losses=%s.\n' "$TASK_HEAD_MODES" "$RANK_MIX_MODES" "$RANK_MIX_WEIGHTS" "$LAMBDAS_VALUES" "$CROSS_LOSSES_VALUES"
    printf 'RANK_WINDOWS_VALUES=[%s]: supervisao conjunta por origem, filtrada por K<=H.\n' "$RANK_WINDOWS_VALUES"
    printf 'd_model128; d_ff256; e_layers2; n_heads8; batch%s; epochs%s; patience%s.\n' "$NS_BATCH_SIZE" "$NS_NUM_EPOCHS" "$NS_PATIENCE"
    local i
    for i in "${!BRANCHES[@]}"; do
        selected "${BRANCHES[i]}" || continue
        printf '%s | %s treinos | %s\n' "${BRANCHES[i]}" "${COUNTS[i]}" "${MASKS[i]}"
    done
    printf 'Controles temporais shared: %s; mistos: %s; ranking puro: %s.\n' "$CONTROL_TOTAL" "$MIXED_TOTAL" "$PURE_TOTAL"
    printf 'TOTAL: %s treinos unicos; %s ramos selecionados.\n' "$TOTAL" "$ACTIVE_BRANCHES"
    printf 'Slurm: particao=%s; GRES=%s; limite=%s; ate %s tarefas por array.\n' "$NS_PARTITION" "${NS_GRES:-vazio}" "$NS_TIME_LIMIT" "$NS_MAX_CONCURRENT_JOBS"
    if [[ "$NS_PARALLEL_BRANCHES" == 0 ]]; then
        printf 'Ramos encadeados: ambos os modos na mesma grade, ate %s tarefas simultaneas.\n' "$NS_MAX_CONCURRENT_JOBS"
    else
        printf 'Arrays paralelos: teto de %s tarefas simultaneas (%s arrays x %s).\n' "$((ACTIVE_BRANCHES*NS_MAX_CONCURRENT_JOBS))" "$ACTIVE_BRANCHES" "$NS_MAX_CONCURRENT_JOBS"
    fi
    printf 'Previsoes: %s\nBacktests: %s\n' "$PRED_ROOT" "$BACKTEST_ROOT"
}

branch_fingerprint() {
    local index=$1
    printf '%s\n' "$(itr_rankmix_fingerprint "$index")" "${BRANCH_HEAD_MODES[index]}" "$RANK_WINDOWS_VALUES" \
      'iTransformer:128:256:2:8:0.1:restored_daily_projection_v1' | sha256sum | cut -d' ' -f1
}

backtests() {
    local i
    for i in "${!BRANCHES[@]}"; do
        selected "${BRANCHES[i]}" || continue
        [[ "${BRANCH_HEAD_MODES[i]}" != separate ]] || die 'Backtest generico nao aceita separate: use ranking_scores no analisador de heads; a saida temporal e auxiliar.'
    done
    itr_rankmix_backtests "$@"
}

# Mesmo contrato do launcher rankmix nativo; apenas o entrypoint/arquitetura
# sao especificos deste arquivo. Treino, losses, exportacao e Slurm sao nativos.
train_launcher() {
    local index=$1 action=$2 dependency=${3:-} tag root
    local -a mix_env=()
    root=$(branch_root "$index"); tag="$EXPERIMENT_TAG/${BRANCH_PATHS[index]}"
    if [[ "${BRANCH_MODES[index]}" == composto ]]; then
        mix_env=("RANK_MIX_WEIGHTS=$RANK_MIX_WEIGHTS" "RANK_MIX_MODE=${BRANCH_MIX_MODES[index]}" "RANK_MIX_REGIONS=$RANK_MIX_REGIONS" "RANK_MIX_FRACTION=$RANK_MIX_FRACTION")
    fi
    env -u RANK_MIX_WEIGHTS -u RANK_MIX_MODE -u RANK_MIX_REGIONS -u RANK_MIX_FRACTION -u B3_INLINE_CONFIG -u B3_CONFIG -u B3_CONFIG_SOURCE -u B3_WORKER_LAUNCHER -u EXPERIMENT \
      TFB_ROOT="$TFB_ROOT" PYTHON_BIN="$NS_PYTHON_BIN" VENV_PATH="$NS_VENV_PATH" \
      ITR_TASK_HEAD_MODE="${BRANCH_HEAD_MODES[index]}" ITR_RANK_WINDOWS="$RANK_WINDOWS_VALUES" \
      DATASET_PATH="$NS_RETURNS_FILE" SOURCE_DATA_KIND=log_return \
      TRAIN_PERCENTAGES="$TRAIN_VALUES" MODELS="$MODEL" LOOKBACKS="$LOOKBACKS_VALUES" HORIZONS="$HORIZON_VALUES" \
      TRADE_WINDOWS="$K_LOSS_VALUES" CROSS_LOSSES="${BRANCH_LOSSES[index]}" LAMBDAS="${BRANCH_LAMBDAS[index]}" DELTAS="$DELTAS_VALUES" \
      CROSS_MASK="${MASKS[index]:-cauchy $NS_MASK_A $NS_MASK_MODE}" \
      BATCH_SIZE="$NS_BATCH_SIZE" NUM_EPOCHS="$NS_NUM_EPOCHS" PATIENCE="$NS_PATIENCE" LEARNING_RATE="$NS_LEARNING_RATE" \
      TEMPORAL_LOSS="${BRANCH_TEMPORALS[index]}" HUBER_DELTA="$NS_HUBER_DELTA" SPEARMAN_TAU="$NS_SPEARMAN_TAU" TRAIN_RATIO_IN_TV="$NS_TRAIN_RATIO_IN_TV" SEED="${BRANCH_SEEDS[index]}" STRIDE="$NS_STRIDE" NUM_ROLLINGS="$NS_NUM_ROLLINGS" \
      CROSS_SCALE="$NS_CROSS_SCALE" RANKNET_ALPHA="$NS_RANKNET_ALPHA" LISTNET_TAU="$NS_LISTNET_TAU" LISTMLE_TAU="$NS_LISTMLE_TAU" APPROXNDCG_TAU="$NS_APPROXNDCG_TAU" HINGE_MARGIN="$NS_HINGE_MARGIN" \
      SCORE_KIND="$NS_SCORE_KIND" CROSS_SCORE_NORMALIZATION="$NS_CROSS_SCORE_NORMALIZATION" LOSS_INVERSE_NORM="$NS_LOSS_INVERSE_NORM" LOSS_TRACK_COMPONENTS="$NS_LOSS_TRACK_COMPONENTS" \
      MOMENTUM_MODE=off MOMENTUM_LOOKBACK='' MOMENTUM_ETA=0 MOMENTUM_GAIN_INIT=0 MOMENTUM_DIAGNOSTICS=0 \
      MOMENTUM_INPUT_MODE=off MOMENTUM_INPUT_LOOKBACK='' LRADJ="$NS_LRADJ" VALIDATION_MODE="$NS_VALIDATION_MODE" \
      RUN_TAG="$tag" OUT_ROOT="$root" RESULT_ROOT="/snfs2/igor.viveiros/result/$tag" LOG_ROOT="$TFB_ROOT/logs/$tag" \
      PARTITION="$NS_PARTITION" GRES="$NS_GRES" TIME_LIMIT="$NS_TIME_LIMIT" NODELIST="$NS_NODELIST" CPUS_PER_TASK="$NS_CPUS_PER_TASK" MEMORY="$NS_MEMORY" \
      MAX_CONCURRENT_JOBS="$NS_MAX_CONCURRENT_JOBS" ARRAY_CHUNK_SIZE="$NS_ARRAY_CHUNK_SIZE" MANTER_CSV="$NS_MANTER_CSV" B3_AFTER_JOB="$dependency" \
      "${mix_env[@]}" bash "$PILOT_SELF" --launch-one "${BRANCH_MODES[index]}" "$action"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    if [[ $# == 0 ]]; then set -- --plan; fi
    main "$@"
fi
