#!/usr/bin/env bash
# Copie este arquivo para scripts/ do TFB, branch experiment/b3-loss-clean-v1.
# Piloto fixo: iTransformer; LB 256/512; H25; K_loss1; seed2026; 6 treinos.
# Requer os launchers rankmix_refinamento_fino e o executor B3 nativos.
# Base verificada: TFB commit 8f6b607e99884ba4eab77900859539166e1acfe5.
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
    MODEL_DEFAULTS_JSON='{"iTransformer":{"batch_size":16,"d_model":128,"d_ff":256,"e_layers":2,"n_heads":8,"dropout":0.1,"norm":true,"lradj":"type1"}}'
    B3_CONFIG_SOURCE="$PILOT_SELF"
    export B3_WORKER_LAUNCHER="$PILOT_SELF"
    source "$TFB_ROOT/scripts/b3_experiment_env.sh"
    exec "$PYTHON_BIN" "$TFB_ROOT/scripts/b3_experiment.py" "$EXPERIMENT" "$@"
fi

# Protocolo cientifico fixo. Exports antigos de outros experimentos nao
# alteram losses, mistura, escala, seeds ou quantidade de treinos deste piloto.
MODEL=iTransformer
TRAIN_VALUES=80
LOOKBACKS_VALUES='256 512'
HORIZON_VALUES=25
K_LOSS_VALUES=1
K_EVAL_VALUES='1 5 10 15 20 25'
SEEDS_VALUES=2026
TEMPORAL_LOSSES_VALUES=mse
CROSS_LOSSES_VALUES='rankmix rankmix_proximity'
DELTAS_VALUES=0
RANK_MIX_INCLUDE_REFERENCES=0
RANK_MIX_FOCUS_SPECS='global_final|global|0.25 0.75 0|0.1|0.005;regional_final|regional|0 0.9 0.1|0.5|0.01'
RANK_MIX_REGIONS='top bottom top'
RANK_MIX_FRACTION=0.3333333333333333
KERNEL_SPECS=cauchy
NS_MASK_MODE=natural
NS_ONLY_BRANCHES=''
NS_BATCH_SIZE=16
NS_NUM_EPOCHS=100
NS_PATIENCE=10
NS_LEARNING_RATE=0.0001
NS_TRAIN_RATIO_IN_TV=0.875
NS_STRIDE=1
NS_NUM_ROLLINGS=48000
NS_CROSS_SCALE=1
NS_RANKNET_ALPHA=1
NS_LISTNET_TAU=1
NS_LISTMLE_TAU=1
NS_APPROXNDCG_TAU=1
NS_HINGE_MARGIN=0.1
NS_HUBER_DELTA=0.5
NS_SPEARMAN_TAU=1
NS_SCORE_KIND=simple_return
NS_CROSS_SCORE_NORMALIZATION=zscore
NS_LOSS_INVERSE_NORM=true
NS_LOSS_TRACK_COMPONENTS=true
NS_VALIDATION_MODE=ordered_sample_mean
NS_LRADJ=type1
MOMENTUM_MODE=off
MOMENTUM_INPUT_MODE=off
MOMENTUM_LOOKBACK=''
MOMENTUM_INPUT_LOOKBACK=''
MOMENTUM_ETA=0
MOMENTUM_GAIN_INIT=0
MOMENTUM_DIAGNOSTICS=0
NS_PARALLEL_BRANCHES=0
NS_SLIPPAGE_BPS=0
NS_COST_BPS=0

# Recursos e destinos podem ser sobrescritos no terminal.
EXPERIMENT_TAG="${EXPERIMENT_TAG:-isolados/itransformer_rankmix_piloto_t80_h25_k1_s2026_v1}"
NS_PARTITION="${NS_PARTITION:-gorgonas}"
NS_GRES="${NS_GRES-gpu:1}"
NS_NODELIST="${NS_NODELIST:-}"
NS_TIME_LIMIT="${NS_TIME_LIMIT:-01:30:00}"
NS_MAX_CONCURRENT_JOBS="${NS_MAX_CONCURRENT_JOBS:-2}"
NS_CPUS_PER_TASK="${NS_CPUS_PER_TASK:-3}"
NS_MEMORY="${NS_MEMORY:-}"

[[ -f "$TFB_ROOT/scripts/run_ns_rankmix_refinamento_fino.sh" ]] || {
    printf 'Launcher rankmix fino ausente em %s/scripts. Atualize o checkout TFB.\n' "$TFB_ROOT" >&2; exit 2;
}
source "$TFB_ROOT/scripts/run_ns_rankmix_refinamento_fino.sh"

usage() {
    cat <<'HELP'
Instalacao: copie este arquivo para scripts/ do checkout TFB atualizado.
Uso: bash scripts/run_itransformer_rankmix_piloto.sh --plan
     bash scripts/run_itransformer_rankmix_piloto.sh --dry-run
     bash scripts/run_itransformer_rankmix_piloto.sh --train
     bash scripts/run_itransformer_rankmix_piloto.sh --backtest [planejar|submeter|consolidar]

Seis treinos: iTransformer, LB256/512, H25, K_loss1, treino80, seed2026.
  MSE pura.
  Global:   [ListNet,ListMLE,ApproxNDCG]=[.25,.75,0], lambda .1, Cauchy A .005.
  Regional: [ListNet,ListMLE,ApproxNDCG]=[0,.9,.1], lambda .5, Cauchy A .01.
Kernels naturais; regional: top/bottom/top, fracao1/3; momentum externo.
Sem controles extras, sem ranking puro e sem repeticao de treino por K_eval.
Arquitetura: d_model128, d_ff256, e_layers2, n_heads8, dropout.1.
Batch16; epochs100; patience10; lr1e-4; validacao ordenada; seed2026.

--plan nao depende de Python/dados/Slurm. --dry-run valida o executor B3.
--train so funciona no cluster e preserva marcas de ramos ja submetidos;
marca de submissao nao comprova conclusao. Confira os logs antes de retomar.
Padrao: gorgonas, GRES gpu:1, 01:30:00, ate 2 tarefas por array ativo.
Ramos encadeados. NS_GRES vazio explicitamente nao solicita GPU.
Recursos editaveis: NS_PARTITION, NS_GRES, NS_NODELIST, NS_TIME_LIMIT,
NS_MAX_CONCURRENT_JOBS, NS_CPUS_PER_TASK, NS_MEMORY, NS_AFTER_JOB.
Caminhos: TFB_ROOT, NS_PYTHON_BIN, NS_VENV_PATH, NS_RETURNS_FILE,
EXPERIMENT_TAG, PRED_ROOT, BACKTEST_ROOT, NS_MARKET_DIR, NS_STRATEGIES_FILE.
O backtest usa os scripts nativos; exige bundle de mercado e estrategias.
HELP
}

pilot_select() {
    # Deixa o construtor nativo validar e nomear todos os ramos. Seleciona
    # apenas os tres objetivos por seus metadados, sem hashes fixos no codigo.
    NS_ONLY_BRANCHES=''
    grid
    local i
    local -a chosen=()
    for i in "${!BRANCHES[@]}"; do
        if [[ "${BRANCH_MODES[i]}" == temporal ]] ||
           [[ "${BRANCH_LOSSES[i]}" == rankmix_proximity && "${BRANCH_LAMBDAS[i]}" != 1 ]]; then
            chosen+=("${BRANCHES[i]}")
        fi
    done
    NS_ONLY_BRANCHES="${chosen[*]}"
    grid
    [[ "$TOTAL" == 6 && "$CONTROL_TOTAL" == 2 && "$MIXED_TOTAL" == 4 && "$PURE_TOTAL" == 0 && "$ACTIVE_BRANCHES" == 3 ]] || die 'Grade inesperada; nenhum job foi submetido.'
}

plan() {
    pilot_select
    printf 'Piloto iTransformer: 6 treinos; treino80; LB256/512; H25; K_loss1; seed2026.\n'
    printf 'MSE: 2; global [.25 .75 0], lambda.1, A.005: 2; regional [0 .9 .1], lambda.5, A.01: 2.\n'
    printf 'd_model128; d_ff256; e_layers2; n_heads8; batch16; epochs100; patience10.\n'
    local i
    for i in "${!BRANCHES[@]}"; do
        selected "${BRANCHES[i]}" || continue
        printf '%s | %s treinos | %s\n' "${BRANCHES[i]}" "${COUNTS[i]}" "${MASKS[i]}"
    done
    printf 'TOTAL: %s treinos unicos; %s ramos selecionados.\n' "$TOTAL" "$ACTIVE_BRANCHES"
    printf 'Slurm: particao=%s; GRES=%s; limite=%s; ate %s tarefas simultaneas.\n' "$NS_PARTITION" "${NS_GRES:-vazio}" "$NS_TIME_LIMIT" "$NS_MAX_CONCURRENT_JOBS"
    printf 'Previsoes: %s\nBacktests: %s\n' "$PRED_ROOT" "$BACKTEST_ROOT"
}

# Mesmo contrato do launcher rankmix nativo; apenas o entrypoint/arquitetura
# sao especificos deste arquivo. Treino, losses, exportacao e Slurm sao nativos.
train_launcher() {
    local index=$1 action=$2 dependency=${3:-} tag root
    local -a mix_env=()
    root=$(branch_root "$index"); tag="$EXPERIMENT_TAG/${BRANCH_PATHS[index]}"
    if [[ "${BRANCH_MODES[index]}" == composto ]]; then
        mix_env=("RANK_MIX_WEIGHTS=${BRANCH_WEIGHTS[index]}" "RANK_MIX_MODE=${BRANCH_MIX_MODES[index]}" "RANK_MIX_REGIONS=$RANK_MIX_REGIONS" "RANK_MIX_FRACTION=$RANK_MIX_FRACTION")
    fi
    env -u RANK_MIX_WEIGHTS -u RANK_MIX_MODE -u RANK_MIX_REGIONS -u RANK_MIX_FRACTION -u B3_INLINE_CONFIG -u B3_CONFIG -u B3_CONFIG_SOURCE -u B3_WORKER_LAUNCHER -u EXPERIMENT \
      TFB_ROOT="$TFB_ROOT" PYTHON_BIN="$NS_PYTHON_BIN" VENV_PATH="$NS_VENV_PATH" \
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
      RUN_TAG="$tag" OUT_ROOT="$root" RESULT_ROOT="/snfs2/igor.viveiros/result/$tag" LOG_ROOT="$TFB_ROOT/logs/$tag" \
      PARTITION="$NS_PARTITION" GRES="$NS_GRES" TIME_LIMIT="$NS_TIME_LIMIT" NODELIST="$NS_NODELIST" CPUS_PER_TASK="$NS_CPUS_PER_TASK" MEMORY="$NS_MEMORY" \
      MAX_CONCURRENT_JOBS="$NS_MAX_CONCURRENT_JOBS" ARRAY_CHUNK_SIZE="$NS_ARRAY_CHUNK_SIZE" MANTER_CSV="$NS_MANTER_CSV" B3_AFTER_JOB="$dependency" \
      "${mix_env[@]}" bash "$PILOT_SELF" --launch-one "${BRANCH_MODES[index]}" "$action"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    if [[ $# == 0 ]]; then set -- --plan; fi
    main "$@"
fi
