#!/usr/bin/env bash
# Experimento de kernels: controle temporal e loss sem kernel treinados uma unica vez.
# Todos os valores deste bloco podem ser sobrescritos com export no terminal.
set -Eeuo pipefail
TFB_ROOT="${TFB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-isolados/ns_kernels_a0.25_t80_v1}"
MODEL="${MODEL:-Nonstationary_Transformer}"
TRAIN_VALUES="${TRAIN_VALUES:-80}"
LOOKBACKS_VALUES="${LOOKBACKS_VALUES:-256 512}"
HORIZON_VALUES="${HORIZON_VALUES:-1 25}"
K_LOSS_VALUES="${K_LOSS_VALUES:-1}"
POSITIVE_LAMBDAS="${POSITIVE_LAMBDAS:-0.5 0.8 1}"
CROSS_LOSSES_VALUES="${CROSS_LOSSES_VALUES:-ranknet ranknet_proximity}"
K_EVAL_VALUES="${K_EVAL_VALUES:-1 5 10 15 20 25}"
DELTAS_VALUES="${DELTAS_VALUES:-0}"
NS_MASK_A="${NS_MASK_A:-0.25}"
NS_MASK_MODE="${NS_MASK_MODE:-natural}"
# Entradas separadas por ponto e virgula ou quebra de linha: nome [forma...].
# A e modo sao compartilhados; para outra escala use outra EXPERIMENT_TAG.
KERNEL_SPECS="${KERNEL_SPECS:-cauchy;student 10;generalized_gaussian 1;generalized_gaussian 2;generalized_gaussian 4;compact 2 0;compact 1 1;compact 2 1;compact 2 2;compact 2 3;cosine;intermediate_band 2}"
NS_ONLY_BRANCHES="${NS_ONLY_BRANCHES:-}" # vazio=todos; ex.: temporal sem_kernel student_10
NS_PYTHON_BIN="${NS_PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"
NS_VENV_PATH="${NS_VENV_PATH:-/sonic_home/igor.viveiros/py310/bin/activate}"
NS_RETURNS_FILE="${NS_RETURNS_FILE:-$TFB_ROOT/dataset/forecasting/b3_daily_tfb_new.csv}"
NS_MARKET_DIR="${NS_MARKET_DIR:-/snfs2/igor.viveiros/analise_resultados/insumos/base_backtest_temporal_001}"
NS_STRATEGIES_FILE="${NS_STRATEGIES_FILE:-$TFB_ROOT/config/estrategias_backtest.json}"
NS_BATCH_SIZE="${NS_BATCH_SIZE:-16}"
NS_NUM_EPOCHS="${NS_NUM_EPOCHS:-100}"
NS_PATIENCE="${NS_PATIENCE:-10}"
NS_LEARNING_RATE="${NS_LEARNING_RATE:-0.0001}"
NS_TEMPORAL_LOSS="${NS_TEMPORAL_LOSS:-mse}"
NS_TRAIN_RATIO_IN_TV="${NS_TRAIN_RATIO_IN_TV:-0.875}"
NS_SEED="${NS_SEED:-2026}"
NS_STRIDE="${NS_STRIDE:-1}"
NS_NUM_ROLLINGS="${NS_NUM_ROLLINGS:-48000}"
NS_CROSS_SCALE="${NS_CROSS_SCALE:-1}"
NS_RANKNET_ALPHA="${NS_RANKNET_ALPHA:-1}"
NS_LISTNET_TAU="${NS_LISTNET_TAU:-1}"
NS_SPEARMAN_TAU="${NS_SPEARMAN_TAU:-}"
NS_LISTMLE_TAU="${NS_LISTMLE_TAU:-}"
NS_APPROXNDCG_TAU="${NS_APPROXNDCG_TAU:-}"
NS_HUBER_DELTA="${NS_HUBER_DELTA:-}"
NS_HINGE_MARGIN="${NS_HINGE_MARGIN:-0.1}"
NS_SCORE_KIND="${NS_SCORE_KIND:-simple_return}"
NS_CROSS_SCORE_NORMALIZATION="${NS_CROSS_SCORE_NORMALIZATION:-zscore}"
NS_LOSS_INVERSE_NORM="${NS_LOSS_INVERSE_NORM:-true}"
NS_LOSS_TRACK_COMPONENTS="${NS_LOSS_TRACK_COMPONENTS:-true}"
MOMENTUM_MODE="${MOMENTUM_MODE:-off}"
MOMENTUM_LOOKBACK="${MOMENTUM_LOOKBACK:-}"
MOMENTUM_ETA="${MOMENTUM_ETA:-0}"
MOMENTUM_GAIN_INIT="${MOMENTUM_GAIN_INIT:-0}"
MOMENTUM_DIAGNOSTICS="${MOMENTUM_DIAGNOSTICS:-0}"
MOMENTUM_INPUT_MODE="${MOMENTUM_INPUT_MODE:-off}"
MOMENTUM_INPUT_LOOKBACK="${MOMENTUM_INPUT_LOOKBACK:-}"
NS_VALIDATION_MODE="${NS_VALIDATION_MODE:-}" # ordered_sample_mean: ordem fixa/media por amostra.
NS_LRADJ="${NS_LRADJ:-}" # Vazio conserva a regra nativa de cada modelo.
NS_PARTITION="${NS_PARTITION:-medusas_dev}"
NS_GRES="${NS_GRES-gpu:1}" # export NS_GRES="" preserva CPU sem solicitar GPU.
NS_NODELIST="${NS_NODELIST:-}"
NS_TIME_LIMIT="${NS_TIME_LIMIT:-01:30:00}"
NS_CPUS_PER_TASK="${NS_CPUS_PER_TASK:-3}"
NS_MEMORY="${NS_MEMORY:-}"
NS_MAX_CONCURRENT_JOBS="${NS_MAX_CONCURRENT_JOBS:-1}"
NS_ARRAY_CHUNK_SIZE="${NS_ARRAY_CHUNK_SIZE:-1000}"
NS_PARALLEL_BRANCHES="${NS_PARALLEL_BRANCHES:-0}" # 0 encadeia ramos; 1 permite todos em paralelo.
NS_AFTER_JOB="${NS_AFTER_JOB:-}" # dependencia Slurm opcional para o primeiro ramo.
NS_MANTER_CSV="${NS_MANTER_CSV:-FALSE}"
NS_CPU_WORKERS="${NS_CPU_WORKERS:-16}"
NS_CPU_N_LOTES="${NS_CPU_N_LOTES:-1}"
NS_CPU_MEMORY="${NS_CPU_MEMORY:-56G}"
NS_SLIPPAGE_BPS="${NS_SLIPPAGE_BPS:-0}"
NS_COST_BPS="${NS_COST_BPS:-0}"
PRED_ROOT="/snfs2/igor.viveiros/previsoes/$EXPERIMENT_TAG"
BACKTEST_ROOT="/snfs2/igor.viveiros/analise_resultados/backtests/$EXPERIMENT_TAG"

die() { printf 'ERRO: %s\n' "$*" >&2; exit 2; }
usage() {
    cat <<HELP
Uso: bash scripts/run_ns_kernels_isolado.sh --plan | --dry-run | --train
     bash scripts/run_ns_kernels_isolado.sh --backtest [planejar|submeter|consolidar]

Default: NS; treino80; LB256/512; H1/25; K_loss1; lambda0/.5/.8/1; A0.25 natural.
12 kernels, RankNet sem kernel uma vez e controle temporal uma vez: 160 treinos.
--plan conta a grade sem Python/dados/Slurm. --dry-run mostra os launchers reais.
--train submete os ramos encadeados (afterany): NS_MAX_CONCURRENT_JOBS por array ativo.
NS_PARALLEL_BRANCHES=1 permite ate numero_de_ramos x NS_MAX_CONCURRENT_JOBS tarefas.
NS_ONLY_BRANCHES seleciona nomes impressos por --plan (ex.: 'temporal sem_kernel cauchy').
--train repetido pula marcas de submissao bem-sucedida; falhas parciais sao bloqueadas.

Todos os valores no bloco inicial aceitam export. Principais:
EXPERIMENT_TAG, MODEL, TRAIN_VALUES, LOOKBACKS_VALUES, HORIZON_VALUES,
K_LOSS_VALUES, POSITIVE_LAMBDAS, CROSS_LOSSES_VALUES, K_EVAL_VALUES, DELTAS_VALUES,
NS_MASK_A, NS_MASK_MODE, KERNEL_SPECS, NS_SEED, NS_BATCH_SIZE, NS_NUM_EPOCHS,
NS_PATIENCE, NS_LEARNING_RATE, NS_PARTITION, NS_GRES, NS_NODELIST,
NS_MAX_CONCURRENT_JOBS, NS_TIME_LIMIT, NS_CPUS_PER_TASK, NS_MEMORY.
NS_SPEARMAN_TAU, NS_LISTMLE_TAU, NS_APPROXNDCG_TAU (default1) e NS_HUBER_DELTA (default0.5) sao opcionais.
MOMENTUM_MODE=off|residual_output, MOMENTUM_LOOKBACK (vazio = LB do modelo),
MOMENTUM_ETA (finito >= 0). Apenas compostos recebem a correcao; controle temporal off.
MOMENTUM_GAIN_INIT (padrao 0), MOMENTUM_DIAGNOSTICS=0|1 e NS_LRADJ tambem aceitam export.
NS_LRADJ vazio preserva o modelo; type1 decai; constant mantem a taxa de aprendizado.
No modo residual, o score final e exportado separadamente em ranking_scores.parquet.
MOMENTUM_INPUT_MODE=off|zero|feature e MOMENTUM_INPUT_LOOKBACK habilitam entrada explicita
(apenas NS). zero usa a mesma arquitetura com feature zerada; feature usa momentum.
Nao combinar entrada explicita com residual_output; a saida continua part.parquet.
NS_VALIDATION_MODE=ordered_sample_mean fixa a ordem e pondera por amostras; vazio preserva legado.

KERNEL_SPECS usa 'kernel parametros;kernel parametros', sem A nem modo.
Ex.: KERNEL_SPECS='cauchy;student 10;generalized_gaussian 1;compact 2 3;cosine;intermediate_band 2'
Aliases gaussian, uniform, laplace, triangular, epanechnikov, biweight, triweight aceitos.
Cada proximity requer sua loss sem kernel na lista CROSS_LOSSES_VALUES.
Formato da mascara enviado: tipo A modo [parametros de forma].

Previsoes: $PRED_ROOT/{temporal,composto/sem_kernel,composto/<kernel>}
R local: --pred-root aponta para $PRED_ROOT (raiz de todos os ramos).
Backtests sao opcionais para o R de IC. O backtest financeiro usa os launchers CPU,
com particao/tempo definidos neles; NS_PARTITION/NS_GRES afetam somente treino.
Retomadas compostas incorporam novas previsoes ao repetir --backtest submeter.
Nao submeta backtest repetido enquanto o job anterior do mesmo ramo estiver ativo.
HELP
}
integer_list() {
    local name=$1 raw=$2 token; local -a tokens
    read -r -a tokens <<< "$raw"
    ((${#tokens[@]})) || die "$name vazio."
    local -A seen=()
    for token in "${tokens[@]}"; do
        [[ "$token" =~ ^[1-9][0-9]*$ ]] || die "$name exige inteiros positivos: $token"
        [[ ! ${seen[$token]+yes} ]] || die "$name tem valor repetido: $token"
        seen[$token]=1
    done
}
selected() { [[ -z "$NS_ONLY_BRANCHES" || " $NS_ONLY_BRANCHES " == *" $1 "* ]]; }
grid() {
    local token spec name label loss h k nplain=0 pairs=0
    [[ "$MOMENTUM_INPUT_MODE" == off || "$MOMENTUM_INPUT_MODE" == zero || "$MOMENTUM_INPUT_MODE" == feature ]] || die 'MOMENTUM_INPUT_MODE deve ser off, zero ou feature.'
    [[ -z "$MOMENTUM_INPUT_LOOKBACK" || "$MOMENTUM_INPUT_LOOKBACK" =~ ^[1-9][0-9]*$ ]] || die 'MOMENTUM_INPUT_LOOKBACK deve ser inteiro positivo ou vazio.'
    [[ -z "$NS_VALIDATION_MODE" || "$NS_VALIDATION_MODE" == ordered_sample_mean ]] || die 'NS_VALIDATION_MODE deve ser vazio ou ordered_sample_mean.'
    [[ "$MOMENTUM_INPUT_MODE" == off || "$MOMENTUM_MODE" == off ]] || die 'Momentum na entrada nao pode ser combinado com residual_output.'
    [[ "$MOMENTUM_MODE" == off || "$MOMENTUM_MODE" == residual_output ]] || die 'MOMENTUM_MODE deve ser off ou residual_output.'
    [[ -z "$MOMENTUM_LOOKBACK" || "$MOMENTUM_LOOKBACK" =~ ^[1-9][0-9]*$ ]] || die 'MOMENTUM_LOOKBACK deve ser inteiro positivo ou vazio.'
    [[ "$MOMENTUM_ETA" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || die 'MOMENTUM_ETA deve ser finito e nao negativo.'
    [[ "$MOMENTUM_GAIN_INIT" =~ ^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || die 'MOMENTUM_GAIN_INIT deve ser numero finito.'
    case "${MOMENTUM_DIAGNOSTICS,,}" in 0|1|true|false) ;; *) die 'MOMENTUM_DIAGNOSTICS deve ser 0, 1, true ou false.';; esac
    case "$NS_LRADJ" in ''|type1|type2|type3|constant) ;; *) die 'NS_LRADJ deve ser vazio, type1, type2, type3 ou constant.';; esac
    [[ "$EXPERIMENT_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_./=-]*$ && "$EXPERIMENT_TAG" != *..* && "$EXPERIMENT_TAG" != *//* && "$EXPERIMENT_TAG" != */ ]] || die 'EXPERIMENT_TAG invalida.'
    [[ "$NS_MASK_MODE" == natural || "$NS_MASK_MODE" == complementary ]] || die 'NS_MASK_MODE deve ser natural ou complementary.'
    [[ "$NS_PARALLEL_BRANCHES" == 0 || "$NS_PARALLEL_BRANCHES" == 1 ]] || die 'NS_PARALLEL_BRANCHES deve ser 0 ou 1.'
    [[ "$NS_MAX_CONCURRENT_JOBS" =~ ^[1-9][0-9]*$ ]] || die 'NS_MAX_CONCURRENT_JOBS deve ser positivo.'
    [[ -z "$NS_AFTER_JOB" || "$NS_AFTER_JOB" =~ ^[0-9]+$ ]] || die 'NS_AFTER_JOB exige um ID Slurm numerico.'
    for name in TRAIN_VALUES LOOKBACKS_VALUES HORIZON_VALUES K_LOSS_VALUES K_EVAL_VALUES; do integer_list "$name" "${!name}"; done
    read -r -a trains <<< "$TRAIN_VALUES"; read -r -a models <<< "$MODEL"
    read -r -a lbs <<< "$LOOKBACKS_VALUES"; read -r -a horizons <<< "$HORIZON_VALUES"
    read -r -a windows <<< "$K_LOSS_VALUES"; read -r -a lambdas <<< "$POSITIVE_LAMBDAS"
    read -r -a deltas <<< "$DELTAS_VALUES"; read -r -a losses <<< "$CROSS_LOSSES_VALUES"
    ((${#models[@]} && ${#losses[@]} && ${#lambdas[@]} && ${#deltas[@]})) || die 'Modelo/loss/lambdas/deltas nao podem ser vazios.'
    local -A seen=()
    for token in "${models[@]}"; do [[ "$token" =~ ^[A-Za-z][A-Za-z0-9_]*$ && ! ${seen[$token]+yes} ]] || die "Modelo invalido/repetido: $token"; seen[$token]=1; done
    if [[ "$MOMENTUM_INPUT_MODE" != off ]]; then
        for token in "${models[@]}"; do [[ "$token" == Nonstationary_Transformer ]] || die "Momentum na entrada ainda nao validado para $token."; done
    fi
    seen=()
    PLAIN_LOSSES=(); PROXIMITY_LOSSES=()
    for loss in "${losses[@]}"; do
        [[ ! ${seen[$loss]+yes} ]] || die "Loss repetida: $loss"; seen[$loss]=1
        case "$loss" in
            ranknet|hinge|pairwise_mse) PLAIN_LOSSES+=("$loss"); nplain=$((nplain+${#deltas[@]}));;
            listnet|listmle|approxndcg|spearman) PLAIN_LOSSES+=("$loss"); nplain=$((nplain+1));;
            ranknet_proximity|hinge_proximity|pairwise_mse_proximity|spearman_proximity|listnet_proximity|listmle_proximity|approxndcg_proximity) PROXIMITY_LOSSES+=("$loss");;
            *) die "Loss desconhecida: $loss";;
        esac
    done
    for loss in "${PROXIMITY_LOSSES[@]}"; do [[ ${seen[${loss%_proximity}]+yes} ]] || die "Inclua ${loss%_proximity} para comparar com $loss sem duplicar baselines."; done
    for h in "${horizons[@]}"; do
        local valid=0
        for k in "${windows[@]}"; do if ((k<=h && h%k==0)); then pairs=$((pairs+1)); valid=1; fi; done
        ((valid)) || die "H=$h nao tem nenhum K_loss valido."
    done
    BRANCHES=(temporal); MASKS=(''); COUNTS=(); BRANCH_LOSSES=('')
    local common=$((${#trains[@]}*${#models[@]}*${#lbs[@]}))
    COUNTS+=("$((common*${#horizons[@]}))")
    if ((${#PLAIN_LOSSES[@]})); then
        BRANCHES+=(sem_kernel); MASKS=('' ''); BRANCH_LOSSES+=("${PLAIN_LOSSES[*]}")
        COUNTS+=("$((common*pairs*${#lambdas[@]}*nplain))")
    fi
    seen=([temporal]=1 [sem_kernel]=1)
    while IFS= read -r spec; do
        read -r -a fields <<< "$spec"; ((${#fields[@]})) || continue
        name=${fields[0]}; [[ "$name" =~ ^[a-z][a-z0-9_]*$ ]] || die "Kernel invalido: $name"
        label="$name"
        for token in "${fields[@]:1}"; do
            [[ "$token" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] || die "Parametro de forma invalido: $token"
            label+="_${token//./p}"; label=${label//+/p}; label=${label//-/m}
        done
        [[ ! ${seen[$label]+yes} ]] || die "Kernel repetido: $label"; seen[$label]=1
        if ((${#PROXIMITY_LOSSES[@]})); then
            BRANCHES+=("$label"); MASKS+=("$name $NS_MASK_A $NS_MASK_MODE ${fields[*]:1}")
            BRANCH_LOSSES+=("${PROXIMITY_LOSSES[*]}")
            COUNTS+=("$((common*pairs*${#lambdas[@]}*${#PROXIMITY_LOSSES[@]}))")
        fi
    done <<< "${KERNEL_SPECS//;/$'\n'}"
    ((${#PROXIMITY_LOSSES[@]} == 0 || ${#BRANCHES[@]} > 2)) || die 'KERNEL_SPECS vazio para losses proximity.'
    for token in $NS_ONLY_BRANCHES; do [[ ${seen[$token]+yes} ]] || die "Ramo inexistente: $token"; done
    TOTAL=0; ACTIVE_BRANCHES=0
    for index in "${!BRANCHES[@]}"; do if selected "${BRANCHES[index]}"; then TOTAL=$((TOTAL+COUNTS[index])); ACTIVE_BRANCHES=$((ACTIVE_BRANCHES+1)); fi; done
    ((ACTIVE_BRANCHES)) || die 'Nenhum ramo selecionado.'
}
plan() {
    grid
    printf 'Modelos: %s; treinos: %s; LB: %s; H: %s; K_loss: %s; lambdas compostos: %s\n' "$MODEL" "$TRAIN_VALUES" "$LOOKBACKS_VALUES" "$HORIZON_VALUES" "$K_LOSS_VALUES" "$POSITIVE_LAMBDAS"
    if [[ "$MOMENTUM_MODE" != off ]]; then
        printf 'Momentum residual nos compostos: LB=%s; eta=%s; controle temporal off.\n' "${MOMENTUM_LOOKBACK:-LB do modelo}" "$MOMENTUM_ETA"
        printf 'Ganho inicial: %s; diagnostico: %s; regra LR: %s.\n' "$MOMENTUM_GAIN_INIT" "$MOMENTUM_DIAGNOSTICS" "${NS_LRADJ:-default do modelo}"
    fi
    if [[ "$MOMENTUM_INPUT_MODE" != off ]]; then
        printf 'Momentum na entrada: modo=%s; LB=%s; sem soma fixa na saida.\n' "$MOMENTUM_INPUT_MODE" "${MOMENTUM_INPUT_LOOKBACK:-LB do modelo}"
    fi
    [[ -z "$NS_VALIDATION_MODE" ]] || printf 'Validacao: %s.\n' "$NS_VALIDATION_MODE"
    for index in "${!BRANCHES[@]}"; do selected "${BRANCHES[index]}" || continue; printf '%-30s %5d treinos  %s\n' "${BRANCHES[index]}" "${COUNTS[index]}" "${MASKS[index]}"; done
    printf 'TOTAL: %d treinos unicos; %d ramos selecionados.\n' "$TOTAL" "$ACTIVE_BRANCHES"
    if [[ "$NS_PARALLEL_BRANCHES" == 0 ]]; then printf 'Ramos encadeados afterany; ate %s tarefas simultaneas por array ativo.\n' "$NS_MAX_CONCURRENT_JOBS";
    else printf 'Ramos paralelos: ate %d tarefas simultaneas.\n' "$((ACTIVE_BRANCHES*NS_MAX_CONCURRENT_JOBS))"; fi
    printf 'Previsoes: %s\n' "$PRED_ROOT"
}
preflight() {
    [[ "$TFB_ROOT" == /* ]] || die 'TFB_ROOT deve ser absoluto.'
    [[ -x "$NS_PYTHON_BIN" ]] || die "Python inexistente: $NS_PYTHON_BIN"
    [[ -f "$NS_RETURNS_FILE" ]] || die "Retornos ausentes: $NS_RETURNS_FILE"
    for name in run_temporal_experiment.sh run_composite_trading_experiment.sh; do [[ -f "$TFB_ROOT/scripts/$name" ]] || die "Launcher ausente: $name"; done
}
branch_root() { if [[ "$1" == temporal ]]; then printf '%s/temporal' "$PRED_ROOT"; else printf '%s/composto/%s' "$PRED_ROOT" "$1"; fi; }
train_launcher() {
    local index=$1 action=$2 dependency=${3:-} branch=${BRANCHES[$1]} launcher tag root momentum_mode=$MOMENTUM_MODE momentum_input_mode=$MOMENTUM_INPUT_MODE
    root=$(branch_root "$branch")
    if [[ "$branch" == temporal ]]; then launcher=run_temporal_experiment.sh; tag="$EXPERIMENT_TAG/temporal"; momentum_mode=off; momentum_input_mode=off;
    else launcher=run_composite_trading_experiment.sh; tag="$EXPERIMENT_TAG/composto/$branch"; fi
    env -u B3_INLINE_CONFIG -u B3_CONFIG -u B3_CONFIG_SOURCE -u B3_WORKER_LAUNCHER -u EXPERIMENT \
      TFB_ROOT="$TFB_ROOT" PYTHON_BIN="$NS_PYTHON_BIN" VENV_PATH="$NS_VENV_PATH" \
      DATASET_PATH="$NS_RETURNS_FILE" SOURCE_DATA_KIND=log_return \
      TRAIN_PERCENTAGES="$TRAIN_VALUES" MODELS="$MODEL" LOOKBACKS="$LOOKBACKS_VALUES" HORIZONS="$HORIZON_VALUES" \
      TRADE_WINDOWS="$K_LOSS_VALUES" CROSS_LOSSES="${BRANCH_LOSSES[index]}" LAMBDAS="$POSITIVE_LAMBDAS" DELTAS="$DELTAS_VALUES" \
      CROSS_MASK="${MASKS[index]:-cauchy $NS_MASK_A $NS_MASK_MODE}" \
      BATCH_SIZE="$NS_BATCH_SIZE" NUM_EPOCHS="$NS_NUM_EPOCHS" PATIENCE="$NS_PATIENCE" LEARNING_RATE="$NS_LEARNING_RATE" \
      TEMPORAL_LOSS="$NS_TEMPORAL_LOSS" TRAIN_RATIO_IN_TV="$NS_TRAIN_RATIO_IN_TV" SEED="$NS_SEED" STRIDE="$NS_STRIDE" NUM_ROLLINGS="$NS_NUM_ROLLINGS" \
      CROSS_SCALE="$NS_CROSS_SCALE" RANKNET_ALPHA="$NS_RANKNET_ALPHA" LISTNET_TAU="$NS_LISTNET_TAU" HINGE_MARGIN="$NS_HINGE_MARGIN" \
      SPEARMAN_TAU="$NS_SPEARMAN_TAU" LISTMLE_TAU="$NS_LISTMLE_TAU" APPROXNDCG_TAU="$NS_APPROXNDCG_TAU" HUBER_DELTA="$NS_HUBER_DELTA" \
      SCORE_KIND="$NS_SCORE_KIND" CROSS_SCORE_NORMALIZATION="$NS_CROSS_SCORE_NORMALIZATION" LOSS_INVERSE_NORM="$NS_LOSS_INVERSE_NORM" LOSS_TRACK_COMPONENTS="$NS_LOSS_TRACK_COMPONENTS" \
      MOMENTUM_MODE="$momentum_mode" MOMENTUM_LOOKBACK="$MOMENTUM_LOOKBACK" MOMENTUM_ETA="$MOMENTUM_ETA" \
      MOMENTUM_GAIN_INIT="$MOMENTUM_GAIN_INIT" MOMENTUM_DIAGNOSTICS="$MOMENTUM_DIAGNOSTICS" LRADJ="$NS_LRADJ" \
      MOMENTUM_INPUT_MODE="$momentum_input_mode" MOMENTUM_INPUT_LOOKBACK="$MOMENTUM_INPUT_LOOKBACK" VALIDATION_MODE="$NS_VALIDATION_MODE" \
      RUN_TAG="$tag" OUT_ROOT="$root" RESULT_ROOT="/snfs2/igor.viveiros/result/$tag" LOG_ROOT="$TFB_ROOT/logs/$tag" \
      PARTITION="$NS_PARTITION" GRES="$NS_GRES" TIME_LIMIT="$NS_TIME_LIMIT" NODELIST="$NS_NODELIST" CPUS_PER_TASK="$NS_CPUS_PER_TASK" MEMORY="$NS_MEMORY" \
      MAX_CONCURRENT_JOBS="$NS_MAX_CONCURRENT_JOBS" ARRAY_CHUNK_SIZE="$NS_ARRAY_CHUNK_SIZE" MANTER_CSV="$NS_MANTER_CSV" B3_AFTER_JOB="$dependency" \
      bash "$TFB_ROOT/scripts/$launcher" "$action"
}
previews() { for index in "${!BRANCHES[@]}"; do selected "${BRANCHES[index]}" || continue; printf '\nPrevia: %s\n' "${BRANCHES[index]}"; train_launcher "$index" --dry-run "$NS_AFTER_JOB"; done; }
branch_fingerprint() {
    {
    printf '%s\n' "$MODEL" "$TRAIN_VALUES" "$LOOKBACKS_VALUES" "$HORIZON_VALUES" "$K_LOSS_VALUES" "$POSITIVE_LAMBDAS" "${BRANCH_LOSSES[$1]}" "${MASKS[$1]}" "$DELTAS_VALUES" \
      "$NS_RETURNS_FILE" "$NS_BATCH_SIZE" "$NS_NUM_EPOCHS" "$NS_PATIENCE" "$NS_LEARNING_RATE" "$NS_TEMPORAL_LOSS" "$NS_TRAIN_RATIO_IN_TV" "$NS_SEED" "$NS_STRIDE" "$NS_NUM_ROLLINGS" \
      "$NS_CROSS_SCALE" "$NS_RANKNET_ALPHA" "$NS_LISTNET_TAU" "$NS_HINGE_MARGIN" "$NS_SCORE_KIND" "$NS_CROSS_SCORE_NORMALIZATION" "$NS_LOSS_INVERSE_NORM" "$NS_LOSS_TRACK_COMPONENTS"
    [[ -z "$NS_SPEARMAN_TAU" ]] || printf 'spearman_tau=%s\n' "$NS_SPEARMAN_TAU"
    [[ -z "$NS_LISTMLE_TAU" ]] || printf 'listmle_tau=%s\n' "$NS_LISTMLE_TAU"
    [[ -z "$NS_APPROXNDCG_TAU" ]] || printf 'approxndcg_tau=%s\n' "$NS_APPROXNDCG_TAU"
    [[ -z "$NS_HUBER_DELTA" ]] || printf 'huber_delta=%s\n' "$NS_HUBER_DELTA"
    if [[ "${BRANCHES[$1]}" != temporal && "$MOMENTUM_MODE" != off ]]; then
        printf '%s\n' "$MOMENTUM_MODE" "$MOMENTUM_LOOKBACK" "$MOMENTUM_ETA"
        # Preserve historical fingerprints for the neutral new defaults.
        if [[ ! "$MOMENTUM_GAIN_INIT" =~ ^[+-]?(0+([.]0*)?|[.]0+)([eE][+-]?[0-9]+)?$ ]]; then
            printf 'momentum_gain_init=%s\n' "$MOMENTUM_GAIN_INIT"
        fi
        case "${MOMENTUM_DIAGNOSTICS,,}" in 1|true) printf 'momentum_diagnostics=true\n';; esac
    fi
    if [[ "${BRANCHES[$1]}" != temporal && "$MOMENTUM_INPUT_MODE" != off ]]; then
        printf 'momentum_input_mode=%s\nmomentum_input_lookback=%s\n' "$MOMENTUM_INPUT_MODE" "$MOMENTUM_INPUT_LOOKBACK"
    fi
    [[ -z "$NS_VALIDATION_MODE" ]] || printf 'validation_mode=%s\n' "$NS_VALIDATION_MODE"
    [[ -z "$NS_LRADJ" ]] || printf 'lradj=%s\n' "$NS_LRADJ"
    } | sha256sum | cut -d' ' -f1
}
train_all() {
    local previous=$NS_AFTER_JOB branch mark fingerprint output code last
    command -v sbatch >/dev/null || die 'sbatch ausente; execute no cluster.'
    mkdir -p "$PRED_ROOT/.submissoes_treino"
    for index in "${!BRANCHES[@]}"; do
        branch=${BRANCHES[index]}; selected "$branch" || continue
        mark="$PRED_ROOT/.submissoes_treino/$branch"; fingerprint=$(branch_fingerprint "$index")
        if ! mkdir "$mark" 2>/dev/null; then
            [[ -f "$mark/fingerprint" && $(cat "$mark/fingerprint") == "$fingerprint" ]] || die "Parametros alterados no ramo $branch; use outra EXPERIMENT_TAG."
            [[ -f "$mark/submitted" && -s "$mark/last_job_id" ]] || die "Submissao incerta/parcial em $mark; confira seu log e o Slurm antes de retomar."
            printf 'Ramo %s ja submetido: job %s; preservado.\n' "$branch" "$(cat "$mark/last_job_id")"
            [[ "$NS_PARALLEL_BRANCHES" == 1 ]] || previous=$(cat "$mark/last_job_id")
            continue
        fi
        printf '%s\n' "$fingerprint" > "$mark/fingerprint"
        printf '%s\n' "$previous" > "$mark/dependency"
        if output=$(train_launcher "$index" submit "$previous" 2>&1); then code=0; else code=$?; fi
        printf '%s\n' "$output" | tee "$mark/submission.log"
        last=$(printf '%s\n' "$output" | awk '/^Job [0-9]+:/ {gsub(":", "", $2); id=$2} END {print id}')
        [[ -z "$last" ]] || printf '%s\n' "$last" > "$mark/last_job_id"
        ((code == 0)) || die "Falha ao submeter $branch (codigo $code). Marca preservada; ramos posteriores nao submetidos."
        [[ "$last" =~ ^[0-9]+$ ]] || die "Nao foi possivel identificar ultimo job de $branch; confira $mark/submission.log."
        : > "$mark/submitted"
        [[ "$NS_PARALLEL_BRANCHES" == 1 ]] || previous=$last
    done
}
backtests() {
    local action=$1 branch launcher id root
    [[ -d "$NS_MARKET_DIR" && -f "$NS_STRATEGIES_FILE" ]] || die 'Bundle de mercado/estrategias ausente.'
    [[ "$action" == planejar || "$action" == submeter || "$action" == consolidar ]] || die 'Acao de backtest invalida.'
    for index in "${!BRANCHES[@]}"; do
        branch=${BRANCHES[index]}; selected "$branch" || continue; root=$(branch_root "$branch")
        if [[ "$branch" == temporal ]]; then launcher=run_backtests_cpu.sh; else launcher=run_backtests_composto_cpu.sh; fi
        [[ -f "$TFB_ROOT/scripts/$launcher" ]] || die "Launcher ausente: $launcher"
        id="kernels_${branch}"
        env -u RUN_ID -u COMPOSITE_RUN_ID TFB_ROOT="$TFB_ROOT" PYTHON_BIN="$NS_PYTHON_BIN" \
          PREDICTIONS_ROOT="$root" MARKET_DIR="$NS_MARKET_DIR" RETURNS_FILE="$NS_RETURNS_FILE" STRATEGIES_FILE="$NS_STRATEGIES_FILE" \
          OUTPUT_ROOT="$BACKTEST_ROOT" RUN_ID="$id" COMPOSITE_RUN_ID="$id" TRAIN_VALUES="$TRAIN_VALUES" K_EVAL_VALUES="$K_EVAL_VALUES" \
          SLIPPAGE_BPS="$NS_SLIPPAGE_BPS" COST_BPS="$NS_COST_BPS" WORKERS="$NS_CPU_WORKERS" N_LOTES="$NS_CPU_N_LOTES" MEMORY="$NS_CPU_MEMORY" \
          bash "$TFB_ROOT/scripts/$launcher" "$action"
    done
}
main() {
    local action=${1:---help}
    case "$action" in
      --help|-h) [[ $# -le 1 ]] || die 'Argumentos excedentes.'; usage;;
      --plan) [[ $# == 1 ]] || die 'Argumentos excedentes.'; plan;;
      --dry-run) [[ $# == 1 ]] || die 'Argumentos excedentes.'; plan; preflight; previews;;
      --train|--submit) [[ $# == 1 ]] || die 'Argumentos excedentes.'; plan; preflight; previews; train_all;;
      --backtest) [[ $# -le 2 ]] || die 'Argumentos excedentes.'; plan; preflight; backtests "${2:-planejar}";;
      *) die 'Acao desconhecida; use --help.';;
    esac
}
main "$@"
