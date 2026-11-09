#!/usr/bin/env bash
# Misturas de ranking: global/regional, com e sem Proximity, controles compartilhados.
# Todos os valores deste bloco podem ser sobrescritos com export no terminal.
set -Eeuo pipefail
TFB_ROOT="${TFB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-isolados/rankmix_a0.025_t80_k1_h1.25_v1}"
MODEL="${MODEL:-Nonstationary_Transformer}"
TRAIN_VALUES="${TRAIN_VALUES:-80}"
LOOKBACKS_VALUES="${LOOKBACKS_VALUES:-32 256 512}"
HORIZON_VALUES="${HORIZON_VALUES:-1 25}"
K_LOSS_VALUES="${K_LOSS_VALUES:-1}"
LAMBDAS_VALUES="${LAMBDAS_VALUES:-0 ${POSITIVE_LAMBDAS:-0.5 0.8 1}}"
TEMPORAL_LOSSES_VALUES="${TEMPORAL_LOSSES_VALUES:-mse huber}"
SEEDS_VALUES="${SEEDS_VALUES:-${NS_SEED:-2026}}"
CROSS_LOSSES_VALUES="${CROSS_LOSSES_VALUES:-rankmix rankmix_proximity}"
# Ordem fixa dos componentes: ListNet, ListMLE, ApproxNDCG. Zero desliga o componente.
RANK_MIX_WEIGHTS="${RANK_MIX_WEIGHTS:-0.5 0.5 0}"
RANK_MIX_MODES="${RANK_MIX_MODES:-global regional}"
RANK_MIX_REGIONS="${RANK_MIX_REGIONS:-top bottom top}"
RANK_MIX_FRACTION="${RANK_MIX_FRACTION:-0.3333333333333333}"
K_EVAL_VALUES="${K_EVAL_VALUES:-1 5 10 15 20 25}"
DELTAS_VALUES="${DELTAS_VALUES:-0}"
NS_MASK_A="${NS_MASK_A:-0.025}"
NS_MASK_MODE="${NS_MASK_MODE:-natural}"
# Entradas separadas por ponto e virgula ou quebra de linha: nome [forma...].
# A e modo sao compartilhados; para outra escala use outra EXPERIMENT_TAG.
KERNEL_SPECS="${KERNEL_SPECS:-cauchy}"
NS_ONLY_BRANCHES="${NS_ONLY_BRANCHES:-}" # vazio=todos; nomes impressos por --plan
NS_PYTHON_BIN="${NS_PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"
NS_VENV_PATH="${NS_VENV_PATH:-/sonic_home/igor.viveiros/py310/bin/activate}"
NS_RETURNS_FILE="${NS_RETURNS_FILE:-$TFB_ROOT/dataset/forecasting/b3_daily_tfb_new.csv}"
NS_MARKET_DIR="${NS_MARKET_DIR:-/snfs2/igor.viveiros/analise_resultados/insumos/base_backtest_temporal_001}"
NS_STRATEGIES_FILE="${NS_STRATEGIES_FILE:-$TFB_ROOT/config/estrategias_backtest.json}"
NS_BATCH_SIZE="${NS_BATCH_SIZE:-16}"
NS_NUM_EPOCHS="${NS_NUM_EPOCHS:-100}"
NS_PATIENCE="${NS_PATIENCE:-10}"
NS_LEARNING_RATE="${NS_LEARNING_RATE:-0.0001}"
NS_HUBER_DELTA="${NS_HUBER_DELTA:-0.5}"
NS_SPEARMAN_TAU="${NS_SPEARMAN_TAU:-1}"
NS_TRAIN_RATIO_IN_TV="${NS_TRAIN_RATIO_IN_TV:-0.875}"
NS_STRIDE="${NS_STRIDE:-1}"
NS_NUM_ROLLINGS="${NS_NUM_ROLLINGS:-48000}"
NS_CROSS_SCALE="${NS_CROSS_SCALE:-1}"
NS_RANKNET_ALPHA="${NS_RANKNET_ALPHA:-1}"
NS_LISTNET_TAU="${NS_LISTNET_TAU:-1}"
NS_LISTMLE_TAU="${NS_LISTMLE_TAU:-1}"
NS_APPROXNDCG_TAU="${NS_APPROXNDCG_TAU:-1}"
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
NS_VALIDATION_MODE="${NS_VALIDATION_MODE:-ordered_sample_mean}" # ordered_sample_mean: ordem fixa/media por amostra.
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
PRED_ROOT="${PRED_ROOT:-/snfs2/igor.viveiros/previsoes/$EXPERIMENT_TAG}"
BACKTEST_ROOT="${BACKTEST_ROOT:-/snfs2/igor.viveiros/analise_resultados/backtests/$EXPERIMENT_TAG}"

die() { printf 'ERRO: %s\n' "$*" >&2; exit 2; }
usage() {
    cat <<HELP
Uso: bash scripts/run_ns_rankmix_isolado.sh --plan | --dry-run | --train
     bash scripts/run_ns_rankmix_isolado.sh --backtest [planejar|submeter|consolidar]

Defaults: Nonstationary_Transformer; treino80; LB32/256/512; H1/25; K_loss1; seed2026.
MSE/Huber; lambda 0/.5/.8/1; misturas global e regional, com e sem Proximity.
Pesos internos [ListNet, ListMLE, ApproxNDCG] = [0.5, 0.5, 0].
Regional: ListNet no topo e ListMLE no fundo; fracao=1/3; Cauchy natural A=.025.
132 treinos: 12 controles temporais + 96 mistos + 24 rankings puros.
Momentum e somente benchmark externo, sem entrada nem residual na rede.

Todos os defaults aceitam export. Principais:
MODEL, TRAIN_VALUES, LOOKBACKS_VALUES, HORIZON_VALUES, K_LOSS_VALUES,
LAMBDAS_VALUES, TEMPORAL_LOSSES_VALUES, SEEDS_VALUES, K_EVAL_VALUES,
RANK_MIX_MODES, RANK_MIX_WEIGHTS, RANK_MIX_REGIONS, RANK_MIX_FRACTION,
KERNEL_SPECS, NS_MASK_A, NS_MASK_MODE, NS_LISTNET_TAU, NS_LISTMLE_TAU,
NS_APPROXNDCG_TAU, NS_HUBER_DELTA, NS_CROSS_SCALE,
NS_BATCH_SIZE, NS_NUM_EPOCHS, NS_PATIENCE, NS_LEARNING_RATE, NS_VALIDATION_MODE,
NS_PARTITION, NS_GRES, NS_NODELIST, NS_MAX_CONCURRENT_JOBS, NS_PARALLEL_BRANCHES,
NS_TIME_LIMIT, NS_CPUS_PER_TASK, NS_MEMORY, NS_ARRAY_CHUNK_SIZE,
PRED_ROOT, BACKTEST_ROOT, NS_ONLY_BRANCHES.
MODEL aceita varios nomes separados por espacos.
RANK_MIX_WEIGHTS exige 3 numeros nao negativos cuja soma seja 1 (tolerancia 1e-8).
Nao ha normalizacao automatica dos pesos. Pesos iguais nao garantem gradientes iguais.
RANK_MIX_MODES='regional' executa so a mistura direcionada: 72 treinos nos demais defaults.
RANK_MIX_WEIGHTS='0 0.5 0.5' troca ListNet por ApproxNDCG, mantendo ListMLE.
RANK_MIX_REGIONS exige 3 termos top/bottom; e aplicado somente no modo regional.
RANK_MIX_FRACTION deve estar em (0,0.5]; as regioes sao definidas por retornos realizados no treino.
CROSS_LOSSES_VALUES exige o par 'rankmix rankmix_proximity'; nunca perde o comparador sem kernel.
KERNEL_SPECS='cauchy;student 10' varia kernels sem repetir a mistura sem kernel.
K_loss deve dividir H: apenas pares validos sao treinados; todo H precisa de um K valido.
Losses temporais aceitas: mse, mae, huber. Lambda0 nao repete por K, modo ou kernel.
Lambda1 nao repete por tarefa temporal. As misturas nao duplicam controles temporais.
--plan conta sem Python/dados/Slurm; --dry-run valida os launchers reais.
--train encadeia ramos por afterany; NS_PARALLEL_BRANCHES=1 remove esse encadeamento.
NS_MAX_CONCURRENT_JOBS limita cada array, nao a soma de arrays paralelos.
Retomadas pulam ramos ja submetidos; falha parcial exige conferir log/Slurm.

Previsoes: $PRED_ROOT/seed=<seed>/temporal_loss=<loss|none>/...
Temporais ficam em temporal; compostos em mistura=<modo>_<assinatura>/composto/<kernel>.
A assinatura identifica pesos, regioes e fracao; a configuracao completa fica no run_config.
Backtests opcionais; repetir submeter apos novos parquets, sem outro job ativo no ramo.
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
add_branch() {
    BRANCHES+=("$1"); BRANCH_MODES+=("$2"); BRANCH_TEMPORALS+=("$3")
    BRANCH_SEEDS+=("$4"); BRANCH_LAMBDAS+=("$5"); BRANCH_LOSSES+=("$6")
    MASKS+=("$7"); COUNTS+=("$8"); BRANCH_PATHS+=("$9"); BRANCH_MIX_MODES+=("${10:-}")
}
grid() {
    local token spec name label loss h k seed temporal index mix_mode pairs=0 nplain=0
    read -r -a mix_weights <<< "$RANK_MIX_WEIGHTS"
    read -r -a mix_modes <<< "$RANK_MIX_MODES"
    read -r -a mix_regions <<< "$RANK_MIX_REGIONS"
    ((${#mix_weights[@]} == 3)) || die 'RANK_MIX_WEIGHTS exige 3 pesos: ListNet ListMLE ApproxNDCG.'
    ((${#mix_regions[@]} == 3)) || die 'RANK_MIX_REGIONS exige 3 regioes top/bottom.'
    ((${#mix_modes[@]})) || die 'RANK_MIX_MODES vazio.'
    for token in "${mix_weights[@]}"; do
        [[ "$token" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] && awk -v x="$token" 'BEGIN {exit !(x>=0 && x<=1)}' || die "Peso interno invalido: $token"
    done
    awk -v a="${mix_weights[0]}" -v b="${mix_weights[1]}" -v c="${mix_weights[2]}" 'BEGIN {d=a+b+c-1; if(d<0)d=-d; exit !(d<=1e-8)}' || die 'RANK_MIX_WEIGHTS deve somar 1; pesos nao sao normalizados automaticamente.'
    for token in "${mix_regions[@]}"; do [[ "$token" == top || "$token" == bottom ]] || die "Regiao invalida: $token"; done
    local -A seen_modes=()
    for token in "${mix_modes[@]}"; do
        [[ "$token" == global || "$token" == regional ]] || die "Modo de mistura invalido: $token"
        [[ ! ${seen_modes[$token]+yes} ]] || die "Modo de mistura repetido: $token"
        seen_modes[$token]=1
    done
    [[ "$RANK_MIX_FRACTION" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] && awk -v x="$RANK_MIX_FRACTION" 'BEGIN {exit !(x>0 && x<=0.5)}' || die 'RANK_MIX_FRACTION deve estar em (0,0.5].'
    # Canonical values keep 0.5 and .50 scientifically identical in labels/fingerprints.
    for index in "${!mix_weights[@]}"; do mix_weights[index]=$(awk -v x="${mix_weights[index]}" 'BEGIN {printf "%.17g",x}'); done
    RANK_MIX_WEIGHTS="${mix_weights[*]}"; RANK_MIX_REGIONS="${mix_regions[*]}"
    RANK_MIX_FRACTION=$(awk -v x="$RANK_MIX_FRACTION" 'BEGIN {printf "%.17g",x}')
    MIX_SIGNATURE=$(printf '%s\n' "$RANK_MIX_WEIGHTS" "$RANK_MIX_REGIONS" "$RANK_MIX_FRACTION" | sha256sum | cut -c1-12)
    [[ "$MOMENTUM_MODE" == off && "$MOMENTUM_INPUT_MODE" == off ]] || die 'Este experimento usa momentum somente como benchmark: MOMENTUM_MODE e MOMENTUM_INPUT_MODE devem ser off.'
    [[ "$EXPERIMENT_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_./=-]*$ && "$EXPERIMENT_TAG" != *..* && "$EXPERIMENT_TAG" != *//* && "$EXPERIMENT_TAG" != */ ]] || die 'EXPERIMENT_TAG invalida.'
    [[ "$NS_MASK_MODE" == natural || "$NS_MASK_MODE" == complementary ]] || die 'NS_MASK_MODE deve ser natural ou complementary.'
    [[ "$NS_PARALLEL_BRANCHES" == 0 || "$NS_PARALLEL_BRANCHES" == 1 ]] || die 'NS_PARALLEL_BRANCHES deve ser 0 ou 1.'
    [[ "$NS_MAX_CONCURRENT_JOBS" =~ ^[1-9][0-9]*$ ]] || die 'NS_MAX_CONCURRENT_JOBS deve ser positivo.'
    [[ -z "$NS_AFTER_JOB" || "$NS_AFTER_JOB" =~ ^[0-9]+$ ]] || die 'NS_AFTER_JOB exige um ID Slurm numerico.'
    [[ -z "$NS_VALIDATION_MODE" || "$NS_VALIDATION_MODE" == ordered_sample_mean ]] || die 'NS_VALIDATION_MODE invalido.'
    case "$NS_LRADJ" in ''|type1|type2|type3|constant) ;; *) die 'NS_LRADJ invalido.';; esac
    for name in NS_HUBER_DELTA NS_SPEARMAN_TAU NS_LISTNET_TAU NS_LISTMLE_TAU NS_APPROXNDCG_TAU NS_MASK_A; do
        [[ "${!name}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] && awk -v x="${!name}" 'BEGIN {exit !(x>0 && x<1e308)}' || die "$name deve ser positivo e finito."
    done
    for name in TRAIN_VALUES LOOKBACKS_VALUES HORIZON_VALUES K_LOSS_VALUES K_EVAL_VALUES; do integer_list "$name" "${!name}"; done
    read -r -a trains <<< "$TRAIN_VALUES"; read -r -a models <<< "$MODEL"
    for token in "${trains[@]}"; do ((token<100)) || die "Treino deve estar entre 1 e 99: $token"; done
    read -r -a lbs <<< "$LOOKBACKS_VALUES"; read -r -a horizons <<< "$HORIZON_VALUES"
    read -r -a windows <<< "$K_LOSS_VALUES"; read -r -a lambdas <<< "$LAMBDAS_VALUES"
    read -r -a deltas <<< "$DELTAS_VALUES"; read -r -a losses <<< "$CROSS_LOSSES_VALUES"
    read -r -a temporals <<< "$TEMPORAL_LOSSES_VALUES"; read -r -a seeds <<< "$SEEDS_VALUES"
    ((${#models[@]} && ${#losses[@]} && ${#lambdas[@]} && ${#deltas[@]} && ${#temporals[@]} && ${#seeds[@]})) || die 'Grade vazia.'
    local -A seen=()
    for token in "${models[@]}"; do [[ "$token" =~ ^[A-Za-z][A-Za-z0-9_]*$ && ! ${seen[$token]+yes} ]] || die "Modelo invalido/repetido: $token"; seen[$token]=1; done
    seen=()
    for seed in "${seeds[@]}"; do [[ "$seed" =~ ^(0|[1-9][0-9]{0,9})$ && ! ${seen[$seed]+yes} ]] && ((seed<=4294967295)) || die "Seed invalida/repetida: $seed"; seen[$seed]=1; done
    seen=()
    for temporal in "${temporals[@]}"; do
        case "$temporal" in mse|mae|huber) ;; *) die "Loss temporal desconhecida: $temporal";; esac
        [[ ! ${seen[$temporal]+yes} ]] || die "Loss temporal repetida: $temporal"; seen[$temporal]=1
    done
    seen=(); mixed_lambdas=(); local controls=0 pure=0 normalized
    for token in "${lambdas[@]}"; do
        [[ "$token" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] && awk -v x="$token" 'BEGIN {exit !(x>=0 && x<=1)}' || die "Lambda fora de [0,1]: $token"
        normalized=$(awk -v x="$token" 'BEGIN {printf "%.15g",x}')
        [[ ! ${seen[$normalized]+yes} ]] || die "Lambda repetido: $token"; seen[$normalized]=1
        case "$normalized" in 0) controls=1;; 1) pure=1;; *) mixed_lambdas+=("$token");; esac
    done
    seen=(); PLAIN_LOSSES=(); PROXIMITY_LOSSES=()
    for loss in "${losses[@]}"; do
        [[ ! ${seen[$loss]+yes} ]] || die "Loss repetida: $loss"; seen[$loss]=1
        case "$loss" in
            rankmix) PLAIN_LOSSES+=("$loss"); nplain=$((nplain+1));;
            rankmix_proximity) PROXIMITY_LOSSES+=("$loss");;
            *) die "Loss desconhecida neste experimento: $loss; use rankmix rankmix_proximity.";;
        esac
    done
    [[ ${seen[rankmix]+yes} && ${seen[rankmix_proximity]+yes} ]] || die 'Inclua rankmix e rankmix_proximity para comparar as duas versoes.'
    for h in "${horizons[@]}"; do
        local valid=0
        for k in "${windows[@]}"; do if ((k<=h && h%k==0)); then pairs=$((pairs+1)); valid=1; fi; done
        ((valid || (!pure && ${#mixed_lambdas[@]}==0))) || die "H=$h nao tem nenhum K_loss valido."
    done
    KERNEL_LABELS=(); KERNEL_MASKS=(); seen=()
    while IFS= read -r spec; do
        read -r -a fields <<< "$spec"; ((${#fields[@]})) || continue
        name=${fields[0]}; [[ "$name" =~ ^[a-z][a-z0-9_]*$ ]] || die "Kernel invalido: $name"
        label="$name"
        for token in "${fields[@]:1}"; do
            [[ "$token" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] || die "Parametro de forma invalido: $token"
            label+="_${token//./p}"; label=${label//+/p}; label=${label//-/m}
        done
        [[ ! ${seen[$label]+yes} ]] || die "Kernel repetido: $label"; seen[$label]=1
        KERNEL_LABELS+=("$label"); KERNEL_MASKS+=("$name $NS_MASK_A $NS_MASK_MODE ${fields[*]:1}")
    done <<< "${KERNEL_SPECS//;/$'\n'}"
    ((${#PROXIMITY_LOSSES[@]} == 0 || ${#KERNEL_LABELS[@]} > 0)) || die 'KERNEL_SPECS vazio.'
    BRANCHES=(); BRANCH_MODES=(); BRANCH_TEMPORALS=(); BRANCH_SEEDS=(); BRANCH_LAMBDAS=(); BRANCH_LOSSES=(); MASKS=(); COUNTS=(); BRANCH_PATHS=(); BRANCH_MIX_MODES=()
    local common=$((${#trains[@]}*${#models[@]}*${#lbs[@]})) stage path count
    for seed in "${seeds[@]}"; do
        for temporal in "${temporals[@]}"; do
            path="seed=$seed/temporal_loss=$temporal"
            if ((controls)); then add_branch "s${seed}_temporal_${temporal}" temporal "$temporal" "$seed" '' '' '' "$((common*${#horizons[@]}))" "$path/temporal"; fi
            if ((${#mixed_lambdas[@]})); then
                count=$((common*pairs*${#mixed_lambdas[@]}))
                for mix_mode in "${mix_modes[@]}"; do
                    local key="${mix_mode}_${MIX_SIGNATURE}"
                    add_branch "s${seed}_misto_${temporal}_${key}_sem_kernel" composto "$temporal" "$seed" "${mixed_lambdas[*]}" rankmix '' "$count" "$path/mistura=$key/composto/sem_kernel" "$mix_mode"
                    for index in "${!KERNEL_LABELS[@]}"; do
                        add_branch "s${seed}_misto_${temporal}_${key}_${KERNEL_LABELS[index]}" composto "$temporal" "$seed" "${mixed_lambdas[*]}" rankmix_proximity "${KERNEL_MASKS[index]}" "$count" "$path/mistura=$key/composto/${KERNEL_LABELS[index]}" "$mix_mode"
                    done
                done
            fi
        done
        if ((pure)); then
            path="seed=$seed/temporal_loss=none"; count=$((common*pairs))
            for mix_mode in "${mix_modes[@]}"; do
                local key="${mix_mode}_${MIX_SIGNATURE}"
                add_branch "s${seed}_puro_${key}_sem_kernel" composto mse "$seed" 1 rankmix '' "$count" "$path/mistura=$key/composto/sem_kernel" "$mix_mode"
                for index in "${!KERNEL_LABELS[@]}"; do
                    add_branch "s${seed}_puro_${key}_${KERNEL_LABELS[index]}" composto mse "$seed" 1 rankmix_proximity "${KERNEL_MASKS[index]}" "$count" "$path/mistura=$key/composto/${KERNEL_LABELS[index]}" "$mix_mode"
                done
            done
        fi
    done
    seen=(); for token in "${BRANCHES[@]}"; do seen[$token]=1; done
    for token in $NS_ONLY_BRANCHES; do [[ ${seen[$token]+yes} ]] || die "Ramo inexistente: $token"; done
    TOTAL=0; ACTIVE_BRANCHES=0; CONTROL_TOTAL=0; MIXED_TOTAL=0; PURE_TOTAL=0
    for index in "${!BRANCHES[@]}"; do
        selected "${BRANCHES[index]}" || continue
        TOTAL=$((TOTAL+COUNTS[index])); ACTIVE_BRANCHES=$((ACTIVE_BRANCHES+1))
        if [[ "${BRANCH_MODES[index]}" == temporal ]]; then CONTROL_TOTAL=$((CONTROL_TOTAL+COUNTS[index]));
        elif [[ "${BRANCH_LAMBDAS[index]}" == 1 ]]; then PURE_TOTAL=$((PURE_TOTAL+COUNTS[index]));
        else MIXED_TOTAL=$((MIXED_TOTAL+COUNTS[index])); fi
    done
    ((ACTIVE_BRANCHES)) || die 'Nenhum ramo selecionado.'
}
plan() {
    grid
    printf 'Modelos: %s; treino: %s; LB: %s; H: %s; K_loss: %s; seeds: %s\n' "$MODEL" "$TRAIN_VALUES" "$LOOKBACKS_VALUES" "$HORIZON_VALUES" "$K_LOSS_VALUES" "$SEEDS_VALUES"
    printf 'Temporais: %s; rankings: %s; lambdas: %s\n' "$TEMPORAL_LOSSES_VALUES" "$CROSS_LOSSES_VALUES" "$LAMBDAS_VALUES"
    printf 'Misturas: %s; pesos [ListNet ListMLE ApproxNDCG]: [%s]; regioes: [%s]; fracao: %s; assinatura: %s\n' "$RANK_MIX_MODES" "$RANK_MIX_WEIGHTS" "$RANK_MIX_REGIONS" "$RANK_MIX_FRACTION" "$MIX_SIGNATURE"
    printf 'Momentum: benchmark externo, 0 treinos; entrada e residual desligados.\n'
    for index in "${!BRANCHES[@]}"; do selected "${BRANCHES[index]}" || continue; printf '%-42s %5d treinos  %s\n' "${BRANCHES[index]}" "${COUNTS[index]}" "${MASKS[index]}"; done
    printf 'Controles lambda=0: %d; mistos: %d; ranking puro lambda=1: %d.\n' "$CONTROL_TOTAL" "$MIXED_TOTAL" "$PURE_TOTAL"
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
branch_root() { printf '%s/%s' "$PRED_ROOT" "${BRANCH_PATHS[$1]}"; }
train_launcher() {
    local index=$1 action=$2 dependency=${3:-} launcher tag root
    local -a mix_env=()
    root=$(branch_root "$index"); tag="$EXPERIMENT_TAG/${BRANCH_PATHS[index]}"
    if [[ "${BRANCH_MODES[index]}" == temporal ]]; then launcher=run_temporal_experiment.sh;
    else
        launcher=run_composite_trading_experiment.sh
        mix_env=("RANK_MIX_WEIGHTS=$RANK_MIX_WEIGHTS" "RANK_MIX_MODE=${BRANCH_MIX_MODES[index]}" "RANK_MIX_REGIONS=$RANK_MIX_REGIONS" "RANK_MIX_FRACTION=$RANK_MIX_FRACTION")
    fi
    env -u RANK_MIX_WEIGHTS -u RANK_MIX_MODE -u RANK_MIX_REGIONS -u RANK_MIX_FRACTION -u B3_INLINE_CONFIG -u B3_CONFIG -u B3_CONFIG_SOURCE -u B3_WORKER_LAUNCHER -u EXPERIMENT \
      TFB_ROOT="$TFB_ROOT" PYTHON_BIN="$NS_PYTHON_BIN" VENV_PATH="$NS_VENV_PATH" \
      DATASET_PATH="$NS_RETURNS_FILE" SOURCE_DATA_KIND=log_return \
      TRAIN_PERCENTAGES="$TRAIN_VALUES" MODELS="$MODEL" LOOKBACKS="$LOOKBACKS_VALUES" HORIZONS="$HORIZON_VALUES" \
      TRADE_WINDOWS="$K_LOSS_VALUES" CROSS_LOSSES="${BRANCH_LOSSES[index]}" LAMBDAS="${BRANCH_LAMBDAS[index]}" DELTAS="$DELTAS_VALUES" \
      CROSS_MASK="${MASKS[index]:-cauchy $NS_MASK_A $NS_MASK_MODE}" \
      BATCH_SIZE="$NS_BATCH_SIZE" NUM_EPOCHS="$NS_NUM_EPOCHS" PATIENCE="$NS_PATIENCE" LEARNING_RATE="$NS_LEARNING_RATE" \
      TEMPORAL_LOSS="${BRANCH_TEMPORALS[index]}" HUBER_DELTA="$NS_HUBER_DELTA" SPEARMAN_TAU="$NS_SPEARMAN_TAU" TRAIN_RATIO_IN_TV="$NS_TRAIN_RATIO_IN_TV" SEED="${BRANCH_SEEDS[index]}" STRIDE="$NS_STRIDE" NUM_ROLLINGS="$NS_NUM_ROLLINGS" \
      CROSS_SCALE="$NS_CROSS_SCALE" RANKNET_ALPHA="$NS_RANKNET_ALPHA" LISTNET_TAU="$NS_LISTNET_TAU" LISTMLE_TAU="$NS_LISTMLE_TAU" APPROXNDCG_TAU="$NS_APPROXNDCG_TAU" HINGE_MARGIN="$NS_HINGE_MARGIN" \
      SCORE_KIND="$NS_SCORE_KIND" CROSS_SCORE_NORMALIZATION="$NS_CROSS_SCORE_NORMALIZATION" LOSS_INVERSE_NORM="$NS_LOSS_INVERSE_NORM" LOSS_TRACK_COMPONENTS="$NS_LOSS_TRACK_COMPONENTS" \
      MOMENTUM_MODE=off MOMENTUM_LOOKBACK="$MOMENTUM_LOOKBACK" MOMENTUM_ETA="$MOMENTUM_ETA" \
      MOMENTUM_GAIN_INIT="$MOMENTUM_GAIN_INIT" MOMENTUM_DIAGNOSTICS="$MOMENTUM_DIAGNOSTICS" LRADJ="$NS_LRADJ" \
      MOMENTUM_INPUT_MODE=off MOMENTUM_INPUT_LOOKBACK="$MOMENTUM_INPUT_LOOKBACK" VALIDATION_MODE="$NS_VALIDATION_MODE" \
      RUN_TAG="$tag" OUT_ROOT="$root" RESULT_ROOT="/snfs2/igor.viveiros/result/$tag" LOG_ROOT="$TFB_ROOT/logs/$tag" \
      PARTITION="$NS_PARTITION" GRES="$NS_GRES" TIME_LIMIT="$NS_TIME_LIMIT" NODELIST="$NS_NODELIST" CPUS_PER_TASK="$NS_CPUS_PER_TASK" MEMORY="$NS_MEMORY" \
      MAX_CONCURRENT_JOBS="$NS_MAX_CONCURRENT_JOBS" ARRAY_CHUNK_SIZE="$NS_ARRAY_CHUNK_SIZE" MANTER_CSV="$NS_MANTER_CSV" B3_AFTER_JOB="$dependency" \
      "${mix_env[@]}" bash "$TFB_ROOT/scripts/$launcher" "$action"
}
previews() { for index in "${!BRANCHES[@]}"; do selected "${BRANCHES[index]}" || continue; printf '\nPrevia: %s\n' "${BRANCHES[index]}"; train_launcher "$index" --dry-run "$NS_AFTER_JOB"; done; }
branch_fingerprint() {
    local index=$1 mix_fingerprint=""
    if [[ "${BRANCH_MODES[index]}" == composto ]]; then mix_fingerprint="${BRANCH_MIX_MODES[index]}|$RANK_MIX_WEIGHTS|$RANK_MIX_REGIONS|$RANK_MIX_FRACTION"; fi
    printf '%s\n' "$MODEL" "$TRAIN_VALUES" "$LOOKBACKS_VALUES" "$HORIZON_VALUES" "$K_LOSS_VALUES" "${BRANCH_LAMBDAS[index]}" "${BRANCH_LOSSES[index]}" "${MASKS[index]}" "$DELTAS_VALUES" \
      "$NS_RETURNS_FILE" "$NS_BATCH_SIZE" "$NS_NUM_EPOCHS" "$NS_PATIENCE" "$NS_LEARNING_RATE" "${BRANCH_TEMPORALS[index]}" "$NS_TRAIN_RATIO_IN_TV" "${BRANCH_SEEDS[index]}" "$NS_STRIDE" "$NS_NUM_ROLLINGS" \
      "$NS_CROSS_SCALE" "$NS_RANKNET_ALPHA" "$NS_LISTNET_TAU" "$NS_LISTMLE_TAU" "$NS_APPROXNDCG_TAU" "$NS_HINGE_MARGIN" "$NS_HUBER_DELTA" "$NS_SPEARMAN_TAU" "$NS_SCORE_KIND" "$NS_CROSS_SCORE_NORMALIZATION" "$NS_LOSS_INVERSE_NORM" "$NS_LOSS_TRACK_COMPONENTS" \
      "$NS_VALIDATION_MODE" "$NS_LRADJ" "$mix_fingerprint" | sha256sum | cut -d' ' -f1
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
        branch=${BRANCHES[index]}; selected "$branch" || continue; root=$(branch_root "$index")
        if [[ "${BRANCH_MODES[index]}" == temporal ]]; then launcher=run_backtests_cpu.sh; else launcher=run_backtests_composto_cpu.sh; fi
        [[ -f "$TFB_ROOT/scripts/$launcher" ]] || die "Launcher ausente: $launcher"
        id="rankmix_${branch}"
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

