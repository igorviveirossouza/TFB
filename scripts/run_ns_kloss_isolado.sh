#!/usr/bin/env bash
# Experimento isolado: nenhum arquivo Python e alterado por este wrapper.
# Auditado no TFB/experiment/b3-loss-clean-v1, commit 28d46152d47af162eef16bf9015eed43c45758cc.
# Os launchers existentes continuam responsaveis pela preparacao, treino e backtest.
set -Eeuo pipefail

# Caminhos explicitos do servidor. Variaveis NS_* evitam herdar outra grade.
TFB_ROOT="${TFB_ROOT:-/sonic_home/igor.viveiros/src/TFB}"
NS_PYTHON_BIN="${NS_PYTHON_BIN:-/sonic_home/igor.viveiros/py310/bin/python}"
NS_VENV_PATH="${NS_VENV_PATH:-/sonic_home/igor.viveiros/py310/bin/activate}"
NS_RETURNS_FILE="${NS_RETURNS_FILE:-$TFB_ROOT/dataset/forecasting/b3_daily_tfb_new.csv}"
NS_MARKET_DIR="${NS_MARKET_DIR:-/snfs2/igor.viveiros/analise_resultados/insumos/base_backtest_temporal_001}"
NS_STRATEGIES_FILE="${NS_STRATEGIES_FILE:-$TFB_ROOT/config/estrategias_backtest.json}"

# O launcher composto atual fixa estes tres prefixos; apenas RUN_TAG e editavel.
# O prefixo novo agrupa os dois ramos para --pred-root do R local.
readonly EXPERIMENT_TAG='ns_kloss_isolado_t80_20261105'
readonly REMOTE_BASE='/snfs2/igor.viveiros'
readonly PRED_ROOT="$REMOTE_BASE/previsoes/$EXPERIMENT_TAG"
readonly RESULT_ROOT_BASE="$REMOTE_BASE/result/$EXPERIMENT_TAG"
readonly BACKTEST_ROOT="$REMOTE_BASE/analise_resultados/backtests/$EXPERIMENT_TAG"
readonly CONTROL_TAG="$EXPERIMENT_TAG/temporal"
readonly COMPOSITE_TAG="$EXPERIMENT_TAG/composto"
readonly CONTROL_PRED_ROOT="$PRED_ROOT/temporal"
readonly COMPOSITE_PRED_ROOT="$PRED_ROOT/composto"

NS_PARTITION="${NS_PARTITION:-medusas_dev}"
NS_GRES="${NS_GRES-gpu:1}"
NS_TIME_LIMIT="${NS_TIME_LIMIT:-01:30:00}"
NS_CPUS_PER_TASK="${NS_CPUS_PER_TASK:-3}"
NS_MEMORY="${NS_MEMORY:-}"
NS_NODELIST="${NS_NODELIST:-}"
NS_CPU_WORKERS="${NS_CPU_WORKERS:-16}"

# Grade editavel neste bloco; listas separadas por espacos. Lambda zero e apenas temporal.
readonly MODEL='Nonstationary_Transformer'
readonly LOOKBACKS_VALUES='256 512'
readonly HORIZON_VALUES='25'
readonly K_LOSS_VALUES='1 5 25'
readonly POSITIVE_LAMBDAS='0.5 0.8 1'
readonly CROSS_LOSSES_VALUES='ranknet ranknet_proximity'
readonly K_EVAL_VALUES_FIXED='1 5 10 15 20 25'

usage() {
    cat <<HELP
Uso: bash run_ns_kloss_isolado.sh ACAO [acao_cpu]

  --help                     Mostra esta ajuda; nao acessa o cluster.
  --plan                     Lista e conta a grade configurada; nao usa Python ou Slurm.
  --dry-run                  Preflight + previas dos dois launchers; nao submete.
  --train | --submit         Preflight e duas previas antes de submeter os treinos.
  --backtest [planejar]       Planeja backtests CPU dos dois ramos; nao submete.
  --backtest submeter         Planeja ambos e depois submete dois jobs CPU.
  --backtest consolidar       Submete consolidacao dos dois backtests existentes.

Grade: treino 80; modelos: $MODEL; LB: $LOOKBACKS_VALUES; H: $HORIZON_VALUES.
K_loss: $K_LOSS_VALUES; lambdas compostos: $POSITIVE_LAMBDAS; controle: lambda 0.
Losses positivas: $CROSS_LOSSES_VALUES; proximity usa Cauchy natural a=.025.
Controle: MSE temporal, uma vez por modelo/LB/H, compartilhado por losses e K_loss.
Composto considera somente K_loss <= H e H divisivel por K_loss.
Todos usam batch 16, seed 2026, 100 epocas, patience 10, LR 1e-4 e split interno .875.

IMPORTANTE: o checkout utilizado deve conter os dois validadores atualizados
para aceitar 0 < lambda <= 1. Este wrapper NAO os altera. --help e --plan
funcionam sem essa verificacao; as demais acoes conferem antes de qualquer job.
O preflight reconhece as duas linhas esperadas; alteracoes diferentes exigem
revisao deste wrapper. As previas reais dos launchers conferem a configuracao.

Caminho do repo: TFB_ROOT=/sonic_home/igor.viveiros/src/TFB (pode ser informado).
Recursos opcionais: NS_PARTITION, NS_GRES, NS_TIME_LIMIT, NS_CPUS_PER_TASK,
NS_MEMORY, NS_NODELIST, NS_CPU_WORKERS. Caminhos opcionais: NS_PYTHON_BIN,
NS_VENV_PATH, NS_RETURNS_FILE, NS_MARKET_DIR, NS_STRATEGIES_FILE.
RUN_TAG e grade sao definidos neste SH; variaveis de outras grades nao entram.

Concorrencia de treino: 1 tarefa por array em cada um dos dois ramos = maximo 2
jobs desta submissao. Arrays adicionais de um mesmo ramo sao encadeados pelo
launcher. Uma marca de submissao impede repetir --train acidentalmente.
Backtest usa N_LOTES=1 por ramo, portanto ate 2 jobs CPU. Execute apos os treinos.
Os launchers CPU existentes fixam a particao gorgonas_dev; nao e sobrescrita aqui.

Baixe a pasta de previsoes deste experimento e execute o R local com --pred-root
apontando para ela. O R le os part.parquet e os respectivos run_config.json.
Os backtests CPU sao opcionais para esse R; nao iniciam o R no servidor.
HELP
}

die() { printf 'ERRO: %s\n' "$*" >&2; exit 2; }

training_grid() {
    local model lb h loss lam k mask value
    local -a models lookbacks horizons windows losses lambdas
    read -r -a models <<< "$MODEL"
    read -r -a lookbacks <<< "$LOOKBACKS_VALUES"
    read -r -a horizons <<< "$HORIZON_VALUES"
    read -r -a windows <<< "$K_LOSS_VALUES"
    read -r -a losses <<< "$CROSS_LOSSES_VALUES"
    read -r -a lambdas <<< "$POSITIVE_LAMBDAS"
    [[ ${#models[@]} -gt 0 && ${#lookbacks[@]} -gt 0 && ${#horizons[@]} -gt 0 &&
       ${#windows[@]} -gt 0 && ${#losses[@]} -gt 0 && ${#lambdas[@]} -gt 0 ]] || die 'Lista vazia na grade.'
    for value in "${lookbacks[@]}" "${horizons[@]}" "${windows[@]}"; do
        [[ "$value" =~ ^[1-9][0-9]*$ ]] || die 'LB, H e K_loss devem ser inteiros positivos sem zeros a esquerda.'
    done
    for model in "${models[@]}"; do
        for lb in "${lookbacks[@]}"; do
            for h in "${horizons[@]}"; do
                printf 'temporal\t%s\tmse\t0\t%d\t%d\tNA\tcontrole compartilhado\n' "$model" "$lb" "$h"
            done
        done
    done
    for loss in "${losses[@]}"; do
        mask='sem mascara'
        [[ "$loss" != *_proximity ]] || mask='cauchy .025 natural'
        for lam in "${lambdas[@]}"; do
            for model in "${models[@]}"; do
                for lb in "${lookbacks[@]}"; do
                    for h in "${horizons[@]}"; do
                        for k in "${windows[@]}"; do
                            (( k <= h && h % k == 0 )) || continue
                            printf 'composto\t%s\t%s\t%s\t%d\t%d\t%d\t%s\n' "$model" "$loss" "$lam" "$lb" "$h" "$k" "$mask"
                        done
                    done
                done
            done
        done
    done
}

grid_counts() {
    local branch rest
    GRID_ROWS=$(training_grid)
    GRID_CONTROLS=0; GRID_COMPOSITES=0
    while IFS=$'\t' read -r branch rest; do
        if [[ "$branch" == temporal ]]; then
            GRID_CONTROLS=$((GRID_CONTROLS+1))
        else
            GRID_COMPOSITES=$((GRID_COMPOSITES+1))
        fi
    done <<< "$GRID_ROWS"
    GRID_TOTAL=$((GRID_CONTROLS+GRID_COMPOSITES))
}

plan() {
    local row index=0
    grid_counts
    printf 'Modelos: %s | treino 80 | H: %s | batch 16 | seed 2026 | LR 1e-4 | epocas 100 | patience 10\n' "$MODEL" "$HORIZON_VALUES"
    printf 'Previsoes: %s\nResultados: %s\nBacktests: %s\n' "$PRED_ROOT" "$RESULT_ROOT_BASE" "$BACKTEST_ROOT"
    printf 'Concorrencia: 1 por array de treino; ate 2 jobs entre os dois ramos.\n'
    printf 'Pre-requisito: checkout com os dois validadores Python aceitando 0 < lambda <= 1.\n'
    printf 'id\tramo\tmodelo\tloss\tlambda\tLB\tH\tK_loss\tmascara\n'
    while IFS= read -r row; do
        index=$((index+1))
        printf '%d\t%s\n' "$index" "$row"
    done <<< "$GRID_ROWS"
    printf 'TOTAL: %d controles + %d compostos = %d treinos unicos.\n' "$GRID_CONTROLS" "$GRID_COMPOSITES" "$GRID_TOTAL"
    printf 'Avaliacao CPU: K_eval=%s; somente K_eval <= H por previsao, sem novos treinos.\n' "$K_EVAL_VALUES_FIXED"
    printf 'Lambda 0 e momentum sao reutilizados nas tabelas; nao duplicar treinamento por painel.\n'
}

lambda_one_preflight() {
    local producer="$TFB_ROOT/scripts/b3_experiment.py"
    local consumer="$TFB_ROOT/scripts/backtest_composto/io.py"
    local producer_compact consumer_compact
    [[ -f "$producer" ]] || die "Runner nao encontrado: $producer. Informe TFB_ROOT."
    [[ -f "$consumer" ]] || die "Validador de backtest nao encontrado: $consumer."
    producer_compact=$(tr -d '[:space:]' < "$producer")
    consumer_compact=$(tr -d '[:space:]' < "$consumer")
    if [[ "$producer_compact" != *"ifany(not0<x<=1forxinconfig['lambdas']):"* ||
          "$consumer_compact" != *'ifany(not0<x<=1forxinlambdas)orany(x<0forxindeltas):'* ]]; then
        cat >&2 <<'BLOCK'
BLOQUEADO: a grade confirmada inclui lambda=1, mas os dois validadores existentes
ainda nao apresentam a regra aprovada 0 < lambda <= 1. Nenhum job foi submetido.
Este SH nao cria Python, nao faz monkeypatch e nao modifica o repositorio.

Atualize o checkout com os dois ajustes ja autorizados:
1) scripts/b3_experiment.py, linha 79 no snapshot auditado:
   if any(not 0 < x < 1 for x in config['lambdas']):
   -> if any(not 0 < x <= 1 for x in config['lambdas']):
   Atualizar tambem a mensagem para "Lambda composto deve estar em (0,1]".
2) scripts/backtest_composto/io.py, linha 139 no snapshot auditado:
   if any(not 0 < x < 1 for x in lambdas) or any(x < 0 for x in deltas):
   -> if any(not 0 < x <= 1 for x in lambdas) or any(x < 0 for x in deltas):

A factory da loss ja aceita [0,1]. Lambda 0 continua no launcher temporal.
Depois de atualizar e validar o checkout, execute primeiro --dry-run.
BLOCK
        exit 3
    fi
}

common_preflight() {
    [[ "$TFB_ROOT" == /* ]] || die 'TFB_ROOT deve ser absoluto.'
    lambda_one_preflight
    [[ -x "$NS_PYTHON_BIN" ]] || die "Python existente nao executavel: $NS_PYTHON_BIN"
    [[ -f "$NS_RETURNS_FILE" ]] || die "Base de retornos ausente: $NS_RETURNS_FILE"
    [[ "$NS_CPUS_PER_TASK" =~ ^[1-9][0-9]*$ ]] || die 'NS_CPUS_PER_TASK invalido.'
    for name in run_temporal_experiment.sh run_composite_trading_experiment.sh; do
        [[ -f "$TFB_ROOT/scripts/$name" ]] || die "Launcher ausente: $name"
    done
    # Garante que BATCH_SIZE alcance tambem NS nos dois launchers atuais.
    local launcher
    for launcher in run_temporal_experiment.sh run_composite_trading_experiment.sh; do
        if ! grep -F '"Nonstationary_Transformer": {"batch_size": ${BATCH_SIZE}' "$TFB_ROOT/scripts/$launcher" >/dev/null; then
            die "$launcher nao expoe batch_size para NS conforme a versao auditada; nao usar controle batch 32."
        fi
    done
}

training_launcher() {
    local branch=$1; shift
    local tag launcher pred result log
    if [[ "$branch" == temporal ]]; then
        tag=$CONTROL_TAG; launcher=run_temporal_experiment.sh; pred=$CONTROL_PRED_ROOT
    else
        tag=$COMPOSITE_TAG; launcher=run_composite_trading_experiment.sh; pred=$COMPOSITE_PRED_ROOT
    fi
    result="$RESULT_ROOT_BASE/$branch"; log="$TFB_ROOT/logs/$EXPERIMENT_TAG/$branch"
    env -u B3_INLINE_CONFIG -u B3_CONFIG -u B3_CONFIG_SOURCE -u B3_WORKER_LAUNCHER -u EXPERIMENT \
        TFB_ROOT="$TFB_ROOT" PYTHON_BIN="$NS_PYTHON_BIN" VENV_PATH="$NS_VENV_PATH" \
        DATASET_PATH="$NS_RETURNS_FILE" SOURCE_DATA_KIND=log_return \
        TRAIN_PERCENTAGES=80 MODELS="$MODEL" LOOKBACKS="$LOOKBACKS_VALUES" HORIZONS="$HORIZON_VALUES" \
        BATCH_SIZE=16 NUM_EPOCHS=100 PATIENCE=10 LEARNING_RATE=0.0001 TEMPORAL_LOSS=mse \
        TRAIN_RATIO_IN_TV=0.875 SEED=2026 STRIDE=1 NUM_ROLLINGS=48000 \
        TRADE_WINDOWS="$K_LOSS_VALUES" CROSS_LOSSES="$CROSS_LOSSES_VALUES" \
        LAMBDAS="$POSITIVE_LAMBDAS" DELTAS=0 CROSS_MASK='cauchy 0.025 natural' \
        CROSS_SCALE=1 RANKNET_ALPHA=1 LISTNET_TAU=1 HINGE_MARGIN=0.1 \
        SCORE_KIND=simple_return CROSS_SCORE_NORMALIZATION=zscore \
        LOSS_INVERSE_NORM=true LOSS_TRACK_COMPONENTS=true \
        RUN_TAG="$tag" OUT_ROOT="$pred" RESULT_ROOT="$result" LOG_ROOT="$log" \
        PARTITION="$NS_PARTITION" GRES="$NS_GRES" TIME_LIMIT="$NS_TIME_LIMIT" \
        CPUS_PER_TASK="$NS_CPUS_PER_TASK" MEMORY="$NS_MEMORY" NODELIST="$NS_NODELIST" \
        MAX_CONCURRENT_JOBS=1 ARRAY_CHUNK_SIZE=1000 MANTER_CSV=FALSE \
        bash "$TFB_ROOT/scripts/$launcher" "$@"
}

training_previews() {
    grid_counts
    printf 'Previa autoritativa do controle (%d tarefas; modelos: %s; H: %s):\n' "$GRID_CONTROLS" "$MODEL" "$HORIZON_VALUES"
    training_launcher temporal --dry-run
    printf 'Previa autoritativa dos compostos (%d tarefas; modelos: %s; H: %s):\n' "$GRID_COMPOSITES" "$MODEL" "$HORIZON_VALUES"
    training_launcher composto --dry-run
}

backtest_preflight() {
    common_preflight
    [[ -d "$NS_MARKET_DIR" ]] || die "Bundle de mercado ausente: $NS_MARKET_DIR"
    [[ -f "$NS_STRATEGIES_FILE" ]] || die "Estrategias de backtest ausentes: $NS_STRATEGIES_FILE"
    [[ "$NS_CPU_WORKERS" =~ ^([1-9]|1[0-6])$ ]] || die 'NS_CPU_WORKERS deve estar entre 1 e 16.'
    local branch path
    for branch in temporal composto; do
        path="$PRED_ROOT/$branch/treino=80/experimento=$branch/run_config.json"
        [[ -f "$path" ]] || die "Treino ainda nao preparado/exportado: $path"
        [[ ! -d "$PRED_ROOT/$branch/treino=70" && ! -d "$PRED_ROOT/$branch/treino=90" ]] ||
            die 'A raiz isolada contem cortes 70/90; nao executar fora do treino 80.'
    done
    for path in run_backtests_cpu.sh run_backtests_composto_cpu.sh; do
        [[ -f "$TFB_ROOT/scripts/$path" ]] || die "Launcher CPU ausente: $path"
    done
}

backtest_launcher() {
    local branch=$1 action=$2 launcher id pred
    if [[ "$branch" == temporal ]]; then
        launcher=run_backtests_cpu.sh; id=ns_kloss_temporal_t80; pred=$CONTROL_PRED_ROOT
    else
        launcher=run_backtests_composto_cpu.sh; id=ns_kloss_composto_t80; pred=$COMPOSITE_PRED_ROOT
    fi
    env -u RUN_ID -u COMPOSITE_RUN_ID \
        TFB_ROOT="$TFB_ROOT" PYTHON_BIN="$NS_PYTHON_BIN" \
        PREDICTIONS_ROOT="$pred" MARKET_DIR="$NS_MARKET_DIR" RETURNS_FILE="$NS_RETURNS_FILE" \
        STRATEGIES_FILE="$NS_STRATEGIES_FILE" OUTPUT_ROOT="$BACKTEST_ROOT" \
        RUN_ID="$id" COMPOSITE_RUN_ID="$id" TRAIN_VALUES=80 K_EVAL_VALUES="$K_EVAL_VALUES_FIXED" \
        SLIPPAGE_BPS=0 COST_BPS=0 WORKERS="$NS_CPU_WORKERS" N_LOTES=1 MEMORY=56G \
        bash "$TFB_ROOT/scripts/$launcher" "$action"
}

submission_mark() {
    local stage=$1
    mkdir -p "$PRED_ROOT"
    local mark="$PRED_ROOT/.${stage}_solicitado"
    if ! mkdir "$mark" 2>/dev/null; then
        die "Ja existe $mark. Confira os jobs/resultados antes de decidir uma nova submissao; a marca nao e removida automaticamente."
    fi
    printf 'Solicitado em %s UTC; host=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${HOSTNAME:-desconhecido}" > "$mark/estado.txt"
    printf 'Marca de submissao: %s (preservada tambem se houver falha parcial).\n' "$mark"
}

main() {
    local action=${1:---help} cpu_action=${2:-planejar}
    case "$action" in
        --help|-h) [[ $# -le 1 ]] || die 'Argumentos excedentes.'; usage ;;
        --plan) [[ $# -eq 1 ]] || die 'Argumentos excedentes.'; plan ;;
        --dry-run)
            [[ $# -eq 1 ]] || die 'Argumentos excedentes.'
            common_preflight
            training_previews
            printf 'Nenhuma submissao realizada.\n'
            ;;
        --train|--submit)
            [[ $# -eq 1 ]] || die 'Argumentos excedentes.'
            common_preflight
            training_previews
            command -v sbatch >/dev/null || die 'sbatch nao encontrado; execute no servidor do cluster.'
            submission_mark treino
            training_launcher temporal submit
            training_launcher composto submit
            printf 'Submissoes solicitadas: %d controles + %d compostos = %d treinos; modelos: %s; H: %s; ate 2 tarefas simultaneas.\n' "$GRID_CONTROLS" "$GRID_COMPOSITES" "$GRID_TOTAL" "$MODEL" "$HORIZON_VALUES"
            ;;
        --backtest)
            [[ $# -le 2 ]] || die 'Argumentos excedentes.'
            case "$cpu_action" in planejar|submeter|consolidar) ;; *) die 'Acao CPU invalida.' ;; esac
            backtest_preflight
            if [[ "$cpu_action" == consolidar ]]; then
                [[ -f "$BACKTEST_ROOT/ns_kloss_temporal_t80/plano.json" && -f "$BACKTEST_ROOT/ns_kloss_composto_t80/plano.json" ]] ||
                    die 'Os dois planos de backtest devem existir antes de consolidar.'
            else
                backtest_launcher temporal planejar
                backtest_launcher composto planejar
            fi
            if [[ "$cpu_action" != planejar ]]; then
                command -v sbatch >/dev/null || die 'sbatch nao encontrado no servidor.'
                [[ "$cpu_action" != submeter ]] || submission_mark backtest
                backtest_launcher temporal "$cpu_action"
                backtest_launcher composto "$cpu_action"
            fi
            ;;
        *) die 'Acao desconhecida; use --help.' ;;
    esac
}

main "$@"

