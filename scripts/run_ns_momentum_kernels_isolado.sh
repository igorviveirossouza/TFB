#!/usr/bin/env bash
# Experimento de momentum com o mesmo pool de kernels e pipeline ja validados.
# Cada valor abaixo e um DEFAULT: export no terminal tem precedencia.
# A execucao e delegada ao launcher de kernels, sem duplicar treino/backtest.
set -Eeuo pipefail

export TFB_ROOT="${TFB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-isolados/ns_momentum_kernels_a0.25_t80_v1}"
export MODEL="${MODEL:-Nonstationary_Transformer}"
export TRAIN_VALUES="${TRAIN_VALUES:-80}"
export LOOKBACKS_VALUES="${LOOKBACKS_VALUES:-256 512}"
export HORIZON_VALUES="${HORIZON_VALUES:-1 25}"
export K_LOSS_VALUES="${K_LOSS_VALUES:-1}"
# Lambda zero e o controle temporal, criado uma vez por modelo/LB/H.
export POSITIVE_LAMBDAS="${POSITIVE_LAMBDAS:-0.5 0.8 1}"
export CROSS_LOSSES_VALUES="${CROSS_LOSSES_VALUES:-ranknet ranknet_proximity}"
export DELTAS_VALUES="${DELTAS_VALUES:-0}"
export K_EVAL_VALUES="${K_EVAL_VALUES:-1 5 10 15 20 25}"

export MOMENTUM_MODE="${MOMENTUM_MODE:-residual_output}"
# Vazio: usa o LB de cada rede (256/512). Ex.: export MOMENTUM_LOOKBACK=512.
export MOMENTUM_LOOKBACK="${MOMENTUM_LOOKBACK:-}"
# Valor inicial de regularizacao, nao resultado de ajuste empirico.
# Zero desliga somente a penalizacao; MOMENTUM_MODE=off desliga o modulo.
export MOMENTUM_ETA="${MOMENTUM_ETA:-0.01}"
export MOMENTUM_GAIN_INIT="${MOMENTUM_GAIN_INIT:-0}"
export MOMENTUM_DIAGNOSTICS="${MOMENTUM_DIAGNOSTICS:-0}"
export NS_MASK_A="${NS_MASK_A:-0.25}"
export NS_MASK_MODE="${NS_MASK_MODE:-natural}"
export KERNEL_SPECS="${KERNEL_SPECS:-cauchy;student 10;generalized_gaussian 1;generalized_gaussian 2;generalized_gaussian 4;compact 2 0;compact 1 1;compact 2 1;compact 2 2;compact 2 3;cosine;intermediate_band 2}"

export NS_BATCH_SIZE="${NS_BATCH_SIZE:-16}"
export NS_NUM_EPOCHS="${NS_NUM_EPOCHS:-100}"
export NS_PATIENCE="${NS_PATIENCE:-10}"
export NS_LEARNING_RATE="${NS_LEARNING_RATE:-0.0001}"
export NS_LRADJ="${NS_LRADJ:-}" # Vazio conserva a regra nativa de cada modelo.
export NS_TEMPORAL_LOSS="${NS_TEMPORAL_LOSS:-mse}"
export NS_TRAIN_RATIO_IN_TV="${NS_TRAIN_RATIO_IN_TV:-0.875}"
export NS_SEED="${NS_SEED:-2026}"
export NS_RANKNET_ALPHA="${NS_RANKNET_ALPHA:-1}"
export NS_SCORE_KIND="${NS_SCORE_KIND:-simple_return}"
export NS_CROSS_SCORE_NORMALIZATION="${NS_CROSS_SCORE_NORMALIZATION:-zscore}"

export NS_PARTITION="${NS_PARTITION:-medusas_dev}"
export NS_GRES="${NS_GRES-gpu:1}" # export NS_GRES="" preserva execucao sem GPU.
export NS_NODELIST="${NS_NODELIST:-}"
export NS_TIME_LIMIT="${NS_TIME_LIMIT:-01:30:00}"
export NS_CPUS_PER_TASK="${NS_CPUS_PER_TASK:-3}"
export NS_MEMORY="${NS_MEMORY:-}"
export NS_MAX_CONCURRENT_JOBS="${NS_MAX_CONCURRENT_JOBS:-1}"
export NS_ARRAY_CHUNK_SIZE="${NS_ARRAY_CHUNK_SIZE:-1000}"
export NS_PARALLEL_BRANCHES="${NS_PARALLEL_BRANCHES:-0}"
export NS_AFTER_JOB="${NS_AFTER_JOB:-}"
export NS_ONLY_BRANCHES="${NS_ONLY_BRANCHES:-}"

usage() {
    cat <<HELP
Uso: bash scripts/run_ns_momentum_kernels_isolado.sh --plan | --dry-run | --train
     bash scripts/run_ns_momentum_kernels_isolado.sh --backtest [planejar|submeter|consolidar]

DEFAULTS (exports existentes prevalecem):
Nonstationary_Transformer; treino80; LB256/512; H1/25; K_loss1; lambda0/.5/.8/1.
RankNet sem kernel + RankNet proximity com 12 kernels; A=0.25, modo natural.
Momentum residual_output nos compostos; L=LB da rede; eta=0.01 (editavel).
Lambda0 continua temporal puro. O momentum historico tambem e benchmark no R.
Grade default: 4 controles + 12 RankNet + 12 kernels x 12 proximity = 160 treinos.

Pool: cauchy; student nu10; generalized_gaussian p1/2/4;
compact (p,q)=(2,0)/(1,1)/(2,1)/(2,2)/(2,3); cosine; intermediate_band p2.
Ganho residual inicial zero (MOMENTUM_GAIN_INIT); eta=0 remove a penalizacao.
MOMENTUM_DIAGNOSTICS=1 registra a evolucao do treino; modo off desliga o modulo.
NS_LRADJ vazio preserva o modelo; type1 decai; constant mantem a taxa de aprendizado.

Todos os defaults deste arquivo aceitam export. Principais parametros:
EXPERIMENT_TAG, MODEL, TRAIN_VALUES, LOOKBACKS_VALUES, HORIZON_VALUES,
K_LOSS_VALUES, POSITIVE_LAMBDAS, CROSS_LOSSES_VALUES, DELTAS_VALUES, K_EVAL_VALUES,
MOMENTUM_MODE, MOMENTUM_LOOKBACK, MOMENTUM_ETA, MOMENTUM_GAIN_INIT,
MOMENTUM_DIAGNOSTICS, NS_LRADJ, NS_MASK_A, NS_MASK_MODE, KERNEL_SPECS.
Ex.: export CROSS_LOSSES_VALUES="hinge hinge_proximity"
Ex.: export KERNEL_SPECS="cauchy;student 10;cosine"
Cada proximity exige sua loss sem kernel. POSITIVE_LAMBDAS nao deve incluir zero.

Treino: batch16; ate100 epocas; patience10; LR1e-4; seed2026; validacao interna12.5%.
Recursos: medusas_dev; gpu:1; 3 CPUs/tarefa; 01:30:00; sem no/memoria fixados.
Por padrao, ramos encadeados afterany e uma tarefa por array ativo.
NS_PARALLEL_BRANCHES=1 retira o encadeamento ENTRE RAMOS;
NS_MAX_CONCURRENT_JOBS limita cada array, nao o total de nos.
NS_AFTER_JOB e uma dependencia inicial opcional; NS_ONLY_BRANCHES seleciona ramos.

--plan apenas conta; --dry-run confere os launchers; --train submete ao Slurm.
O backtest e separado: --backtest submeter. O R usa a raiz completa das previsoes.
Outras opcoes do launcher run_ns_kernels_isolado.sh continuam disponiveis via export,
incluindo NS_PYTHON_BIN, NS_RETURNS_FILE, NS_CPU_WORKERS, NS_CPU_N_LOTES e custos.
Os prefixos remotos continuam os do projeto; EXPERIMENT_TAG define a subpasta.
Use nova tag ao alterar o desenho. --plan mostra a grade e o caminho efetivos.
HELP
}

case "${1:---help}" in
    --help|-h)
        [[ $# -le 1 ]] || { printf 'ERRO: Argumentos excedentes.\n' >&2; exit 2; }
        usage
        exit 0
        ;;
esac

launcher="$TFB_ROOT/scripts/run_ns_kernels_isolado.sh"
[[ -f "$launcher" ]] || { printf 'ERRO: Launcher ausente: %s\n' "$launcher" >&2; exit 2; }
exec bash "$launcher" "$@"
