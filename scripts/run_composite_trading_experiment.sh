#!/usr/bin/env bash
#SBATCH --exclude=medusa5
# Edite a grade, os hiperparâmetros e o cluster diretamente neste arquivo.
# Use: bash scripts/run_composite_trading_experiment.sh --dry-run
set -euo pipefail

# -----------------------------------------------------------------------------
# Modo de execução:
# temporal	- Use run_temporal_experiment.sh para a loss temporal
# composto	- Treinamento com loss temporal mais componente cross-sectional
# var	      - Use run_var_experiment.sh para o baseline VAR.
# -----------------------------------------------------------------------------

if [[ "${EXPERIMENT:-composto}" != composto ]]; then
  echo "Este script executa apenas composto; use run_temporal_experiment.sh para temporal ou run_var_experiment.sh para VAR." >&2
  exit 2
fi
EXPERIMENT="composto"

# -----------------------------------------------------------------------------
# GRADE E TREINAMENTO — edite aqui
# -----------------------------------------------------------------------------
#TRAIN_PERCENTAGES=(70 80 90)
read -r -a TRAIN_PERCENTAGES <<< "${TRAIN_PERCENTAGES[*]:-80}"
read -r -a MODELS <<< "${MODELS[*]:-DUET Nonstationary_Transformer FEDformer TimesNet simpleFORMER simpleMLP}"
read -r -a LOOKBACKS <<< "${LOOKBACKS[*]:-32 128 256}"
read -r -a HORIZONS <<< "${HORIZONS[*]:-1 5 10 15 20 25}"

NUM_EPOCHS="${NUM_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PATIENCE="${PATIENCE:-10}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
TEMPORAL_LOSS="${TEMPORAL_LOSS:-mse}"
TRAIN_RATIO_IN_TV="${TRAIN_RATIO_IN_TV:-0.875}"   # Validação = 12,5% do bloco anterior ao teste
SEED="${SEED:-2026}"
STRIDE="${STRIDE:-1}"
NUM_ROLLINGS="${NUM_ROLLINGS:-48000}"

# Apenas K <= H e H % K == 0 (16 pares na grade padrão).
# DELTAS varia em ranknet, hinge e pairwise_mse; ListNet roda uma vez por combinação.
#TRADE_WINDOWS=(1 5 10 15 20 25)

read -r -a TRADE_WINDOWS <<< "${TRADE_WINDOWS[*]:-1}"
read -r -a CROSS_LOSSES <<< "${CROSS_LOSSES[*]:-hinge_proximity ranknet_proximity pairwise_mse_proximity}"
#CROSS_LOSSES=(ranknet hinge listnet pairwise_mse)
read -r -a LAMBDAS <<< "${LAMBDAS[*]:-0.1 0.5 0.8}"
#DELTAS=(0 0.025 0.05 0.1)
read -r -a DELTAS <<< "${DELTAS[*]:-0}"

# Novas losses opcionais: hinge_proximity, ranknet_proximity, pairwise_mse_proximity.
# Para executar apenas as novas, substitua CROSS_LOSSES acima por:
# CROSS_LOSSES=(hinge_proximity ranknet_proximity pairwise_mse_proximity)
# Vetor: máscara, a (escala do score realizado), modo natural/complementary.
# uniform: |R_i-R_j| <= a (a >= 0); gaussian: exp(-gap²/(2a²)) (a > 0).
# cauchy: 1/(1+(gap/a)²) (a > 0); complementary usa 1-peso e zera empates.
# Dois elementos continuam válidos: modo natural. DELTAS não filtra essas losses.
# Cada execução usa um único vetor; mude-o para comparar outra máscara/a/modo.

read -r -a CROSS_MASK <<< "${CROSS_MASK[*]:-gaussian 0.025 natural}"   # Ex.: CROSS_MASK=(cauchy 0.025 complementary)

CROSS_SCALE="${CROSS_SCALE:-1}"
RANKNET_ALPHA="${RANKNET_ALPHA:-1}"
LISTNET_TAU="${LISTNET_TAU:-1}"
HINGE_MARGIN="${HINGE_MARGIN:-0.1}"
SCORE_KIND="${SCORE_KIND:-simple_return}"
CROSS_SCORE_NORMALIZATION="${CROSS_SCORE_NORMALIZATION:-zscore}"
LOSS_INVERSE_NORM="${LOSS_INVERSE_NORM:-true}"
LOSS_TRACK_COMPONENTS="${LOSS_TRACK_COMPONENTS:-true}"
# Todos os H passos ficam cobertos por blocos completos de K.
# ListNet não usa delta: pasta/auditoria usam delta=0 e delta_effective=False.
# Grade padrão: 33.696 tarefas (10.368 por loss de pares; 2.592 para ListNet).

# -----------------------------------------------------------------------------
# HIPERPARÂMETROS ESPECÍFICOS DOS MODELOS — também editáveis neste script
# lr, num_epochs, patience, seq_len e pred_len vêm das variáveis acima.
# Os campos restantes herdam os defaults dos adaptadores do TFB.
# -----------------------------------------------------------------------------
MODEL_DEFAULTS_JSON=$(cat <<JSON
{
  "DUET": {"CI": 1, "batch_size": ${BATCH_SIZE}, "d_ff": 32, "d_model": 32, "dropout": 0.5, "e_layers": 1, "factor": 3, "fc_dropout": 0.2, "hidden_size": 256, "k": 1, "lradj": "type1", "n_heads": 2, "norm": true, "num_experts": 4, "patch_len": 48},
  "TimesNet": {"batch_size": ${BATCH_SIZE}, "d_ff": 512, "d_model": 256, "factor": 3, "norm": true, "top_k": 5},
  "FEDformer": {"batch_size": ${BATCH_SIZE}, "d_ff": 512, "d_model": 256, "factor": 3, "norm": true},
  "Nonstationary_Transformer": {"batch_size": ${BATCH_SIZE}, "d_ff": 256, "d_model": 128, "dropout": 0.1, "factor": 3, "norm": true, "p_hidden_dims": [32, 32], "p_hidden_layers": 2},
  "simpleFORMER": {"d_model": 512, "e_layers": 2, "d_ff": 2048, "n_heads": 8, "temporal_hidden_dim": 0, "dropout": 0.1, "norm": true, "use_norm": true, "norm_eps": 1e-05, "batch_size": ${BATCH_SIZE}, "lradj": "type1"},
  "simpleMLP": {"d_model": 512, "e_layers": 2, "d_ff": 2048, "n_heads": 8, "temporal_hidden_dim": 0, "dropout": 0.1, "norm": true, "use_norm": true, "norm_eps": 1e-05, "batch_size": ${BATCH_SIZE}, "lradj": "type1"}
}
JSON
)

# -----------------------------------------------------------------------------
# CAMINHOS E CLUSTER — edite estes valores neste próprio script
# -----------------------------------------------------------------------------
TFB_ROOT="${TFB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MASK_MODE_TAG=""
if [[ "${CROSS_MASK[2]:-natural}" != natural ]]; then
  MASK_MODE_TAG="_${CROSS_MASK[2]}"
fi
export RUN_TAG="${RUN_TAG:-composto_${CROSS_MASK[0]}_a${CROSS_MASK[1]}${MASK_MODE_TAG}_k1_t80_v1}"


VENV_PATH="${VENV_PATH:-/sonic_home/igor.viveiros/py310/bin/activate}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_PATH="${DATASET_PATH:-$TFB_ROOT/dataset/forecasting/b3_daily_tfb_new.csv}"
export SOURCE_DATA_KIND="${SOURCE_DATA_KIND:-log_return}"  # log_return: base nova já corrigida; price: calcular diff(log(P))


OUT_ROOT="/snfs2/igor.viveiros/previsoes/${RUN_TAG}"
RESULT_ROOT="/snfs2/igor.viveiros/result/${RUN_TAG}"
LOG_ROOT="$TFB_ROOT/logs/${RUN_TAG}"


PARTITION="${PARTITION:-gorgonas}"    # Partição do Slurm
# GPU: valor passado literalmente a sbatch --gres (ex.: gpu:1 ou gpu:<tipo>:1).
# O tipo deve existir no cluster; vazio = não solicitar GPU.
# Não fixe CUDA_VISIBLE_DEVICES: preserve a GPU alocada pelo Slurm.
GRES="${GRES-gpu:0}"
TIME_LIMIT="${TIME_LIMIT:-01:30:00}"     # Tempo máximo por job
NODELIST="${NODELIST:-}"                # Ex.: medusa3; vazio = Slurm escolhe
CPUS_PER_TASK="${CPUS_PER_TASK:-3}"
MEMORY="${MEMORY:-}"                    # Ex.: 16G; vazio = padrão da partição
MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-6}"
ARRAY_CHUNK_SIZE="${ARRAY_CHUNK_SIZE:-1000}"
MANTER_CSV="${MANTER_CSV:-FALSE}"

# --dry-run mostra o sbatch do primeiro array: GPU, tempo, CPUs e logs.
# A execução abaixo é compartilhada; não é necessário editar arquivos Python.
B3_CONFIG_SOURCE="$TFB_ROOT/scripts/run_composite_trading_experiment.sh"
source "$TFB_ROOT/scripts/b3_experiment_env.sh"
exec "$PYTHON_BIN" "$TFB_ROOT/scripts/b3_experiment.py" "$EXPERIMENT" "$@"
