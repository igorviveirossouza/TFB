#!/usr/bin/env bash
# Edite a grade, os hiperparâmetros e o cluster diretamente neste arquivo.
# Use: bash scripts/run_composite_trading_experiment.sh --dry-run
set -euo pipefail

EXPERIMENT="composto"

# -----------------------------------------------------------------------------
# GRADE E TREINAMENTO — edite aqui
# -----------------------------------------------------------------------------
TRAIN_PERCENTAGES=(70 80 90)
MODELS=(DUET Nonstationary_Transformer FEDformer TimesNet simpleFORMER simpleMLP)
LOOKBACKS=(32 128 256)
HORIZONS=(1 5 10 15 20 25)

NUM_EPOCHS=100
PATIENCE=10
LEARNING_RATE=0.0001
TEMPORAL_LOSS="mse"
TRAIN_RATIO_IN_TV=0.875   # Validação = 12,5% do bloco anterior ao teste
SEED=2026
STRIDE=1
NUM_ROLLINGS=48000

# Apenas K <= H e H % K == 0 (16 pares na grade padrão).
# DELTAS varia em ranknet, hinge e pairwise_mse; ListNet roda uma vez por combinação.
TRADE_WINDOWS=(1 5 10 15 20 25)
CROSS_LOSSES=(ranknet hinge listnet pairwise_mse)
LAMBDAS=(0.1 0.5 0.8)
DELTAS=(0 0.025 0.05 0.1)

CROSS_SCALE=1
RANKNET_ALPHA=1
LISTNET_TAU=1
HINGE_MARGIN=0.1
SCORE_KIND="simple_return"
CROSS_SCORE_NORMALIZATION="zscore"
LOSS_INVERSE_NORM=true
LOSS_TRACK_COMPONENTS=true
# Todos os H passos ficam cobertos por blocos completos de K.
# ListNet não usa delta: pasta/auditoria usam delta=0 e delta_effective=False.
# Grade padrão: 33.696 tarefas (10.368 por loss de pares; 2.592 para ListNet).

# -----------------------------------------------------------------------------
# HIPERPARÂMETROS ESPECÍFICOS DOS MODELOS — também editáveis neste script
# lr, num_epochs, patience, seq_len e pred_len vêm das variáveis acima.
# Os campos restantes herdam os defaults dos adaptadores do TFB.
# -----------------------------------------------------------------------------
MODEL_DEFAULTS_JSON=$(cat <<'JSON'
{
  "DUET": {"CI": 1, "batch_size": 32, "d_ff": 32, "d_model": 32, "dropout": 0.5, "e_layers": 1, "factor": 3, "fc_dropout": 0.2, "hidden_size": 256, "k": 1, "lradj": "type1", "n_heads": 2, "norm": true, "num_experts": 4, "patch_len": 48},
  "TimesNet": {"batch_size": 32, "d_ff": 512, "d_model": 256, "factor": 3, "norm": true, "top_k": 5},
  "FEDformer": {"batch_size": 32, "d_ff": 512, "d_model": 256, "factor": 3, "norm": true},
  "Nonstationary_Transformer": {"d_ff": 256, "d_model": 128, "dropout": 0.1, "factor": 3, "norm": true, "p_hidden_dims": [32, 32], "p_hidden_layers": 2},
  "simpleFORMER": {"d_model": 512, "e_layers": 2, "d_ff": 2048, "n_heads": 8, "temporal_hidden_dim": 0, "dropout": 0.1, "norm": true, "use_norm": true, "norm_eps": 1e-05, "batch_size": 32, "lradj": "type1"},
  "simpleMLP": {"d_model": 512, "e_layers": 2, "d_ff": 2048, "n_heads": 8, "temporal_hidden_dim": 0, "dropout": 0.1, "norm": true, "use_norm": true, "norm_eps": 1e-05, "batch_size": 32, "lradj": "type1"}
}
JSON
)

# -----------------------------------------------------------------------------
# CAMINHOS E CLUSTER — edite estes valores neste próprio script
# -----------------------------------------------------------------------------
TFB_ROOT="${TFB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PATH="${VENV_PATH:-/sonic_home/igor.viveiros/py310/bin/activate}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_PATH="${DATASET_PATH:-$TFB_ROOT/dataset/forecasting/b3_daily_tfb_new.csv}"
OUT_ROOT="${OUT_ROOT:-/snfs2/igor.viveiros/previsoes/novas_previsoes}"
RESULT_ROOT="${RESULT_ROOT:-/snfs2/igor.viveiros/result/b3_new}"
LOG_ROOT="${LOG_ROOT:-$TFB_ROOT/logs/b3_new}"

PARTITION="${PARTITION:-medusas_dev}"    # Partição do Slurm
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"     # Tempo máximo por job
NODELIST="${NODELIST:-}"                # Ex.: medusa3; vazio = Slurm escolhe
CPUS_PER_TASK="${CPUS_PER_TASK:-3}"
MEMORY="${MEMORY:-}"                    # Ex.: 16G; vazio = padrão da partição
MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-6}"
ARRAY_CHUNK_SIZE="${ARRAY_CHUNK_SIZE:-1000}"
MANTER_CSV="${MANTER_CSV:-FALSE}"

# A execução abaixo é compartilhada; não é necessário editar arquivos Python.
B3_CONFIG_SOURCE="$TFB_ROOT/scripts/run_composite_trading_experiment.sh"
source "$TFB_ROOT/scripts/b3_experiment_env.sh"
exec "$PYTHON_BIN" "$TFB_ROOT/scripts/b3_experiment.py" "$EXPERIMENT" "$@"
