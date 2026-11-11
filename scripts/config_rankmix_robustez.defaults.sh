#!/usr/bin/env bash
# Defaults completos: edite e use source neste arquivo. Nao submete jobs.
# Carregar este arquivo define a grade; nenhuma submissao e automatica.
export TFB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export EXPERIMENT_TAG=isolados/rankmix_robustez_t60-90_v1

# Grade cientifica
export MODEL=iTransformer
export TRAIN_VALUES='60 70 80 90'
export LOOKBACKS_VALUES='32 256 512'
export HORIZON_VALUES='1 25'
export NS_RANKING_PROTOCOL=origin_anchored_windows
export K_LOSS_VALUES=1
export RANK_WINDOWS_VALUES='1 5 10 15 20 25'
export LAMBDAS_VALUES='0 0.1 0.5 0.8 1'
export SEEDS_VALUES='2026 2027 2028'
export TEMPORAL_LOSSES_VALUES=mse
export CROSS_LOSSES_VALUES='rankmix rankmix_proximity'
export RANK_MIX_WEIGHT_SPECS='0.5 0.5 0;0.25 0.75 0;0 0.75 0.25;0 1 0'
export RANK_MIX_MODES='global regional'
export RANK_MIX_REGIONS='top bottom top'
export RANK_MIX_FRACTION=0.3333333333333333
export RANK_MIX_INCLUDE_ABLATIONS=1
export RANK_MIX_ABLATION_SPECS='1 0 0;0 0 1'
export RANK_MIX_ABLATION_LAMBDAS=1
export KERNEL_SPECS=cauchy
export NS_MASK_A_VALUES='0.01 0.025 0.05'
export NS_MASK_MODE=natural

# Arquiteturas e treinamento
export NS_MODEL_DEFAULTS_JSON='{"iTransformer":{"d_model":128,"d_ff":256,"e_layers":2,"n_heads":8,"dropout":0.1,"norm":true,"lradj":"type1"},"Nonstationary_Transformer":{"d_model":128,"d_ff":256,"e_layers":2,"d_layers":1,"n_heads":8,"dropout":0.1,"factor":3,"norm":true,"p_hidden_dims":[32,32],"p_hidden_layers":2,"lradj":"type1"},"simpleFORMER":{"d_model":512,"d_ff":2048,"e_layers":2,"n_heads":8,"temporal_hidden_dim":0,"dropout":0.1,"norm":true,"use_norm":true,"norm_eps":0.00001,"lradj":"type1"}}'
export NS_BATCH_SIZE=16
export NS_NUM_EPOCHS=100
export NS_PATIENCE=10
export NS_LEARNING_RATE=0.0001
export NS_LRADJ=''
export NS_TRAIN_RATIO_IN_TV=0.875
export NS_VALIDATION_MODE=ordered_sample_mean
export NS_STRIDE=1
export NS_NUM_ROLLINGS=48000
export NS_CROSS_SCALE=1
export NS_LISTNET_TAU=1
export NS_LISTMLE_TAU=1
export NS_APPROXNDCG_TAU=1
export NS_HUBER_DELTA=0.5
export NS_SCORE_KIND=simple_return
export NS_CROSS_SCORE_NORMALIZATION=zscore
export NS_LOSS_INVERSE_NORM=true
export NS_LOSS_TRACK_COMPONENTS=true

# Recursos e submissao
export NS_PARTITION=gorgonas_dev
export NS_GRES=''
export NS_NODELIST=''
export NS_TIME_LIMIT=01:30:00
export NS_CPUS_PER_TASK=3
export NS_MEMORY=''
export NS_MAX_CONCURRENT_JOBS=3
export NS_ARRAY_CHUNK_SIZE=1000
export NS_PARALLEL_BRANCHES=0
export NS_AFTER_JOB=''
export NS_RESUME_CHAIN_OLD_JOBS=0
export NS_ONLY_BRANCHES=''
export NS_MANTER_CSV=FALSE

# Caminhos
export NS_PYTHON_BIN=/sonic_home/igor.viveiros/py310/bin/python
export NS_VENV_PATH=/sonic_home/igor.viveiros/py310/bin/activate
export NS_RETURNS_FILE="$TFB_ROOT/dataset/forecasting/b3_daily_tfb_new.csv"
export PRED_ROOT="/snfs2/igor.viveiros/previsoes/$EXPERIMENT_TAG"
export NS_LOG_ROOT="$TFB_ROOT/logs/$EXPERIMENT_TAG"
export NS_RESULT_ROOT="/snfs2/igor.viveiros/result/$EXPERIMENT_TAG"
export BACKTEST_ROOT="/snfs2/igor.viveiros/analise_resultados/backtests/$EXPERIMENT_TAG"

# Avaliacao opcional pelo backtest existente
export K_EVAL_VALUES='1 5 10 15 20 25'
export NS_MARKET_DIR=/snfs2/igor.viveiros/analise_resultados/insumos/base_backtest_temporal_001
export NS_STRATEGIES_FILE="$TFB_ROOT/config/estrategias_backtest.json"
export NS_CPU_WORKERS=16
export NS_CPU_N_LOTES=1
export NS_CPU_MEMORY=56G
export NS_SLIPPAGE_BPS=0
export NS_COST_BPS=0
