#!/bin/bash
#SBATCH -p gorgonas
#SBATCH --cpus-per-task=32
#SBATCH --output=/sonic_home/igor.viveiros/logs/VARIOS_dm16-%j.out

source /sonic_home/igor.viveiros/py310/bin/activate
PYTHON_BIN=/sonic_home/igor.viveiros/py310/bin/python

export MPLCONFIGDIR=/tmp/$USER-mpl

cd /sonic_home/igor.viveiros/src/TFB || exit 1

SEQ_LEN=68

SALVAR_EM="seqlen${SEQ_LEN}/agregacao/h24"

# echo "TIMESNET"
# $PYTHON_BIN ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 24, "train_ratio_in_tv": {"__default__": 1.0, "b3_daily_tfb.csv": 1.0}}' \
# --model-name "time_series_library.TimesNet" --model-hyper-params '{"seq_len": '$SEQ_LEN', "pred_len": 24, "batch_size": 32, "d_ff": 512, "d_model": 256, "factor": 3, "horizon": 24, "norm": true, "top_k": 5}' --adapter "transformer_adapter"  --eval-backend sequential --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM" --save-true-pred False 

# echo "DUET"
# $PYTHON_BIN ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv"  --strategy-args '{"horizon": 24, "train_ratio_in_tv": {"__default__": 1.0, "b3_daily_tfb.csv": 1.0}}' \
# --model-name "duet.DUET" --model-hyper-params '{"seq_len":'$SEQ_LEN', "pred_len": 24, "CI": 1, "batch_size": 32, "d_ff": 32, "d_model": 32, "dropout": 0.5, "e_layers": 1, "factor": 3, "fc_dropout": 0.2, "hidden_size": 256, "k": 1, "loss": "MAE", "lr": 0.01, "lradj": "type1", "n_heads": 2, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 10}' --deterministic "full" --eval-backend sequential --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM" --save-true-pred False 

# echo "TIOMS"
# $PYTHON_BIN ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 24}' --model-name "TIOMS.BandWiseAdapter" \
# --model-hyper-params '{"seq_len": '$SEQ_LEN', "pred_len": 24, "label_len": 18,  "batch_size": 32, "aggregator_type": "pesos", "aggregator_hidden_dim": 64, "dropout": 0.1, "eps": 1e-5, "expert_type": "attention", "temporal_pool_type": "max", "expert_d_model": 64,  "expert_n_heads": 16, "channel_n_heads": 4}' --deterministic "full"   --eval-backend sequential  --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM" --save-true-pred False 

# echo "TIOMS"
# $PYTHON_BIN ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 24}' --model-name "TIOMS.BandWiseAdapter" \
# --model-hyper-params '{"seq_len": '$SEQ_LEN', "pred_len": 24, "label_len": 18,  "batch_size": 32, "aggregator_type": "pesos", "aggregator_hidden_dim": 64, "dropout": 0.1, "eps": 1e-5, "expert_type": "attention", "temporal_pool_type": "last", "expert_d_model": 64,  "expert_n_heads": 16, "channel_n_heads": 4}' --deterministic "full"   --eval-backend sequential  --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM" --save-true-pred False 

# echo "TIOMS"
# $PYTHON_BIN ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 24}' --model-name "TIOMS.BandWiseAdapter" \
# --model-hyper-params '{"seq_len": '$SEQ_LEN', "pred_len": 24, "label_len": 18,  "batch_size": 32, "aggregator_type": "pesos", "aggregator_hidden_dim": 64, "dropout": 0.1, "eps": 1e-5, "expert_type": "attention", "temporal_pool_type": "attn", "expert_d_model": 64,  "expert_n_heads": 16, "channel_n_heads": 4}' --deterministic "full"   --eval-backend sequential  --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM" --save-true-pred False 

# echo "TIOMS"
# $PYTHON_BIN ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 24}' --model-name "TIOMS.BandWiseAdapter" \
# --model-hyper-params '{"seq_len": '$SEQ_LEN', "pred_len": 24, "label_len": 18,  "batch_size": 32, "aggregator_type": "pesos", "aggregator_hidden_dim": 64, "dropout": 0.1, "eps": 1e-5, "expert_type": "attention", "temporal_pool_type": "flat", "expert_d_model": 64,  "expert_n_heads": 16, "channel_n_heads": 4}' --deterministic "full"   --eval-backend sequential  --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM" --save-true-pred False 

# echo "TIOMS NO BAND SEM CHANEL"
# CUDA_VISIBLE_DEVICES="" $PYTHON_BIN ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 24}' --model-name "TIOMS.NoBandWiseAdapter" \
# --model-hyper-params '{"seq_len": '$SEQ_LEN', "pred_len": 24, "label_len": 18,  "batch_size": 32, "dropout": 0.1, "eps": 1e-5, "expert_type": "attention", "expert_d_model": 64,  "expert_n_heads": 16}' --deterministic "full"   --eval-backend sequential  --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM" --save-true-pred False 

# echo "TIOMS"
# $PYTHON_BIN ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 24}' --model-name "TIOMS.NoBandWiseAdapterChanel" \
# --model-hyper-params '{"seq_len": '$SEQ_LEN', "pred_len": 24, "label_len": 18,  "batch_size": 32, "aggregator_type": "pesos", "aggregator_hidden_dim": 64, "dropout": 0.1, "eps": 1e-5, "expert_type": "attention", "temporal_pool_type": "last", "expert_d_model": 64,  "expert_n_heads": 16, "channel_n_heads": 4}' --deterministic "full"   --eval-backend sequential  --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM" --save-true-pred False 

# echo "TIOMS COM BAND SEM CHANEL"
# $PYTHON_BIN ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 24}' --model-name "TIOMS.LearnableBandWiseAdapter" \
# --model-hyper-params '{"seq_len": '$SEQ_LEN', "pred_len": 24, "label_len": 18,  "batch_size": 32, "dropout": 0.1, "eps": 1e-5, "expert_type": "attention", "expert_d_model": 64,  "expert_n_heads": 16, "temporal_pool_type": "flat"}' --deterministic "full"   --eval-backend sequential  --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM" --save-true-pred False 

# echo "TIOMS COM BAND SEM CHANEL"
# $PYTHON_BIN ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 24}' --model-name "TIOMS.LearnableBandWiseAdapter" \
# --model-hyper-params '{"seq_len": '$SEQ_LEN', "pred_len": 24, "label_len": 18,  "batch_size": 32, "dropout": 0.1, "eps": 1e-5, "expert_type": "attention", "expert_d_model": 64,  "expert_n_heads": 16, "temporal_pool_type": "avg"}' --deterministic "full"   --eval-backend sequential  --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM" --save-true-pred False 

# echo "TIOMS COM BAND SEM CHANEL"
$PYTHON_BIN ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 24}' --model-name "TIOMS.LearnableBandWiseAdapterAudit" \
--model-hyper-params '{"channel_agg_type": "none", "seq_len": '$SEQ_LEN', "pred_len": 24, "label_len": 18,  "batch_size": 32, "dropout": 0.1, "eps": 1e-5, "expert_type": "attention", "expert_d_model": 64,  "expert_n_heads": 8, "channel_n_heads": 4, "temporal_pool_type": "last"}' --deterministic "full"   --eval-backend sequential  --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM" --save-true-pred False 

$PYTHON_BIN ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 24}' --model-name "TIOMS.LearnableBandWiseAdapterAudit" \
--model-hyper-params '{"channel_agg_type": "linear", "seq_len": '$SEQ_LEN', "pred_len": 24, "label_len": 18,  "batch_size": 32, "dropout": 0.1, "eps": 1e-5, "expert_type": "attention", "expert_d_model": 64,  "expert_n_heads": 8, "channel_n_heads": 4, "temporal_pool_type": "last"}' --deterministic "full"   --eval-backend sequential  --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM" --save-true-pred False 

$PYTHON_BIN ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 24}' --model-name "TIOMS.LearnableBandWiseAdapterAudit" \
--model-hyper-params '{"channel_agg_type": "residual_gated", "seq_len": '$SEQ_LEN', "pred_len": 24, "label_len": 18,  "batch_size": 32, "dropout": 0.1, "eps": 1e-5, "expert_type": "attention", "expert_d_model": 64,  "expert_n_heads": 8, "channel_n_heads": 4, "temporal_pool_type": "last"}' --deterministic "full"   --eval-backend sequential  --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM" --save-true-pred False 


$PYTHON_BIN ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 24}' --model-name "TIOMS.LearnableBandWiseAdapterAudit" \
--model-hyper-params '{"channel_agg_type": "attention", "seq_len": '$SEQ_LEN', "pred_len": 24, "label_len": 18,  "batch_size": 32, "dropout": 0.1, "eps": 1e-5, "expert_type": "attention", "expert_d_model": 64,  "expert_n_heads": 8, "channel_n_heads": 4, "temporal_pool_type": "last"}' --deterministic "full"   --eval-backend sequential  --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM" --save-true-pred False 


$PYTHON_BIN ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 24}' --model-name "TIOMS.LearnableBandWiseAdapterAudit" \
--model-hyper-params '{"channel_agg_type": "linear_residual", "seq_len": '$SEQ_LEN', "pred_len": 24, "label_len": 18,  "batch_size": 32, "dropout": 0.1, "eps": 1e-5, "expert_type": "attention", "expert_d_model": 64,  "expert_n_heads": 8, "channel_n_heads": 4, "temporal_pool_type": "last"}' --deterministic "full"   --eval-backend sequential  --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM" --save-true-pred False 

$PYTHON_BIN ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 24}' --model-name "TIOMS.LearnableBandWiseAdapterAudit" \
--model-hyper-params '{"channel_agg_type": "linear_lowrank_residual", "seq_len": '$SEQ_LEN', "pred_len": 24, "label_len": 18,  "batch_size": 32, "dropout": 0.1, "eps": 1e-5, "expert_type": "attention", "expert_d_model": 64,  "expert_n_heads": 8, "channel_n_heads": 4, "temporal_pool_type": "last"}' --deterministic "full"   --eval-backend sequential  --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM" --save-true-pred False 

$PYTHON_BIN ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 24}' --model-name "TIOMS.LearnableBandWiseAdapterAudit" \
--model-hyper-params '{"channel_agg_type": "mlp_mixer", "seq_len": '$SEQ_LEN', "pred_len": 24, "label_len": 18,  "batch_size": 32, "dropout": 0.1, "eps": 1e-5, "expert_type": "attention", "expert_d_model": 64,  "expert_n_heads": 8, "channel_n_heads": 4, "temporal_pool_type": "last"}' --deterministic "full"   --eval-backend sequential  --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM" --save-true-pred False 


# echo "TIOMS COM BAND SEM CHANEL"
# $PYTHON_BIN ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 24}' --model-name "TIOMS.LearnableBandWiseAdapter" \
# --model-hyper-params '{"seq_len": '$SEQ_LEN', "pred_len": 24, "label_len": 18,  "batch_size": 32, "dropout": 0.1, "eps": 1e-5, "expert_type": "attention", "expert_d_model": 64,  "expert_n_heads": 16, "temporal_pool_type": "attn"}' --deterministic "full"   --eval-backend sequential  --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM" --save-true-pred False 


# echo "TIOMS COM BAND SEM CHANEL"
# $PYTHON_BIN ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 24}' --model-name "TIOMS.LearnableBandWiseAdapter" \
# --model-hyper-params '{"seq_len": '$SEQ_LEN', "pred_len": 24, "label_len": 18,  "batch_size": 32, "dropout": 0.1, "eps": 1e-5, "expert_type": "attention", "expert_d_model": 64,  "expert_n_heads": 16, "temporal_pool_type": "max"}' --deterministic "full"   --eval-backend sequential  --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM" --save-true-pred False 


echo "Benchmarks finalizados"

# echo "Decodificando previsões..."

# RESULT_BASE="result/experimentos/$SALVAR_EM"
# DECODE_SCRIPT="ts_benchmark/utils/decode_prediction.py"

# find "$RESULT_BASE" -type f -name "*.tar.gz" | while read -r tarfile; do
#     echo "Extraindo: $tarfile"

#     extract_dir="${tarfile%.tar.gz}_extracted"
#     mkdir -p "$extract_dir"

#     tar -xzf "$tarfile" -C "$extract_dir"

#     find "$extract_dir" -type f -name "*.csv" | while read -r csvfile; do
#         echo "Decodificando CSV: $csvfile"
#         python "$DECODE_SCRIPT" "$csvfile"
#     done
# done

# echo "Decodificação finalizada"