#!/bin/bash
#SBATCH -p gorgonas
#SBATCH --cpus-per-task=32
#SBATCH --output=/sonic_home/igor.viveiros/logs/VARIOS_dm16-%j.out

source /sonic_home/igor.viveiros/py310/bin/activate
PYTHON_BIN=/sonic_home/igor.viveiros/py310/bin/python

export MPLCONFIGDIR=/tmp/$USER-mpl

cd /sonic_home/igor.viveiros/src/TFB || exit 1

SEQ_LEN=36
PRED_LEN=24

SALVAR_EM="B3/seqlen${SEQ_LEN}/h${PRED_LEN}"

echo "DUET"
$PYTHON_BIN  ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": '$PRED_LEN'}' --model-name "duet.DUET" --model-hyper-params '{"CI": 1, "batch_size": 8, "d_ff": 1024, "d_model": 512, "dropout": 0.2, "e_layers": 2, "factor": 3, "pred_len": '$PRED_LEN', "k": 2, "loss": "MAE", "lr": 0.0005, "lradj": "type1", "n_heads": 1, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 5, "seq_len": '$SEQ_LEN'}' --deterministic "full" --gpus 0 --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM" --save-true-pred False

echo "TIMESNET"
$PYTHON_BIN  ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": '$PRED_LEN'}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 512, "d_model": 256, "factor": 3, "pred_len": '$PRED_LEN', "norm": true, "seq_len": '$SEQ_LEN', "top_k": 5}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM" --save-true-pred False

echo "TIOMS LINEAR"
$PYTHON_BIN ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": '$PRED_LEN'}' --model-name "TIOMS.LearnableBandWiseAdapterAudit" \
--model-hyper-params '{"channel_agg_type": "linear", "seq_len": '$SEQ_LEN', "pred_len": '$PRED_LEN', "label_len": 18,  "batch_size": 32, "dropout": 0.1, "eps": 1e-5, "expert_type": "attention", "expert_d_model": 64,  "expert_n_heads": 8, "channel_n_heads": 4, "temporal_pool_type": "last"}' --deterministic "full"   --eval-backend sequential  --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM" --save-true-pred False 

$PYTHON_BIN ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": '$PRED_LEN'}' --model-name "TIOMS.LearnableBandWiseAdapterAudit" \
--model-hyper-params '{"channel_agg_type": "none", "seq_len": '$SEQ_LEN', "pred_len": '$PRED_LEN', "label_len": 18,  "batch_size": 32, "dropout": 0.1, "eps": 1e-5, "expert_type": "attention", "expert_d_model": 64,  "expert_n_heads": 8, "channel_n_heads": 4, "temporal_pool_type": "last"}' --deterministic "full"   --eval-backend sequential  --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM" --save-true-pred False 

echo "FEDFormer"
$PYTHON_BIN ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": '$PRED_LEN'}' --model-name "time_series_library.FEDformer" --model-hyper-params '{"batch_size": 8, "d_ff": 512, "d_model": 256, "dropout": 0.05, "factor": 3, "pred_len": '$PRED_LEN', "lr": 0.001, "moving_avg": 25, "norm": true, "seq_len": '$SEQ_LEN'}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM" --save-true-pred False

echo "NonStationary Transformer"

$PYTHON_BIN ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": '$PRED_LEN'}' --model-name "time_series_library.Nonstationary_Transformer" --model-hyper-params '{"d_ff": 2048, "d_model": 512, "dropout": 0.05, "factor": 3, "pred_len": '$PRED_LEN', "norm": true, "p_hidden_dims": [32, 32], "p_hidden_layers": 2, "seq_len": '$SEQ_LEN'}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM" --save-true-pred False


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