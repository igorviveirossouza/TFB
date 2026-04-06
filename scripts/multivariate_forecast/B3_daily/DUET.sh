#!/bin/bash
#SBATCH -p gorgonas
#SBATCH --cpus-per-task=32
#SBATCH --output=/sonic_home/igor.viveiros/logs/DUET-%j.out

# Ativar ambiente
source /sonic_home/igor.viveiros/py310/bin/activate
PYTHON_BIN=/sonic_home/igor.viveiros/py310/bin/python

export MPLCONFIGDIR=/tmp/$USER-mpl

# Prefixo central de salvamento
SALVAR_EM="DUET/b3_teste_1"

cd /sonic_home/igor.viveiros/src/TFB || exit 1

echo "Starting DUET..."
echo "Salvando resultados em: experimentos/$SALVAR_EM"

$PYTHON_BIN ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 24}' --model-name "duet.DUET" --model-hyper-params '{"CI": 1, "batch_size": 32, "d_ff": 32, "d_model": 32, "dropout": 0.5, "e_layers": 1, "factor": 3, "fc_dropout": 0.2, "hidden_size": 256, "horizon": 24, "k": 1, "loss": "MAE", "lr": 0.01, "lradj": "type1", "n_heads": 2, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 10, "seq_len": 104}' --deterministic "full" --eval-backend sequential --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM/h24/" --save-true-pred False 

# python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 36}' --model-name "duet.DUET" --model-hyper-params '{"CI": 1, "batch_size": 32, "d_ff": 16, "d_model": 16, "dropout": 0.5, "e_layers": 1, "factor": 3, "fc_dropout": 0.2, "horizon": 36, "k": 1, "loss": "MAE", "lr": 0.01, "lradj": "type1", "n_heads": 2, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 10, "seq_len": 36}' --deterministic "full" --gpus 1 --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM/h36" --save-true-pred True 

# python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 48}' --model-name "duet.DUET" --model-hyper-params '{"CI": 1, "batch_size": 32, "d_ff": 16, "d_model": 16, "dropout": 0.5, "e_layers": 1, "factor": 3, "fc_dropout": 0.2, "horizon": 48, "k": 1, "loss": "MAE", "lr": 0.01, "lradj": "type1", "n_heads": 2, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 10, "seq_len": 36}' --deterministic "full" --gpus 1 --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM/h48" --save-true-pred True 

# python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 60}' --model-name "duet.DUET" --model-hyper-params '{"CI": 1, "batch_size": 32, "d_ff": 16, "d_model": 16, "dropout": 0.5, "e_layers": 1, "factor": 3, "fc_dropout": 0.2, "horizon": 60, "k": 1, "loss": "MAE", "lr": 0.01, "lradj": "type1", "n_heads": 2, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 10, "seq_len": 36}' --deterministic "full" --gpus 1 --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM/h60" --save-true-pred True 



# echo "Benchmarks finalizados"
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