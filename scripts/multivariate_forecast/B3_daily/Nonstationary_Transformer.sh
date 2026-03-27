#!/bin/bash
#SBATCH -p gorgonas
#SBATCH --cpus-per-task=32
#SBATCH --output=/sonic_home/igor.viveiros/logs/slurm-%j.out

# Ativar ambiente (ajuste se necessário)
source /sonic_home/igor.viveiros/py310/bin/activate

export MPLCONFIGDIR=/tmp/$USER-mpl

cd /sonic_home/igor.viveiros/src/TFB || exit 1

echo "Starting MY TIMES NET..."

SALVAR_EM="NonStaionaryTransformer/b3_teste_1"

echo "Salvando resultados em: experimentos/$SALVAR_EM"


python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 24}' --model-name "time_series_library.Nonstationary_Transformer" --model-hyper-params '{"d_ff": 256, "d_model": 128, "dropout": 0.05, "factor": 3, "horizon": 24, "norm": true, "p_hidden_dims": [32, 32], "p_hidden_layers": 2, "seq_len": 36}' --adapter "transformer_adapter" --gpus 1 --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM/h24" &

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 36}' --model-name "time_series_library.Nonstationary_Transformer" --model-hyper-params '{"d_ff": 256, "d_model": 128, "dropout": 0.05, "factor": 3, "horizon": 36, "norm": true, "p_hidden_dims": [32, 32], "p_hidden_layers": 2, "seq_len": 36}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM/h36" & 

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 48}' --model-name "time_series_library.Nonstationary_Transformer" --model-hyper-params '{"d_ff": 256, "d_model": 128, "dropout": 0.05, "factor": 3, "horizon": 48, "norm": true, "p_hidden_dims": [32, 32], "p_hidden_layers": 2, "seq_len": 36}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM/h48" &

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 60}' --model-name "time_series_library.Nonstationary_Transformer" --model-hyper-params '{"d_ff": 256, "d_model": 128, "dropout": 0.05, "factor": 3, "horizon": 60, "norm": true, "p_hidden_dims": [32, 32], "p_hidden_layers": 2, "seq_len": 36}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "experimentos/$SALVAR_EM/h60" &


wait

echo "Benchmarks finalizados"
echo "Decodificando previsões..."

RESULT_BASE="result/experimentos/$SALVAR_EM"
DECODE_SCRIPT="ts_benchmark/utils/decode_prediction.py"

find "$RESULT_BASE" -type f -name "*.tar.gz" | while read -r tarfile; do
    echo "Extraindo: $tarfile"

    extract_dir="${tarfile%.tar.gz}_extracted"
    mkdir -p "$extract_dir"

    tar -xzf "$tarfile" -C "$extract_dir"

    find "$extract_dir" -type f -name "*.csv" | while read -r csvfile; do
        echo "Decodificando CSV: $csvfile"
        python "$DECODE_SCRIPT" "$csvfile"
    done
done

echo "Decodificação finalizada"