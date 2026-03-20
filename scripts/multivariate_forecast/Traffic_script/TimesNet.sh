#!/bin/bash
#SBATCH -p gorgonas
#SBATCH --cpus-per-task=32
#SBATCH --output=/sonic_home/igor.viveiros/logs/slurm-%j.out

# Ativar ambiente (ajuste se necessário)
source /sonic_home/igor.viveiros/py310/bin/activate

export MPLCONFIGDIR=/tmp/$USER-mpl

cd /sonic_home/igor.viveiros/src/TFB || exit 1

echo "Starting MY TIMES NET..."

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"horizon": 96}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"batch_size": 8, "d_ff": 512, "d_model": 256, "horizon": 96, "norm": true, "seq_len": 96}' --adapter "transformer_adapter" --gpus 1 --num-workers 8 --timeout 60000 --save-path "experimentos/Traffic/TimesNet_gpu" &

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"horizon": 192}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 2048, "d_model": 512, "factor": 3, "horizon": 192, "norm": true, "seq_len": 96, "top_k": 5}' --adapter "transformer_adapter" --gpus 1 --num-workers 8 --timeout 60000 --save-path "experimentos/Traffic/TimesNet_gpu" &

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"horizon": 336}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"batch_size": 16, "d_ff": 512, "d_model": 256, "horizon": 336, "norm": true, "seq_len": 96}' --adapter "transformer_adapter" --gpus 1 --num-workers 8 --timeout 60000 --save-path "experimentos/Traffic/TimesNet_gpu" &

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"horizon": 720}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"batch_size": 16, "d_ff": 512, "d_model": 512, "factor": 3, "horizon": 720, "norm": true, "seq_len": 192, "top_k": 5}' --adapter "transformer_adapter" --gpus 1 --num-workers 8 --timeout 60000 --save-path "experimentos/Traffic/TimesNet_gpu" &

wait

echo "Finalizado"