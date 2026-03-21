#!/bin/bash
#SBATCH -p gorgonas
#SBATCH --cpus-per-task=32
#SBATCH --output=/sonic_home/igor.viveiros/logs/slurm-%j.out

# Ativar ambiente (ajuste se necessário)
source /sonic_home/igor.viveiros/py310/bin/activate

export MPLCONFIGDIR=/tmp/$USER-mpl

cd /sonic_home/igor.viveiros/src/TFB || exit 1

echo "Starting MY TIMES NET..."



python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 24}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 512, "d_model": 256, "factor": 3, "horizon": 24, "norm": true, "seq_len": 36, "top_k": 5}' --adapter "transformer_adapter" --gpus 1 --num-workers 1 --timeout 60000 --save-path "experimentos/TimesNet" &

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 36}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 2048, "d_model": 512, "factor": 3, "horizon": 36, "norm": true, "seq_len": 36, "top_k": 5}' --adapter "transformer_adapter" --gpus 1 --num-workers 1 --timeout 60000 --save-path "experimentos/TimesNet" &

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 48}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 256, "d_model": 128, "factor": 3, "horizon": 48, "norm": true, "seq_len": 36, "top_k": 5}' --adapter "transformer_adapter" --gpus 1 --num-workers 1 --timeout 60000 --save-path "experimentos/TimesNet" & 

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 60}' --model-name "time_series_library.TimesNet" --model-hyper-params '{"d_ff": 768, "d_model": 768, "factor": 3, "horizon": 60, "norm": true, "seq_len": 36, "top_k": 5}' --adapter "transformer_adapter" --gpus 1 --num-workers 1 --timeout 60000 --save-path "experimentos/TimesNet" &

wait

echo "Finalizado"
