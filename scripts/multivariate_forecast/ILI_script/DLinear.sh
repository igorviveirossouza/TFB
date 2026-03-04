#!/bin/bash
#SBATCH --job-name=job_name
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gorgonas
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
cd /sonic_home/igor.viveiros/src/TFB || exit 1

/sonic_home/igor.viveiros/py310/bin/python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon": 24}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 16, "d_ff": 512, "d_model": 256, "horizon": 24, "lr": 0.01, "norm": true, "seq_len": 104}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ILI/DLinear"

/sonic_home/igor.viveiros/py310/bin/python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon": 36}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 64, "d_ff": 512, "d_model": 256, "horizon": 36, "lr": 0.01, "norm": true, "seq_len": 104}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ILI/DLinear"

/sonic_home/igor.viveiros/py310/bin/python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon": 48}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 64, "d_ff": 512, "d_model": 256, "horizon": 48, "lr": 0.01, "norm": true, "seq_len": 104}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ILI/DLinear"

/sonic_home/igor.viveiros/py310/bin/python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon": 60}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 64, "d_ff": 512, "d_model": 256, "horizon": 60, "lr": 0.01, "norm": true, "seq_len": 104}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ILI/DLinear"

/sonic_home/igor.viveiros/py310/bin/python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ILI.csv" --strategy-args '{"horizon": 96}' --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 8, "d_ff": 2048, "d_model": 512, "horizon": 96, "lr": 0.005, "norm": true, "seq_len": 512}' --adapter "transformer_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ILI/DLinear"
