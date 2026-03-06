#!/bin/bash

# ==============================
# MyTimesNet - Execução no SPEED
# ==============================

# Ativar ambiente (ajuste se necessário)
source /sonic_home/igor.viveiros/py310/bin/activate

export MPLCONFIGDIR=/tmp/$USER-mpl

cd /sonic_home/igor.viveiros/src/TFB || exit 1

echo "Starting MY TIMES NET..."

echo "Rolling Forecast - ETTh1"
#python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAdapter" --model-hyper-params '{"batch_size":16, "num_epochs":2, "seq_len":96, "pred_len":336, "d_model":128, "d_ff":256, "top_k":2}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNet_local"
#python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --deterministic "efficient" --model-name "mytimesnet.MyTimesOriginalNetAdapter" --model-hyper-params '{"batch_size":16, "num_epochs":2, "seq_len":96, "pred_len":336, "d_model":128, "d_ff":256, "top_k":2}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNet_local"

#python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAdapter" --model-hyper-params '{"batch_size":16, "num_epochs":2, "seq_len":96, "pred_len":336, "d_model":128, "d_ff":256, "top_k":2}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNet_local"
#python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesOriginalNetAdapter" --model-hyper-params '{"batch_size":16, "num_epochs":2, "seq_len":96, "pred_len":336, "d_model":128, "d_ff":256, "top_k":2}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNet_local"


echo "Iniciando MyTimesNet - FIXED FORECAST ILI"
# Modificado
python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_weekly.json" --data-name-list "ILI.csv" --deterministic "efficient" --strategy-args '{"horizon":24}' --model-name "mytimesnet.MyTimesNetAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":104,"pred_len":24,"d_model":128,"d_ff":256,"top_k":2}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNet_fixed"
# Original
python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_weekly.json" --data-name-list "ILI.csv" --deterministic "efficient" --strategy-args '{"horizon":24}' --model-name "mytimesnet.MyTimesNetOriginalAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":104,"pred_len":24,"d_model":128,"d_ff":256,"top_k":2}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNet_fixed"




echo "Finalizado."
