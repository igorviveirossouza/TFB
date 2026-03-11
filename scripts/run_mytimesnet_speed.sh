#!/bin/bash

# ==============================
# MyTimesNet - Execução no SPEED
# ==============================

# Ativar ambiente (ajuste se necessário)
source /sonic_home/igor.viveiros/py310/bin/activate

export MPLCONFIGDIR=/tmp/$USER-mpl

cd /sonic_home/igor.viveiros/src/TFB || exit 1

echo "Starting MY TIMES NET..."

# echo "Rolling Forecast - ETTh1"

#python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAdapter" --model-hyper-params '{"batch_size":16, "num_epochs":2, "seq_len":96, "pred_len":336, "d_model":128, "d_ff":256, "top_k":2}' --num-workers 2 --timeout 60000 --save-path "results_MyTimesNet_modificado_autovalor"
#python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetOriginalAdapter" --model-hyper-params '{"batch_size":16, "num_epochs":2, "seq_len":96, "pred_len":336, "d_model":128, "d_ff":256, "top_k":2}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNetOriginal"


#echo "Rolling Forecast - Exchange"

#python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAdapter" --model-hyper-params '{"batch_size":16, "num_epochs":2, "seq_len":96, "pred_len":96, "d_model":128, "d_ff":256, "top_k":5}' --num-workers 4 --timeout 60000 --save-path "results_Exchange1"
#python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetOriginalAdapter" --model-hyper-params '{"batch_size":16, "num_epochs":2, "seq_len":96, "pred_len":96, "d_model":128, "d_ff":256, "top_k":5}' --num-workers 4 --timeout 60000 --save-path "results_Exchange1"
#python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAutovalorAdapter" --model-hyper-params '{"batch_size":16, "num_epochs":2, "seq_len":96, "pred_len":96, "d_model":128, "d_ff":256, "top_k":5}' --num-workers 4 --timeout 60000 --save-path "results_Exchange1"
#python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAlphaAdapter" --model-hyper-params '{"batch_size":16, "num_epochs":2, "seq_len":96, "pred_len":96, "d_model":128, "d_ff":256, "top_k":5}' --num-workers 4 --timeout 60000 --save-path "results_Exchange1"

#echo "Fixed Forecast - Electricity - topk_2"

#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-name-list "Electricity.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":96,"pred_len":96,"d_model":128,"d_ff":256,"top_k":2}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNetAlphaAprendivel_Electricity_topk2"
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-name-list "Electricity.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetOriginalAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":96,"pred_len":96,"d_model":128,"d_ff":256,"top_k":2}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNetAlphaAprendivel_Electricity_topk2"
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-name-list "Electricity.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAutovalorAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":96,"pred_len":96,"d_model":128,"d_ff":256,"top_k":2}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNetAlphaAprendivel_Electricity_topk2"
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-name-list "Electricity.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAlphaAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":96,"pred_len":96,"d_model":128,"d_ff":256,"top_k":2}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNetAlphaAprendivel_Electricity_topk2"

#echo "Fixed Forecast - Electricity - topk_5"

#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-name-list "Electricity.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":96,"pred_len":96,"d_model":128,"d_ff":256,"top_k":5}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNetAlphaAprendivel_Electricity_topk5"
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-name-list "Electricity.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetOriginalAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":96,"pred_len":96,"d_model":128,"d_ff":256,"top_k":5}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNetAlphaAprendivel_Electricity_topk5"
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-name-list "Electricity.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAutovalorAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":96,"pred_len":96,"d_model":128,"d_ff":256,"top_k":5}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNetAlphaAprendivel_Electricity_topk5"
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-name-list "Electricity.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAlphaAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":96,"pred_len":96,"d_model":128,"d_ff":256,"top_k":5}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNetAlphaAprendivel_Electricity_topk5"

#echo "Fixed Forecast - Traffic - topk_2"

#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-name-list "Traffic.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":96,"pred_len":96,"d_model":128,"d_ff":256,"top_k":2}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNetAlphaAprendivel_Traffic_topk2"
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-name-list "Traffic.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetOriginalAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":96,"pred_len":96,"d_model":128,"d_ff":256,"top_k":2}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNetAlphaAprendivel_Traffic_topk2"
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-name-list "Traffic.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAutovalorAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":96,"pred_len":96,"d_model":128,"d_ff":256,"top_k":2}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNetAlphaAprendivel_Traffic_topk2"
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-name-list "Traffic.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAlphaAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":96,"pred_len":96,"d_model":128,"d_ff":256,"top_k":2}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNetAlphaAprendivel_Traffic_topk2"

#echo "Fixed Forecast - Traffic - topk_5"

#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-name-list "Traffic.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":96,"pred_len":96,"d_model":128,"d_ff":256,"top_k":5}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNetAlphaAprendivel_Traffic_topk5"
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-name-list "Traffic.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetOriginalAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":96,"pred_len":96,"d_model":128,"d_ff":256,"top_k":5}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNetAlphaAprendivel_Traffic_topk5"
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-name-list "Traffic.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAutovalorAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":96,"pred_len":96,"d_model":128,"d_ff":256,"top_k":5}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNetAlphaAprendivel_Traffic_topk5"
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-name-list "Traffic.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAlphaAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":96,"pred_len":96,"d_model":128,"d_ff":256,"top_k":5}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNetAlphaAprendivel_Traffic_topk5"


echo "Fixed Forecast - Weather - topk_2"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-name-list "Weather.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":96,"pred_len":96,"d_model":128,"d_ff":256,"top_k":2}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNetAlphaAprendivel_Weather_topk2"
python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-name-list "Weather.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetOriginalAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":96,"pred_len":96,"d_model":128,"d_ff":256,"top_k":2}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNetAlphaAprendivel_Weather_topk2"
python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-name-list "Weather.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAutovalorAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":96,"pred_len":96,"d_model":128,"d_ff":256,"top_k":2}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNetAlphaAprendivel_Weather_topk2"
python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-name-list "Weather.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAlphaAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":96,"pred_len":96,"d_model":128,"d_ff":256,"top_k":2}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNetAlphaAprendivel_Weather_topk2"

echo "Fixed Forecast - Weather - topk_5"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-name-list "Weather.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":96,"pred_len":96,"d_model":128,"d_ff":256,"top_k":5}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNetAlphaAprendivel_Weather_topk5"
python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-name-list "Weather.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetOriginalAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":96,"pred_len":96,"d_model":128,"d_ff":256,"top_k":5}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNetAlphaAprendivel_Weather_topk5"
python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-name-list "Weather.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAutovalorAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":96,"pred_len":96,"d_model":128,"d_ff":256,"top_k":5}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNetAlphaAprendivel_Weather_topk5"
python ./scripts/run_benchmark.py --config-path "fixed_forecast_config.json" --data-name-list "Weather.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAlphaAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":96,"pred_len":96,"d_model":128,"d_ff":256,"top_k":5}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNetAlphaAprendivel_Weather_topk5"



#python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAdapter" --model-hyper-params '{"batch_size":16, "num_epochs":2, "seq_len":96, "pred_len":336, "d_model":128, "d_ff":256, "top_k":2}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNet_modificado"
#python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --deterministic "efficient" --model-name "mytimesnet.MyTimesOriginalNetAdapter" --model-hyper-params '{"batch_size":16, "num_epochs":2, "seq_len":96, "pred_len":336, "d_model":128, "d_ff":256, "top_k":2}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNetOriginal"

#python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAdapter" --model-hyper-params '{"batch_size":16, "num_epochs":2, "seq_len":96, "pred_len":336, "d_model":128, "d_ff":256, "top_k":2}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNet_local"
#python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesOriginalNetAdapter" --model-hyper-params '{"batch_size":16, "num_epochs":2, "seq_len":96, "pred_len":336, "d_model":128, "d_ff":256, "top_k":2}' --num-workers 4 --timeout 60000 --save-path "results_MyTimesNet_local"


# echo "Iniciando MyTimesNet - FIXED FORECAST ILI"
# Modificado
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_weekly.json" --data-name-list "ILI.csv" --deterministic "efficient" --strategy-args '{"horizon":24}' --model-name "mytimesnet.MyTimesNetAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":104,"pred_len":24,"d_model":128,"d_ff":256,"top_k":2}' --num-workers 1 --timeout 60000 --save-path "results_MyTimesNet_fixed2"
# Original
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_weekly.json" --data-name-list "ILI.csv" --deterministic "efficient" --strategy-args '{"horizon":24}' --model-name "mytimesnet.MyTimesNetOriginalAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":104,"pred_len":24,"d_model":128,"d_ff":256,"top_k":2}' --num-workers 1 --timeout 60000 --save-path "results_MyTimesNet_fixed3"




echo "Finalizado."
