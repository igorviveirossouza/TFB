#!/bin/bash
#SBATCH -p gorgonas
#SBATCH --cpus-per-task=8
#SBATCH --output=/sonic_home/igor.viveiros/logs/slurm-%j.out



# ==============================
# MyTimesNet - Execução no SPEED
# ==============================

# Ativar ambiente (ajuste se necessário)
source /sonic_home/igor.viveiros/py310/bin/activate

export MPLCONFIGDIR=/tmp/$USER-mpl

cd /sonic_home/igor.viveiros/src/TFB || exit 1

echo "Starting MY TIMES NET..."

#echo "Rolling Forecast - Exchange"

#python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAdapter" --model-hyper-params '{"batch_size":16, "num_epochs":2, "seq_len":96, "pred_len":96, "d_model":128, "d_ff":256, "top_k":5}' --num-workers 4 --timeout 60000 --save-path "results_Exchange1"
#python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetOriginalAdapter" --model-hyper-params '{"batch_size":16, "num_epochs":2, "seq_len":96, "pred_len":96, "d_model":128, "d_ff":256, "top_k":5}' --num-workers 4 --timeout 60000 --save-path "results_Exchange1"
#python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAutovalorAdapter" --model-hyper-params '{"batch_size":16, "num_epochs":2, "seq_len":96, "pred_len":96, "d_model":128, "d_ff":256, "top_k":5}' --num-workers 4 --timeout 60000 --save-path "results_Exchange1"
#python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Exchange.csv" --deterministic "efficient" --model-name "mytimesnet.MyTimesNetAlphaAdapter" --model-hyper-params '{"batch_size":16, "num_epochs":2, "seq_len":96, "pred_len":96, "d_model":128, "d_ff":256, "top_k":5}' --num-workers 4 --timeout 60000 --save-path "results_Exchange1"


# ------------------------------
# Adapters
# ------------------------------

adapters=(
"MyTimesNetAdapter"
#"MyTimesNetOriginalAdapter"
#"MyTimesNetAutovalorAdapter"
#"MyTimesNetAlphaAdapter"
)

# ------------------------------
# Datasets
# ------------------------------

datasets=(
"Electricity"
"Traffic"
)

# ------------------------------
# Rodar Electricity e Traffic
# ------------------------------

for dataset in "${datasets[@]}"; do
  for topk in 2 5; do
    echo "Fixed Forecast - ${dataset} - topk_${topk}"

    for adapter in "${adapters[@]}"; do

      python ./scripts/run_benchmark.py \
        --config-path "fixed_forecast_config_hourly.json" \
        --data-name-list "${dataset}.csv" \
        --deterministic "efficient" \
        --model-name "mytimesnet.${adapter}" \
        --model-hyper-params "{\"batch_size\":16,\"num_epochs\":2,\"seq_len\":96,\"pred_len\":96,\"d_model\":128,\"d_ff\":256,\"top_k\":${topk}}"\
        --gpus 1 \
        --num-workers 4 \
        --timeout 60000 \
        --save-path "results_${dataset}_topk${topk}"

    done
  done
done


# ------------------------------
# Weather (config diferente)
# ------------------------------

for topk in 2 5; do
  echo "Fixed Forecast - Weather - topk_${topk}"

  for adapter in "${adapters[@]}"; do

    python ./scripts/run_benchmark.py \
      --config-path "fixed_forecast_config_hourly.json" \
      --data-name-list "Weather.csv" \
      --deterministic "efficient" \
      --model-name "mytimesnet.${adapter}" \
      --model-hyper-params "{\"batch_size\":16,\"num_epochs\":2,\"seq_len\":96,\"pred_len\":96,\"d_model\":128,\"d_ff\":256,\"top_k\":${topk}}" \
      --gpus 1 \
      --num-workers 4 \
      --timeout 60000 \
      --save-path "results_Weather_topk${topk}"

  done
done


# echo "Iniciando MyTimesNet - FIXED FORECAST ILI"
# Modificado
# python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_weekly.json" --data-name-list "ILI.csv" --deterministic "efficient" --strategy-args '{"horizon":24}' --model-name "mytimesnet.MyTimesNetAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":104,"pred_len":24,"d_model":128,"d_ff":256,"top_k":2}' --num-workers 1 --timeout 60000 --save-path "results_MyTimesNet_fixed2"
# Original
#python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_weekly.json" --data-name-list "ILI.csv" --deterministic "efficient" --strategy-args '{"horizon":24}' --model-name "mytimesnet.MyTimesNetOriginalAdapter" --model-hyper-params '{"batch_size":16,"num_epochs":2,"seq_len":104,"pred_len":24,"d_model":128,"d_ff":256,"top_k":2}' --num-workers 1 --timeout 60000 --save-path "results_MyTimesNet_fixed3"


echo "Finalizado."


