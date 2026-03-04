#!/bin/bash

# ==============================
# MyTimesNet - Execução no SPEED
# ==============================

source /sonic_home/igor.viveiros/tfb_clean/bin/activate
export MPLCONFIGDIR=/tmp/$USER-mpl
cd /sonic_home/igor.viveiros/src/TFB || exit 1

echo "Starting MY TIMES NET..."


/sonic_home/igor.viveiros/py310/bin/python 
# Ativar ambiente (ajuste se necessário)
source /sonic_home/igor.viveiros/py310/bin/activate

cd  /sonic_home/igor.viveiros/src/TFB

python ./scripts/run_benchmark.py \
  --config-path "rolling_forecast_config.json" \
  --data-name-list "ETTh1.csv" \
  --deterministic "efficient"\
  --model-name "mytimesnet.MyTimesNetAdapter" \
  --model-hyper-params '{
      "batch_size":16,
      "num_epochs":2,
      "seq_len":96,
      "pred_len":336,
      "d_model":128,
      "d_ff":256,
      "top_k":2
  }' \
  --num-workers 4 \
  --timeout 60000 \
  --save-path "results_MyTimesNet_local"

echo "Finalizado."
