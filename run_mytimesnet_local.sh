#!/bin/bash

# ==============================
# MyTimesNet - Execução Local
# ==============================

# Ativar ambiente (ajuste se necessário)
source /home/igor/venvs/py310/bin/activate

# Ir para a pasta do projeto
cd ~/Documentos/DCC/Dissertação/src/TFB || exit 1

echo "Iniciando MyTimesNet..."

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
