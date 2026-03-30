#!/bin/bash
#SBATCH -p gorgonas
#SBATCH --cpus-per-task=32
#SBATCH --output=/sonic_home/igor.viveiros/logs/slurm-%j.out

source /sonic_home/igor.viveiros/py310/bin/activate
export MPLCONFIGDIR=/tmp/$USER-mpl

cd /sonic_home/igor.viveiros/src/TFB || exit 1

SALVAR_EM="TIOMS/b3_teste_1"


python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 24}' --model-name "TIOMS.BandWiseAdapter" --model-hyper-params '{"seq_len": 36, "pred_len": 24, "label_len": 18, "expert_hidden_dim": 128, "aggregator_type": "sum", "aggregator_hidden_dim": 64, "dropout": 0.1, "eps": 1e-5}' --deterministic "full" --gpus 1 --num-workers 4 --timeout 60000 --save-path "experimentos/$SALVAR_EM/h24" --save-true-pred True &

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 36}' --model-name "TIOMS.BandWiseAdapter" --model-hyper-params '{"seq_len": 36, "pred_len": 24, "label_len": 18, "expert_hidden_dim": 128, "aggregator_type": "sum", "aggregator_hidden_dim": 64, "dropout": 0.1, "eps": 1e-5}' --deterministic "full" --gpus 1 --num-workers 4 --timeout 60000 --save-path "experimentos/$SALVAR_EM/h36" --save-true-pred True &

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 48}' --model-name "TIOMS.BandWiseAdapter" --model-hyper-params '{"seq_len": 36, "pred_len": 24, "label_len": 18, "expert_hidden_dim": 128, "aggregator_type": "sum", "aggregator_hidden_dim": 64, "dropout": 0.1, "eps": 1e-5}' --deterministic "full" --gpus 1 --num-workers 4 --timeout 60000 --save-path "experimentos/$SALVAR_EM/h48" --save-true-pred True &

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "b3_daily_tfb.csv" --strategy-args '{"horizon": 60}' --model-name "TIOMS.BandWiseAdapter" --model-hyper-params '{"seq_len": 36, "pred_len": 24, "label_len": 18, "expert_hidden_dim": 128, "aggregator_type": "sum", "aggregator_hidden_dim": 64, "dropout": 0.1, "eps": 1e-5}' --deterministic "full" --gpus 1 --num-workers 4 --timeout 60000 --save-path "experimentos/$SALVAR_EM/h60" --save-true-pred True &

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