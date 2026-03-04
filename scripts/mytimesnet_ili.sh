#!/bin/bash
#SBATCH --job-name=MyTimesNet_ILI
#SBATCH --output=/sonic_home/igor.viveiros/src/TFB/logs/%x_%j.out
#SBATCH --error=/sonic_home/igor.viveiros/src/TFB/logs/%x_%j.err
#SBATCH --partition=gorgonas
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=24:00:00

echo "======================================="
echo "Job started on $(hostname)"
echo "Start time: $(date)"
echo "======================================="

cd /sonic_home/igor.viveiros/src/TFB || exit 1

# Opcional: controle de thread
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK


/sonic_home/igor.viveiros/py311_cluster/bin/python ./scripts/run_benchmark.py \
  --config-path "rolling_forecast_config.json" \
  --data-name-list "ILI.csv" \
  --strategy-args '{"horizon":24}' \
  --model-name "mytimesnet.MyTimesNetAdapter" \
  --model-hyper-params '{"batch_size":16,"num_epochs":1,"seq_len":32,"pred_len":24,"num_workers":4,"d_model":128,"d_ff":256,"top_k":2}'\
  --num-workers 1 \
  --timeout 60000 \
  --save-path "results_MyTimesNet_ILI"

echo "======================================="
echo "End time: $(date)"
echo "======================================="
