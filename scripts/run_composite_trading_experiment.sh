
  rm -rf "$result_dir"
  mkdir -p "$result_dir"

  "$PYTHON_BIN" ./scripts/run_benchmark_composite_trading_loss_v3.py \
    --config-path "$CONFIG_FILE" \
    --data-name-list "$data_file" \
    --strategy-args "{\"horizon\":${h},\"tv_ratio\":${TV_RATIO},\"train_ratio_in_tv\":${TRAIN_RATIO_IN_TV},\"stride\":${STRIDE},\"num_rollings\":${NUM_ROLLINGS},\"seed\":${SEED}}" \
    --model-name "$model_name" \
    --model-hyper-params "$model_hyper" \
    "${adapter_args[@]}" \
    --deterministic "$deterministic" \
    --gpus 0 \
    --num-workers 1 \
    --timeout 60000 \
    --save-path "$result_dir" \
    --save-true-pred True
}

decode_predictions() {
  local result_dir="$1" h="$2" decoded_dir="$3"
  rm -rf "$decoded_dir"
  mkdir -p "$decoded_dir"
  mapfile -t tars < <(find "$result_dir" -maxdepth 1 -type f -name '*.csv.tar.gz' | sort)
  (( ${#tars[@]} > 0 )) || { echo "ERRO: nenhum .csv.tar.gz em $result_dir" >&2; exit 4; }

  local copy_index=0 tarfile extracted_dir decoded_csv rows
  for tarfile in "${tars[@]}"; do
    "$PYTHON_BIN" ts_benchmark/utils/decode_prediction.py "$tarfile"
    extracted_dir="$(dirname "$tarfile")/$(basename "$tarfile" .tar.gz)_extracted"
    [[ -d "$extracted_dir" ]] || { echo "ERRO: pasta extraída não encontrada: $extracted_dir" >&2; exit 4; }
    while IFS= read -r decoded_csv; do
      rows=$(($(wc -l < "$decoded_csv") - 1))
      [[ "$rows" -eq "$h" ]] || continue
      cp "$decoded_csv" "${decoded_dir}/csv_sample_${copy_index}_inference_data.csv"
      copy_index=$((copy_index + 1))
    done < <(find "$extracted_dir" -type f -name 'inference_data.csv' | sort)
  done
  (( copy_index > 0 )) || { echo "ERRO: nenhuma previsão decodificada com h=$h." >&2; exit 4; }
}

convert_predictions() {
  local decoded_dir="$1" original_dataset="$2" h="$3" lb="$4" step_offset="$5"
  local dataset_label="$6" model_key="$7" k="$8" csv_dir="$9"

  local cmd=(
    "$PYTHON_BIN" scripts/convert_composite_trading_predictions.py
    --decoded-dir "$decoded_dir"
    --dataset "$original_dataset"
    --pred-len "$h"
    --lookback "$lb"
    --step-offset "$step_offset"
    --tv-ratio "$TV_RATIO"
    --parquet-root "$PARQUET_ROOT"
    --dataset-label "$dataset_label"
    --model "$model_key"
    --k "$k"
  )

  if [[ "$MANTER_CSV" == "TRUE" ]]; then
    cmd+=(--output-dir "$csv_dir" --keep-csv)
  fi