#!/bin/bash

cd "$(dirname "$0")/../.."

task="$1"
lr="$2"

for seed in 1 2 3 4 5; do
  output=glue_output/350m/full_full/"${task}_$lr"
  bash run_glue_full.sh --model_name_or_path checkpoints/llama_350m_full_512_batch1152_lr0.001_step40000/converted_mdoel --lr "$lr" --output_dir "$output" --task_name "$task" --seed "$seed"
  rm -rf "$output/checkpoint-*"
  mv "$output/metric_results.json" "$output/metric_results$seed.json"
done
