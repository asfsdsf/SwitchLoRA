#!/bin/bash

# Default values for parameters
param1="v0"
param2="v1"
task_name="stsb"
seed=1234
max_seq_length=512
num_train_epochs=30
batch_size=16

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --param1)
            param1="$2"
            shift 2
            ;;
        --param2)
            param2="$2"
            shift 2
            ;;
        --model_name_or_path)
            model_name_or_path="$2"
            shift 2
            ;;
        --task_name)
            task_name="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        --max_seq_length)
            max_seq_length="$2"
            shift 2
            ;;
        --num_train_epochs)
            num_train_epochs="$2"
            shift 2
            ;;
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
        --lr)
            lr="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# if output_dir is not set, set it to ./glue_output/$task_name
if [ -z "$output_dir" ]; then
    output_dir="./glue_output/$task_name"
fi

# if model_name_or_path is not set, exit
if [ -z "$model_name_or_path" ]; then
    echo "model_name_or_path is not set"
    exit 1
fi

# if lr is not set, exit
if [ -z "$lr" ]; then
    echo "lr is not set"
    exit 1
fi

# Print the parameter values
echo "param1: $param1"
echo "param2: $param2"
echo "model_name_or_path: $model_name_or_path"
echo "task_name: $task_name"
echo "seed: $seed"
echo "output_dir: $output_dir"
echo "max_seq_length: $max_seq_length"
echo "num_train_epochs: $num_train_epochs"
echo "batch_size: $batch_size"


python run_glue.py \
--model_name_or_path "$model_name_or_path" \
--task_name "$task_name" \
--use_llama True \
--do_train \
--do_eval \
--evaluation_strategy epoch \
--save_total_limit 2 \
--save_safetensors False \
--save_steps 100000 \
--overwrite_output_dir \
--fp16 \
--cache_dir ./glue \
--seed "$seed" \
--output_dir "$output_dir" \
--max_seq_length "$max_seq_length" \
--num_train_epochs "$num_train_epochs" \
--per_device_train_batch_size "$batch_size" \
--learning_rate "$lr" \
--scheduler linear
