cd "$(dirname "$0")/../.."

torchrun --master_port 14268 --nproc-per-node 8 main.py --model_config configs/llama_250m.json --dataset_path preprocessed_data/allenai/c4_en_t5-base_512 --batch_size 72 --total_batch_size 1152 --lr 0.01 --scheduler cosine_with_min_lr --max_length 512 --num_training_steps 20000 --save_every 2000 --eval_every 1000 --keep_checkpoints 3 --num_workers 8 --lora_rank 128 --lora_dropout 0.  --switch_lora --switch_lora_descent_rate 0.1 --zero_switch_step_state  --zero_switch_state ---continuous_switch -warmed_up_model ../relora/checkpoints/relora_250m_full_checkpoint/model_1000 --save_dir checkpoints/llama_250m_switchlora_512_batch1152_fromfullwarm1k --autoresume True

