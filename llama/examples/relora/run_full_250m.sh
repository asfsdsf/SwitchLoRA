torchrun --nproc-per-node 8 torchrun_main.py --model_config configs/llama_250m.json --dataset_path preprocessed_data/allenai/c4_en_t5-base_512 --batch_size 72 --total_batch_size 1152 --lr  5e-4 --max_length 512 --save_every 1000 --eval_every 1000 ---keep_checkpoints 100 -num_training_steps 20000 --tags warm_start_250M --save_dir checkpoints/relora_250m_full


