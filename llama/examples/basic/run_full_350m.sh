cd "$(dirname "$0")/../.."

torchrun --nproc-per-node 8 --master_port 14214 main.py --model_config configs/llama_350m.json --dataset_path preprocessed_data/allenai/c4_en_t5-base_512 --batch_size 72 --total_batch_size 1152 --lr 1e-3 --max_length 512 --num_training_steps 40000 --save_every 2000 --eval_every 1000 --keep_checkpoints 3 --num_workers 8 --save_dir checkpoints/llama_350m_full_512_batch1152_lr0.001_step40000 --autoresume True




