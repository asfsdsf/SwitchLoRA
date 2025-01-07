"""
Examples run:
# run a switchlora training
python main.py --model_config configs/llama_35m.json --dataset_path preprocessed_data/allenai/c4_en_t5-base_128 --batch_size 24 --total_batch_size 1152 --lr 1e-3 --max_length 128 --num_training_steps 20000 --save_every 100 --eval_every 100 --keep_checkpoints 3 --num_workers 8 \
--switch_lora --switch_lora_descent_rate 0.1 --switch_lora_interval 40 --lora_rank 16
# run a full-rank training
python main.py --model_config configs/llama_35m.json --dataset_path preprocessed_data/allenai/c4_en_t5-base_128 --batch_size 24 --total_batch_size 1152 --lr 1e-3 --max_length 128 --num_training_steps 20000 --save_every 100 --eval_every 100 --keep_checkpoints 3 --num_workers 8
# run a small test for debugging
python main.py --model_config configs/llama_9m.json --dataset_path preprocessed_data/allenai/c4_en_t5-base_128 --batch_size 4 --total_batch_size 8 --lr 1e-3 --max_length 128 --num_training_steps 96 --save_every 8 --eval_every 8 --keep_checkpoints 3 --num_workers 8 \
--switch_lora --switch_lora_descent_rate 0.1 --switch_lora_interval 40 --lora_rank 16\
--autoresume true --save_dir checkpoints/small_switchlora_test
"""

import os
import sys
from torch.utils.data.dataloader import DataLoader
import yaml
import time
import json
import random
import argparse
from typing import Union

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    default_data_collator,
)

from tokenizers import Tokenizer

import datasets
import datasets.distributed

from tqdm import tqdm
from loguru import logger

from modeling_llama import LlamaForCausalLM

import training_utils
from training_utils import reduce_avg_values

from switchlora import switch_lora, lora_utils
from switchlora.optimizer import StepAdamW

transformers.logging.set_verbosity_error()


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    switch_lora.add_parse_switch_lora_args(parser)

    parser.add_argument("--training_config", type=str, default=None,
                        help="Alternative to providing the parameters. Overrides all parameters. Path to a yaml file with training run config")

    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--model_revision", type=str, default=None,
                        help="Tag name, branch name, or commit hash of the model from HuggingFace Hub. E.g., v2.0.1 or step1000")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Continue training, loading optimizer from the checkpoint. See also --autoresume to automatically set checkpoint resume dir.")
    parser.add_argument("--warmed_up_model", type=str, default=None,
                        help="Start with warmed-up model weights. Does not restore optimizer and scheduler.")
    parser.add_argument("--load_optimizer_state_on_resume", default=True, type=lambda x: x.lower() == "true",
                        help="Load optimizer state from the checkpoint when resuming training. "
                             "If False, optimizer state will be initialized from scratch. Setting it to False is useful for some very specific experiments.")

    parser.add_argument("--dataset_path", type=str, default=None, help="Path to a huggingface dataset directory")
    parser.add_argument("--from_json_data", action='store_true',
                        help="Load json data instead of huggingface. Training data should be put as <json_data_path>/train/*.json.gz. Validation data should be put as <dataset_path>/validation/*.json.gz.")
    parser.add_argument("--padding_data", action='store_true', help="Data is preprocessed by padding method.")
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)

    parser.add_argument("--optimizer", default="Adam",
                        help="Could be adam (for AdamW) or adam_zero for ZeroRedundancyOptimizer(AdamW)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["constant", "linear", "cosine", "cosine_with_min_lr"])
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps for scheduler.")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--eval_every", type=int, default=1_000)

    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Subdirectory under ./checkpoints to save checkpoints and tensorboard logs. When --autoresume is true, checkpoints in this directory will be resumed automatically.")
    parser.add_argument("--keep_checkpoints", type=int, default=None,
                        help="Number of checkpoints to keep. By default, keep all checkpoints.")
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--profile", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--plot_rank", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--autoresume", default=False, type=lambda x: x.lower() == "true",
                        help="Automatically resume training from the last checkpoint in the save_dir. ")

    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(args)

    args = check_args(args)

    return args


def check_args(args):
    if args.training_config is not None:
        logger.info(
            f"Yaml config provided for the run. The file {args.training_config} is used to provide all the parameters.")
        if len(sys.argv) > 3:
            logger.error(f"argv length is {len(sys.argv)}")
            raise RuntimeError(
                "You provided both a yaml config and command line arguments. "
                "Please use only one of the two options."
            )
        with open(args.training_config) as f:
            training_config = yaml.safe_load(f)
        for k, v in training_config.items():
            if k == "lr": v = float(v)
            setattr(args, k, v)

    if args.batch_size is None:
        raise ValueError("batch_size must be specified")

    if args.switch_lora:
        args.use_lora = True

    if args.total_batch_size is None:
        args.gradient_accumulation = args.gradient_accumulation or 1
        args.total_batch_size = args.batch_size * args.gradient_accumulation

    assert args.total_batch_size % args.batch_size == 0, "total_batch_size must be divisible by batch_size"

    if args.dtype in ["fp16", "float16"]:
        raise NotImplementedError("fp16 is not supported")

    if args.dataset_path is None:
        raise ValueError("dataset_path must be specified")

    if args.model_config is None:
        raise ValueError("model_config must be specified")

    return args


def single_gpu_env():
    if not "LOCAL_RANK" in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

    if not "MASTER_PORT" in os.environ:
        os.environ["MASTER_PORT"] = "15647"


def maybe_make_profiler(args):
    if not args.profile: return None
    global_rank = dist.get_rank()
    profiler_logging_dir = os.path.join(f"profiler_logs/{args.run_name}")
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_logging_dir, worker_name=f"rank{global_rank}"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )
    print(f"Rank {global_rank} profiling results will be saved to {profiler_logging_dir}")
    prof.start()
    return prof


def main(args):
    # --- seed ----------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    logger.info("Script finished successfully")

    # --- multi gpu env -------------------------------
    single_gpu_env()
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"
    if global_rank != 0: logger.remove()

    # --- batch size ----------------------------------
    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # --- automatically resume config -----------------
    # Obtain automatically resume config.
    # resume will be done later, after the model and optimizer are initialized.
    if args.save_dir is not None and os.path.exists(args.save_dir):
        if not args.autoresume:
            raise ValueError(f"Save directory {args.save_dir} already exists and --autoresume is off. Interrupting...")

        _old_train_config = os.path.join(args.save_dir, "training_config.yaml")
        if os.path.exists(_old_train_config):
            with open(os.path.join(args.save_dir, "training_config.yaml")) as f:
                old_args = yaml.safe_load(f)
            if old_args != vars(args):
                logger.warning(f"Arguments have changed since the last run.")
                logger.warning(f"Training config will be overwritten with new args")

                for k, v in vars(args).items():
                    if old_args.get(k) != v:
                        logger.warning(f"{k:30} {old_args.get(k)} -> {v}")
        else:
            logger.warning(f"Training config not found in the existing save directory {args.save_dir}.")

        training_state, resume_from = training_utils.get_last_training_state(args.save_dir)

        if args.resume_from is None:
            args.resume_from = resume_from

        logger.info(f"Resuming training from {resume_from}")

    dist.barrier()  # guarantees none of the workers will read save_dir above here before it's created by rank 0

    # --- set checkpoint dir --------------------------
    args.run_name = os.path.basename(args.model_config)
    args.run_name = os.path.splitext(args.run_name)[0]
    args.run_name = args.run_name + "_" + str(args.max_length)
    if args.switch_lora:
        switch_lora.set_hyper_args(args)
        args.run_name += f"_switchlora"
    elif args.use_lora:
        switch_lora.set_hyper_args(args)
        args.run_name += f"_lora"
    else:  # full-rank training
        args.run_name += f"_full"
    if global_rank == 0:
        if args.save_dir is None:
            args.save_dir = f"checkpoints/{args.run_name}"

        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "training_config.yaml"), "w") as f:
            yaml.dump(vars(args), f)

    dist.barrier()  # guarantees that save_dir exists and wand initialized on rank 0

    if args.save_dir is None:
        args.save_dir = f"checkpoints/{args.run_name}"

    if global_rank == 0:
        logger.add(os.path.join(args.save_dir, "output.log"))

    # --- Finish args config --------------------------
    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    # --- load dataset --------------------------------
    logger.info("Loading Huggingface dataset from directory")
    if args.from_json_data:
        def get_files(file_path):
            import glob
            files_path = glob.glob(file_path)
            files_path.sort()
            return files_path

        train_path = get_files(os.path.join(args.dataset_path, "train/*.json.gz"))
        eval_path = get_files(os.path.join(args.dataset_path, "validation/*.json.gz"))
        dataset_dict = datasets.load_dataset("json", data_files={
            'train': train_path, 'validation': eval_path}, streaming=True)
    else:
        dataset_dict = datasets.load_from_disk(args.dataset_path)
        logger.info(f"Applying set_format")
        dataset_dict.set_format(type='torch', columns=["input_ids"])

    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["validation"]

    if args.seed != 0:
        # TODO: check whether this condition is needed
        # this weird condition is due to backward compatibility
        train_dataset = train_dataset.shuffle(seed=args.seed)
        eval_dataset = eval_dataset.shuffle(seed=args.seed)

    # --- verify dataset ------------------------------
    logger.info("Checking datasets size")
    minimum_n_tokens = args.total_batch_size * args.num_training_steps
    if not args.from_json_data:
        dataset_n_tokens = len(train_dataset) * args.max_length
        if dataset_n_tokens < minimum_n_tokens:
            raise ValueError(f"Dataset only has {dataset_n_tokens} tokens, but we need at least {minimum_n_tokens}")

    logger.info("Loading dataset preprocessing args to check on seq_length")
    with open(os.path.join(args.dataset_path, "args.json")) as f:
        dataset_preprocessing_args = json.load(f)
    assert dataset_preprocessing_args["sequence_length"] == args.max_length
    logger.info("All good! Loading tokenizer now")

    # --- load tokenizer ------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        dataset_preprocessing_args["tokenizer"],
        model_max_length=args.max_length,
        cache_dir="tokenizer" + dataset_preprocessing_args["tokenizer"]
    )
    logger.info("Tokenizer loaded")

    # --- load model ----------------------------------
    if args.model_config is not None:
        model_config = AutoConfig.from_pretrained(args.model_config)
        t_vocab_size = tokenizer.get_vocab_size() if isinstance(tokenizer, Tokenizer) else tokenizer.vocab_size

        if model_config.vocab_size != t_vocab_size:
            logger.warning(
                f"Model config vocab size ({model_config.vocab_size}) does not match tokenizer vocab size ({t_vocab_size})")
            if model_config.vocab_size == 32000 and t_vocab_size == 32100:
                logger.warning("You are most likely reusing old checkpoints. This is alright, but not recommended.")
            else:
                raise ValueError(
                    f"Model config vocab size ({model_config.vocab_size}) does not match tokenizer vocab size ({t_vocab_size})")

        if not isinstance(model_config, LlamaConfig):
            raise NotImplementedError(f"Unknown model config type {type(model_config)}, only LLaMA is supported")

        logger.info("Using local version of LLaMA")
        model = LlamaForCausalLM(model_config)
    else:
        raise ValueError("Model config must be provided")

    # --- step values ---------------------------------
    global_step = 0
    update_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    params_before = sum(p.numel() for p in model.parameters())

    # -- load from full pretraining -------------------
    if args.warmed_up_model is not None:
        logger.info("*" * 40)
        logger.info(f"Loading a warmed-up model from {args.warmed_up_model}")
        checkpoint_path = os.path.join(args.warmed_up_model, "pytorch_model.bin")  # !! won't work with sharded models
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        logger.info(f"Model successfully loaded (strict=True policy)")

    # --- wrap with lora ------------------------------
    if args.use_lora:
        logger.info(f"Wrapping model with LoRA")
        model = switch_lora.SwitchLoRAModel(
            model,
            to_lora_layer_name=["attn", "attention", "mlp"],
            r=args.lora_rank,
            lora_alpha=1.,
            lora_dropout=args.lora_dropout,
            quantize=args.quantize,
            use_double_quant=args.use_double_quant,
        )

    # --- resume checkpoints --------------------------
    if args.resume_from:
        logger.info(f"Loading model from {args.resume_from}")
        checkpoint_path = os.path.join(args.resume_from, "pytorch_model.bin")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)

        logger.info(f"Model successfully loaded (strict=True policy)")

        logger.info(f"Loading training state like global_step, update_step, and tokens_seen from {args.resume_from}")
        with open(os.path.join(args.resume_from, "training_state.json")) as f:
            _old_state = json.load(f)
        global_step = _old_state["global_step"]
        update_step = _old_state["update_step"]
        tokens_seen = _old_state["tokens_seen"]
        tokens_seen_before = _old_state["tokens_seen_before"]
        logger.info(f"global_step       : {global_step}")
        logger.info(f"update_step       : {update_step}")
        logger.info(f"tokens_seen       : {tokens_seen}")
        logger.info(f"tokens_seen_before: {tokens_seen_before}")
        logger.info(f"Will train for {args.num_training_steps - update_step} update steps")

    # --- print params and trainable params -----------
    params_after = sum(p.numel() for p in model.parameters())
    params_candidates = sum(p[1].numel() for p in model.named_parameters() if
                            p[0].endswith("candidates_A") or p[0].endswith("candidates_B"))
    added_floats = params_after - params_before
    logger.info(f"Total params  before LoRA: {params_before / 1_000_000:.2f}M")
    logger.info(f"Total params  after  LoRA (Including candidates parameters): {params_after / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    logger.info(f"Candidates params: {params_candidates / 1_000_000:.2f}M")
    logger.info(f"In total, added {added_floats / 1_000_000:.2f}M parameters to the model")

    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")

    # --- fixed precision -----------------------------
    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    n_total_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    p_trainable_params = n_trainable_params / n_total_params

    # --- Distributed wrapping ------------------------
    logger.info("Wrapping model with DDP")
    model: Union[switch_lora.SwitchLoRAModel, LlamaForCausalLM] = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )
    switch_lora.init(model)

    lora_A_params, lora_B_params, other_params, trainable_params = lora_utils.obtain_lora_parameters(model)

    if args.use_lora and len(lora_A_params) + len(lora_B_params) == 0:
        raise ValueError("No LoRA parameters found")

    # --- set run_config ------------------------------
    run_config = dict(vars(args))
    run_config.update({
        "tokenizer": dataset_preprocessing_args["tokenizer"],
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "trainable_params_M": n_trainable_params / 1_000_000,
        "equivalent_params_M": params_before / 1_000_000,
        "percent_trainable_params": p_trainable_params,
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
        "dataset_preprocessing_args": dataset_preprocessing_args,
    })

    # --- optimizer -----------------------------------
    optimizer_kwargs = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "betas": (args.adam_beta1, args.adam_beta2),
    }
    if args.optimizer.lower() == "adam":
        params = lora_utils.obtain_params_mark_lora(model, args.lr)
        if args.switch_lora and not (
                args.remain_switch_step_state and args.remain_switch_state and args.zero_all_state):
            logger.info("Using StepAdam optimizer")
            optimizer = StepAdamW(params, **optimizer_kwargs)
        else:
            logger.info("Using Adam optimizer")
            optimizer = torch.optim.AdamW(params, **optimizer_kwargs)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    # --- scheduler -----------------------------------
    if args.scheduler == "constant":
        scheduler = transformers.get_constant_schedule(optimizer)
    elif args.scheduler == "linear":
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, args.warmup_steps, args.num_training_steps)
    elif args.scheduler == "cosine":
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.num_training_steps)
    elif args.scheduler == "cosine_with_min_lr":
        # like transformers.get_cosine_schedule_with_warmup() but keep ratio grater than 0.1
        scheduler = training_utils.get_cosine_schedule_with_warmup_min_lr(optimizer, args.warmup_steps,
                                                                          args.num_training_steps)
    else:
        raise ValueError(f"Scheduler {args.scheduler} not supported")

    # --- resume optimizer & scheduler ----------------
    if args.resume_from:
        if args.load_optimizer_state_on_resume:
            _optimizer_dir = args.resume_from
            optimizer_checkpoint = torch.load(os.path.join(_optimizer_dir, "optimizer.pt"), map_location="cpu")
            optimizer.load_state_dict(optimizer_checkpoint["optimizer"])
            scheduler.load_state_dict(optimizer_checkpoint["scheduler"])
            update_step = optimizer_checkpoint["update_step"]
            global_step = optimizer_checkpoint["global_step"]
            logger.info(f"Optimizer restored from {_optimizer_dir}")

            # Resume all random generator states
            torch.set_rng_state(optimizer_checkpoint['torch_rng_state'])
            torch.cuda.set_rng_state(optimizer_checkpoint['cuda_rng_state'])
            np.random.set_state(optimizer_checkpoint['numpy_rng_state'])

        # check that batch_size did not change or dataloader rewinding won't work
        _training_config_path = os.path.join(args.resume_from, "training_config.yaml")
        if os.path.exists(_training_config_path):
            with open(_training_config_path) as f:
                _old_training_config = yaml.safe_load(f)
            if args.batch_size != _old_training_config["batch_size"]:
                raise RuntimeError("Cannot resume from a checkpoint with a different batch size.")

    # --- dataloader ----------------------------------
    # Huggingface dataset to dataloader
    if not args.from_json_data:
        logger.info(f"Full training set size: {len(train_dataset)}")
    logger.info(repr(train_dataset))

    train_dataset = datasets.distributed.split_dataset_by_node(train_dataset, rank=global_rank, world_size=world_size)
    eval_dataset = datasets.distributed.split_dataset_by_node(eval_dataset, rank=global_rank, world_size=world_size)

    _skip_batches = update_step * args.gradient_accumulation
    train_loader = training_utils.SkipDataLoader(
        train_dataset,
        skip_batches=_skip_batches,
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        num_workers=args.num_workers,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        num_workers=args.num_workers,
    )

    test_loader = None

    # -------------------------------------------------

    update_time = time.time()
    loss_info = torch.tensor([0.0, 0.0, 0.0], device=device)  # loss, n_batches, n_NaNs
    loss_record_list = []
    updated_loss_record_list = []
    test_loss_list = []
    n_skipped_batches = 0
    if args.padding_data:
        pad_idx = tokenizer.pad_token_id
    else:
        pad_idx = None

    prof = maybe_make_profiler(args)

    # --- training loop -------------------------------
    logger.info(f"Starting training. update_step: {update_step}, global_step: {global_step}")
    if global_rank == 0:
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)

    for batch in train_loader:
        global_step += 1
        if global_step == 1: logger.info(f"Starting first step")
        if update_step >= args.num_training_steps:
            logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
            print(f"Rank {global_rank} stopping training.")
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        if args.padding_data:
            labels = batch["input_ids"].clone()
            labels[labels == pad_idx] = -100
        else:
            labels = batch["input_ids"]

        tokens_seen += batch["input_ids"].numel() * world_size
        loss = model(**batch, labels=labels).loss

        loss_info[0] += loss.detach()
        loss_info[1] += 1
        loss_info[2] += torch.isnan(loss).float()

        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()

        loss_record_list.append(loss.item())

        if global_step % args.gradient_accumulation != 0:
            continue

        # --- print delta norm ----------------------------
        if global_step > args.gradient_accumulation and (update_step + 1) % args.save_every == 0 and global_rank == 0:
        # if (update_step + 1) % args.save_every == 1 and global_rank == 0:
            if args.use_lora:
                current_model_directory = f"{args.save_dir}/model_{update_step + 1}"
                os.makedirs(current_model_directory, exist_ok=True)
                delta_norm = lora_utils.cal_delta_norm(model, use_lora=args.use_lora, print_abs=True)
                loss_file_path = os.path.join(current_model_directory, "delta_norm.json")
                with open(loss_file_path, "w") as f:
                    json.dump(delta_norm, f, indent=4)

        # -------------------------------------------------

        updated_loss_record_list.append(sum(loss_record_list) / len(loss_record_list))
        loss_record_list = []

        # The below code is only executed during the update step
        if global_rank == 0: pbar.update(1)

        if args.clip_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.clip_grad_norm, error_if_nonfinite=True)
            # if global_rank == 0:
            #     if grad_norm >= args.clip_grad_norm:
            #         logger.info(f"Grad norm clipped: {grad_norm}")

        dist.all_reduce(loss_info, op=dist.ReduceOp.SUM)
        _loss = loss_info[0] / loss_info[1]

        if loss_info[2] == 0:  # no NaNs, update model
            # >>> switchlora switch matrix >>>>>>>>>>>>>>>>>>>>
            if args.switch_lora:
                switch_lora.switch_lora(
                    model,
                    optimizer,
                    update_step,
                    args.switch_lora_descent_rate * args.num_training_steps
                )
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            optimizer.step()
            scheduler.step()

        else:
            logger.error(f"Nan detected in loss_info, {_loss=}, skipping update")
            n_skipped_batches += 1

            if n_skipped_batches > 0.05 * args.num_training_steps:
                logger.error(f"More than 5% of batches skipped due to NaNs, stopping training.")
                break

        optimizer.zero_grad()
        update_step += 1
        update_time = time.time() - update_time

        loss_info = torch.zeros_like(loss_info)

        # --- evaluation ----------------------------------
        if update_step % args.eval_every == 0:  # TODO: delete 'and False'
            logger.info(f"Performing evaluation at step {update_step}")
            total_loss, evaluated_on_tokens = training_utils.evaluate_model(model, eval_loader, device, pad_idx)
            logger.info(f"Eval loss at step {update_step}: {total_loss}")
            test_loss_list.append((update_step, total_loss.item()))

        if global_step > args.gradient_accumulation and update_step % args.save_every == 0:
        # if update_step % args.save_every == 1:
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "update_time": update_time,
            }
            training_utils.save_model(
                model,
                dict(lora_utils.iter_lora_layers(model, with_name=True)) if args.use_lora else [],
                optimizer=optimizer,
                scheduler=scheduler,
                training_state_checkpoint=training_state_checkpoint,
                run_config=run_config,
                save_dir=current_model_directory,
            )
            if args.keep_checkpoints is not None and global_rank == 0:
                training_utils.delete_old_checkpoints(args.save_dir, keep=args.keep_checkpoints)

            # save loss
            loss_file_path = os.path.join(current_model_directory, "loss.json")
            update_loss = reduce_avg_values(updated_loss_record_list, dtype=torch.float32, device=loss.device)
            if global_rank == 0:
                with open(loss_file_path, "w") as f:
                    json.dump({
                        "update_step_begin": update_step - len(updated_loss_record_list),
                        "update_step_end": update_step,
                        "update_loss": update_loss,
                        "test_loss": test_loss_list,
                    }, f, indent=4)

                # save rank distribution
                if args.plot_rank:
                    if args.use_lora:
                        lora_utils.rank_dist(model.module.origin_model.model, args.use_lora,
                                             os.path.join(current_model_directory, "rank_dist.json"), True)
                    else:
                        lora_utils.rank_dist(model.module.model, args.use_lora,
                                             os.path.join(current_model_directory, "rank_dist.json"), True)

        update_time = time.time()
        if prof is not None: prof.step()
    else:  # for-else statement
        print(f"Warning: reached the end of the dataset. Training stopped, {global_rank=}, {update_step=}")
        logger.warning("Reached the end of the dataset. Training stopped")

    # --- end training loop ---------------------------
    if prof is not None: prof.stop()

    # --- finish training -----------------------------
    logger.info("Training finished")
    if global_rank == 0: pbar.close()

    current_model_directory = f"{args.save_dir}/model_{update_step}"

    if not os.path.exists(current_model_directory):
        logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
        }
        training_utils.save_model(
            model,
            dict(lora_utils.iter_lora_layers(model, with_name=True)) if args.use_lora else [],
            optimizer=optimizer,
            scheduler=scheduler,
            training_state_checkpoint=training_state_checkpoint,
            run_config=run_config,
            save_dir=current_model_directory,
        )

    # --- final evaluation ----------------------------
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer
    import gc;
    gc.collect()
    torch.cuda.empty_cache()

    total_loss, evaluated_on_tokens = training_utils.evaluate_model(
        model, eval_loader, device, pad_idx,
        target_eval_tokens=100_000_000,
    )

    if global_rank == 0:
        logger.info(f"Final eval loss: {total_loss}")

    # --- test dataset evaluation ---------------------
    if test_loader is not None:
        logger.info("Running test evaluation (full test set!)")
        total_loss, evaluated_on_tokens = training_utils.evaluate_model(
            model, test_loader, device, pad_idx,
            target_eval_tokens=-1,
        )

        if global_rank == 0:
            logger.info(f"Test loss: {total_loss}")

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    args = parse_args()
    main(args)
