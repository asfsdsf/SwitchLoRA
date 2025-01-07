import os
import sys
import yaml
import time
import json
import random
import argparse
from typing import Union

import numpy as np

import torch
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


import datasets
import datasets.distributed

from loguru import logger

from modeling_llama import LlamaForCausalLM

import training_utils

from switchlora import switch_lora, lora_utils

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
    parser.add_argument("--load_optimizer_state_on_resume", default=True, type=lambda x: x.lower() == "true",
                        help="Load optimizer state from the checkpoint when resuming training. "
                             "If False, optimizer state will be initialized from scratch. Setting it to False is useful for some very specific experiments.")

    parser.add_argument("--dataset_path", type=str, default=None, help="Path to a huggingface dataset directory")
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)

    parser.add_argument("--optimizer", default="Adam",
                        help="Could be adam (for AdamW) or adam_zero for ZeroRedundancyOptimizer(AdamW)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["constant", "linear", "cosine"])
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

    parser.add_argument("--quantize", default=None, type=str, choices=[None, "4bit", "8bit"])
    parser.add_argument("--use_double_quant", default=True, type=lambda x: x.lower() == "true")

    parser.add_argument("--profile", default=False, type=lambda x: x.lower() == "true")
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



def merge_weights(model):
    def T(w, layer):
        return w.transpose(0, 1) if layer.fan_in_fan_out else w
    @torch.no_grad()
    def layer2linear(origin_layer, device):
        if isinstance(origin_layer, switch_lora.SwitchLoRAMergedLinear):
            weight_transposed = True
        else:
            weight_transposed = False

        in_features = origin_layer.in_features
        out_features = origin_layer.out_features
        if isinstance(origin_layer, switch_lora.SwitchLoraLinear):
            linear_layer = torch.nn.Linear(in_features, out_features, bias=origin_layer.bias is not None,
                                           device=device, dtype=origin_layer.weight.dtype)
        elif isinstance(origin_layer, switch_lora.SwitchLoRAMergedLinear):
            linear_layer = transformers.pytorch_utils.Conv1D(out_features, in_features)
                                           #                   bias=origin_layer.bias is not None, device=device, dtype=origin_layer.weight.dtype)
        else:
            raise NotImplementedError("Unknown LoRA adapted layer.")

        if origin_layer.bias is not None:
            linear_layer.bias.copy_(origin_layer.bias)
        linear_layer.weight.copy_(origin_layer.weight.transpose(0, 1).contiguous() if weight_transposed else origin_layer.weight.contiguous())
        linear_layer.weight.data += origin_layer.merge_AB() * origin_layer.scaling 

        return linear_layer


    replace_list = ["attn", "attention", "mlp"]
    lora_class_list = [switch_lora.SwitchLoraLayer]

    for name, layer in model.named_modules():
        if not any(isinstance(layer, lora_class) for lora_class in lora_class_list):
            continue
        if any(name_to_replace in name for name_to_replace in replace_list):
            parent, _, target_name, = lora_utils._get_submodules(model, name)
            linear_layer = layer2linear(layer, model.device)
            setattr(parent, target_name, linear_layer)

def save_inner_model(model, lora_layers, run_config, save_dir):
    global_rank = dist.get_rank()
    _time = time.time()

    if global_rank == 0:
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

        merge_weights(model)
        _model = model.module.origin_model
        _model.save_pretrained(save_dir, safe_serialization=False)

        for layer_name in lora_layers:
            lora_layers[layer_name] = lora_layers[layer_name].state_dict()

    dist.barrier()

    if global_rank == 0:
        lora_checkpoint = {
            "lora": lora_layers,
        }
        torch.save(lora_checkpoint, f"{save_dir}/optimizer.pt")
        run_config_checkpoint = {
            "config": run_config,
            "dtype": run_config["dtype"],
        }
        torch.save(run_config_checkpoint, f"{save_dir}/run_config.pt")

    logger.info(f"Saving took {time.time() - _time:.2f} seconds")
    dist.barrier()

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


    # --- load model ----------------------------------
    if args.model_config is not None:
        model_config = AutoConfig.from_pretrained(args.model_config)

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
    added_floats = params_after - params_before
    logger.info(f"Total params  before LoRA: {params_before / 1_000_000:.2f}M")
    logger.info(f"Total params  after  LoRA (Including candidates parameters): {params_after / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
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

    lora_A_params, lora_B_params, other_params, trainable_params = lora_utils.obtain_lora_parameters(model)

    if args.use_lora and len(lora_A_params) + len(lora_B_params) == 0:
        raise ValueError("No LoRA parameters found")

    # --- set run_config ------------------------------
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "trainable_params_M": n_trainable_params / 1_000_000,
        "equivalent_params_M": params_before / 1_000_000,
        "percent_trainable_params": p_trainable_params,
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })


    # --- finish training -----------------------------
    logger.info("Training finished")

    current_model_directory = f"{args.save_dir}/converted_model"

    if not os.path.exists(current_model_directory):
        logger.info(f"Saving model to {current_model_directory}")
        save_inner_model(
            model,
            dict(lora_utils.iter_lora_layers(model, with_name=True)) if args.use_lora else [],
            run_config=run_config,
            save_dir=current_model_directory,
        )

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    args = parse_args()
    main(args)
