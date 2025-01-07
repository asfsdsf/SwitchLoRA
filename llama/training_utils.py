import os
import json
import time
import math
from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
import numpy as np

import transformers

from loguru import logger
import torch.distributed as dist


def delete_old_checkpoints(save_dir, keep):
    if keep is None:
        return

    checkpoints = [d for d in os.listdir(save_dir) if d.startswith(f"model_")]
    if len(checkpoints) <= keep:
        return

    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))
    for checkpoint in checkpoints[:-keep]:
        checkpoint_path = os.path.join(save_dir, checkpoint)
        logger.info(f"Deleting checkpoint {checkpoint_path}")
        os.system(f"rm -rf {checkpoint_path}")


def save_model(model, lora_layers, optimizer, scheduler, training_state_checkpoint, run_config, save_dir):
    global_rank = dist.get_rank()
    _time = time.time()

    if global_rank == 0:
        update_step = training_state_checkpoint["update_step"]
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

        _model = model.module
        _model.save_pretrained(save_dir, safe_serialization=False, max_shard_size="20GB")

        for layer_name in lora_layers:
            lora_layers[layer_name] = lora_layers[layer_name].state_dict()

    dist.barrier()

    if global_rank == 0:
        optimizer_checkpoint = {
            "lora": lora_layers,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "update_step": update_step,
            "global_step": training_state_checkpoint["global_step"],
            "config": run_config,
            "dtype": run_config["dtype"],
            'torch_rng_state': torch.get_rng_state(),
            # If you're using CUDA:
            'cuda_rng_state': torch.cuda.get_rng_state(),
            # If you're using NumPy:
            'numpy_rng_state': np.random.get_state()
        }
        torch.save(optimizer_checkpoint, f"{save_dir}/optimizer.pt")

        with open(f"{save_dir}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

    logger.info(f"Saving took {time.time() - _time:.2f} seconds")
    dist.barrier()


@torch.no_grad()
def evaluate_model(model: nn.Module, eval_dataloader, device, pad_idx, target_eval_tokens=10_000_000):
    _time = time.time()
    was_training = model.training
    model.eval()

    ddp_loss_info = torch.zeros(3).to(device)  # [loss, n_batches, n_tokens]
    tokens_in_batch_info = torch.zeros(1).to(device)

    rank = dist.get_rank()
    for i, batch in enumerate(eval_dataloader):
        if i == 0:
            # this way of estiming the number of eval steps
            # is needed to avoid a deadlock when using FSDP
            batch["input_ids"]: torch.Tensor
            tokens_in_batch_info[0] += batch["input_ids"].numel()
            dist.all_reduce(tokens_in_batch_info, op=dist.ReduceOp.SUM)
            n_eval_iters = int(target_eval_tokens / tokens_in_batch_info[0])

        if target_eval_tokens != -1 and i > n_eval_iters: break

        batch = {k: v.to(device) for k, v in batch.items()}

        if pad_idx is not None:
            labels = batch["input_ids"].clone()
            labels[labels == pad_idx] = -100
        else:
            labels = batch["input_ids"]

        loss = model(**batch, labels=labels).loss
        if torch.isnan(ddp_loss_info[0]):
            print(f"Rank {dist.get_rank()} got nan loss. This is probably a bug.")

        tokens_in_batch = batch["input_ids"].numel()
        assert tokens_in_batch > 0, "Batch size is zero"
        ddp_loss_info[0] += loss.detach()
        ddp_loss_info[1] += 1
        ddp_loss_info[2] += tokens_in_batch

    # check if loss is nan
    if torch.isnan(ddp_loss_info[0]):
        raise RuntimeError(f"Rank {rank} got nan loss. This is probably a bug.")

    # Gather losses across all GPUs
    dist.all_reduce(ddp_loss_info, op=dist.ReduceOp.SUM)
    eval_loss = ddp_loss_info[0] / ddp_loss_info[1]
    evaluated_on_tokens = ddp_loss_info[2].item()
    logger.info(f"Evaluated on {evaluated_on_tokens} tokens, eval loss: {eval_loss:.4f}")

    logger.info(f"Evaluation took {time.time() - _time:.2f} seconds")

    if was_training: model.train()
    return eval_loss, evaluated_on_tokens


def get_last_training_state(save_dir):
    # list all directories in the save_dir
    # find the model with the highest number of iterations "{args.save_dir}/model_{update_step}"
    model_dirs = [d for d in os.listdir(save_dir) if d.startswith(f"model_")]
    if len(model_dirs) == 0:
        logger.warning(f"Save directory {save_dir} exists, but does not contain any models.")
        logger.warning("Starting training from scratch.")
        return None, None

    model_dirs = sorted(model_dirs, key=lambda x: int(x.split("_")[-1]))
    resume_from = os.path.join(save_dir, model_dirs[-1])

    logger.info(f"Restarting training from {resume_from}")
    with open(os.path.join(resume_from, "training_state.json")) as f:
        training_state = json.load(f)

    return training_state, resume_from


def reduce_avg_values(value_list, dtype, device):
    """
    Gather across all processes to obtain average values for single value of list values.
    Note that this is not in place replacement.
    @param value_list: list or single value
    @return: reduced value
    """
    value_tensors = torch.tensor(value_list, dtype=dtype, device=device)

    if len(value_tensors) == 0:
        value_tensors = [value_tensors]
        dist.all_reduce(value_tensors, op=dist.ReduceOp.AVG)
        value_list = value_tensors[0].item()
    else:
        dist.all_reduce(value_tensors, op=dist.ReduceOp.AVG)
        value_list = [v.item() for v in value_tensors]
    return value_list


class SkipDataLoader(DataLoader):
    """
    Subclass of a PyTorch `DataLoader` that will skip the first batches.

    Args:
        dataset (`torch.utils.data.dataset.Dataset`):
            The dataset to use to build this datalaoder.
        skip_batches (`int`, *optional*, defaults to 0):
            The number of batches to skip at the beginning.
        kwargs:
            All other keyword arguments to pass to the regular `DataLoader` initialization.
    """

    def __init__(self, dataset, skip_batches=0, **kwargs):
        super().__init__(dataset, **kwargs)
        self.skip_batches = skip_batches

    def __iter__(self):
        for index, batch in enumerate(super().__iter__()):
            if index >= self.skip_batches:
                yield batch


def _get_cosine_schedule_with_warmup_min_lr_lambda(
        current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_lr_ratio: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup_min_lr(
        optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1,
        min_lr_ratio: float = 0.1
):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_min_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_ratio=min_lr_ratio
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
