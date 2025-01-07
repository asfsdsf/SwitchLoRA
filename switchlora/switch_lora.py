import loralib as _lora
import torch.nn.functional as _F
from . import lora_utils
import torch as _torch
import torch.nn as _nn
import math as _math
import numpy.random as _random
import os as _os
import json as _json
import gc as _gc
from typing import Mapping as _Mapping, Any as _Any, List as _List
from functools import partial as _partial
from torch.optim.lr_scheduler import MultiplicativeLR as _MultiplicativeLR
from torch.optim.lr_scheduler import ChainedScheduler as _ChainedScheduler
import numpy as _np
import bitsandbytes as bnb
import bitsandbytes.functional as bnbF
from .lora_utils import obtain_lora_Adam_lr, obtain_lora_parameters
from transformers.pytorch_utils import Conv1D as GptConv1D

# Min step adam optimizer needed to warm-up
_ADAM_WARM_STEP = 5
# Type to descend switch_lora interval
_SWITCH_DESCEND_TYPE = "exponential"
# _SWITCH_DESCEND_TYPE = "Z"
# whether to drop candidates
_FIX_SWITCH_LORA_INTERVAL = False
_ZERO_INIT_B = False
_CANDIDATES_DROP_RATE = 0.
_SWITCH_LORA_INTERVAL = 40
_ADJUST_LORA_SCHEDULE = False
_ZERO_SWITCH_STATE = True
_ZERO_SWITCH_STEP_STATE = True
_ZERO_ALL_STATE = False
_ADD_WEIGHTED_RANK = False
_FORCE_DISABLE_CANDIDATES = False
_CONTINUOUS_SWITCH = True
_OFFLOAD_CANDIDATES = True


def add_parse_switch_lora_args(parser):
    """
    Recommended arguments for switch_lora
    @param parser: parser = argparse.ArgumentParser()
    """
    parser.add_argument("--use_lora", action='store_true')
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=float, default=1.)

    parser.add_argument("--switch_lora_drop", type=float, default=0,
                        help="Rate of candidates to drop.")
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--switch_lora_interval", type=int, default=40)
    parser.add_argument("--adjust_lora_schedule", action='store_true')
    parser.add_argument("--remain_switch_state", action='store_true',
                        help="Do not zero out optimizer states when switching LoRA.")
    parser.add_argument("--remain_switch_step_state", action='store_true',
                        help="Do not zero out optimizer states 'step' when switching LoRA.")
    parser.add_argument("--zero_all_state", action='store_true')
    parser.add_argument("--add_weighted_rank", action='store_true')
    parser.add_argument("--fix_switch_lora_interval", action='store_true',
                        help="Whether to fix switch lora interval.")
    parser.add_argument("--switch_lora_descent_rate", type=float, default=0.1)
    parser.add_argument("--adam_warm_step", type=int, default=5,
                        help="Min step adam optimizer needed to warm-up. Switched LoRA will be fixed in this step. Set to -1 means no warm-up(Do not use 0)")
    parser.add_argument("--switch_descend_type", type=str, default="exponential",
                        help="Type of descend rate. Z or exponential.")
    parser.add_argument("--force_disable_candidates", action='store_true',
                        help="Force to disable init candidates. This option is for test purpose.")
    parser.add_argument("--zero_init_B", action='store_true')
    parser.add_argument("--init_lora_type", type=str, default=None,
                        help="Set to origin_lora to use origin LoRA initialization method.")
    parser.add_argument("--switch_lora", action='store_true',
                        help="Use switched LoRA which will overlap --lora option.")
    parser.add_argument("--cal_delta_norm", action='store_true')
    parser.add_argument("--lora_scheduler", action='store_true')
    parser.add_argument("--change_lora_lr", action='store_true')
    parser.add_argument("--no_lora_lr_ratio", type=float, default=1.,
                        help="Change the learning rate of none LoRA parameters. (Only used when use_lora is true)")
    parser.add_argument("--quantize", default=None, type=str, choices=[None, "4bit", "8bit"])
    parser.add_argument("--use_double_quant", action='store_true')
    parser.add_argument("--discontinuous_switch", action='store_true',
                        help="Randomly select candidates instead of continuous selection.")
    parser.add_argument("--no_offload_candidates", action='store_true')


def set_hyper_args(args):
    if args.switch_lora:
        args.use_lora = True
    global _ADAM_WARM_STEP
    _ADAM_WARM_STEP = args.adam_warm_step
    global _SWITCH_DESCEND_TYPE
    _SWITCH_DESCEND_TYPE = args.switch_descend_type
    global _FIX_SWITCH_LORA_INTERVAL
    _FIX_SWITCH_LORA_INTERVAL = args.fix_switch_lora_interval
    global _CANDIDATES_DROP_RATE
    _CANDIDATES_DROP_RATE = args.switch_lora_drop
    global _SWITCH_LORA_INTERVAL
    _SWITCH_LORA_INTERVAL = args.switch_lora_interval
    global _ADJUST_LORA_SCHEDULE
    _ADJUST_LORA_SCHEDULE = args.adjust_lora_schedule
    global _ZERO_SWITCH_STATE
    _ZERO_SWITCH_STATE = not args.remain_switch_state
    global _ZERO_SWITCH_STEP_STATE
    _ZERO_SWITCH_STEP_STATE = not args.remain_switch_step_state
    global _ZERO_ALL_STATE
    _ZERO_ALL_STATE = args.zero_all_state
    global _ADD_WEIGHTED_RANK
    _ADD_WEIGHTED_RANK = args.add_weighted_rank
    global _ZERO_INIT_B
    _ZERO_INIT_B = args.zero_init_B
    global _FORCE_DISABLE_CANDIDATES
    _FORCE_DISABLE_CANDIDATES = args.force_disable_candidates
    global _CONTINUOUS_SWITCH
    _CONTINUOUS_SWITCH = not args.discontinuous_switch
    global _OFFLOAD_CANDIDATES
    _OFFLOAD_CANDIDATES = not args.no_offload_candidates
    lora_utils.CAL_DELTA_NORM = args.cal_delta_norm
    lora_utils.set_init_lora_method(args.init_lora_type)


@_torch.no_grad()
def _correct_switched_lora(model):
    def get_all_continuous_indices_pieces(fixed_steps):
        index_begin = -1
        index_end = -1
        for i, step in enumerate(fixed_steps):
            if step > 0:
                if index_end != i:
                    index_begin = i
                index_end = i + 1
                fixed_steps[i] -= 1
            elif index_end == i:
                yield index_begin, index_end
            if step < 0:
                fixed_steps[i] = 0

        if index_end == len(fixed_steps):
            yield index_begin, index_end

    lora_layers = lora_utils.iter_lora_layers(model)
    if _CONTINUOUS_SWITCH:
        for layer in lora_layers:
            for index_begin, index_end in get_all_continuous_indices_pieces(layer.fixed_A_steps):
                layer.get_lora_A_grad()[index_begin: index_end, :] = 0
            for index_begin, index_end in get_all_continuous_indices_pieces(layer.fixed_B_steps):
                layer.get_lora_B_grad()[:, index_begin: index_end] = 0
    else:
        for layer in lora_layers:
            for i, step in enumerate(layer.fixed_A_steps):
                if step > 0:
                    # lora_A is split by line index
                    layer.get_lora_A_grad()[i, :] = 0
                    layer.fixed_A_steps[i] -= 1
                elif step < 0:
                    layer.fixed_A_steps[i] = 0

            for i, step in enumerate(layer.fixed_B_steps):
                if step > 0:
                    # lora_B is split by column index
                    layer.get_lora_B_grad()[:, i] = 0
                    layer.fixed_B_steps[i] -= 1
                elif step < 0:
                    layer.fixed_B_steps[i] = 0


def _init_orthogonal_lora(mat_size: int, init_size: float, rank: int, src_mat, candidate_num: int):
    """

    :param mat_size: max of matrix row and column size
    :param init_size: the average value of elements in the matrix
    :return: orthogonal vectors list whose size is mat_size and the length of vector is mat_size too
    """
    # obtain candidates list
    # matrix = _torch.randn(mat_size, mat_size, device=src_mat.device, dtype=src_mat.dtype)
    # q, r = _torch.linalg.qr(matrix)
    # q *= init_size

    # TODO: check orthogonal or random is better
    # candidates_list = list(_torch.chunk(q, mat_size, dim=0))
    # candidates_list = [candidates_list[i] for i in len(candidates_list) if i < candidate_num]
    candidates_list = [src_mat.new_zeros(mat_size) for _ in range(candidate_num)]
    for candidate in candidates_list:
        candidate.uniform_(-init_size, init_size)

    # init candidates list
    candidates_weight = [1. / rank for _ in range(candidate_num)]
    global _CONTINUOUS_SWITCH
    if _CONTINUOUS_SWITCH:
        selected_indices = list(range(candidate_num - rank, candidate_num))
    else:
        selected_indices = _random.choice(list(range(candidate_num)), size=rank, replace=False)
    return candidates_list, candidates_weight, selected_indices


def _get_lora_schedule(global_step,
                       base_interval,
                       expect_switch_descend_step):
    ratio = 1.
    if _FIX_SWITCH_LORA_INTERVAL:
        interval = base_interval
    else:
        interval = base_interval / _get_switch_rate(global_step, expect_switch_descend_step)
    # in interval steps, fixed steps is _ADAM_WARM_STEP.
    if _ADAM_WARM_STEP > 0 and _ADJUST_LORA_SCHEDULE:
        ratio = ratio * (interval / (interval - _ADAM_WARM_STEP))
    return ratio


def _get_other_schedule(global_step):
    return 1.


def obtain_lora_scheduler(
        optimizer,
        base_interval,
        expect_switch_descend_step,
        optim_beta,
        origin_scheduler,
        last_epoch=-1,
):
    lr_lambda = []
    for elem in optim_beta:
        if "is_lora_A" in elem or "is_lora_B" in elem:
            schedule = _partial(
                _get_lora_schedule,
                base_interval=base_interval,
                expect_switch_descend_step=expect_switch_descend_step
            )
        else:
            schedule = _partial(
                _get_other_schedule
            )
        lr_lambda.append(schedule)
    switch_lora_scheduler = _MultiplicativeLR(optimizer, lr_lambda, last_epoch)
    return _ChainedScheduler([origin_scheduler, switch_lora_scheduler])


def _get_switch_rate(global_step: int, expect_switch_descend_step):
    if _SWITCH_DESCEND_TYPE == "Z":
        # Slowly decrease when step is little.
        # Fast decrease when step is close to expect_switch_descend_step.
        k = 10. / expect_switch_descend_step
        value = 1 - 1 / (1 + _math.exp(-k * (global_step - expect_switch_descend_step)))
    elif _SWITCH_DESCEND_TYPE == "exponential":
        # decrease exponentially
        # value is 0.3 when reaching expect_switch_descend_step
        x = 0.3 ** (1 / expect_switch_descend_step)
        value = x ** (global_step + 1)
    else:
        raise ValueError("Unsupported descend type.")
    # return max(value, 0.0001)
    return max(value, 1e-8)


def _get_switch_replace_num(select_num, global_step, base_switch_interval, expect_switch_descend_step):
    if _FIX_SWITCH_LORA_INTERVAL:
        interval = base_switch_interval
    else:
        interval = base_switch_interval / _get_switch_rate(global_step, expect_switch_descend_step)
    if interval < _ADAM_WARM_STEP * 2:  # *2 since only one of lora_A and lora_B can be fixed
        raise RuntimeError("Switch interval can not be less than adam warm up steps.")
    replace_num = select_num / interval
    replace_num_decimal = replace_num - int(replace_num)
    replace_num = int(replace_num) + (1 if _random.random() < replace_num_decimal else 0)
    return replace_num


def init(model):
    lora_layers = lora_utils.iter_lora_layers(model)
    if _OFFLOAD_CANDIDATES:
        for layer in lora_layers:
            layer.offload_candidates()
    _gc.collect()
    _torch.cuda.empty_cache()


@_torch.no_grad()
def switch_lora(model, optimizer, global_step, expect_switch_descend_step):
    def T(w, layer):
        return w.transpose(0, 1) if layer.fan_in_fan_out else w

    gather_estimated_rank(model, global_step)
    lora_layers = lora_utils.iter_lora_layers(model)
    for layer in lora_layers:
        layer.switch(global_step, expect_switch_descend_step)
        layer.zero_lora_states(optimizer)
        layer.correct_switched_lora()


@_torch.no_grad()
def gather_estimated_rank(model, step):
    lora_layers = lora_utils.iter_lora_layers(model)
    for layer in lora_layers:
        if not hasattr(layer, "gathered_ranks"):
            continue
        if not hasattr(layer, "ranks"):
            layer.ranks = {}
        layer.ranks[step] = [0] * layer.r
        for i in range(layer.r):
            gathered_rank = layer.gathered_ranks[i]
            if len(gathered_rank) == 0:
                continue
            std = _np.std(gathered_rank)
            avg = _np.average(gathered_rank)
            layer.ranks[step][i] = avg
            gathered_rank.clear()

        n = layer.lora_A.shape[1]
        dest_rank = (2 / (n * _torch.pi)) ** 0.5

        avg_rank = _np.average(layer.ranks[step])
        for i in range(layer.r):
            rank = layer.ranks[step][i]
            layer.candidates_A_weight[layer.selected_A_indices[i]].fill_(1. / n / (dest_rank / rank) ** 3)


def _is_continuous(num_list):
    old_value = num_list[0] - 1
    for num in num_list:
        if num != old_value + 1:
            return False
        old_value = num
    return True


@_torch.no_grad()
def _switch_chosen_indices(layer, replace_map: dict[int, int], replace_type: str, switch_drop: float):
    def T(w, layer):
        return w.transpose(0, 1) if layer.fan_in_fan_out else w

    def to_candidate(mat, candidate):
        assert not _OFFLOAD_CANDIDATES, "Offload is not implemented in this case."
        # set new_mat as candidate
        # new_mat = (1-switch_drop)*candidate+(mat-candidate)
        new_mat = mat - switch_drop * candidate
        # normalize(new_mat) so that its norm is the same as mat
        new_mat *= _torch.norm(mat) / _torch.norm(new_mat)
        candidate.copy_(new_mat)
        return candidate

    def switch_continuous_rows(weight, mat1, candidates1, selected_indices1, mat2, fixed_steps2,
                               mat_index_begin, mat_index_end, dest_index_begin, dest_index_end):
        if _OFFLOAD_CANDIDATES:
            candidates_piece = candidates1[dest_index_begin: dest_index_end].to(mat1.device)
        else:
            candidates_piece = candidates1[dest_index_begin: dest_index_end]
        if layer.quantize is None:
            weight += (T(mat2[:, mat_index_begin: mat_index_end] @
                         (mat1[mat_index_begin: mat_index_end, :] - candidates_piece), layer)
                       * layer.scaling)
        else:
            assert False, "Not implemented."

        # save mat1 value to candidates
        origin_candidate_indices = [selected_indices1[mat_index] for mat_index in range(mat_index_begin, mat_index_end)]
        if _is_continuous(origin_candidate_indices):
            if _OFFLOAD_CANDIDATES:
                candidates1[selected_indices1[mat_index_begin]: selected_indices1[mat_index_end - 1] + 1, :] = mat1[
                                                                                                               mat_index_begin: mat_index_end,
                                                                                                               :].to(
                    'cpu')
            else:
                candidates1[selected_indices1[mat_index_begin]: selected_indices1[mat_index_end - 1] + 1, :] = mat1[
                                                                                                               mat_index_begin: mat_index_end,
                                                                                                               :]
        else:
            for mat_index, dest_index in replace_map.items():
                origin_candidate_index = selected_indices1[mat_index]
                if switch_drop != 0:
                    # lora_A is split by line index
                    to_candidate(mat1[mat_index, :], candidates1[origin_candidate_index])
                else:
                    if _OFFLOAD_CANDIDATES:
                        candidates1[origin_candidate_index].copy_(mat1[mat_index, :].to('cpu'))
                    else:
                        candidates1[origin_candidate_index].copy_(mat1[mat_index, :])

        # change lora_A value to candidates
        mat1[mat_index_begin: mat_index_end, :] = candidates_piece
        del candidates_piece
        for mat_index, dest_index in replace_map.items():
            # lora_A is split by line index
            selected_indices1[mat_index] = dest_index

            fixed_steps2[mat_index] = _ADAM_WARM_STEP

    def switch_rows(weight, mat1, candidates1, selected_indices1, mat2, fixed_steps2):
        # update weight
        for mat_index, dest_index in replace_map.items():
            if layer.quantize is None:
                weight += T(
                    mat2[:, None, mat_index] @ (mat1[mat_index, None, :] - candidates1[dest_index].to(mat1.device)),
                    layer) * layer.scaling
            else:
                odd_mat = layer.odd_mat
                bnbF.dequantize_4bit(weight.data, weight.quant_state, out=odd_mat)
                odd_mat += T(
                    mat2[:, None, mat_index] @ (mat1[mat_index, None, :] - candidates1[dest_index]),
                    layer) * layer.scaling
                weight.data, weight.quant_state = bnbF.quantize_4bit(
                    odd_mat,
                    quant_type=weight.quant_type,
                    compress_statistics=weight.compress_statistics,
                )

        # save mat1 value to candidates
        for mat_index, dest_index in replace_map.items():
            origin_candidate_index = selected_indices1[mat_index]
            if switch_drop != 0:
                # lora_A is split by line index
                to_candidate(mat1[mat_index, :], candidates1[origin_candidate_index])
            else:
                if _OFFLOAD_CANDIDATES:
                    candidates1[origin_candidate_index].copy_(mat1[mat_index, :].to('cpu'))
                else:
                    candidates1[origin_candidate_index].copy_(mat1[mat_index, :])

        # change lora_A value to candidates
        for mat_index, dest_index in replace_map.items():
            # lora_A is split by line index
            mat1[mat_index, :] = candidates1[dest_index].to(mat1.device)
            selected_indices1[mat_index] = dest_index

            fixed_steps2[mat_index] = _ADAM_WARM_STEP

    global _CONTINUOUS_SWITCH
    if len(replace_map) > 1 and _CONTINUOUS_SWITCH:
        mat_indices = list(replace_map.keys())
        dest_indices = list(replace_map.values())
        continuous = _is_continuous(mat_indices) and _is_continuous(dest_indices)
    else:
        continuous = False
    if continuous:
        mat_index_begin = mat_indices[0]
        mat_index_end = mat_indices[-1] + 1
        dest_index_begin = dest_indices[0]
        dest_index_end = dest_indices[-1] + 1
    if replace_type == "A":
        if continuous:
            switch_continuous_rows(layer.weight, layer.lora_A, layer.candidates_A, layer.selected_A_indices,
                                   layer.lora_B, layer.fixed_B_steps,
                                   mat_index_begin, mat_index_end, dest_index_begin, dest_index_end)
        else:
            switch_rows(layer.weight, layer.lora_A, layer.candidates_A, layer.selected_A_indices, layer.lora_B,
                        layer.fixed_B_steps)
    else:
        if continuous:
            switch_continuous_rows(layer.weight.transpose(0, 1), layer.lora_B.transpose(0, 1), layer.candidates_B,
                                   layer.selected_B_indices, layer.lora_A.transpose(0, 1), layer.fixed_A_steps,
                                   mat_index_begin, mat_index_end, dest_index_begin, dest_index_end)
        else:
            switch_rows(layer.weight.transpose(0, 1), layer.lora_B.transpose(0, 1), layer.candidates_B,
                        layer.selected_B_indices, layer.lora_A.transpose(0, 1), layer.fixed_A_steps)


def _select_indices(candidate_num: int,
                    indices_to_replace: list[int],
                    current_indices,
                    fixed_steps,
                    weights) -> dict[int, int]:
    """
    :param Total candidate_num: number of candidates (including unavailable ones)
    :param indices_to_replace: indices in lora_A/lora_B to be replaced.
    :param current_indices: current selected candidate indices. i.e. selected_A_indices or selected_B_indices
    :param fixed_steps: remained fixed steps for selected elements.
    :param weights: weights of possibility to select candidate elements
    :return:
    """
    global _CONTINUOUS_SWITCH
    if _CONTINUOUS_SWITCH:
        mat_index_begin = indices_to_replace[0]
        candidate_index_begin = current_indices[mat_index_begin - 1] + 1
        replace_map = {index: (candidate_index_begin + i) % candidate_num for i, index in enumerate(indices_to_replace)}
        return replace_map

    all_indices = set(range(candidate_num))
    current_indices_set = set(current_indices)
    candidates_indices_to_replace = set([current_indices[i] for i in indices_to_replace])
    fixed_indices = set([current_indices[i] for i in range(len(current_indices)) if fixed_steps[i] > 0])
    available_indices = all_indices - (current_indices_set - candidates_indices_to_replace) - fixed_indices

    if len(fixed_indices.intersection(candidates_indices_to_replace)) > 0:
        raise RuntimeError("Fixed indices are chosen to switch")

    available_indices = list(available_indices)
    p = [weights[i] for i in available_indices]
    sum_p = sum(p)
    p = [p_i / sum_p for p_i in p]
    selected_indices = _random.choice(available_indices, size=len(indices_to_replace),
                                      replace=False, p=p)
    replace_map = dict(zip(indices_to_replace, selected_indices))
    return replace_map


def _get_int_para(value):
    return _nn.Parameter(_torch.tensor(value, dtype=_torch.int32), requires_grad=False)


def _get_float_para(value):
    return _nn.Parameter(_torch.tensor(value, dtype=_torch.float32), requires_grad=False)


class SwitchLoRAModel(_torch.nn.Module):
    def __init__(
            self,
            origin_model,
            to_lora_layer_name,
            r: int = 128,
            lora_alpha: float = 1,
            lora_dropout: float = 0.1,
            quantize=None,
            use_double_quant: bool = False
    ):
        super().__init__()
        self.origin_model = origin_model
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.to_lora_layer_name = to_lora_layer_name
        self.quantize = quantize
        self.use_double_quant = use_double_quant

        lora_utils.set_use_lora(layer_replace_dict)
        lora_utils.replace_with_lora_auto(
            self.origin_model, to_lora_layer_name,
            lora_rank=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            quantize=quantize,
            merge_weights=quantize is None,
            bnb_4bit_use_double_quant=use_double_quant
        )
        self._copy_forward_signature()

    def _copy_forward_signature(self):
        from inspect import signature
        import types
        origin_signature = signature(self.origin_model.__class__.forward)

        def new_forward(self, *args, **kwargs):
            return self.origin_model.forward(*args, **kwargs)

        new_forward.__signature__ = origin_signature

        # Dynamically set the new forward method with the correct signature
        self.forward = types.MethodType(new_forward, self)
        self.__class__.forward = new_forward

    def _pre_save_candidates_list(self):
        def set_value(param, value):
            param.fill_(value)

        def set_list_value(param_list, values):
            for i, param in enumerate(param_list):
                param.fill_(values[i])

        lora_layers = lora_utils.iter_lora_layers(self)
        for layer in lora_layers:
            if not hasattr(layer, "candidates_A_len"):
                continue
            set_value(layer.candidates_A_len_param, layer.candidates_A_len)
            set_value(layer.candidates_B_len_param, layer.candidates_B_len)
            set_value(layer.candidate_A_index_param, layer.candidate_A_index)
            set_value(layer.candidate_B_index_param, layer.candidate_B_index)
            set_list_value(layer.candidates_A_weight_param, layer.candidates_A_weight)
            set_list_value(layer.candidates_B_weight_param, layer.candidates_B_weight)
            set_list_value(layer.selected_A_indices_param, layer.selected_A_indices)
            set_list_value(layer.selected_B_indices_param, layer.selected_B_indices)
            set_list_value(layer.fixed_A_steps_param, layer.fixed_A_steps)
            set_list_value(layer.fixed_B_steps_param, layer.fixed_B_steps)

    def save_pretrained(self, path, **kwargs):
        self._pre_save_candidates_list()
        self.origin_model.save_pretrained(path, **kwargs)
        with open(_os.path.join(path, "switch_lora_config.json"), "w") as f:
            _json.dump({
                "r": self.r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                "to_lora_layer_name": self.to_lora_layer_name,
                "quantize": self.quantize,
                "use_double_quant": self.use_double_quant,
                # "switch_lora_descent_rate": self.switch_lora_descent_rate,
                # "switch_lora_interval": self.switch_lora_interval,
                # "switch_lora_drop": self.switch_lora_drop,
            }, f, indent=4)

    @classmethod
    def from_pretrained(cls, path):
        from transformers import AutoModelForCausalLM, AutoConfig
        with open(_os.path.join(path, "switch_lora_config.json"), "r") as f:
            relora_config = _json.load(f)

        config = AutoConfig.from_pretrained(path)

        base_model = AutoModelForCausalLM.from_config(config)

        model = cls(base_model, **relora_config)

        with open(_os.path.join(path, "pytorch_model.bin"), "rb") as f:
            state_dict = _torch.load(f, map_location="cpu")

        model.origin_model.load_state_dict(state_dict, strict=True)
        return model

    def load_state_dict(self, state_dict: _Mapping[str, _Any], strict: bool = True, assign: bool = False):
        result = self.origin_model.load_state_dict(state_dict, strict, assign)
        return result


class SwitchLoraLayer():
    def __init__(
            self,
            in_features=None,
            out_features=None,
            r=None,
    ):
        if not hasattr(self, 'lora_A'):
            raise RuntimeError("init of SwitchLoraLayer should be used before loralib LoRALayer init.")
        in_features = self.lora_A.shape[1]
        out_features = self.lora_B.shape[0]
        r = self.lora_A.shape[0]

        if not _FORCE_DISABLE_CANDIDATES:
            self._init_candidates()

        # flag to judge whether to estimate at next forward propagation
        self.to_estimate_rank = False

        if lora_utils.CAL_DELTA_NORM:
            # For test
            # Used to calculate the norm of gradients
            self.Wx = _nn.Parameter(_torch.empty(50, out_features), requires_grad=False)  # 50 is from transformer
            self.Ax = _nn.Parameter(_torch.empty(50, r), requires_grad=False)  # 50 is from transformer
            self.BAx = _nn.Parameter(_torch.empty(50, out_features), requires_grad=False)  # 50 is from transformer
            self.BA = _nn.Parameter(_torch.empty(out_features, in_features), requires_grad=False)
            self.deltaA = _nn.Parameter(_torch.empty(r, in_features), requires_grad=False)
            self.deltaB = _nn.Parameter(_torch.empty(out_features, r), requires_grad=False)
            self.deltaW = _nn.Parameter(_torch.empty(out_features, in_features), requires_grad=False)

    def forward(self, x: _torch.Tensor):
        if self.training and self.to_estimate_rank:
            with _torch.no_grad():
                rank = lora_utils.estimate_rank(self.lora_A.transpose(0, 1), x)
                if not hasattr(self, "gathered_ranks"):
                    self.gathered_ranks = []
                for i, r in enumerate(rank):
                    if len(self.gathered_ranks) <= i:
                        self.gathered_ranks.append([])
                    self.gathered_ranks[i].append(r.item())

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            bound_A, bound_B = lora_utils.get_init_lora_bound(self.in_features, self.out_features, self.r)
            with _torch.no_grad():
                self.lora_A.uniform_(-bound_A, bound_A)
                self.lora_B.uniform_(-bound_B, bound_B)

    @_torch.no_grad()
    def _init_candidates(self):
        """
        Initialize the candidates for lora_A and lora_B.
        """
        if hasattr(self, "candidates_A"):
            return

        def get_parameter_list(list_of_parameters):
            shape = (len(list_of_parameters), len(list_of_parameters[0]))
            new_list = _nn.Parameter(_torch.empty(
                shape, dtype=list_of_parameters[0].dtype, device=list_of_parameters[0].device), requires_grad=False)
            for i in range(shape[0]):
                new_list[i] = list_of_parameters[i]
            return new_list

        # Follow Xavier and Kaiming initialization to keep std of output of lora_B*lora_A uniform.
        # But make A*deltaB and B*deltaA the same size
        bound_A, bound_B = lora_utils.get_init_lora_bound(self.in_features, self.out_features, self.r)
        if _ZERO_INIT_B:
            bound_B = 0

        candidate_num = min(self.in_features, self.out_features)

        self.candidates_A, self.candidates_A_weight, self.selected_A_indices \
            = _init_orthogonal_lora(self.in_features, bound_A, self.lora_A.shape[0], self.lora_A, candidate_num)
        self.candidates_B, self.candidates_B_weight, self.selected_B_indices \
            = _init_orthogonal_lora(self.out_features, bound_B, self.lora_B.shape[1], self.lora_B, candidate_num)
        self.fixed_A_steps = [0] * len(self.selected_A_indices)
        self.fixed_B_steps = [0] * len(self.selected_B_indices)

        # register candidates as model parameters
        # so that they can be saved and loaded when checkpoints are saved and loaded
        self.candidates_A = get_parameter_list(self.candidates_A)
        self.candidates_B = get_parameter_list(self.candidates_B)

        self.candidate_A_index_param = _get_int_para(0)
        # set to self.r / 2 to make sure selected A and selected B are different
        # since select strategy is incremental selection
        self.candidate_B_index_param = _get_int_para(self.r // 2)
        # shift so that continuous switch can go in correct order
        self.selected_B_indices = [self.selected_B_indices[i - self.r // 2] for i in range(self.r)]

        # set some variables for candidates as model parameters
        self.candidates_A_len_param = _get_int_para(len(self.candidates_A))
        self.candidates_B_len_param = _get_int_para(len(self.candidates_B))
        self.candidates_A_weight_param = _get_float_para(self.candidates_A_weight)
        self.candidates_B_weight_param = _get_float_para(self.candidates_B_weight)
        self.selected_A_indices_param = _get_int_para(self.selected_A_indices)
        self.selected_B_indices_param = _get_int_para(self.selected_B_indices)

        self.fixed_A_steps_param = _get_int_para([0] * len(self.selected_A_indices))
        self.fixed_B_steps_param = _get_int_para([0] * len(self.selected_B_indices))

        for i, index in enumerate(self.selected_A_indices):
            self.lora_A[i, :] = self.candidates_A[index]
        for i, index in enumerate(self.selected_B_indices):
            self.lora_B[:, i] = self.candidates_B[index]

    def register_control_param(self):
        def get_value(param):
            return param.item()

        def set_list_value(param_list, value_list):
            for i, param in enumerate(param_list):
                value_list[i] = param.item()

        if hasattr(self, "candidates_A_len"):
            return

        self.candidates_A_len = get_value(self.candidates_A_len_param)
        self.candidates_B_len = get_value(self.candidates_B_len_param)
        self.candidate_A_index = get_value(self.candidate_A_index_param)
        self.candidate_B_index = get_value(self.candidate_B_index_param)
        set_list_value(self.candidates_A_weight_param, self.candidates_A_weight)
        set_list_value(self.candidates_B_weight_param, self.candidates_B_weight)
        set_list_value(self.selected_A_indices_param, self.selected_A_indices)
        set_list_value(self.selected_B_indices_param, self.selected_B_indices)
        set_list_value(self.fixed_A_steps_param, self.fixed_A_steps)
        set_list_value(self.fixed_B_steps_param, self.fixed_B_steps)

    def offload_candidates(self):
        if _OFFLOAD_CANDIDATES:
            self.candidates_A.data = self.candidates_A.data.to('cpu')
            self.candidates_B.data = self.candidates_B.data.to('cpu')

    def switch(self, global_step, expect_switch_descend_step):
        self.register_control_param()
        origin_available_candidate_num = min(self.in_features, self.out_features) - self.r
        if origin_available_candidate_num <= 0:
            return

        replace_num = _get_switch_replace_num(self.r, global_step, _SWITCH_LORA_INTERVAL, expect_switch_descend_step)
        fixed_A_num = sum([1 for s in self.fixed_A_steps if s > 0])
        fixed_B_num = sum([1 for s in self.fixed_B_steps if s > 0])
        available_num = min((self.candidate_A_index - fixed_A_num - self.candidate_B_index) % self.r,
                            (self.candidate_B_index - fixed_B_num - self.candidate_A_index) % self.r)
        if replace_num > available_num:
            replace_num = available_num

        if replace_num == 0:
            return

        replace_num_A = replace_num * (self.lora_A.shape[0] // self.r)
        replace_num_B = replace_num * (self.lora_B.shape[1] // self.r)

        to_replace_A = [i % self.lora_A.shape[0] for i in
                        range(self.candidate_A_index, self.candidate_A_index + replace_num_A)]
        to_replace_B = [i % self.lora_B.shape[1] for i in
                        range(self.candidate_B_index, self.candidate_B_index + replace_num_B)]
        self.candidate_A_index = (self.candidate_A_index + replace_num_A) % self.lora_A.shape[0]
        self.candidate_B_index = (self.candidate_B_index + replace_num_B) % self.lora_B.shape[1]
        replace_A_map = _select_indices(self.candidates_A_len, to_replace_A, self.selected_A_indices,
                                        self.fixed_A_steps, self.candidates_A_weight)
        _switch_chosen_indices(self, replace_A_map, "A", _CANDIDATES_DROP_RATE)
        replace_B_map = _select_indices(self.candidates_B_len, to_replace_B, self.selected_B_indices,
                                        self.fixed_B_steps, self.candidates_B_weight)
        if _ZERO_INIT_B:
            replace_B_map.clear()
        _switch_chosen_indices(self, replace_B_map, "B", _CANDIDATES_DROP_RATE)

    def _get_all_continuous_indices_pieces(self, fixed_steps):
        index_begin = -1
        index_end = -1
        for i, step in enumerate(fixed_steps):
            if step > 0:
                if index_end != i:
                    index_begin = i
                index_end = i + 1
                fixed_steps[i] -= 1
            elif index_end == i:
                yield index_begin, index_end
            if step < 0:
                fixed_steps[i] = 0

        if index_end == len(fixed_steps):
            yield index_begin, index_end

    def correct_switched_lora(self):
        if _CONTINUOUS_SWITCH:
            for index_begin, index_end in self._get_all_continuous_indices_pieces(self.fixed_A_steps):
                self.get_lora_A_grad()[index_begin: index_end, :] = 0
            for index_begin, index_end in self._get_all_continuous_indices_pieces(self.fixed_B_steps):
                self.get_lora_B_grad()[:, index_begin: index_end] = 0
        else:
            for i, step in enumerate(self.fixed_A_steps):
                if step > 0:
                    # lora_A is split by line index
                    self.get_lora_A_grad()[i, :] = 0
                    self.fixed_A_steps[i] -= 1
                elif step < 0:
                    self.fixed_A_steps[i] = 0

            for i, step in enumerate(self.fixed_B_steps):
                if step > 0:
                    # lora_B is split by column index
                    self.get_lora_B_grad()[:, i] = 0
                    self.fixed_B_steps[i] -= 1
                elif step < 0:
                    self.fixed_B_steps[i] = 0

    def _continuous_zero(self, optimizer, index_begin, index_end, A_or_B):
        if self.get_optimizer_lora(optimizer, "exp_avg", A_or_B) is None:
            return
        if A_or_B == "A":
            self.get_optimizer_lora(optimizer, "exp_avg", A_or_B)[index_begin: index_end, :] = 0
            self.get_optimizer_lora(optimizer, "exp_avg_sq", A_or_B)[index_begin: index_end, :] = 0
            if _ZERO_SWITCH_STEP_STATE:
                self.get_optimizer_lora(optimizer, "step", A_or_B)[index_begin: index_end, :] = 0
        else:
            self.get_optimizer_lora(optimizer, "exp_avg", A_or_B)[:, index_begin: index_end] = 0
            self.get_optimizer_lora(optimizer, "exp_avg_sq", A_or_B)[:, index_begin: index_end] = 0
            if _ZERO_SWITCH_STEP_STATE:
                self.get_optimizer_lora(optimizer, "step", A_or_B)[:, index_begin: index_end] = 0

    def zero_lora_states(self, optimizer):
        if not _ZERO_SWITCH_STATE:
            return
        fixed_A_indices = [i for i, step in enumerate(self.fixed_A_steps) if step == _ADAM_WARM_STEP or step < 0]
        fixed_B_indices = [i for i, step in enumerate(self.fixed_B_steps) if step == _ADAM_WARM_STEP or step < 0]
        for A_or_B in ("A", "B"):
            fixed_indices = fixed_A_indices if A_or_B == "A" else fixed_B_indices
            if _CONTINUOUS_SWITCH and len(fixed_indices) > 1 and _is_continuous(fixed_indices):
                self._continuous_zero(optimizer, fixed_indices[0], fixed_indices[-1] + 1, A_or_B)
            else:
                fixed_steps = self.fixed_A_steps if A_or_B == "A" else self.fixed_B_steps
                for i, step in enumerate(fixed_steps):
                    if step == _ADAM_WARM_STEP or step < 0:
                        if self.get_optimizer_lora(optimizer, "exp_avg", A_or_B) is None:
                            break
                        if _ZERO_ALL_STATE:
                            self.get_optimizer_lora()
                            self.get_optimizer_lora(optimizer, "exp_avg", A_or_B).zero_()
                            self.get_optimizer_lora(optimizer, "exp_avg_sq", A_or_B).zero_()
                            if _ZERO_SWITCH_STEP_STATE:
                                self.get_optimizer_lora(optimizer, "step", A_or_B).zero_()
                            break
                        else:
                            if A_or_B == "A":
                                self.get_optimizer_lora(optimizer, "exp_avg", A_or_B)[i, :] = 0
                                self.get_optimizer_lora(optimizer, "exp_avg_sq", A_or_B)[i, :] = 0
                                if _ZERO_SWITCH_STEP_STATE:
                                    self.get_optimizer_lora(optimizer, "step", A_or_B)[i, :] = 0
                            else:
                                self.get_optimizer_lora(optimizer, "exp_avg", A_or_B)[:, i] = 0
                                self.get_optimizer_lora(optimizer, "exp_avg_sq", A_or_B)[:, i] = 0
                                if _ZERO_SWITCH_STEP_STATE:
                                    self.get_optimizer_lora(optimizer, "step", A_or_B)[:, i] = 0

    def get_lora_A_grad(self):
        return self.lora_A.grad

    def get_lora_B_grad(self):
        return self.lora_B.grad

    def get_optimizer_lora(self, optimizer, state_name, A_or_B):
        if A_or_B == "A":
            return self.get_optimizer_lora_A(optimizer, state_name)
        elif A_or_B == "B":
            return self.get_optimizer_lora_B(optimizer, state_name)
        else:
            raise Exception("The parameter should be string 'A' or 'B'.")

    def get_optimizer_lora_A(self, optimizer, state_name):
        state = optimizer.state[self.lora_A]
        if state_name not in state:
            return None
        return state[state_name]

    def get_optimizer_lora_B(self, optimizer, state_name):
        state = optimizer.state[self.lora_B]
        if state_name not in state:
            return None
        return state[state_name]


class SwitchLoraLinear(_lora.Linear, SwitchLoraLayer):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = True,
            quantize=None,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            **kwargs):
        _lora.Linear.__init__(self, in_features, out_features, r, lora_alpha, lora_dropout, fan_in_fan_out,
                              merge_weights, **kwargs)

        SwitchLoraLayer.__init__(self)
        self.weight.requires_grad = False

        if merge_weights and quantize is not None:
            raise NotImplementedError("Merging is not yet supported when quantization is enabled.")

        self.quantize = quantize
        if quantize is None:
            pass
        elif quantize == "4bit":
            self.weight = bnb.nn.Params4bit(
                self.weight.data,
                requires_grad=False,
                compress_statistics=bnb_4bit_use_double_quant,
                quant_type=bnb_4bit_quant_type,
            )
        elif quantize == "8bit":
            # logger.warning("Int8 currently does not support merge_and_reinit! It will fail")
            raise NotImplementedError(
                "merge_and_reinit_functional for quantized models is not implemented yet. Use non-functional implementation")
            self.weight = bnb.nn.Int8Params(
                self.weight.data,
                requires_grad=False,
            )
        else:
            raise ValueError(f"Unknown quantize type: {quantize}")
        if quantize is not None:
            # Used for convenience. Memory overhead here can be diminished.
            # Memory usage of this parameter is not included in our paper.
            self.odd_mat = _nn.Parameter(_torch.empty(
                self.weight.data.shape,
                dtype=self.lora_B.dtype,
                device=self.weight.data.device), requires_grad=False)

    def forward(self, x: _torch.Tensor):
        # Replace _lora.Linear for better efficiency
        # result = _lora.Linear.forward(self, x)
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            if self.quantize == "4bit":
                result = bnb.matmul_4bit(x, T(self.weight.t()), bias=self.bias, quant_state=self.weight.quant_state)
            elif self.quantize == "8bit":
                result = bnb.matmul(x, T(self.weight.t()), bias=self.bias, quant_state=self.weight.quant_state)
            else:
                result = _F.linear(x, T(self.weight), bias=self.bias)
            # result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            result += _F.linear(_F.linear(self.lora_dropout(x), self.lora_A), self.lora_B) * self.scaling
        else:
            if self.quantize:
                raise NotImplementedError("Merging is not yet supported when quantization is enabled.")
            result = _F.linear(x, T(self.weight), bias=self.bias)

        SwitchLoraLayer.forward(self, x)
        return result

    def reset_parameters(self):
        _lora.Linear.reset_parameters(self)
        SwitchLoraLayer.reset_parameters(self)

    def merge_AB(self):
        def T(w, layer):
            return w.transpose(0, 1) if layer.fan_in_fan_out else w

        return T(self.lora_B @ self.lora_A, self)


class SwitchLoraConv(_lora.ConvLoRA, SwitchLoraLayer):
    def __init__(self,
                 conv_module,
                 in_channels,
                 out_channels,
                 kernel_size,
                 r=0,
                 lora_alpha=1,
                 lora_dropout=0.,
                 merge_weights=True,
                 quantize=None,
                 bnb_4bit_use_double_quant=False,
                 bnb_4bit_quant_type="nf4",
                 **kwargs):
        _lora.ConvLoRA.__init__(self, conv_module, in_channels, out_channels, kernel_size, r=r,
                                lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights, **kwargs)

        SwitchLoraLayer.__init__(self)
        self.quantize = quantize

        if quantize is not None:
            raise NotImplementedError("Quantization is not yet supported for ConvLoRA")
        self.in_features = self.lora_A.shape[1]
        self.out_features = self.lora_B.shape[0]
        self.conv.weight.requires_grad = False

    def forward(self, x: _torch.Tensor):
        result = _lora.ConvLoRA.forward(self, x)
        SwitchLoraLayer.forward(self, x)
        return result

    def reset_parameters(self):
        _lora.ConvLoRA.reset_parameters(self)
        SwitchLoraLayer.reset_parameters(self)

    def merge_AB(self):
        return (self.lora_B @ self.lora_A).view(self.conv.weight.shape)


class SwitchLoRAMergedLinearInner(SwitchLoraLayer):
    def __init__(
            self,
            parent_layer,
            group_index,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = True,
            quantize=None,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            **kwargs):
        self.parent_layer = parent_layer
        self.group_index = group_index
        self.r = r
        self.lora_alpha = lora_alpha
        # self.weight = weight
        if r > 0:
            # self.lora_A = lora_A
            # self.lora_B = lora_B
            self.scaling = self.lora_alpha / self.r
        self.in_features = in_features
        self.out_features = out_features
        self.fan_in_fan_out = fan_in_fan_out
        self.merge_weights = merge_weights
        self.quantize = quantize
        SwitchLoraLayer.__init__(self)
        self.reset_parameters()

    def reset_parameters(self):
        SwitchLoraLayer.reset_parameters(self)

    def merge_AB(self):
        def T(w, layer):
            return w.transpose(0, 1) if layer.fan_in_fan_out else w

        return T(self.lora_B @ self.lora_A, self)

    def offload_candidates(self):
        pass

    @property
    def weight(self):
        g = self.group_index
        o = self.out_features
        return self.parent_layer.weight[g * o:(g + 1) * o]

    @property
    def lora_A(self):
        g = self.group_index
        r = self.r
        return self.parent_layer.lora_A[g * r:(g + 1) * r]

    @property
    def lora_B(self):
        g = self.group_index
        o = self.out_features
        return self.parent_layer.lora_B[g * o:(g + 1) * o]

    def get_lora_A_grad(self):
        g = self.group_index
        r = self.r
        return self.parent_layer.lora_A.grad[g * r:(g + 1) * r]

    def get_lora_B_grad(self):
        g = self.group_index
        o = self.out_features
        return self.parent_layer.lora_B.grad[g * o:(g + 1) * o]

    def get_optimizer_lora_A(self, optimizer, state_name):
        g = self.group_index
        r = self.r
        state = optimizer.state[self.parent_layer.lora_A]
        if state_name not in state:
            return None
        return state[state_name][g * r:(g + 1) * r]

    def get_optimizer_lora_B(self, optimizer, state_name):
        g = self.group_index
        o = self.out_features
        state = optimizer.state[self.parent_layer.lora_B]
        if state_name not in state:
            return None
        return state[state_name][g * o:(g + 1) * o]


class SwitchLoRAMergedLinear(_lora.MergedLinear, SwitchLoraLayer):
    # See also https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/examples/NLG/src/model.py
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            enable_lora: _List[bool] = [True],
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = False,
            quantize=None,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quajt_type="nf4",
            **kwargs):
        # Set r = 0 here to disable default lora adapter
        _lora.MergedLinear.__init__(self, in_features, out_features, r, lora_alpha, lora_dropout, enable_lora,
                                    fan_in_fan_out, merge_weights, **kwargs)

        SwitchLoraLayer.__init__(self, in_features, out_features, r)
        self.quantize = quantize
        if quantize is not None:
            raise NotImplementedError("Quantization is not yet supported.")
        self.groups = sum(enable_lora)
        if 'bias' in kwargs and kwargs['bias']:
            kwargs['bias'] = False
        out_features_split = out_features // self.groups
        self.lora_adapters = [SwitchLoRAMergedLinearInner(
            # self.weight[g * out_features_split: (g + 1) * out_features_split],
            # self.lora_A[g * in_features:(g + 1) * in_features],
            # self.lora_B[g * out_features_split:(g + 1) * out_features_split],
            self,
            g,
            in_features,
            out_features_split,
            r,
            lora_alpha,
            lora_dropout,
            fan_in_fan_out,
            merge_weights,
            quantize,
            bnb_4bit_use_double_quant,
            bnb_4bit_quajt_type,
            **kwargs
        ) for g in range(self.groups)]
        self.reset_parameters()
        self.register_candidates()
        if quantize is not None:
            raise NotImplementedError("Quantization is not yet supported for ConvLoRA")

    def forward(self, x: _torch.Tensor):
        result = _lora.MergedLinear.forward(self, x)
        SwitchLoraLayer.forward(self, x)
        return result

    def reset_parameters(self):
        if not hasattr(self, 'groups'):
            # Prevent parameters reset in lora.MergedLinear class
            return
        _lora.MergedLinear.reset_parameters(self)
        for g in range(self.groups):
            self.lora_adapters[g].reset_parameters()
        # SwitchLoraLayer.reset_parameters(self)

    def offload_candidates(self):
        super().offload_candidates()
        if (self.candidates_A.device != self.lora_adapters[0].candidates_A.device or
                self.candidate_A_index_param.device != self.candidate_A_index_param.device or
                self.candidates_A.dtype != self.lora_adapters[0].candidates_A.dtype or
                self.candidate_A_index_param.dtype != self.candidate_A_index_param.dtype
        ):
            self.register_candidates()

    def switch(self, global_step, expect_switch_descend_step):
        for g in range(self.groups):
            self.lora_adapters[g].switch(global_step, expect_switch_descend_step)

    def correct_switched_lora(self):
        for g in range(self.groups):
            self.lora_adapters[g].correct_switched_lora()

    def zero_lora_states(self, optimizer):
        for g in range(self.groups):
            self.lora_adapters[g].zero_lora_states(optimizer)

    def register_candidates(self):
        def stack_parameters(parameter_name):
            parameter_list = [getattr(self.lora_adapters[g], parameter_name) for g in range(self.groups)]
            shape = list(parameter_list[0].shape)
            origin_param = getattr(self, parameter_name)
            dtype = origin_param.dtype
            device = origin_param.device
            if shape == []:
                dim0 = 1
                shape = [self.groups]
            else:
                dim0 = shape[0]
                shape[0] = shape[0] * self.groups
            if list(origin_param.shape) == shape:
                parameters = origin_param
            else:
                parameters = _nn.Parameter(_torch.empty(
                    shape, dtype=dtype, device=device), requires_grad=False)
                setattr(self, parameter_name, parameters)
            for g in range(self.groups):
                parameters[g * dim0: (g + 1) * dim0] = parameter_list[g]
                setattr(self.lora_adapters[g], parameter_name, parameters[g * dim0: (g + 1) * dim0])
            return parameters

        stack_parameters("candidates_A")
        stack_parameters("candidates_B")
        stack_parameters("candidate_A_index_param")
        stack_parameters("candidate_B_index_param")

        stack_parameters("candidates_A_len_param")
        stack_parameters("candidates_B_len_param")
        stack_parameters("candidates_A_weight_param")
        stack_parameters("candidates_B_weight_param")
        stack_parameters("selected_A_indices_param")
        stack_parameters("selected_B_indices_param")
        stack_parameters("fixed_A_steps_param")
        stack_parameters("fixed_B_steps_param")


class SwitchLoraGptConv1D(_lora.LoRALayer, GptConv1D, SwitchLoraLayer):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = True,
            quantize=None,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            **kwargs):
        GptConv1D.__init__(self, out_features, in_features)
        # The shape of self.weight in is different with that in Linear layer
        # To make the code consistent, swap in_features, out_features
        in_features, out_features = out_features, in_features
        self.in_features = in_features
        self.out_features = out_features
        _lora.LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                 merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        if r > 0:
            self.lora_A = _nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = _nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

        SwitchLoraLayer.__init__(self)

        if quantize is not None:
            raise NotImplementedError("Quantization is not supported.")

        self.quantize = quantize

    def forward(self, x: _torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        result = GptConv1D.forward(self, x)
        if self.r > 0 and not self.merged:
            # result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            result += _F.linear(_F.linear(self.lora_dropout(x), self.lora_B.transpose(0, 1)),
                                self.lora_A.transpose(0, 1)) * self.scaling

        SwitchLoraLayer.forward(self, x)
        return result

    def reset_parameters(self):
        GptConv1D.reset_parameters(self)
        SwitchLoraLayer.reset_parameters(self)

    def merge_AB(self):
        def T(w, layer):
            return w.transpose(0, 1) if layer.fan_in_fan_out else w

        return T(self.lora_B @ self.lora_A, self)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        GptConv1D.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True


class Conv1d(SwitchLoraConv):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(_nn.Conv1d, *args, **kwargs)


class Conv2d(SwitchLoraConv):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(_nn.Conv2d, *args, **kwargs)


class Conv3d(SwitchLoraConv):
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(_nn.Conv3d, *args, **kwargs)


layer_replace_dict = {
    _nn.Linear: SwitchLoraLinear,
    _nn.Conv1d: Conv1d,
    _nn.Conv2d: Conv2d,
    _nn.Conv3d: Conv3d,
    # GptConv1D: SwitchLoraGptConv1D
    GptConv1D: SwitchLoRAMergedLinear
}
