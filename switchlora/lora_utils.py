import os.path

import loralib as _lora
import torch as _torch
import torch.nn as _nn
import math as _math
from typing import List as _List
from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
import json as _json
import os as _os
import torchinfo
import matplotlib.pyplot as _plt
from copy import deepcopy as _deepcopy
import bitsandbytes as bnb
import bitsandbytes.functional as bnbF
from transformers.pytorch_utils import Conv1D as GptConv1D

_lora_layer = _lora.Linear
_init_lora_type = None


def set_use_lora(lora_linear_layer):
    global _lora_layer
    _lora_layer = lora_linear_layer


def layer2lora(origin_layer, device, rank: int | None = None, **kwargs):
    if hasattr(origin_layer, "in_features"):
        in_features = origin_layer.in_features
        out_features = origin_layer.out_features
    elif hasattr(origin_layer, "nf"):  # is GPT Conv layer
        # nf (`int`): The number of output features.
        # nx (`int`): The number of input features. But not set as member variable
        # Note that Conv.weight is (in_features, out_features)
        in_features = origin_layer.weight.shape[0]
        out_features = origin_layer.nf
    else:
        raise NotImplementedError("Unknown layer to convert to LoRA.")
    if rank is None or rank <= 0:
        rank = max(min(in_features, out_features) // 8, 8)
    if _lora_layer is None:
        lora_linear = NoLoraLinear(in_features, out_features, bias=origin_layer.bias is not None)
    elif isinstance(_lora_layer, _Mapping):
        lora_layer_cls = _lora_layer[type(origin_layer)]
        if "r" in kwargs:
            lora_linear = lora_layer_cls(in_features, out_features, bias=origin_layer.bias is not None, **kwargs)
        else:
            lora_linear = lora_layer_cls(in_features, out_features, r=rank, bias=origin_layer.bias is not None,
                                         **kwargs)
    elif issubclass(_lora_layer, _nn.Module):
        if "r" in kwargs:
            lora_linear = _lora_layer(in_features, out_features, bias=origin_layer.bias is not None, **kwargs)
        else:
            lora_linear = _lora_layer(in_features, out_features, r=rank, bias=origin_layer.bias is not None, **kwargs)
    else:
        raise ValueError("lora_layer should be a subclass of lora or a dict mapping normal module to lora subclass")
    weight_transposed = False
    if hasattr(lora_linear, 'lora_adapters'):
        # The shape of MergedLinear.weight is (out_features, in_features)
        # while the shape of GPT Conv.weight is (in_features, out_features)
        weight_transposed = True
    lora_linear = lora_linear.to(device)
    def T(w, layer):
        return w.transpose(0, 1) if layer.fan_in_fan_out else w
    with _torch.no_grad():
        lora_linear.weight.copy_(origin_layer.weight.transpose(0, 1).contiguous() if weight_transposed
                                 else origin_layer.weight.contiguous())
        if hasattr(lora_linear, "lora_A"):
            lora_linear.weight.data -= lora_linear.merge_AB() * lora_linear.scaling
        if origin_layer.bias is not None:
            lora_linear.bias.copy_(origin_layer.bias)
    return lora_linear


def _replace_with_lora(layer, linear_list, device, **kwargs):
    if type(linear_list) == str:
        if linear_list == '*':
            for i in range(len(layer)):
                layer[i] = layer2lora(layer[i], device, **kwargs)
        elif linear_list.isnumeric():
            index = int(linear_list)
            layer[index] = layer2lora(layer[index], device, **kwargs)
        else:  # is normal name string
            setattr(layer, linear_list, layer2lora(getattr(layer, linear_list), device, **kwargs))
    elif isinstance(linear_list, _Mapping):
        contain_star = False
        for linear_name in linear_list:
            if linear_name == '*':
                contain_star = True
                break
        if contain_star:
            for sublayer in layer:
                _replace_with_lora(getattr(layer, sublayer), linear_list['*'], device, **kwargs)
        else:
            for linear_name in linear_list:
                _replace_with_lora(getattr(layer, linear_name), linear_list[linear_name], device, **kwargs)
    elif isinstance(linear_list, _Iterable):
        i = 0
        linear_name = None
        for v in linear_list:
            i += 1
            if i == 1:
                linear_name = v
            elif i == 2:
                rank = v
        is_list = False
        if i == 2 and isinstance(rank, int):  # is [linear_name, rank] format
            if linear_name.isnumeric():
                index = int(linear_name)
                layer[index] = layer2lora(layer[index], device, rank, **kwargs)
            elif linear_name == '*':
                for i in range(len(layer)):
                    layer[i] = layer2lora(layer[i], device, rank, **kwargs)
            else:  # is normal name string
                setattr(layer, linear_name, layer2lora(getattr(layer, linear_name), device, rank, **kwargs))
        else:  # is layers list
            is_list = True

        if is_list:
            for linear_name in linear_list:
                _replace_with_lora(layer, linear_name, device, **kwargs)


def replace_with_lora(layer_dicts, device, **kwargs):
    """
    :param layer_dicts:
    :return:

    Examples:
    replace_with_lora({
        model.layer_a:"linear",
        model.layer_b:["linear", 4],
        model.layer_c:["0", 4],
        model.layer_d:[["linear1", 4], "linear2", ["linear3", 12]],
        model.layer_e:[["0", 4], ["1"], ["2", 12]],
        model.layer_f:[["0", 4], ["1"], ["2", 12]],
    }, defualt_rank=4)
    """
    for layer, linear_list in layer_dicts.items():
        _replace_with_lora(layer, linear_list, device, **kwargs)


def _get_submodules(model, key):
    """
    follow way of peft package to get submodule used to replace linear layers with lora
    """
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def _get_lora_class_list(base_layer=False):
    if isinstance(_lora_layer, _Mapping):
        if base_layer:
            lora_class_list = _lora_layer.keys()
        else:
            lora_class_list = _lora_layer.values()
    else:
        raise ValueError("Variable _lora_layer should be a subclass of dict mapping")
    return lora_class_list



def replace_with_lora_auto(model: _nn.Module, replace_list: _List[str], lora_rank: int | None, **kwargs):
    base_lora_class_list = _get_lora_class_list(base_layer=True)
    for name, layer in model.named_modules():
        if not any(isinstance(layer, lora_class) for lora_class in base_lora_class_list):
            continue
        kwargs0 = kwargs.copy()
        if any(name_to_replace in name for name_to_replace in replace_list):
            parent, _, target_name, = _get_submodules(model, name)
            if name.endswith("c_attn"):
                kwargs0['enable_lora'] = [True, True, True]
            if name.endswith('c_fc'):
                kwargs0['enable_lora'] = [True, True, True, True]
            lora_layer = layer2lora(layer, model.device, lora_rank, **kwargs0)
            setattr(parent, target_name, lora_layer)


def obtain_lora_parameters(model):
    lora_A_para = []
    lora_B_para = []
    other_para = []
    all_trainable_para = []
    for name, param in model.named_parameters():
        if 'lora_A' in name:
            lora_A_para.append(param)
        elif 'lora_B' in name:
            lora_B_para.append(param)
            # print("lora grad", param.grad)
        else:
            other_para.append(param)
            # print("other grad", param.grad)
        all_trainable_para.append(param)
        # print(name, param.shape)
    # print("*****************************************************")
    return lora_A_para, lora_B_para, other_para, all_trainable_para


def obtain_lora_Adam_lr(model, lr, change_lora_lr, no_lora_lr_ratio=1.):
    named_parameters = dict(model.named_parameters())
    other_para = []
    lora_A_para = []
    lora_B_para = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("lora_A"):
            lora_A_para.append(param)
        elif name.endswith("lora_B"):
            lora_B_para.append(param)
        else:
            other_para.append(param)
    optim_args_list = []
    optim_args_list.append({"params": lora_A_para, "lr": lr, "is_lora_A": True})
    optim_args_list.append({"params": lora_B_para, "lr": lr, "is_lora_B": True})
    optim_args_list.append({"params": other_para, "lr": lr})
    return optim_args_list



def iter_lora_layers(model, with_name=False):
    lora_class_list = _get_lora_class_list()
    for name, layer in model.named_modules():
        if any(isinstance(layer, lora_class) for lora_class in lora_class_list):
            if with_name:
                yield name, layer
            else:
                yield layer


@_torch.no_grad()
def load_lora(lora_layer, state_dicts):
    def T(w, layer):
        return w.transpose(0, 1) if layer.fan_in_fan_out else w
    lora_layer.weight.data += T(lora_layer.lora_B @ lora_layer.lora_A, lora_layer) * lora_layer.scaling
    for key, value in state_dicts.items():
        if key.endswith("lora_A"):
            lora_layer.lora_A.copy_(value)
        if key.endswith("lora_B"):
            lora_layer.lora_B.copy_(value)
    lora_layer.weight.data -= T(lora_layer.lora_B @ lora_layer.lora_A, lora_layer) * lora_layer.scaling


_mat_gathered_info_dict = {}
_mat_detailed_info_dict = {}


def _print_mat_info(layer_name, mat, mat_name):
    vec = _torch.flatten(mat)
    avg = _torch.mean(vec)
    max = _torch.max(vec)
    min = _torch.min(vec)
    std = _torch.std(vec)
    print(
        f"matrix ({mat.shape[0]}x{mat.shape[1]}) {mat_name}. avg: {avg:.2g}, max: {max:.2g}, min: {min:.2g}, std: {std:.2g}.")
    if mat_name not in _mat_gathered_info_dict:
        _mat_gathered_info_dict[mat_name] = {
            "count": 0,
            "avg": 0.,
            "max": 0.,
            "min": 0.,
            "std": 0.
        }
    _mat_gathered_info_dict[mat_name]["count"] += 1
    _mat_gathered_info_dict[mat_name]["avg"] += avg
    _mat_gathered_info_dict[mat_name]["max"] += max
    _mat_gathered_info_dict[mat_name]["min"] += min
    _mat_gathered_info_dict[mat_name]["std"] += std

    if layer_name not in _mat_detailed_info_dict:
        _mat_detailed_info_dict[layer_name] = {}
    _mat_detailed_info_dict[layer_name][mat_name] = {
        "avg": avg.item(),
        "max": max.item(),
        "min": min.item(),
        "std": std.item(),
    }


def _clear_mat_info():
    mat_info = {}
    for k, v in _mat_gathered_info_dict.items():
        count = v["count"]
        print(f"avg of all layers matrix {k}. avg: {v['avg'] / count:.2g},"
              f" max: {v['max'] / count:.2g},"
              f" min: {v['min'] / count:.2g},"
              f" std: {v['std'] / count:.2g}.")
        mat_info[k] = {"avg": v['avg'] / count,
                       "max": v['max'] / count,
                       "min": v['min'] / count,
                       "std": v['std'] / count,
                       }
    returned_dict = _mat_detailed_info_dict.copy()
    _mat_gathered_info_dict.clear()
    _mat_detailed_info_dict.clear()
    return returned_dict


CAL_DELTA_NORM = False


def count_parameters(model, only_trainable):
    torchinfo.summary(model, (1, 128), dtypes=[_torch.long], verbose=2)
    print(f"Modules,   Parameters. only_trainable={only_trainable}")
    total_params = 0
    table = {}
    for name, parameter in model.named_parameters():
        if only_trainable and not parameter.requires_grad:
            continue
        params = parameter.numel()
        if name not in table:
            table[name] = 0
        table[name] += params
        total_params += params
    return total_params


@_torch.no_grad()
def cal_delta_norm(model, use_lora, print_abs=False):
    if not CAL_DELTA_NORM:
        return None

    def print_or_abs(layer_name, mat, odd_for_abs, mat_name):
        if print_abs:
            _torch.abs(mat, out=odd_for_abs)
            _print_mat_info(layer_name, odd_for_abs, mat_name)
        else:
            _print_mat_info(layer_name, mat, mat_name)

    if use_lora:
        for name, layer in iter_lora_layers(model, with_name=True):
            if layer.Ax.device != layer.lora_B.device:
                device = layer.lora_B.device
                layer.Ax = layer.Ax.to(device)
                layer.BAx = layer.BAx.to(device)
                layer.Wx = layer.Wx.to(device)
                layer.BA = layer.BA.to(device)
                layer.deltaA = layer.deltaA.to(device)
                layer.deltaB = layer.deltaB.to(device)
                layer.deltaW = layer.deltaW.to(layer.weight.device)
            print_or_abs(name, layer.weight, layer.BA, "weight")
            print_or_abs(name, layer.lora_A, layer.deltaA, "A")
            print_or_abs(name, layer.lora_A.grad, layer.deltaA, "deltaA")
            print_or_abs(name, layer.lora_B, layer.deltaB, "B")
            print_or_abs(name, layer.lora_B.grad, layer.deltaB, "deltaB")
            _torch.matmul(layer.lora_B, layer.lora_A.grad, out=layer.BA)
            print_or_abs(name, layer.BA, layer.BA, "B*deltaA")
            _torch.matmul(layer.lora_B.grad, layer.lora_A, out=layer.BA)
            print_or_abs(name, layer.BA, layer.BA, "deltaB*A")
            if layer.weight.requires_grad:
                print_or_abs(name, layer.weight, layer.deltaW, "weight")
                print_or_abs(name, layer.weight.grad, layer.deltaW, "delta weight")

            print("----------------------------------------------------")
    else:
        for name, layer in iter_lora_layers(model, with_name=True):
            if layer.deltaW.device != layer.weight.device:
                layer.deltaW = layer.deltaW.to(layer.weight.device)
            print_or_abs(name, layer.weight, layer.deltaW, "weight")
            print_or_abs(name, layer.weight.grad, layer.deltaW, "delta weight")
            print("----------------------------------------------------")
    return _clear_mat_info()



def set_init_lora_method(method: str):
    global _init_lora_type
    _init_lora_type = method


def get_init_lora_bound(in_features: int, out_features: int, rank: int):
    """
    Follow Xavier and Kaiming initialization to keep std of output of lora_B*lora_A uniform.
    But make A*deltaB and B*deltaA the same size.
    """
    global _init_lora_type
    m = out_features
    n = in_features
    r = rank
    gain = _nn.init.calculate_gain("leaky_relu", param=_math.sqrt(5))
    if _init_lora_type == "origin_lora":
        fan = n
        std_A = gain / _math.sqrt(fan)
        std_B = 0.
    else:
        std_A = ((m ** 0.5 * r) / (n ** (3. / 2))) ** (1. / 4) * gain ** 0.5
        std_B = (r / (m * n) ** 0.5) ** (1. / 4) * gain ** 0.5
    bound_A = _math.sqrt(3.0) * std_A
    bound_B = _math.sqrt(3.0) * std_B
    return bound_A, bound_B


import os


def clip_path(file_path):
    """
    Clips the directory path of the given file path so that it contains at most two directory names.

    Parameters:
        file_path (str): The full path to the file.

    Returns:
        str: The clipped directory path containing at most two directory names.
    """
    # Normalize the file path to handle different OS path conventions
    normalized_path = os.path.normpath(file_path)

    # Extract the directory part of the path
    directory_path = os.path.dirname(normalized_path)

    # Split the directory path into its components
    path_parts = directory_path.split(os.sep)

    # Clip the path to include at most the last two directories
    clipped_path = os.sep.join(path_parts[-2:]) if len(path_parts) >= 2 else path_parts[-1]

    return clipped_path


@_torch.no_grad()
def rank_dist(model, use_lora=True, save_path=None, to_plot=False):
    def get_merged_weight(layer):
        if use_lora:
            if layer.quantize is None:
                return layer.weight + layer.lora_B @ layer.lora_A * layer.scaling
            else:
                odd_mat = layer.odd_mat
                bnbF.dequantize_4bit(layer.weight.data, layer.weight.quant_state, out=odd_mat)
                odd_mat += layer.lora_B @ layer.lora_A * layer.scaling
                layer.weight.data, layer.weight.quant_state = bnbF.quantize_4bit(
                    odd_mat,
                    quant_type=layer.weight.quant_type,
                    compress_statistics=layer.weight.compress_statistics,
                )
                return odd_mat

        else:
            return layer.weight

    # Store the original model's dtype to restore later
    # original_dtype = next(model.parameters()).dtype

    # Convert model to float32 since torch.svd does not support half precision
    model = _deepcopy(model)
    model.float()

    def save_fig(base_path, ranks, rank_name):
        plot_path = _os.path.join(base_path, rank_name)
        plot_label = _os.path.join(clip_path(plot_path))
        _plt.hist([r for r_list in ranks for r in r_list], density=True, bins=100, alpha=0.9,
                  label=plot_label)
        _plt.legend()
        _plt.savefig(plot_path, bbox_inches='tight')
        _plt.clf()

    # get singular values of all layers
    q_projs = []
    k_projs = []
    v_projs = []
    o_projs = []
    gate_projs = []
    down_projs = []
    up_projs = []

    for layer in model.layers:
        q_projs_weight = get_merged_weight(layer.self_attn.q_proj).detach()
        singular_values = _torch.svd(q_projs_weight).S
        q_projs.append(singular_values.cpu().tolist())

        k_projs_weight = get_merged_weight(layer.self_attn.k_proj).detach()
        singular_values = _torch.svd(k_projs_weight).S
        k_projs.append(singular_values.cpu().tolist())

        v_projs_weight = get_merged_weight(layer.self_attn.v_proj).detach()
        singular_values = _torch.svd(v_projs_weight).S
        v_projs.append(singular_values.cpu().tolist())

        o_projs_weight = get_merged_weight(layer.self_attn.o_proj).detach()
        singular_values = _torch.svd(o_projs_weight).S
        o_projs.append(singular_values.cpu().tolist())

        gate_projs_weight = get_merged_weight(layer.mlp.gate_proj).detach()
        singular_values = _torch.svd(gate_projs_weight).S
        gate_projs.append(singular_values.cpu().tolist())

        down_projs_weight = get_merged_weight(layer.mlp.down_proj).detach()
        singular_values = _torch.svd(down_projs_weight).S
        down_projs.append(singular_values.cpu().tolist())

        up_projs_weight = get_merged_weight(layer.mlp.up_proj).detach()
        singular_values = _torch.svd(up_projs_weight).S
        up_projs.append(singular_values.cpu().tolist())

    rank_dict = {
        "q_projs": q_projs,
        "k_projs": k_projs,
        "v_projs": v_projs,
        "o_projs": o_projs,
        "gate_projs": gate_projs,
        "down_projs": down_projs,
        "up_projs": up_projs
    }

    if save_path is not None:
        with open(save_path, "w") as f:
            _json.dump(rank_dict, f, indent=4)

    if to_plot:
        if save_path is None:
            raise RuntimeError("save_path not provided.")
        plot_base_path = _os.path.dirname(save_path)
        save_fig(plot_base_path, q_projs, "q_projs")
        save_fig(plot_base_path, k_projs, "k_projs")
        save_fig(plot_base_path, v_projs, "v_projs")
        save_fig(plot_base_path, o_projs, "o_projs")
        save_fig(plot_base_path, gate_projs, "gate_projs")
        save_fig(plot_base_path, down_projs, "down_projs")
        save_fig(plot_base_path, up_projs, "up_projs")

    # Restore the original precision
    # if original_dtype == _torch.bfloat16:
    #     model.bfloat16()
    # elif original_dtype == _torch.float16:
    #     model.half()
    return rank_dict


def estimate_rank(mat, x):
    """
    Estimate the rank of matrix mat with respect to x

    Consider following problem:
    There are n independent variables with same distribution. There mean value is 0 and their variance is 1/9. Please calculate expectation of the absolute value of their sum.
    The expectation is about √(2n/9π) when n is large
    That's to say, when all elements of vectors x and y are uniformly random distributed in [-1, 1], dot(x,y) ~ √(2n/9π)
    It is known that if all elements of v is uniformly random distributed in [-1, 1], norm(v) ~ √(n/3)
    So we can obtain that expectation of absolute value of dot product of two random vector is √(2/nπ)
    """
    x = _torch.nn.functional.normalize(x, p=2, dim=-1)
    mat = _torch.nn.functional.normalize(mat, p=2, dim=0)

    s = abs(x @ mat)

    # √(2/nπ)
    n = mat.shape[0]
    expectation = (2 / (n * _torch.pi)) ** 0.5

    rank = s.mean(dim=list(range(len(x.shape) - 1)))
    return rank / expectation


class NoLoraLinear(_torch.nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            **kwargs
    ):
        _torch.nn.Linear.__init__(self, in_features, out_features, **kwargs)
        if CAL_DELTA_NORM:
            self.deltaW = _torch.empty(out_features, in_features)


