from torch.optim import AdamW, Optimizer
from torch import Tensor
from typing import List, Optional, Tuple, Union
import torch
import math
from torch._utils import is_compiling
import functools
from torch.utils._foreach_utils import (
    _get_foreach_kernels_supported_devices,
    _get_fused_kernels_supported_devices,
)

_foreach_supported_types = [torch.Tensor, torch.nn.parameter.Parameter]


# For any optimizer with a faster implementation, we attempt to default to the
# fastest + stablest whenever possible. For foreach, the requirements are to have
# native params all on CUDA. For fused, there's currently the additional requirement
# that the tensors' dtypes must be floating point. Neither alternative supports
# torch.jit.script nor differentiable, so we fall back to the single tensor
# implementation in those cases.
def _default_to_fused_or_foreach(params: List[torch.Tensor],
                                 differentiable: bool,
                                 use_fused: bool = False) -> Tuple[bool, bool]:
    if torch.jit.is_scripting() or differentiable:
        return False, False

    fused_supported_devices = _get_fused_kernels_supported_devices()
    foreach_supported_devices = _get_foreach_kernels_supported_devices()
    fused = use_fused and all(
        p is None or (type(p) in _foreach_supported_types and
                      p.device.type in fused_supported_devices and
                      torch.is_floating_point(p)) for p in params
    )
    foreach = not fused and all(
        p is None or (type(p) in _foreach_supported_types and
                      p.device.type in foreach_supported_devices) for p in params
    )
    return fused, foreach


def _foreach_addcdiv(inputs, tensor1s, tensor2s, value=1.0):
    """
    Apply the addcdiv operation to each group of tensors from the inputs, tensor1s, and tensor2s lists.

    Args:
    inputs (list of Tensor): The tensors to which the operation will be applied.
    tensor1s (list of Tensor): The first tensor in the division.
    tensor2s (list of Tensor): The second tensor in the division.
    value (float): A scaling factor to apply to the tensor1/tensor2 division result before adding it to input.

    Returns:
    list of Tensor: The modified list of input tensors after the addcdiv operation.
    """
    # Check that all lists have the same length
    if not (len(inputs) == len(tensor1s) == len(tensor2s)):
        raise ValueError("All input lists must have the same length.")

    # Perform the operation in-place for each group of tensors
    for i in range(len(inputs)):
        inputs[i].addcdiv_(tensor1s[i], tensor2s[i], value=value)

    return inputs


def _use_grad_for_differentiable(func):
    def _use_grad(self, *args, **kwargs):
        import torch._dynamo
        prev_grad = torch.is_grad_enabled()
        try:
            # Note on graph break below:
            # we need to graph break to ensure that aot respects the no_grad annotation.
            # This is important for perf because without this, functionalization will generate an epilogue
            # which updates the mutated parameters of the optimizer which is *not* visible to inductor, as a result,
            # inductor will allocate for every parameter in the model, which is horrible.
            # With this, aot correctly sees that this is an inference graph, and functionalization will generate
            # an epilogue which is appended to the graph, which *is* visible to inductor, as a result, inductor sees that
            # step is in place and is able to avoid the extra allocation.
            # In the future, we will either 1) continue to graph break on backward, so this graph break does not matter
            # or 2) have a fully fused forward and backward graph, which will have no_grad by default, and we can remove this
            # graph break to allow the fully fused fwd-bwd-optimizer graph to be compiled.
            # see https://github.com/pytorch/pytorch/issues/104053
            torch.set_grad_enabled(self.defaults['differentiable'])
            torch._dynamo.graph_break()
            ret = func(self, *args, **kwargs)
        finally:
            torch._dynamo.graph_break()
            torch.set_grad_enabled(prev_grad)
        return ret

    functools.update_wrapper(_use_grad, func)
    return _use_grad


def _get_value(x):
    # item is significantly faster than a cpu tensor in eager mode
    if not torch.jit.is_scripting() and is_compiling():
        return x
    else:
        return x.item()


def _stack_if_compiling(x):
    if not torch.jit.is_scripting() and is_compiling():
        return torch.stack(x)
    else:
        return x


def _dispatch_sqrt(x: float):  # float annotation is needed because of torchscript type inference
    if not torch.jit.is_scripting() and isinstance(x, torch.Tensor):
        return x.sqrt()
    else:
        return math.sqrt(x)


class StepAdamW(AdamW):
    def _init_group(
            self,
            group,
            params_with_grad,
            grads,
            amsgrad,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            element_wise_step
    ):
        to_init = set()
        for p in group["params"]:
            if p.grad is None:
                continue
            if len(self.state[p]) == 0:
                to_init.add(p)
            if 'step' in self.state[p]:
                if self.state[p]['step'].device != p.device:
                    # step is put to 'cpu' when initialization. Here we change to p.device
                    self.state[p]['step'] = self.state[p]['step'].to(p.device)
        has_complex = super()._init_group(
            group,
            params_with_grad,
            grads,
            amsgrad,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps=[],
        )
        for p in group["params"]:
            state = self.state[p]
            if p in to_init:
                if element_wise_step == "is_lora_A":
                    # state["step"] = torch.zeros_like(
                    #     p, memory_format=torch.preserve_format
                    # ).float()  # cast to float32 since beta1**step and beta2**step will get wrong value with bf16
                    state["step"] = torch.zeros(p.shape[0], 1, device=p.device, dtype=p.dtype).float()
                elif element_wise_step == "is_lora_B":
                    state["step"] = torch.zeros(1, p.shape[1], device=p.device, dtype=p.dtype).float()
                else:
                    assert False
            state_steps.append(state["step"])

        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group["amsgrad"]
            beta1, beta2 = group["betas"]
            if "is_lora_A" in group:
                element_wise_step = "is_lora_A"
            elif "is_lora_B" in group:
                element_wise_step = "is_lora_B"
            else:
                element_wise_step = False

            if element_wise_step:
                has_complex = self._init_group(
                    group,
                    params_with_grad,
                    grads,
                    amsgrad,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    element_wise_step
                )
            else:
                has_complex = super()._init_group(
                    group,
                    params_with_grad,
                    grads,
                    amsgrad,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps
                )

            adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                has_complex=has_complex,
                element_wise_step=element_wise_step
            )

        return loss


def adamw(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        state_steps: List[Tensor],
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
        grad_scale: Optional[Tensor] = None,
        found_inf: Optional[Tensor] = None,
        has_complex: bool = False,
        *,
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: Union[float, Tensor],
        weight_decay: float,
        eps: float,
        maximize: bool,
        element_wise_step: bool
):
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """
    if not torch._utils.is_compiling() and not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if fused or differentiable or capturable:
        raise RuntimeError("Fused and differentiable and capturable are not supported")
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
        # Do not flip on foreach for the unsupported case where lr is a Tensor and capturable=False.
        if foreach and isinstance(lr, Tensor) and not capturable:
            foreach = False
    if fused is None:
        fused = False
    if foreach is None:
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    if fused and not torch.jit.is_scripting():
        raise NotImplementedError()
    elif foreach and not torch.jit.is_scripting():
        if element_wise_step:
            func = _multi_tensor_adamw_element_wise_step
        else:
            func = _multi_tensor_adamw
    else:
        raise NotImplementedError()

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
        has_complex=has_complex,
    )


def _foreach_correction(beta1, beta2, device_state_steps,
                        exp_avg_sq_sqrt, eps,
                        device_exp_avgs, lr,
                        device_params):
    for i in range(len(device_state_steps)):
        bias_correction2_sqrt = _dispatch_sqrt(1 - beta2 ** device_state_steps[i])
        exp_avg_sq_sqrt[i].div_(bias_correction2_sqrt)
        exp_avg_sq_sqrt[i].add_(eps)

        bias_correction1 = 1 - beta1 ** device_state_steps[i]
        step_size = (lr / bias_correction1) * -1
        device_exp_avgs_step = device_exp_avgs[i] * step_size

        device_params[i].data.add_(device_exp_avgs_step / exp_avg_sq_sqrt[i])


def _foreach_correction2(beta2, device_state_steps):
    bias_correction2_sqrt = [_dispatch_sqrt(1 - beta2 ** step)
                             for step in device_state_steps]
    return bias_correction2_sqrt


def _foreach_correction1(beta1, device_state_steps,
                         device_exp_avgs, lr):
    odd_var = []
    for i in range(len(device_state_steps)):
        bias_correction1 = 1 - beta1 ** device_state_steps[i]
        step_size = (lr / bias_correction1) * -1
        device_exp_avgs_step = device_exp_avgs[i] * step_size
        odd_var.append(device_exp_avgs_step)
    return odd_var


def _multi_tensor_adamw_element_wise_step_old(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        state_steps: List[Tensor],
        grad_scale: Optional[Tensor],
        found_inf: Optional[Tensor],
        *,
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: Union[Tensor, float],
        weight_decay: float,
        eps: float,
        maximize: bool,
        capturable: bool,
        differentiable: bool,
        has_complex: bool,
):
    if len(params) == 0:
        return

    if isinstance(lr, Tensor) and not capturable:
        raise RuntimeError("lr as a Tensor is not supported for capturable=False and foreach=True")

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch._utils.is_compiling() and capturable:
        assert all(
            p.is_cuda and step.is_cuda for p, step in zip(params, state_steps)
        ), "If capturable=True, params and state_steps must be CUDA tensors."

    assert not differentiable, "_foreach ops don't support autograd"

    assert grad_scale is None and found_inf is None

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([
        params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps])
    for ((
                 device_params,
                 device_grads,
                 device_exp_avgs,
                 device_exp_avg_sqs,
                 device_max_exp_avg_sqs,
                 device_state_steps,
         ), _) in grouped_tensors.values():
        if maximize:
            device_grads = torch._foreach_neg(device_grads)

        if has_complex:
            raise NotImplementedError()
            # if amsgrad:
            #     _view_as_real(device_params, device_grads, device_exp_avgs, device_exp_avg_sqs, device_max_exp_avg_sqs)
            # else:
            #     _view_as_real(device_params, device_grads, device_exp_avgs, device_exp_avg_sqs)

        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        if device_state_steps[0].is_cpu:
            torch._foreach_add_(device_state_steps, torch.tensor(1.0, device='cpu'), alpha=1.0)
        else:
            torch._foreach_add_(device_state_steps, 1)

        # Perform stepweight decay
        if weight_decay != 0:
            torch._foreach_mul_(device_params, 1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - beta1)

        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads, 1 - beta2)

        # Delete the local intermediate since it won't be used anymore to save on peak memory
        del device_grads

        if capturable:
            raise NotImplementedError()
            bias_correction1 = torch._foreach_pow(beta1, device_state_steps)
            bias_correction2 = torch._foreach_pow(beta2, device_state_steps)
            # foreach_sub doesn't allow a scalar as the first arg
            torch._foreach_sub_(bias_correction1, 1)
            torch._foreach_sub_(bias_correction2, 1)
            # we do not negate bias_correction1 as it'll need to be negated later anyway
            torch._foreach_neg_(bias_correction2)

            # foreach_div doesn't allow a scalar as the first arg
            torch._foreach_div_(bias_correction1, lr)
            torch._foreach_reciprocal_(bias_correction1)

            torch._foreach_sqrt_(bias_correction2)

            # Re-assign for clarity as we maintain minimal intermediates: we'll have
            # step_size = - lr / (1 - beta1 ^ t) where t = num_steps
            # bias_correction2_sqrt = sqrt(1 - beta2 ^ t)
            step_size = bias_correction1
            bias_correction2_sqrt = bias_correction2

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)

                # Use the max. for normalizing running avg. of gradient
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_div_(exp_avg_sq_sqrt, step_size)

            # at this point, exp_avg_sq_sqrt = - (1 - beta^t) * [sqrt(exp_avg_sq / (1 - beta2^t)) + eps] / lr
            torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt)
        else:
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)

                # Use the max. for normalizing running avg. of gradient
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            _foreach_correction(beta1, beta2, device_state_steps,
                                exp_avg_sq_sqrt, eps,
                                device_exp_avgs, lr,
                                device_params)
            continue

            odd_var = _foreach_correction2(beta2, device_state_steps)

            # torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_div_(exp_avg_sq_sqrt, odd_var)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)

            odd_var = _foreach_correction1(beta1, device_state_steps, device_exp_avgs, lr)

            # torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt, step_size)
            #torch._foreach_addcdiv_(device_params, device_exp_avgs_step, exp_avg_sq_sqrt)
            torch._foreach_addcdiv_(device_params, odd_var, exp_avg_sq_sqrt)


def _multi_tensor_adamw_element_wise_step(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        state_steps: List[Tensor],
        grad_scale: Optional[Tensor],
        found_inf: Optional[Tensor],
        *,
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: Union[Tensor, float],
        weight_decay: float,
        eps: float,
        maximize: bool,
        capturable: bool,
        differentiable: bool,
        has_complex: bool,
):
    if len(params) == 0:
        return

    if isinstance(lr, Tensor) and not capturable:
        raise RuntimeError("lr as a Tensor is not supported for capturable=False and foreach=True")

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch._utils.is_compiling() and capturable:
        assert all(
            p.is_cuda and step.is_cuda for p, step in zip(params, state_steps)
        ), "If capturable=True, params and state_steps must be CUDA tensors."

    assert not differentiable, "_foreach ops don't support autograd"

    assert grad_scale is None and found_inf is None

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([
        params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps])
    for ((
                 device_params,
                 device_grads,
                 device_exp_avgs,
                 device_exp_avg_sqs,
                 device_max_exp_avg_sqs,
                 device_state_steps,
         ), _) in grouped_tensors.values():
        if maximize:
            device_grads = torch._foreach_neg(device_grads)

        if has_complex:
            if amsgrad:
                _view_as_real(device_params, device_grads, device_exp_avgs, device_exp_avg_sqs, device_max_exp_avg_sqs)
            else:
                _view_as_real(device_params, device_grads, device_exp_avgs, device_exp_avg_sqs)

        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        if device_state_steps[0].is_cpu:
            torch._foreach_add_(device_state_steps, torch.tensor(1.0, device='cpu'), alpha=1.0)
        else:
            torch._foreach_add_(device_state_steps, 1)

        # Perform stepweight decay
        if weight_decay != 0:
            torch._foreach_mul_(device_params, 1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - beta1)

        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads, 1 - beta2)

        # Delete the local intermediate since it won't be used anymore to save on peak memory
        del device_grads

        if capturable:
            raise NotImplementedError()
            bias_correction1 = torch._foreach_pow(beta1, device_state_steps)
            bias_correction2 = torch._foreach_pow(beta2, device_state_steps)
            # foreach_sub doesn't allow a scalar as the first arg
            torch._foreach_sub_(bias_correction1, 1)
            torch._foreach_sub_(bias_correction2, 1)
            # we do not negate bias_correction1 as it'll need to be negated later anyway
            torch._foreach_neg_(bias_correction2)

            # foreach_div doesn't allow a scalar as the first arg
            torch._foreach_div_(bias_correction1, lr)
            torch._foreach_reciprocal_(bias_correction1)

            torch._foreach_sqrt_(bias_correction2)

            # Re-assign for clarity as we maintain minimal intermediates: we'll have
            # step_size = - lr / (1 - beta1 ^ t) where t = num_steps
            # bias_correction2_sqrt = sqrt(1 - beta2 ^ t)
            step_size = bias_correction1
            bias_correction2_sqrt = bias_correction2

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)

                # Use the max. for normalizing running avg. of gradient
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_div_(exp_avg_sq_sqrt, step_size)

            # at this point, exp_avg_sq_sqrt = - (1 - beta^t) * [sqrt(exp_avg_sq / (1 - beta2^t)) + eps] / lr
            torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt)
        else:
            bias_correction1 = [1 - beta1 ** step for step in device_state_steps]
            bias_correction2 = [1 - beta2 ** step for step in device_state_steps]

            step_size = _stack_if_compiling([(lr / bc) * -1 for bc in bias_correction1])

            bias_correction2_sqrt = [_dispatch_sqrt(bc) for bc in bias_correction2]

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)

                # Use the max. for normalizing running avg. of gradient
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            #torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt, step_size)
            torch._foreach_div_(exp_avg_sq_sqrt, step_size)
            torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt)

def _multi_tensor_adamw(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        state_steps: List[Tensor],
        grad_scale: Optional[Tensor],
        found_inf: Optional[Tensor],
        *,
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: Union[Tensor, float],
        weight_decay: float,
        eps: float,
        maximize: bool,
        capturable: bool,
        differentiable: bool,
        has_complex: bool,
):
    if len(params) == 0:
        return

    if isinstance(lr, Tensor) and not capturable:
        raise RuntimeError("lr as a Tensor is not supported for capturable=False and foreach=True")

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch._utils.is_compiling() and capturable:
        assert all(
            p.is_cuda and step.is_cuda for p, step in zip(params, state_steps)
        ), "If capturable=True, params and state_steps must be CUDA tensors."

    assert not differentiable, "_foreach ops don't support autograd"

    assert grad_scale is None and found_inf is None

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([
        params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps])
    for ((
                 device_params,
                 device_grads,
                 device_exp_avgs,
                 device_exp_avg_sqs,
                 device_max_exp_avg_sqs,
                 device_state_steps,
         ), _) in grouped_tensors.values():
        if maximize:
            device_grads = torch._foreach_neg(device_grads)

        if has_complex:
            if amsgrad:
                _view_as_real(device_params, device_grads, device_exp_avgs, device_exp_avg_sqs, device_max_exp_avg_sqs)
            else:
                _view_as_real(device_params, device_grads, device_exp_avgs, device_exp_avg_sqs)

        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        if device_state_steps[0].is_cpu:
            torch._foreach_add_(device_state_steps, torch.tensor(1.0, device='cpu'), alpha=1.0)
        else:
            torch._foreach_add_(device_state_steps, 1)

        # Perform stepweight decay
        if weight_decay != 0:
            torch._foreach_mul_(device_params, 1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - beta1)

        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads, 1 - beta2)

        # Delete the local intermediate since it won't be used anymore to save on peak memory
        del device_grads

        if capturable:
            bias_correction1 = torch._foreach_pow(beta1, device_state_steps)
            bias_correction2 = torch._foreach_pow(beta2, device_state_steps)
            # foreach_sub doesn't allow a scalar as the first arg
            torch._foreach_sub_(bias_correction1, 1)
            torch._foreach_sub_(bias_correction2, 1)
            # we do not negate bias_correction1 as it'll need to be negated later anyway
            torch._foreach_neg_(bias_correction2)

            # foreach_div doesn't allow a scalar as the first arg
            torch._foreach_div_(bias_correction1, lr)
            torch._foreach_reciprocal_(bias_correction1)

            torch._foreach_sqrt_(bias_correction2)

            # Re-assign for clarity as we maintain minimal intermediates: we'll have
            # step_size = - lr / (1 - beta1 ^ t) where t = num_steps
            # bias_correction2_sqrt = sqrt(1 - beta2 ^ t)
            step_size = bias_correction1
            bias_correction2_sqrt = bias_correction2

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)

                # Use the max. for normalizing running avg. of gradient
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_div_(exp_avg_sq_sqrt, step_size)

            # at this point, exp_avg_sq_sqrt = - (1 - beta^t) * [sqrt(exp_avg_sq / (1 - beta2^t)) + eps] / lr
            torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt)
        else:
            bias_correction1 = [1 - beta1 ** _get_value(step) for step in device_state_steps]
            bias_correction2 = [1 - beta2 ** _get_value(step) for step in device_state_steps]

            step_size = _stack_if_compiling([(lr / bc) * -1 for bc in bias_correction1])

            bias_correction2_sqrt = [_dispatch_sqrt(bc) for bc in bias_correction2]

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)

                # Use the max. for normalizing running avg. of gradient
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt, step_size)
