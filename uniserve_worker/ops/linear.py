"""Dense projection operations with an explicit accumulator contract."""

import torch


@torch.library.custom_op("uniserve_worker::linear_fp32_accum", mutates_args=())
def linear_fp32_accum(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    """Retain FP32 reductions through the final FP16/BF16 output rounding."""

    from uniserve_kernel.dense_linear import linear_fp32_accum as project

    flat = input.reshape(-1, input.shape[-1]).contiguous()
    return project(flat, weight, bias).reshape(*input.shape[:-1], weight.shape[0])


@linear_fp32_accum.register_fake
def _linear_fp32_accum_fake(input, weight, bias):
    return input.new_empty((*input.shape[:-1], weight.shape[0]))
