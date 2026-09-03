"""H3 inference operations with checkpoint-defined FP32 accumulation boundaries."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.nn import functional as F

__all__ = [
    "apply_partial_rope",
    "qk_rmsnorm_rope",
    "row_modulated_rmsnorm",
    "value_first_swiglu",
]

_FFN_SIZE = 14336
_FFN_SIZE_TL = tl.constexpr(14336)


@triton.jit
def _value_first_swiglu_kernel(value_gate_ptr, output_ptr, elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    row = offsets // _FFN_SIZE_TL
    column = offsets - row * _FFN_SIZE_TL
    value = tl.load(
        value_gate_ptr + row * (2 * _FFN_SIZE_TL) + column,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        value_gate_ptr + row * (2 * _FFN_SIZE_TL) + _FFN_SIZE_TL + column,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    activated_gate = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16)
    tl.store(output_ptr + offsets, value * activated_gate, mask=mask)


def _rmsnorm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    normalized = value.float() * torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + eps)
    return (normalized * weight.float()).to(value.dtype)


def row_modulated_rmsnorm(
    value: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    row_indices: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    normalized = _rmsnorm(value, weight, eps)
    return normalized * (1.0 + scale.index_select(0, row_indices)) + shift.index_select(
        0, row_indices
    )


def apply_partial_rope(
    value: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    rotary = cosine.shape[-1]
    head = value[..., :rotary]
    tail = value[..., rotary:]
    first, second = head.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return torch.cat((head * cosine + rotated * sine, tail), dim=-1)


def qk_rmsnorm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = _rmsnorm(query, query_weight, eps)
    key = _rmsnorm(key, key_weight, eps)
    return apply_partial_rope(query, cosine, sine), apply_partial_rope(key, cosine, sine)


def value_first_swiglu(value_gate: torch.Tensor) -> torch.Tensor:
    if (
        value_gate.is_cuda
        and value_gate.dtype == torch.bfloat16
        and value_gate.shape[-1] == 2 * _FFN_SIZE
    ):
        output = torch.empty(
            (*value_gate.shape[:-1], _FFN_SIZE),
            dtype=value_gate.dtype,
            device=value_gate.device,
        )
        elements = output.numel()
        _value_first_swiglu_kernel[(triton.cdiv(elements, 1024),)](
            value_gate,
            output,
            elements,
            BLOCK=1024,
            num_warps=4,
        )
        return output
    value, gate = value_gate.chunk(2, dim=-1)
    return value * F.silu(gate.float()).to(gate.dtype)
