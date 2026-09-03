"""H3 inference operations with checkpoint-defined FP32 accumulation boundaries."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.nn import functional as F

__all__ = [
    "apply_partial_rope",
    "attention_residual_modulated_rmsnorm",
    "attention_residual_modulated_rmsnorm_fp8",
    "dual_gated_residual",
    "qk_rmsnorm_rope",
    "row_modulated_rmsnorm",
    "value_first_swiglu",
    "value_first_swiglu_fp8",
]

_HIDDEN_SIZE = 5376
_FFN_SIZE = 14336
_HIDDEN_SIZE_TL = tl.constexpr(5376)
_FFN_SIZE_TL = tl.constexpr(14336)
_FP8_MAX_TL = tl.constexpr(448.0)
_SCALE_EPS_TL = tl.constexpr(1.0e-12)


@triton.jit
def _row_modulated_rmsnorm_kernel(
    value_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    row_indices_ptr,
    output_ptr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    accumulator = tl.zeros((BLOCK,), tl.float32)
    for start in tl.range(0, _HIDDEN_SIZE_TL, BLOCK):
        columns = start + offsets
        mask = columns < _HIDDEN_SIZE_TL
        value = tl.load(
            value_ptr + row * _HIDDEN_SIZE_TL + columns,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        accumulator += tl.where(mask, value * value, 0.0)
    inverse_rms = tl.rsqrt(tl.sum(accumulator, axis=0) / _HIDDEN_SIZE_TL + eps)
    modulation_row = tl.load(row_indices_ptr + row)
    for start in tl.range(0, _HIDDEN_SIZE_TL, BLOCK):
        columns = start + offsets
        mask = columns < _HIDDEN_SIZE_TL
        value = tl.load(
            value_ptr + row * _HIDDEN_SIZE_TL + columns,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(
            shift_ptr + modulation_row * _HIDDEN_SIZE_TL + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        scale = tl.load(
            scale_ptr + modulation_row * _HIDDEN_SIZE_TL + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        output = value * inverse_rms * weight * (1.0 + scale) + shift
        tl.store(output_ptr + row * _HIDDEN_SIZE_TL + columns, output, mask=mask)


@triton.jit
def _attention_residual_modulated_rmsnorm_kernel(
    hidden_ptr,
    attention_ptr,
    gate_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    row_indices_ptr,
    output_ptr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    mask = columns < _HIDDEN_SIZE_TL
    modulation_row = tl.load(row_indices_ptr + row)
    hidden = tl.load(
        hidden_ptr + row * _HIDDEN_SIZE_TL + columns,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    attention = tl.load(
        attention_ptr + row * _HIDDEN_SIZE_TL + columns,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    gate = tl.load(
        gate_ptr + modulation_row * _HIDDEN_SIZE_TL + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    residual = hidden + gate * attention
    inverse_rms = tl.rsqrt(tl.sum(residual * residual, axis=0) / _HIDDEN_SIZE_TL + eps)
    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(
        shift_ptr + modulation_row * _HIDDEN_SIZE_TL + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scale = tl.load(
        scale_ptr + modulation_row * _HIDDEN_SIZE_TL + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    output = residual * inverse_rms * weight * (1.0 + scale) + shift
    tl.store(output_ptr + row * _HIDDEN_SIZE_TL + columns, output, mask=mask)


@triton.jit
def _attention_residual_modulated_rmsnorm_fp8_kernel(
    hidden_ptr,
    attention_ptr,
    gate_ptr,
    weight_ptr,
    shift_ptr,
    modulation_scale_ptr,
    row_indices_ptr,
    output_ptr,
    output_scale_ptr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    mask = columns < _HIDDEN_SIZE_TL
    modulation_row = tl.load(row_indices_ptr + row)
    hidden = tl.load(
        hidden_ptr + row * _HIDDEN_SIZE_TL + columns,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    attention = tl.load(
        attention_ptr + row * _HIDDEN_SIZE_TL + columns,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    gate = tl.load(
        gate_ptr + modulation_row * _HIDDEN_SIZE_TL + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    residual = hidden + gate * attention
    inverse_rms = tl.rsqrt(tl.sum(residual * residual, axis=0) / _HIDDEN_SIZE_TL + eps)
    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(
        shift_ptr + modulation_row * _HIDDEN_SIZE_TL + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    modulation_scale = tl.load(
        modulation_scale_ptr + modulation_row * _HIDDEN_SIZE_TL + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    output = tl.where(
        mask,
        residual * inverse_rms * weight * (1.0 + modulation_scale) + shift,
        0.0,
    )
    output = output.to(tl.bfloat16).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(output), axis=0), _SCALE_EPS_TL) / _FP8_MAX_TL
    quantized = tl.maximum(tl.minimum(output / scale, _FP8_MAX_TL), -_FP8_MAX_TL)
    tl.store(output_ptr + row * _HIDDEN_SIZE_TL + columns, quantized, mask=mask)
    tl.store(output_scale_ptr + row, scale)


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
    output = value * gate / (1.0 + tl.exp(-gate))
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _value_first_swiglu_fp8_kernel(
    value_gate_ptr,
    output_ptr,
    output_scale_ptr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    mask = columns < _FFN_SIZE_TL
    value = tl.load(
        value_gate_ptr + row * (2 * _FFN_SIZE_TL) + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        value_gate_ptr + row * (2 * _FFN_SIZE_TL) + _FFN_SIZE_TL + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    output = tl.where(mask, value * gate / (1.0 + tl.exp(-gate)), 0.0)
    output = output.to(tl.bfloat16).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(output), axis=0), _SCALE_EPS_TL) / _FP8_MAX_TL
    quantized = tl.maximum(tl.minimum(output / scale, _FP8_MAX_TL), -_FP8_MAX_TL)
    tl.store(output_ptr + row * _FFN_SIZE_TL + columns, quantized, mask=mask)
    tl.store(output_scale_ptr + row, scale)


@triton.jit
def _dual_gated_residual_kernel(
    hidden_ptr,
    attention_ptr,
    feed_forward_ptr,
    attention_gate_ptr,
    feed_forward_gate_ptr,
    row_indices_ptr,
    elements,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    row = offsets // _HIDDEN_SIZE_TL
    column = offsets - row * _HIDDEN_SIZE_TL
    modulation_row = tl.load(row_indices_ptr + row, mask=mask, other=0)
    hidden = tl.load(hidden_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    attention = tl.load(attention_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    feed_forward = tl.load(feed_forward_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    attention_gate = tl.load(
        attention_gate_ptr + modulation_row * _HIDDEN_SIZE_TL + column,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    feed_forward_gate = tl.load(
        feed_forward_gate_ptr + modulation_row * _HIDDEN_SIZE_TL + column,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    output = hidden + attention_gate * attention + feed_forward_gate * feed_forward
    tl.store(attention_ptr + offsets, output, mask=mask)


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
    if value.is_cuda and value.dtype == torch.bfloat16 and value.shape[-1] == _HIDDEN_SIZE:
        output = torch.empty_like(value)
        rows = value.numel() // _HIDDEN_SIZE
        _row_modulated_rmsnorm_kernel[(rows,)](
            value,
            weight,
            shift,
            scale,
            row_indices,
            output,
            eps,
            BLOCK=1024,
            num_warps=8,
        )
        return output
    normalized = _rmsnorm(value, weight, eps)
    return normalized * (1.0 + scale.index_select(0, row_indices)) + shift.index_select(
        0, row_indices
    )


def attention_residual_modulated_rmsnorm(
    hidden: torch.Tensor,
    attention: torch.Tensor,
    attention_gate: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    row_indices: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    output = torch.empty_like(hidden)
    rows = hidden.numel() // _HIDDEN_SIZE
    _attention_residual_modulated_rmsnorm_kernel[(rows,)](
        hidden,
        attention,
        attention_gate,
        weight,
        shift,
        scale,
        row_indices,
        output,
        eps,
        BLOCK=8192,
        num_warps=16,
    )
    return output


def attention_residual_modulated_rmsnorm_fp8(
    hidden: torch.Tensor,
    attention: torch.Tensor,
    attention_gate: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    row_indices: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = hidden.numel() // _HIDDEN_SIZE
    output = torch.empty_like(hidden, dtype=torch.float8_e4m3fn)
    output_scale = torch.empty((rows, 1), dtype=torch.float32, device=hidden.device)
    _attention_residual_modulated_rmsnorm_fp8_kernel[(rows,)](
        hidden,
        attention,
        attention_gate,
        weight,
        shift,
        scale,
        row_indices,
        output,
        output_scale,
        eps,
        BLOCK=8192,
        num_warps=16,
    )
    return output, output_scale


def dual_gated_residual(
    hidden: torch.Tensor,
    attention: torch.Tensor,
    feed_forward: torch.Tensor,
    attention_gate: torch.Tensor,
    feed_forward_gate: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor:
    elements = hidden.numel()
    _dual_gated_residual_kernel[(triton.cdiv(elements, 1024),)](
        hidden,
        attention,
        feed_forward,
        attention_gate,
        feed_forward_gate,
        row_indices,
        elements,
        BLOCK=1024,
        num_warps=4,
    )
    return attention


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
        output = torch.empty((*value_gate.shape[:-1], _FFN_SIZE), dtype=value_gate.dtype, device=value_gate.device)
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


def value_first_swiglu_fp8(value_gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not value_gate.is_cuda or value_gate.dtype != torch.bfloat16:
        raise RuntimeError("fused SwiGLU FP8 execution requires bfloat16 CUDA input")
    if value_gate.shape[-1] != 2 * _FFN_SIZE:
        raise ValueError(f"fused SwiGLU FP8 width must be {2 * _FFN_SIZE}")
    rows = value_gate.numel() // (2 * _FFN_SIZE)
    output = torch.empty(
        (*value_gate.shape[:-1], _FFN_SIZE),
        dtype=torch.float8_e4m3fn,
        device=value_gate.device,
    )
    output_scale = torch.empty((rows, 1), dtype=torch.float32, device=value_gate.device)
    _value_first_swiglu_fp8_kernel[(rows,)](
        value_gate,
        output,
        output_scale,
        BLOCK=16384,
        num_warps=16,
    )
    return output, output_scale
