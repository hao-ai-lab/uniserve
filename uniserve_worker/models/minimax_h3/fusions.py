"""Implements H3 transformer fusions with checkpoint-defined BF16 boundaries.

The operations combine row-indexed modulation, residual gates, normalization,
partial rotary embedding, and value-first SwiGLU. Fixed-width Triton paths make
the rounding order explicit; FP8 variants publish the scale required to recover
each quantized output row.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.nn import functional as F

__all__ = [
    "apply_partial_rope",
    "attention_residual_modulated_rmsnorm",
    "attention_residual_modulated_rmsnorm_fp8",
    "gated_residual",
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
def _round_to_bf16(value):
    """Round FP32 values through BF16 with explicit nearest-even PTX semantics."""

    return tl.inline_asm_elementwise(
        asm="""{
        .reg .b16 rounded;
        cvt.rn.bf16.f32 rounded, $1;
        cvt.f32.bf16 $0, rounded;
        }""",
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _divide_rn(dividend, divisor):
    """Divide FP32 operands with explicit nearest-even PTX semantics."""

    return tl.inline_asm_elementwise(
        asm="div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[dividend, divisor],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _multiply_rn(left, right):
    """Multiply FP32 operands with explicit nearest-even PTX semantics."""

    return tl.inline_asm_elementwise(
        asm="mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _row_normalized_modulation_kernel(
    value_ptr,
    weight_ptr,
    inverse_rms_ptr,
    shift_ptr,
    scale_ptr,
    row_indices_ptr,
    output_ptr,
    shift_row_stride,
    scale_row_stride,
    elements,
    BLOCK: tl.constexpr,
):
    """Fuse RMS scaling with row-selected affine modulation at BF16 boundaries."""

    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    row = offsets // _HIDDEN_SIZE_TL
    column = offsets - row * _HIDDEN_SIZE_TL
    modulation_row = tl.load(row_indices_ptr + row, mask=mask, other=0)

    # Rounding after weight and scale multiplication matches checkpoint inference.
    value = tl.load(
        value_ptr + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    weight = tl.load(weight_ptr + column, mask=mask, other=0.0).to(tl.float32)
    inverse_rms = tl.load(inverse_rms_ptr + row, mask=mask, other=0.0).to(tl.float32)
    normalized = _multiply_rn(value, inverse_rms)
    normalized = _round_to_bf16(_multiply_rn(normalized, weight))
    shift = tl.load(
        shift_ptr + modulation_row * shift_row_stride + column,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scale = tl.load(
        scale_ptr + modulation_row * scale_row_stride + column,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scale = _round_to_bf16(1.0 + scale)
    modulated = _round_to_bf16(normalized * scale)
    output = (modulated + shift).to(tl.bfloat16)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _gated_residual_kernel(
    hidden_ptr,
    update_ptr,
    gate_ptr,
    row_indices_ptr,
    output_ptr,
    gate_row_stride,
    elements,
    BLOCK: tl.constexpr,
):
    """Add a row-selected gated update using checkpoint BF16 rounding points."""

    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    row = offsets // _HIDDEN_SIZE_TL
    column = offsets - row * _HIDDEN_SIZE_TL
    modulation_row = tl.load(row_indices_ptr + row, mask=mask, other=0)
    hidden = tl.load(hidden_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    update = tl.load(update_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(
        gate_ptr + modulation_row * gate_row_stride + column,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    update = _round_to_bf16(gate * update)
    output = (hidden + update).to(tl.bfloat16)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _row_normalized_modulation_fp8_kernel(
    value_ptr,
    weight_ptr,
    inverse_rms_ptr,
    shift_ptr,
    scale_ptr,
    row_indices_ptr,
    output_ptr,
    output_scale_ptr,
    shift_row_stride,
    scale_row_stride,
    BLOCK: tl.constexpr,
):
    """Modulate one hidden row and emit tensorwise-scaled E4M3 values."""

    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    mask = columns < _HIDDEN_SIZE_TL
    modulation_row = tl.load(row_indices_ptr + row)
    value = tl.load(
        value_ptr + row * _HIDDEN_SIZE_TL + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    inverse_rms = tl.load(inverse_rms_ptr + row).to(tl.float32)
    normalized = _multiply_rn(value, inverse_rms)
    normalized = _round_to_bf16(_multiply_rn(normalized, weight))
    shift = tl.load(
        shift_ptr + modulation_row * shift_row_stride + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scale = tl.load(
        scale_ptr + modulation_row * scale_row_stride + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scale = _round_to_bf16(1.0 + scale)
    modulated = _round_to_bf16(normalized * scale)
    output = (modulated + shift).to(tl.bfloat16)

    # Each row carries the reciprocal dequantization scale consumed by FP8 GEMM.
    output_fp32 = tl.where(mask, output.to(tl.float32), 0.0)
    output_scale = tl.maximum(tl.max(tl.abs(output_fp32), axis=0), _SCALE_EPS_TL)
    output_scale /= _FP8_MAX_TL
    quantized = tl.maximum(
        tl.minimum(_divide_rn(output_fp32, output_scale), _FP8_MAX_TL),
        -_FP8_MAX_TL,
    )
    tl.store(output_ptr + row * _HIDDEN_SIZE_TL + columns, quantized, mask=mask)
    tl.store(output_scale_ptr + row, output_scale)


@triton.jit
def _value_first_swiglu_kernel(value_gate_ptr, output_ptr, elements, BLOCK: tl.constexpr):
    """Evaluate fixed-width SwiGLU from a packed value-then-gate projection."""

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
    output = (value.to(tl.bfloat16) * activated_gate).to(tl.bfloat16)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _value_first_swiglu_fp8_kernel(
    value_gate_ptr,
    output_ptr,
    output_scale_ptr,
    BLOCK: tl.constexpr,
):
    """Evaluate one packed SwiGLU row and quantize it with a rowwise scale."""

    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    mask = columns < _FFN_SIZE_TL
    value = tl.load(
        value_gate_ptr + row * (2 * _FFN_SIZE_TL) + columns,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    gate = tl.load(
        value_gate_ptr + row * (2 * _FFN_SIZE_TL) + _FFN_SIZE_TL + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    activated_gate = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16)
    output = (value * activated_gate).to(tl.bfloat16)
    output_fp32 = tl.where(mask, output.to(tl.float32), 0.0)
    output_scale = tl.maximum(tl.max(tl.abs(output_fp32), axis=0), _SCALE_EPS_TL)
    output_scale /= _FP8_MAX_TL
    quantized = tl.maximum(
        tl.minimum(_divide_rn(output_fp32, output_scale), _FP8_MAX_TL),
        -_FP8_MAX_TL,
    )
    tl.store(output_ptr + row * _FFN_SIZE_TL + columns, quantized, mask=mask)
    tl.store(output_scale_ptr + row, output_scale)


def _rmsnorm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply RMS normalization in FP32 and restore the operand dtype."""

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
    """RMS-normalize hidden rows and apply row-indexed shift and scale vectors."""

    if value.is_cuda and value.dtype == torch.bfloat16 and value.shape[-1] == _HIDDEN_SIZE:
        inverse_rms = torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + eps)
        output = torch.empty_like(value)
        elements = value.numel()
        _row_normalized_modulation_kernel[(triton.cdiv(elements, 1024),)](
            value,
            weight,
            inverse_rms,
            shift,
            scale,
            row_indices,
            output,
            shift.stride(0),
            scale.stride(0),
            elements,
            BLOCK=1024,
            num_warps=4,
        )
        return output

    normalized = _rmsnorm(value, weight, eps)
    return normalized * (1.0 + scale.index_select(0, row_indices)) + shift.index_select(
        0, row_indices
    )


def gated_residual(
    hidden: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor:
    """Add row-indexed gated updates to the residual stream.

    The fixed-width CUDA path reuses ``update`` as its output buffer; callers must
    treat that input as consumed after the call.
    """

    if hidden.is_cuda and hidden.dtype == torch.bfloat16 and hidden.shape[-1] == _HIDDEN_SIZE:
        elements = hidden.numel()
        _gated_residual_kernel[(triton.cdiv(elements, 1024),)](
            hidden,
            update,
            gate,
            row_indices,
            update,
            gate.stride(0),
            elements,
            BLOCK=1024,
            num_warps=4,
        )
        return update

    return hidden + gate.index_select(0, row_indices) * update


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the gated attention residual and its row-modulated normalization."""

    residual = gated_residual(hidden, attention, attention_gate, row_indices)
    normalized = row_modulated_rmsnorm(
        residual,
        weight,
        shift,
        scale,
        row_indices,
        eps=eps,
    )
    return residual, normalized


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Produce a gated residual plus row-scaled FP8 modulated normalization."""

    residual = gated_residual(hidden, attention, attention_gate, row_indices)
    inverse_rms = torch.rsqrt(residual.float().pow(2).mean(-1, keepdim=True) + eps)
    rows = residual.numel() // _HIDDEN_SIZE
    output = torch.empty_like(residual, dtype=torch.float8_e4m3fn)
    output_scale = torch.empty((rows, 1), dtype=torch.float32, device=hidden.device)
    _row_normalized_modulation_fp8_kernel[(rows,)](
        residual,
        weight,
        inverse_rms,
        shift,
        scale,
        row_indices,
        output,
        output_scale,
        shift.stride(0),
        scale.stride(0),
        BLOCK=8192,
        num_warps=16,
    )
    return residual, output, output_scale


def apply_partial_rope(
    value: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    """Apply split-half rotary embedding to the leading coordinates of each head."""

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
    """RMS-normalize query and key heads before applying partial rotary embedding."""

    query = _rmsnorm(query, query_weight, eps)
    key = _rmsnorm(key, key_weight, eps)
    return apply_partial_rope(query, cosine, sine), apply_partial_rope(key, cosine, sine)


def value_first_swiglu(value_gate: torch.Tensor) -> torch.Tensor:
    """Apply SwiGLU to a projection packed in value-then-gate order."""

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


def value_first_swiglu_fp8(value_gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed-width SwiGLU output in E4M3 form with one scale per row."""

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
