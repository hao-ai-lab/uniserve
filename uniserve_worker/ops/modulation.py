"""Indexed affine modulation and gated residuals with explicit BF16 rounding.

RMS statistics and weighted normalization accumulate in FP32. The weighted
result, unit-offset scale, scaled result and shifted result round separately
through the activation dtype. Row indices select strided modulation tables.
FP8 outputs retain the BF16 result's per-row dequantization scale.
"""

from __future__ import annotations

import torch

from ..backends.triton import triton_available

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

__all__ = [
    "modulated_rms_norm",
    "gated_residual",
    "gated_residual_rms_norm",
    "gated_residual_rms_norm_fp8",
]

if triton is not None:
    from .silu import _fp8_divide_rn

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
        width: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Fuse RMS scaling with row-selected affine modulation at BF16 boundaries."""

        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < elements
        row = offsets // width
        column = offsets - row * width
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
        width: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Add a row-selected gated update using checkpoint BF16 rounding points."""

        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < elements
        row = offsets // width
        column = offsets - row * width
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
        width: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Modulate one hidden row and emit row-scaled E4M3 values."""

        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        mask = columns < width
        modulation_row = tl.load(row_indices_ptr + row)
        value = tl.load(
            value_ptr + row * width + columns,
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

        # Each row carries the dequantization scale consumed by FP8 GEMM.
        output_fp32 = tl.where(mask, output.to(tl.float32), 0.0)
        output_scale = tl.maximum(tl.max(tl.abs(output_fp32), axis=0), _SCALE_EPS_TL)
        output_scale /= _FP8_MAX_TL
        quantized = tl.maximum(
            tl.minimum(_fp8_divide_rn(output_fp32, output_scale), _FP8_MAX_TL),
            -_FP8_MAX_TL,
        )
        tl.store(output_ptr + row * width + columns, quantized, mask=mask)
        tl.store(output_scale_ptr + row, output_scale)


def _modulation_inputs_eligible(value: torch.Tensor, *operands: torch.Tensor) -> bool:
    """Accept contiguous BF16 values and co-located, contiguous feature axes."""

    return (
        triton is not None
        and value.is_cuda
        and value.dtype == torch.bfloat16
        and value.is_contiguous()
        and int(value.shape[-1]) > 0
        and triton_available(value.device)
        and all(operand.device == value.device and operand.stride(-1) == 1 for operand in operands)
    )


def _rmsnorm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply RMS normalization in FP32 and restore the operand dtype."""

    normalized = value.float() * torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + eps)
    return (normalized * weight.float()).to(value.dtype)


def modulated_rms_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    row_indices: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Normalize ``[rows, width]`` values and apply indexed affine vectors.

    ``row_indices`` contains one valid table row per value row. Weight is a
    width vector; scale and shift share a table shape and may have strided rows.
    All affine arithmetic rounds through the value dtype at the stated edges.
    """

    if _modulation_inputs_eligible(value, weight, shift, scale, row_indices):
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
            width=int(value.shape[-1]),
            BLOCK=1024,
            num_warps=4,
        )
        return output

    normalized = _rmsnorm(value, weight, eps)
    affine_scale = (1.0 + scale.index_select(0, row_indices).float()).to(value.dtype)
    scaled = (normalized * affine_scale).to(value.dtype)
    return (scaled.float() + shift.index_select(0, row_indices).float()).to(value.dtype)


def gated_residual(
    hidden: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor:
    """Add row-indexed gated updates to the residual stream.

    An eligible CUDA provider may reuse ``update`` as its output buffer; callers must
    treat that input as consumed after the call.
    """

    if update.is_contiguous() and _modulation_inputs_eligible(hidden, update, gate, row_indices):
        elements = hidden.numel()
        _gated_residual_kernel[(triton.cdiv(elements, 1024),)](
            hidden,
            update,
            gate,
            row_indices,
            update,
            gate.stride(0),
            elements,
            width=int(hidden.shape[-1]),
            BLOCK=1024,
            num_warps=4,
        )
        return update

    gated = (gate.index_select(0, row_indices).float() * update.float()).to(hidden.dtype)
    return hidden + gated


def gated_residual_rms_norm(
    hidden: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    row_indices: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the gated residual and its row-modulated normalization."""

    residual = gated_residual(hidden, update, gate, row_indices)
    normalized = modulated_rms_norm(
        residual,
        weight,
        shift,
        scale,
        row_indices,
        eps=eps,
    )
    return residual, normalized


def gated_residual_rms_norm_fp8(
    hidden: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    row_indices: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Produce a gated residual plus row-scaled FP8 modulated normalization."""

    residual = gated_residual(hidden, update, gate, row_indices)
    width = int(residual.shape[-1])
    if (
        not _modulation_inputs_eligible(residual, weight, shift, scale, row_indices)
        or width > 32768
    ):
        from ..nn.quant.fp8 import quantize_fp8_rowwise

        normalized = modulated_rms_norm(residual, weight, shift, scale, row_indices, eps=eps)
        values, scales = quantize_fp8_rowwise(normalized.reshape(-1, width))
        return residual, values.reshape(normalized.shape), scales
    inverse_rms = torch.rsqrt(residual.float().pow(2).mean(-1, keepdim=True) + eps)
    rows = residual.numel() // width
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
        width=width,
        BLOCK=triton.next_power_of_2(width),
        num_warps=16,
    )
    return residual, output, output_scale
