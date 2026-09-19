"""Indexed RMS modulation and gated residuals.

Statistics and affine expressions accumulate in FP32. Providers may fuse
calls and choose their reduction order; outputs use the activation dtype
or carry a per-row E4M3 dequantization scale. Row indices select modulation
parameters whose leading stride may include other parameter groups.
"""

from __future__ import annotations

import torch

from uniserve.runtime.triton import triton_available

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

    @triton.jit
    def _modulated_rms_kernel(
        hidden_ptr,
        update_ptr,
        gate_ptr,
        weight_ptr,
        shift_ptr,
        scale_ptr,
        row_indices_ptr,
        output_ptr,
        output_scale_ptr,
        retain_ptr,
        gate_row_stride,
        shift_row_stride,
        scale_row_stride,
        WIDTH: tl.constexpr,  # noqa: N803
        BLOCK: tl.constexpr,  # noqa: N803
        EPS: tl.constexpr,  # noqa: N803
        HAS_UPDATE: tl.constexpr,  # noqa: N803
        FP8_OUTPUT: tl.constexpr,  # noqa: N803
        RETAIN: tl.constexpr,  # noqa: N803
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        mask = columns < WIDTH
        offsets = row * WIDTH + columns
        modulation_row = tl.load(row_indices_ptr + row)

        # value: one [WIDTH] activation row, accumulated in FP32.
        value = tl.load(hidden_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        if HAS_UPDATE:
            update = tl.load(update_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            gate = tl.load(
                gate_ptr + modulation_row * gate_row_stride + columns,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            value = value + gate * update
            tl.store(update_ptr + offsets, value, mask=mask)
        if RETAIN:
            # Keep the normalized row's source as the residual for the caller,
            # written in the same pass that reads it.
            tl.store(retain_ptr + offsets, value, mask=mask)

        mean_square = tl.sum(value * value, axis=0) / WIDTH
        weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(
            tl.float32
        )
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
        output = (
            value * tl.rsqrt(mean_square + EPS) * weight * (1.0 + scale) + shift
        )

        if FP8_OUTPUT:
            # Per-row E4M3 dequantization scale from the row's absolute max,
            # clamped away from zero so the division stays well-defined.
            output = tl.where(mask, output, 0.0)
            output_scale = (
                tl.maximum(tl.max(tl.abs(output), axis=0), 1.0e-12) / 448.0
            )
            output = tl.minimum(
                tl.maximum(output / output_scale, -448.0), 448.0
            )
            tl.store(output_scale_ptr + row, output_scale)

        tl.store(output_ptr + offsets, output, mask=mask)

    @triton.jit
    def _gated_residual_kernel(
        hidden_ptr,
        update_ptr,
        gate_ptr,
        row_indices_ptr,
        gate_row_stride,
        ELEMENTS: tl.constexpr,  # noqa: N803
        WIDTH: tl.constexpr,  # noqa: N803
        BLOCK: tl.constexpr,  # noqa: N803
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < ELEMENTS
        row = offsets // WIDTH
        columns = offsets % WIDTH
        modulation_row = tl.load(row_indices_ptr + row, mask=mask, other=0)
        hidden = tl.load(hidden_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        update = tl.load(update_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        gate = tl.load(
            gate_ptr + modulation_row * gate_row_stride + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        tl.store(update_ptr + offsets, hidden + gate * update, mask=mask)


def _modulation_inputs_eligible(
    value: torch.Tensor, *operands: torch.Tensor
) -> bool:
    return (
        triton is not None
        and value.is_cuda
        and value.dtype == torch.bfloat16
        and value.is_contiguous()
        and value.numel() > 0
        and triton_available(value.device)
        and all(
            operand.device == value.device and operand.stride(-1) == 1
            for operand in operands
        )
    )


def _modulate(
    value: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    row_indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Evaluate the real-valued modulation expression.

    Evaluate the real-valued expression with FP32 statistics and affine math.
    """
    value = value.float()
    inverse_rms = torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    return (
        value
        * inverse_rms
        * weight.float()
        * (1.0 + scale.index_select(0, row_indices).float())
        + shift.index_select(0, row_indices).float()
    )


def _fused_modulation(
    value: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    row_indices: torch.Tensor,
    eps: float,
    *,
    update: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    fp8: bool = False,
    retain: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Own output storage for the fused modulation.

    Own output storage for one row-wise fused reduction and affine
    expression.
    """
    width = int(value.shape[-1])
    rows = value.numel() // width
    output = torch.empty_like(
        value, dtype=torch.float8_e4m3fn if fp8 else value.dtype
    )
    output_scale = (
        torch.empty((rows, 1), dtype=torch.float32, device=value.device)
        if fp8
        else None
    )

    # Pointer slots disabled by the constexpr flags still need valid tensors;
    # reuse an existing buffer as a placeholder the kernel never dereferences.
    _modulated_rms_kernel[(rows,)](
        value,
        value if update is None else update,
        shift if gate is None else gate,
        weight,
        shift,
        scale,
        row_indices,
        output,
        output if output_scale is None else output_scale,
        output if retain is None else retain,
        0 if gate is None else gate.stride(0),
        shift.stride(0),
        scale.stride(0),
        WIDTH=width,
        BLOCK=triton.next_power_of_2(width),
        EPS=eps,
        HAS_UPDATE=update is not None,
        FP8_OUTPUT=fp8,
        RETAIN=retain is not None,
        num_warps=4 if width < 2048 else 8,
    )
    return output, output_scale


def modulated_rms_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    row_indices: torch.Tensor,
    *,
    eps: float,
    retain: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return RMS-normalized rows with their indexed scale and shift.

    ``retain`` receives a copy of ``value`` in the same pass, so a caller
    that keeps the residual while ``value`` lives in borrowed scratch does
    not read it twice.
    """
    if retain is not None and (
        retain.shape != value.shape
        or retain.dtype != value.dtype
        or not retain.is_contiguous()
    ):
        raise ValueError("retained rows must match the normalized value")
    if value.shape[-1] <= 32768 and _modulation_inputs_eligible(
        value, weight, shift, scale, row_indices
    ):
        return _fused_modulation(
            value, weight, shift, scale, row_indices, eps, retain=retain
        )[0]

    if retain is not None:
        retain.copy_(value)
    return _modulate(value, weight, shift, scale, row_indices, eps).to(
        value.dtype
    )


def gated_residual(
    hidden: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor:
    """Add indexed gated updates to hidden rows.

    Add indexed gated updates; an eligible provider consumes update storage.
    """
    if update.is_contiguous() and _modulation_inputs_eligible(
        hidden, update, gate, row_indices
    ):
        _gated_residual_kernel[(triton.cdiv(hidden.numel(), 1024),)](
            hidden,
            update,
            gate,
            row_indices,
            gate.stride(0),
            ELEMENTS=hidden.numel(),
            WIDTH=int(hidden.shape[-1]),
            BLOCK=1024,
            num_warps=4,
        )
        return update

    return (
        hidden.float()
        + gate.index_select(0, row_indices).float() * update.float()
    ).to(hidden.dtype)


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
    """Return a gated residual and its modulated normalization.

    The fused expression may retain the residual sum in FP32 for normalization.
    Update is consumed and may back the returned activation-dtype residual.
    """
    if (
        hidden.shape[-1] <= 32768
        and update.is_contiguous()
        and _modulation_inputs_eligible(
            hidden, update, gate, weight, shift, scale, row_indices
        )
    ):
        normalized, _ = _fused_modulation(
            hidden,
            weight,
            shift,
            scale,
            row_indices,
            eps,
            update=update,
            gate=gate,
        )
        return update, normalized

    residual = (
        hidden.float()
        + gate.index_select(0, row_indices).float() * update.float()
    )
    normalized = _modulate(residual, weight, shift, scale, row_indices, eps)
    return residual.to(hidden.dtype), normalized.to(hidden.dtype)


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
    """Return the residual and row-scaled E4M3 modulated normalization.

    Update is consumed and may back the returned residual. Scales describe the
    returned quantized values, independently of the provider's reduction order.
    """
    if (
        hidden.shape[-1] <= 32768
        and update.is_contiguous()
        and _modulation_inputs_eligible(
            hidden, update, gate, weight, shift, scale, row_indices
        )
    ):
        values, scales = _fused_modulation(
            hidden,
            weight,
            shift,
            scale,
            row_indices,
            eps,
            update=update,
            gate=gate,
            fp8=True,
        )
        assert scales is not None
        return update, values, scales

    from uniserve.quantization import Quantizer

    residual = (
        hidden.float()
        + gate.index_select(0, row_indices).float() * update.float()
    )
    normalized = _modulate(residual, weight, shift, scale, row_indices, eps)
    encoded = Quantizer("fp8", axis=0).quantize(
        normalized.reshape(-1, normalized.shape[-1])
    )
    buffers = encoded.buffers()
    return (
        residual.to(hidden.dtype),
        buffers["values"].reshape(normalized.shape),
        buffers["scale"],
    )
