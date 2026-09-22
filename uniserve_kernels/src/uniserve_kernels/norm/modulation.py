"""Indexed RMS modulation and gated residuals over BF16 rows.

Row indices select modulation parameters whose leading stride may include
other parameter groups. Statistics and affine expressions accumulate in FP32;
outputs use the activation dtype or carry a per-row E4M3 dequantization scale.
"""

from __future__ import annotations

import torch

from uniserve_kernels.triton import launchable, tl, triton

#: One program reduces a complete row, bounding the normalized width.
MAX_WIDTH = 32768


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


def can_run(value: torch.Tensor, *operands: torch.Tensor) -> bool:
    """Return whether contiguous BF16 CUDA rows and indexed operands fit.

    Operands may be strided between rows but are unit-strided per channel.
    """
    return (
        launchable(value.device)
        and value.is_cuda
        and value.dtype == torch.bfloat16
        and value.is_contiguous()
        and value.numel() > 0
        and all(
            operand.device == value.device and operand.stride(-1) == 1
            for operand in operands
        )
    )


def modulated_rms_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    row_indices: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    *,
    update: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
    retain: torch.Tensor | None = None,
) -> None:
    """Store ``rms(value) * weight * (1 + scale[i]) + shift[i]`` per row.

    With ``update`` and ``gate``, ``value + gate[i] * update`` is normalized
    and also stored into ``update``. ``output_scale`` selects E4M3 output with
    one ``[rows, 1]`` scale. ``retain`` receives the normalized source rows.
    """
    width = int(value.shape[-1])
    rows = value.numel() // width

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
        out,
        out if output_scale is None else output_scale,
        out if retain is None else retain,
        0 if gate is None else gate.stride(0),
        shift.stride(0),
        scale.stride(0),
        WIDTH=width,
        BLOCK=triton.next_power_of_2(width),
        EPS=eps,
        HAS_UPDATE=update is not None,
        FP8_OUTPUT=output_scale is not None,
        RETAIN=retain is not None,
        num_warps=4 if width < 2048 else 8,
    )


def gated_residual(
    hidden: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    row_indices: torch.Tensor,
) -> None:
    """Store ``hidden + gate[i] * update`` into contiguous ``update``."""
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
