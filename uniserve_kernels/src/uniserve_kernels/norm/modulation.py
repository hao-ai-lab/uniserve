"""Indexed RMS modulation and gated residuals over floating rows.

Row indices select modulation parameters whose leading stride may include
other parameter groups. Statistics and affine expressions accumulate in FP32;
outputs use the activation dtype or carry a per-row E4M3 dequantization scale.

``uniserve.nn.functional`` owns validation and raises on CUDA when
:func:`unsupported` reports a reason.
"""

from __future__ import annotations

import torch

from uniserve_kernels.triton import tl, triton, unsupported_operands

#: One program reduces a complete row, bounding the normalized width.
MAX_WIDTH = 32768
_FLOATING = (torch.float16, torch.bfloat16, torch.float32)


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
        # One program per activation row. ``BLOCK`` is ``WIDTH`` rounded up
        # to a power of two, and masked lanes load zeros so they add nothing
        # to the mean square. ``modulation_row`` selects this row's shift,
        # scale and gate from rows ``*_row_stride`` elements apart.
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
            # ``update`` receives the gated residual rounded to its dtype,
            # while the normalization below reads the unrounded FP32 sum.
            tl.store(update_ptr + offsets, value, mask=mask)
        if RETAIN:
            # Copy the normalized row's source (the gated sum when HAS_UPDATE)
            # for the caller in the same pass that reads it.
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
            # clamped away from zero so the division stays well-defined. 448
            # is the largest finite E4M3 magnitude.
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
        # Elementwise over the flat [rows, WIDTH] activation: each element
        # recovers its row to select the gate row, so ``MAX_WIDTH`` does not
        # apply. The result overwrites ``update``.
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


def unsupported(
    value: torch.Tensor,
    *operands: torch.Tensor,
    rows: tuple[torch.Tensor, ...] = (),
    normalizes: bool = True,
) -> str | None:
    """Return why the kernels cannot take these operands, or ``None``.

    ``value`` is the contiguous floating ``[..., width]`` activation.
    ``operands`` are indexed parameters (weight, shift, scale, gate) and row
    indices, which may be strided between rows but are unit-strided per
    channel. ``rows`` are further tensors addressed as full activation rows
    (``update``, ``retain``), which must be contiguous. A ``normalizes``
    launch reduces a complete row in one program, bounding ``width`` by
    :data:`MAX_WIDTH`.
    """
    reason = unsupported_operands(value, *operands, *rows)
    if reason is not None:
        return reason
    width = int(value.shape[-1]) if value.ndim else 0
    if value.dtype not in _FLOATING:
        return f"dtype {value.dtype} is not float16, bfloat16 or float32"
    if value.numel() == 0 or (normalizes and not 0 < width <= MAX_WIDTH):
        return f"rows are empty or wider than the kernel's {MAX_WIDTH}"
    if not value.is_contiguous() or any(
        not row.is_contiguous() for row in rows
    ):
        return "the activation or an updated row tensor is not contiguous"
    if any(operand.stride(-1) != 1 for operand in operands):
        return "a modulation operand is not unit-strided per channel"
    return None


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

    ``i`` is ``row_indices[row]``. ``value``, ``out``, ``update`` and
    ``retain`` are contiguous ``[..., width]`` rows with ``width`` at most
    ``MAX_WIDTH``; ``weight`` has ``width`` elements; ``shift``, ``scale``
    and ``gate`` are indexed along their leading axis with unit channel
    stride.

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
    """Store ``hidden + gate[i] * update`` into contiguous ``update``.

    ``i`` is ``row_indices[row]``; ``gate`` rows may be strided but are
    unit-strided per channel. ``hidden`` must be contiguous.
    """
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
