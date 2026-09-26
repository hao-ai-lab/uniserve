"""Row-wise RMS normalization and residual-add RMS normalization.

These Triton kernels back ``uniserve.nn.functional.rms_norm`` and
``add_rms_norm``, which check the weight, residual and outputs against ``x``,
allocate outputs the caller does not supply, and raise on CUDA whenever
:func:`unsupported` or :func:`unsupported_add` reports a reason. One program
normalizes one last-axis row with FP32 statistics and rounds once when
storing to the output dtype.
"""

from __future__ import annotations

import torch

from uniserve_kernels.triton import tl, triton, unsupported_operands

#: One program reduces a complete row, bounding the normalized width.
MAX_WIDTH = 8192
#: Leading axes a row address may span after merging contiguous axes, e.g.
#: ``[tokens, heads]`` rows of a merged projection view.
MAX_ROW_AXES = 4
# Power-of-two row blocks at least this wide launch with 8 warps, narrower
# blocks with 4 (see :func:`row_launch`).
_WIDE_BLOCK = 2048
_FLOATING = (torch.float16, torch.bfloat16, torch.float32)


if triton is not None:

    @triton.jit
    def _row_offset(row, shape: tl.constexpr, strides: tl.constexpr):
        """Return the element offset of flattened row ``row``.

        ``shape`` holds the extents of every leading axis after the first
        and ``strides`` the element strides of all leading axes, both as
        compile-time tuples; the first extent only bounds the launch.
        """
        offset = tl.full((), 0, tl.int64)
        remaining = row.to(tl.int64)
        for axis in tl.static_range(len(shape) - 1, -1, -1):
            offset += (remaining % shape[axis]) * strides[axis + 1]
            remaining //= shape[axis]
        return offset + remaining * strides[0]

    @triton.jit
    def _rms_norm_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        x_shape: tl.constexpr,
        x_strides: tl.constexpr,
        y_shape: tl.constexpr,
        y_strides: tl.constexpr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        block: tl.constexpr,
    ):
        """Normalize one hidden-state row per Triton program.

        Input and output rows are addressed through their own leading-axis
        strides with unit channel stride. Every program loads its whole row
        before storing, so the output may alias the input.
        """
        # ``n_cols``, ``eps`` and the row geometry are constexpr, so Triton
        # compiles one variant per distinct width, epsilon and view layout;
        # the leading extent (token count) is not part of the key.
        row = tl.program_id(0)
        offs = tl.arange(0, block)
        mask = offs < n_cols
        x_row = _row_offset(row, x_shape, x_strides)
        y_row = _row_offset(row, y_shape, y_strides)
        x = tl.load(x_ptr + x_row + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        # Variance and scaling stay in fp32; the output store performs the
        # conversion to the destination tensor's dtype.
        var = tl.sum(x * x, axis=0) / n_cols
        y = x * tl.rsqrt(var + eps) * w
        tl.store(y_ptr + y_row + offs, y, mask=mask)

    @triton.jit
    def _add_rms_norm_kernel(
        x_ptr,
        r_ptr,
        w_ptr,
        y_ptr,
        c_ptr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        block: tl.constexpr,
    ):
        """Add a residual, preserve the sum, and normalize it in one program."""
        # Rows are contiguous; int64 row bases keep offsets past 2**31 valid.
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, block)
        mask = offs < n_cols
        x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        r = tl.load(r_ptr + row * n_cols + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        c = x + r

        # Both outputs derive from the same fp32 sum: ``c_ptr`` carries the
        # residual stream and ``y_ptr`` carries its normalized projection.
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(c * c, axis=0) / n_cols
        y = c * tl.rsqrt(var + eps) * w
        tl.store(c_ptr + row * n_cols + offs, c, mask=mask)
        tl.store(y_ptr + row * n_cols + offs, y, mask=mask)


def row_axes(value: torch.Tensor) -> tuple[tuple, tuple] | None:
    """Return the merged leading-axis geometry of ``value``'s rows.

    Rows are the last axis. Adjacent leading axes merge when one steps over
    the other exactly (a contiguous prefix merges to a single row axis), and
    extent-one axes drop out. The result is ``(inner extents, strides)``:
    the extents of every merged leading axis after the first and the element
    stride of each merged axis. ``None`` means the channels are not
    unit-strided or more than :data:`MAX_ROW_AXES` axes remain.
    """
    if value.ndim == 0 or (value.shape[-1] > 1 and value.stride(-1) != 1):
        return None
    axes: list[list[int]] = []
    for extent, stride in zip(value.shape[:-1], value.stride()[:-1]):
        if extent == 1:
            continue
        if axes and axes[-1][1] == extent * stride:
            axes[-1] = [axes[-1][0] * extent, stride]
        else:
            axes.append([int(extent), int(stride)])
    if not axes:
        axes = [[1, int(value.shape[-1])]]
    if len(axes) > MAX_ROW_AXES:
        return None
    return (
        tuple(extent for extent, _ in axes[1:]),
        tuple(stride for _, stride in axes),
    )


def unsupported(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor
) -> str | None:
    """Return why ``rms_norm`` has no kernel for these operands, or ``None``.

    ``x`` and ``out`` are floating ``[..., width]`` rows with unit channel
    stride whose leading axes merge into at most :data:`MAX_ROW_AXES`
    strided axes (see :func:`row_axes`); ``out`` may alias ``x``. The weight
    is a contiguous ``[width]`` vector and ``width`` is at most
    :data:`MAX_WIDTH`.
    """
    reason = unsupported_operands(x, weight, out)
    if reason is not None:
        return reason
    width = int(x.shape[-1])
    if x.dtype not in _FLOATING or out.dtype not in _FLOATING:
        return f"dtypes {x.dtype} -> {out.dtype} are not floating types"
    if not 0 < width <= MAX_WIDTH:
        return f"row width {width} is outside the kernel's 1..{MAX_WIDTH}"
    if weight.shape != (width,) or not weight.is_contiguous():
        return "the weight is not one contiguous vector over the row width"
    if out.shape != x.shape:
        return "the output shape differs from the input"
    if row_axes(x) is None or row_axes(out) is None:
        return (
            "input or output channels are not unit-strided or their rows "
            f"span more than {MAX_ROW_AXES} strided leading axes"
        )
    return None


def unsupported_add(
    x: torch.Tensor, weight: torch.Tensor, *rows: torch.Tensor
) -> str | None:
    """Return why ``add_rms_norm`` has no kernel for these operands.

    ``rows`` lists the residual input and both outputs; with ``x`` they are
    contiguous floating rows of one shape, and the weight is a contiguous
    ``[width]`` vector of at most :data:`MAX_WIDTH` elements. Outputs may
    alias the inputs: each program loads its whole row before storing.
    """
    reason = unsupported_operands(x, weight, *rows)
    if reason is not None:
        return reason
    width = int(x.shape[-1])
    if any(value.dtype not in _FLOATING for value in (x, *rows)):
        return "row dtypes are not all floating types"
    if not 0 < width <= MAX_WIDTH:
        return f"row width {width} is outside the kernel's 1..{MAX_WIDTH}"
    if weight.shape != (width,) or not weight.is_contiguous():
        return "the weight is not one contiguous vector over the row width"
    if any(
        value.shape != x.shape or not value.is_contiguous()
        for value in (x, *rows)
    ):
        return (
            "input, residual and outputs are not contiguous rows of one shape"
        )
    return None


def row_launch(width: int) -> tuple[int, int]:
    """Return the power-of-two row block covering ``width`` and its warps."""
    block = triton.next_power_of_2(width)
    return block, 8 if block >= _WIDE_BLOCK else 4


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, out: torch.Tensor
) -> None:
    """Store ``x * rsqrt(mean(x^2) + eps) * weight`` into ``out``."""
    width = int(x.shape[-1])
    rows = x.numel() // width
    if rows == 0:
        return
    block, warps = row_launch(width)
    x_shape, x_strides = row_axes(x)
    out_shape, out_strides = row_axes(out)
    _rms_norm_kernel[(rows,)](
        x,
        weight,
        out,
        x_shape,
        x_strides,
        out_shape,
        out_strides,
        width,
        float(eps),
        block,
        num_warps=warps,
    )


def add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    summed: torch.Tensor,
) -> None:
    """Store the residual sum and its RMS normalization.

    Both outputs derive from the unrounded FP32 sum ``x + residual``.
    """
    width = int(x.shape[-1])
    rows = x.numel() // width
    if rows == 0:
        return
    block, warps = row_launch(width)
    _add_rms_norm_kernel[(rows,)](
        x,
        residual,
        weight,
        out,
        summed,
        width,
        float(eps),
        block,
        num_warps=warps,
    )
