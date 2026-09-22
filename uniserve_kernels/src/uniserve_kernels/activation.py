"""Gated SiLU activations over packed or separate value and gate channels.

Bias, SiLU and multiplication accumulate in FP32. FP8 variants store one E4M3
dequantization scale per row; magnitude variants store one absolute maximum
per program for :func:`uniserve_kernels.reduction.absmax`.
"""

from __future__ import annotations

import torch

from uniserve_kernels.triton import launchable, tl, triton

_ACT_BLOCK = 1024
_MAX_SIGNED_INDEX = (1 << 31) - 1
#: FP8 rows reduce one complete row per program.
MAX_FP8_WIDTH = 32768
#: Programs that also reduce a magnitude cover this many elements.
ABSMAX_BLOCK = 32768


if triton is not None:
    _FP8_MAX_TL = tl.constexpr(448.0)
    _FP8_SCALE_EPS_TL = tl.constexpr(1.0e-12)

    @triton.jit
    def _silu_and_mul_kernel(
        x_ptr, out_ptr, n_cols: tl.constexpr, block: tl.constexpr
    ):
        """Apply ``silu(gate) * value`` to one packed activation row."""
        row = tl.program_id(0)
        cols = tl.program_id(1) * block + tl.arange(0, block)
        mask = cols < n_cols
        base = row * (n_cols * 2)

        # The gate occupies the first half of each row and its multiplicative
        # value occupies the second half.
        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(x_ptr + base + n_cols + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        silu = x / (1.0 + tl.exp(-x))
        out = silu * y
        tl.store(out_ptr + row * n_cols + cols, out, mask=mask)

    @triton.jit
    def _silu_and_mul_fp8_kernel(
        x_ptr,
        out_ptr,
        scale_ptr,
        n_cols: tl.constexpr,
        block: tl.constexpr,
    ):
        """Apply packed SwiGLU and emit its row-scaled E4M3 output."""
        row = tl.program_id(0)
        cols = tl.arange(0, block)
        mask = cols < n_cols
        base = row * (n_cols * 2)

        gate = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        value = tl.load(x_ptr + base + n_cols + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        activated = gate / (1.0 + tl.exp(-gate))
        output = activated * value

        # One E4M3 dequantization scale per row from the row's absolute max.
        output_fp32 = tl.where(mask, output.to(tl.float32), 0.0)
        scale = tl.maximum(
            tl.max(tl.abs(output_fp32), axis=0), _FP8_SCALE_EPS_TL
        )
        scale = scale / _FP8_MAX_TL
        quantized = tl.maximum(
            tl.minimum(output_fp32 / scale, _FP8_MAX_TL),
            -_FP8_MAX_TL,
        )
        tl.store(out_ptr + row * n_cols + cols, quantized, mask=mask)
        tl.store(scale_ptr + row, scale)


if triton is not None:

    @triton.jit
    def _value_first_swiglu_kernel(
        value_gate_ptr,
        bias_ptr,
        output_ptr,
        partials_ptr,
        elements,
        width: tl.constexpr,
        HAS_BIAS: tl.constexpr,  # noqa: N803
        RETURN_ABSMAX: tl.constexpr,  # noqa: N803
        BLOCK: tl.constexpr,  # noqa: N803
    ):
        """Evaluate value-first SwiGLU over a packed projection.

        Evaluate value-first SwiGLU over a packed ``[value, gate]``
        projection.
        """
        # Flattened projections can exceed int32 offsets, so index in int64.
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        offsets = offsets.to(tl.int64)
        mask = offsets < elements
        row = offsets // width
        column = offsets - row * width

        value = tl.load(
            value_gate_ptr + row * (2 * width) + column, mask=mask, other=0.0
        ).to(tl.float32)
        gate = tl.load(
            value_gate_ptr + row * (2 * width) + width + column,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if HAS_BIAS:
            value += tl.load(bias_ptr + column, mask=mask, other=0.0).to(
                tl.float32
            )
            gate += tl.load(bias_ptr + width + column, mask=mask, other=0.0).to(
                tl.float32
            )

        output = (value * gate / (1.0 + tl.exp(-gate))).to(
            output_ptr.dtype.element_ty
        )
        tl.store(output_ptr + offsets, output, mask=mask)

        if RETURN_ABSMAX:
            partial = tl.max(
                tl.where(mask, tl.abs(output.to(tl.float32)), 0.0), axis=0
            )
            tl.store(partials_ptr + tl.program_id(0), partial)

    @triton.jit
    def _swiglu_kernel(
        value_ptr,
        gate_ptr,
        value_bias_ptr,
        gate_bias_ptr,
        output_ptr,
        partials_ptr,
        elements,
        width: tl.constexpr,
        value_row_stride: tl.constexpr,
        gate_row_stride: tl.constexpr,
        HAS_VALUE_BIAS: tl.constexpr,  # noqa: N803
        HAS_GATE_BIAS: tl.constexpr,  # noqa: N803
        RETURN_ABSMAX: tl.constexpr,  # noqa: N803
        BLOCK: tl.constexpr,  # noqa: N803
    ):
        """Evaluate SwiGLU from separate value and gate channel views."""
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        offsets = offsets.to(tl.int64)
        mask = offsets < elements
        row = offsets // width
        column = offsets - row * width

        value = tl.load(
            value_ptr + row * value_row_stride + column,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.load(
            gate_ptr + row * gate_row_stride + column,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if HAS_VALUE_BIAS:
            value += tl.load(value_bias_ptr + column, mask=mask, other=0.0).to(
                tl.float32
            )
        if HAS_GATE_BIAS:
            gate += tl.load(gate_bias_ptr + column, mask=mask, other=0.0).to(
                tl.float32
            )

        output = (value * gate / (1.0 + tl.exp(-gate))).to(
            output_ptr.dtype.element_ty
        )
        tl.store(output_ptr + offsets, output, mask=mask)

        if RETURN_ABSMAX:
            partial = tl.max(
                tl.where(mask, tl.abs(output.to(tl.float32)), 0.0), axis=0
            )
            tl.store(partials_ptr + tl.program_id(0), partial)

    @triton.jit
    def _value_first_swiglu_fp8_kernel(
        value_gate_ptr,
        output_ptr,
        output_scale_ptr,
        width: tl.constexpr,
        BLOCK: tl.constexpr,  # noqa: N803
    ):
        """Evaluate one packed SwiGLU row with rowwise quantization.

        Evaluate one packed SwiGLU row and quantize it with a rowwise
        scale.
        """
        row = tl.program_id(0).to(tl.int64)
        columns = tl.arange(0, BLOCK)
        mask = columns < width
        value = tl.load(
            value_gate_ptr + row * (2 * width) + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.load(
            value_gate_ptr + row * (2 * width) + width + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        activated_gate = gate / (1.0 + tl.exp(-gate))
        output = value * activated_gate

        output_fp32 = tl.where(mask, output.to(tl.float32), 0.0)
        output_scale = tl.maximum(
            tl.max(tl.abs(output_fp32), axis=0), _FP8_SCALE_EPS_TL
        )
        output_scale /= _FP8_MAX_TL
        quantized = tl.maximum(
            tl.minimum(output_fp32 / output_scale, _FP8_MAX_TL),
            -_FP8_MAX_TL,
        )
        tl.store(output_ptr + row * width + columns, quantized, mask=mask)
        tl.store(output_scale_ptr + row, output_scale)


def can_run(x: torch.Tensor, *operands: torch.Tensor | None) -> bool:
    """Return whether contiguous CUDA inputs and operands fit the kernels."""
    return (
        launchable(x.device)
        and not torch.is_grad_enabled()
        and x.is_cuda
        and x.is_contiguous()
        and all(
            operand is None
            or (operand.device == x.device and operand.is_contiguous())
            for operand in operands
        )
    )


def silu_and_mul(x: torch.Tensor, out: torch.Tensor) -> None:
    """Store ``silu(gate) * value`` for packed ``[gate, value]`` rows."""
    width = int(x.shape[-1]) // 2
    rows = out.numel() // width
    x_rows, out_rows = x.view(rows, 2 * width), out.view(rows, width)

    # Chunk rows so flattened offsets stay within Triton's signed indexing.
    step = max(1, _MAX_SIGNED_INDEX // (2 * width))
    for start in range(0, rows, step):
        stop = min(rows, start + step)
        _silu_and_mul_kernel[(stop - start, triton.cdiv(width, _ACT_BLOCK))](
            x_rows[start:stop],
            out_rows[start:stop],
            width,
            _ACT_BLOCK,
            num_warps=4,
        )


def silu_and_mul_fp8(
    x: torch.Tensor, out: torch.Tensor, scale: torch.Tensor
) -> None:
    """Store row-scaled E4M3 packed SwiGLU values and ``[rows, 1]`` scales."""
    width = int(x.shape[-1]) // 2
    block = triton.next_power_of_2(width)
    _silu_and_mul_fp8_kernel[(x.numel() // (2 * width),)](
        x,
        out,
        scale,
        width,
        block,
        num_warps=32 if block >= 32_768 else 16,
    )


def value_first_swiglu(
    value_gate: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    partials: torch.Tensor | None = None,
) -> None:
    """Store ``value * silu(gate)`` for packed ``[value, gate]`` rows."""
    elements = out.numel()
    block = _ACT_BLOCK if partials is None else ABSMAX_BLOCK
    _value_first_swiglu_kernel[(triton.cdiv(elements, block),)](
        value_gate,
        bias,
        out,
        partials,
        elements=elements,
        width=int(out.shape[-1]),
        HAS_BIAS=bias is not None,
        RETURN_ABSMAX=partials is not None,
        BLOCK=block,
        num_warps=4 if partials is None else 8,
    )


def value_first_swiglu_fp8(
    value_gate: torch.Tensor, out: torch.Tensor, scale: torch.Tensor
) -> None:
    """Store row-scaled E4M3 value-first SwiGLU and ``[rows, 1]`` scales."""
    width = int(out.shape[-1])
    block = triton.next_power_of_2(width)
    _value_first_swiglu_fp8_kernel[(out.numel() // width,)](
        value_gate,
        out,
        scale,
        width=width,
        BLOCK=block,
        num_warps=32 if block >= 32768 else 16,
    )


def swiglu(
    value: torch.Tensor,
    gate: torch.Tensor,
    value_bias: torch.Tensor | None,
    gate_bias: torch.Tensor | None,
    out: torch.Tensor,
    partials: torch.Tensor | None = None,
) -> None:
    """Store ``(value + value_bias) * silu(gate + gate_bias)``.

    ``value`` and ``gate`` are separate ``[rows, width]`` views with unit
    channel stride; ``out`` is contiguous.
    """
    width = int(value.shape[-1])
    value_rows, gate_rows = value.reshape(-1, width), gate.reshape(-1, width)
    elements = out.numel()
    block = _ACT_BLOCK if partials is None else ABSMAX_BLOCK
    _swiglu_kernel[(triton.cdiv(elements, block),)](
        value_rows,
        gate_rows,
        value_bias,
        gate_bias,
        out,
        partials,
        elements=elements,
        width=width,
        value_row_stride=value_rows.stride(0),
        gate_row_stride=gate_rows.stride(0),
        HAS_VALUE_BIAS=value_bias is not None,
        HAS_GATE_BIAS=gate_bias is not None,
        RETURN_ABSMAX=partials is not None,
        BLOCK=block,
        num_warps=4 if partials is None else 8,
    )
