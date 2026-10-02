"""Gated activations over packed or separate value and gate channels.

Bias, activation and multiplication accumulate in FP32. FP8 variants store
one E4M3 dequantization scale per row; magnitude variants store one absolute
maximum per program for :func:`uniserve_kernels.reduction.absmax`.

Launchers write caller-supplied outputs and validate nothing. The callers in
``uniserve.nn.functional`` run :func:`unsupported` first, size the outputs,
and raise on CUDA when it reports a reason; tensor operations evaluate the
same formulas only off CUDA.

Every input is addressed as rows with unit channel stride: packed inputs
``[..., 2 * width]`` whose leading axes flatten to rows of one stride (see
:func:`row_stride`), separate SwiGLU views likewise, and contiguous outputs.

:func:`softcap` bounds contiguous values to ``(-cap, cap)`` as
``tanh(x * (1 / cap)) * cap`` in one pass, with PyTorch's CUDA arithmetic.
"""

from __future__ import annotations

import numpy as np
import torch

from uniserve_kernels.triton import (
    dependent_launch,
    pdl_prologue,
    tl,
    triton,
    unsupported_operands,
)

#: Gate activations of the packed ``act_and_mul`` kernels: ``silu``, exact
#: (erf) ``gelu`` and ``gelu_tanh``, PyTorch's ``approximate="tanh"`` form.
ACTIVATIONS = ("silu", "gelu", "gelu_tanh")
#: Elements per program for the kernels that neither quantize per row nor
#: report absmax partials.
_ACT_BLOCK = 1024
#: Output elements per program and warps of the packed ``act_and_mul``
#: kernel. Programs tile the flattened output, so no row width leaves a
#: mostly masked tail program; 16 elements per thread keep enough loads in
#: flight to stream HBM (measured on SM100 against the registered copy line).
_GATED_BLOCK = 4096
_GATED_WARPS = 8
#: Widest row an FP8 kernel accepts: one program loads and reduces a complete
#: row to find its scale.
MAX_FP8_WIDTH = 32768
#: Programs that also reduce a magnitude cover this many elements; callers
#: size ``partials`` as ``ceil(elements / ABSMAX_BLOCK)`` FP32 entries.
ABSMAX_BLOCK = 32768
#: Elements per program and warps of the softcap kernel, which streams a
#: logits tensor once (the gated kernel's streaming shape).
_SOFTCAP_BLOCK = 4096
_SOFTCAP_WARPS = 8

_FLOATING = (torch.float16, torch.bfloat16, torch.float32)


if triton is not None:
    from triton.language.extra.cuda import libdevice

    # Largest finite float8_e4m3fn magnitude. The epsilon keeps the scale of
    # an all-zero row positive so the FP8 kernels' division stays finite.
    _FP8_MAX_TL = tl.constexpr(448.0)
    _FP8_SCALE_EPS_TL = tl.constexpr(1.0e-12)
    # sqrt(8 / pi) and sqrt(1 / 2): constants of the two GELU forms.
    _SQRT_8_OVER_PI = tl.constexpr(1.5957691216057308)
    _SQRT_HALF = tl.constexpr(0.7071067811865476)

    @triton.jit
    def _activate(gate, ACTIVATION: tl.constexpr):  # noqa: N803
        """Apply the named gate activation to an FP32 tile."""
        if ACTIVATION == "silu":
            return gate / (1.0 + tl.exp(-gate))
        elif ACTIVATION == "gelu_tanh":
            # 0.5 * x * (1 + tanh(u)) equals x * sigmoid(2u) for
            # u = sqrt(2 / pi) * (x + 0.044715 x^3); the sigmoid form needs
            # one exponential and saturates to exact 0 and x at both ends.
            inner = _SQRT_8_OVER_PI * (gate + 0.044715 * gate * gate * gate)
            return gate / (1.0 + tl.exp(-inner))
        else:
            return 0.5 * gate * (1.0 + libdevice.erf(gate * _SQRT_HALF))

    @triton.jit
    def _act_and_mul_kernel(
        x_ptr,
        out_ptr,
        elements,
        x_row_stride,
        n_cols: tl.constexpr,
        ACTIVATION: tl.constexpr,  # noqa: N803
        BLOCK: tl.constexpr,  # noqa: N803
        WIDE: tl.constexpr,  # noqa: N803
        PDL: tl.constexpr,  # noqa: N803
    ):
        """Store ``act(gate) * value`` for one block of the flattened output.

        Grid: ``ceil(elements / BLOCK)`` programs over the contiguous
        ``[rows, n_cols]`` output. Input rows start ``x_row_stride``
        elements apart. ``WIDE`` computes offsets in int64, which launches
        whose input or output spans ``2**31`` elements require. ``PDL``
        launches run :func:`pdl_prologue` before the first load.
        """
        pdl_prologue(PDL)

        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        if WIDE:
            offsets = offsets.to(tl.int64)
        mask = offsets < elements
        row = offsets // n_cols
        column = offsets - row * n_cols
        base = row * x_row_stride + column

        # The gate occupies the first half of each row and its multiplicative
        # value occupies the second half.
        gate = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
        value = tl.load(x_ptr + base + n_cols, mask=mask, other=0.0).to(
            tl.float32
        )
        out = _activate(gate, ACTIVATION) * value
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _act_and_mul_fp8_kernel(
        x_ptr,
        out_ptr,
        scale_ptr,
        x_row_stride,
        n_cols: tl.constexpr,
        ACTIVATION: tl.constexpr,  # noqa: N803
        block: tl.constexpr,
    ):
        """Apply packed gating and emit its row-scaled E4M3 output.

        One program owns one complete ``[gate, value]`` row, with ``block``
        at least ``n_cols``, so the row maximum is a single block reduction.
        Row offsets are int64, so one launch covers every row of a packed
        input past ``2**31`` elements.
        """
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, block)
        mask = cols < n_cols
        base = row * x_row_stride

        gate = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        value = tl.load(x_ptr + base + n_cols + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        output = _activate(gate, ACTIVATION) * value

        # One E4M3 dequantization scale per row from the absolute max of the
        # unrounded FP32 products; stored codes decode as ``code * scale``.
        output_fp32 = tl.where(mask, output, 0.0)
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

    @triton.jit
    def _value_first_swiglu_kernel(
        value_gate_ptr,
        bias_ptr,
        output_ptr,
        partials_ptr,
        elements,
        input_row_stride,
        width: tl.constexpr,
        HAS_BIAS: tl.constexpr,  # noqa: N803
        RETURN_ABSMAX: tl.constexpr,  # noqa: N803
        BLOCK: tl.constexpr,  # noqa: N803
    ):
        """Evaluate ``value * silu(gate)`` over packed ``[value, gate]`` rows.

        Programs tile the flattened contiguous ``[rows, width]`` output in
        ``BLOCK`` elements; input rows start ``input_row_stride`` elements
        apart and the optional bias is packed ``[value_bias, gate_bias]``
        like the input. With ``RETURN_ABSMAX``, each program also stores the
        absolute maximum of its outputs, after rounding to the output dtype,
        to ``partials_ptr[program]``.
        """
        # Flattened projections can exceed int32 offsets, so index in int64.
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        offsets = offsets.to(tl.int64)
        mask = offsets < elements
        row = offsets // width
        column = offsets - row * width

        value = tl.load(
            value_gate_ptr + row * input_row_stride + column,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.load(
            value_gate_ptr + row * input_row_stride + width + column,
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
        """Evaluate SwiGLU from separate value and gate channel views.

        Each view has unit channel stride and its own row stride; the output
        is contiguous. Programs tile the output and report magnitude partials
        exactly as :func:`_value_first_swiglu_kernel` does.
        """
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
        input_row_stride,
        width: tl.constexpr,
        BLOCK: tl.constexpr,  # noqa: N803
    ):
        """Evaluate one packed ``[value, gate]`` row and quantize it to E4M3.

        One program owns one complete row (``BLOCK >= width``) and stores its
        dequantization scale to ``output_scale_ptr[row]``.
        """
        row = tl.program_id(0).to(tl.int64)
        columns = tl.arange(0, BLOCK)
        mask = columns < width
        value = tl.load(
            value_gate_ptr + row * input_row_stride + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.load(
            value_gate_ptr + row * input_row_stride + width + columns,
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


if triton is not None:

    @triton.jit
    def _softcap_kernel(
        input_ptr,
        output_ptr,
        elements,
        inverse,
        cap,
        BLOCK: tl.constexpr,  # noqa: N803
    ):
        """Store ``tanh(x * inverse) * cap`` of contiguous elements.

        Every step is one FP32 operation PyTorch's CUDA kernels apply to
        ``torch.tanh(x.float() / cap) * cap``: the division by a scalar
        multiplies by its FP32 reciprocal ``inverse``, ``tanh`` is libdevice
        ``tanhf`` without flushing subnormals, and the product by ``cap``
        rounds once, so the output is bit-identical to that expression.
        """
        program = tl.program_id(0).to(tl.int64)
        offsets = program * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < elements
        values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        capped = libdevice.tanh(values * inverse) * cap
        tl.store(
            output_ptr + offsets,
            capped.to(output_ptr.dtype.element_ty),
            mask=mask,
        )


def unsupported_softcap(x: torch.Tensor, out: torch.Tensor) -> str | None:
    """Return why :func:`softcap` cannot map ``x`` to ``out``, or ``None``.

    Both are contiguous floating tensors of one shape on one CUDA device.
    """
    reason = unsupported_operands(x, out)
    if reason is not None:
        return reason
    if x.dtype not in _FLOATING or out.dtype not in _FLOATING:
        return "softcap reads and writes float16, bfloat16 or float32"
    if x.shape != out.shape:
        return "the softcap output does not match the input shape"
    if not (x.is_contiguous() and out.is_contiguous()):
        return "softcap operands are not contiguous"
    return None


def softcap(x: torch.Tensor, cap: float, out: torch.Tensor) -> None:
    """Store ``tanh(x.float() / cap) * cap`` into ``out`` in its dtype.

    ``cap`` is a positive finite number; the division uses its FP32
    reciprocal, as PyTorch divides a CUDA tensor by a scalar. Callers first
    check :func:`unsupported_softcap`.
    """
    elements = x.numel()
    if elements == 0:
        return
    # The FP32 reciprocal PyTorch computes for a scalar divisor, and the
    # FP32 multiplier it applies for ``* cap``.
    inverse = float(np.float32(1.0) / np.float32(cap))
    _softcap_kernel[(triton.cdiv(elements, _SOFTCAP_BLOCK),)](
        x,
        out,
        elements,
        inverse,
        float(np.float32(cap)),
        _SOFTCAP_BLOCK,
        num_warps=_SOFTCAP_WARPS,
        # PyTorch's tanhf keeps subnormal arguments; libdevice's flushing
        # variant would return zero for |x / cap| below FP32's normal range.
        enable_reflect_ftz=False,
    )


def row_stride(x: torch.Tensor) -> int | None:
    """Return the element stride between the flattened rows of ``x``.

    Rows are the last axis. ``None`` means the channels are not unit-strided
    or the leading axes do not flatten to rows of one common stride, which
    the kernels require to address a row as ``row * stride``.
    """
    if x.ndim == 0 or x.stride(-1) != 1:
        return None
    try:
        return int(x.view(-1, x.shape[-1]).stride(0))
    except RuntimeError:
        return None


def unsupported(
    x: torch.Tensor, *operands: torch.Tensor | None, fp8_width: int = 0
) -> str | None:
    """Return why the kernels cannot evaluate packed rows ``x``, or ``None``.

    ``x`` is a floating tensor whose rows (see :func:`row_stride`) carry the
    gating channels. ``operands`` are the outputs and biases the launch
    touches (``None`` for an absent bias); each must be contiguous.
    ``fp8_width`` is the per-row output width of an FP8 launch, which one
    program reduces completely and is bounded by :data:`MAX_FP8_WIDTH`.
    """
    reason = unsupported_operands(x, *operands)
    if reason is not None:
        return reason
    if x.dtype not in _FLOATING:
        return f"input dtype {x.dtype} is not float16, bfloat16 or float32"
    if row_stride(x) is None:
        return (
            "input channels are not unit-strided or its leading axes do not "
            "flatten to rows of one stride"
        )
    if any(
        operand is not None and not operand.is_contiguous()
        for operand in operands
    ):
        return "an output or bias operand is not contiguous"
    if fp8_width > MAX_FP8_WIDTH:
        return (
            f"FP8 row width {fp8_width} exceeds the single-program row "
            f"reduction limit {MAX_FP8_WIDTH}"
        )
    return None


def act_and_mul(x: torch.Tensor, out: torch.Tensor, *, activation: str) -> None:
    """Store ``activation(gate) * value`` for packed ``[gate, value]`` rows.

    ``x`` is ``[..., 2 * width]`` with rows of one stride and ``out`` is
    contiguous ``[..., width]``; ``out`` receives the result in its own
    dtype. ``activation`` is one of :data:`ACTIVATIONS`.
    """
    width = int(x.shape[-1]) // 2
    rows = out.numel() // width
    if rows == 0:
        return
    stride = row_stride(x)
    elements = rows * width
    # The last program's offsets reach elements + BLOCK; input offsets reach
    # rows * stride (packed rows span at least 2 * width).
    span = max(rows * max(stride, 2 * width), elements) + _GATED_BLOCK
    pdl = dependent_launch(x.device)
    _act_and_mul_kernel[(triton.cdiv(elements, _GATED_BLOCK),)](
        x,
        out,
        elements,
        stride,
        width,
        activation,
        _GATED_BLOCK,
        span >= 2**31,
        pdl,
        num_warps=_GATED_WARPS,
        launch_pdl=pdl,
    )


def act_and_mul_fp8(
    x: torch.Tensor, out: torch.Tensor, scale: torch.Tensor, *, activation: str
) -> None:
    """Store row-scaled E4M3 packed gating values and ``[rows, 1]`` scales.

    ``x`` packs ``[gate, value]`` rows like :func:`act_and_mul`. ``out`` is
    contiguous float8_e4m3fn ``[rows, width]`` and ``scale`` is FP32; one
    program handles each complete row, so ``width`` is at most
    :data:`MAX_FP8_WIDTH`.
    """
    width = int(x.shape[-1]) // 2
    rows = out.numel() // width
    if rows == 0:
        return
    block = triton.next_power_of_2(width)
    _act_and_mul_fp8_kernel[(rows,)](
        x,
        out,
        scale,
        row_stride(x),
        width,
        activation,
        block,
        num_warps=32 if block >= 32_768 else 16,
    )


def value_first_swiglu(
    value_gate: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    partials: torch.Tensor | None = None,
) -> None:
    """Store ``value * silu(gate)`` for packed ``[value, gate]`` rows.

    ``bias`` is ``None`` or contiguous ``[2 * width]``. When ``partials`` is
    given it must hold ``ceil(out.numel() / ABSMAX_BLOCK)`` FP32 entries,
    which :func:`uniserve_kernels.reduction.absmax` then reduces.
    """
    elements = out.numel()
    if elements == 0:
        return
    block = _ACT_BLOCK if partials is None else ABSMAX_BLOCK
    _value_first_swiglu_kernel[(triton.cdiv(elements, block),)](
        value_gate,
        bias,
        out,
        partials,
        elements=elements,
        input_row_stride=row_stride(value_gate),
        width=int(out.shape[-1]),
        HAS_BIAS=bias is not None,
        RETURN_ABSMAX=partials is not None,
        BLOCK=block,
        num_warps=4 if partials is None else 8,
    )


def value_first_swiglu_fp8(
    value_gate: torch.Tensor, out: torch.Tensor, scale: torch.Tensor
) -> None:
    """Store row-scaled E4M3 value-first SwiGLU and ``[rows, 1]`` scales.

    ``value_gate`` holds packed ``[value, gate]`` rows of one stride, ``out``
    is contiguous float8_e4m3fn ``[..., width]`` and ``scale`` is FP32. One
    program loads each complete row, so callers keep ``width`` within
    :data:`MAX_FP8_WIDTH`.
    """
    width = int(out.shape[-1])
    rows = out.numel() // width
    if rows == 0:
        return
    block = triton.next_power_of_2(width)
    _value_first_swiglu_fp8_kernel[(rows,)](
        value_gate,
        out,
        scale,
        row_stride(value_gate),
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

    ``value`` and ``gate`` are separate ``[..., width]`` views with unit
    channel stride whose leading axes flatten to rows; ``out`` is contiguous.
    ``partials`` follows the same contract as in :func:`value_first_swiglu`.
    """
    width = int(value.shape[-1])
    value_rows, gate_rows = value.reshape(-1, width), gate.reshape(-1, width)
    elements = out.numel()
    if elements == 0:
        return
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
