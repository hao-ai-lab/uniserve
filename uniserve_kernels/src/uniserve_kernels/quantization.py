"""Dynamic E4M3 encoding and magnitude statistics over floating rows.

``uniserve.quantization.quantizer.Quantizer`` runs these kernels for FP8
quantization of CUDA inputs, with one scale per row of a rank-2 matrix or
one scale for the whole tensor, and for the FP32 magnitude statistics
(``Quantizer.amax``) that dynamic scales and NVFP4 tensor scales derive
from. It raises when :func:`unsupported_rowwise_fp8` reports a reason. Off
CUDA, the quantizer takes its FP8 tensor-operation branch, which repeats the
scale floor and E4M3 bound as literals, so the constants here must stay in
agreement with it.

Every input is a ``[rows, width]`` matrix with unit column stride and any
row stride; outputs are contiguous.

:func:`nvfp4_encode` is a Triton device function, not a launch: kernels
that produce a row in registers call it to store the row directly in a
calibrated NVFP4 encoding, bit for bit as FlashInfer's NVFP4 encoder
would store it.
"""

from __future__ import annotations

import torch

from uniserve_kernels.triton import tl, triton, unsupported_operands

#: One program reduces a complete row, bounding the encoded width.
MAX_WIDTH = 32768
#: Programs of the tensor-magnitude reduction. The count is fixed, so the
#: partial buffer and its final reduction compile once for every row count.
ABSMAX_PROGRAMS = 1024


if triton is not None:
    # Largest finite ``float8_e4m3fn`` magnitude.
    _FP8_MAX_TL = tl.constexpr(448.0)
    # Floor on the row max-abs, so an all-zero row gets a nonzero scale.
    _FP8_SCALE_EPS_TL = tl.constexpr(1.0e-12)

    @triton.jit
    def _rowwise_fp8_quant_kernel(
        input_ptr,
        output_ptr,
        scale_ptr,
        amax_ptr,
        input_row_stride,
        amax_row_stride,
        width: tl.constexpr,
        HAS_AMAX: tl.constexpr,  # noqa: N803
        ROW_SCALES: tl.constexpr,  # noqa: N803
        BLOCK: tl.constexpr,  # noqa: N803
    ):
        """Quantize one floating activation row and publish its E4M3 scale.

        Input rows are addressed by ``input_row_stride`` with unit column
        stride; outputs are contiguous ``[rows, width]`` values. Without
        ``HAS_AMAX`` the scale derives from this row's absolute maximum;
        with it, from ``amax_ptr[row * amax_row_stride]`` (stride 0 reads
        one tensor-wide statistic). ``ROW_SCALES`` stores one scale per row;
        otherwise program zero stores the single tensor scale.
        """
        # Row bases widen to int64 once per program, so tensors past 2**31
        # elements address correctly while column offsets stay 32-bit.
        row = tl.program_id(0).to(tl.int64)
        source = input_ptr + row * input_row_stride
        destination = output_ptr + row * width
        columns = tl.arange(0, BLOCK)
        mask = columns < width
        values = tl.load(source + columns, mask=mask, other=0.0).to(tl.float32)

        if HAS_AMAX:
            maximum = tl.load(amax_ptr + row * amax_row_stride)
        else:
            maximum = tl.max(tl.abs(values), axis=0)
        # The one scale per program divides with correct rounding, so the
        # published scale equals ``Tensor.div``'s. Element quotients use the
        # fast full-range division (at most 2 ulp in FP32): correctly rounded
        # division per element makes the kernel compute-bound, and the E4M3
        # rounding that follows is coarser by 2^19.
        scale = tl.math.div_rn(
            tl.maximum(maximum, _FP8_SCALE_EPS_TL), _FP8_MAX_TL
        )
        quantized = tl.maximum(
            tl.minimum(values / scale, _FP8_MAX_TL),
            -_FP8_MAX_TL,
        )

        # The store converts the clamped FP32 quotient to the values dtype.
        tl.store(destination + columns, quantized, mask=mask)
        if ROW_SCALES:
            tl.store(scale_ptr + row, scale)
        elif row == 0:
            tl.store(scale_ptr, scale)

    @triton.jit
    def _row_absmax_kernel(
        input_ptr,
        output_ptr,
        input_row_stride,
        width: tl.constexpr,
        BLOCK: tl.constexpr,  # noqa: N803
    ):
        """Store the FP32 absolute maximum of one row to ``output_ptr[row]``."""
        row = tl.program_id(0).to(tl.int64)
        source = input_ptr + row * input_row_stride
        columns = tl.arange(0, BLOCK)
        values = tl.load(source + columns, mask=columns < width, other=0.0).to(
            tl.float32
        )
        tl.store(output_ptr + row, tl.max(tl.abs(values), axis=0))

    @triton.jit(do_not_specialize=["rows"])
    def _absmax_partials_kernel(
        input_ptr,
        partials_ptr,
        rows,
        input_row_stride,
        width: tl.constexpr,
        PROGRAMS: tl.constexpr,  # noqa: N803
        BLOCK: tl.constexpr,  # noqa: N803
    ):
        """Store the absolute maximum over rows ``program, program + PROGRAMS,
        ...`` to ``partials_ptr[program]``.

        Programs without rows store zero, the maximum of no magnitudes.
        """  # noqa: D205
        program = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        maximum = tl.zeros((BLOCK,), tl.float32)
        for row in range(program, rows, PROGRAMS):
            source = input_ptr + row.to(tl.int64) * input_row_stride
            values = tl.load(
                source + columns, mask=columns < width, other=0.0
            ).to(tl.float32)
            maximum = tl.maximum(maximum, tl.abs(values))
        tl.store(partials_ptr + program, tl.max(maximum, axis=0))

    @triton.jit
    def _rcp_approx(x):
        """``rcp.approx.ftz.f32``: the reciprocal the NVFP4 encoder uses."""
        return tl.inline_asm_elementwise(
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            [x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _e4m3_byte(x):
        """The saturating round-to-nearest E4M3 byte of FP32 ``x``.

        ``cvt.rn.satfinite.e4m3x2.f32`` stores its first operand in the
        upper byte; the lower byte encodes ``x``.
        """
        pair = tl.inline_asm_elementwise(
            "{ .reg .b16 pair; "
            "cvt.rn.satfinite.e4m3x2.f32 pair, $2, $1; "
            "cvt.u32.u16 $0, pair; }",
            "=r,f,f",
            [x, tl.zeros_like(x)],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        return (pair & 0xFF).to(tl.uint8)

    @triton.jit
    def _e2m1_pair(even, odd):
        """One byte of two saturating round-to-nearest E2M1 codes.

        The lower nibble holds ``even`` (the lower column), as FlashInfer's
        packed NVFP4 values store them.
        """
        pair = tl.inline_asm_elementwise(
            "{ .reg .b8 pair; .reg .b16 wide; "
            "cvt.rn.satfinite.e2m1x2.f32 pair, $2, $1; "
            "mov.b16 wide, {pair, 0}; cvt.u32.u16 $0, wide; }",
            "=r,f,f",
            [even, odd],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        return pair.to(tl.uint8)

    @triton.jit
    def nvfp4_encode(values, tensor_scale, BLOCK: tl.constexpr):  # noqa: N803
        """Encode one row of ``BLOCK`` FP32 values with a static tensor scale.

        ``values`` hold the 16-bit row values in FP32 (padding columns
        zero); ``tensor_scale`` is the calibrated NVFP4 activation scale
        ``s`` whose reciprocal is the encoder's global scale. Returns the
        ``[BLOCK // 2]`` packed E2M1 bytes and the ``[BLOCK // 16]`` E4M3
        block-scale bytes of the linear layout. The arithmetic is
        FlashInfer's ``fp4_quantize`` (TensorRT-LLM
        ``cvt_warp_fp16_to_fp4``, quantization_utils.cuh:431-690) step for
        step, so the bytes equal its output: each K16 block's amax times
        ``rcp.approx(6)`` times ``1 / s`` rounds to the E4M3 block scale;
        values multiply by ``rcp.approx(scale * rcp.approx(1 / s))`` (zero
        for an all-zero block) and round to E2M1, saturating.
        """
        blocks = tl.reshape(values, [BLOCK // 16, 16])
        amax = tl.max(tl.abs(blocks), axis=1)

        # The global scale is the correctly rounded FP32 1 / s, as the
        # reference encoders compute it on the device.
        global_scale = tl.math.div_rn(1.0, tensor_scale)
        sixth = _rcp_approx(tl.full([BLOCK // 16], 6.0, tl.float32))
        scale_bytes = _e4m3_byte(global_scale * (amax * sixth))
        decoded = scale_bytes.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        inverse_global = _rcp_approx(tl.zeros_like(amax) + global_scale)
        output_scale = tl.where(
            amax != 0.0, _rcp_approx(decoded * inverse_global), 0.0
        )

        scaled = blocks * output_scale[:, None]
        even, odd = tl.split(tl.reshape(scaled, [BLOCK // 2, 2]))
        return _e2m1_pair(even, odd), scale_bytes


def unsupported_rowwise_fp8(x: torch.Tensor) -> str | None:
    """Return why the kernels cannot read rows ``x``, or ``None``.

    ``x`` is a floating ``[rows, width]`` matrix with unit column stride and
    any row stride; one program reduces a complete row, bounding ``width``
    by :data:`MAX_WIDTH`.
    """
    reason = unsupported_operands(x)
    if reason is not None:
        return reason
    if x.ndim != 2:
        return "the kernels read a rank-2 matrix of rows"
    if x.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        return f"dtype {x.dtype} is not a floating dtype"
    if x.shape[1] > 1 and x.stride(1) != 1:
        return "the columns are not unit-strided"
    if x.shape[1] > MAX_WIDTH:
        return f"row width {x.shape[1]} exceeds the kernel's {MAX_WIDTH}"
    return None


def _block(width: int) -> int:
    """Return the power-of-two row tile; zero-width rows still reduce."""
    return triton.next_power_of_2(max(width, 1))


def rowwise_fp8(
    x: torch.Tensor,
    values: torch.Tensor,
    scale: torch.Tensor,
    amax: torch.Tensor | None = None,
) -> None:
    """Store E4M3 rows of ``x`` and their dequantization scales.

    A scale is ``max(amax, 1e-12) / 448`` and each value is
    ``clamp(row / scale, -448, 448)``. Without ``amax`` every row uses its
    own absolute maximum. A supplied FP32 ``amax`` holds one contiguous
    statistic per row, or one element for the whole tensor. ``scale`` is
    contiguous FP32 with one entry per row, or one element when ``amax`` is
    tensor-wide; ``values`` is contiguous ``float8_e4m3fn`` ``[rows,
    width]``. Neither output is checked here.
    """
    rows, width = x.shape
    if rows == 0:
        return
    per_tensor = amax is not None and amax.numel() == 1 and rows != 1
    _rowwise_fp8_quant_kernel[(rows,)](
        x,
        values,
        scale,
        scale if amax is None else amax,
        x.stride(0),
        0 if amax is None or per_tensor else 1,
        width=width,
        HAS_AMAX=amax is not None,
        ROW_SCALES=not per_tensor,
        BLOCK=_block(width),
        num_warps=8 if width <= 5120 else 16,
    )


def row_absmax(x: torch.Tensor, out: torch.Tensor) -> None:
    """Store the FP32 absolute maximum of each row into contiguous ``out``."""
    rows, width = x.shape
    if rows == 0:
        return
    _row_absmax_kernel[(rows,)](
        x,
        out,
        x.stride(0),
        width=width,
        BLOCK=_block(width),
        num_warps=8 if width <= 5120 else 16,
    )


def tensor_absmax(x: torch.Tensor, out: torch.Tensor) -> None:
    """Store the FP32 absolute maximum of every element into scalar ``out``.

    Empty matrices have maximum zero. Two launches: fixed-count per-program
    partials, then :func:`uniserve_kernels.reduction.absmax`.
    """
    from uniserve_kernels import reduction

    rows, width = x.shape
    partials = torch.empty(
        (ABSMAX_PROGRAMS,), dtype=torch.float32, device=x.device
    )
    _absmax_partials_kernel[(ABSMAX_PROGRAMS,)](
        x,
        partials,
        rows,
        x.stride(0),
        width=width,
        PROGRAMS=ABSMAX_PROGRAMS,
        BLOCK=_block(width),
        num_warps=4 if width <= 2048 else 8,
    )
    reduction.absmax(partials, out)
