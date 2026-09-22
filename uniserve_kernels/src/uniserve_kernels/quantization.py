"""Row-statistic E4M3 encoding over caller-supplied outputs."""

from __future__ import annotations

import torch

from uniserve_kernels.triton import launchable, tl, triton

#: One program reduces a complete row, bounding the encoded width.
MAX_WIDTH = 32768


if triton is not None:
    _FP8_MAX_TL = tl.constexpr(448.0)
    _FP8_SCALE_EPS_TL = tl.constexpr(1.0e-12)

    @triton.jit
    def _rowwise_fp8_quant_kernel(
        input_ptr,
        output_ptr,
        scale_ptr,
        input_row_stride,
        width: tl.constexpr,
        BLOCK: tl.constexpr,  # noqa: N803
    ):
        """Quantize one BF16 activation row and publish its E4M3 scale."""
        # One program per row; each program publishes one max-abs E4M3 scale.
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        mask = columns < width
        values = tl.load(
            input_ptr + row * input_row_stride + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        maximum = tl.maximum(tl.max(tl.abs(values), axis=0), _FP8_SCALE_EPS_TL)
        scale = maximum / _FP8_MAX_TL
        quantized = tl.maximum(
            tl.minimum(values / scale, _FP8_MAX_TL),
            -_FP8_MAX_TL,
        )

        tl.store(output_ptr + row * width + columns, quantized, mask=mask)
        tl.store(scale_ptr + row, scale)


def can_run_rowwise_fp8(x: torch.Tensor) -> bool:
    """Return whether unit-strided BF16/FP16 CUDA rows fit the kernel."""
    return (
        launchable(x.device)
        and x.is_cuda
        and x.ndim == 2
        and x.dtype in {torch.float16, torch.bfloat16}
        and x.stride(1) == 1
        and 0 < x.shape[1] <= MAX_WIDTH
        and x.shape[0] > 0
    )


def rowwise_fp8(
    x: torch.Tensor, values: torch.Tensor, scale: torch.Tensor
) -> None:
    """Store E4M3 rows of ``x`` and one max-abs ``[rows, 1]`` scale per row."""
    width = x.shape[1]
    _rowwise_fp8_quant_kernel[(x.shape[0],)](
        x,
        values,
        scale,
        x.stride(0),
        width=width,
        BLOCK=triton.next_power_of_2(width),
        num_warps=8 if width <= 5120 else 16,
    )
