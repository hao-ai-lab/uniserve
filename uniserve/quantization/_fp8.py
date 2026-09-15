"""Fused row-statistic FP8 encoding over borrowed output buffers."""

import torch

from uniserve.runtime.triton import triton_available

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


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
        BLOCK: tl.constexpr,
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


def rowwise(x: torch.Tensor, values: torch.Tensor, scale: torch.Tensor) -> bool:
    """Encode eligible BF16/FP16 rows using the existing single-kernel formula."""

    if not (
        triton is not None
        and x.is_cuda
        and x.ndim == 2
        and x.dtype in {torch.float16, torch.bfloat16}
        and x.stride(1) == 1
        and 0 < x.shape[1] <= 32768
        and x.shape[0] > 0
        and triton_available(x.device)
    ):
        return False

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
    return True
