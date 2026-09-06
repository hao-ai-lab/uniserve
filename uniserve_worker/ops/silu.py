"""Provider dispatch for packed SiLU-gated multiplication.

Inputs store gate and value halves contiguously along the final dimension.
Providers preserve all leading dimensions and return a tensor with half the
input width.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

import torch
import torch.nn.functional as F

from ..backends.triton import triton_available
from .core import Dispatcher, Operator
from .requests import SiluAndMulReq

try:  # pragma: no cover - availability depends on the serving environment.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

_TRITON_ACT_BLOCK = 1024
_TRITON_MAX_SIGNED_INDEX = (1 << 31) - 1
_SGL_ALIGNMENT_BYTES = 16


@lru_cache(maxsize=1)
def _sgl_silu_and_mul_kernel():
    """Resolve the optional SGL fused activation once per process."""

    try:  # pragma: no cover - depends on the installed accelerator stack.
        from sgl_kernel import silu_and_mul
    except Exception:
        return None
    return silu_and_mul


def _triton_act_row_chunks(rows: int, n_cols: int) -> Iterator[tuple[int, int]]:
    """Yield row ranges whose flattened offsets fit Triton's signed index width."""

    rows = int(rows)
    n_cols = int(n_cols)
    if rows < 0 or n_cols <= 0:
        raise ValueError("activation row geometry must be non-negative with positive columns")

    # Each input row contains two ``n_cols`` halves, so the launch range is
    # bounded by twice the output width.
    rows_per_launch = max(1, _TRITON_MAX_SIGNED_INDEX // (2 * n_cols))
    for start in range(0, rows, rows_per_launch):
        yield start, min(rows, start + rows_per_launch)


if triton is not None:
    _FP8_MAX_TL = tl.constexpr(448.0)
    _FP8_SCALE_EPS_TL = tl.constexpr(1.0e-12)

    @triton.jit
    def _fp8_divide_rn(dividend, divisor):
        """Divide FP32 operands with explicit nearest-even PTX semantics."""

        return tl.inline_asm_elementwise(
            asm="div.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[dividend, divisor],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )


    @triton.jit
    def _silu_and_mul_kernel(x_ptr, out_ptr, n_cols: tl.constexpr, block: tl.constexpr):
        """Apply ``silu(gate) * value`` to one packed activation row."""

        row = tl.program_id(0)
        cols = tl.program_id(1) * block + tl.arange(0, block)
        mask = cols < n_cols
        base = row * (n_cols * 2)

        # The gate occupies the first half of each row and its multiplicative
        # value occupies the second half.
        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(x_ptr + base + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
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
        """Apply packed SwiGLU and emit its BF16-rounded row-scaled E4M3 output."""

        row = tl.program_id(0)
        cols = tl.arange(0, block)
        mask = cols < n_cols
        base = row * (n_cols * 2)
        gate = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        value = tl.load(x_ptr + base + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        activated = gate / (1.0 + tl.exp(-gate))
        output = (activated * value).to(tl.bfloat16)
        output_fp32 = tl.where(mask, output.to(tl.float32), 0.0)
        scale = tl.maximum(tl.max(tl.abs(output_fp32), axis=0), _FP8_SCALE_EPS_TL)
        scale = scale / _FP8_MAX_TL
        quantized = tl.maximum(
            tl.minimum(_fp8_divide_rn(output_fp32, scale), _FP8_MAX_TL),
            -_FP8_MAX_TL,
        )
        tl.store(out_ptr + row * n_cols + cols, quantized, mask=mask)
        tl.store(scale_ptr + row, scale)


def _act_inputs_eligible(x: torch.Tensor) -> bool:
    """Check the device, layout, gradient, and packed-width kernel contract."""

    return not (
        not x.is_cuda
        or torch.is_grad_enabled()
        or not x.is_contiguous()
        or x.shape[-1] % 2 != 0
    )


def silu_and_mul_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply packed SwiGLU and emit its BF16-rounded row-scaled E4M3 output."""

    if not _act_inputs_eligible(x) or triton is None or not triton_available(x.device):
        from ..nn.quant.fp8 import quantize_fp8_rowwise

        output = F.silu(x[..., : x.shape[-1] // 2]) * x[..., x.shape[-1] // 2 :]
        rows = output.reshape(-1, output.shape[-1])
        quantized, scale = quantize_fp8_rowwise(rows)
        return quantized.reshape(output.shape), scale

    n_cols = int(x.shape[-1] // 2)
    rows = x.numel() // (2 * n_cols)
    output = torch.empty((*x.shape[:-1], n_cols), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    block = triton.next_power_of_2(n_cols)
    _silu_and_mul_fp8_kernel[(rows,)](
        x,
        output,
        scale,
        n_cols,
        block,
        num_warps=32 if block >= 32_768 else 16,
    )
    return output, scale


class TritonSiluAndMul(Operator):
    """Triton provider for packed SiLU gating."""

    def __init__(self) -> None:
        """Register the Triton provider identity."""

        super().__init__("triton", "silu_and_mul")

    def can_run(self, req: SiluAndMulReq) -> bool:
        """Accept contiguous even-width CUDA inputs when Triton is available."""

        if not _act_inputs_eligible(req.x):
            return False
        if triton is None or not triton_available(req.x.device):
            return False
        return int(req.x.shape[-1] // 2) > 0

    def run(self, req: SiluAndMulReq) -> torch.Tensor:
        """Launch gated activation over flattened rows and restore input rank."""

        n_cols = int(req.x.shape[-1] // 2)
        out = torch.empty((*req.x.shape[:-1], n_cols), dtype=req.x.dtype, device=req.x.device)
        rows = out.numel() // n_cols
        block = _TRITON_ACT_BLOCK

        # Flatten leading dimensions into independent activation rows; chunking
        # keeps pointer arithmetic within the kernel's signed indexing range.
        x_rows = req.x.view(rows, 2 * n_cols)
        out_rows = out.view(rows, n_cols)
        for start, end in _triton_act_row_chunks(rows, n_cols):
            grid = (end - start, triton.cdiv(n_cols, block))
            _silu_and_mul_kernel[grid](
                x_rows[start:end],
                out_rows[start:end],
                n_cols,
                block,
                num_warps=4,
            )
        return out


class SglSiluAndMul(Operator):
    """SGL extension provider for packed SiLU gating."""

    def __init__(self) -> None:
        """Register the SGL provider identity."""

        super().__init__("sgl_kernel", "silu_and_mul")

    def can_run(self, req: SiluAndMulReq) -> bool:
        """Accept aligned packed inputs supported by the SGL extension."""

        if _sgl_silu_and_mul_kernel() is None:
            return False
        if not _act_inputs_eligible(req.x):
            return False
        return int(req.x.shape[-1]) * int(req.x.element_size()) % _SGL_ALIGNMENT_BYTES == 0

    def run(self, req: SiluAndMulReq) -> torch.Tensor:
        """Run the resolved SGL fused activation kernel."""

        kernel = _sgl_silu_and_mul_kernel()
        if kernel is None:
            raise RuntimeError("sgl silu_and_mul kernel unavailable")
        return kernel(req.x)


class EagerSiluAndMul(Operator):
    """Portable tensor provider for packed SiLU gating."""

    def __init__(self) -> None:
        """Register the eager provider identity."""

        super().__init__("eager", "silu_and_mul")

    def can_run(self, req: SiluAndMulReq) -> bool:
        """Accept any request; tensor operations enforce shape compatibility."""

        del req
        return True

    def run(self, req: SiluAndMulReq) -> torch.Tensor:
        """Split packed gate/value halves and multiply the activated gate."""

        x, y = req.x.chunk(2, dim=-1)
        return F.silu(x) * y


@lru_cache(maxsize=1)
def silu_and_mul_dispatcher() -> Dispatcher[SiluAndMulReq, torch.Tensor]:
    """Return the process-wide SiLU provider dispatcher."""

    return Dispatcher(
        "silu_and_mul",
        [TritonSiluAndMul(), SglSiluAndMul(), EagerSiluAndMul()],
        env_override="UNISERVE_SILU_AND_MUL_PROVIDER",
    )
