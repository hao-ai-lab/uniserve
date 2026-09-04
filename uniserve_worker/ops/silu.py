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


def _act_inputs_eligible(x: torch.Tensor) -> bool:
    """Check the device, layout, gradient, and packed-width kernel contract."""

    return not (
        not x.is_cuda
        or torch.is_grad_enabled()
        or not x.is_contiguous()
        or x.shape[-1] % 2 != 0
    )


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
