"""Activation helpers shared by model definitions."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backends.triton import triton_available

__all__ = [
    'SiluAndMul',
    'GeluAndMul',
    'get_act_fn',
]

try:  # pragma: no cover - availability depends on the serving environment.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


# The 1024-column tile gives each thread eight contiguous BF16 elements with
# four warps per program. SiLU and multiplication are evaluated in FP32 before
# the output is converted to its storage dtype.
_TRITON_ACT_BLOCK = 1024
_TRITON_MAX_SIGNED_INDEX = (1 << 31) - 1


def _triton_act_row_chunks(rows: int, n_cols: int) -> Iterator[tuple[int, int]]:
    """Partition rows so every kernel launch uses int32-safe element offsets."""

    rows = int(rows)
    n_cols = int(n_cols)
    if rows < 0 or n_cols <= 0:
        raise ValueError("activation row geometry must be non-negative with positive columns")
    rows_per_launch = max(1, _TRITON_MAX_SIGNED_INDEX // (2 * n_cols))
    for start in range(0, rows, rows_per_launch):
        yield start, min(rows, start + rows_per_launch)


if triton is not None:

    @triton.jit
    def _silu_and_mul_kernel(x_ptr, out_ptr, n_cols: tl.constexpr, block: tl.constexpr):
        # 2D grid: axis 0 walks rows, axis 1 walks column blocks. Offsets are
        # contiguous within a program, so no integer div/mod per element.
        row = tl.program_id(0)
        cols = tl.program_id(1) * block + tl.arange(0, block)
        mask = cols < n_cols
        base = row * (n_cols * 2)
        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(x_ptr + base + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        silu = x / (1.0 + tl.exp(-x))
        out = silu * y
        tl.store(out_ptr + row * n_cols + cols, out, mask=mask)


def _act_inputs_eligible(x: torch.Tensor, y: torch.Tensor | None) -> bool:
    """Preconditions shared by every fused SiLU-and-mul backend."""

    return not (
        y is not None
        or not x.is_cuda
        or torch.is_grad_enabled()
        or not x.is_contiguous()
        or x.shape[-1] % 2 != 0
    )



class _TritonSiluAndMul:
    def is_eligible(self, x: torch.Tensor, y: torch.Tensor | None) -> bool:
        if not _act_inputs_eligible(x, y):
            return False
        if not (
            triton is not None
            and triton_available(x.device)
        ):
            return False
        return int(x.shape[-1] // 2) > 0

    def run(self, x: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        n_cols = int(x.shape[-1] // 2)
        out = torch.empty((*x.shape[:-1], n_cols), dtype=x.dtype, device=x.device)
        rows = out.numel() // n_cols
        block = _TRITON_ACT_BLOCK
        x_rows = x.view(rows, 2 * n_cols)
        out_rows = out.view(rows, n_cols)
        for start, end in _triton_act_row_chunks(rows, n_cols):
            # Rows on axis 0 and column blocks on axis 1. Each launch rebases
            # its pointers so the kernel's fast int32 offsets cannot overflow.
            grid = (end - start, triton.cdiv(n_cols, block))
            _silu_and_mul_kernel[grid](
                x_rows[start:end],
                out_rows[start:end],
                n_cols,
                block,
                num_warps=4,
            )
        return out


class _EagerSiluAndMul:
    def is_eligible(self, x: torch.Tensor, y: torch.Tensor | None) -> bool:
        return True

    def run(self, x: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        if y is None:
            x, y = x.chunk(2, dim=-1)
        return F.silu(x) * y



class SiluAndMul(nn.Module):
    """Apply SiLU to one half of a tensor and multiply by the other half."""

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        if y is not None:
            return _EagerSiluAndMul().run(x, y)
        from uniserve_worker import ops

        return ops.silu_and_mul(x)


class GeluAndMul(nn.Module):
    """Apply GELU to one half of a tensor and multiply by the other half."""

    def __init__(self, approximate: str = "none") -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        if y is None:
            x, y = x.chunk(2, dim=-1)
        return F.gelu(x, approximate=self.approximate) * y


# Maps each accepted activation name to a factory producing a fresh module.
_ACT_FN_REGISTRY: dict[str, Callable[[], nn.Module]] = {
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "gelu": nn.GELU,
    "gelu_pytorch_tanh": lambda: nn.GELU(approximate="tanh"),
    "gelu_fast": lambda: nn.GELU(approximate="tanh"),
    "gelu_approx": lambda: nn.GELU(approximate="tanh"),
    "relu": nn.ReLU,
    "silu_and_mul": SiluAndMul,
    "swiglu": SiluAndMul,
    "gelu_and_mul": GeluAndMul,
    "geglu": GeluAndMul,
}


def get_act_fn(name: str) -> nn.Module:
    factory = _ACT_FN_REGISTRY.get(name.lower())
    if factory is None:
        raise ValueError(f"unknown activation {name!r}")
    return factory()
