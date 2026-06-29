"""Activation helpers shared by model definitions."""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..foundation.triton_compat import triton_device_supported, triton_fused_layers_enabled

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


# Triton block tile for the fused SiLU-and-mul kernel; fixed by the kernel build.
_TRITON_ACT_BLOCK = 256


if triton is not None:

    @triton.jit
    def _silu_and_mul_kernel(x_ptr, out_ptr, n_cols: tl.constexpr, total: tl.constexpr, block: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * block + tl.arange(0, block)
        mask = offs < total
        row = offs // n_cols
        col = offs - row * n_cols
        base = row * (n_cols * 2)
        x = tl.load(x_ptr + base + col, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(x_ptr + base + n_cols + col, mask=mask, other=0.0).to(tl.float32)
        silu = (x / (1.0 + tl.exp(-x))).to(tl.bfloat16).to(tl.float32)
        out = silu * y
        tl.store(out_ptr + offs, out, mask=mask)


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
            and triton_fused_layers_enabled()
            and triton_device_supported(x.device)
        ):
            return False
        return int(x.shape[-1] // 2) > 0

    def run(self, x: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        n_cols = int(x.shape[-1] // 2)
        out = torch.empty((*x.shape[:-1], n_cols), dtype=x.dtype, device=x.device)
        total = out.numel()
        block = _TRITON_ACT_BLOCK
        grid = (triton.cdiv(total, block),)
        _silu_and_mul_kernel[grid](x, out, n_cols, total, block, num_warps=4)
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
