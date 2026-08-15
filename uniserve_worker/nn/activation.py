"""Activation helpers shared by model definitions."""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from uniserve_worker import ops

__all__ = [
    "SiluAndMul",
    "GeluAndMul",
    "get_act_fn",
]


class SiluAndMul(nn.Module):
    """Apply SiLU to one half of a tensor and multiply by the other half."""

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return ops.silu_and_mul(x, y)


class GeluAndMul(nn.Module):
    """Apply GELU to one half of a tensor and multiply by the other half."""

    def __init__(self, approximate: str = "none") -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        if y is None:
            x, y = x.chunk(2, dim=-1)
        return F.gelu(x, approximate=self.approximate) * y


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
