"""Activation helpers shared by model definitions."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from uniserve import ops

__all__ = [
    "SiluAndMul",
    "GeluAndMul",
    "get_act_fn",
]


class SiluAndMul(nn.Module):
    """Applies SiLU gating to explicit inputs or equal halves of one packed tensor."""

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Apply SiLU to the first operand and multiply by the second or packed half."""

        return ops.silu_and_mul(x, y)


class GeluAndMul(nn.Module):
    """Applies GELU gating to explicit inputs or equal halves of one packed tensor."""

    def __init__(self, approximate: str = "none") -> None:
        """Select the PyTorch GELU approximation used by the gate."""

        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Apply GELU to the first operand and multiply by the second or packed half."""

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
    """Construct the registered activation module for a checkpoint activation name."""

    factory = _ACT_FN_REGISTRY.get(name.lower())
    if factory is None:
        raise ValueError(f"unknown activation {name!r}")
    return factory()
