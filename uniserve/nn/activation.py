"""Activation helpers shared by model definitions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn

from . import functional

__all__ = [
    "SiLUAndMul",
    "GELUAndMul",
    "get_act_fn",
]


class SiLUAndMul(nn.Module):
    """Apply SiLU gating to equal channel halves of a packed tensor."""

    def forward(
        self, x: torch.Tensor, *, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply SiLU to the gate half and multiply by the value half."""
        return functional.silu_and_mul(x, out=out)


class GELUAndMul(nn.Module):
    """Apply GELU gating to equal channel halves of a packed tensor."""

    def __init__(self, approximate: Literal["none", "tanh"] = "none") -> None:
        """Select the PyTorch GELU approximation used by the gate."""
        super().__init__()
        self.approximate = approximate

    def forward(
        self, x: torch.Tensor, *, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply GELU to the gate half and multiply by the value half."""
        return functional.gelu_and_mul(x, approximate=self.approximate, out=out)


# Checkpoint activation aliases resolve to one module per numerical formula.
_ACT_FN_REGISTRY: dict[str, Callable[[], nn.Module]] = {
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "gelu": nn.GELU,
    "gelu_pytorch_tanh": lambda: nn.GELU(approximate="tanh"),
    "gelu_fast": lambda: nn.GELU(approximate="tanh"),
    "gelu_approx": lambda: nn.GELU(approximate="tanh"),
    "relu": nn.ReLU,
    "silu_and_mul": SiLUAndMul,
    "swiglu": SiLUAndMul,
    "gelu_and_mul": GELUAndMul,
    "geglu": GELUAndMul,
}


def get_act_fn(name: str) -> nn.Module:
    """Construct the registered activation module for a checkpoint activation
    name.
    """  # noqa: D205
    factory = _ACT_FN_REGISTRY.get(name.lower())
    if factory is None:
        raise ValueError(f"unknown activation {name!r}")
    return factory()
