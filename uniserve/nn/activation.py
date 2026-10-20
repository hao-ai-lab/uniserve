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
    """Apply SiLU gating to packed channel halves or a ``(gate, value)`` pair.

    ``rounding`` states whether the activated gate rounds to the input dtype
    before the product (see ``functional.Rounding``).
    """

    def __init__(self, rounding: functional.Rounding) -> None:
        super().__init__()
        self.rounding = rounding

    def forward(
        self, x: functional.GatingInput, *, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply SiLU to the gate channels and multiply by the value ones."""
        return functional.silu_and_mul(x, out=out, rounding=self.rounding)


class GELUAndMul(nn.Module):
    """Apply GELU gating to packed channel halves or a gate/value pair."""

    def __init__(self, approximate: Literal["none", "tanh"] = "none") -> None:
        """Select the PyTorch GELU approximation used by the gate."""
        super().__init__()
        self.approximate = approximate

    def forward(
        self, x: functional.GatingInput, *, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply GELU to the gate channels and multiply by the value ones."""
        return functional.gelu_and_mul(x, approximate=self.approximate, out=out)


# Elementwise activations of non-gated MLPs, by the names checkpoints use. A
# name resolves to one module per numerical formula; gated activations take
# gate and value channels and are the MLP's own (``SiLUAndMul``,
# ``GELUAndMul``).
_ACT_FN_REGISTRY: dict[str, Callable[[], nn.Module]] = {
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "gelu": nn.GELU,
    "gelu_pytorch_tanh": lambda: nn.GELU(approximate="tanh"),
    "gelu_fast": lambda: nn.GELU(approximate="tanh"),
    "relu": nn.ReLU,
}


def get_act_fn(name: str) -> nn.Module:
    """Construct the registered activation module for a checkpoint activation
    name.
    """  # noqa: D205
    factory = _ACT_FN_REGISTRY.get(name.lower())
    if factory is None:
        raise ValueError(f"unknown activation {name!r}")
    return factory()
