"""Shared Qwen-style decoder components."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from ...forward import MeshView
from ..activation import GeluAndMul
from ..layer import LayerSpec
from ..linear import MergedColumnParallelLinear, RowParallelLinear
from ..placement import WeightMode

__all__ = [
    "Qwen3MLP",
    "qwen3_gate_up_activation",
]


def _silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    from uniserve_worker import ops

    return ops.silu_and_mul(x)


def qwen3_gate_up_activation(hidden_act: str | None) -> Callable[[torch.Tensor], torch.Tensor]:
    key = str(hidden_act or "silu").lower()
    if key in {"silu", "swish", "silu_and_mul", "swiglu"}:
        return _silu_and_mul
    if key in {"gelu", "gelu_and_mul", "geglu"}:
        return GeluAndMul()
    if key in {"gelu_pytorch_tanh", "gelu_tanh"}:
        return GeluAndMul(approximate="tanh")
    raise ValueError(f"Qwen3MLP does not support hidden_act={hidden_act!r}")


class Qwen3MLP(nn.Module):
    """SwiGLU-style feed-forward block with a fused gate/up projection."""

    def __init__(
        self,
        config: Any,
        *,
        spec: LayerSpec,
        weight_mode: WeightMode = WeightMode.VANILLA,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            int(config.hidden_size),
            (int(config.intermediate_size), int(config.intermediate_size)),
            spec=spec,
            bias=False,
            weight_mode=weight_mode,
        )
        self.act = qwen3_gate_up_activation(getattr(config, "hidden_act", "silu"))
        self.down_proj = RowParallelLinear(
            int(config.intermediate_size),
            int(config.hidden_size),
            spec=spec,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor, mesh: MeshView) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_up_proj(hidden_states)), mesh)
