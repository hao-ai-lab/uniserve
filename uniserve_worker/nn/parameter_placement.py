"""Parameter creation and load placement policy."""
from __future__ import annotations

import torch
from torch import nn

from .placement import ShardPlan, get_shard_plan, set_shard_plan

__all__ = [
    "ParameterPlacementPolicy",
]


class ParameterPlacementPolicy:
    """Owns parameter creation, load placement, shard plans, and finalization."""

    def create_parameter(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device | str = "meta",
        requires_grad: bool = False,
    ) -> nn.Parameter:
        return nn.Parameter(
            torch.empty(tuple(int(dim) for dim in shape), dtype=dtype, device=device),
            requires_grad=requires_grad,
        )

    def load_tensor(self, parameter: nn.Parameter, tensor: torch.Tensor, *, dtype: torch.dtype | None = None) -> nn.Parameter:
        value = tensor.to(dtype=dtype or parameter.dtype, device=parameter.device)
        with torch.no_grad():
            parameter.copy_(value)
        return parameter

    def attach_shard_plan(self, parameter: nn.Parameter, plan: ShardPlan | None) -> nn.Parameter:
        if plan is not None:
            set_shard_plan(parameter, plan)
        return parameter

    def finalize(self, module: nn.Module) -> nn.Module:
        finalize = getattr(module, "finalize_quantization", None)
        if callable(finalize):
            finalize()
        return module

    def apply(self, parameter: nn.Parameter) -> nn.Parameter:
        plan = get_shard_plan(parameter)
        if plan is not None:
            return parameter
        return parameter
