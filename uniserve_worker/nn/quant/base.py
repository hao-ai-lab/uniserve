"""Weight creation, loading finalization, and execution for shared linear layers."""

from __future__ import annotations

import abc
from typing import ClassVar, Iterable, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "QuantizeMethodBase",
    "UnquantizedLinearMethod",
    "process_quantized_modules",
]


class QuantizeMethodBase(abc.ABC):
    # Cross-cutting flag read by consumers to decide quantized-only handling.
    is_quantized: ClassVar[bool] = False

    @abc.abstractmethod
    def create_weights(
        self,
        module: nn.Module,
        *,
        input_size: int,
        output_size: int,
        bias: bool,
        **_: object,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def apply(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def process_weights_after_loading(self, module: nn.Module) -> None:
        return None


class UnquantizedLinearMethod(QuantizeMethodBase):
    def create_weights(
        self,
        module: nn.Module,
        *,
        input_size: int,
        output_size: int,
        bias: bool,
        **_: object,
    ) -> None:
        module.register_parameter(
            "weight",
            nn.Parameter(torch.empty(int(output_size), int(input_size))),
        )
        module.register_parameter(
            "bias",
            nn.Parameter(torch.empty(int(output_size))) if bias else None,
        )
        from ...loader.weight_loaders import attach_weight_loader, default_weight_loader

        weight = cast(nn.Parameter, module.weight)
        attach_weight_loader(weight, default_weight_loader)
        bias_parameter = cast(nn.Parameter | None, module.bias)
        if bias_parameter is not None:
            attach_weight_loader(bias_parameter, default_weight_loader)

    def apply(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        from ..linear import LinearBase

        linear = cast(LinearBase, module)
        return F.linear(x, linear.weight, linear.bias)

    def apply_sequence_parallel(
        self,
        module: nn.Module,
        x: torch.Tensor,
        mesh: object,
        workspace: torch.Tensor,
        *,
        group: str,
    ) -> torch.Tensor:
        from ..linear import LinearBase
        from ..mesh import DeviceMesh

        linear = cast(LinearBase, module)
        device_mesh = cast(DeviceMesh, mesh)
        global_rows = int(x.shape[0]) * device_mesh.size(group)
        gathered = workspace.view(x.dtype)[: global_rows * x.shape[1]].view(
            global_rows,
            x.shape[1],
        )
        device_mesh.all_gather_into_tensor(gathered, x, group)
        return F.linear(gathered, linear.weight, linear.bias)


def process_quantized_modules(modules: Iterable[nn.Module]) -> None:
    for module in modules:
        method = getattr(module, "quant_method", None)
        if isinstance(method, QuantizeMethodBase):
            method.process_weights_after_loading(module)
