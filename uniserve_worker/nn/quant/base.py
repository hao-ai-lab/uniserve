"""Quantization method contract for shared layers.

Only the unquantized method is active today.  The important production seam is
that the method owns parameter creation, so future fp8/int8 layouts do not have
to fight a dense weight that was already registered by ``LinearBase``.
"""
from __future__ import annotations

import abc
from typing import ClassVar, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'QuantizeMethodBase',
    'UnquantizedLinearMethod',
    'process_quantized_modules',
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

    def apply(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, module.weight, module.bias)


def process_quantized_modules(modules: Iterable[nn.Module]) -> None:
    for module in modules:
        method = getattr(module, "quant_method", None)
        if isinstance(method, QuantizeMethodBase):
            method.process_weights_after_loading(module)
