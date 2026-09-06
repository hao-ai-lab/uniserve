"""Weight creation, loading finalization, and execution for shared linear layers."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import ClassVar, Iterable, Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "QuantizeMethodBase",
    "UnquantizedLinearMethod",
    "process_quantized_modules",
]


@dataclass(frozen=True)
class PreparedLinearInput:
    """Quantized values, optional per-block scales, and their shared scale domain."""

    values: torch.Tensor
    block_scales: torch.Tensor | None = None
    global_scale: torch.Tensor | None = None


class QuantizeMethodBase(abc.ABC):
    """Defines parameter creation, loading finalization, and execution for a linear quantization method."""

    # Cross-cutting flag read by consumers to decide quantized-only handling.
    is_quantized: ClassVar[bool] = False

    @property
    def weight_scale_domain(self) -> Literal["tensor", "row", "block"]:
        """Axes whose extrema must cover the logical weight before physical sharding."""

        return "block"

    @property
    def input_scale_domain(self) -> Literal["tensor", "row", "block"]:
        """Whether physical input-column shards share an activation scale."""

        return "block"

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
        """Register weight, scale, and optional bias storage on a linear module."""

        raise NotImplementedError

    @abc.abstractmethod
    def apply(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Project activations through a materialized linear module."""

        raise NotImplementedError

    def process_weights_after_loading(self, module: nn.Module) -> None:
        """Finalize loaded tensors into the representation consumed by execution."""

        return None

    def apply_prequantized(
        self,
        module: nn.Module,
        x: torch.Tensor,
        scale: torch.Tensor,
        *,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Project caller-quantized activations when the method supports that boundary."""

        del module, x, scale, output_dtype
        raise RuntimeError("linear quantization method cannot consume prequantized activations")

    def input_scale(self, x: torch.Tensor) -> torch.Tensor | None:
        """Return local activation scales for the declared tensor or row domain."""

        raise RuntimeError("linear format has no explicit input-scale preparation")

    def prepare_input(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None,
    ) -> PreparedLinearInput:
        raise RuntimeError("linear format has no prepared-input representation")

    def apply_prepared(
        self,
        module: nn.Module,
        prepared: PreparedLinearInput,
        *,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        raise RuntimeError("linear format has no prepared GEMM")


class UnquantizedLinearMethod(QuantizeMethodBase):
    """Implements dense linear execution with loader-managed unquantized weights."""

    def create_weights(
        self,
        module: nn.Module,
        *,
        input_size: int,
        output_size: int,
        bias: bool,
        **_: object,
    ) -> None:
        """Register dense weight and optional bias parameters with checkpoint loaders."""

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
        """Project activations with the module's dense weight and optional bias."""

        from ..linear import LinearBase

        linear = cast(LinearBase, module)
        return F.linear(x, linear.weight, linear.execution_bias)

    def input_scale(self, x: torch.Tensor) -> torch.Tensor | None:
        return None

    def prepare_input(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None,
    ) -> PreparedLinearInput:
        return PreparedLinearInput(x)

    def apply_prepared(
        self,
        module: nn.Module,
        prepared: PreparedLinearInput,
        *,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        from ..linear import LinearBase

        linear = cast(LinearBase, module)
        values = prepared.values
        if output_dtype == values.dtype:
            return self.apply(module, values)
        if output_dtype != torch.float32 or values.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("dense prepared GEMM supports input dtype or FP32 output")
        if values.is_cuda:
            output = torch.mm(values, linear.weight.T, out_dtype=output_dtype)
            bias = linear.execution_bias
            return output if bias is None else output + bias
        bias = linear.execution_bias
        return F.linear(
            values.float(), linear.weight.float(), None if bias is None else bias.float()
        )


def process_quantized_modules(modules: Iterable[nn.Module]) -> None:
    """Finalize the loaded weight representation of every quantized module."""

    for module in modules:
        method = getattr(module, "quant_method", None)
        if isinstance(method, QuantizeMethodBase):
            from ..linear import LinearBase

            if isinstance(module, LinearBase):
                module.finalize_weights()
            else:
                method.process_weights_after_loading(module)
