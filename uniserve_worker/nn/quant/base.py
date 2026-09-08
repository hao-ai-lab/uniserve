"""Weight creation, loading finalization, and execution for shared linear layers."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Iterable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from ..linear import LinearBase

__all__ = [
    "LinearMethod",
    "UnquantizedLinearMethod",
    "process_quantized_modules",
]


BlockScaleLayout = Literal["linear", "128x4"]


@dataclass(frozen=True, slots=True)
class PreparedLinearInput:
    """Values and format-specific dequantization scales for a linear projection.

    Leading value dimensions retain the logical input shape. NVFP4 packs two
    input columns per byte. Row scales index flattened logical rows; a tensor
    scale belongs to the complete logical quantization domain. Block scales
    retain their declared physical layout, including accelerator padding.
    """

    values: torch.Tensor
    block_scales: torch.Tensor | None = None
    tensor_scale: torch.Tensor | None = None
    row_scales: torch.Tensor | None = None
    block_scale_layout: BlockScaleLayout = "linear"


class LinearMethod(abc.ABC):
    """Defines parameter creation, loading finalization, and execution for a linear quantization method."""

    # Cross-cutting flag read by consumers to decide quantized-only handling.
    is_quantized: ClassVar[bool] = False
    preferred_block_scale_layout: ClassVar[BlockScaleLayout] = "linear"

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
        module: LinearBase,
        *,
        input_size: int,
        output_size: int,
        bias: bool,
        **_: object,
    ) -> None:
        """Register weight, scale, and optional bias storage on a linear module."""

        raise NotImplementedError

    @abc.abstractmethod
    def apply(self, module: LinearBase, x: torch.Tensor) -> torch.Tensor:
        """Project activations through a materialized linear module."""

        raise NotImplementedError

    @abc.abstractmethod
    def process_weights_after_loading(self, module: LinearBase) -> None:
        """Finalize loaded tensors into the representation consumed by execution."""

        raise NotImplementedError

    @abc.abstractmethod
    def input_scale(
        self, x: torch.Tensor, *, absmax: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """Return local scales, optionally reusing the complete tensor magnitude."""

        raise NotImplementedError

    @abc.abstractmethod
    def prepare_input(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None,
        *,
        block_scale_layout: BlockScaleLayout = "linear",
    ) -> PreparedLinearInput:
        raise NotImplementedError

    @abc.abstractmethod
    def apply_prepared(
        self,
        module: LinearBase,
        prepared: PreparedLinearInput,
        *,
        output_dtype: torch.dtype,
        include_bias: bool = True,
    ) -> torch.Tensor:
        """Project prepared values without changing their quantization domains."""

        raise NotImplementedError


class UnquantizedLinearMethod(LinearMethod):
    """Implements dense linear execution with loader-managed unquantized weights."""

    def create_weights(
        self,
        module: LinearBase,
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

        weight = module.weight
        attach_weight_loader(weight, default_weight_loader)
        bias_parameter = module.bias
        if bias_parameter is not None:
            attach_weight_loader(bias_parameter, default_weight_loader)

    def process_weights_after_loading(self, module: LinearBase) -> None:
        """Dense checkpoint storage is already the execution representation."""

    def apply(self, module: LinearBase, x: torch.Tensor) -> torch.Tensor:
        """Project activations with the module's dense weight and optional bias."""

        return F.linear(x, module.weight, module.execution_bias)

    def input_scale(
        self, x: torch.Tensor, *, absmax: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        return None

    def prepare_input(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None,
        *,
        block_scale_layout: BlockScaleLayout = "linear",
    ) -> PreparedLinearInput:
        return PreparedLinearInput(x)

    def apply_prepared(
        self,
        module: LinearBase,
        prepared: PreparedLinearInput,
        *,
        output_dtype: torch.dtype,
        include_bias: bool = True,
    ) -> torch.Tensor:
        values = prepared.values
        bias = module.execution_bias if include_bias else None
        if output_dtype == values.dtype:
            return F.linear(values, module.weight, bias)
        if output_dtype != torch.float32 or values.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("dense prepared GEMM supports input dtype or FP32 output")
        if values.is_cuda:
            output = torch.mm(values, module.weight.T, out_dtype=output_dtype)
            return output if bias is None else output + bias
        return F.linear(
            values.float(), module.weight.float(), None if bias is None else bias.float()
        )


def process_quantized_modules(modules: Iterable[nn.Module]) -> None:
    """Finalize the loaded weight representation of every quantized module."""

    from ..linear import LinearBase

    for module in modules:
        if isinstance(module, LinearBase):
            module.finalize_weights()
