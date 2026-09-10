"""Shared gated feed-forward layers and prepared activation composition."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
from torch import nn

from uniserve_worker import ops

from .activation import GeluAndMul
from .layer import LayerConfig
from .linear import (
    LinearBase,
    MergedColumnParallelLinear,
    RowParallelLinear,
    project_with_deferred_bias,
)
from .quant.base import LinearMethod, PreparedLinearInput
from .quant.fp8 import DynamicW8A8Fp8LinearMethod
from .shard import WeightMode

__all__ = ["GatedMLP"]


class GatedMLP(nn.Module):
    """Gated expansion and tensor-parallel reduction with explicit packed order.

    The packed projection retains the checkpoint-facing ``gate_up_proj`` name.
    ``order`` identifies its physical halves. Activation providers may fuse
    SiLU and multiplication before rounding to the output dtype.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        layer_config: LayerConfig,
        hidden_act: str = "silu",
        weight_mode: WeightMode = WeightMode.VANILLA,
        quant_method: LinearMethod | None = None,
        order: Literal["gate_value", "value_gate"] = "gate_value",
        bias: bool = False,
    ) -> None:
        super().__init__()
        activation = hidden_act.lower()
        self._silu = activation in {"silu", "swish", "silu_and_mul", "swiglu"}
        self.order = order
        self.act: Callable[[torch.Tensor], torch.Tensor]
        if order == "value_gate":
            if not self._silu:
                raise ValueError("value-first gated MLP requires SiLU")
            self.act = ops.value_first_swiglu
        elif order == "gate_value":
            if self._silu:
                self.act = ops.silu_and_mul
            elif activation in {"gelu", "gelu_and_mul", "geglu"}:
                self.act = GeluAndMul()
            elif activation in {"gelu_pytorch_tanh", "gelu_tanh"}:
                self.act = GeluAndMul(approximate="tanh")
            else:
                raise ValueError(f"gated MLP does not support hidden_act={hidden_act!r}")
        else:
            raise ValueError("gated MLP has an unsupported packed order")
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            (intermediate_size, intermediate_size),
            layer_config=layer_config,
            quant_method=(
                layer_config.quant_method("gate_up_proj", packed_names=("gate_proj", "up_proj"))
                if quant_method is None
                else quant_method
            ),
            prefix="gate_up_proj",
            bias=bias,
            weight_mode=weight_mode,
        )
        # A singleton projection includes dense bias in GEMM. Sharded outputs
        # apply bias after summing partial results across their input columns.
        projection = LinearBase if layer_config.communicator.world_size == 1 else RowParallelLinear
        self.down_proj = projection(
            intermediate_size,
            hidden_size,
            layer_config=layer_config,
            quant_method=quant_method,
            prefix="down_proj",
            bias=bias,
        )

    @property
    def accepts_prequantized_fp8(self) -> bool:
        """Whether row-scaled E4M3 producers can feed this SiLU projection pair."""

        return self._silu and all(
            isinstance(projection.quant_method, DynamicW8A8Fp8LinearMethod)
            and not projection.quant_method.tensorwise
            for projection in (self.gate_up_proj, self.down_proj)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply gated expansion and reduce the output projection across its mesh."""

        return self._project_output(self.gate_up_proj(hidden_states))

    def forward_prepared(self, prepared: PreparedLinearInput) -> torch.Tensor:
        """Consume row-scaled FP8 input without quantizing it again."""

        if not self.accepts_prequantized_fp8:
            raise ValueError("prepared gated MLP requires row-scaled FP8 SiLU projections")
        return self._project_output(self.gate_up_proj.forward_prepared(prepared))

    def forward_deferred(
        self, hidden: torch.Tensor, *, input_absmax: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Fuse projection bias into value-first gating and the caller's residual.

        The optional magnitude describes the complete input tensor. Tensor-scaled
        down projections reuse the magnitude of the rounded gated activation.
        Dense providers apply bias in their GEMM epilogue.
        """

        if self.order != "value_gate":
            raise ValueError("deferred gated MLP requires value-first SiLU with FP32 activation")
        packed, bias = project_with_deferred_bias(self.gate_up_proj, hidden, absmax=input_absmax)
        if self.down_proj.quant_method.input_scale_domain == "tensor":
            activated, maximum = ops.value_first_swiglu_absmax(packed, bias)
        else:
            activated = ops.value_first_swiglu(packed, bias)
            maximum = None
        return project_with_deferred_bias(self.down_proj, activated, absmax=maximum)

    def _project_output(self, packed: torch.Tensor) -> torch.Tensor:
        """Gate the expansion and preserve the reduction's logical scale domain.

        Sharded down projections resolve activation scales over their complete
        input domain. Their local activation cannot publish a final row scale.
        """

        if not self.accepts_prequantized_fp8 or self.down_proj.weight_group.world_size > 1:
            return self.down_proj(self.act(packed))
        if self.order == "value_gate":
            values, scales = ops.value_first_swiglu_fp8(packed)
        else:
            values, scales = ops.silu_and_mul_fp8(packed)
        output = self.down_proj.forward_prepared(PreparedLinearInput(values, row_scales=scales))
        return (
            self.down_proj.reduce_output(output)
            if isinstance(self.down_proj, RowParallelLinear)
            else output
        )
