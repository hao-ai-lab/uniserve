"""Shared gated expansion and tensor-parallel output reduction."""

import torch
from torch import nn

from uniserve.quantization import Quantizer
from uniserve.tensors import _join_channels

from .activation import GELUAndMul, SiLUAndMul
from .linear import MergedColumnParallelLinear, RowParallelLinear


class GatedMLP(nn.Module):
    """Compose named gate/up projections, activation, and a reduced down
    projection.
    """  # noqa: D205

    activation: SiLUAndMul | GELUAndMul

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        activation: str = "silu",
        bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if activation == "silu":
            self.activation = SiLUAndMul()
        elif activation in {"gelu", "gelu_pytorch_tanh"}:
            self.activation = GELUAndMul(
                approximate="tanh" if activation.endswith("tanh") else "none"
            )
        else:
            raise ValueError(f"unsupported gated activation {activation!r}")
        self.gate_up = MergedColumnParallelLinear(
            hidden_size,
            {"gate": intermediate_size, "up": intermediate_size},
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.down = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = self.gate_up(x)
        packed = _join_channels((branches["gate"], branches["up"]))

        if (
            isinstance(self.activation, SiLUAndMul)
            and self.down.input_quantizer == Quantizer("fp8", axis=0)
            and self.down.group.size == 1
        ):
            # The fused producer computes FP32 gating and encodes once before
            # the contraction. K-sharded scales must instead cover all shards.
            shape = (*packed.shape[:-1], packed.shape[-1] // 2)
            matrix = packed.reshape(-1, packed.shape[-1])
            encoded = self.down.input_quantizer.empty(
                (matrix.shape[0], shape[-1]),
                dtype=packed.dtype,
                device=packed.device,
            )
            hidden = self.activation(matrix, out=encoded)
            return self.down(hidden).reshape(
                *shape[:-1], self.down.out_features
            )

        return self.down(self.activation(packed))
