"""Shared gated expansion and tensor-parallel output reduction."""

import torch
from torch import nn

from uniserve.quantization import Quantizer
from uniserve.tensors import _join_channels

from .activation import GELUAndMul, SiLUAndMul
from .functional import Rounding, calibrated_nvfp4
from .linear import MergedColumnParallelLinear, RowParallelLinear


class GatedMLP(nn.Module):
    """Compose named gate/up projections, activation, and a reduced down
    projection.

    A SiLU-gated MLP states the ``rounding`` recipe its model's reference
    evaluates (see ``SiLUAndMul``); a GELU-gated one has a single recipe and
    takes none.
    """  # noqa: D205

    activation: SiLUAndMul | GELUAndMul

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        activation: str = "silu",
        rounding: Rounding | None = None,
        bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if (activation == "silu") != (rounding is not None):
            raise ValueError(
                "SiLU gating requires a rounding recipe, and no other "
                "gating takes one"
            )
        if activation == "silu":
            assert rounding is not None
            self.activation = SiLUAndMul(rounding)
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
        gate, up = branches["gate"], branches["up"]
        # Branches written back to back gate as packed rows; separately
        # written branches gate as a pair, without joining their storage.
        packed = _join_channels((gate, up), copy=False)
        operands = (gate, up) if packed is None else packed

        quantizer = self.down.input_quantizer
        if (
            quantizer is not None
            and self.down.group.size == 1
            and (
                quantizer == Quantizer("fp8", axis=0)
                or (calibrated_nvfp4(quantizer) and gate.is_cuda)
            )
        ):
            # The fused producer computes the gating under the activation's
            # rounding recipe and encodes once before the contraction: one
            # E4M3 scale per row, or NVFP4 blocks against the calibrated
            # tensor scale, so the gated rows are never stored. K-sharded
            # row scales must instead cover all shards.
            width = gate.shape[-1]
            rows = gate.numel() // width
            matrix = (
                (gate.reshape(rows, width), up.reshape(rows, width))
                if packed is None
                else packed.reshape(rows, 2 * width)
            )
            encoded = quantizer.empty(
                (rows, width), dtype=gate.dtype, device=gate.device
            )
            hidden = self.activation(matrix, out=encoded)
            return self.down(hidden).reshape(
                *gate.shape[:-1], self.down.out_features
            )

        return self.down(self.activation(operands))
