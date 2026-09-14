"""Resident diffusion modulation products prepared from streamed checkpoint projections."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["ModulationPlan"]


class ModulationPlan(nn.Module):
    """Own timestep-only products without retaining their projection weights.

    A fixed solver ladder permits preparation once at load time. Projection
    inputs retain the caller's batch geometry and dtype conversion; individual
    weights may be materialized and released by a checkpoint iterator. Requests
    borrow products from this immutable model resource.
    """

    blocks: torch.Tensor
    final: torch.Tensor | None

    def __init__(self, blocks: torch.Tensor, final: torch.Tensor | None) -> None:
        super().__init__()
        if blocks.ndim != 4 or (final is not None and final.ndim != 3):
            raise ValueError("modulation products require step, layer, row, and output axes")
        if final is not None and (
            blocks.shape[0] != final.shape[0] or blocks.shape[2] != final.shape[1]
        ):
            raise ValueError("block and final modulation products must share the timestep ladder")
        if min(blocks.shape) < 1 or (final is not None and min(final.shape) < 1):
            raise ValueError("modulation products require nonempty dimensions")
        self.register_buffer("blocks", blocks)
        self.register_buffer("final", final)

    @classmethod
    @torch.inference_mode()
    def materialize(
        cls,
        activated_timesteps: torch.Tensor,
        layer_projections: Iterable[tuple[torch.Tensor, torch.Tensor | None]],
        final_projection: tuple[torch.Tensor, torch.Tensor | None] | None,
        *,
        layer_count: int,
    ) -> ModulationPlan:
        """Prepare [step, layer, row, output] products from each streamed weight.

        Inputs have [step, row, input] shape. Each projection executes one GEMM
        over all step/row pairs, preserving the declared logical projection
        batch. The iterator must yield exactly ``layer_count`` dense weight and
        optional bias pairs, with a common output width, device, and dtype.
        Pipeline stages without a final normalization omit its projection.
        """

        if activated_timesteps.ndim != 3 or min(activated_timesteps.shape) < 1 or layer_count < 1:
            raise ValueError("modulation preparation requires a nonempty fixed timestep ladder")
        steps, rows, width = activated_timesteps.shape
        inputs = activated_timesteps.flatten(0, 1)

        def project(weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
            if weight.ndim != 2 or weight.shape[1] != width:
                raise ValueError("modulation checkpoint projection has incompatible input width")
            if bias is not None and tuple(bias.shape) != (weight.shape[0],):
                raise ValueError("modulation checkpoint bias has incompatible output width")
            return F.linear(inputs.to(weight.dtype), weight, bias).view(steps, rows, -1)

        blocks = None
        count = 0
        for index, (weight, bias) in enumerate(layer_projections):
            if index >= layer_count:
                raise ValueError("modulation checkpoint contains too many layer projections")
            projected = project(weight, bias)
            if blocks is None:
                blocks = projected.new_empty((steps, layer_count, rows, projected.shape[-1]))
            if (
                projected.shape != blocks[:, index].shape
                or projected.dtype != blocks.dtype
                or projected.device != blocks.device
            ):
                raise ValueError(
                    "modulation layer projections must share output geometry and format"
                )
            blocks[:, index].copy_(projected)
            count += 1
            del weight, bias, projected
        if blocks is None or count != layer_count:
            raise ValueError("modulation checkpoint is missing layer projections")
        return cls(blocks, None if final_projection is None else project(*final_projection))

    @torch.inference_mode()
    def copy_step(self, step: int, block_output: torch.Tensor, final_output: torch.Tensor) -> None:
        """Copy one solver step into caller-owned fixed-address execution buffers."""

        if not 0 <= step < self.blocks.shape[0]:
            raise ValueError("modulation step is outside the prepared ladder")
        final_matches = (
            final_output.numel() == 0
            if self.final is None
            else final_output.shape == self.final[step].shape
        )
        if block_output.shape != self.blocks[step].shape or not final_matches:
            raise ValueError("modulation execution buffers have incompatible geometry")
        block_output.copy_(self.blocks[step])
        if self.final is not None:
            final_output.copy_(self.final[step])
