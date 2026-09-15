"""Resident diffusion modulation products prepared from streamed checkpoint projections."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["Modulation"]


class Modulation(nn.Module):
    """Own timestep-only products without retaining their projection weights.

    A fixed solver ladder permits preparation once at load time. Projection
    inputs retain the caller's batch dimensions and dtype conversion; individual
    weights may be materialized and released by a checkpoint iterator. Requests
    borrow products from this immutable model resource.
    """

    products: torch.Tensor
    output_products: torch.Tensor | None

    def __init__(self, products: torch.Tensor, output_products: torch.Tensor | None) -> None:
        super().__init__()
        if products.ndim != 4 or (output_products is not None and output_products.ndim != 3):
            raise ValueError("modulation products require step, layer, row, and output axes")
        if output_products is not None and (
            products.shape[0] != output_products.shape[0]
            or products.shape[2] != output_products.shape[1]
        ):
            raise ValueError("layer and output modulation products must share the timestep ladder")
        if min(products.shape) < 1 or (
            output_products is not None and min(output_products.shape) < 1
        ):
            raise ValueError("modulation products require nonempty dimensions")
        self.register_buffer("products", products)
        self.register_buffer("output_products", output_products)

    @classmethod
    @torch.inference_mode()
    def from_projections(
        cls,
        activated_timesteps: torch.Tensor,
        layer_projections: Iterable[tuple[torch.Tensor, torch.Tensor | None]],
        final_projection: tuple[torch.Tensor, torch.Tensor | None] | None,
        *,
        layer_count: int,
    ) -> Modulation:
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
                raise ValueError("modulation layer projections must share output shape and format")
            blocks[:, index].copy_(projected)
            count += 1
            del weight, bias, projected
        if blocks is None or count != layer_count:
            raise ValueError("modulation checkpoint is missing layer projections")
        return cls(blocks, None if final_projection is None else project(*final_projection))

    def forward(self, step_index: int, layer_index: int) -> torch.Tensor:
        """Borrow one layer's modulation products at a prepared solver step."""
        if (
            not 0 <= step_index < self.products.shape[0]
            or not 0 <= layer_index < self.products.shape[1]
        ):
            raise ValueError("modulation indices must lie within the prepared step/layer domain")
        return self.products[step_index, layer_index]

    def output(self, step_index: int) -> torch.Tensor:
        """Borrow final-normalization products for one prepared solver step."""
        if self.output_products is None:
            raise ValueError("this modulation has no output projection")
        if not 0 <= step_index < self.output_products.shape[0]:
            raise ValueError("modulation step must lie within the prepared ladder")
        return self.output_products[step_index]
