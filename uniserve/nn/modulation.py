"""Resident diffusion modulation products prepared from streamed checkpoint
projections.
"""  # noqa: D205

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["Modulation"]


class Modulation(nn.Module):
    """Own timestep-only products without retaining their projection weights.

    A fixed solver schedule permits preparation once at load time. The
    products are stored once per distinct timestep value (an *entry*); a
    step reads one entry per timestep group (for example the generated
    video, the generated audio and a condition held at a fixed level), which
    ``groups`` names. Projection inputs retain the caller's row order and
    dtype conversion; individual weights may be materialized and released by
    a checkpoint iterator. Requests borrow products from this immutable model
    resource.

    Attributes:
        products: [layer, entry, output] products of every layer projection.
        output_products: [entry, output] products of the final projection, or
            None on a pipeline stage without it.
        groups: [step, group] int64 entry each timestep group reads at each
            solver step.
    """

    products: torch.Tensor
    output_products: torch.Tensor | None
    groups: torch.Tensor

    def __init__(
        self,
        products: torch.Tensor,
        output_products: torch.Tensor | None,
        groups: torch.Tensor,
    ) -> None:
        super().__init__()
        if products.ndim != 3 or (
            output_products is not None and output_products.ndim != 2
        ):
            raise ValueError(
                "modulation products require layer, entry and output axes"
            )
        if output_products is not None and (
            products.shape[1] != output_products.shape[0]
        ):
            raise ValueError(
                "layer and output modulation products must share their "
                "timestep entries"
            )
        if min(products.shape) < 1 or (
            output_products is not None and min(output_products.shape) < 1
        ):
            raise ValueError("modulation products require nonempty dimensions")
        if (
            groups.ndim != 2
            or min(groups.shape) < 1
            or groups.dtype != torch.int64
            or (
                not groups.is_meta
                and (
                    bool((groups < 0).any())
                    or bool((groups >= products.shape[1]).any())
                )
            )
        ):
            raise ValueError(
                "modulation groups must name one entry per step and group"
            )
        self.register_buffer("products", products)
        self.register_buffer("output_products", output_products)
        self.register_buffer("groups", groups)

    @classmethod
    @torch.inference_mode()
    def from_projections(
        cls,
        activated_timesteps: torch.Tensor,
        layer_projections: Iterable[tuple[torch.Tensor, torch.Tensor | None]],
        final_projection: tuple[torch.Tensor, torch.Tensor | None] | None,
        *,
        groups: torch.Tensor,
        layer_count: int,
    ) -> Modulation:
        """Prepare [layer, entry, output] products from each streamed weight.

        ``activated_timesteps`` holds one [entry, input] row per distinct
        timestep value. Each projection executes one GEMM over every entry,
        in the caller's row order. The iterator must yield exactly
        ``layer_count`` dense weight and optional bias pairs, with a common
        output width, device, and dtype. Pipeline stages without a final
        normalization omit its projection. ``groups`` is moved to the
        products' device.
        """  # noqa: D205
        if (
            activated_timesteps.ndim != 2
            or min(activated_timesteps.shape) < 1
            or layer_count < 1
        ):
            raise ValueError(
                "modulation preparation requires nonempty fixed timestep "
                "entries"
            )
        entries, width = activated_timesteps.shape

        def project(
            weight: torch.Tensor, bias: torch.Tensor | None
        ) -> torch.Tensor:
            if weight.ndim != 2 or weight.shape[1] != width:
                raise ValueError(
                    "modulation checkpoint projection has incompatible input "
                    "width"
                )
            if bias is not None and tuple(bias.shape) != (weight.shape[0],):
                raise ValueError(
                    "modulation checkpoint bias has incompatible output width"
                )
            return F.linear(activated_timesteps.to(weight.dtype), weight, bias)

        blocks = None
        count = 0
        for index, (weight, bias) in enumerate(layer_projections):
            if index >= layer_count:
                raise ValueError(
                    "modulation checkpoint contains too many layer projections"
                )

            projected = project(weight, bias)
            if blocks is None:
                blocks = projected.new_empty(
                    (layer_count, entries, projected.shape[-1])
                )
            if (
                projected.shape != blocks[index].shape
                or projected.dtype != blocks.dtype
                or projected.device != blocks.device
            ):
                raise ValueError(
                    "modulation layer projections must share output shape "
                    "and format"
                )

            blocks[index].copy_(projected)
            count += 1
            # Release the streamed weight before the iterator yields the next.
            del weight, bias, projected

        if blocks is None or count != layer_count:
            raise ValueError(
                "modulation checkpoint is missing layer projections"
            )
        return cls(
            blocks,
            None if final_projection is None else project(*final_projection),
            groups.to(blocks.device),
        )

    def _entries(self, step: torch.Tensor) -> torch.Tensor:
        if step.dtype != torch.int64 or tuple(step.shape) != (1,):
            raise ValueError("modulation step must be a [1] int64 index")
        return self.groups.index_select(0, step)[0]

    def forward(self, step: torch.Tensor) -> torch.Tensor:
        """Gather every layer's modulation products at one solver step.

        ``step`` is a [1] int64 device index into the prepared schedule
        (``Schedule.step``). The gathers read it on the device, so a captured
        computation serves every step. Returns [layer, group, output]
        products; indices outside the schedule are not checked on the host.
        """
        return self.products.index_select(1, self._entries(step))

    def output(self, step: torch.Tensor) -> torch.Tensor:
        """Gather final-normalization products at one solver step.

        ``step`` is a [1] int64 device index, as for ``forward``. Returns
        [group, output] products.
        """
        if self.output_products is None:
            raise ValueError("this modulation has no output projection")
        return self.output_products.index_select(0, self._entries(step))
