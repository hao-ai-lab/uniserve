"""Precomputed diffusion modulation plans over fixed timestep ladders."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

__all__ = [
    "modulation_plan_shapes",
    "prepare_modulation_plan",
    "select_modulation_step",
]


def modulation_plan_shapes(
    layer_projections: Sequence[nn.Module],
    final_projection: nn.Module,
    *,
    steps: int,
    input_rows: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Derive resident block and final-projection storage shapes for a timestep ladder."""

    if not layer_projections or steps < 1 or input_rows < 1:
        raise ValueError(
            "modulation planning requires projections, steps, and input rows"
        )
    widths = tuple(
        int(getattr(projection, "out_features")) for projection in layer_projections
    )
    if len(set(widths)) != 1:
        raise ValueError("modulation layer projections must have one output width")
    final_width = int(getattr(final_projection, "out_features"))
    return (
        (int(steps), len(layer_projections), int(input_rows), widths[0]),
        (int(steps), int(input_rows), final_width),
    )


@torch.inference_mode()
def prepare_modulation_plan(
    activated_timesteps: torch.Tensor,
    layer_projections: Sequence[nn.Module],
    final_projection: nn.Module,
    block_storage: torch.Tensor,
    final_storage: torch.Tensor,
) -> None:
    """Project all activated timesteps into caller-owned per-step modulation storage."""

    expected_block, expected_final = modulation_plan_shapes(
        layer_projections,
        final_projection,
        steps=int(block_storage.shape[0]),
        input_rows=int(block_storage.shape[2]),
    )
    if (
        tuple(block_storage.shape) != expected_block
        or tuple(final_storage.shape) != expected_final
    ):
        raise ValueError("caller-provided modulation storage has incompatible geometry")
    flat_rows = expected_block[0] * expected_block[2]
    if activated_timesteps.ndim != 2 or activated_timesteps.shape[0] != flat_rows:
        raise ValueError("activated timestep ladder has incompatible geometry")
    for layer, projection in enumerate(layer_projections):
        projected = projection(activated_timesteps.to(projection.weight.dtype))
        block_storage[:, layer].copy_(projected.view_as(block_storage[:, layer]))
    final_storage.copy_(
        final_projection(activated_timesteps.to(final_projection.weight.dtype)).view_as(
            final_storage
        )
    )


@torch.inference_mode()
def select_modulation_step(
    block_plan: torch.Tensor,
    final_plan: torch.Tensor,
    step: int,
    block_output: torch.Tensor,
    final_output: torch.Tensor,
) -> None:
    """Copy one prepared timestep slice into fixed execution output buffers."""

    index = int(step)
    if (
        not 0 <= index < block_plan.shape[0]
        or final_plan.shape[0] != block_plan.shape[0]
    ):
        raise ValueError("modulation step is outside the prepared ladder")
    block_output.copy_(block_plan[index])
    final_output.copy_(final_plan[index])
