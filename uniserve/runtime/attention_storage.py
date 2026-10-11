"""Allocate bounded attention communication storage before graph capture."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType

import torch

from uniserve.distributed.mesh import Communicator
from uniserve.nn.attention._parallel import (
    AttentionBuffers,
    ParallelAttention,
)


def allocate_attention_context(
    *,
    group: Communicator,
    rows: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> AttentionBuffers:
    """Allocate the compact context K/V domain.

    Every rank holds the whole context's rows: its own window is where it
    publishes, and the gather fills the rest.
    """
    if min(rows, heads, head_dim) < 1:
        raise ValueError("attention context extents must be positive")

    shape = (rows, heads, head_dim)
    # Replicated compact domain: each rank owns one contiguous row window.
    key = torch.empty(
        (rows * group.size, *shape[1:]), dtype=dtype, device=group.device
    )
    value = torch.empty_like(key)
    begin = group.rank * rows
    local_key, local_value = (
        key[begin : begin + rows],
        value[begin : begin + rows],
    )
    return AttentionBuffers(key, value, local_key, local_value)


def allocate_context_storage(
    layers: Iterable[ParallelAttention],
    *,
    rows: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> Mapping[ParallelAttention, AttentionBuffers]:
    """Allocate context capacity shared by serialized layers of one caller.

    Rows describe each context partition. Numerical layers only borrow the
    resulting views when the caller enters ``context_scope``.
    """
    # All layers in this call share head dimensions and dtype.
    # Only the communicator and the row count can vary per layer.
    allocations: dict[tuple[Communicator, int], AttentionBuffers] = {}
    bindings = {}
    for layer in layers:
        if layer.context_group.size == 1:
            continue

        group = layer.context_group
        key = (group, rows)
        if key not in allocations:
            allocations[key] = allocate_attention_context(
                group=group,
                rows=rows,
                heads=heads,
                head_dim=head_dim,
                dtype=dtype,
            )

        bindings[layer] = allocations[key]

    return MappingProxyType(bindings)
