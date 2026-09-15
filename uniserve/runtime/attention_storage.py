"""Allocate bounded attention communication storage before graph capture."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from uniserve.distributed.mesh import Communicator
from uniserve.nn.attention._parallel import AttentionBuffers, OutputBuffers, ParallelAttention
from uniserve.runtime._peer_memory import allocate_peer_workspace
from uniserve.runtime.tensor_buffers import TensorBuffers
from uniserve.tensors import BufferConfig


@dataclass(frozen=True)
class OutputStorage:
    """Keep symmetric attention destinations alive through their final reader."""

    allocations: tuple[TensorBuffers, ...]
    views: Mapping[ParallelAttention, OutputBuffers]

    def __post_init__(self) -> None:
        object.__setattr__(self, "views", MappingProxyType(dict(self.views)))


def allocate_output_storage(
    layers: Iterable[ParallelAttention],
    *,
    rows: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> OutputStorage:
    """Allocate peer outputs shared only by one caller's serialized layers.

    The dimensions describe query rows and heads after Ulysses exchange. Each peer
    receives its sequence rows with every head in that logical group. Runtime
    owns registration, synchronization values, and allocation retirement.
    """

    if min(rows, heads, head_dim) < 1:
        raise ValueError("attention output extents must be positive")

    allocations = {}
    bindings = {}
    for layer in layers:
        group = layer.ulysses_group
        if rows % group.size:
            raise ValueError("attention output rows must divide Ulysses membership")

        schema = {
            "output": BufferConfig((rows // group.size, heads * group.size, head_dim), dtype),
            "receive": BufferConfig((rows // group.size, heads * group.size, head_dim), dtype),
            "sync_input": BufferConfig((1,), torch.int32),
            "sync_output": BufferConfig((group.size,), torch.int32),
        }
        if group not in allocations:
            allocation = TensorBuffers.allocate(
                schema, device=group.device, symmetric={"output": group, "receive": group}
            )
            # Each rank seeds its own sync slot for the peer rendezvous.
            allocation.view(schema)["sync_input"].fill_(group.rank)
            allocations[group] = allocation

        allocation = allocations[group]
        views = allocation.view(schema)
        bindings[layer] = OutputBuffers(
            allocation.peers("output"),
            views["receive"],
            views["sync_input"],
            views["sync_output"],
        )
    return OutputStorage(tuple(allocations.values()), bindings)


def allocate_attention_context(
    *,
    group: Communicator,
    rows: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    block_size: int,
    mapped: bool,
) -> AttentionBuffers:
    """Allocate context K/V and fences using their actual physical row capacity."""

    if min(rows, heads, head_dim, block_size) < 1:
        raise ValueError("attention context extents must be positive")
    if rows % block_size:
        raise ValueError("attention context rows must align to its validity blocks")

    shape = (rows, heads, head_dim)
    if mapped:
        keys = allocate_peer_workspace(
            group,
            shape,
            dtype=dtype,
            row_multiple=block_size,
        )
        values = allocate_peer_workspace(
            group,
            shape,
            dtype=dtype,
            row_multiple=block_size,
        )
        key, value = keys.global_tensor, values.global_tensor
        local_key, local_value = keys.local, values.local
    else:
        # Replicated compact domain: each rank owns one contiguous row window.
        key = torch.empty((rows * group.size, *shape[1:]), dtype=dtype, device=group.device)
        value = torch.empty_like(key)
        begin = group.rank * rows
        local_key, local_value = key[begin : begin + rows], value[begin : begin + rows]

    # Trailing fields: per-block valid-row counts, local fence, per-peer fences.
    return AttentionBuffers(
        key,
        value,
        local_key,
        local_value,
        torch.empty(key.shape[0] // block_size, dtype=torch.int32, device=group.device),
        torch.zeros(1, dtype=torch.int32, device=group.device),
        torch.empty(group.size, dtype=torch.int32, device=group.device),
    )


def allocate_context_storage(
    layers: Iterable[ParallelAttention],
    *,
    rows: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    block_size: int,
) -> Mapping[ParallelAttention, AttentionBuffers]:
    """Allocate context capacity shared by serialized layers of one caller.

    Rows describe each context partition before any column gather. Mapped
    allocation preserves page padding and peer offsets. Numerical layers only
    borrow the resulting views when the caller enters ``context_scope``.
    """

    # All layers in this call share head dimensions, dtype and block size.
    # Only communicator, gathered row count and mapping can vary per layer.
    allocations: dict[tuple[Communicator, int, bool], AttentionBuffers] = {}
    bindings = {}
    for layer in layers:
        if layer.context_group.size == 1:
            continue

        group = layer.key_group
        gathered_rows = rows * (layer.col_group.size if layer.col_group is not None else 1)
        key = (group, gathered_rows, layer.mapped)
        if key not in allocations:
            allocations[key] = allocate_attention_context(
                group=group,
                rows=gathered_rows,
                heads=heads,
                head_dim=head_dim,
                mapped=layer.mapped,
                dtype=dtype,
                block_size=block_size,
            )

        bindings[layer] = allocations[key]

    return MappingProxyType(bindings)
