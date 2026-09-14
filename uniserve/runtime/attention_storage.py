"""Allocate bounded attention communication storage before graph capture."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from uniserve.distributed.mesh import Communicator
from uniserve.distributed.peer_memory import allocate_peer_workspace
from uniserve.nn.attention import RadixAttention
from uniserve.nn.attention_storage import ExchangeBuffers
from uniserve.nn.parallel_attention import AttentionBuffers, OutputBuffers, ParallelAttention
from uniserve.runtime.tensor_buffers import TensorBuffers
from uniserve.tensors import BufferConfig


@dataclass(frozen=True)
class AttentionStorage:
    """Retain exchange allocations while exposing only borrowed numerical views.

    The caller keeps this owner alive until its streams, captured graphs, and
    output readers finish. Scope-bound numerical layers only receive ``views``.
    """

    allocations: tuple[TensorBuffers, ...]
    views: Mapping[Communicator, ExchangeBuffers]

    def __post_init__(self) -> None:
        object.__setattr__(self, "views", MappingProxyType(dict(self.views)))


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

    Geometry describes query rows and heads after Ulysses exchange. Each peer
    receives its sequence rows with every head in that logical group. Runtime
    owns registration, synchronization values, and allocation retirement.
    """

    if min(rows, heads, head_dim) < 1:
        raise ValueError("attention output extents must be positive")
    allocations = {}
    bindings = {}
    for layer in layers:
        group = layer.ulysses_group
        if rows % group.world_size:
            raise ValueError("attention output rows must divide Ulysses membership")
        if group not in allocations:
            allocation = TensorBuffers.allocate(
                {
                    "output": BufferConfig(
                        (rows // group.world_size, heads * group.world_size, head_dim), dtype
                    ),
                    "sync_input": BufferConfig((1,), torch.int32),
                    "sync_output": BufferConfig((group.world_size,), torch.int32),
                },
                group.device,
                symmetric={"output": group},
                fill={"sync_input": group.rank_in_group},
            )
            allocations[group] = allocation
        allocation = allocations[group]
        bindings[layer] = OutputBuffers(
            allocation.peers("output"),
            allocation.capacity["sync_input"],
            allocation.capacity["sync_output"],
        )
    return OutputStorage(tuple(allocations.values()), bindings)


def allocate_attention_exchange_storage(
    modules: Iterable[RadixAttention],
    *,
    max_tokens: int,
    dtype: torch.dtype,
) -> AttentionStorage:
    """Share each group's maximum payload capacity across serialized layers.

    Each scope must own an independent stream execution domain. The logical
    row bound includes transport padding; head widths come from loaded module
    geometry. NCCL buffers use symmetric windows; other backends use ordinary
    device storage with the same capacity and lifetime contract.
    """

    widths: dict[Communicator, tuple[int, int]] = {}
    for module in modules:
        group = module.exchange.ulysses_group
        if group.world_size == 1:
            continue
        query, key = widths.get(group, (0, 0))
        widths[group] = (
            max(query, module.num_heads * module.head_dim),
            max(key, module.num_kv_heads * module.head_dim),
        )
    result = {}
    allocations = []
    for group, (query, key) in widths.items():
        rows = (max_tokens + group.world_size - 1) // group.world_size * group.world_size
        symmetric = torch.distributed.get_backend(group._require()) == "nccl"
        schema = {
            f"{role}_{direction}": BufferConfig((rows * width * dtype.itemsize,), torch.uint8)
            for role, width in (("query", query), ("key", key), ("value", key), ("output", query))
            for direction in (
                ("send", "receive") if role == "output" else ("send", "receive", "staging")
            )
        }
        allocation = TensorBuffers.allocate(
            schema, group.device, symmetric={name: group for name in schema} if symmetric else {}
        )
        allocations.append(allocation)
        result[group] = ExchangeBuffers(allocation.capacity)
    return AttentionStorage(tuple(allocations), result)


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
        key = torch.empty((rows * group.world_size, *shape[1:]), dtype=dtype, device=group.device)
        value = torch.empty_like(key)
        begin = group.rank_in_group * rows
        local_key, local_value = key[begin : begin + rows], value[begin : begin + rows]
    return AttentionBuffers(
        key,
        value,
        local_key,
        local_value,
        torch.empty(key.shape[0] // block_size, dtype=torch.int32, device=group.device),
        torch.zeros(1, dtype=torch.int32, device=group.device),
        torch.empty(group.world_size, dtype=torch.int32, device=group.device),
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
        if layer.context_group.world_size == 1:
            continue
        group = layer.key_group
        gathered_rows = rows * (layer.col_group.world_size if layer.col_group is not None else 1)
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
