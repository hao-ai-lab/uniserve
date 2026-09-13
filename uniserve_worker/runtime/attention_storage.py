"""Allocate bounded attention communication storage before graph capture."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from ..nn.attention import RadixAttention
from ..nn.attention_storage import ExchangeBuffers
from ..nn.mesh import Communicator
from ..nn.parallel_attention import AttentionBuffers, OutputBuffers, ParallelAttention
from ..runtime.tensor_buffers import TensorBuffers, TensorSchema
from .peer_memory import allocate_peer_workspace


@dataclass(frozen=True)
class AttentionContextGeometry:
    """Declare the key domain and physical communication used by context attention."""

    group: Communicator
    rows: int
    heads: int
    mapped: bool
    head_dim: int
    dtype: torch.dtype
    block_size: int

    def __post_init__(self) -> None:
        if min(self.rows, self.heads, self.head_dim, self.block_size) < 1:
            raise ValueError("attention context extents must be positive")
        if self.rows % self.block_size:
            raise ValueError("attention context rows must align to its validity blocks")


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
                    "output": TensorSchema(
                        (rows // group.world_size, heads * group.world_size, head_dim),
                        dtype,
                        memory="symmetric",
                        group=group,
                    ),
                    "sync_input": TensorSchema((1,), torch.int32, fill=group.rank_in_group),
                    "sync_output": TensorSchema((group.world_size,), torch.int32),
                },
                group.device,
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
            f"{role}_{direction}": TensorSchema(
                (rows * width * dtype.itemsize,),
                torch.uint8,
                memory="symmetric" if symmetric else "device",
                group=group if symmetric else None,
            )
            for role, width in (("query", query), ("key", key), ("value", key), ("output", query))
            for direction in (
                ("send", "receive") if role == "output" else ("send", "receive", "staging")
            )
        }
        allocation = TensorBuffers.allocate(schema, group.device)
        allocations.append(allocation)
        result[group] = ExchangeBuffers(allocation.capacity)
    return AttentionStorage(tuple(allocations), result)


def allocate_attention_context(geometry: AttentionContextGeometry) -> AttentionBuffers:
    """Allocate context K/V and fences using their actual physical row capacity."""

    group, rows = geometry.group, geometry.rows
    shape = (rows, geometry.heads, geometry.head_dim)
    if geometry.mapped:
        keys = allocate_peer_workspace(
            group,
            shape,
            dtype=geometry.dtype,
            row_multiple=geometry.block_size,
        )
        values = allocate_peer_workspace(
            group,
            shape,
            dtype=geometry.dtype,
            row_multiple=geometry.block_size,
        )
        key, value = keys.global_tensor, values.global_tensor
        local_key, local_value = keys.local, values.local
    else:
        key = torch.empty(
            (rows * group.world_size, *shape[1:]), dtype=geometry.dtype, device=group.device
        )
        value = torch.empty_like(key)
        begin = group.rank_in_group * rows
        local_key, local_value = key[begin : begin + rows], value[begin : begin + rows]
    return AttentionBuffers(
        key,
        value,
        local_key,
        local_value,
        torch.empty(key.shape[0] // geometry.block_size, dtype=torch.int32, device=group.device),
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

    allocations: dict[AttentionContextGeometry, AttentionBuffers] = {}
    bindings = {}
    for layer in layers:
        if layer.context_group.world_size == 1:
            continue
        geometry = AttentionContextGeometry(
            group=layer.key_group,
            rows=rows * (layer.col_group.world_size if layer.col_group is not None else 1),
            heads=heads,
            head_dim=head_dim,
            mapped=layer.mapped,
            dtype=dtype,
            block_size=block_size,
        )
        if geometry not in allocations:
            allocations[geometry] = allocate_attention_context(geometry)
        bindings[layer] = allocations[geometry]
    return MappingProxyType(bindings)
