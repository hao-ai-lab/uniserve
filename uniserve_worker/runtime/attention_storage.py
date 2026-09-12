"""Allocate bounded attention communication storage before graph capture."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from ..nn.attention import RadixAttention
from ..nn.attention_storage import ExchangeBuffers
from ..nn.mesh import Communicator
from ..nn.parallel_attention import AttentionBuffers, AttentionContextGeometry
from ..runtime.tensor_buffers import TensorBuffers, TensorSchema
from .peer_memory import allocate_peer_workspace


def allocate_attention_exchange_storage(
    modules: Iterable[RadixAttention],
    *,
    max_tokens: int,
    dtype: torch.dtype,
) -> dict[Communicator, ExchangeBuffers]:
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
        result[group] = ExchangeBuffers(allocation)
    return result


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
