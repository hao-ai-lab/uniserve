"""Allocate bounded attention communication storage before graph capture."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from ..execution.bounded_storage import BoundedTensorStorage, TensorSchema
from ..nn.attention import RadixAttention
from ..nn.attention_storage import AttentionExchangeStorage
from ..nn.mesh import Communicator
from .distributed import DistributedEnvironment


def allocate_attention_exchange_storage(
    modules: Iterable[RadixAttention],
    environment: DistributedEnvironment,
    *,
    max_tokens: int,
    dtype: torch.dtype,
    scope: tuple[object, ...],
) -> dict[Communicator, AttentionExchangeStorage]:
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
        symmetric = environment.backend == "nccl"
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
        allocation = BoundedTensorStorage.allocate(
            schema, group.device, environment=environment, layout=scope
        )
        result[group] = AttentionExchangeStorage(allocation.capacity)
    return result
