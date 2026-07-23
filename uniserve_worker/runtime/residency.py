"""System-owned physical KV residency provisioned from immutable specs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from ..capabilities import EngineCaps
from ..foundation.runtime_config import decode_graph_padding_block_count
from ..foundation.sizing import ceil_div
from ..spec import ModelSpec, ResourcePlan
from .block_allocator import BlockFreeList
from .kv_pool import PagedKVPool

__all__ = ["ResidencyStore", "ScratchKvPool"]


class ScratchKvPool(PagedKVPool):
    """Paged KV storage with an internal allocator for transaction scratch."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._allocator = BlockFreeList(self.schedulable_num_blocks)

    def allocate_blocks(self, count: int) -> list[int]:
        return self._allocator.allocate(int(count), label="scratch KV pool")

    def release_blocks(self, block_ids: Iterable[int]) -> None:
        values = tuple(int(value) for value in block_ids)
        if any(value < 0 or value >= self.schedulable_num_blocks for value in values):
            raise RuntimeError("scratch KV pool cannot release an out-of-range block")
        self._allocator.release(values)


class ResidencyStore:
    """Own the physical request and scratch KV pools for one ready worker."""

    def __init__(
        self,
        *,
        kv: PagedKVPool | None,
        scratch: ScratchKvPool | None,
    ) -> None:
        self.kv = kv
        self.scratch = scratch

    @classmethod
    def from_spec(
        cls,
        spec: ModelSpec,
        capabilities: EngineCaps,
        resources: ResourcePlan,
        *,
        device: str,
    ) -> ResidencyStore:
        """Provision all physical KV storage from resolved declarations."""

        if resources.kv_block is None and resources.scratch is None:
            return cls(kv=None, scratch=None)

        cache = spec.cache
        block_size = int(capabilities.block_size)
        reserved = decode_graph_padding_block_count(block_size)
        kv = PagedKVPool(
            num_layers=int(cache.num_layers),
            num_blocks=int(capabilities.num_blocks) + reserved,
            block_size=block_size,
            num_kv_heads=int(cache.num_kv_heads),
            head_dim=int(cache.head_dim),
            device=device,
            dtype=_torch_dtype(cache.dtype),
            store_dtype=cache.store_dtype,
            reserved_tail_blocks=reserved,
        )
        scratch = None
        if resources.scratch is not None:
            scratch_blocks = ceil_div(
                int(capabilities.scratch_capacity_tokens),
                block_size,
            )
            if scratch_blocks < 1:
                raise ValueError("declared scratch residency has zero physical capacity")
            scratch = ScratchKvPool(
                num_layers=int(cache.num_layers),
                num_blocks=scratch_blocks + reserved,
                block_size=block_size,
                num_kv_heads=int(cache.num_kv_heads),
                head_dim=int(cache.head_dim),
                device=device,
                dtype=_torch_dtype(cache.dtype),
                store_dtype=cache.store_dtype,
                reserved_tail_blocks=reserved,
            )
        return cls(kv=kv, scratch=scratch)


def _torch_dtype(name: str) -> torch.dtype:
    value = getattr(torch, str(name).removeprefix("torch."), None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unsupported cache dtype {name!r}")
    return value
