"""System-owned physical KV residency provisioned from immutable specs."""

from __future__ import annotations

import torch

from ..capabilities import EngineCaps
from ..foundation.runtime_config import graph_padding_block_count
from ..foundation.sizing import ceil_div
from ..spec import ModelSpec, ResourcePlan
from .kv_pool import PagedKVPool

__all__ = ["ResidencyStore"]


class ResidencyStore:
    """Own the physical KV storage for one ready worker."""

    def __init__(self, *, kv: PagedKVPool | None) -> None:
        self.kv = kv

    @classmethod
    def from_spec(
        cls,
        spec: ModelSpec,
        capabilities: EngineCaps,
        resources: ResourcePlan,
        *,
        device: str,
    ) -> ResidencyStore:
        """Provision all physical KV storage from resolved declarations.

        Leased capacity and transaction-branch capacity are ranges of one
        storage tensor, so any batch can attend over both at once.
        """

        if resources.kv_block is None and resources.scratch is None:
            return cls(kv=None)

        cache = spec.cache
        block_size = int(capabilities.block_size)
        reserved = graph_padding_block_count(block_size)
        branch_blocks = 0
        if resources.scratch is not None:
            branch_blocks = ceil_div(int(capabilities.scratch_capacity_tokens), block_size)
            if branch_blocks < 1:
                raise ValueError("declared scratch residency has zero physical capacity")
        kv = PagedKVPool(
            num_layers=int(cache.num_layers),
            num_blocks=int(capabilities.num_blocks) + branch_blocks + reserved,
            block_size=block_size,
            num_kv_heads=int(cache.num_kv_heads),
            head_dim=int(cache.head_dim),
            device=device,
            dtype=_torch_dtype(cache.dtype),
            store_dtype=cache.store_dtype,
            reserved_tail_blocks=reserved,
            branch_blocks=branch_blocks,
        )
        return cls(kv=kv)


def _torch_dtype(name: str) -> torch.dtype:
    value = getattr(torch, str(name).removeprefix("torch."), None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unsupported cache dtype {name!r}")
    return value
