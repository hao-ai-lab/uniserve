"""Small reusable block-id allocators for runtime-owned cache pools."""
from __future__ import annotations

import bisect
from collections.abc import Iterable

__all__ = ["BlockFreeList"]


class BlockFreeList:
    """Sorted unique free-list for integer block ids."""

    def __init__(self, num_blocks: int = 0) -> None:
        self._free = list(range(max(0, int(num_blocks))))

    @property
    def available(self) -> int:
        return len(self._free)

    def reset(self, num_blocks: int) -> None:
        self._free = list(range(max(0, int(num_blocks))))

    def allocate(self, count: int, *, label: str = "block pool") -> list[int]:
        count = int(count)
        if count <= 0:
            return []
        if len(self._free) < count:
            raise RuntimeError(f"{label} exhausted: need {count}, have {len(self._free)}")
        out = self._free[:count]
        del self._free[:count]
        return out

    def release(self, block_ids: Iterable[int]) -> None:
        for block_id in block_ids:
            block_id = int(block_id)
            position = bisect.bisect_left(self._free, block_id)
            if position < len(self._free) and self._free[position] == block_id:
                continue
            self._free.insert(position, block_id)
