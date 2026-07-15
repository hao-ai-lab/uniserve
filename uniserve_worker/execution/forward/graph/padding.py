"""Reserved paged-KV storage for CUDA-graph padding rows."""

from __future__ import annotations

from typing import Any

from ....foundation.errors import invalid_descriptor

__all__ = ["decode_graph_padding_block_ids"]


def decode_graph_padding_block_ids(pool: Any) -> tuple[int, ...]:
    block_ids = tuple(int(block_id) for block_id in getattr(pool, "reserved_block_ids", ()))
    if len(set(block_ids)) != len(block_ids):
        raise invalid_descriptor("decode graph padding block ids must be unique")
    num_blocks = int(getattr(pool, "num_blocks", 0) or 0)
    if any(block_id < 0 or block_id >= num_blocks for block_id in block_ids):
        raise invalid_descriptor("decode graph padding block id is out of range")
    return block_ids
