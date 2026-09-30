"""Exact key validity for FlashAttention-4 over 64-row attention blocks."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from flash_attn.cute.utils import scalar_to_ssa


@cute.jit
def block_validity(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    q_idx: cute.TensorSSA,
    kv_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors: list,
) -> cute.TensorSSA:
    """Keep the first ``valid_sizes[tile]`` keys of each 64-row block.

    FlashAttention calls this predicate for each score element. The borrowed
    auxiliary tensor is a contiguous int32 vector of live row counts, shared
    across heads and queries. Bounds protect its final partial kernel tile.
    """
    del batch, head, q_idx, seqlen_info
    valid_sizes = aux_tensors[0]
    tile = kv_idx[0] // 64
    valid = cutlass.Int32(0)
    if tile < valid_sizes.shape[0]:
        valid = valid_sizes[tile]
    return kv_idx % 64 < scalar_to_ssa(valid, cutlass.Int32)


__all__ = ["block_validity"]
