"""Defines the per-query visible-end mask for jagged forward attention."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute

try:  # pragma: no cover - optional FlashAttention CUTE runtime dependency.
    from flash_attn.cute.utils import scalar_to_ssa  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "FlashAttention CUTE utils are unavailable. Install a provider package "
        "that exposes flash_attn.cute.utils.scalar_to_ssa."
    ) from exc


@cute.jit
def hybrid_multimodal_mask(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    q_idx: cute.TensorSSA,
    kv_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors: list,
    aux_scalars: list,
) -> cute.TensorSSA:
    """Keep key positions below the visible-end limit for each batch/query pair."""

    del head, seqlen_info, aux_scalars
    visible_end = aux_tensors[0]
    limit = scalar_to_ssa(visible_end[batch[0], q_idx[0]], cutlass.Int32)
    return kv_idx < limit


__all__ = ["hybrid_multimodal_mask"]
