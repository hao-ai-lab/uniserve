"""Per-query visible-end mask for FlashAttention-4 CuTe attention.

``uniserve.runtime.backends.attention.flash_attn_4`` passes
:func:`visible_end_mask` as the FlashAttention-4 ``mask_mod`` together with
one auxiliary tensor: an int32 ``[batch, query rows]`` table whose entry is
the exclusive end of the key range each sequence-local query row may see.
The mask is independent of the attention head.
"""

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
def visible_end_mask(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    q_idx: cute.TensorSSA,
    kv_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors: list,
) -> cute.TensorSSA:
    """Keep key positions below the visible-end limit per batch/query pair.

    The signature is FlashAttention-4's ``mask_mod`` callback. The function
    declares no ``__vec_size__``, so the kernel calls it for one score element
    at a time with shape-(1,) index vectors; element 0 of ``batch`` and
    ``q_idx`` addresses ``aux_tensors[0]``. Returns the keep predicate
    ``kv_idx < visible_end[batch, q_idx]``.
    """
    del head, seqlen_info
    visible_end = aux_tensors[0]
    limit = scalar_to_ssa(visible_end[batch[0], q_idx[0]], cutlass.Int32)
    return kv_idx < limit


@cute.jit
def paged_causal_mask(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    q_idx: cute.TensorSSA,
    kv_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors: list,
) -> cute.TensorSSA:
    """Apply a sequence's live causal flag over its paged prefix and block.

    Auxiliary columns are int32 causal flags and absolute prefix lengths.
    Non-causal rows see the entire block; causal rows include their own key.
    The attention kernel separately masks positions past each live length.
    """
    del head, seqlen_info
    causal = scalar_to_ssa(aux_tensors[0][batch[0]], cutlass.Int32)
    prefix = scalar_to_ssa(aux_tensors[1][batch[0]], cutlass.Int32)
    return (causal == 0) | (kv_idx <= prefix + q_idx)


__all__ = ["paged_causal_mask", "visible_end_mask"]
