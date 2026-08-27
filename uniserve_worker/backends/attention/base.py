"""Attention backend interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch

from ...execution.forward_batch import ForwardBatch
from ..triton import triton_available

try:  # pragma: no cover - availability depends on the serving environment.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

__all__ = [
    "AttentionCapabilities",
    "AttentionBackend",
    "PagedAttentionBackend",
    "SegmentedAttentionBackend",
    "VarlenAttentionBackend",
    "VisibleEndAttentionBackend",
    "merge_attention_states",
]


if triton is not None:

    @triton.jit
    def _merge_attention_states_kernel(
        first_output_ptr,
        first_lse_ptr,
        second_output_ptr,
        second_lse_ptr,
        output_ptr,
        merged_lse_ptr,
        state_count,
        head_dim: tl.constexpr,
        block_rows: tl.constexpr,
        block_dim: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
        columns = tl.arange(0, block_dim)
        row_mask = rows < state_count
        first_lse = tl.load(first_lse_ptr + rows, mask=row_mask, other=-float("inf"))
        second_lse = tl.load(second_lse_ptr + rows, mask=row_mask, other=-float("inf"))
        maximum = tl.maximum(first_lse, second_lse)
        both_empty = (first_lse == -float("inf")) & (second_lse == -float("inf"))
        first_scale = tl.exp(first_lse - maximum)
        second_scale = tl.exp(second_lse - maximum)
        denominator = first_scale + second_scale
        first_weight = tl.where(both_empty, 0.0, first_scale / denominator)
        second_weight = tl.where(both_empty, 0.0, second_scale / denominator)
        merged_lse = tl.where(both_empty, -float("inf"), maximum + tl.log(denominator))

        offsets = rows[:, None] * head_dim + columns[None, :]
        mask = row_mask[:, None] & (columns[None, :] < head_dim)
        first_output = tl.load(first_output_ptr + offsets, mask=mask, other=0.0)
        second_output = tl.load(second_output_ptr + offsets, mask=mask, other=0.0)
        merged = (
            first_output.to(tl.float32) * first_weight[:, None]
            + second_output.to(tl.float32) * second_weight[:, None]
        )
        tl.store(output_ptr + offsets, merged, mask=mask)
        tl.store(merged_lse_ptr + rows, merged_lse, mask=row_mask)


def merge_attention_states(
    first_output: torch.Tensor,
    first_lse: torch.Tensor,
    second_output: torch.Tensor,
    second_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable online-softmax merge for independently evaluated KV segments."""

    if first_output.shape != second_output.shape or first_lse.shape != second_lse.shape:
        raise ValueError("attention states must have matching shapes")
    if _triton_merge_eligible(first_output, first_lse, second_output, second_lse):
        output = torch.empty_like(first_output)
        merged_lse = torch.empty_like(first_lse)
        state_count = int(first_lse.numel())
        head_dim = int(first_output.shape[-1])
        block_dim = triton.next_power_of_2(head_dim)
        block_rows = max(1, min(8, 1024 // block_dim))
        _merge_attention_states_kernel[(triton.cdiv(state_count, block_rows),)](
            first_output,
            first_lse,
            second_output,
            second_lse,
            output,
            merged_lse,
            state_count,
            head_dim,
            block_rows,
            block_dim,
            num_warps=8,
        )
        return output, merged_lse
    merged_lse = torch.logaddexp(first_lse, second_lse)
    first_weight = torch.exp(first_lse - merged_lse).nan_to_num(0.0)
    second_weight = torch.exp(second_lse - merged_lse).nan_to_num(0.0)
    output = (
        first_output.float() * first_weight.unsqueeze(-1)
        + second_output.float() * second_weight.unsqueeze(-1)
    ).to(first_output.dtype)
    return output, merged_lse


def _triton_merge_eligible(
    first_output: torch.Tensor,
    first_lse: torch.Tensor,
    second_output: torch.Tensor,
    second_lse: torch.Tensor,
) -> bool:
    tensors = (first_output, first_lse, second_output, second_lse)
    return bool(
        triton is not None
        and not torch.is_grad_enabled()
        and first_output.ndim == first_lse.ndim + 1
        and tuple(first_output.shape[:-1]) == tuple(first_lse.shape)
        and int(first_output.shape[-1]) > 0
        and int(first_lse.numel()) > 0
        and all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors)
        and len({tensor.device for tensor in tensors}) == 1
        and first_output.dtype == second_output.dtype
        and first_lse.dtype == second_lse.dtype
        and first_output.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and first_lse.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and triton_available(first_output.device)
    )


@dataclass(frozen=True)
class AttentionCapabilities:
    # Whether the provider's runtime dependency is currently usable. Optional
    # provider modules may still register an unavailable backend so explicit
    # selection and diagnostics can name it, but dispatch must not route generic
    # dense/paged attention through a backend whose kernel import failed.
    available: bool = True
    paged_kv: bool = False
    varlen_attention: bool = False
    varlen_paged_kv: bool = False
    # True when ``forward_varlen`` only supports paged KV cache inputs and must
    # not be selected for contiguous q/k/v varlen prefill.
    requires_paged_varlen: bool = False
    visible_end: bool = False
    segmented_attention: bool = False
    segmented_attention_cuda_graph: bool = False
    paged_block_size_multiple: int = 1
    min_head_dim: int = 1
    # The backend's paged-KV kernel only supports single-token decode (one query
    # token per row), so it must not be dispatched for multi-token paged prefill.
    # Backends whose paged kernel handles arbitrary query lengths leave this False.
    paged_decode_only: bool = False
    # The backend's paged-varlen prefill path can be captured directly in a CUDA
    # graph because it consumes live tensor inputs and does not bake mutable host
    # wrapper plan state that another request can later overwrite.
    paged_varlen_cuda_graph: bool = False
    # The backend's paged visible-end path consumes only live tensors and is
    # safe to capture for mixed causal/bidirectional segment compositions.
    visible_end_cuda_graph: bool = False
    # (q, k, v) head-dim geometries the backend's kernel can run. Empty means the
    # backend imposes no fixed-geometry restriction (the common case); a non-empty
    # set declares the exact tuples a geometry-restricted kernel (e.g. fa4_cute's
    # unified trunk path) accepts, so callers and the registry can pre-emptively
    # avoid dispatching shapes the kernel would hard-reject. This is the single
    # authoritative source for that table.
    trunk_geometries: frozenset[tuple[int, int, int]] = field(default_factory=frozenset)
    # Dense-forward constraints. ``cuda_only`` and ``min_cuda_capability`` also
    # apply to every other regime; ``dense_ranks`` is the set of accepted q/k/v
    # ndims for the contiguous dense path; ``accepts_dense_mask`` is whether
    # ``DenseAttention.attn_mask`` may be non-None.
    cuda_only: bool = False
    min_cuda_capability: tuple[int, int] | None = None
    dense_ranks: frozenset[int] = field(default_factory=lambda: frozenset({3, 4}))
    accepts_dense_mask: bool = False

    def supports_trunk_geometry(self, q_head_dim: int, k_head_dim: int, v_head_dim: int) -> bool:
        """Whether the backend kernel accepts this exact ``(q, k, v)`` geometry.

        An empty :attr:`trunk_geometries` means the backend is not
        geometry-restricted and accepts any shape it is otherwise capable of.
        """
        if not self.trunk_geometries:
            return True
        return (int(q_head_dim), int(k_head_dim), int(v_head_dim)) in self.trunk_geometries


class AttentionBackend(Protocol):
    """Universal attention-backend contract.

    Only the members declared here are mandatory for *every* registered
    backend. The paged- and
    varlen-specific entry points are intentionally *not* part of this base
    Protocol because they are optional and capability-gated: a backend
    implements ``forward_paged`` only when it advertises
    ``capabilities().paged_kv`` and ``forward_varlen`` only when it advertises
    ``capabilities().varlen_attention``. Callers must narrow via the
    corresponding capability flag (or the ``PagedAttentionBackend`` /
    ``VarlenAttentionBackend`` Protocols below) before invoking those methods.
    """

    name: str

    def capabilities(self) -> AttentionCapabilities: ...

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        scale: float,
        attn_mask: torch.Tensor | None = None,
        context: ForwardBatch | None = None,
    ) -> torch.Tensor: ...


class PagedAttentionBackend(AttentionBackend, Protocol):
    """Backends that support paged KV decode.

    Implemented only by backends reporting ``capabilities().paged_kv`` (e.g.
    ``flash_attn``, ``fa4_cute``, ``flashinfer``, ``sgl_kernel``). ``paged_kv``
    being ``True`` is the precondition for calling ``forward_paged``.
    """

    def forward_paged(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        causal: bool,
        scale: float,
        context: ForwardBatch | None = None,
    ) -> torch.Tensor: ...


class VarlenAttentionBackend(AttentionBackend, Protocol):
    """Backends that support variable-length (cu_seqlens) prefill.

    Implemented only by backends reporting ``capabilities().varlen_attention``
    ``varlen_attention`` being ``True`` is the precondition for calling
    ``forward_varlen``.
    """

    def forward_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool,
        scale: float,
        block_table: torch.Tensor | None = None,
        context: ForwardBatch | None = None,
    ) -> torch.Tensor: ...


class VisibleEndAttentionBackend(AttentionBackend, Protocol):
    """Backends that support the hybrid ``visible_end`` mask path.

    Implemented only by backends reporting ``capabilities().visible_end``.
    Callers must check the capability before invoking ``forward_visible_end``.

    ``q`` may be fixed ``[B, L, H, D]`` or varlen ``[total, H, D]``;
    ``visible_end`` is padded ``[B, max_q]`` and indexed locally per sequence.
    """

    def forward_visible_end(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        visible_end: torch.Tensor,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        page_table: torch.Tensor | None = None,
        seqused_k: torch.Tensor | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        scale: float | None = None,
        use_prefix_bounds: bool = False,
        fully_visible: bool = False,
        context: ForwardBatch | None = None,
    ) -> torch.Tensor: ...


class SegmentedAttentionBackend(AttentionBackend, Protocol):
    """Backends that compose paged-prefix and dense-current attention state."""

    def forward_segmented(
        self,
        q: torch.Tensor,
        current_k: torch.Tensor,
        current_v: torch.Tensor,
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
        *,
        page_table: torch.Tensor,
        prefix_lens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        visible_current_end: torch.Tensor,
        scale: float,
        fully_visible_current: bool,
        context: ForwardBatch | None = None,
    ) -> torch.Tensor: ...
