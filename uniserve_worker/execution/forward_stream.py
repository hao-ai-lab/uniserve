"""Side-table builder for packed multimodal forward streams."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Literal

import torch

from ..backends.paged_kv_math import paged_kv_write
from ..contracts.forward_mode import ForwardMode
from ..foundation.errors import invalid_descriptor
from ..runtime.cache_protocols import KVCacheView
from ..runtime.kv_pool import PagedKVPool

__all__ = [
    'SegmentClass',
    'SegmentModality',
    'VisiblePolicy',
    'SegmentClassId',
    'SegmentModalityId',
    'VisiblePolicyKind',
    'ForwardSegmentSpec',
    'ForwardStream',
    'ForwardPagedKVSegment',
    'ForwardPagedKVView',
    'ForwardStreamBuilder',
    'build_text_position_indexes',
]

SegmentClass = Literal["extend", "decode", "denoise", "reencode"]
SegmentModality = Literal["und", "gen"]
VisiblePolicy = Literal["causal", "bidirectional"]


class SegmentClassId(IntEnum):
    """Canonical integer encoding for a forward segment class.

    Member names are the wire/string ``SegmentClass`` labels; the integer
    values are the single source of truth for any kernel that needs to pack the
    class as an int. The builder validates string inputs against the member
    names (see ``add_segment``).
    """

    extend = 0
    decode = 1
    denoise = 2
    reencode = 3


class SegmentModalityId(IntEnum):
    """Canonical integer encoding for a forward segment modality."""

    und = 0
    gen = 1


class VisiblePolicyKind(str, Enum):
    """Attention visibility policy for a forward segment.

    Member values are the wire/string ``VisiblePolicy`` labels; each member
    owns the rule that maps ``(prefix_len, q_len)`` to its per-token visible-end
    extent, keeping the build logic out of ``ForwardStreamBuilder.build``.
    """

    causal = "causal"
    bidirectional = "bidirectional"

    def __str__(self) -> str:
        return self.value

    def visible_end(self, prefix_len: int, q_len: int, device: torch.device) -> torch.Tensor:
        """Per-token visible key extent for this policy."""

        if self is VisiblePolicyKind.causal:
            return prefix_len + torch.arange(
                1, q_len + 1, dtype=torch.int32, device=device
            )
        return torch.full((q_len,), prefix_len + q_len, dtype=torch.int32, device=device)


@dataclass(frozen=True)
class ForwardSegmentSpec:
    """One ragged segment in a packed multimodal forward stream."""

    op_index: int
    req_id: int
    kind: str
    mode: ForwardMode
    modality: SegmentModality
    segment_class: SegmentClass
    q_len: int
    prefix_len: int
    branch_id: int = 0
    visible_policy: VisiblePolicy = "causal"
    indexes: torch.Tensor | None = None


@dataclass(frozen=True)
class ForwardStream:
    """Packed attention side-table for multimodal forwards.

    The paged visible-end path reads ``cu_seqlens_q``, ``visible_end``, and
    ``indexes``; K extents come from the paged kv_view, not this table.
    ``fully_visible`` means every row can attend its whole effective K extent,
    so the attention backend can skip visible-end masking entirely.
    """

    segments: tuple[ForwardSegmentSpec, ...]
    cu_seqlens_q: torch.Tensor
    visible_end: torch.Tensor
    indexes: torch.Tensor
    fully_visible: bool = False
    und_indices: torch.Tensor | None = None
    gen_indices: torch.Tensor | None = None


@dataclass(frozen=True)
class ForwardPagedKVSegment:
    block_ids: tuple[int, ...]
    base_len: int
    q_len: int
    # Physical write flag for the current forward span. Denoise rows use this as
    # a transient in-pool staging write so visible-end attention can see current
    # image-token K/V without making those tokens part of the persistent cache.
    write_kv: bool = True
    # Whether this segment's current span is allowed to become part of the owning
    # request/cache logical length after the forward. Text segments persist; image
    # denoise segments remain transient and leave their conditioning cache length
    # unchanged.
    persist_kv: bool = True
    branch_id: int = 0


class ForwardPagedKVView:
    """One-pool paged-KV view for ragged visible-end attention segments."""

    def __init__(
        self,
        pool: PagedKVPool,
        segments: list[ForwardPagedKVSegment] | tuple[ForwardPagedKVSegment, ...],
    ) -> None:
        if not segments:
            raise invalid_descriptor("forward paged KV view requires at least one segment")
        self.pool = pool
        self.segments = tuple(
            ForwardPagedKVSegment(
                block_ids=tuple(pool.validate_block_ids(seg.block_ids)),
                base_len=int(seg.base_len),
                q_len=int(seg.q_len),
                write_kv=bool(seg.write_kv),
                persist_kv=bool(seg.persist_kv),
                branch_id=int(seg.branch_id),
            )
            for seg in segments
        )
        for seg in self.segments:
            if seg.base_len < 0:
                raise invalid_descriptor("forward paged segment base_len must be non-negative")
            if seg.q_len <= 0:
                raise invalid_descriptor("forward paged segment q_len must be positive")
            if seg.persist_kv and not seg.write_kv:
                raise invalid_descriptor("persistent forward paged segments must write current K/V")
            end = seg.base_len + seg.q_len
            required = end if seg.write_kv else seg.base_len
            if required > len(seg.block_ids) * pool.block_size:
                raise invalid_descriptor("forward paged segment blocks do not cover current append")
        self._block_table_cache: dict[torch.device, torch.Tensor] = {}
        self._cache_seqlens_before_cache: dict[torch.device, torch.Tensor] = {}
        self._cache_seqlens_after_cache: dict[torch.device, torch.Tensor] = {}
        self._cu_seqlens_after_cache: dict[torch.device, torch.Tensor] = {}
        self._persistent_cache_seqlens_after_cache: dict[torch.device, torch.Tensor] = {}
        self._write_plan_cache: dict[
            torch.device,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
        ] = {}

    def _target_device(self, device: torch.device | str | None = None) -> torch.device:
        return torch.device(device if device is not None else self.pool.k.device)

    def block_table(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        target = self._target_device(device)
        cached = self._block_table_cache.get(target)
        if cached is not None:
            return cached
        max_blocks = max(len(seg.block_ids) for seg in self.segments)
        out = torch.zeros(
            (len(self.segments), max_blocks),
            dtype=torch.int32,
            device=target,
        )
        for row, seg in enumerate(self.segments):
            if seg.block_ids:
                out[row, : len(seg.block_ids)] = torch.tensor(
                    seg.block_ids,
                    dtype=torch.int32,
                    device=out.device,
                )
        self._block_table_cache[target] = out
        return out

    def cache_seqlens_before(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        return self._cached_int_vector(
            self._cache_seqlens_before_cache,
            (seg.base_len for seg in self.segments),
            device=device,
        )

    def cache_seqlens_after(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        return self._cached_int_vector(
            self._cache_seqlens_after_cache,
            (seg.base_len + seg.q_len for seg in self.segments),
            device=device,
        )

    def cu_seqlens_after(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        target = self._target_device(device)
        cached = self._cu_seqlens_after_cache.get(target)
        if cached is not None:
            return cached
        lengths = self.cache_seqlens_after(device=target)
        out = torch.empty((int(lengths.numel()) + 1,), dtype=torch.int32, device=target)
        out[0] = 0
        out[1:] = torch.cumsum(lengths, dim=0)
        self._cu_seqlens_after_cache[target] = out
        return out

    def persistent_cache_seqlens_after(
        self,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        return self._cached_int_vector(
            self._persistent_cache_seqlens_after_cache,
            (
                seg.base_len + seg.q_len if seg.persist_kv else seg.base_len
                for seg in self.segments
            ),
            device=device,
        )

    def _cached_int_vector(
        self,
        cache: dict[torch.device, torch.Tensor],
        values: Iterable[int],
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        target = self._target_device(device)
        cached = cache.get(target)
        if cached is not None:
            return cached
        out = torch.tensor(list(values), dtype=torch.int32, device=target)
        cache[target] = out
        return out

    def append_packed(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        if k.shape != v.shape:
            raise invalid_descriptor("forward packed KV key/value shapes must match")
        if k.ndim != 3:
            raise invalid_descriptor("forward packed KV append expects [total_tokens, heads, dim]")
        expected = sum(seg.q_len for seg in self.segments)
        if int(k.shape[0]) != expected:
            raise invalid_descriptor(
                f"forward packed KV has {int(k.shape[0])} tokens, expected {expected}"
            )
        if not getattr(self.pool, "is_quantized", False):
            k_cache, v_cache = self.pool.layer_cache(layer)
            page_ids, offsets, token_indices = self._write_plan(device=k.device)
            k_src = k if token_indices is None else k.index_select(0, token_indices)
            v_src = v if token_indices is None else v.index_select(0, token_indices)
            paged_kv_write(
                k_cache,
                v_cache,
                page_ids,
                offsets,
                k_src,
                v_src,
                cast=k_src.dtype != k_cache.dtype or v_src.dtype != v_cache.dtype,
            )
            return
        offset = 0
        for seg in self.segments:
            end = offset + seg.q_len
            if seg.write_kv:
                self.pool.write(
                    layer,
                    list(seg.block_ids),
                    start=seg.base_len,
                    k=k[offset:end],
                    v=v[offset:end],
                )
            offset = end

    def _write_plan(
        self,
        *,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        target = self._target_device(device)
        cached = self._write_plan_cache.get(target)
        if cached is not None:
            return cached

        page_ids: list[int] = []
        offsets: list[int] = []
        token_indices: list[int] = []
        flat = 0
        all_written = True
        for seg in self.segments:
            if not seg.write_kv:
                all_written = False
                flat += seg.q_len
                continue
            for local in range(seg.q_len):
                position = int(seg.base_len) + local
                block_slot = position // self.pool.block_size
                if block_slot >= len(seg.block_ids):
                    raise invalid_descriptor("forward paged segment blocks do not cover current append")
                page_ids.append(int(seg.block_ids[block_slot]))
                offsets.append(position % self.pool.block_size)
                token_indices.append(flat + local)
            flat += seg.q_len

        page_tensor = torch.tensor(page_ids, dtype=torch.int64, device=target)
        offset_tensor = torch.tensor(offsets, dtype=torch.int64, device=target)
        index_tensor: torch.Tensor | None
        if all_written and token_indices == list(range(flat)):
            index_tensor = None
        else:
            index_tensor = torch.tensor(token_indices, dtype=torch.long, device=target)
        cached = (page_tensor, offset_tensor, index_tensor)
        self._write_plan_cache[target] = cached
        return cached

    @classmethod
    def from_request_caches(
        cls,
        request_caches: Sequence[KVCacheView],
        q_lens: Sequence[int],
    ) -> "ForwardPagedKVView":
        if len(request_caches) != len(q_lens):
            raise invalid_descriptor("forward request caches and q_lens mismatch")
        if not request_caches:
            raise invalid_descriptor("forward request cache list must not be empty")
        pool = request_caches[0].pool
        if not isinstance(pool, PagedKVPool):
            raise invalid_descriptor("forward request cache must expose a PagedKVPool")
        segments = []
        for cache, q_len in zip(request_caches, q_lens):
            if cache.pool is not pool:
                raise invalid_descriptor("forward request caches must share one PagedKVPool")
            # ``base_len`` is the persistent sequence length on ``KVCacheView``.
            segments.append(
                ForwardPagedKVSegment(
                    block_ids=tuple(int(block_id) for block_id in getattr(cache, "block_ids", ())),
                    base_len=int(cache.base_len),
                    q_len=int(q_len),
                )
            )
        return cls(pool, segments)


class ForwardStreamBuilder:
    """Accumulates forward segment specs and materializes a ``ForwardStream``."""

    def __init__(self) -> None:
        self._segments: list[ForwardSegmentSpec] = []

    def add_segment(
        self,
        *,
        op_index: int,
        req_id: int,
        kind: str,
        mode: ForwardMode,
        modality: SegmentModality,
        segment_class: SegmentClass,
        q_len: int,
        prefix_len: int,
        branch_id: int = 0,
        visible_policy: VisiblePolicy = "causal",
        indexes: torch.Tensor | None = None,
    ) -> None:
        if modality not in SegmentModalityId.__members__:
            raise invalid_descriptor(f"unknown forward segment modality {modality!r}")
        if segment_class not in SegmentClassId.__members__:
            raise invalid_descriptor(f"unknown forward segment class {segment_class!r}")
        q_len = int(q_len)
        prefix_len = int(prefix_len)
        if q_len <= 0:
            raise invalid_descriptor("forward segment q_len must be positive")
        if prefix_len < 0:
            raise invalid_descriptor("forward segment prefix_len must be non-negative")
        if indexes is not None and tuple(indexes.shape) != (3, q_len):
            raise invalid_descriptor("forward segment indexes must be shaped [3, q_len]")
        if visible_policy not in VisiblePolicyKind.__members__:
            raise invalid_descriptor(f"unknown forward visible policy {visible_policy!r}")
        self._segments.append(
            ForwardSegmentSpec(
                op_index=int(op_index),
                req_id=int(req_id),
                kind=str(kind),
                mode=mode,
                modality=modality,
                segment_class=segment_class,
                q_len=q_len,
                prefix_len=prefix_len,
                branch_id=int(branch_id),
                visible_policy=visible_policy,
                indexes=indexes,
            )
        )

    def build(self, *, device: torch.device | str | None = None) -> ForwardStream:
        if not self._segments:
            raise invalid_descriptor("forward stream requires at least one segment")
        first_index = next((seg.indexes for seg in self._segments if seg.indexes is not None), None)
        target_device = torch.device(device) if device is not None else (
            first_index.device if first_index is not None else torch.device("cpu")
        )
        max_q = max(seg.q_len for seg in self._segments)
        cu_q = [0]
        visible_rows = []
        index_chunks = []
        und_indices: list[int] = []
        gen_indices: list[int] = []
        fully_visible = True
        for seg in self._segments:
            token_start = cu_q[-1]
            cu_q.append(cu_q[-1] + seg.q_len)
            target_indices = gen_indices if seg.modality == "gen" else und_indices
            target_indices.extend(range(token_start, token_start + seg.q_len))
            row = torch.zeros(max_q, dtype=torch.int32, device=target_device)
            policy = VisiblePolicyKind(seg.visible_policy)
            visible = policy.visible_end(seg.prefix_len, seg.q_len, target_device)
            row[: seg.q_len] = visible
            visible_rows.append(row)
            if not (policy is VisiblePolicyKind.bidirectional or seg.q_len == 1):
                fully_visible = False
            if seg.indexes is None:
                index_chunks.append(
                    build_text_position_indexes(seg.prefix_len, seg.q_len, target_device)
                )
            else:
                index_chunks.append(seg.indexes.to(device=target_device, dtype=torch.long))

        return ForwardStream(
            segments=tuple(self._segments),
            cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, device=target_device),
            visible_end=torch.stack(visible_rows, dim=0),
            indexes=torch.cat(index_chunks, dim=1),
            fully_visible=fully_visible,
            und_indices=torch.tensor(und_indices, dtype=torch.long, device=target_device),
            gen_indices=torch.tensor(gen_indices, dtype=torch.long, device=target_device),
        )


def build_text_position_indexes(
    start: int, length: int, device: torch.device | str | None
) -> torch.Tensor:
    """Build the ``[3, length]`` position-index table for a run of text tokens.

    Row 0 is the contiguous ``[start, start + length)`` temporal position; the
    height/width rows are zero because text tokens carry no spatial position.
    Shared by the forward-stream side-table and the interleaved text caches.
    """
    t = torch.arange(start, start + length, dtype=torch.long, device=device)
    zeros = torch.zeros(length, dtype=torch.long, device=device)
    return torch.stack([t, zeros, zeros], dim=0)
