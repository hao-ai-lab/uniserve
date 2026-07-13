"""Side-table builder for packed multimodal forward streams."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Literal

import torch

from ...backends.paged_kv_math import paged_kv_write
from ...contracts.forward_mode import ForwardMode
from ...foundation.errors import invalid_descriptor
from ...runtime.cache_protocols import KVCacheView
from ...runtime.host_staging import fill_cpu_ints, is_pinned
from ...runtime.kv_pool import PagedKVPool
from ...runtime.tensor_staging import TextTensorStager, TextTensorStagingSlot

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
    'ForwardGraphStreamState',
    'ForwardGraphPagedKVView',
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
    index_start: int | None = None


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


def _normalize_forward_paged_segments(
    pool: PagedKVPool,
    segments: list[ForwardPagedKVSegment] | tuple[ForwardPagedKVSegment, ...],
) -> tuple[ForwardPagedKVSegment, ...]:
    if not segments:
        raise invalid_descriptor("forward paged KV view requires at least one segment")
    normalized = tuple(
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
    for seg in normalized:
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
    return normalized


def _forward_paged_segment_signature(
    segments: tuple[ForwardPagedKVSegment, ...],
) -> tuple[tuple[int, bool, bool, int], ...]:
    return tuple(
        (
            int(seg.q_len),
            bool(seg.write_kv),
            bool(seg.persist_kv),
            int(seg.branch_id),
        )
        for seg in segments
    )


class ForwardPagedKVView:
    """One-pool paged-KV view for ragged visible-end attention segments."""

    def __init__(
        self,
        pool: PagedKVPool,
        segments: list[ForwardPagedKVSegment] | tuple[ForwardPagedKVSegment, ...],
    ) -> None:
        self.pool = pool
        self.segments = _normalize_forward_paged_segments(pool, segments)
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


class ForwardGraphStreamState:
    """Stable tensor identities for one captured packed-forward stream geometry."""

    def __init__(self, stream: ForwardStream) -> None:
        self._signature = self._stream_signature(stream)
        self.stream = ForwardStream(
            segments=tuple(stream.segments),
            cu_seqlens_q=stream.cu_seqlens_q.detach().clone(),
            visible_end=stream.visible_end.detach().clone(),
            indexes=stream.indexes.detach().clone(),
            fully_visible=bool(stream.fully_visible),
            und_indices=self._clone_optional(stream.und_indices),
            gen_indices=self._clone_optional(stream.gen_indices),
        )

    @classmethod
    def from_stream(cls, stream: ForwardStream) -> "ForwardGraphStreamState":
        return cls(stream)

    def refresh(self, stream: ForwardStream) -> ForwardStream:
        if self._stream_signature(stream) != self._signature:
            raise invalid_descriptor("forward graph stream geometry mismatch")
        # cu_seqlens_q is fixed by captured segment geometry. Base positions and
        # exact expert routes may vary while preserving their captured shapes.
        self.stream.visible_end.copy_(stream.visible_end, non_blocking=True)
        self.stream.indexes.copy_(stream.indexes, non_blocking=True)
        if self.stream.und_indices is not None and stream.und_indices is not None:
            self.stream.und_indices.copy_(stream.und_indices, non_blocking=True)
        if self.stream.gen_indices is not None and stream.gen_indices is not None:
            self.stream.gen_indices.copy_(stream.gen_indices, non_blocking=True)
        object.__setattr__(self.stream, "segments", tuple(stream.segments))
        return self.stream

    @staticmethod
    def _clone_optional(tensor: torch.Tensor | None) -> torch.Tensor | None:
        return None if tensor is None else tensor.detach().clone()

    @staticmethod
    def _stream_signature(stream: ForwardStream) -> tuple[object, ...]:
        return (
            tuple(
                (
                    int(seg.q_len),
                    seg.modality,
                    seg.segment_class,
                    seg.visible_policy,
                    int(seg.branch_id),
                )
                for seg in stream.segments
            ),
            tuple(stream.cu_seqlens_q.shape),
            str(stream.cu_seqlens_q.dtype),
            str(stream.cu_seqlens_q.device),
            tuple(stream.visible_end.shape),
            str(stream.visible_end.dtype),
            str(stream.visible_end.device),
            tuple(stream.indexes.shape),
            str(stream.indexes.dtype),
            str(stream.indexes.device),
            bool(stream.fully_visible),
            None
            if stream.und_indices is None
            else (tuple(stream.und_indices.shape), str(stream.und_indices.dtype), str(stream.und_indices.device)),
            None
            if stream.gen_indices is None
            else (tuple(stream.gen_indices.shape), str(stream.gen_indices.dtype), str(stream.gen_indices.device)),
        )


class ForwardGraphPagedKVView:
    """Graph-owned paged-KV view with refreshable values and stable tensors."""

    def __init__(
        self,
        pool: PagedKVPool,
        segments: list[ForwardPagedKVSegment] | tuple[ForwardPagedKVSegment, ...],
        *,
        device: torch.device | str | None = None,
    ) -> None:
        self.pool = pool
        self.segments = _normalize_forward_paged_segments(pool, segments)
        self._signature = _forward_paged_segment_signature(self.segments)
        self._device = torch.device(device if device is not None else pool.k.device)
        if self._device != pool.k.device:
            raise invalid_descriptor("forward graph paged KV view device must match the pool")
        self._block_width = max(len(seg.block_ids) for seg in self.segments)
        self._total_tokens = sum(int(seg.q_len) for seg in self.segments)
        self._write_tokens = sum(int(seg.q_len) for seg in self.segments if seg.write_kv)
        segment_count = len(self.segments)
        block_table_numel = segment_count * self._block_width
        cache_before_start = block_table_numel
        cache_after_start = cache_before_start + segment_count
        persistent_after_start = cache_after_start + segment_count
        cu_after_start = persistent_after_start + segment_count
        int32_numel = cu_after_start + segment_count + 1
        self._int32_inputs = torch.empty(int32_numel, dtype=torch.int32, device=self._device)
        self._block_table = self._int32_inputs[:block_table_numel].view(
            segment_count,
            self._block_width,
        )
        self._cache_seqlens_before = self._int32_inputs[
            cache_before_start:cache_after_start
        ]
        self._cache_seqlens_after = self._int32_inputs[
            cache_after_start:persistent_after_start
        ]
        self._persistent_cache_seqlens_after = self._int32_inputs[
            persistent_after_start:cu_after_start
        ]
        self._cu_seqlens_after = self._int32_inputs[cu_after_start:int32_numel]
        self._write_plan_inputs = torch.empty(
            2 * self._write_tokens,
            dtype=torch.int64,
            device=self._device,
        )
        self._page_ids = self._write_plan_inputs[: self._write_tokens]
        self._offsets = self._write_plan_inputs[self._write_tokens :]
        self._stager = TextTensorStager(ring_depth=3)
        token_indices = self._static_token_indices(self.segments)
        self._token_indices = (
            None
            if token_indices is None
            else torch.tensor(token_indices, dtype=torch.long, device=self._device)
        )
        self.refresh(self.segments)

    def refresh(
        self,
        segments: list[ForwardPagedKVSegment] | tuple[ForwardPagedKVSegment, ...],
    ) -> "ForwardGraphPagedKVView":
        normalized = _normalize_forward_paged_segments(self.pool, segments)
        if _forward_paged_segment_signature(normalized) != self._signature:
            raise invalid_descriptor("forward graph paged KV geometry mismatch")
        if max(len(seg.block_ids) for seg in normalized) > self._block_width:
            raise invalid_descriptor("forward graph paged KV block-table width exceeded")
        self.segments = normalized
        cache_before = [int(seg.base_len) for seg in normalized]
        cache_after = [int(seg.base_len) + int(seg.q_len) for seg in normalized]
        persistent_after = [
            int(seg.base_len) + int(seg.q_len) if seg.persist_kv else int(seg.base_len)
            for seg in normalized
        ]
        total = 0
        cu_after = [0]
        for value in cache_after:
            total += int(value)
            cu_after.append(total)
        slot = self._stager.acquire_slot(device=self._device)
        try:
            self._copy_int32_inputs(
                normalized,
                cache_before,
                cache_after,
                persistent_after,
                cu_after,
                slot=slot,
            )
            page_ids, offsets = self._write_plan_values(normalized)
            self._copy_write_plan_inputs(page_ids, offsets, slot=slot)
        finally:
            self._stager.mark_slot_submitted(slot, device=self._device)
        return self

    def _copy_int32_inputs(
        self,
        segments: tuple[ForwardPagedKVSegment, ...],
        cache_before: list[int],
        cache_after: list[int],
        persistent_after: list[int],
        cu_after: list[int],
        *,
        slot: TextTensorStagingSlot,
    ) -> None:
        segment_count = len(segments)
        if (
            len(cache_before) != segment_count
            or len(cache_after) != segment_count
            or len(persistent_after) != segment_count
            or len(cu_after) != segment_count + 1
        ):
            raise invalid_descriptor("forward graph paged KV side-table length mismatch")
        block_table_numel = segment_count * self._block_width
        total_numel = block_table_numel + 3 * segment_count + segment_count + 1
        if int(self._int32_inputs.numel()) != total_numel:
            raise invalid_descriptor("forward graph paged KV side-table geometry mismatch")
        flat = slot.int_buffer(
            "int32_inputs",
            total_numel,
            pin=self._device.type == "cuda",
        )
        flat.zero_()
        for row, seg in enumerate(segments):
            if seg.block_ids:
                start = row * self._block_width
                fill_cpu_ints(flat[start : start + len(seg.block_ids)], seg.block_ids)
        offset = block_table_numel
        for values in (cache_before, cache_after, persistent_after, cu_after):
            fill_cpu_ints(flat[offset : offset + len(values)], values)
            offset += len(values)
        self._int32_inputs.copy_(flat, non_blocking=self._non_blocking_cpu_copy(flat))

    def _copy_write_plan_inputs(
        self,
        page_ids: list[int],
        offsets: list[int],
        *,
        slot: TextTensorStagingSlot,
    ) -> None:
        if len(page_ids) != self._write_tokens or len(offsets) != self._write_tokens:
            raise invalid_descriptor("forward graph paged KV write-plan length mismatch")
        if self._write_tokens == 0:
            return
        cpu = slot.long_buffer(
            "write_plan_inputs",
            2 * self._write_tokens,
            pin=self._device.type == "cuda",
        )
        fill_cpu_ints(cpu[: self._write_tokens], page_ids)
        fill_cpu_ints(cpu[self._write_tokens :], offsets)
        self._write_plan_inputs.copy_(cpu, non_blocking=self._non_blocking_cpu_copy(cpu))

    def _non_blocking_cpu_copy(self, cpu: torch.Tensor) -> bool:
        return self._device.type == "cuda" and is_pinned(cpu)

    def _target_device(self, device: torch.device | str | None = None) -> torch.device:
        target = torch.device(device if device is not None else self._device)
        if target != self._device:
            raise invalid_descriptor("forward graph paged KV view is bound to one device")
        return target

    def block_table(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        self._target_device(device)
        return self._block_table

    @property
    def base_lens(self) -> tuple[int, ...]:
        return tuple(int(seg.base_len) for seg in self.segments)

    @property
    def base_len(self) -> int:
        return max(self.base_lens, default=0)

    def cache_seqlens(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        return self.cache_seqlens_before(device=device)

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.append_packed(layer, k, v)

    def append_varlen(
        self,
        layer: int,
        k: torch.Tensor,
        v: torch.Tensor,
        query_lens: Sequence[int],
        *,
        block_table: torch.Tensor | None = None,
        cache_seqlens: torch.Tensor | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
    ) -> None:
        del query_lens, block_table, cache_seqlens, cu_seqlens_q
        self.append_packed(layer, k, v)

    def cache_seqlens_before(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        self._target_device(device)
        return self._cache_seqlens_before

    def cache_seqlens_after(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        self._target_device(device)
        return self._cache_seqlens_after

    def cu_seqlens_after(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        self._target_device(device)
        return self._cu_seqlens_after

    def persistent_cache_seqlens_after(
        self,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        self._target_device(device)
        return self._persistent_cache_seqlens_after

    def max_seqlen_k(self) -> int:
        return max((int(seg.base_len) + int(seg.q_len) for seg in self.segments), default=0)

    def append_packed(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        if k.shape != v.shape:
            raise invalid_descriptor("forward packed KV key/value shapes must match")
        if k.ndim != 3:
            raise invalid_descriptor("forward packed KV append expects [total_tokens, heads, dim]")
        if int(k.shape[0]) != self._total_tokens:
            raise invalid_descriptor(
                f"forward packed KV has {int(k.shape[0])} tokens, expected {self._total_tokens}"
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
        self._target_device(device)
        return self._page_ids, self._offsets, self._token_indices

    def _write_plan_values(
        self,
        segments: tuple[ForwardPagedKVSegment, ...],
    ) -> tuple[list[int], list[int]]:
        page_ids: list[int] = []
        offsets: list[int] = []
        for seg in segments:
            if not seg.write_kv:
                continue
            for local in range(seg.q_len):
                position = int(seg.base_len) + local
                block_slot = position // self.pool.block_size
                if block_slot >= len(seg.block_ids):
                    raise invalid_descriptor("forward paged segment blocks do not cover current append")
                page_ids.append(int(seg.block_ids[block_slot]))
                offsets.append(position % self.pool.block_size)
        return page_ids, offsets

    @staticmethod
    def _static_token_indices(
        segments: tuple[ForwardPagedKVSegment, ...],
    ) -> list[int] | None:
        token_indices: list[int] = []
        flat = 0
        all_written = True
        for seg in segments:
            if not seg.write_kv:
                all_written = False
                flat += seg.q_len
                continue
            for local in range(seg.q_len):
                token_indices.append(flat + local)
            flat += seg.q_len
        if all_written and token_indices == list(range(flat)):
            return None
        return token_indices


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
        index_start: int | None = None,
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
        if indexes is not None and index_start is not None:
            raise invalid_descriptor("forward segment index_start is only valid for generated indexes")
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
                index_start=None if index_start is None else int(index_start),
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
        visible_values: list[int] = []
        index_chunks: list[torch.Tensor] = []
        generated_index_positions: list[int] = []
        und_indices: list[int] = []
        gen_indices: list[int] = []
        fully_visible = True

        def flush_generated_indexes() -> None:
            if generated_index_positions:
                index_chunks.append(
                    _build_text_position_indexes_from_positions(
                        generated_index_positions,
                        target_device,
                    )
                )
                generated_index_positions.clear()

        for seg in self._segments:
            token_start = cu_q[-1]
            cu_q.append(cu_q[-1] + seg.q_len)
            target_indices = gen_indices if seg.modality == "gen" else und_indices
            target_indices.extend(range(token_start, token_start + seg.q_len))
            policy = VisiblePolicyKind(seg.visible_policy)
            if policy is VisiblePolicyKind.causal:
                visible_values.extend(int(seg.prefix_len) + offset + 1 for offset in range(seg.q_len))
            else:
                visible_values.extend([int(seg.prefix_len) + int(seg.q_len)] * int(seg.q_len))
            visible_values.extend([0] * (max_q - int(seg.q_len)))
            if not (policy is VisiblePolicyKind.bidirectional or seg.q_len == 1):
                fully_visible = False
            if seg.indexes is None:
                start = int(seg.index_start) if seg.index_start is not None else int(seg.prefix_len)
                generated_index_positions.extend(range(start, start + int(seg.q_len)))
            else:
                flush_generated_indexes()
                index_chunks.append(seg.indexes.to(device=target_device, dtype=torch.long))
        flush_generated_indexes()

        return ForwardStream(
            segments=tuple(self._segments),
            cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, device=target_device),
            visible_end=torch.tensor(
                visible_values,
                dtype=torch.int32,
                device=target_device,
            ).view(len(self._segments), max_q),
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


def _build_text_position_indexes_from_positions(
    positions: Sequence[int],
    device: torch.device | str | None,
) -> torch.Tensor:
    t = torch.tensor([int(pos) for pos in positions], dtype=torch.long, device=device)
    zeros = torch.zeros((2, int(t.numel())), dtype=torch.long, device=device)
    return torch.cat([t.view(1, -1), zeros], dim=0)
