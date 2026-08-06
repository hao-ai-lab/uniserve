"""Transactional authority for sequence KV block tables and lengths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from threading import RLock
from typing import TypeAlias, cast

import torch

from ..backends.paged_kv_math import paged_kv_write
from ..batch import (
    Admission,
    FixedPoint,
    ProductKind,
    ProductRef,
    RequestKey,
    StorageClass,
    VersionRef,
)
from ..foundation.errors import invalid_descriptor
from ..foundation.sizing import bucketed_page_count, ceil_div
from .host_staging import (
    TensorStagingSlot,
    copy_cpu_to_device,
    cpu_int_staging_buffer,
    fill_cpu_ints,
    is_pinned,
)
from .kv_pool import PagedKVPool
from .transfer import Locator, Transport


@dataclass(frozen=True, slots=True)
class KvSnapshot:
    """One immutable incremental publication of an exact committed KV view."""

    locators: tuple[str, ...]
    source_version: VersionRef
    source_digest: str
    destination: str
    base_version: VersionRef | None
    base_extent: int
    published_extent: int
    block_ids: tuple[int, ...]
    logical_blocks: tuple[int, ...]
    group_id: int
    mapping_generation: int
    scale_identity: str

    def __post_init__(self) -> None:
        point = self.source_version.point
        if (
            not isinstance(point, FixedPoint)
            or point.semantic_digest != self.source_digest
            or len(self.source_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.source_digest)
        ):
            raise invalid_descriptor("KV publication source digest is not exact")
        if not self.destination:
            raise invalid_descriptor("KV publication destination is empty")
        if self.base_extent < 0 or self.published_extent < self.base_extent:
            raise invalid_descriptor("KV publication extents are not contiguous")
        if self.base_version is None and self.base_extent != 0:
            raise invalid_descriptor("KV publication base identity does not match its extent")
        if self.base_version is not None:
            base_point = self.base_version.point
            if self.base_version.request_key != self.source_version.request_key or not isinstance(
                base_point, FixedPoint
            ):
                raise invalid_descriptor("KV publication base version is not exact")
        if any(block_id < 0 for block_id in self.block_ids) or len(set(self.block_ids)) != len(
            self.block_ids
        ):
            raise invalid_descriptor("KV publication block mapping is invalid")
        if (
            len(self.logical_blocks) != len(self.block_ids)
            or any(block_id < 0 for block_id in self.logical_blocks)
            or len(set(self.logical_blocks)) != len(self.logical_blocks)
        ):
            raise invalid_descriptor("KV publication logical lease is invalid")
        if self.group_id < 0 or self.mapping_generation < 1 or not self.scale_identity:
            raise invalid_descriptor("KV publication mapping metadata is invalid")
        if self.published_extent == self.base_extent and self.locators:
            raise invalid_descriptor("empty KV publication suffix carries transfer locators")

    def to_wire(self) -> dict[str, object]:
        return {
            "locators": list(self.locators),
            "source_version": self.source_version.to_wire(),
            "source_digest": self.source_digest,
            "destination": self.destination,
            "base_version": None if self.base_version is None else self.base_version.to_wire(),
            "base_extent": self.base_extent,
            "published_extent": self.published_extent,
            "block_ids": list(self.block_ids),
            "logical_blocks": list(self.logical_blocks),
            "group_id": self.group_id,
            "mapping_generation": self.mapping_generation,
            "scale_identity": self.scale_identity,
        }

    @classmethod
    def from_wire(cls, value: object) -> KvSnapshot:
        if not isinstance(value, Mapping):
            raise invalid_descriptor("KV publication descriptor is not a mapping")
        base = value.get("base_version")
        raw_block_ids = value.get("block_ids", ())
        raw_logical_blocks = value.get("logical_blocks", ())
        if not isinstance(raw_block_ids, Sequence) or isinstance(
            raw_block_ids, (str, bytes, bytearray)
        ):
            raise invalid_descriptor("KV publication block mapping is not a sequence")
        block_ids: list[int] = []
        for item in raw_block_ids:
            if not isinstance(item, int) or isinstance(item, bool):
                raise invalid_descriptor("KV publication block mapping contains a non-integer")
            block_ids.append(item)
        if not isinstance(raw_logical_blocks, Sequence) or isinstance(
            raw_logical_blocks, (str, bytes, bytearray)
        ):
            raise invalid_descriptor("KV publication logical lease is not a sequence")
        logical_blocks: list[int] = []
        for item in raw_logical_blocks:
            if not isinstance(item, int) or isinstance(item, bool):
                raise invalid_descriptor("KV publication logical lease contains a non-integer")
            logical_blocks.append(item)
        return cls(
            locators=tuple(str(item) for item in cast(Sequence[object], value.get("locators", ()))),
            source_version=VersionRef.from_wire(
                value.get("source_version"), "KV publication.source_version"
            ),
            source_digest=str(value.get("source_digest", "")),
            destination=str(value.get("destination", "")),
            base_version=None
            if base is None
            else VersionRef.from_wire(base, "KV publication.base_version"),
            base_extent=int(value.get("base_extent", 0)),
            published_extent=int(value.get("published_extent", 0)),
            block_ids=tuple(block_ids),
            logical_blocks=tuple(logical_blocks),
            group_id=int(value.get("group_id", 0)),
            mapping_generation=int(value.get("mapping_generation", 0)),
            scale_identity=str(value.get("scale_identity", "")),
        )

    def for_tensor_rank(self, rank: int, size: int) -> KvSnapshot:
        """Select this rank's contiguous locator group from a TP publication."""

        rank = int(rank)
        size = int(size)
        if size < 1 or rank < 0 or rank >= size:
            raise invalid_descriptor("published KV tensor-parallel rank is invalid")
        if not self.locators:
            return self
        if len(self.locators) % size:
            raise invalid_descriptor(
                "published KV locators do not divide across tensor-parallel ranks"
            )
        width = len(self.locators) // size
        start = rank * width
        return replace(self, locators=self.locators[start : start + width])


@dataclass(slots=True)
class KvEntry:
    block_ids: list[int] = field(default_factory=list)
    logical_blocks: list[int] = field(default_factory=list)
    prefix_len: int = 0
    group_id: int = 0
    reserved_len: int = 0
    initialized_len: int = 0
    visible_len: int = 0
    committed_len: int = 0
    published_by_destination: dict[str, int] = field(default_factory=dict)
    mapping_generation: int = 1
    scale_identity: str = "none"

    @property
    def length(self) -> int:
        """Attention-visible extent retained for existing runner ownership."""

        return self.visible_len

    @length.setter
    def length(self, value: int) -> None:
        self.visible_len = int(value)

    @property
    def published_len(self) -> int:
        return max(self.published_by_destination.values(), default=0)

    def extents(self) -> KvExtents:
        value = KvExtents(
            reserved=self.reserved_len,
            initialized=self.initialized_len,
            visible=self.visible_len,
            committed=self.committed_len,
            published=self.published_len,
        )
        value.validate()
        return value


@dataclass(frozen=True, slots=True)
class KvExtents:
    reserved: int = 0
    initialized: int = 0
    visible: int = 0
    committed: int = 0
    published: int = 0

    def validate(self) -> None:
        if (
            not 0
            <= self.published
            <= self.committed
            <= self.visible
            <= self.initialized
            <= self.reserved
        ):
            raise invalid_descriptor("KV extents are not monotonically contained")


@dataclass(slots=True)
class _RetainedPage:
    key: torch.Tensor
    value: torch.Tensor
    key_scale: torch.Tensor | None
    value_scale: torch.Tensor | None
    key_scale_set: torch.Tensor | None
    value_scale_set: torch.Tensor | None


_KvEntryState: TypeAlias = tuple[
    bool,
    tuple[int, ...],
    tuple[int, ...],
    int,
    int,
    int,
    int,
    int,
    int,
    dict[str, int],
    int,
    str,
]


_LogicalPageKey: TypeAlias = int


@dataclass(frozen=True, slots=True)
class _LogicalPageUndo:
    key: _LogicalPageKey
    page: int


@dataclass(frozen=True, slots=True)
class _KvSnapshot:
    entries: dict[int, _KvEntryState]
    branches: dict[tuple[ProductRef, str], tuple[tuple[int, ...], int]]
    publications: dict[ProductRef, KvSnapshot]
    destination_bases: dict[tuple[int, str], tuple[VersionRef, int, tuple[int, ...], int, str]]
    installed_bases: dict[tuple[int, str], tuple[VersionRef, int]]
    retained_counts: dict[int, int]


@dataclass(frozen=True, slots=True)
class _KvEntrySnapshot:
    entries: dict[int, _KvEntryState]


@dataclass(frozen=True, slots=True)
class _KvAuxiliarySnapshot:
    branches: dict[tuple[ProductRef, str], tuple[tuple[int, ...], int]]
    publications: dict[ProductRef, KvSnapshot]
    destination_bases: dict[tuple[int, str], tuple[VersionRef, int, tuple[int, ...], int, str]]
    installed_bases: dict[tuple[int, str], tuple[VersionRef, int]]
    retained_counts: dict[int, int]


@dataclass(frozen=True, slots=True)
class KvPageState:
    key: torch.Tensor
    value: torch.Tensor
    key_scale: torch.Tensor | None = None
    value_scale: torch.Tensor | None = None
    key_scale_set: torch.Tensor | None = None
    value_scale_set: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class KvBranchState:
    owner: ProductRef
    branch: str
    length: int
    block_count: int
    pages: KvPageState


@dataclass(frozen=True, slots=True)
class KvCommittedState:
    session_id: int
    block_ids: tuple[int, ...]
    logical_blocks: tuple[int, ...]
    prefix_len: int
    length: int
    group_id: int
    reserved_len: int
    initialized_len: int
    committed_len: int
    published_by_destination: tuple[tuple[str, int], ...]
    mapping_generation: int
    scale_identity: str
    pages: KvPageState | None
    branches: tuple[KvBranchState, ...]
    publications: tuple[tuple[ProductRef, KvSnapshot], ...]
    destination_bases: tuple[
        tuple[str, VersionRef, int, tuple[int, ...], int, str],
        ...,
    ]
    installed_bases: tuple[tuple[str, VersionRef, int], ...]


@dataclass(frozen=True, slots=True)
class _VarlenAppendPlan:
    key: tuple[object, ...]
    page_ids: torch.Tensor
    offsets: torch.Tensor


class KvBatchView:
    """Forward-bounded view over one ordered set of request KV leases."""

    def __init__(
        self,
        pool: PagedKVPool,
        block_ids: Sequence[Sequence[int]],
        base_lens: Sequence[int],
        query_lens: Sequence[int] | None = None,
    ) -> None:
        if not block_ids:
            raise invalid_descriptor("KV batch view requires at least one row")
        if len(block_ids) != len(base_lens):
            raise invalid_descriptor("KV batch view rows and lengths mismatch")
        self._pool = pool
        self._block_ids = tuple(tuple(pool.validate_block_ids(ids)) for ids in block_ids)
        self._base_lens = tuple(int(value) for value in base_lens)
        if any(value < 0 for value in self._base_lens):
            raise invalid_descriptor("KV batch view lengths must be non-negative")
        self._query_lens = None if query_lens is None else tuple(int(value) for value in query_lens)
        if self._query_lens is not None and len(self._query_lens) != len(base_lens):
            raise invalid_descriptor("packed KV query lengths do not match cache rows")
        self._block_table_width = bucketed_page_count(max(len(ids) for ids in self._block_ids))
        self._block_tables: dict[torch.device, torch.Tensor] = {}
        self._cache_lengths: dict[torch.device, torch.Tensor] = {}
        self._append_plan: _VarlenAppendPlan | None = None

    @property
    def block_size(self) -> int:
        return self._pool.block_size

    @property
    def supports_paged_attention_storage(self) -> bool:
        return bool(self._pool.supports_paged_attention_storage)

    @property
    def base_lens(self) -> tuple[int, ...]:
        return self._base_lens

    def with_synthetic_row(
        self,
        block_ids: Sequence[int],
        *,
        base_len: int,
        query_len: int,
    ) -> KvBatchView:
        """Return a view with one non-session row over explicit pool storage."""
        if self._query_lens is None:
            raise invalid_descriptor("synthetic KV rows require declared query lengths")
        return KvBatchView(
            self._pool,
            (*self._block_ids, tuple(int(value) for value in block_ids)),
            (*self._base_lens, int(base_len)),
            (*self._query_lens, int(query_len)),
        )

    def layer_kv(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._pool.layer_cache(layer)

    def append(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> None:
        if key.shape != value.shape:
            raise invalid_descriptor("batched KV append key/value shapes must match")
        if key.ndim != 4:
            raise invalid_descriptor("batched KV append expects [batch, tokens, heads, dim]")
        if int(key.shape[0]) != len(self._block_ids):
            raise invalid_descriptor("batched KV append batch size does not match cache rows")
        for row, block_ids in enumerate(self._block_ids):
            self._pool.write(
                layer,
                list(block_ids),
                start=self._base_lens[row],
                k=key[row],
                v=value[row],
            )

    def append_varlen(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        row_lengths: Sequence[int],
        *,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        query_offsets: torch.Tensor,
    ) -> None:
        if key.shape != value.shape:
            raise invalid_descriptor("ragged KV append key/value shapes must match")
        if key.ndim != 3:
            raise invalid_descriptor("ragged KV append expects [tokens, heads, dim]")
        lengths = tuple(int(value) for value in row_lengths)
        if len(lengths) != len(self._block_ids):
            raise invalid_descriptor("ragged KV append lengths do not match cache rows")
        if any(value < 0 for value in lengths):
            raise invalid_descriptor("ragged KV append lengths must be non-negative")
        total = sum(lengths)
        if total != int(key.shape[0]):
            raise invalid_descriptor("ragged KV append token count does not match query lengths")
        if self._append_varlen_indexed(
            int(layer),
            key,
            value,
            total=total,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            query_offsets=query_offsets,
        ):
            return
        offset = 0
        for row, (block_ids, length) in enumerate(zip(self._block_ids, lengths, strict=True)):
            if length:
                self._pool.write(
                    layer,
                    list(block_ids),
                    start=self._base_lens[row],
                    k=key[offset : offset + length],
                    v=value[offset : offset + length],
                )
            offset += length

    def append_packed(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        page_ids: torch.Tensor,
        page_offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> None:
        del layer, key, value, page_ids, page_offsets, token_indices
        raise RuntimeError("packed attention requires a packed KV transaction view")

    def block_table(
        self,
        device: torch.device,
        *,
        slot: TensorStagingSlot | None = None,
    ) -> torch.Tensor:
        target = torch.device(device)
        cached = self._block_tables.get(target)
        if cached is not None:
            return cached
        row_count = len(self._block_ids)
        cpu = cpu_int_staging_buffer(
            row_count * self._block_table_width,
            dtype=torch.int32,
            pin=target.type == "cuda",
            slot=slot,
            name="kv_block_table",
        )
        offset = 0
        for block_ids in self._block_ids:
            count = len(block_ids)
            fill_cpu_ints(cpu[offset : offset + count], block_ids)
            if count < self._block_table_width:
                cpu[offset + count : offset + self._block_table_width].zero_()
            offset += self._block_table_width
        result = copy_cpu_to_device(
            cpu,
            device=target,
            non_blocking=target.type == "cuda" and is_pinned(cpu),
            slot=slot,
            name="kv_block_table",
        ).view(row_count, self._block_table_width)
        self._block_tables[target] = result
        return result

    def cache_seqlens(
        self,
        device: torch.device,
        *,
        slot: TensorStagingSlot | None = None,
    ) -> torch.Tensor:
        target = torch.device(device)
        cached = self._cache_lengths.get(target)
        if cached is not None:
            return cached
        cpu = cpu_int_staging_buffer(
            len(self._base_lens),
            dtype=torch.int32,
            pin=target.type == "cuda",
            slot=slot,
            name="kv_cache_lengths",
        )
        fill_cpu_ints(cpu, self._base_lens)
        result = copy_cpu_to_device(
            cpu,
            device=target,
            non_blocking=target.type == "cuda" and is_pinned(cpu),
            slot=slot,
            name="kv_cache_lengths",
        )
        self._cache_lengths[target] = result
        return result

    def _append_varlen_indexed(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        total: int,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        query_offsets: torch.Tensor,
    ) -> bool:
        if total <= 0:
            return True
        if self._pool.is_quantized or not self._pool.supports_paged_attention_storage:
            return False
        if not (key.is_cuda and value.is_cuda and self._pool.k.is_cuda and self._pool.v.is_cuda):
            return False
        if key.device != self._pool.k.device or value.device != self._pool.v.device:
            return False
        row_count = len(self._block_ids)
        if int(block_table.shape[0]) != row_count or int(cache_seqlens.shape[0]) != row_count:
            return False
        if int(query_offsets.numel()) != row_count + 1:
            return False
        if (
            block_table.device != key.device
            or cache_seqlens.device != key.device
            or query_offsets.device != key.device
        ):
            return False
        if key.shape[1:] != (self._pool.n_kv, self._pool.head_dim):
            return False
        plan = self._varlen_append_plan(
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            query_offsets=query_offsets,
            total=total,
        )
        self._pool.k[layer, plan.page_ids, plan.offsets] = key.to(dtype=self._pool.k.dtype)
        self._pool.v[layer, plan.page_ids, plan.offsets] = value.to(dtype=self._pool.v.dtype)
        return True

    def _varlen_append_plan(
        self,
        *,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        query_offsets: torch.Tensor,
        total: int,
    ) -> _VarlenAppendPlan:
        cache_key = (
            int(block_table.data_ptr()),
            int(cache_seqlens.data_ptr()),
            int(query_offsets.data_ptr()),
            tuple(int(value) for value in block_table.shape),
            tuple(int(value) for value in cache_seqlens.shape),
            tuple(int(value) for value in query_offsets.shape),
            int(total),
        )
        cached = self._append_plan
        if cached is not None and cached.key == cache_key:
            return cached
        token_offsets = torch.arange(total, device=block_table.device, dtype=torch.int64)
        if len(self._block_ids) == 1:
            positions = cache_seqlens[0].to(dtype=torch.int64) + token_offsets
            block_slots = torch.div(positions, self._pool.block_size, rounding_mode="floor")
            page_ids = (
                block_table[0].to(dtype=torch.int64).index_select(0, block_slots).contiguous()
            )
        else:
            offsets = query_offsets.to(dtype=torch.int64)
            row_ids = torch.bucketize(token_offsets, offsets[1:].contiguous(), right=True)
            positions = (
                cache_seqlens.to(dtype=torch.int64)[row_ids] + token_offsets - offsets[row_ids]
            )
            block_slots = torch.div(positions, self._pool.block_size, rounding_mode="floor")
            page_ids = block_table.to(dtype=torch.int64)[row_ids, block_slots].contiguous()
        page_offsets = torch.remainder(positions, self._pool.block_size).contiguous()
        plan = _VarlenAppendPlan(cache_key, page_ids, page_offsets)
        self._append_plan = plan
        return plan


class _PackedKvView:
    """Transaction-bounded packed view over logical KV spans in one pool."""

    def __init__(
        self,
        pool: PagedKVPool,
        rows: Sequence[tuple[KvEntry, int, bool]],
    ) -> None:
        if not rows:
            raise invalid_descriptor("packed KV view requires at least one row")
        self._pool = pool
        self._rows = tuple((entry, int(query), bool(write)) for entry, query, write in rows)
        for entry, query, write in self._rows:
            if query < 1:
                raise invalid_descriptor("packed KV query length must be positive")
            required = entry.length + query if write else entry.length
            if required > len(entry.block_ids) * pool.block_size:
                raise invalid_descriptor("packed KV row exceeds its block capacity")

    @property
    def block_size(self) -> int:
        return self._pool.block_size

    @property
    def supports_paged_attention_storage(self) -> bool:
        return bool(self._pool.supports_paged_attention_storage)

    @property
    def base_lens(self) -> tuple[int, ...]:
        return tuple(entry.length for entry, _query, _write in self._rows)

    @property
    def query_lens(self) -> tuple[int, ...]:
        return tuple(query for _entry, query, _write in self._rows)

    def layer_kv(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._pool.layer_cache(layer)

    def append(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> None:
        del layer, key, value
        raise RuntimeError("packed KV writes require an explicit write plan")

    def append_varlen(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        row_lengths: Sequence[int],
        *,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        query_offsets: torch.Tensor,
    ) -> None:
        del layer, key, value, block_table, cache_seqlens, query_offsets
        if tuple(int(value) for value in row_lengths) != self.query_lens:
            raise invalid_descriptor("packed KV append row lengths changed")
        raise RuntimeError("packed KV writes require an explicit write plan")

    def append_packed(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        page_ids: torch.Tensor,
        page_offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> None:
        if key.shape != value.shape or key.ndim != 3:
            raise invalid_descriptor("packed KV values must align as [tokens, heads, dim]")
        total = sum(self.query_lens)
        if int(key.shape[0]) != total:
            raise invalid_descriptor(
                f"packed KV received {int(key.shape[0])} tokens, expected {total}"
            )
        written = sum(query for _entry, query, write in self._rows if write)
        if not (
            tuple(page_ids.shape) == (written,)
            and tuple(page_offsets.shape) == (written,)
            and tuple(token_indices.shape) == (written,)
        ):
            raise invalid_descriptor("packed KV write-plan tensors do not match the writable span")
        if page_ids.device != key.device or page_offsets.device != key.device:
            raise invalid_descriptor("packed KV write locations must be on the key device")
        if token_indices.device != key.device:
            raise invalid_descriptor("packed KV token indices must be on the key device")
        if not self._pool.is_quantized:
            key_cache, value_cache = self._pool.layer_cache(layer)
            source_key = key.index_select(0, token_indices)
            source_value = value.index_select(0, token_indices)
            paged_kv_write(
                key_cache,
                value_cache,
                page_ids,
                page_offsets,
                source_key,
                source_value,
                cast=source_key.dtype != key_cache.dtype or source_value.dtype != value_cache.dtype,
            )
            return
        offset = 0
        for entry, query, write in self._rows:
            end = offset + query
            if write:
                self._pool.write(
                    layer,
                    entry.block_ids,
                    start=entry.length,
                    k=key[offset:end],
                    v=value[offset:end],
                )
            offset = end

    def block_table(self, device: torch.device) -> torch.Tensor:
        width = bucketed_page_count(
            max(len(entry.block_ids) for entry, _query, _write in self._rows)
        )
        result = torch.zeros((len(self._rows), width), dtype=torch.int32, device=device)
        for index, (entry, _query, _write) in enumerate(self._rows):
            if entry.block_ids:
                result[index, : len(entry.block_ids)] = torch.tensor(
                    entry.block_ids,
                    dtype=torch.int32,
                    device=device,
                )
        return result

    def seqused_k(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [entry.length + query for entry, query, _write in self._rows],
            dtype=torch.int32,
            device=device,
        )

    def write_plan(
        self,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pages: list[int] = []
        offsets: list[int] = []
        selected: list[int] = []
        flat = 0
        for entry, query, write in self._rows:
            if not write:
                flat += query
                continue
            for local in range(query):
                position = entry.length + local
                pages.append(entry.block_ids[position // self.block_size])
                offsets.append(position % self.block_size)
                selected.append(flat + local)
            flat += query
        return (
            torch.tensor(pages, dtype=torch.int64, device=device),
            torch.tensor(offsets, dtype=torch.int64, device=device),
            torch.tensor(selected, dtype=torch.long, device=device),
        )


class KvStore:
    """Own committed block leases, prefix boundaries, and physical lengths."""

    def __init__(self, pool: PagedKVPool | None = None) -> None:
        self.pool = pool
        self._entries: dict[int, KvEntry] = {}
        self._logical_pages: dict[_LogicalPageKey, int] = {}
        self._logical_groups: dict[_LogicalPageKey, int] = {}
        self._page_logical: dict[int, _LogicalPageKey] = {}
        self._branches: dict[tuple[ProductRef, str], KvEntry] = {}
        self._holders: dict[int, set[int]] = {}
        self._published: dict[
            int,
            list[tuple[ProductRef, tuple[str, ...], object]],
        ] = {}
        self._destination_bases: dict[
            tuple[int, str], tuple[VersionRef, int, tuple[int, ...], int, str]
        ] = {}
        self._installed_bases: dict[tuple[int, str], tuple[VersionRef, int]] = {}
        self._publication_products: dict[ProductRef, KvSnapshot] = {}
        self._lock = RLock()

    def bind_pool(self, pool: PagedKVPool) -> None:
        with self._lock:
            if self.pool is not None and self.pool is not pool:
                raise RuntimeError("KV store is already bound to a physical pool")
            self.pool = pool

    def resident_block_count(self) -> int:
        """Return the number of worker pages with live session holders."""

        with self._lock:
            return sum(bool(holders) for holders in self._holders.values())

    def scratch_token_count(self) -> int:
        """Return physically allocated branch capacity in token slots."""

        with self._lock:
            block_ids = {
                block_id for entry in self._branches.values() for block_id in entry.block_ids
            }
            block_size = 0 if self.pool is None else int(self.pool.block_size)
            return len(block_ids) * block_size

    def published_locator_count(self) -> int:
        """Return exact live transport registrations owned by KV operations."""

        with self._lock:
            return sum(
                len(locators)
                for entries in self._published.values()
                for _product, locators, _transport in entries
            )

    def admit(self, admission: Admission) -> None:
        session_id = admission.request_key.session_id
        with self._lock:
            und = admission.und
            if und is None:
                self._entries.setdefault(session_id, KvEntry())
                return
            if session_id in self._entries:
                return
            metadata = und.kv
            entry = KvEntry(
                prefix_len=metadata.prefix_len,
                group_id=metadata.group_id,
                reserved_len=metadata.prefix_len,
                initialized_len=metadata.prefix_len,
                visible_len=metadata.prefix_len,
                committed_len=metadata.prefix_len,
                scale_identity=self._scale_identity(),
            )
            self._entries[session_id] = entry

    def reserve_logical_page_delta(
        self,
        request_key: RequestKey,
        logical_page_delta: Sequence[int],
        *,
        expected_capacity_pages: int,
    ) -> tuple[_LogicalPageUndo, ...]:
        """Extend one logical lease with worker-selected physical pages."""

        pool = self.pool
        if pool is None:
            raise RuntimeError("KV registration requires a physical pool")
        session_id = int(request_key.session_id)
        delta = [int(value) for value in logical_page_delta]
        expected_capacity = int(expected_capacity_pages)
        if expected_capacity < 0 or expected_capacity > pool.leasable_num_blocks:
            raise invalid_descriptor("KV reservation declares an invalid logical capacity")
        if len(set(delta)) != len(delta):
            raise invalid_descriptor("KV reservation repeats a logical page")
        if any(value < 0 or value >= pool.leasable_num_blocks for value in delta):
            raise invalid_descriptor("KV reservation exceeds negotiated logical capacity")
        with self._lock:
            entry = self.get(session_id)
            if len(entry.logical_blocks) + len(delta) != expected_capacity:
                raise invalid_descriptor("KV reservation does not establish operation capacity")
            if set(delta).intersection(entry.logical_blocks):
                raise invalid_descriptor("KV reservation repeats a committed logical page")
            pages = list(entry.block_ids)
            undo: list[_LogicalPageUndo] = []
            try:
                for logical_block in delta:
                    key = logical_block
                    group = self._logical_groups.get(key)
                    if group is not None and group != int(entry.group_id):
                        raise invalid_descriptor("KV logical block belongs to another cache group")
                    page = self._logical_pages.get(key)
                    if page is None:
                        page = self._allocate_logical_page(key, int(entry.group_id))
                        undo.append(_LogicalPageUndo(key=key, page=page))
                    pages.append(page)
                if entry.prefix_len > len(pages) * pool.block_size:
                    raise invalid_descriptor("KV prefix exceeds its logical lease capacity")
                self._validate_blocks(session_id, pages, entry.prefix_len)
            except BaseException:
                self._undo_logical_pages(tuple(reversed(undo)))
                raise
            added = pages[len(entry.block_ids) :]
            entry.logical_blocks.extend(delta)
            entry.block_ids.extend(added)
            self._register(session_id, added)
            entry.reserved_len = len(pages) * pool.block_size
            if added:
                entry.mapping_generation += 1
            return tuple(undo)

    def _allocate_logical_page(
        self,
        key: _LogicalPageKey,
        group_id: int,
    ) -> int:
        pool = self.pool
        if pool is None:
            raise RuntimeError("KV registration requires a physical pool")
        page = pool.allocate_session_blocks(1)[0]
        self._logical_pages[key] = page
        self._logical_groups[key] = int(group_id)
        self._page_logical[page] = key
        return page

    def _undo_logical_pages(self, undo: Sequence[_LogicalPageUndo]) -> None:
        pool = self.pool
        if pool is None:
            raise RuntimeError("KV registration lost its physical pool")
        for change in undo:
            if self._logical_pages.get(change.key) != change.page:
                raise RuntimeError("KV logical-page mapping changed during registration")
            del self._logical_pages[change.key]
            del self._logical_groups[change.key]
            del self._page_logical[change.page]
            pool.release_session_blocks((change.page,))

    def get(self, session_id: int) -> KvEntry:
        with self._lock:
            try:
                return self._entries[int(session_id)]
            except KeyError:
                raise invalid_descriptor(f"session {session_id} has no KV state") from None

    def validate_write(self, session_id: int, begin: int, end: int) -> None:
        entry = self.get(session_id)
        if begin < entry.prefix_len:
            raise invalid_descriptor(
                f"session {session_id} writes KV below its prefix reference boundary"
            )
        if end < begin:
            raise invalid_descriptor("KV write range is inverted")
        if self.pool is not None and end > len(entry.block_ids) * self.pool.block_size:
            raise invalid_descriptor(f"session {session_id} KV write exceeds its block lease")

    def advance(self, session_id: int, tokens: int) -> None:
        with self._lock:
            entry = self.get(session_id)
            begin = entry.visible_len
            end = begin + int(tokens)
            self.validate_write(session_id, begin, end)
            entry.initialized_len = max(entry.initialized_len, end)
            entry.visible_len = end

    def initialize(self, session_id: int, tokens: int) -> int:
        """Record an enqueued append without making its suffix attention-visible."""

        with self._lock:
            entry = self.get(session_id)
            end = entry.visible_len + int(tokens)
            self.validate_write(session_id, entry.visible_len, end)
            entry.initialized_len = max(entry.initialized_len, end)
            return end

    def select(self, session_id: int, length: int) -> None:
        """Select one initialized prefix as the device-visible extent."""

        with self._lock:
            entry = self.get(session_id)
            selected = int(length)
            if selected < entry.committed_len or selected > entry.initialized_len:
                raise invalid_descriptor("KV selection is outside initialized branch state")
            entry.visible_len = selected

    def commit(self, session_id: int, length: int) -> None:
        """Advance the semantic extent to an exact already visible prefix."""

        with self._lock:
            entry = self.get(session_id)
            selected = int(length)
            if selected < entry.committed_len or selected > entry.visible_len:
                raise invalid_descriptor("KV commit is outside the visible extent")
            entry.committed_len = selected

    def rewind(self, session_id: int, length: int) -> None:
        """Select an already initialized prefix as the lineage's visible KV extent."""

        with self._lock:
            entry = self.get(session_id)
            selected = int(length)
            if selected < entry.prefix_len or selected > entry.initialized_len:
                raise invalid_descriptor("KV rewind selects an uninitialized prefix")
            entry.visible_len = selected
            entry.committed_len = min(entry.committed_len, selected)

    def view(
        self,
        session_ids: Sequence[int],
        *,
        query_lens: Sequence[int] | None = None,
    ) -> KvBatchView:
        if self.pool is None:
            raise RuntimeError("KV execution requires a physical pool")
        with self._lock:
            try:
                entries = [self._entries[int(value)] for value in session_ids]
            except KeyError as error:
                raise invalid_descriptor(f"session {error.args[0]} has no KV state") from None
        return self.view_entries(entries, query_lens=query_lens)

    def view_entries(
        self,
        entries: Sequence[KvEntry],
        *,
        query_lens: Sequence[int] | None = None,
    ) -> KvBatchView:
        if self.pool is None:
            raise RuntimeError("KV execution requires a physical pool")
        return KvBatchView(
            self.pool,
            [entry.block_ids for entry in entries],
            [entry.length for entry in entries],
            query_lens,
        )

    def packed_view(self, rows: Sequence[tuple[KvEntry, int, bool]]) -> _PackedKvView:
        if self.pool is None:
            raise RuntimeError("packed KV execution requires its declared physical pool")
        return _PackedKvView(self.pool, rows)

    def scratch_entry(
        self,
        owner: ProductRef,
        branch: str,
        *,
        capacity_tokens: int,
        copy_conditioning: bool,
    ) -> tuple[KvEntry, bool]:
        """Return one product-owned branch prefix, provisioning it atomically."""

        pool = self.pool
        if pool is None:
            raise RuntimeError("branch KV execution requires a physical pool")
        if (
            owner.kind is not ProductKind.LATENT
            or owner.storage_class is not StorageClass.LATENT_ARENA
            or int(owner.generation) < 1
        ):
            raise invalid_descriptor("branch KV owner is not an exact latent product")
        session_id = int(owner.request_key.session_id)
        key = (owner, str(branch))
        with self._lock:
            existing = self._branches.get(key)
            if existing is not None:
                self._ensure_scratch_capacity(existing, capacity_tokens)
                return existing, False
            source = self.get(session_id)
            prefix = source.visible_len if copy_conditioning else 0
            entry = KvEntry(
                initialized_len=prefix,
                visible_len=prefix,
                committed_len=prefix,
                scale_identity=self._scale_identity(),
            )
            try:
                self._ensure_scratch_capacity(entry, max(prefix, int(capacity_tokens)))
                if copy_conditioning and prefix:
                    pages = ceil_div(prefix, pool.block_size)
                    pool.copy_pages(source.block_ids[:pages], entry.block_ids[:pages])
            except BaseException:
                self._release_scratch(entry.block_ids)
                raise
            self._branches[key] = entry
            return entry, True

    def rebind_scratch_owner(self, source: ProductRef, target: ProductRef) -> None:
        """Move branch KV ownership to a successor latent product."""

        for owner in (source, target):
            if (
                owner.kind is not ProductKind.LATENT
                or owner.storage_class is not StorageClass.LATENT_ARENA
                or int(owner.generation) < 1
            ):
                raise invalid_descriptor("branch KV owner is not an exact latent product")
        if source.request_key != target.request_key:
            raise invalid_descriptor("branch KV ownership cannot cross request keys")
        if source == target:
            return
        with self._lock:
            moves = tuple(
                (key, (target, key[1]), entry)
                for key, entry in self._branches.items()
                if key[0] == source
            )
            if any(target_key in self._branches for _source_key, target_key, _entry in moves):
                raise invalid_descriptor("successor latent already owns branch KV state")
            for source_key, target_key, entry in moves:
                del self._branches[source_key]
                self._branches[target_key] = entry

    def release_scratch_owner(self, owner: ProductRef) -> None:
        """Release every branch prefix owned by one exact latent product."""

        if (
            owner.kind is not ProductKind.LATENT
            or owner.storage_class is not StorageClass.LATENT_ARENA
            or int(owner.generation) < 1
        ):
            raise invalid_descriptor("branch KV owner is not an exact latent product")
        with self._lock:
            keys = tuple(key for key in self._branches if key[0] == owner)
            for key in keys:
                self._release_scratch(self._branches.pop(key).block_ids)

    def advance_entry(self, entry: KvEntry, tokens: int) -> None:
        pool = self.pool
        if pool is None:
            raise RuntimeError("KV advance requires a physical pool")
        end = entry.length + int(tokens)
        if end > len(entry.block_ids) * pool.block_size:
            raise invalid_descriptor("KV advance exceeds its block capacity")
        entry.initialized_len = max(entry.initialized_len, end)
        entry.visible_len = end
        entry.committed_len = end

    def publish(
        self,
        session_id: int,
        *,
        source_version: VersionRef,
        source_digest: str,
        destination: str,
        expected_base: VersionRef | None,
        product: ProductRef,
        transport: object | None = None,
    ) -> KvSnapshot:
        """Publish only the missing suffix of one exact committed version.

        Destination state is immutable: every successful call appends a new
        snapshot and advances its exact installed base. A caller cannot skip,
        approximate, or overwrite that base by scalar length.
        """

        if int(source_version.request_key.session_id) != int(session_id):
            raise invalid_descriptor("KV publication source belongs to another session")
        if (
            product.request_key != source_version.request_key
            or product.kind is not ProductKind.KV
            or product.producer_op_id == source_version.producer_op_id
        ):
            raise invalid_descriptor("KV publication product has an invalid identity")
        point = source_version.point
        if not isinstance(point, FixedPoint) or point.semantic_digest != source_digest:
            raise invalid_descriptor("KV publication requires an exact fixed source digest")
        destination = str(destination)
        if not destination:
            raise invalid_descriptor("KV publication destination is empty")
        entry = self.get(session_id)
        extents = entry.extents()
        if extents.committed != extents.visible:
            raise invalid_descriptor("KV publication source is not the committed visible version")
        installed = self._destination_bases.get((int(session_id), destination))
        if installed is None:
            if expected_base is not None:
                raise invalid_descriptor("KV publication expected base is not installed")
            base_extent = 0
        else:
            installed_version, base_extent, base_blocks, group_id, scale_identity = installed
            if expected_base != installed_version:
                raise invalid_descriptor("KV publication expected base does not match destination")
            base_pages = ceil_div(base_extent, self.pool.block_size) if self.pool is not None else 0
            if (
                tuple(entry.block_ids[:base_pages]) != base_blocks[:base_pages]
                or group_id != entry.group_id
                or scale_identity != entry.scale_identity
            ):
                raise invalid_descriptor("KV publication base has incompatible mapping metadata")
        if base_extent > entry.committed_len:
            raise invalid_descriptor("KV publication destination is ahead of its source")
        locators: list[str] = []
        suffix = entry.committed_len - base_extent
        if transport is not None and suffix:
            if self.pool is None:
                raise RuntimeError("KV publication requires a physical pool")
            if not bool(getattr(transport, "supports_async_publication", False)):
                raise invalid_descriptor("KV publication transport is not asynchronous")
            publish = getattr(transport, "publish_async")
            for layer in range(self.pool.num_layers):
                key, value = self.pool.read(
                    layer,
                    entry.block_ids,
                    start=base_extent,
                    length=suffix,
                )
                if key is None or value is None:
                    raise RuntimeError("published KV span is incomplete")
                locators.append(publish(key.contiguous()).to_wire_json())
                locators.append(publish(value.contiguous()).to_wire_json())
            self._retain_published(session_id, product, tuple(locators), transport)
        snapshot = KvSnapshot(
            locators=tuple(locators),
            source_version=source_version,
            source_digest=source_digest,
            destination=destination,
            base_version=expected_base,
            base_extent=base_extent,
            published_extent=entry.committed_len,
            block_ids=tuple(entry.block_ids),
            logical_blocks=tuple(entry.logical_blocks),
            group_id=entry.group_id,
            mapping_generation=entry.mapping_generation,
            scale_identity=entry.scale_identity,
        )
        self._destination_bases[(int(session_id), destination)] = (
            source_version,
            entry.committed_len,
            tuple(entry.block_ids),
            entry.group_id,
            entry.scale_identity,
        )
        entry.published_by_destination[destination] = entry.committed_len
        self._publication_products[product] = snapshot
        entry.extents().validate()
        return snapshot

    def publication(self, product: ProductRef) -> KvSnapshot:
        with self._lock:
            try:
                return self._publication_products[product]
            except KeyError:
                raise invalid_descriptor("KV publication product is not resident") from None

    def stage_publication(self, product: ProductRef, snapshot: KvSnapshot) -> None:
        """Register one exact remote publication before its install operation runs."""

        if (
            product.kind is not ProductKind.KV
            or product.request_key != snapshot.source_version.request_key
            or product.producer_op_id == snapshot.source_version.producer_op_id
        ):
            raise invalid_descriptor("staged KV publication has an invalid product identity")
        with self._lock:
            existing = self._publication_products.get(product)
            if existing is not None and existing != snapshot:
                raise invalid_descriptor("staged KV publication conflicts with its exact identity")
            self._publication_products[product] = snapshot

    def destination_base(self, session_id: int, destination: str) -> VersionRef | None:
        """Return the exact immutable base currently installed for a destination."""

        with self._lock:
            base = self._destination_bases.get((int(session_id), str(destination)))
            return None if base is None else base[0]

    def validate_conditioning(self, session_id: int, product: ProductRef) -> KvSnapshot:
        """Validate a local published prefix without waiting for its remote copy."""

        with self._lock:
            try:
                snapshot = self._publication_products[product]
            except KeyError:
                raise invalid_descriptor("KV conditioning product is not resident") from None
            if int(product.request_key.session_id) != int(session_id):
                raise invalid_descriptor("KV conditioning product belongs to another session")
            point = snapshot.source_version.point
            if not isinstance(point, FixedPoint) or point.semantic_digest != snapshot.source_digest:
                raise invalid_descriptor("KV conditioning product has no exact source digest")
            entry = self.get(session_id)
            if (
                entry.initialized_len < snapshot.published_extent
                or entry.visible_len < snapshot.published_extent
                or entry.committed_len < snapshot.published_extent
            ):
                raise invalid_descriptor("KV conditioning prefix is not locally committed")
            if self.pool is None:
                raise RuntimeError("KV conditioning validation requires a physical pool")
            pages = ceil_div(snapshot.published_extent, self.pool.block_size)
            if (
                tuple(entry.block_ids[:pages]) != snapshot.block_ids[:pages]
                or entry.group_id != snapshot.group_id
                or entry.scale_identity != snapshot.scale_identity
            ):
                raise invalid_descriptor("KV conditioning mapping does not match its publication")
            return snapshot

    def install_publication(
        self,
        session_id: int,
        source: ProductRef,
        installed_product: ProductRef,
        transport: Transport,
        transferred_tensors: tuple[torch.Tensor, ...] | None = None,
    ) -> KvSnapshot:
        """Install or validate one immutable publication and bind its installed product."""

        snapshot = self.publication(source)
        if (
            installed_product.request_key != source.request_key
            or installed_product.kind is not ProductKind.KV
            or installed_product.producer_op_id == source.producer_op_id
        ):
            raise invalid_descriptor("installed KV product has an invalid identity")
        entry = self.get(session_id)
        local_exact = (
            entry.visible_len == snapshot.published_extent
            and entry.committed_len == snapshot.published_extent
            and tuple(entry.block_ids) == snapshot.block_ids
            and tuple(entry.logical_blocks) == snapshot.logical_blocks
            and entry.mapping_generation == snapshot.mapping_generation
            and entry.scale_identity == snapshot.scale_identity
        )
        if local_exact:
            self._installed_bases[(int(session_id), snapshot.destination)] = (
                snapshot.source_version,
                snapshot.published_extent,
            )
        else:
            self.import_snapshot(
                session_id,
                snapshot,
                transport,
                transferred_tensors=transferred_tensors,
            )
        with self._lock:
            entry = self.get(session_id)
            lease_pages = len(snapshot.logical_blocks)
            local_snapshot = replace(
                snapshot,
                block_ids=tuple(entry.block_ids[:lease_pages]),
                logical_blocks=tuple(entry.logical_blocks[:lease_pages]),
                mapping_generation=entry.mapping_generation,
                scale_identity=entry.scale_identity,
            )
            self._publication_products[source] = local_snapshot
            self._publication_products[installed_product] = local_snapshot
        return local_snapshot

    def validate_installed(self, session_id: int, product: ProductRef) -> KvSnapshot:
        snapshot = self.publication(product)
        installed = self._installed_bases.get((int(session_id), snapshot.destination))
        if installed != (snapshot.source_version, snapshot.published_extent):
            raise invalid_descriptor("KV input is not the exact installed publication")
        return snapshot

    def _retain_published(
        self,
        session_id: int,
        product: ProductRef,
        locators: tuple[str, ...],
        transport: object,
    ) -> None:
        if locators:
            with self._lock:
                self._published.setdefault(int(session_id), []).append(
                    (product, locators, transport)
                )

    def retain_restored_publications(
        self,
        states: Sequence[KvCommittedState],
        transport: Transport,
    ) -> None:
        """Bind restored locator assets to their exact producing operations."""

        retained: list[tuple[int, ProductRef, tuple[str, ...]]] = []
        for state in states:
            for product, snapshot in state.publications:
                if snapshot.locators:
                    retained.append((state.session_id, product, snapshot.locators))
        with self._lock:
            for session_id, product, locators in retained:
                entries = self._published.setdefault(int(session_id), [])
                candidate = (product, locators, transport)
                if candidate not in entries:
                    entries.append(candidate)

    def release_operations(
        self,
        releases: Sequence[tuple[RequestKey, int]],
    ) -> None:
        """Release branch and transport state owned by exact producer operations."""

        identities = {(request_key, int(op_id)) for request_key, op_id in releases}
        if not identities:
            return
        retained: list[tuple[tuple[str, ...], object]] = []
        with self._lock:
            for key in tuple(self._branches):
                owner, _branch = key
                if (owner.request_key, int(owner.producer_op_id)) in identities:
                    self._release_scratch(self._branches.pop(key).block_ids)
            for session_id, entries in tuple(self._published.items()):
                keep = []
                for product, locators, transport in entries:
                    if (product.request_key, int(product.producer_op_id)) in identities:
                        retained.append((locators, transport))
                    else:
                        keep.append((product, locators, transport))
                if keep:
                    self._published[session_id] = keep
                else:
                    self._published.pop(session_id, None)
            for product in tuple(self._publication_products):
                if (product.request_key, int(product.producer_op_id)) in identities:
                    del self._publication_products[product]
        self._release_publication_locators(retained)

    def release_published(self, session_id: int) -> None:
        """Return the device copies one session handed the transport."""

        with self._lock:
            retained = self._published.pop(int(session_id), ())
            for key in [key for key in self._destination_bases if key[0] == int(session_id)]:
                del self._destination_bases[key]
            for key in [key for key in self._installed_bases if key[0] == int(session_id)]:
                del self._installed_bases[key]
            for product in [
                product
                for product in self._publication_products
                if product.request_key.session_id == int(session_id)
            ]:
                del self._publication_products[product]
        if not retained:
            return
        self._release_publication_locators(
            [(locators, transport) for _product, locators, transport in retained]
        )

    @staticmethod
    def _release_publication_locators(
        retained: Sequence[tuple[tuple[str, ...], object]],
    ) -> None:
        for locators, transport in retained:
            release = getattr(transport, "release", None)
            if not callable(release):
                continue
            for encoded in locators:
                release(Locator.from_wire_json(encoded))

    def rewrite_locators(self, session_ids: set[int], replacements: dict[str, str]) -> None:
        """Attach durable fallbacks to every live KV publication descriptor."""

        requested = {int(value) for value in session_ids}
        if not replacements:
            return

        def rewrite(snapshot: KvSnapshot) -> KvSnapshot:
            locators = tuple(replacements.get(raw, raw) for raw in snapshot.locators)
            return (
                snapshot if locators == snapshot.locators else replace(snapshot, locators=locators)
            )

        with self._lock:
            for product, snapshot in tuple(self._publication_products.items()):
                if product.request_key.session_id in requested:
                    self._publication_products[product] = rewrite(snapshot)
            for session_id, entries in tuple(self._published.items()):
                if session_id not in requested:
                    continue
                self._published[session_id] = [
                    (
                        product,
                        tuple(replacements.get(raw, raw) for raw in locators),
                        transport,
                    )
                    for product, locators, transport in entries
                ]

    def import_snapshot(
        self,
        session_id: int,
        snapshot: KvSnapshot,
        transport: Transport,
        *,
        transferred_tensors: tuple[torch.Tensor, ...] | None = None,
    ) -> None:
        """Install a complete transferred snapshot without exposing partial state.

        Every locator is fetched and its tensor geometry is validated before a
        pool write or block-table mutation occurs. Raw pages that overlap the
        prior committed entry are retained until the surrounding ``KvTxn`` is
        finalized, so a later forward, validation, or commit failure can restore
        both metadata and bytes.
        """

        transaction = self.begin_step({int(session_id)})
        try:
            transaction.import_snapshot(
                session_id,
                snapshot,
                transport,
                transferred_tensors=transferred_tensors,
            )
            transaction.prepare()
            transaction.publish()
            transaction.finalize()
        except BaseException:
            transaction.rollback()
            raise

    def begin_step(self, request_ids: set[int]) -> KvTxn:
        return KvTxn(self, frozenset(int(value) for value in request_ids))

    def _import_snapshot(
        self,
        session_id: int,
        snapshot: KvSnapshot,
        transport: Transport,
        transaction: KvTxn,
        transferred_tensors: tuple[torch.Tensor, ...] | None = None,
    ) -> None:
        if self.pool is None:
            raise RuntimeError("KV snapshot import requires a physical pool")
        suffix = snapshot.published_extent - snapshot.base_extent
        expected_locators = 2 * self.pool.num_layers if suffix else 0
        if len(snapshot.locators) != expected_locators:
            raise invalid_descriptor("published KV locator count does not match cache layers")
        from .transfer import Locator, fetch_locator

        with self._lock:
            resident = self.get(session_id)
            if resident.group_id != snapshot.group_id:
                raise invalid_descriptor("published KV group does not match the local session")
            installed = self._installed_bases.get((int(session_id), snapshot.destination))
            if snapshot.base_version is None:
                if installed is not None or snapshot.base_extent != 0:
                    raise invalid_descriptor("KV installation base does not match destination")
            elif installed != (snapshot.base_version, snapshot.base_extent):
                raise invalid_descriptor("KV installation base does not match destination")
            if resident.visible_len != snapshot.base_extent:
                raise invalid_descriptor("KV installation resident extent does not match base")
            if resident.group_id != snapshot.group_id:
                raise invalid_descriptor("published KV group does not match the local session")
            base_pages = ceil_div(snapshot.base_extent, self.pool.block_size)
            if tuple(resident.logical_blocks[:base_pages]) != snapshot.logical_blocks[:base_pages]:
                raise invalid_descriptor("KV installation logical base does not match")
            if resident.scale_identity != snapshot.scale_identity:
                raise invalid_descriptor("KV installation scale identity does not match base")

        if transferred_tensors is not None and len(transferred_tensors) != len(snapshot.locators):
            raise invalid_descriptor("prepared KV transfer tensor count does not match locators")
        tensors: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in range(self.pool.num_layers):
            key = (
                fetch_locator(transport, Locator.from_wire_json(snapshot.locators[2 * layer]))
                if transferred_tensors is None
                else transferred_tensors[2 * layer]
            )
            value = (
                fetch_locator(
                    transport,
                    Locator.from_wire_json(snapshot.locators[2 * layer + 1]),
                )
                if transferred_tensors is None
                else transferred_tensors[2 * layer + 1]
            )
            expected = (suffix, self.pool.n_kv, self.pool.head_dim)
            if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                raise invalid_descriptor("published KV transport returned a non-tensor value")
            if tuple(key.shape) != expected or tuple(value.shape) != expected:
                raise invalid_descriptor(f"published KV tensors must have shape {expected}")
            if key.dtype != self.pool.dtype or value.dtype != self.pool.dtype:
                raise invalid_descriptor(
                    f"published KV tensors must use compute dtype {self.pool.dtype}"
                )
            tensors.append((key, value))

        with self._lock:
            entry = self.get(session_id)
            if entry.group_id != snapshot.group_id:
                raise invalid_descriptor("published KV group does not match the local session")
            if (
                tuple(entry.logical_blocks[: len(snapshot.logical_blocks)])
                != snapshot.logical_blocks
            ):
                raise invalid_descriptor("KV installation logical lease does not match reservation")
            blocks = entry.block_ids[: len(snapshot.logical_blocks)]
            self._validate_blocks(session_id, blocks, snapshot.published_extent)
            write_page = snapshot.base_extent // self.pool.block_size
            transaction._retain_pages(blocks[write_page:])
            for layer, (key, value) in enumerate(tensors):
                self.pool.write(
                    layer,
                    blocks,
                    start=snapshot.base_extent,
                    k=key.to(self.pool.k.device),
                    v=value.to(self.pool.v.device),
                )
            entry.reserved_len = len(entry.block_ids) * self.pool.block_size
            entry.initialized_len = snapshot.published_extent
            entry.visible_len = snapshot.published_extent
            entry.committed_len = snapshot.published_extent
            entry.published_by_destination[snapshot.destination] = snapshot.published_extent
            entry.scale_identity = snapshot.scale_identity
            entry.prefix_len = min(entry.prefix_len, entry.visible_len)
            self._installed_bases[(int(session_id), snapshot.destination)] = (
                snapshot.source_version,
                snapshot.published_extent,
            )

    def copy(self, copies: Sequence[tuple[int, int]]) -> None:
        if self.pool is None:
            raise RuntimeError("KV copy requires a physical pool")
        with self._lock:
            for source, target in copies:
                source = int(source)
                target = int(target)
                self.pool.validate_block_ids((source, target))
                if source == target:
                    continue
                self.pool.k[:, target].copy_(self.pool.k[:, source])
                self.pool.v[:, target].copy_(self.pool.v[:, source])
                for name in ("k_scale", "v_scale", "k_scale_set", "v_scale_set"):
                    value = getattr(self.pool, name)
                    if value is not None:
                        value[:, target].copy_(value[:, source])

    def drop(self, session_id: int) -> None:
        self.release_published(session_id)
        with self._lock:
            entry = self._entries.pop(int(session_id), None)
            if entry is None:
                pass
            else:
                self._unregister(int(session_id), entry.block_ids)
            keys = [
                key
                for key in self._branches
                if int(key[0].request_key.session_id) == int(session_id)
            ]
            for key in keys:
                self._release_scratch(self._branches.pop(key).block_ids)

    def snapshot_committed(
        self,
        request_ids: set[int],
        committed_lengths: Mapping[int, int] | None = None,
    ) -> tuple[KvCommittedState, ...]:
        requested = sorted(int(value) for value in request_ids)
        with self._lock:
            states: list[KvCommittedState] = []
            for session_id in requested:
                entry = self._entries.get(session_id)
                if entry is None:
                    raise invalid_descriptor(f"session {session_id} has no KV state")
                length = (
                    entry.length
                    if committed_lengths is None
                    else int(committed_lengths[session_id])
                )
                if length < entry.prefix_len or length > entry.length:
                    raise invalid_descriptor("committed KV snapshot extent is invalid")
                pages = self._snapshot_pages(self.pool, entry.block_ids)
                branches = tuple(
                    KvBranchState(
                        owner=key[0],
                        branch=key[1],
                        length=branch.length,
                        block_count=len(branch.block_ids),
                        pages=self._require_pages(
                            self._snapshot_pages(self.pool, branch.block_ids),
                            "branch KV",
                        ),
                    )
                    for key, branch in sorted(
                        self._branches.items(),
                        key=lambda item: (
                            int(item[0][0].producer_op_id),
                            int(item[0][0].output_index),
                            int(item[0][0].generation),
                            item[0][1],
                        ),
                    )
                    if int(key[0].request_key.session_id) == session_id
                )
                states.append(
                    KvCommittedState(
                        session_id=session_id,
                        block_ids=tuple(entry.block_ids),
                        logical_blocks=tuple(entry.logical_blocks),
                        prefix_len=entry.prefix_len,
                        length=length,
                        group_id=entry.group_id,
                        reserved_len=entry.reserved_len,
                        initialized_len=entry.initialized_len,
                        committed_len=min(entry.committed_len, length),
                        published_by_destination=tuple(
                            sorted(entry.published_by_destination.items())
                        ),
                        mapping_generation=entry.mapping_generation,
                        scale_identity=entry.scale_identity,
                        pages=pages,
                        branches=branches,
                        publications=tuple(
                            sorted(
                                (
                                    (product, publication)
                                    for product, publication in self._publication_products.items()
                                    if product.request_key.session_id == session_id
                                ),
                                key=lambda item: (
                                    int(item[0].producer_op_id),
                                    int(item[0].output_index),
                                    int(item[0].generation),
                                ),
                            )
                        ),
                        destination_bases=tuple(
                            sorted(
                                (
                                    (
                                        destination,
                                        version,
                                        extent,
                                        blocks,
                                        group_id,
                                        scale_identity,
                                    )
                                    for (owner, destination), (
                                        version,
                                        extent,
                                        blocks,
                                        group_id,
                                        scale_identity,
                                    ) in self._destination_bases.items()
                                    if owner == session_id
                                ),
                                key=lambda item: item[0],
                            )
                        ),
                        installed_bases=tuple(
                            sorted(
                                (
                                    (destination, version, extent)
                                    for (owner, destination), (
                                        version,
                                        extent,
                                    ) in self._installed_bases.items()
                                    if owner == session_id
                                ),
                                key=lambda item: item[0],
                            )
                        ),
                    )
                )
            return tuple(states)

    def restore_committed(
        self,
        states: Sequence[KvCommittedState],
        request_ids: set[int] | None = None,
    ) -> None:
        staged = tuple(states)
        staged_ids = {int(state.session_id) for state in staged}
        if len(staged_ids) != len(staged):
            raise invalid_descriptor("KV snapshot repeats a session identity")
        session_ids = staged_ids if request_ids is None else {int(value) for value in request_ids}
        if not staged_ids <= session_ids:
            raise invalid_descriptor("KV snapshot contains an undeclared session")
        self._validate_committed_states(staged)
        with self._lock:
            prior = self.snapshot_committed(session_ids & set(self._entries))
            try:
                self._install_committed(staged, session_ids)
            except BaseException:
                self._install_committed(prior, session_ids)
                raise

    def _validate_committed_states(self, states: Sequence[KvCommittedState]) -> None:
        main_pages: dict[int, tuple[KvPageState, int]] = {}
        for state in states:
            if (
                min(
                    state.prefix_len,
                    state.length,
                    state.group_id,
                    state.reserved_len,
                    state.initialized_len,
                    state.committed_len,
                    state.mapping_generation,
                )
                < 0
            ):
                raise invalid_descriptor("KV snapshot metadata must be non-negative")
            published_extent = max(
                (extent for _destination, extent in state.published_by_destination), default=0
            )
            if (
                not published_extent
                <= state.committed_len
                <= state.length
                <= state.initialized_len
                <= state.reserved_len
            ):
                raise invalid_descriptor("KV snapshot extents are not monotonically contained")
            if len(state.logical_blocks) != len(state.block_ids) or len(
                set(state.logical_blocks)
            ) != len(state.logical_blocks):
                raise invalid_descriptor("KV snapshot logical lease is invalid")
            if any(logical_block < 0 for logical_block in state.logical_blocks):
                raise invalid_descriptor("KV snapshot logical lease is invalid")
            if state.prefix_len > state.committed_len:
                raise invalid_descriptor("KV snapshot prefix exceeds its committed length")
            if not state.scale_identity or state.mapping_generation < 1:
                raise invalid_descriptor("KV snapshot mapping metadata is invalid")
            destinations = [destination for destination, _extent in state.published_by_destination]
            if any(not destination for destination in destinations) or len(
                set(destinations)
            ) != len(destinations):
                raise invalid_descriptor("KV snapshot publication destinations are invalid")
            publication_refs: set[ProductRef] = set()
            for product, publication in state.publications:
                publication_pages_match = not publication.block_ids
                if self.pool is not None:
                    publication_pages = ceil_div(
                        publication.published_extent,
                        self.pool.block_size,
                    )
                    publication_pages_match = (
                        tuple(state.block_ids[:publication_pages])
                        == publication.block_ids[:publication_pages]
                    )
                if (
                    product in publication_refs
                    or product.request_key.session_id != state.session_id
                    or product.kind is not ProductKind.KV
                    or publication.source_version.request_key != product.request_key
                    or publication.published_extent > state.committed_len
                    or not publication_pages_match
                    or tuple(state.logical_blocks[: len(publication.logical_blocks)])
                    != publication.logical_blocks
                ):
                    raise invalid_descriptor("KV snapshot publication identity is invalid")
                publication_refs.add(product)
            destination_base_names = [value[0] for value in state.destination_bases]
            installed_base_names = [value[0] for value in state.installed_bases]
            if (
                any(not value for value in destination_base_names + installed_base_names)
                or len(set(destination_base_names)) != len(destination_base_names)
                or len(set(installed_base_names)) != len(installed_base_names)
            ):
                raise invalid_descriptor("KV snapshot exact destination bases are invalid")
            for (
                _destination,
                version,
                extent,
                blocks,
                group_id,
                scale_identity,
            ) in state.destination_bases:
                base_pages_match = not blocks
                if self.pool is not None:
                    base_pages = ceil_div(extent, self.pool.block_size)
                    base_pages_match = tuple(state.block_ids[:base_pages]) == blocks[:base_pages]
                if (
                    version.request_key.session_id != state.session_id
                    or not isinstance(version.point, FixedPoint)
                    or extent < 0
                    or extent > state.committed_len
                    or group_id != state.group_id
                    or scale_identity != state.scale_identity
                    or not base_pages_match
                ):
                    raise invalid_descriptor("KV snapshot producer destination base is invalid")
            for _destination, version, extent in state.installed_bases:
                if (
                    version.request_key.session_id != state.session_id
                    or not isinstance(version.point, FixedPoint)
                    or extent < 0
                    or extent > state.committed_len
                ):
                    raise invalid_descriptor("KV snapshot consumer destination base is invalid")
            if self.pool is None:
                if state.block_ids or state.reserved_len or state.pages is not None:
                    raise invalid_descriptor("KV snapshot requires an absent physical pool")
            else:
                if state.reserved_len != len(state.block_ids) * self.pool.block_size:
                    raise invalid_descriptor(
                        "KV snapshot reservation does not match block capacity"
                    )
                if any(
                    logical_block >= self.pool.leasable_num_blocks
                    for logical_block in state.logical_blocks
                ):
                    raise invalid_descriptor("KV snapshot logical lease exceeds capacity")
                self.pool.validate_block_ids(state.block_ids)
                if any(value >= self.pool.leasable_num_blocks for value in state.block_ids):
                    raise invalid_descriptor("KV snapshot uses a block outside the leased range")
                if state.block_ids:
                    pages = self._require_pages(state.pages, "request KV")
                    self._validate_page_state(self.pool, pages, len(state.block_ids))
                    for index, block_id in enumerate(state.block_ids):
                        previous = main_pages.get(block_id)
                        if previous is None:
                            main_pages[block_id] = (pages, index)
                        elif not self._page_equal(previous[0], previous[1], pages, index):
                            raise invalid_descriptor(
                                f"KV snapshot gives shared block {block_id} conflicting contents"
                            )
                elif state.pages is not None:
                    raise invalid_descriptor("empty KV snapshot must not contain page tensors")
            seen_branches: set[tuple[ProductRef, str]] = set()
            for branch in state.branches:
                key = (branch.owner, branch.branch)
                if key in seen_branches:
                    raise invalid_descriptor("KV snapshot repeats a branch")
                seen_branches.add(key)
                if (
                    branch.owner.request_key.session_id != state.session_id
                    or branch.owner.kind is not ProductKind.LATENT
                    or branch.owner.storage_class is not StorageClass.LATENT_ARENA
                    or branch.owner.generation < 1
                    or branch.length < 0
                    or branch.block_count < 1
                ):
                    raise invalid_descriptor("KV snapshot branch is invalid")
                if not branch.branch:
                    raise invalid_descriptor("KV snapshot branch name is empty")
                if self.pool is None:
                    raise invalid_descriptor("KV snapshot requires an absent physical pool")
                if branch.length > branch.block_count * self.pool.block_size:
                    raise invalid_descriptor("KV snapshot branch length exceeds capacity")
                self._validate_page_state(self.pool, branch.pages, branch.block_count)

    def _install_committed(
        self,
        states: Sequence[KvCommittedState],
        session_ids: set[int],
    ) -> None:
        for branch_key in [
            key for key in self._branches if key[0].request_key.session_id in session_ids
        ]:
            self._release_scratch(self._branches.pop(branch_key).block_ids)
        for session_id in session_ids:
            current = self._entries.pop(session_id, None)
            if current is not None:
                self._unregister(session_id, current.block_ids)
        for product in tuple(self._publication_products):
            if product.request_key.session_id in session_ids:
                del self._publication_products[product]
        for destination_key in tuple(self._destination_bases):
            if destination_key[0] in session_ids:
                del self._destination_bases[destination_key]
        for installed_key in tuple(self._installed_bases):
            if installed_key[0] in session_ids:
                del self._installed_bases[installed_key]

        block_sources: dict[int, tuple[KvPageState, int]] = {}
        for state in states:
            physical_blocks = []
            for logical_block in state.logical_blocks:
                key = int(logical_block)
                group = self._logical_groups.get(key)
                if group is not None and group != int(state.group_id):
                    raise invalid_descriptor("KV logical block belongs to another cache group")
                page = self._logical_pages.get(key)
                if page is None:
                    page = self._allocate_logical_page(key, int(state.group_id))
                physical_blocks.append(page)
            entry = KvEntry(
                block_ids=physical_blocks,
                logical_blocks=list(state.logical_blocks),
                prefix_len=state.prefix_len,
                group_id=state.group_id,
                reserved_len=state.reserved_len,
                initialized_len=state.initialized_len,
                visible_len=state.length,
                committed_len=state.committed_len,
                published_by_destination=dict(state.published_by_destination),
                mapping_generation=state.mapping_generation
                + int(tuple(physical_blocks) != state.block_ids),
                scale_identity=state.scale_identity,
            )
            block_size = 0 if self.pool is None else self.pool.block_size
            shared_boundary = entry.prefix_len // block_size if block_size else 0
            for index, block_id in enumerate(entry.block_ids):
                holders = self._holders.get(block_id, set())
                if holders and index >= shared_boundary:
                    raise invalid_descriptor(
                        f"KV snapshot writable block {block_id} is held by another session"
                    )
            self._entries[state.session_id] = entry
            self._register(state.session_id, entry.block_ids)
            if state.pages is not None:
                for index, block_id in enumerate(entry.block_ids):
                    block_sources.setdefault(block_id, (state.pages, index))
            self._publication_products.update(
                {
                    product: replace(
                        publication,
                        block_ids=tuple(entry.block_ids[: len(publication.logical_blocks)]),
                        logical_blocks=tuple(
                            entry.logical_blocks[: len(publication.logical_blocks)]
                        ),
                        mapping_generation=entry.mapping_generation,
                        scale_identity=entry.scale_identity,
                    )
                    for product, publication in state.publications
                }
            )
            self._destination_bases.update(
                {
                    (state.session_id, destination): (
                        version,
                        extent,
                        tuple(entry.block_ids[: ceil_div(extent, block_size) if block_size else 0]),
                        group_id,
                        scale_identity,
                    )
                    for destination, version, extent, _blocks, group_id, scale_identity in (
                        state.destination_bases
                    )
                }
            )
            self._installed_bases.update(
                {
                    (state.session_id, destination): (version, extent)
                    for destination, version, extent in state.installed_bases
                }
            )

        if self.pool is not None:
            for block_id, (pages, index) in block_sources.items():
                self._copy_page_to_pool(self.pool, block_id, pages, index)

        for state in states:
            for branch in state.branches:
                pool = self.pool
                if pool is None:
                    raise RuntimeError("validated physical KV pool disappeared")
                block_ids = tuple(pool.allocate_branch_blocks(branch.block_count))
                if len(block_ids) != branch.block_count:
                    if block_ids:
                        pool.release_branch_blocks(block_ids)
                    raise RuntimeError("branch KV allocator returned an incomplete lease")
                try:
                    for index, block_id in enumerate(block_ids):
                        self._copy_page_to_pool(pool, block_id, branch.pages, index)
                except BaseException:
                    pool.release_branch_blocks(block_ids)
                    raise
                self._branches[(branch.owner, branch.branch)] = KvEntry(
                    block_ids=list(block_ids),
                    reserved_len=len(block_ids) * pool.block_size,
                    initialized_len=branch.length,
                    visible_len=branch.length,
                    committed_len=branch.length,
                    scale_identity=self._scale_identity(),
                )

    @staticmethod
    def _snapshot_pages(
        pool: PagedKVPool | None,
        block_ids: Sequence[int],
    ) -> KvPageState | None:
        if not block_ids:
            return None
        if pool is None:
            raise RuntimeError("KV metadata references an absent physical pool")
        index = torch.tensor(tuple(block_ids), dtype=torch.long, device=pool.k.device)

        def take(value: torch.Tensor | None) -> torch.Tensor | None:
            if value is None:
                return None
            return value.index_select(1, index).detach().cpu().contiguous()

        return KvPageState(
            key=pool.k.index_select(1, index).detach().cpu().contiguous(),
            value=pool.v.index_select(1, index).detach().cpu().contiguous(),
            key_scale=take(pool.k_scale),
            value_scale=take(pool.v_scale),
            key_scale_set=take(pool.k_scale_set),
            value_scale_set=take(pool.v_scale_set),
        )

    @staticmethod
    def _validate_page_state(
        pool: PagedKVPool,
        pages: KvPageState,
        block_count: int,
    ) -> None:
        expected = (pool.num_layers, block_count, pool.block_size, pool.n_kv, pool.head_dim)
        if tuple(pages.key.shape) != expected or tuple(pages.value.shape) != expected:
            raise invalid_descriptor(f"KV snapshot pages must have shape {expected}")
        if pages.key.dtype != pool.k.dtype or pages.value.dtype != pool.v.dtype:
            raise invalid_descriptor("KV snapshot page dtype does not match physical storage")
        for name, value, target in (
            ("key_scale", pages.key_scale, pool.k_scale),
            ("value_scale", pages.value_scale, pool.v_scale),
            ("key_scale_set", pages.key_scale_set, pool.k_scale_set),
            ("value_scale_set", pages.value_scale_set, pool.v_scale_set),
        ):
            if (value is None) != (target is None):
                raise invalid_descriptor(f"KV snapshot {name} presence does not match storage")
            if value is not None and target is not None:
                expected_aux = (target.shape[0], block_count, *target.shape[2:])
                if tuple(value.shape) != expected_aux or value.dtype != target.dtype:
                    raise invalid_descriptor(f"KV snapshot {name} geometry does not match storage")

    @staticmethod
    def _copy_page_to_pool(
        pool: PagedKVPool,
        block_id: int,
        pages: KvPageState,
        index: int,
    ) -> None:
        pool.k[:, block_id].copy_(pages.key[:, index].to(device=pool.k.device))
        pool.v[:, block_id].copy_(pages.value[:, index].to(device=pool.v.device))
        for source, target in (
            (pages.key_scale, pool.k_scale),
            (pages.value_scale, pool.v_scale),
            (pages.key_scale_set, pool.k_scale_set),
            (pages.value_scale_set, pool.v_scale_set),
        ):
            if source is not None and target is not None:
                target[:, block_id].copy_(source[:, index].to(device=target.device))

    @staticmethod
    def _page_equal(
        left: KvPageState,
        left_index: int,
        right: KvPageState,
        right_index: int,
    ) -> bool:
        for lhs, rhs in (
            (left.key, right.key),
            (left.value, right.value),
            (left.key_scale, right.key_scale),
            (left.value_scale, right.value_scale),
            (left.key_scale_set, right.key_scale_set),
            (left.value_scale_set, right.value_scale_set),
        ):
            if lhs is None or rhs is None:
                if lhs is not rhs:
                    return False
            elif not torch.equal(lhs[:, left_index], rhs[:, right_index]):
                return False
        return True

    @staticmethod
    def _require_pages(value: KvPageState | None, label: str) -> KvPageState:
        if value is None:
            raise invalid_descriptor(f"{label} snapshot pages are missing")
        return value

    def snapshot_entries(self, request_ids: set[int]) -> _KvEntrySnapshot:
        """Bound rollback state for the main request entries only."""

        requested = {int(value) for value in request_ids}
        with self._lock:
            return _KvEntrySnapshot(entries=self._snapshot_entries_locked(requested))

    def snapshot_auxiliary(self, request_ids: set[int]) -> _KvAuxiliarySnapshot:
        """Bound branch and transport state before an operation first mutates it."""

        requested = {int(value) for value in request_ids}
        with self._lock:
            return self._snapshot_auxiliary_locked(requested)

    def snapshot_requests(self, request_ids: set[int]) -> _KvSnapshot:
        requested = {int(value) for value in request_ids}
        with self._lock:
            entries = self._snapshot_entries_locked(requested)
            auxiliary = self._snapshot_auxiliary_locked(requested)
            return _KvSnapshot(
                entries=entries,
                branches=auxiliary.branches,
                publications=auxiliary.publications,
                destination_bases=auxiliary.destination_bases,
                installed_bases=auxiliary.installed_bases,
                retained_counts=auxiliary.retained_counts,
            )

    def _snapshot_entries_locked(
        self,
        request_ids: set[int],
    ) -> dict[int, _KvEntryState]:
        result: dict[int, _KvEntryState] = {}
        for session_id in request_ids:
            entry = self._entries.get(session_id)
            if entry is None:
                result[session_id] = (False, (), (), 0, 0, 0, 0, 0, 0, {}, 0, "none")
            else:
                result[session_id] = (
                    True,
                    tuple(entry.block_ids),
                    tuple(entry.logical_blocks),
                    entry.prefix_len,
                    entry.group_id,
                    entry.reserved_len,
                    entry.initialized_len,
                    entry.visible_len,
                    entry.committed_len,
                    dict(entry.published_by_destination),
                    entry.mapping_generation,
                    entry.scale_identity,
                )
        return result

    def _snapshot_auxiliary_locked(self, request_ids: set[int]) -> _KvAuxiliarySnapshot:
        return _KvAuxiliarySnapshot(
            branches={
                key: (tuple(entry.block_ids), entry.length)
                for key, entry in self._branches.items()
                if int(key[0].request_key.session_id) in request_ids
            },
            publications={
                product: publication
                for product, publication in self._publication_products.items()
                if product.request_key.session_id in request_ids
            },
            destination_bases={
                key: value
                for key, value in self._destination_bases.items()
                if key[0] in request_ids
            },
            installed_bases={
                key: value for key, value in self._installed_bases.items() if key[0] in request_ids
            },
            retained_counts={
                session_id: len(self._published.get(session_id, ())) for session_id in request_ids
            },
        )

    def restore_requests(
        self,
        request_ids: set[int],
        snapshot: object,
    ) -> None:
        if not isinstance(snapshot, _KvSnapshot):
            raise RuntimeError("KV snapshot has an invalid type")
        requested = {int(value) for value in request_ids}
        with self._lock:
            self._restore_entries_locked(requested, snapshot.entries)
            self._restore_auxiliary_locked(
                requested,
                _KvAuxiliarySnapshot(
                    branches=snapshot.branches,
                    publications=snapshot.publications,
                    destination_bases=snapshot.destination_bases,
                    installed_bases=snapshot.installed_bases,
                    retained_counts=snapshot.retained_counts,
                ),
            )

    def restore_entries(self, request_ids: set[int], snapshot: _KvEntrySnapshot) -> None:
        requested = {int(value) for value in request_ids}
        with self._lock:
            self._restore_entries_locked(requested, snapshot.entries)

    def restore_auxiliary(
        self,
        request_ids: set[int],
        snapshot: _KvAuxiliarySnapshot,
    ) -> None:
        requested = {int(value) for value in request_ids}
        with self._lock:
            self._restore_auxiliary_locked(requested, snapshot)

    def _restore_entries_locked(
        self,
        request_ids: set[int],
        entries: dict[
            int,
            tuple[
                bool,
                tuple[int, ...],
                tuple[int, ...],
                int,
                int,
                int,
                int,
                int,
                int,
                dict[str, int],
                int,
                str,
            ],
        ],
    ) -> None:
        for session_id in request_ids:
            current = self._entries.pop(session_id, None)
            if current is not None:
                self._unregister(session_id, current.block_ids)
            (
                existed,
                blocks,
                logical_blocks,
                prefix_len,
                group_id,
                reserved_len,
                initialized_len,
                visible_len,
                committed_len,
                published,
                mapping_generation,
                scale_identity,
            ) = entries.get(
                session_id,
                (False, (), (), 0, 0, 0, 0, 0, 0, {}, 0, "none"),
            )
            if existed:
                entry = KvEntry(
                    block_ids=list(blocks),
                    logical_blocks=list(logical_blocks),
                    prefix_len=prefix_len,
                    group_id=group_id,
                    reserved_len=reserved_len,
                    initialized_len=initialized_len,
                    visible_len=visible_len,
                    committed_len=committed_len,
                    published_by_destination=dict(published),
                    mapping_generation=mapping_generation,
                    scale_identity=scale_identity,
                )
                self._entries[session_id] = entry
                self._register(session_id, entry.block_ids)

    def _restore_auxiliary_locked(
        self,
        request_ids: set[int],
        snapshot: _KvAuxiliarySnapshot,
    ) -> None:
        current_keys = [
            key for key in self._branches if int(key[0].request_key.session_id) in request_ids
        ]
        for key in current_keys:
            current = self._branches.pop(key)
            prior_blocks = set(snapshot.branches.get(key, ((), 0))[0])
            self._release_scratch(
                tuple(block for block in current.block_ids if block not in prior_blocks)
            )
        for key, (blocks, length) in snapshot.branches.items():
            if int(key[0].request_key.session_id) in request_ids:
                self._branches[key] = KvEntry(
                    block_ids=list(blocks),
                    reserved_len=len(blocks) * (0 if self.pool is None else self.pool.block_size),
                    initialized_len=length,
                    visible_len=length,
                    committed_len=length,
                    scale_identity=self._scale_identity(),
                )
        for product in [
            product
            for product in self._publication_products
            if product.request_key.session_id in request_ids
        ]:
            del self._publication_products[product]
        self._publication_products.update(snapshot.publications)
        for destination_key in [key for key in self._destination_bases if key[0] in request_ids]:
            del self._destination_bases[destination_key]
        self._destination_bases.update(snapshot.destination_bases)
        for installed_key in [key for key in self._installed_bases if key[0] in request_ids]:
            del self._installed_bases[installed_key]
        self._installed_bases.update(snapshot.installed_bases)
        for session_id in request_ids:
            retained = self._published.get(session_id)
            if retained is not None:
                del retained[snapshot.retained_counts.get(session_id, 0) :]
                if not retained:
                    self._published.pop(session_id, None)

    def _ensure_scratch_capacity(self, entry: KvEntry, tokens: int) -> None:
        pool = self.pool
        if pool is None:
            raise RuntimeError("branch KV capacity requires a physical pool")
        required = ceil_div(max(0, int(tokens)), pool.block_size)
        missing = required - len(entry.block_ids)
        if missing > 0:
            allocated = pool.allocate_branch_blocks(missing)
            if len(allocated) != missing:
                if allocated:
                    pool.release_branch_blocks(allocated)
                raise RuntimeError("branch KV allocator returned an incomplete lease")
            entry.block_ids.extend(int(value) for value in allocated)
            entry.reserved_len = len(entry.block_ids) * pool.block_size
            entry.mapping_generation += 1

    def _scale_identity(self) -> str:
        pool = self.pool
        if pool is None:
            return "none"
        return ":".join(
            (
                str(pool.store_dtype),
                str(pool.dtype),
                str(pool.block_size),
                str(pool.n_kv),
                str(pool.head_dim),
                "quantized" if pool.is_quantized else "direct",
            )
        )

    def _release_scratch(self, blocks: Sequence[int]) -> None:
        if blocks and self.pool is not None:
            self.pool.release_branch_blocks(tuple(int(value) for value in blocks))

    def _validate_blocks(self, session_id: int, blocks: Sequence[int], boundary: int) -> None:
        if len(set(blocks)) != len(blocks):
            raise invalid_descriptor(f"session {session_id} KV lease repeats a block")
        if self.pool is not None and any(
            value < 0 or value >= self.pool.leasable_num_blocks for value in blocks
        ):
            raise invalid_descriptor(f"session {session_id} KV lease is outside the pool")
        block_size = 0 if self.pool is None else self.pool.block_size
        shared_prefix_blocks = boundary // block_size if block_size else 0
        for index, block in enumerate(blocks):
            other_holders = self._holders.get(int(block), set()) - {int(session_id)}
            if other_holders and index >= shared_prefix_blocks:
                raise invalid_descriptor(
                    f"session {session_id} receives writable KV block {block} held by another session"
                )

    def _register(self, session_id: int, blocks: Sequence[int]) -> None:
        for block in blocks:
            self._holders.setdefault(int(block), set()).add(int(session_id))

    def _unregister(self, session_id: int, blocks: Sequence[int]) -> None:
        for block in blocks:
            holders = self._holders.get(int(block))
            if holders is None:
                continue
            holders.discard(int(session_id))
            if not holders:
                del self._holders[int(block)]


class KvTxn:
    """Rollback scope for KV metadata and the rare imported-page overwrite."""

    def __init__(self, store: KvStore, session_ids: frozenset[int]) -> None:
        self._store = store
        self._session_ids = session_ids
        self._entry_snapshot = store.snapshot_entries(set(session_ids))
        self._auxiliary_snapshot: _KvAuxiliarySnapshot | None = None
        self._retained: dict[int, _RetainedPage] = {}
        self._logical_page_undo: list[_LogicalPageUndo] = []
        self._closed = False

    def view(
        self,
        session_ids: Sequence[int],
        *,
        query_lens: Sequence[int] | None = None,
    ) -> KvBatchView:
        self._require_open()
        if any(int(value) not in self._session_ids for value in session_ids):
            raise RuntimeError("KV view includes a session outside this step")
        return self._store.view(session_ids, query_lens=query_lens)

    def entries(self, session_ids: Sequence[int]) -> tuple[KvEntry, ...]:
        self._require_open()
        requested = tuple(int(value) for value in session_ids)
        if any(value not in self._session_ids for value in requested):
            raise RuntimeError("KV entry batch includes a session outside this step")
        with self._store._lock:
            try:
                return tuple(self._store._entries[value] for value in requested)
            except KeyError as error:
                raise invalid_descriptor(f"session {error.args[0]} has no KV state") from None

    def view_entries(
        self,
        entries: Sequence[KvEntry],
        *,
        query_lens: Sequence[int] | None = None,
    ) -> KvBatchView:
        self._require_open()
        return self._store.view_entries(entries, query_lens=query_lens)

    def packed_view(self, rows: Sequence[tuple[KvEntry, int, bool]]) -> _PackedKvView:
        self._require_open()
        return self._store.packed_view(rows)

    def scratch_entry(
        self,
        owner: ProductRef,
        branch: str,
        *,
        capacity_tokens: int,
        copy_conditioning: bool,
    ) -> tuple[KvEntry, bool]:
        self._require_open()
        if int(owner.request_key.session_id) not in self._session_ids:
            raise RuntimeError("scratch KV entry targets a session outside this step")
        self._capture_auxiliary()
        return self._store.scratch_entry(
            owner,
            branch,
            capacity_tokens=capacity_tokens,
            copy_conditioning=copy_conditioning,
        )

    def rebind_scratch_owner(self, source: ProductRef, target: ProductRef) -> None:
        self._require_open()
        if int(source.request_key.session_id) not in self._session_ids:
            raise RuntimeError("scratch KV source targets a session outside this step")
        if int(target.request_key.session_id) not in self._session_ids:
            raise RuntimeError("scratch KV target targets a session outside this step")
        self._capture_auxiliary()
        self._store.rebind_scratch_owner(source, target)

    def release_scratch_owner(self, owner: ProductRef) -> None:
        self._require_open()
        if int(owner.request_key.session_id) not in self._session_ids:
            raise RuntimeError("scratch KV owner targets a session outside this step")
        self._capture_auxiliary()
        self._store.release_scratch_owner(owner)

    def advance_entry(self, entry: KvEntry, tokens: int) -> None:
        self._require_open()
        self._capture_auxiliary()
        self._store.advance_entry(entry, tokens)

    def advance(self, session_id: int, tokens: int) -> None:
        self._require_open()
        if int(session_id) not in self._session_ids:
            raise RuntimeError("KV advance targets a session outside this step")
        self._store.advance(int(session_id), int(tokens))

    def initialize(self, session_id: int, tokens: int) -> int:
        self._require_open()
        if int(session_id) not in self._session_ids:
            raise RuntimeError("KV initialization targets a session outside this step")
        return self._store.initialize(int(session_id), int(tokens))

    def select(self, session_id: int, length: int) -> None:
        self._require_open()
        if int(session_id) not in self._session_ids:
            raise RuntimeError("KV selection targets a session outside this step")
        self._store.select(int(session_id), int(length))

    def commit(self, session_id: int, length: int) -> None:
        self._require_open()
        if int(session_id) not in self._session_ids:
            raise RuntimeError("KV commit targets a session outside this step")
        self._store.commit(int(session_id), int(length))

    def extents(self, session_id: int) -> KvExtents:
        self._require_open()
        if int(session_id) not in self._session_ids:
            raise RuntimeError("KV extent query targets a session outside this step")
        return self._store.get(int(session_id)).extents()

    def reserve_logical_page_delta(
        self,
        request_key: RequestKey,
        logical_page_delta: Sequence[int],
        *,
        expected_capacity_pages: int,
    ) -> None:
        self._require_open()
        if int(request_key.session_id) not in self._session_ids:
            raise RuntimeError("KV reservation targets a session outside this step")
        self._logical_page_undo.extend(
            self._store.reserve_logical_page_delta(
                request_key,
                logical_page_delta,
                expected_capacity_pages=expected_capacity_pages,
            )
        )

    def import_snapshot(
        self,
        session_id: int,
        snapshot: KvSnapshot,
        transport: Transport,
        *,
        transferred_tensors: tuple[torch.Tensor, ...] | None = None,
    ) -> None:
        self._require_open()
        if int(session_id) not in self._session_ids:
            raise RuntimeError("KV import targets a session outside this step")
        self._capture_auxiliary()
        self._store._import_snapshot(
            int(session_id),
            snapshot,
            transport,
            self,
            transferred_tensors,
        )

    def stage_publication(self, product: ProductRef, snapshot: KvSnapshot) -> None:
        self._require_open()
        self._capture_auxiliary()
        self._store.stage_publication(product, snapshot)

    def destination_base(self, session_id: int, destination: str) -> VersionRef | None:
        self._require_open()
        return self._store.destination_base(session_id, destination)

    def publish_kv(
        self,
        session_id: int,
        *,
        source_version: VersionRef,
        source_digest: str,
        destination: str,
        expected_base: VersionRef | None,
        product: ProductRef,
        transport: object | None,
    ) -> KvSnapshot:
        self._require_open()
        if int(session_id) not in self._session_ids:
            raise RuntimeError("KV publication targets a session outside this step")
        self._capture_auxiliary()
        return self._store.publish(
            session_id,
            source_version=source_version,
            source_digest=source_digest,
            destination=destination,
            expected_base=expected_base,
            product=product,
            transport=transport,
        )

    def install_publication(
        self,
        session_id: int,
        source: ProductRef,
        installed_product: ProductRef,
        transport: Transport,
        transferred_tensors: tuple[torch.Tensor, ...] | None = None,
    ) -> KvSnapshot:
        self._require_open()
        if int(session_id) not in self._session_ids:
            raise RuntimeError("KV installation targets a session outside this step")
        self._capture_auxiliary()
        return self._store.install_publication(
            session_id,
            source,
            installed_product,
            transport,
            transferred_tensors=transferred_tensors,
        )

    def prepare(self) -> None:
        self._require_open()

    def publish(self) -> None:
        self._require_open()

    def finalize(self) -> None:
        self._require_open()
        self._closed = True
        self._clear_retained()

    def rollback(self) -> None:
        if self._closed:
            return
        pool = self._store.pool
        with self._store._lock:
            if pool is not None:
                for block, retained in self._retained.items():
                    pool.k[:, block].copy_(retained.key)
                    pool.v[:, block].copy_(retained.value)
                    for target, source in (
                        (pool.k_scale, retained.key_scale),
                        (pool.v_scale, retained.value_scale),
                        (pool.k_scale_set, retained.key_scale_set),
                        (pool.v_scale_set, retained.value_scale_set),
                    ):
                        if target is not None and source is not None:
                            target[:, block].copy_(source)
            self._store.restore_entries(set(self._session_ids), self._entry_snapshot)
            if self._auxiliary_snapshot is not None:
                self._store.restore_auxiliary(
                    set(self._session_ids),
                    self._auxiliary_snapshot,
                )
            self._store._undo_logical_pages(tuple(reversed(self._logical_page_undo)))
        self._closed = True
        self._clear_retained()

    def _capture_auxiliary(self) -> None:
        if self._auxiliary_snapshot is None:
            self._auxiliary_snapshot = self._store.snapshot_auxiliary(set(self._session_ids))

    def _retain_pages(self, blocks: Sequence[int]) -> None:
        pool = self._store.pool
        if pool is None:
            raise RuntimeError("KV page retention requires a physical pool")
        for block in blocks:
            block = int(block)
            if block in self._retained:
                continue
            pool.validate_block_ids((block,))
            self._retained[block] = _RetainedPage(
                key=pool.k[:, block].clone(),
                value=pool.v[:, block].clone(),
                key_scale=None if pool.k_scale is None else pool.k_scale[:, block].clone(),
                value_scale=None if pool.v_scale is None else pool.v_scale[:, block].clone(),
                key_scale_set=(
                    None if pool.k_scale_set is None else pool.k_scale_set[:, block].clone()
                ),
                value_scale_set=(
                    None if pool.v_scale_set is None else pool.v_scale_set[:, block].clone()
                ),
            )

    def _clear_retained(self) -> None:
        self._retained.clear()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("KV transaction is closed")


__all__ = [
    "KvBatchView",
    "KvBranchState",
    "KvCommittedState",
    "KvEntry",
    "KvPageState",
    "KvStore",
    "KvTxn",
]
