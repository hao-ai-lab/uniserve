"""Transactional authority for sequence KV block tables and lengths."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from threading import RLock

import torch

from ..backends.paged_kv_math import paged_kv_write
from ..batch import Admission, KvLeaseDelta, PublishedKv
from ..foundation.errors import invalid_descriptor
from ..foundation.sizing import ceil_div
from .host_staging import copy_cpu_to_device, cpu_int_staging_buffer, fill_cpu_ints, is_pinned
from .kv_pool import PagedKVPool
from .transfer import Transport


@dataclass(slots=True)
class KvEntry:
    block_ids: list[int] = field(default_factory=list)
    prefix_len: int = 0
    length: int = 0
    group_id: int = 0


@dataclass(slots=True)
class _RetainedPage:
    key: torch.Tensor
    value: torch.Tensor
    key_scale: torch.Tensor | None
    value_scale: torch.Tensor | None
    key_scale_set: torch.Tensor | None
    value_scale_set: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class _KvSnapshot:
    entries: dict[int, tuple[bool, tuple[int, ...], int, int, int]]
    branches: dict[tuple[int, int, str], tuple[tuple[int, ...], int]]


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
    generation: int
    branch: str
    length: int
    block_count: int
    pages: KvPageState


@dataclass(frozen=True, slots=True)
class KvCommittedState:
    session_id: int
    block_ids: tuple[int, ...]
    prefix_len: int
    length: int
    group_id: int
    pages: KvPageState | None
    branches: tuple[KvBranchState, ...]


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
        self._block_table_width = max(len(ids) for ids in self._block_ids)
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

    def block_table(self, device: torch.device) -> torch.Tensor:
        target = torch.device(device)
        cached = self._block_tables.get(target)
        if cached is not None:
            return cached
        row_count = len(self._block_ids)
        cpu = cpu_int_staging_buffer(
            row_count * self._block_table_width,
            dtype=torch.int32,
            pin=target.type == "cuda",
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
            slot=None,
            name="kv_block_table",
        ).view(row_count, self._block_table_width)
        self._block_tables[target] = result
        return result

    def cache_seqlens(self, device: torch.device) -> torch.Tensor:
        target = torch.device(device)
        cached = self._cache_lengths.get(target)
        if cached is not None:
            return cached
        cpu = cpu_int_staging_buffer(
            len(self._base_lens),
            dtype=torch.int32,
            pin=target.type == "cuda",
            name="kv_cache_lengths",
        )
        fill_cpu_ints(cpu, self._base_lens)
        result = copy_cpu_to_device(
            cpu,
            device=target,
            non_blocking=target.type == "cuda" and is_pinned(cpu),
            slot=None,
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
            page_ids = block_table[0].to(dtype=torch.int64).index_select(0, block_slots).contiguous()
        else:
            offsets = query_offsets.to(dtype=torch.int64)
            row_ids = torch.bucketize(token_offsets, offsets[1:].contiguous(), right=True)
            positions = (
                cache_seqlens.to(dtype=torch.int64)[row_ids]
                + token_offsets
                - offsets[row_ids]
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
        width = max(len(entry.block_ids) for entry, _query, _write in self._rows)
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

    def __init__(
        self,
        pool: PagedKVPool | None = None,
        scratch: PagedKVPool | None = None,
    ) -> None:
        self.pool = pool
        self.scratch_pool = scratch
        self._entries: dict[int, KvEntry] = {}
        self._branches: dict[tuple[int, int, str], KvEntry] = {}
        self._holders: dict[int, set[int]] = {}
        self._lock = RLock()

    def bind_pool(self, pool: PagedKVPool) -> None:
        with self._lock:
            if self.pool is not None and self.pool is not pool:
                raise RuntimeError("KV store is already bound to a physical pool")
            self.pool = pool

    def bind_scratch_pool(self, pool: PagedKVPool) -> None:
        with self._lock:
            if self.scratch_pool is not None and self.scratch_pool is not pool:
                raise RuntimeError("KV store is already bound to a scratch pool")
            if not callable(getattr(pool, "allocate_blocks", None)) or not callable(
                getattr(pool, "release_blocks", None)
            ):
                raise TypeError("scratch KV pool must own block allocation")
            self.scratch_pool = pool

    def resident_block_count(self) -> int:
        """Return the number of distinct host-leased blocks with live holders."""

        with self._lock:
            return sum(bool(holders) for holders in self._holders.values())

    def scratch_token_count(self) -> int:
        """Return physically allocated scratch capacity in token slots."""

        with self._lock:
            block_ids = {
                block_id for entry in self._branches.values() for block_id in entry.block_ids
            }
            block_size = 0 if self.scratch_pool is None else int(self.scratch_pool.block_size)
            return len(block_ids) * block_size

    def admit(self, admission: Admission) -> None:
        with self._lock:
            sequence = admission.sequence
            if sequence is None:
                self._entries.setdefault(admission.session_id, KvEntry())
                return
            if admission.session_id in self._entries:
                return
            allocation = sequence.kv
            entry = KvEntry(
                block_ids=list(allocation.block_ids),
                prefix_len=allocation.prefix_len,
                length=allocation.prefix_len,
                group_id=allocation.group_id,
            )
            self._validate_blocks(admission.session_id, entry.block_ids, entry.prefix_len)
            self._entries[admission.session_id] = entry
            self._register(admission.session_id, entry.block_ids)

    def get(self, session_id: int) -> KvEntry:
        with self._lock:
            try:
                return self._entries[int(session_id)]
            except KeyError:
                raise invalid_descriptor(f"session {session_id} has no KV state") from None

    def apply_lease(self, session_id: int, lease: KvLeaseDelta) -> None:
        with self._lock:
            entry = self.get(session_id)
            if lease.group_id != entry.group_id:
                raise invalid_descriptor(
                    f"session {session_id} KV group {lease.group_id} does not match {entry.group_id}"
                )
            new = list(lease.new_blocks)
            if not new:
                return
            if entry.block_ids[-len(new) :] == new:
                return
            duplicate = set(entry.block_ids) & set(new)
            if duplicate:
                raise invalid_descriptor(
                    f"session {session_id} KV lease repeats blocks {sorted(duplicate)}"
                )
            self._validate_blocks(session_id, new, entry.length)
            entry.block_ids.extend(new)
            self._register(session_id, new)

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
            end = entry.length + int(tokens)
            self.validate_write(session_id, entry.length, end)
            entry.length = end

    def view(
        self,
        session_ids: Sequence[int],
        *,
        query_lens: Sequence[int] | None = None,
    ) -> KvBatchView:
        if self.pool is None:
            raise RuntimeError("KV execution requires a physical pool")
        entries = [self.get(value) for value in session_ids]
        return KvBatchView(
            self.pool,
            [entry.block_ids for entry in entries],
            [entry.length for entry in entries],
            query_lens,
        )

    def packed_view(
        self,
        rows: Sequence[tuple[KvEntry, int, bool]],
        *,
        scratch: bool,
    ) -> _PackedKvView:
        pool = self.scratch_pool if scratch else self.pool
        if pool is None:
            raise RuntimeError("packed KV execution requires its declared physical pool")
        return _PackedKvView(pool, rows)

    def scratch_entry(
        self,
        session_id: int,
        generation: int,
        branch: str,
        *,
        capacity_tokens: int,
        copy_conditioning: bool,
    ) -> tuple[KvEntry, bool]:
        """Return one generation-scoped branch prefix, provisioning it atomically."""

        pool = self.scratch_pool
        if pool is None:
            raise RuntimeError("flow execution requires a scratch KV pool")
        key = (int(session_id), int(generation), str(branch))
        with self._lock:
            existing = self._branches.get(key)
            if existing is not None:
                self._ensure_scratch_capacity(existing, capacity_tokens)
                return existing, False
            source = self.get(session_id)
            prefix = source.length if copy_conditioning else 0
            entry = KvEntry(length=prefix)
            try:
                self._ensure_scratch_capacity(entry, max(prefix, int(capacity_tokens)))
                if copy_conditioning and prefix:
                    self._copy_span(
                        self.pool,
                        source.block_ids,
                        pool,
                        entry.block_ids,
                        start=0,
                        length=prefix,
                    )
            except BaseException:
                self._release_scratch(entry.block_ids)
                raise
            self._branches[key] = entry
            return entry, True

    def advance_entry(self, entry: KvEntry, tokens: int, *, scratch: bool) -> None:
        pool = self.scratch_pool if scratch else self.pool
        if pool is None:
            raise RuntimeError("KV advance requires a physical pool")
        end = entry.length + int(tokens)
        if end > len(entry.block_ids) * pool.block_size:
            raise invalid_descriptor("KV advance exceeds its block capacity")
        entry.length = end

    def promote_scratch(self, session_id: int, source: KvEntry, tokens: int) -> None:
        if self.pool is None or self.scratch_pool is None:
            raise RuntimeError("KV promotion requires request and scratch pools")
        target = self.get(session_id)
        count = int(tokens)
        self.validate_write(session_id, target.length, target.length + count)
        self._copy_span(
            self.scratch_pool,
            source.block_ids,
            self.pool,
            target.block_ids,
            start=source.length,
            target_start=target.length,
            length=count,
        )
        target.length += count

    def release_generation(self, session_id: int, generation: int) -> None:
        with self._lock:
            keys = [
                key
                for key in self._branches
                if key[0] == int(session_id) and key[1] == int(generation)
            ]
            for key in keys:
                self._release_scratch(self._branches.pop(key).block_ids)

    def publish(
        self,
        session_id: int,
        *,
        source_version: int,
        position: int,
        transport: object | None = None,
    ) -> PublishedKv:
        entry = self.get(session_id)
        locators: list[str] = []
        if transport is not None:
            if self.pool is None:
                raise RuntimeError("KV publication requires a physical pool")
            publish = getattr(transport, "publish")
            for layer in range(self.pool.num_layers):
                key, value = self.pool.read(
                    layer,
                    entry.block_ids,
                    start=0,
                    length=entry.length,
                )
                if key is None or value is None:
                    raise RuntimeError("published KV span is incomplete")
                locators.append(publish(key.contiguous()).to_wire_json())
                locators.append(publish(value.contiguous()).to_wire_json())
        return PublishedKv(
            handle=int(session_id),
            locators=tuple(locators),
            source_version=int(source_version),
            kv_tokens=entry.length,
            block_ids=tuple(entry.block_ids),
            group_id=entry.group_id,
            position=int(position),
        )

    def import_snapshot(self, session_id: int, snapshot: PublishedKv, transport: Transport) -> None:
        """Install a complete transferred snapshot without exposing partial state.

        Every locator is fetched and its tensor geometry is validated before a
        pool write or block-table mutation occurs. Raw pages that overlap the
        prior committed entry are retained until the surrounding ``KvTxn`` is
        finalized, so a later forward, validation, or commit failure can restore
        both metadata and bytes.
        """

        transaction = self.begin_step({int(session_id)})
        try:
            transaction.import_snapshot(session_id, snapshot, transport)
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
        snapshot: PublishedKv,
        transport: Transport,
        transaction: KvTxn,
    ) -> None:
        if self.pool is None:
            raise RuntimeError("KV snapshot import requires a physical pool")
        if len(snapshot.locators) != 2 * self.pool.num_layers:
            raise invalid_descriptor("published KV locator count does not match cache layers")
        from .transfer import Locator, fetch_locator

        with self._lock:
            resident = self.get(session_id)
            if resident.group_id != snapshot.group_id:
                raise invalid_descriptor("published KV group does not match the local session")
            if (
                resident.length == snapshot.kv_tokens
                and tuple(resident.block_ids) == tuple(snapshot.block_ids)
            ):
                # Flow conditioning is re-published on every denoise step, but it
                # is fixed across an image's steps. Once the session entry already
                # holds exactly these blocks and tokens, re-fetching over the
                # transport and re-copying the KV block-by-block is pure waste
                # (it dominated the travel workload). The entry is unchanged, so
                # there is nothing to write and nothing to roll back -- skip the
                # fetch, the copy, and the per-block rollback snapshot entirely.
                return

        tensors: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in range(self.pool.num_layers):
            key = fetch_locator(transport, Locator.from_wire_json(snapshot.locators[2 * layer]))
            value = fetch_locator(
                transport, Locator.from_wire_json(snapshot.locators[2 * layer + 1])
            )
            expected = (snapshot.kv_tokens, self.pool.n_kv, self.pool.head_dim)
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
            blocks = list(snapshot.block_ids)
            self._validate_blocks(session_id, blocks, snapshot.kv_tokens)
            shared_blocks = 0
            for block in blocks:
                if self._holders.get(block, set()) - {int(session_id)}:
                    shared_blocks += 1
                else:
                    break
            write_start = min(snapshot.kv_tokens, shared_blocks * self.pool.block_size)
            transaction._retain_pages(blocks[shared_blocks:])
            for layer, (key, value) in enumerate(tensors):
                if write_start >= snapshot.kv_tokens:
                    continue
                self.pool.write(
                    layer,
                    blocks,
                    start=write_start,
                    k=key[write_start:].to(self.pool.k.device),
                    v=value[write_start:].to(self.pool.v.device),
                )
            self._unregister(session_id, entry.block_ids)
            entry.block_ids = blocks
            entry.length = snapshot.kv_tokens
            entry.prefix_len = min(entry.prefix_len, entry.length)
            self._register(session_id, blocks)

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
        with self._lock:
            entry = self._entries.pop(int(session_id), None)
            if entry is None:
                pass
            else:
                self._unregister(int(session_id), entry.block_ids)
            keys = [key for key in self._branches if key[0] == int(session_id)]
            for key in keys:
                self._release_scratch(self._branches.pop(key).block_ids)

    def snapshot_committed(self, request_ids: set[int]) -> tuple[KvCommittedState, ...]:
        requested = sorted(int(value) for value in request_ids)
        with self._lock:
            states: list[KvCommittedState] = []
            for session_id in requested:
                entry = self._entries.get(session_id)
                if entry is None:
                    raise invalid_descriptor(f"session {session_id} has no KV state")
                pages = self._snapshot_pages(self.pool, entry.block_ids)
                branches = tuple(
                    KvBranchState(
                        generation=key[1],
                        branch=key[2],
                        length=branch.length,
                        block_count=len(branch.block_ids),
                        pages=self._require_pages(
                            self._snapshot_pages(self.scratch_pool, branch.block_ids),
                            "scratch KV",
                        ),
                    )
                    for key, branch in sorted(self._branches.items())
                    if key[0] == session_id
                )
                states.append(
                    KvCommittedState(
                        session_id=session_id,
                        block_ids=tuple(entry.block_ids),
                        prefix_len=entry.prefix_len,
                        length=entry.length,
                        group_id=entry.group_id,
                        pages=pages,
                        branches=branches,
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
            if min(state.prefix_len, state.length, state.group_id) < 0:
                raise invalid_descriptor("KV snapshot metadata must be non-negative")
            if state.prefix_len > state.length:
                raise invalid_descriptor("KV snapshot prefix exceeds its committed length")
            if self.pool is None:
                if state.block_ids or state.length or state.pages is not None:
                    raise invalid_descriptor("KV snapshot requires an absent physical pool")
            else:
                if state.length > len(state.block_ids) * self.pool.block_size:
                    raise invalid_descriptor("KV snapshot length exceeds its block capacity")
                self.pool.validate_block_ids(state.block_ids)
                if any(value >= self.pool.schedulable_num_blocks for value in state.block_ids):
                    raise invalid_descriptor("KV snapshot uses a reserved physical block")
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
            seen_branches: set[tuple[int, str]] = set()
            for branch in state.branches:
                key = (branch.generation, branch.branch)
                if key in seen_branches:
                    raise invalid_descriptor("KV snapshot repeats a scratch branch")
                seen_branches.add(key)
                if branch.generation < 0 or branch.length < 0 or branch.block_count < 1:
                    raise invalid_descriptor("KV snapshot scratch branch is invalid")
                if not branch.branch:
                    raise invalid_descriptor("KV snapshot scratch branch name is empty")
                if self.scratch_pool is None:
                    raise invalid_descriptor("KV snapshot requires an absent scratch pool")
                if branch.length > branch.block_count * self.scratch_pool.block_size:
                    raise invalid_descriptor("KV snapshot scratch length exceeds capacity")
                self._validate_page_state(self.scratch_pool, branch.pages, branch.block_count)

    def _install_committed(
        self,
        states: Sequence[KvCommittedState],
        session_ids: set[int],
    ) -> None:
        for key in [key for key in self._branches if key[0] in session_ids]:
            self._release_scratch(self._branches.pop(key).block_ids)
        for session_id in session_ids:
            current = self._entries.pop(session_id, None)
            if current is not None:
                self._unregister(session_id, current.block_ids)

        block_sources: dict[int, tuple[KvPageState, int]] = {}
        for state in states:
            entry = KvEntry(
                block_ids=list(state.block_ids),
                prefix_len=state.prefix_len,
                length=state.length,
                group_id=state.group_id,
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
                for index, block_id in enumerate(state.block_ids):
                    block_sources.setdefault(block_id, (state.pages, index))

        if self.pool is not None:
            for block_id, (pages, index) in block_sources.items():
                self._copy_page_to_pool(self.pool, block_id, pages, index)

        for state in states:
            for branch in state.branches:
                pool = self.scratch_pool
                if pool is None:
                    raise RuntimeError("validated scratch KV pool disappeared")
                block_ids = tuple(getattr(pool, "allocate_blocks")(branch.block_count))
                if len(block_ids) != branch.block_count:
                    if block_ids:
                        getattr(pool, "release_blocks")(block_ids)
                    raise RuntimeError("scratch KV allocator returned an incomplete lease")
                try:
                    for index, block_id in enumerate(block_ids):
                        self._copy_page_to_pool(pool, block_id, branch.pages, index)
                except BaseException:
                    getattr(pool, "release_blocks")(block_ids)
                    raise
                self._branches[(state.session_id, branch.generation, branch.branch)] = KvEntry(
                    block_ids=list(block_ids),
                    length=branch.length,
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

    def snapshot_requests(self, request_ids: set[int]) -> _KvSnapshot:
        with self._lock:
            result: dict[int, tuple[bool, tuple[int, ...], int, int, int]] = {}
            for session_id in request_ids:
                entry = self._entries.get(session_id)
                if entry is None:
                    result[session_id] = (False, (), 0, 0, 0)
                else:
                    result[session_id] = (
                        True,
                        tuple(entry.block_ids),
                        entry.prefix_len,
                        entry.length,
                        entry.group_id,
                    )
            branches = {
                key: (tuple(entry.block_ids), entry.length)
                for key, entry in self._branches.items()
                if key[0] in request_ids
            }
            return _KvSnapshot(entries=result, branches=branches)

    def restore_requests(
        self,
        request_ids: set[int],
        snapshot: object,
    ) -> None:
        if not isinstance(snapshot, _KvSnapshot):
            raise RuntimeError("KV snapshot has an invalid type")
        with self._lock:
            for session_id in request_ids:
                current = self._entries.pop(session_id, None)
                if current is not None:
                    self._unregister(session_id, current.block_ids)
                existed, blocks, prefix_len, length, group_id = snapshot.entries.get(
                    session_id, (False, (), 0, 0, 0)
                )
                if existed:
                    entry = KvEntry(list(blocks), prefix_len, length, group_id)
                    self._entries[session_id] = entry
                    self._register(session_id, entry.block_ids)
            current_keys = [key for key in self._branches if key[0] in request_ids]
            for key in current_keys:
                current = self._branches.pop(key)
                prior_blocks = set(snapshot.branches.get(key, ((), 0))[0])
                self._release_scratch(
                    tuple(block for block in current.block_ids if block not in prior_blocks)
                )
            for key, (blocks, length) in snapshot.branches.items():
                if key[0] in request_ids:
                    self._branches[key] = KvEntry(list(blocks), length=length)

    def _ensure_scratch_capacity(self, entry: KvEntry, tokens: int) -> None:
        pool = self.scratch_pool
        if pool is None:
            raise RuntimeError("scratch KV pool is not bound")
        required = ceil_div(max(0, int(tokens)), pool.block_size)
        missing = required - len(entry.block_ids)
        if missing > 0:
            allocated = getattr(pool, "allocate_blocks")(missing)
            if len(allocated) != missing:
                if allocated:
                    getattr(pool, "release_blocks")(allocated)
                raise RuntimeError("scratch KV allocator returned an incomplete lease")
            entry.block_ids.extend(int(value) for value in allocated)

    def _release_scratch(self, blocks: Sequence[int]) -> None:
        if blocks and self.scratch_pool is not None:
            getattr(self.scratch_pool, "release_blocks")(tuple(int(value) for value in blocks))

    @staticmethod
    def _copy_span(
        source_pool: PagedKVPool | None,
        source_blocks: Sequence[int],
        target_pool: PagedKVPool,
        target_blocks: Sequence[int],
        *,
        start: int,
        length: int,
        target_start: int | None = None,
    ) -> None:
        if source_pool is None:
            raise RuntimeError("KV copy source pool is absent")
        destination = int(start) if target_start is None else int(target_start)
        for layer in range(source_pool.num_layers):
            key, value = source_pool.read(
                layer,
                list(source_blocks),
                start=int(start),
                length=int(length),
            )
            if key is None or value is None:
                raise RuntimeError("KV copy source span is incomplete")
            target_pool.write(
                layer,
                list(target_blocks),
                start=destination,
                k=key.to(target_pool.k.device),
                v=value.to(target_pool.v.device),
            )

    def _validate_blocks(self, session_id: int, blocks: Sequence[int], boundary: int) -> None:
        if len(set(blocks)) != len(blocks):
            raise invalid_descriptor(f"session {session_id} KV lease repeats a block")
        if self.pool is not None and any(
            value < 0 or value >= self.pool.schedulable_num_blocks for value in blocks
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
        self._snapshot = store.snapshot_requests(set(session_ids))
        self._retained: dict[int, _RetainedPage] = {}
        self._pending_generation_releases: set[tuple[int, int]] = set()
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

    def packed_view(
        self,
        rows: Sequence[tuple[KvEntry, int, bool]],
        *,
        scratch: bool,
    ) -> _PackedKvView:
        self._require_open()
        return self._store.packed_view(rows, scratch=scratch)

    def scratch_entry(
        self,
        session_id: int,
        generation: int,
        branch: str,
        *,
        capacity_tokens: int,
        copy_conditioning: bool,
    ) -> tuple[KvEntry, bool]:
        self._require_open()
        if int(session_id) not in self._session_ids:
            raise RuntimeError("scratch KV entry targets a session outside this step")
        return self._store.scratch_entry(
            session_id,
            generation,
            branch,
            capacity_tokens=capacity_tokens,
            copy_conditioning=copy_conditioning,
        )

    def advance_entry(self, entry: KvEntry, tokens: int, *, scratch: bool) -> None:
        self._require_open()
        self._store.advance_entry(entry, tokens, scratch=scratch)

    def advance(self, session_id: int, tokens: int) -> None:
        self._require_open()
        if int(session_id) not in self._session_ids:
            raise RuntimeError("KV advance targets a session outside this step")
        self._store.advance(int(session_id), int(tokens))

    def promote_scratch(self, session_id: int, source: KvEntry, tokens: int) -> None:
        self._require_open()
        if int(session_id) not in self._session_ids:
            raise RuntimeError("KV promotion targets a session outside this step")
        self._store.promote_scratch(session_id, source, tokens)

    def release_generation(self, session_id: int, generation: int) -> None:
        self._require_open()
        if int(session_id) not in self._session_ids:
            raise RuntimeError("KV release targets a session outside this step")
        self._pending_generation_releases.add((int(session_id), int(generation)))

    def import_snapshot(
        self,
        session_id: int,
        snapshot: PublishedKv,
        transport: Transport,
    ) -> None:
        self._require_open()
        if int(session_id) not in self._session_ids:
            raise RuntimeError("KV import targets a session outside this step")
        self._store._import_snapshot(int(session_id), snapshot, transport, self)

    def prepare(self) -> None:
        self._require_open()

    def publish(self) -> None:
        self._require_open()

    def finalize(self) -> None:
        self._require_open()
        for session_id, generation in self._pending_generation_releases:
            self._store.release_generation(session_id, generation)
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
            self._store.restore_requests(set(self._session_ids), self._snapshot)
        self._closed = True
        self._clear_retained()

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
