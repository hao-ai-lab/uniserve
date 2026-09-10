"""Fixed physical KV storage indexed by scheduler-assigned page IDs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from threading import RLock

import torch

from ..backends.paged_kv_math import paged_kv_write
from ..execution.batch import BufferId, ProductRef, RequestKey
from ..foundation.errors import compute_error, invalid_descriptor, resource_error, unsupported_setup
from ..nn.quant.kv_cache import (
    dequantize_fp8_block,
    fp8_quantize,
    is_fp8_kv_dtype,
    resolve_kv_store_dtype,
    scale_for_fp8_block,
)
from .cache_transfer import CacheTransfers, CacheWrite

__all__ = ["CachePool"]


@dataclass(slots=True)
class CacheSource:
    """An immutable token interval retained by its physical publications.

    Ranges map each physical page to its token offset and token count. A publication
    covers every layer's K/V for these ranges, so appends outside the interval
    remain independent even when they share its final physical page.
    """

    buffer: BufferId
    ranges: dict[int, tuple[int, int]]
    retirements: tuple[Future[None], ...] = ()
    released: bool = False


@dataclass(eq=False, slots=True)
class CacheExecution:
    """Physical page intervals retained by one computation's existing completion fence."""

    completion: Future[None]
    requests: set[RequestKey]
    ranges: dict[int, tuple[int, int]]


class CachePool:
    """Startup-sized layer-major KV tensors with no allocation authority."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_pages: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
        total_kv_heads: int | None = None,
        kv_head_offset: int = 0,
        total_layers: int | None = None,
        layer_offset: int = 0,
        store_dtype: torch.dtype | str | None = None,
        group_ranges: Sequence[tuple[int, int]] | None = None,
        import_capacity: int = 1,
    ) -> None:
        """Allocate layer-major KV pages and optional per-page FP8 scales."""

        # Normalize scheduler-visible geometry before device allocation so
        # every tensor shares one validated page interpretation.
        self.num_layers = int(num_layers)
        self.total_layers = self.num_layers if total_layers is None else int(total_layers)
        self.layer_offset = int(layer_offset)
        self.num_pages = int(num_pages)
        self.num_blocks = self.num_pages
        self.block_size = int(page_size)
        self.n_kv = int(num_kv_heads)
        self.total_kv_heads = self.n_kv if total_kv_heads is None else int(total_kv_heads)
        self.kv_head_offset = int(kv_head_offset)
        self.head_dim = int(head_dim)
        self.group_ranges = self._group_ranges(group_ranges)
        self.group_count = len(self.group_ranges)
        self.dtype = dtype
        self.store_dtype = resolve_kv_store_dtype(dtype, store_dtype)
        self.is_quantized = is_fp8_kv_dtype(self.store_dtype)
        self.supports_paged_attention_storage = not self.is_quantized
        if (
            self.num_layers < 1
            or self.layer_offset < 0
            or self.layer_offset + self.num_layers > self.total_layers
            or self.num_pages < 1
            or self.block_size < 1
            or self.n_kv < 1
            or self.kv_head_offset < 0
            or self.kv_head_offset + self.n_kv > self.total_kv_heads
            or self.head_dim < 1
        ):
            raise invalid_descriptor("CachePool geometry is invalid")
        if self.is_quantized and (
            self.total_kv_heads % self.n_kv or self.kv_head_offset % self.n_kv
        ):
            raise invalid_descriptor("FP8 KV heads must form aligned, uniform scale groups")

        # Keys and values use identical layer/page/token/head geometry. FP8
        # storage adds one scale and initialization flag per layer-page pair.
        shape = (
            self.num_layers,
            self.num_pages,
            self.block_size,
            self.n_kv,
            self.head_dim,
        )
        self.k = torch.zeros(shape, device=device, dtype=self.store_dtype)
        self.v = torch.zeros(shape, device=device, dtype=self.store_dtype)
        scale_shape = (self.num_layers, self.num_pages, 1, 1, 1)
        self._scales = (
            torch.ones((2, *scale_shape), device=device, dtype=torch.float32)
            if self.is_quantized
            else None
        )
        self.k_scale = None if self._scales is None else self._scales[0]
        self.v_scale = None if self._scales is None else self._scales[1]
        # Initialization is host-owned page metadata. A scale is written only
        # on the first append after page allocation, so an immutable published
        # scale never receives even a same-value device write during an append.
        scale_flags = self.num_layers * self.num_pages
        self._k_scale_set = bytearray(scale_flags) if self.is_quantized else None
        self._v_scale_set = bytearray(scale_flags) if self.is_quantized else None

        # Reuse normalized page tuples after their bounds and group ownership
        # have been established by the validation path.
        self._validated_page_tuples: dict[
            tuple[tuple[int, ...], bool, int | None], tuple[int, ...]
        ] = {}
        self._sources: dict[BufferId, CacheSource] = {}
        self._executions: dict[Future[None], CacheExecution] = {}
        self._execution_pages: dict[int, set[CacheExecution]] = {}
        self._execution_lock = RLock()
        self.imports = CacheTransfers(self, capacity=import_capacity)

    @property
    def has_pending_accesses(self) -> bool:
        """Whether cache intervals still have publication, computation or import owners."""

        return bool(self._sources) or bool(self._executions) or bool(self.imports)

    def retain_execution(
        self,
        request: RequestKey,
        page_ids: Sequence[int],
        *,
        group: int,
        length: int,
        completion: Future[None],
    ) -> None:
        """Retain scheduler-authorized model accesses until their device work completes.

        Model calls are ordered by the public runner. Their existing output fence
        also prevents independent import streams and page allocation from reusing
        these ranges while a producer or consumer kernel is still running.
        """

        if not length:
            return
        pages = self.validate_pages(page_ids, group=group)
        ranges = tuple(self._spans(pages, 0, length))
        with self._execution_lock:
            if completion.done():
                completion.result()
                return
            execution = self._executions.get(completion)
            register = execution is None
            if execution is None:
                execution = CacheExecution(completion, set(), {})
                self._executions[completion] = execution
            execution.requests.add(request)
            for page, offset, count in ranges:
                previous = execution.ranges.get(page)
                if previous is not None:
                    end = max(previous[0] + previous[1], offset + count)
                    offset = min(previous[0], offset)
                    count = end - offset
                execution.ranges[page] = (offset, count)
                self._execution_pages.setdefault(page, set()).add(execution)
        if register:
            completion.add_done_callback(self._execution_completed)

    def _execution_completed(self, completion: Future[None]) -> None:
        with self._execution_lock:
            if completion.cancelled() or completion.exception() is not None:
                return
            execution = self._executions.pop(completion, None)
            if execution is None:
                return
            for page in execution.ranges:
                uses = self._execution_pages[page]
                uses.remove(execution)
                if not uses:
                    del self._execution_pages[page]

    def _execution_dependencies(
        self, ranges: Sequence[tuple[int, int, int]]
    ) -> tuple[Future[None], ...]:
        with self._execution_lock:
            return tuple(
                {
                    execution.completion
                    for page, offset, count in ranges
                    for execution in self._execution_pages.get(page, ())
                    if self._ranges_overlap(execution.ranges, ((page, offset, count),))
                }
            )

    def require_reusable(
        self, page_ids: Sequence[int], *, group: int, start: int, length: int
    ) -> None:
        """Authorize page initialization or independent-stream import before submission."""

        self.require_writable(page_ids, group=group, start=start, length=length)
        if self._execution_dependencies(tuple(self._spans(page_ids, start, length))):
            raise resource_error("KV interval still has an executing producer or consumer")

    def reserve_publication(
        self,
        product: ProductRef,
        page_ids: Sequence[int],
        *,
        group: int,
        start: int,
        length: int,
    ) -> CacheSource:
        """Retain the exact published interval before exporting any of its views.

        The caller attaches every registration's retirement future and releases
        this reservation if publication is abandoned before semantic visibility.
        """

        self._reap_sources()
        pages = self.validate_pages(page_ids, group=group)
        ranges = {
            page: (offset, count) for page, offset, count in self._spans(pages, start, length)
        }
        if not ranges or product.buffer_id in self._sources:
            raise invalid_descriptor("KV publication has an empty or already registered interval")
        source = CacheSource(product.buffer_id, ranges)
        self._sources[source.buffer] = source
        return source

    def retain_publication(self, source: CacheSource, retirement: Future[None]) -> None:
        """Retain the published interval until this physical registration retires."""

        if self._sources.get(source.buffer) is not source or source.released:
            raise invalid_descriptor("KV publication reservation is no longer active")
        source.retirements = (*source.retirements, retirement)

    def release_buffers(self, buffers: Iterable[BufferId]) -> None:
        """Revoke semantic ownership while preserving every pending physical read."""

        selected = tuple(buffers)
        self.imports.release(selected)
        for buffer in selected:
            source = self._sources.get(buffer)
            if source is not None:
                source.released = True
        self._reap_sources()

    def retirement_ready(
        self,
        *,
        buffers: Iterable[BufferId] = (),
        requests: Iterable[RequestKey] = (),
        retained: frozenset[BufferId] = frozenset(),
    ) -> bool:
        """Observe completion errors only for the selected allocation owners."""

        selected = set(buffers)
        owners = set(requests)
        sources = tuple(
            source
            for buffer, source in self._sources.items()
            if buffer in selected or (buffer.owner in owners and buffer not in retained)
        )
        for source in sources:
            for future in source.retirements:
                if future.done():
                    future.result()
        self._reap_sources()
        # Free retires a publication, not the request's resident KV pages.
        # Unrelated products can share that request while later computation
        # still reads its prefix. Only request retirement waits for all such
        # computations; physical page reuse separately checks their ranges.
        with self._execution_lock:
            executions = tuple(
                execution
                for execution in self._executions.values()
                if execution.requests.intersection(owners)
            )
        for execution in executions:
            if execution.completion.done():
                execution.completion.result()
        return (
            not executions
            and all(source.buffer not in self._sources for source in sources)
            and self.imports.retirement_ready(selected, owners, retained)
        )

    def write_dependencies(
        self, page_ids: Sequence[int], *, group: int, start: int, length: int
    ) -> tuple[Future[None], ...]:
        """Return retirements that must precede writing this physical interval."""

        if not self.has_pending_accesses:
            return ()
        self._reap_sources()
        pages = self.validate_pages(page_ids, group=group)
        ranges = tuple(self._spans(pages, start, length))
        return (
            self._execution_dependencies(ranges)
            + self.imports.dependencies(ranges)
            + tuple(
                future
                for source in self._sources.values()
                if self._ranges_overlap(source.ranges, ranges)
                for future in source.retirements
            )
        )

    def require_writable(
        self, page_ids: Sequence[int], *, group: int, start: int, length: int
    ) -> None:
        """Authorize the interval before staging any kernel that can write it.

        Device-indexed attention kernels borrow raw cache views. Their caller
        must validate the scheduler's write interval here before dispatch; no
        device-to-host read of per-token addresses is needed in the kernel path.
        """

        # The runner orders model accesses. Only independent imports and
        # published immutable ranges add write conflicts at this boundary.
        if not self._sources and not self.imports:
            return
        self._reap_sources()
        pages = self.validate_pages(page_ids, group=group)
        ranges = tuple(self._spans(pages, start, length))
        if any(self._ranges_overlap(source.ranges, ranges) for source in self._sources.values()):
            raise resource_error("KV interval still has a published version")
        if self.imports.dependencies(ranges):
            raise resource_error("KV interval still has an import destination")

    @staticmethod
    def _ranges_overlap(
        left: Mapping[int, tuple[int, int]], right: Sequence[tuple[int, int, int]]
    ) -> bool:
        return any(
            (other := left.get(page)) is not None
            and offset < other[0] + other[1]
            and other[0] < offset + count
            for page, offset, count in right
        )

    def _reap_sources(self) -> None:
        for buffer, source in tuple(self._sources.items()):
            if source.released and all(
                future.done() and not future.cancelled() and future.exception() is None
                for future in source.retirements
            ):
                del self._sources[buffer]

    def close(self) -> None:
        """Release semantic publications and require known physical retirement."""

        self.imports.stop()
        self.release_buffers(tuple(self._sources))
        self.imports.require_retired()
        if self._executions:
            raise resource_error("KV cache still has executing producers or consumers")
        if self._sources:
            raise resource_error("KV cache still has unretired physical publications")

    def _group_ranges(
        self,
        declared: Sequence[tuple[int, int]] | None,
    ) -> tuple[tuple[int, int], ...]:
        """Normalize physical cache-group ranges and require exact non-overlapping page coverage."""

        ranges = (
            ((0, self.num_pages),)
            if declared is None
            else tuple((int(offset), int(count)) for offset, count in declared)
        )
        if not ranges:
            raise invalid_descriptor("CachePool declares no KV groups")
        covered = [False] * self.num_pages
        for group, (offset, count) in enumerate(ranges):
            end = offset + count
            if offset < 0 or count < 1 or end > self.num_pages:
                raise invalid_descriptor(f"KV group {group} has invalid physical page bounds")
            for page in range(offset, end):
                if covered[page]:
                    raise invalid_descriptor("KV group physical page ranges overlap")
                covered[page] = True
        if not all(covered):
            raise invalid_descriptor("KV group physical page ranges do not cover the request pool")
        return ranges

    def validate_group(self, group: int) -> int:
        """Validate a cache-group index and return its normalized integer value."""

        value = int(group)
        if value < 0 or value >= self.group_count:
            raise invalid_descriptor(
                f"KV group {value} outside pool group count {self.group_count}"
            )
        return value

    def page_ids(self, group: int) -> range:
        """Expose allocatable non-sentinel page ids assigned to one cache group."""

        group_id = self.validate_group(group)
        offset, count = self.group_ranges[group_id]
        return range(max(1, offset), offset + count)

    def validate_pages(
        self,
        page_ids: Iterable[int],
        *,
        allow_sentinel: bool = False,
        group: int | None = None,
    ) -> tuple[int, ...]:
        """Validate physical page identifiers against one group, optionally accepting the sentinel page."""

        # Resident scheduler tables already use immutable integer tuples. Check
        # their validated identity before normalizing every element again.
        pages = tuple(page_ids)
        key = (pages, bool(allow_sentinel), group)
        cached = self._validated_page_tuples.get(key)
        if cached is not None:
            return cached
        pages = tuple(int(page) for page in pages)
        key = (pages, bool(allow_sentinel), group)
        validated = self._validate_page_tuple(pages, allow_sentinel, group)
        if len(self._validated_page_tuples) >= 16_384:
            self._validated_page_tuples.clear()
        self._validated_page_tuples[key] = validated
        return validated

    def _validate_page_tuple(
        self,
        pages: tuple[int, ...],
        allow_sentinel: bool,
        group: int | None,
    ) -> tuple[int, ...]:
        """Validate page identifiers, sentinel policy, and optional group ownership once per tuple."""

        real_pages = tuple(page for page in pages if page != 0)
        if len(set(real_pages)) != len(real_pages):
            raise invalid_descriptor("KV allocation repeats a physical page")
        lower = 0 if allow_sentinel else 1
        upper = self.num_pages
        if pages and (min(pages) < lower or max(pages) >= upper):
            raise invalid_descriptor("KV allocation exceeds the fixed physical pool")
        if group is not None:
            group_id = self.validate_group(group)
            offset, count = self.group_ranges[group_id]
            end = offset + count
            if any(page < offset or page >= end for page in real_pages):
                raise invalid_descriptor("KV allocation addresses another cache group")
        return pages

    def zero_pages(self, group: int, page_ids: Iterable[int]) -> None:
        """Zero every layer and field for the selected physical KV pages."""

        pages = self.validate_pages(page_ids, group=group)
        if not pages:
            return
        self.require_reusable(pages, group=group, start=0, length=len(pages) * self.block_size)
        self._clear_pages(pages)

    def initialize_import(self, write: CacheWrite) -> None:
        """Initialize the new pages covered by this active import reservation."""

        if not self.imports.owns(write):
            raise invalid_descriptor("KV initialization has no destination reservation")
        if write.initialized_pages:
            self._clear_pages(write.initialized_pages)

    def mark_import_scales(self, write: CacheWrite) -> None:
        """Publish initialization metadata for physically copied FP8 page scales."""

        if self._k_scale_set is None or self._v_scale_set is None:
            return
        if not self.imports.owns(write):
            raise invalid_descriptor("KV scale import has no destination reservation")
        publication = write.publication
        for page, _, _ in self._spans(
            write.pages,
            publication.base_extent,
            publication.published_extent - publication.base_extent,
        ):
            for layer in range(self.num_layers):
                index = layer * self.num_pages + page
                self._k_scale_set[index] = 1
                self._v_scale_set[index] = 1

    def _clear_pages(self, pages: tuple[int, ...]) -> None:
        # Clearing has no logical page order. Merge adjacent physical ranges so
        # initialization needs neither a gathered GPU index tensor nor one
        # kernel per page when the scheduler grants a contiguous allocation.
        ranges: list[tuple[int, int]] = []
        for page in sorted(pages):
            if ranges and ranges[-1][1] == page:
                ranges[-1] = (ranges[-1][0], page + 1)
            else:
                ranges.append((page, page + 1))
        for start, end in ranges:
            for store in (self.k, self.v):
                # E4M3 +0 has an all-zero byte representation on CPU and CUDA.
                values = store.view(torch.uint8) if self.is_quantized else store
                values[:, start:end].zero_()
            if self._scales is not None:
                self._scales[:, :, start:end].fill_(1)
        if self._k_scale_set is not None and self._v_scale_set is not None:
            for layer in range(self.num_layers):
                for page in pages:
                    index = layer * self.num_pages + page
                    self._k_scale_set[index] = 0
                    self._v_scale_set[index] = 0

    def layer_cache(self, layer: int, group: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Borrow layer storage; kernel callers authorize writes with require_writable()."""

        self.validate_group(group)
        layer_id = self._validate_layer(layer)
        if self.is_quantized:
            raise unsupported_setup(
                "paged attention cannot consume quantized KV storage without scale-aware kernels"
            )
        return self.k[layer_id], self.v[layer_id]

    def transfer_views(
        self,
        page_ids: Sequence[int],
        *,
        group: int,
        start: int,
        length: int,
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        """Borrow raw K, V and optional FP8 scales without packing or conversion.

        K/V have logical shape [tokens, layers, heads, dim]. Scales have shape
        [pages, 2, layers, 1], including both boundary pages of the token interval.
        The final scale axis represents this rank's group of local heads.
        Each field retains one allocation and its physical page order. The
        caller must hold the source publication or destination write ownership
        for the complete lifetime of these views and every asynchronous access.
        """

        pages = self.validate_pages(page_ids, group=group)
        spans = tuple(self._spans(pages, int(start), int(length)))
        if not spans:
            return ()
        fields = tuple(
            tuple(
                store[:, page, offset : offset + count].permute(1, 0, 2, 3)
                for page, offset, count in spans
            )
            for store in (self.k, self.v)
        )
        if self._scales is not None:
            fields += (
                tuple(
                    self._scales[:, :, page, 0, 0, 0].unsqueeze(0).unsqueeze(-1)
                    for page, _, _ in spans
                ),
            )
        return fields

    def read(
        self,
        layer: int,
        page_ids: Sequence[int],
        *,
        group: int = 0,
        start: int,
        length: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Gather a bounded token interval from selected pages into contiguous K/V tensors."""

        layer_id = self._validate_layer(layer)
        pages = self.validate_pages(page_ids, allow_sentinel=True, group=group)
        if length <= 0:
            return None, None
        spans = self._spans(pages, int(start), int(length))
        keys = [
            self._read_span(self.k, self.k_scale, layer_id, page, offset, count)
            for page, offset, count in spans
        ]
        values = [
            self._read_span(self.v, self.v_scale, layer_id, page, offset, count)
            for page, offset, count in spans
        ]
        if len(keys) == 1:
            return keys[0], values[0]
        return torch.cat(keys, dim=0), torch.cat(values, dim=0)

    def write(
        self,
        layer: int,
        page_ids: Sequence[int],
        *,
        group: int = 0,
        start: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Scatter contiguous K/V tensors into a bounded token interval of selected pages."""

        layer_id = self._validate_layer(layer)
        pages = self.validate_pages(page_ids, group=group)
        if k.shape != v.shape:
            raise invalid_descriptor("KV write key/value tensors do not align")
        if k.ndim != 3 or tuple(k.shape[1:]) != (self.n_kv, self.head_dim):
            raise invalid_descriptor("KV write tensors must have shape [tokens, heads, dim]")
        self.require_writable(pages, group=group, start=start, length=int(k.shape[0]))
        written = 0
        for page, offset, count in self._spans(pages, int(start), int(k.shape[0])):
            self._write_span(
                self.k,
                self.k_scale,
                self._k_scale_set,
                layer_id,
                page,
                offset,
                k[written : written + count],
            )
            self._write_span(
                self.v,
                self.v_scale,
                self._v_scale_set,
                layer_id,
                page,
                offset,
                v[written : written + count],
            )
            written += count

    def write_locations(
        self,
        layer: int,
        locations: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Write packed tokens within the caller's already authorized cache interval."""

        layer_id = self._validate_layer(layer)
        flat_locations = locations.reshape(-1).to(device=k.device, dtype=torch.int64)
        if k.shape != v.shape or k.ndim != 3 or int(k.shape[0]) != int(flat_locations.numel()):
            raise invalid_descriptor("KV output locations do not align with current K/V")
        if not self.is_quantized:
            persists = flat_locations > 0
            pages = torch.where(
                persists,
                torch.div(flat_locations, self.block_size, rounding_mode="floor"),
                -1,
            )
            offsets = torch.remainder(flat_locations, self.block_size)
            paged_kv_write(
                self.k[layer_id],
                self.v[layer_id],
                pages,
                offsets,
                k,
                v,
                cast=k.dtype != self.k.dtype or v.dtype != self.v.dtype,
            )
            return
        selected = torch.nonzero(flat_locations > 0, as_tuple=False).reshape(-1)
        if int(selected.numel()) == 0:
            return
        encoded = flat_locations.index_select(0, selected)
        pages = torch.div(encoded, self.block_size, rounding_mode="floor")
        offsets = torch.remainder(encoded, self.block_size)
        selected_k = k.index_select(0, selected)
        selected_v = v.index_select(0, selected)
        for index in range(int(selected.numel())):
            page = int(pages[index].item())
            offset = int(offsets[index].item())
            self._write_span(
                self.k,
                self.k_scale,
                self._k_scale_set,
                layer_id,
                page,
                offset,
                selected_k[index : index + 1],
            )
            self._write_span(
                self.v,
                self.v_scale,
                self._v_scale_set,
                layer_id,
                page,
                offset,
                selected_v[index : index + 1],
            )

    def _validate_layer(self, layer: int) -> int:
        """Validate and normalize a cache layer index."""

        value = int(layer)
        if value < 0 or value >= self.num_layers:
            raise invalid_descriptor(f"layer {value} outside KV pool layers {self.num_layers}")
        return value

    def _spans(
        self,
        page_ids: Sequence[int],
        start: int,
        length: int,
    ) -> tuple[tuple[int, int, int], ...]:
        """Partition a logical token interval into contiguous physical page spans."""

        if start < 0 or length < 0 or start + length > len(page_ids) * self.block_size:
            raise invalid_descriptor("KV token range exceeds its scheduler block table")
        spans: list[tuple[int, int, int]] = []
        cursor = start
        remaining = length
        while remaining:
            page_slot = cursor // self.block_size
            offset = cursor % self.block_size
            count = min(remaining, self.block_size - offset)
            spans.append((page_ids[page_slot], offset, count))
            cursor += count
            remaining -= count
        return tuple(spans)

    def _read_span(
        self,
        store: torch.Tensor,
        scales: torch.Tensor | None,
        layer: int,
        page: int,
        offset: int,
        count: int,
    ) -> torch.Tensor:
        """Read and dequantize one contiguous physical cache span."""

        span = store[layer, page, offset : offset + count]
        if not self.is_quantized:
            return span
        if scales is None:
            raise compute_error("quantized KV storage has no scale table", phase="kv_read")
        return dequantize_fp8_block(span, scales[layer, page], dtype=self.dtype)

    def _write_span(
        self,
        store: torch.Tensor,
        scales: torch.Tensor | None,
        scale_set: bytearray | None,
        layer: int,
        page: int,
        offset: int,
        values: torch.Tensor,
    ) -> None:
        """Write a contiguous logical span across pages with scale-aware FP8 storage."""

        count = int(values.shape[0])
        if not self.is_quantized:
            store[layer, page, offset : offset + count] = values.to(
                device=store.device,
                dtype=store.dtype,
            )
            return
        if scales is None or scale_set is None:
            raise compute_error("quantized KV storage has no scale state", phase="kv_write")
        values_f32 = values.to(device=store.device, dtype=torch.float32)
        index = layer * self.num_pages + page
        scale = scales[layer, page]
        if not scale_set[index]:
            scale.copy_(scale_for_fp8_block(values_f32))
            scale_set[index] = 1
        store[layer, page, offset : offset + count] = fp8_quantize(values_f32, scale)
