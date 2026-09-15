"""Request page assignment, cache publications, imports, and physical retirement."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from threading import RLock

import torch

from uniserve.quantization import QuantizedTensor
from uniserve.runtime import PrefixCache
from uniserve_worker.bootstrap.worker_info import KVCacheInfo

from ..foundation.errors import invalid_descriptor, resource_error
from ..protocol.identity import BufferId, ComputationId, RequestKey
from ..protocol.transfer import KvTransfer, Locator, TensorTransfer
from ..transfer.exports import ExportLocations, release_exports
from ..transfer.tickets import Transport, publish_tensor
from .block_tables import BlockTables, page_spans
from .cache_imports import CacheImport, CacheImports

__all__ = ["CacheManager"]


@dataclass(slots=True)
class CacheExport:
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
class CacheAccess:
    """Physical page intervals retained by one computation's existing completion fence."""

    completion: Future[None]
    requests: set[RequestKey]
    ranges: dict[int, tuple[int, int]]


class CacheManager:
    """Coordinate request ownership around a numerical cache without duplicating it."""

    def __init__(
        self,
        cache: PrefixCache,
        *,
        info: KVCacheInfo,
        group_ranges: Sequence[tuple[int, int]] | None = None,
        import_capacity: int = 1,
        request_pool_size: int = 1,
        max_blocks_per_request: int | None = None,
        staging_depth: int = 1,
    ) -> None:
        self.cache, self.info = cache, info
        self.layers = tuple(cache.config.layers)
        if len(self.layers) != info.num_layers:
            raise ValueError("cache backing must cover the advertised resident layers")
        for name in self.layers:
            state = cache.state(name)
            if state.key.shape != (
                info.num_blocks,
                info.block_size,
                info.num_kv_heads,
                info.head_dim,
            ):
                raise ValueError("cache backing must match the advertised page and head extents")
        self.compute_dtype = cache.config.layers[self.layers[0]].compute_dtype
        self.group_ranges = self._group_ranges(group_ranges)
        self.group_count = len(self.group_ranges)
        # Reuse normalized page tuples after their bounds and group ownership
        # have been established by the validation path.
        self._validated_page_tuples: dict[
            tuple[tuple[int, ...], bool, int | None], tuple[int, ...]
        ] = {}
        self.exports: dict[BufferId, ExportLocations] = {}
        self.export_releases: dict[BufferId, tuple[Future[None], ...]] = {}
        self._sources: dict[BufferId, CacheExport] = {}
        self._executions: dict[Future[None], CacheAccess] = {}
        self._execution_pages: dict[int, set[CacheAccess]] = {}
        self._execution_lock = RLock()
        self.block_tables = BlockTables(
            group_count=self.group_count,
            request_pool_size=request_pool_size,
            max_blocks_per_request=max(1, self.info.num_blocks - 1)
            if max_blocks_per_request is None
            else max_blocks_per_request,
            block_size=self.info.block_size,
            device=cache.device,
            staging_depth=staging_depth,
        )
        self._publications: dict[BufferId, KvTransfer] = {}
        self._destination_bases: dict[tuple[RequestKey, str], tuple[BufferId, int]] = {}
        self._installed_bases: dict[tuple[RequestKey, str], tuple[BufferId, int]] = {}
        self.imports = CacheImports(self, capacity=import_capacity)

    @contextmanager
    def startup_pages(self, count: int, *, group: int = 0):
        """Borrow bounded scratch pages before scheduler admission.

        The startup caller serializes this lease with other preparation and
        returns only after its stream finishes. Serving allocation authority
        remains with the scheduler; no request or publication is introduced.
        """

        if self.has_pending_accesses:
            raise RuntimeError("startup scratch requires an idle KV pool")
        available = self.page_ids(group)
        if not 0 <= count <= len(available):
            raise ValueError("startup scratch exceeds KV capacity")
        pages = tuple(available[:count])
        self.zero_pages(group, pages)
        try:
            yield pages
        finally:
            if self.cache.device.type == "cuda":
                torch.cuda.current_stream(self.cache.device).synchronize()
            self.zero_pages(group, pages)

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
        ranges = tuple(page_spans(pages, 0, length, page_size=self.info.block_size))
        with self._execution_lock:
            if completion.done():
                completion.result()
                return
            execution = self._executions.get(completion)
            register = execution is None
            if execution is None:
                execution = CacheAccess(completion, set(), {})
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
        if self._execution_dependencies(
            tuple(page_spans(page_ids, start, length, page_size=self.info.block_size))
        ):
            raise resource_error("KV interval still has an executing producer or consumer")

    def reserve_publication(
        self,
        buffer: BufferId,
        page_ids: Sequence[int],
        *,
        group: int,
        start: int,
        length: int,
    ) -> CacheExport:
        """Retain the exact published interval before exporting any of its views.

        The caller attaches every registration's retirement future and releases
        this reservation if publication is abandoned before semantic visibility.
        """

        self._reap_sources()
        pages = self.validate_pages(page_ids, group=group)
        ranges = {
            page: (offset, count)
            for page, offset, count in page_spans(
                pages, start, length, page_size=self.info.block_size
            )
        }
        if not ranges or buffer in self._sources:
            raise invalid_descriptor("KV publication has an empty or already registered interval")
        source = CacheExport(buffer, ranges)
        self._sources[source.buffer] = source
        return source

    def retain_publication(self, source: CacheExport, retirement: Future[None]) -> None:
        """Retain the published interval until this physical registration retires."""

        if self._sources.get(source.buffer) is not source or source.released:
            raise invalid_descriptor("KV publication reservation is no longer active")
        source.retirements = (*source.retirements, retirement)

    def release_buffers(self, buffers: Iterable[BufferId]) -> None:
        """Revoke semantic ownership while preserving every pending physical read."""

        selected = tuple(buffers)
        release_exports(self.exports, self.export_releases, selected)
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
        ranges = tuple(page_spans(pages, start, length, page_size=self.info.block_size))
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
        ranges = tuple(page_spans(pages, start, length, page_size=self.info.block_size))
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
        self.exports.clear()
        self.export_releases.clear()
        self.block_tables.close()
        self._publications.clear()
        self._destination_bases.clear()
        self._installed_bases.clear()
        self.cache.close()

    def _group_ranges(
        self,
        declared: Sequence[tuple[int, int]] | None,
    ) -> tuple[tuple[int, int], ...]:
        """Normalize physical cache-group ranges and require exact non-overlapping page coverage."""

        ranges = (
            ((0, self.info.num_blocks),)
            if declared is None
            else tuple((int(offset), int(count)) for offset, count in declared)
        )
        if not ranges:
            raise invalid_descriptor("KVCache declares no KV groups")
        covered = [False] * self.info.num_blocks
        for group, (offset, count) in enumerate(ranges):
            end = offset + count
            if offset < 0 or count < 1 or end > self.info.num_blocks:
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
        real_pages = tuple(page for page in pages if page != 0)
        if len(set(real_pages)) != len(real_pages):
            raise invalid_descriptor("KV allocation repeats a physical page")
        lower = 0 if allow_sentinel else 1
        upper = self.info.num_blocks
        if pages and (min(pages) < lower or max(pages) >= upper):
            raise invalid_descriptor("KV allocation exceeds the fixed physical pool")
        if group is not None:
            group_id = self.validate_group(group)
            offset, count = self.group_ranges[group_id]
            end = offset + count
            if any(page < offset or page >= end for page in real_pages):
                raise invalid_descriptor("KV allocation addresses another cache group")
        if len(self._validated_page_tuples) >= 16_384:
            self._validated_page_tuples.clear()
        self._validated_page_tuples[key] = pages
        return pages

    def zero_pages(self, group: int, page_ids: Iterable[int]) -> None:
        """Zero every layer and field for the selected physical KV pages."""

        pages = self.validate_pages(page_ids, group=group)
        if not pages:
            return
        self.require_reusable(pages, group=group, start=0, length=len(pages) * self.info.block_size)
        for name in self.layers:
            self.cache.zero_blocks(name, pages)

    def initialize_import(self, write: CacheImport) -> None:
        """Initialize the new pages covered by this active import reservation."""

        if not self.imports.owns(write):
            raise invalid_descriptor("KV initialization has no destination reservation")
        if write.initialized_pages:
            for name in self.layers:
                self.cache.zero_blocks(name, write.initialized_pages)

    def mark_import_scales(self, write: CacheImport) -> None:
        """Commit copied scale initialization for a still-owned import destination."""

        if not self.imports.owns(write):
            raise invalid_descriptor("KV scale import has no destination reservation")
        publication = write.publication
        pages = tuple(
            page
            for page, _, _ in page_spans(
                write.pages,
                publication.base_extent,
                publication.published_extent - publication.base_extent,
                self.info.block_size,
            )
        )
        for name in self.layers:
            self.cache.mark_initialized(name, pages, fields=("key", "value"))

    def destination_base(self, request_key: RequestKey, destination: str) -> BufferId | None:
        """Resolve the newest publication published to one destination for a request."""

        value = self._destination_bases.get((request_key, str(destination)))
        return None if value is None else value[0]

    def publish(
        self,
        *,
        request_pool_idx: int,
        group_id: int,
        visible_length: int,
        destination: str,
        expected_base: BufferId | None,
        buffer: BufferId,
        transports: Mapping[str, Transport],
    ) -> KvTransfer:
        """Export a visible KV extent under its exact buffer identity."""

        installed = self._destination_bases.get((buffer.owner, destination))
        if installed is None:
            if expected_base is not None:
                raise invalid_descriptor("KV publication expected base is not installed")
            base_extent = 0
        else:
            installed_buffer, base_extent = installed
            if installed_buffer != expected_base:
                raise invalid_descriptor("KV publication expected base does not match destination")
        pages = self.block_tables.pages(request_pool_idx, group_id)
        visible = int(visible_length)
        if visible > self.block_tables.allocated_length(request_pool_idx):
            raise invalid_descriptor("KV publication exceeds its scheduler block table")
        if visible < base_extent:
            raise invalid_descriptor("KV publication destination is ahead of its source")
        suffix = visible - base_extent
        source = (
            self.reserve_publication(
                buffer, pages, group=group_id, start=base_extent, length=suffix
            )
            if suffix
            else None
        )
        locators: list[Locator] = []
        tensors: list[TensorTransfer] = []
        try:
            if suffix:
                assert source is not None
                spans = page_spans(pages, base_extent, suffix, self.info.block_size)
                fields = ("key", "value")
                for field in fields:
                    locations = []
                    for layer, name in enumerate(self.layers, self.info.layer_offset):
                        tensor = self.cache.state(name).tensors[field]
                        values = (
                            tensor.buffers()["values"]
                            if isinstance(tensor, QuantizedTensor)
                            else tensor
                        )
                        views = tuple(
                            values[page, start : start + count].unsqueeze(1)
                            for page, start, count in spans
                        )
                        if isinstance(tensor, QuantizedTensor):
                            # Appending can enlarge a block's scale and re-encode
                            # its prefix. Freeze exported bytes so an immutable
                            # publication survives later numerical block updates.
                            views = (torch.cat(views, dim=0),)
                        exported = publish_tensor(
                            transports,
                            views,
                            retain=partial(self.retain_publication, source),
                            offset=(0, layer, self.info.kv_head_offset, 0),
                        )
                        locations.extend(exported)
                        locators.extend(exported)
                    tensors.append(
                        TensorTransfer(
                            shape=(
                                suffix,
                                self.info.total_layers,
                                self.info.total_kv_heads,
                                self.info.head_dim,
                            ),
                            locations=tuple(locations),
                        )
                    )
                if self.info.dtype == "float8_e4m3fn":
                    locations = []
                    for layer, name in enumerate(self.layers, self.info.layer_offset):
                        for field_index, field in enumerate(fields):
                            scales = self.cache.state(name).tensors[field].buffers()["scale"]
                            views = (
                                torch.cat(
                                    tuple(scales[page : page + 1] for page, _, _ in spans), dim=0
                                ),
                            )
                            exported = publish_tensor(
                                transports,
                                views,
                                retain=partial(self.retain_publication, source),
                                offset=(
                                    0,
                                    field_index,
                                    layer,
                                    self.info.kv_head_offset // self.info.num_kv_heads,
                                ),
                            )
                            locations.extend(exported)
                            locators.extend(exported)
                    tensors.append(
                        TensorTransfer(
                            shape=(
                                len(spans),
                                2,
                                self.info.total_layers,
                                self.info.total_kv_heads // self.info.num_kv_heads,
                            ),
                            locations=tuple(locations),
                        )
                    )
        except BaseException:
            for locator in locators:
                transports[locator.backend].release(locator)
            self.release_buffers((buffer,))
            raise
        publication = KvTransfer(
            tensors=tuple(tensors),
            source=buffer,
            destination=destination,
            base=expected_base,
            base_extent=base_extent,
            published_extent=visible,
            group_id=int(group_id),
            compute_dtype=str(self.compute_dtype).removeprefix("torch."),
            page_size=self.info.block_size,
        )
        return publication

    def publication(self, buffer: BufferId) -> KvTransfer:
        """Require the resident KV publication identified by a buffer identity."""

        try:
            return self._publications[buffer]
        except KeyError:
            raise invalid_descriptor("KV publication buffer is not resident") from None

    def resident(self, buffer: BufferId) -> KvTransfer | None:
        """Look up a resident KV publication without treating absence as an error."""

        return self._publications.get(buffer)

    def validate_conditioning(
        self,
        request_key: RequestKey,
        buffer: BufferId,
        *,
        request_pool_idx: int,
        group_id: int,
        visible_length: int,
        publication: KvTransfer | None = None,
    ) -> KvTransfer:
        """Verify that an installed buffer extends the request’s current compatible KV base."""

        publication = self.publication(buffer) if publication is None else publication
        if buffer.owner != request_key:
            raise invalid_descriptor("KV conditioning buffer belongs to another request")
        if (
            int(visible_length) < publication.published_extent
            or int(group_id) != publication.group_id
            or self.block_tables.allocated_length(request_pool_idx) < publication.published_extent
        ):
            raise invalid_descriptor("KV conditioning allocation disagrees with its publication")
        self.block_tables.pages(request_pool_idx, group_id)
        return publication

    def _validate_install(
        self, source: BufferId, publication: KvTransfer, *, group_id: int
    ) -> None:
        """Check semantic lineage and the raw representation before destination access."""

        if source != publication.source:
            raise invalid_descriptor("KV installation source identity is invalid")
        installed = self._installed_bases.get((source.owner, publication.destination))
        if publication.base is None:
            if installed is not None or publication.base_extent != 0:
                raise invalid_descriptor("KV installation base is invalid")
        elif installed != (publication.base, publication.base_extent):
            raise invalid_descriptor("KV installation base does not match destination")
        if int(group_id) != publication.group_id:
            raise invalid_descriptor("KV installation group disagrees with publication")
        if publication.tensors:
            suffix = publication.published_extent - publication.base_extent
            expected = (
                suffix,
                self.info.total_layers,
                self.info.total_kv_heads,
                self.info.head_dim,
            )
            if publication.tensors[0].shape != expected:
                raise invalid_descriptor("KV transfer shape does not match destination layers")

    def prepare_install(
        self,
        source: BufferId,
        publication: KvTransfer,
        *,
        request_pool_idx: int,
        group_id: int,
        page_ids: tuple[int, ...],
        allocated_length: int,
        initialized_pages: tuple[int, ...],
        transports: Mapping[str, Transport],
    ) -> CacheImport:
        """Reserve scheduler pages and start their bounded physical import."""

        self._validate_install(source, publication, group_id=group_id)
        pages = self.validate_pages(page_ids, group=group_id)
        initialized = self.validate_pages(initialized_pages, group=group_id)
        if (
            allocated_length > len(pages) * self.info.block_size
            or allocated_length < publication.published_extent
            or not set(initialized).issubset(pages)
        ):
            raise invalid_descriptor("KV import exceeds its scheduler block table")
        if publication.base_extent:
            base_pages = (
                publication.base_extent + self.info.block_size - 1
            ) // self.info.block_size
            installed_pages = self.block_tables.pages(request_pool_idx, group_id)[:base_pages]
            if pages[:base_pages] != installed_pages or set(initialized).intersection(
                installed_pages
            ):
                raise invalid_descriptor("KV import would replace its installed base pages")
        return self.imports.reserve(
            source,
            publication,
            request_pool_idx=request_pool_idx,
            group=group_id,
            pages=pages,
            initialized_pages=initialized,
            transports=transports,
        )

    def install(
        self,
        *,
        request_pool_idx: int,
        group_id: int,
        request_key: RequestKey,
        source: BufferId,
        installed_buffer: BufferId,
        write: CacheImport,
    ) -> KvTransfer:
        """Adopt a completed physical import under its exact source and base version."""

        publication = write.publication
        if (
            installed_buffer.owner != source.owner
            or source.owner != request_key
            or write.buffer != source
            or write.request_pool_idx != request_pool_idx
            or write.group_id != group_id
        ):
            raise invalid_descriptor("installed KV buffer identity is invalid")
        self._validate_install(source, publication, group_id=group_id)
        if (
            self.block_tables.pages(request_pool_idx, group_id) != write.pages
            or self.block_tables.allocated_length(request_pool_idx) < publication.published_extent
        ):
            raise invalid_descriptor("KV installation scheduler block table changed")
        self.imports.adopt(write)
        self.block_tables.set_verified(
            torch.tensor((request_pool_idx,), device=self.block_tables.page_tables.device),
            torch.tensor(
                (publication.published_extent,), device=self.block_tables.page_tables.device
            ),
        )
        return publication

    def validate_publications(
        self,
        publications: Sequence[tuple[BufferId, KvTransfer]],
        installations: Sequence[tuple[BufferId, BufferId, KvTransfer]],
    ) -> None:
        """Validate touched KV versions before any group resource becomes visible."""

        publications_by_buffer: dict[BufferId, KvTransfer] = {}
        destination_bases: dict[tuple[RequestKey, str], tuple[BufferId, int]] = {}
        installed_bases: dict[tuple[RequestKey, str], tuple[BufferId, int]] = {}
        for buffer, publication in publications:
            if buffer != publication.source:
                raise invalid_descriptor("KV publication buffer identity is invalid")
            existing = publications_by_buffer.get(buffer, self._publications.get(buffer))
            if existing is not None and existing != publication:
                raise invalid_descriptor("KV publication conflicts with its buffer identity")
            destination_key = (buffer.owner, publication.destination)
            current = destination_bases.get(
                destination_key, self._destination_bases.get(destination_key)
            )
            expected = (
                None if publication.base is None else (publication.base, publication.base_extent)
            )
            if current != expected:
                raise invalid_descriptor("KV publication base changed before publication")
            publications_by_buffer[buffer] = publication
            destination_bases[destination_key] = (
                publication.source,
                publication.published_extent,
            )
        for source, installed_buffer, publication in installations:
            if source != publication.source or installed_buffer.owner != source.owner:
                raise invalid_descriptor("installed KV buffer identity is invalid")
            destination_key = (installed_buffer.owner, publication.destination)
            current = installed_bases.get(
                destination_key, self._installed_bases.get(destination_key)
            )
            expected = (
                None if publication.base is None else (publication.base, publication.base_extent)
            )
            if current != expected:
                raise invalid_descriptor("KV installation base changed before publication")
            publications_by_buffer[source] = publication
            publications_by_buffer[installed_buffer] = publication
            installed_bases[destination_key] = (
                publication.source,
                publication.published_extent,
            )

    def commit_publications(
        self,
        publications: Sequence[tuple[BufferId, KvTransfer]],
        installations: Sequence[tuple[BufferId, BufferId, KvTransfer]],
    ) -> None:
        """Publish validated KV versions by updating only affected directory entries."""

        self.validate_publications(publications, installations)
        for buffer, publication in publications:
            self._publications[buffer] = publication
            self._destination_bases[(buffer.owner, publication.destination)] = (
                publication.source,
                publication.published_extent,
            )
        for source, installed_buffer, publication in installations:
            self._publications[source] = publication
            self._publications[installed_buffer] = publication
            self._installed_bases[(installed_buffer.owner, publication.destination)] = (
                publication.source,
                publication.published_extent,
            )

    def release_operations(
        self, releases: Sequence[tuple[RequestKey, ComputationId]]
    ) -> tuple[BufferId, ...]:
        """Forget semantic publications and identify buffers for the execution owner.

        Locator registration and physical retirement belong to execution's
        canonical publication table. Imported references may have no local
        registration; removing their semantic record does not release a remote
        publisher's storage.
        """

        identities = {(key, op_id) for key, op_id in releases}
        publications_by_buffer = tuple(
            buffer
            for buffer in self._publications
            if (buffer.owner, buffer.producer_op_id) in identities
        )
        for buffer in publications_by_buffer:
            del self._publications[buffer]
        return publications_by_buffer

    def drop(self, request_id: int) -> None:
        """Discard semantic KV state while execution retires the request's registrations."""

        selected = tuple(
            buffer
            for buffer in self._publications
            if int(buffer.owner.request_id) == int(request_id)
        )
        for buffer in selected:
            del self._publications[buffer]
        for table in (self._destination_bases, self._installed_bases):
            for key in tuple(key for key in table if int(key[0].request_id) == int(request_id)):
                del table[key]
