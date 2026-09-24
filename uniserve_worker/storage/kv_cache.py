"""Request page assignment, cache publications, imports, and retirement.

``KVCacheManager`` wraps one ``PrefixCache`` of paged MHA K/V state on a
worker rank and decides when each physical page interval may be rewritten
or reused. Page allocation belongs to the engine scheduler; this manager
validates the page ids it assigns and owns the request block tables
(``BlockTables``) that model calls index. Three kinds of owner retain page
intervals, each as ``page -> (token offset, token count)``:

- Execution accesses (``CacheAccess``): the pages a model call reads or
  writes, retained until the batch's completion future succeeds.
- Publications (``CacheExport``): an immutable token interval exported
  through transports, retained until the buffer is released and every
  physical registration has retired.
- Imports (``CacheImports``): scheduler-assigned destination pages that an
  import stream fills from a publication.

Writers ask ``write_dependencies`` which futures must resolve first, and
``require_writable`` and ``require_reusable`` reject writes that still
overlap a retained interval. The manager also keeps the semantic
publication directory: the ``KvTransfer`` each buffer identity names and,
per ``(request, destination)``, the lineage base that the next incremental
publication or installation must extend.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from threading import RLock

import torch

from uniserve.cache import block_spans, mha
from uniserve.runtime import PrefixCache
from uniserve_worker.errors import invalid_descriptor, resource_error
from uniserve_worker.protocol.identity import BufferId, CallId, RequestKey
from uniserve_worker.protocol.transfer import (
    KvTransfer,
    Locator,
    TensorTransfer,
)
from uniserve_worker.protocol.worker_info import KVCacheInfo
from uniserve_worker.storage.block_tables import BlockTables
from uniserve_worker.storage.cache_imports import CacheImport, CacheImports
from uniserve_worker.transport.exports import ExportLocations, release_exports
from uniserve_worker.transport.interface import Transport
from uniserve_worker.transport.publication import publish_tensor

__all__ = ["KVCacheManager"]


@dataclass(slots=True)
class CacheExport:
    """An immutable token interval retained by its physical publications.

    Ranges map each physical page to its token offset and token count. A
    publication covers every layer's K/V for these ranges, so appends outside
    the interval remain independent even when they share its final physical
    page.

    Attributes:
        buffer: Buffer identity the publication is registered under.
        ranges: Retained interval per physical page.
        retirements: One future per physical registration, attached through
            ``KVCacheManager.retain_publication``.
        released: Whether semantic ownership has been revoked. The entry
            leaves the manager only when it is released and every retirement
            has succeeded.
    """

    buffer: BufferId
    ranges: dict[int, tuple[int, int]]
    retirements: tuple[Future[None], ...] = ()
    released: bool = False


@dataclass(eq=False, slots=True)
class CacheAccess:
    """Physical page intervals retained by one computation's completion.

    Every model access that shares one ``completion`` future joins one
    access; ``requests`` names their request keys and ``ranges`` keeps one
    interval per page.
    """

    completion: Future[None]
    requests: set[RequestKey]
    ranges: dict[int, tuple[int, int]]


class KVCacheManager:
    """Coordinate request ownership around one numerical cache.

    Physical page ``0`` is the padding sentinel and is never allocatable.
    """

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
        """Validate the cache backing and create the ownership tables.

        Args:
            cache: Paged numerical K/V storage; closed by ``close``.
            info: Advertised page count, page size, head and layer extents.
            group_ranges: Physical ``(first page, page count)`` per cache
                group. Defaults to one group covering the whole pool.
            import_capacity: Maximum copy tasks ``CacheImports`` admits at
                once.
            request_pool_size: Request slots of the block tables.
            max_blocks_per_request: Block-table width; defaults to every
                non-sentinel page.
            staging_depth: Depth of the block tables' host staging rings.

        Raises:
            ValueError: When the backing does not match ``info``, does not
                hold MHA state, or ``staging_depth`` is below 1.
            WorkerError: ``invalid_descriptor`` when the group ranges do not
                tile the physical pool exactly or a block-table dimension is
                below 1.
        """
        self.cache, self.info = cache, info
        self.layers = tuple(cache.config.layers)
        if len(self.layers) != info.num_layers:
            raise ValueError(
                "cache backing must cover the advertised resident layers"
            )
        for name in self.layers:
            state = cache.state(name)
            if not isinstance(state, mha.State):
                raise ValueError("cache backing must hold MHA state layers")
            # Physical K/V pages are
            # [page, token within page, KV head, head dim].
            if state.key.shape != (
                info.num_blocks,
                info.block_size,
                info.num_kv_heads,
                info.head_dim,
            ):
                raise ValueError(
                    "cache backing must match the advertised page and head "
                    "extents"
                )

        layout = cache.config.layers[self.layers[0]]
        if not isinstance(layout, mha.Config):
            raise ValueError("cache backing must hold MHA state layers")
        self.compute_dtype = layout.compute_dtype
        self.group_ranges = self._group_ranges(group_ranges)
        self.group_count = len(self.group_ranges)
        # Reuse normalized page tuples after their bounds and group ownership
        # have been established by the validation path.
        self._validated_page_tuples: dict[
            tuple[tuple[int, ...], bool, int | None], tuple[int, ...]
        ] = {}

        # Transport locations of committed exports, updated by the batch
        # commit; publication intervals retained under their buffers.
        self.exports: dict[BufferId, ExportLocations] = {}
        self._sources: dict[BufferId, CacheExport] = {}

        # Execution accesses by completion future, indexed by page.
        # ``_execution_completed`` runs as a future callback on the thread
        # that resolves the completion, so both maps are mutated under
        # ``_execution_lock``.
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

        # Semantic directory, changed by ``apply_publications``,
        # ``release_calls`` and ``drop`` and cleared by ``close``. Base maps
        # hold the latest ``(source buffer, published extent)`` per
        # ``(request, destination)``: ``_destination_bases`` for what this
        # rank published and ``_installed_bases`` for what it installed.
        self._publications: dict[BufferId, KvTransfer] = {}
        self._destination_bases: dict[
            tuple[RequestKey, str], tuple[BufferId, int]
        ] = {}
        self._installed_bases: dict[
            tuple[RequestKey, str], tuple[BufferId, int]
        ] = {}

        self.imports = CacheImports(self, capacity=import_capacity)

    @contextmanager
    def startup_pages(self, count: int, *, group: int = 0):
        """Borrow bounded scratch pages before scheduler admission.

        Yields the first ``count`` allocatable pages of ``group``, zeroed on
        entry and again on exit, after the current stream drains on CUDA. The
        startup caller serializes this lease with other preparation. Serving
        allocation authority remains with the scheduler; no request or
        publication is introduced.

        Raises:
            RuntimeError: When any cache interval is still retained.
            ValueError: When ``count`` is negative or exceeds the group's
                allocatable pages.
            WorkerError: ``invalid_descriptor`` when ``group`` is out of
                range.
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
        """Whether any publication, execution access or import is retained."""
        return (
            bool(self._sources) or bool(self._executions) or bool(self.imports)
        )

    def retain_execution(
        self,
        request: RequestKey,
        page_ids: Sequence[int],
        *,
        group: int,
        length: int,
        completion: Future[None],
    ) -> None:
        """Retain a model access until its device work completes.

        The access covers tokens ``[0, length)`` of ``page_ids`` in
        ``group``. Model calls are ordered by the runner; this retention
        keeps independent import streams and page reuse
        (``require_reusable``, ``write_dependencies``) away from these ranges
        while a producer or consumer kernel may still run. A zero ``length``
        retains nothing, and an already resolved ``completion`` retains
        nothing and re-raises its failure or cancellation.

        Raises:
            WorkerError: ``invalid_descriptor`` when the pages are invalid
                for ``group``.
        """
        if not length:
            return
        pages = self.validate_pages(page_ids, group=group)
        ranges = tuple(
            block_spans(pages, 0, length, block_size=self.info.block_size)
        )

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

            # Each page keeps one interval: the hull of every span retained
            # on it by this access.
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
        # A failed or cancelled completion leaves its access registered: its
        # ranges stay blocked, ``retirement_ready`` re-raises the failure for
        # its requests, and ``close`` refuses to proceed.
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
        """Return completions of accesses that overlap the given spans."""
        with self._execution_lock:
            return tuple(
                {
                    execution.completion
                    for page, offset, count in ranges
                    for execution in self._execution_pages.get(page, ())
                    if self._ranges_overlap(
                        execution.ranges, ((page, offset, count),)
                    )
                }
            )

    def require_reusable(
        self, page_ids: Sequence[int], *, group: int, start: int, length: int
    ) -> None:
        """Authorize page initialization or stream import before submission.

        Applies ``require_writable`` and also rejects intervals that an
        execution access still retains, including one whose completion
        failed or was cancelled. ``zero_pages`` and
        ``CacheImports.reserve`` call this before writing pages.

        Raises:
            WorkerError: A resource error when the interval overlaps a
                publication, an import destination or an execution access;
                ``invalid_descriptor`` when the pages are invalid and a
                publication or import is retained.
        """
        self.require_writable(page_ids, group=group, start=start, length=length)
        if self._execution_dependencies(
            tuple(
                block_spans(
                    page_ids, start, length, block_size=self.info.block_size
                )
            )
        ):
            raise resource_error(
                "KV interval still has an executing producer or consumer"
            )

    def reserve_publication(
        self,
        buffer: BufferId,
        page_ids: Sequence[int],
        *,
        group: int,
        start: int,
        length: int,
    ) -> CacheExport:
        """Retain the exact published interval before exporting any view.

        The caller attaches every registration's retirement future with
        ``retain_publication`` and releases this reservation
        (``release_buffers``) if publication is abandoned before semantic
        visibility.

        Raises:
            WorkerError: ``invalid_descriptor`` when the pages are invalid,
                the interval is empty, or ``buffer`` already has a retained
                interval.
        """
        self._reap_sources()
        pages = self.validate_pages(page_ids, group=group)
        ranges = {
            page: (offset, count)
            for page, offset, count in block_spans(
                pages, start, length, block_size=self.info.block_size
            )
        }
        if not ranges or buffer in self._sources:
            raise invalid_descriptor(
                "KV publication has an empty or already registered interval"
            )
        source = CacheExport(buffer, ranges)
        self._sources[source.buffer] = source
        return source

    def retain_publication(
        self, source: CacheExport, retirement: Future[None]
    ) -> None:
        """Retain the published interval until this registration retires.

        Raises:
            WorkerError: ``invalid_descriptor`` when the reservation has been
                released or is no longer the one registered for its buffer.
        """
        if self._sources.get(source.buffer) is not source or source.released:
            raise invalid_descriptor(
                "KV publication reservation is no longer active"
            )
        source.retirements = (*source.retirements, retirement)

    def release_buffers(self, buffers: Iterable[BufferId]) -> None:
        """Revoke semantic ownership while preserving every pending read.

        Committed export registrations stop accepting new readers, imports
        of these buffers are abandoned, and their publication intervals are
        marked released. An interval is dropped only once all of its
        retirements have succeeded.
        """
        selected = tuple(buffers)
        release_exports(self.exports, selected)
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
        """Report whether the selected owners retain no cache interval.

        Publication intervals and imports are selected when their buffer is
        in ``buffers``, or when a request in ``requests`` owns it and it is
        not in ``retained``. Execution accesses are selected by request only.
        Failures of the selected publications' retirements and execution
        completions that have already resolved are re-raised here, so errors
        surface only for these owners.
        """
        selected = set(buffers)
        owners = set(requests)
        sources = tuple(
            source
            for buffer, source in self._sources.items()
            if buffer in selected
            or (buffer.owner in owners and buffer not in retained)
        )
        for source in sources:
            for future in source.retirements:
                if future.done():
                    future.result()
        self._reap_sources()

        # Free retires a publication, not the request's resident KV pages.
        # Unrelated products can share that request while later computation
        # still reads its prefix. Only request retirement waits for all such
        # call kinds; physical page reuse separately checks their ranges.
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
        """Return the futures that must resolve before writing an interval.

        These are the completions of overlapping execution accesses and the
        retirements of overlapping import destinations and publications.
        Returns an empty tuple without validating the pages when nothing is
        retained.

        Raises:
            WorkerError: ``invalid_descriptor`` when the pages are invalid.
        """
        if not self.has_pending_accesses:
            return ()
        self._reap_sources()
        pages = self.validate_pages(page_ids, group=group)
        ranges = tuple(
            block_spans(pages, start, length, block_size=self.info.block_size)
        )
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
        When no publication or import is retained, this returns without
        validating the pages.

        Raises:
            WorkerError: A resource error when the interval overlaps a
                publication or an import destination; ``invalid_descriptor``
                when the pages are invalid.
        """
        # The runner orders model accesses. Only independent imports and
        # published immutable ranges add write conflicts at this boundary.
        if not self._sources and not self.imports:
            return
        self._reap_sources()
        pages = self.validate_pages(page_ids, group=group)
        ranges = tuple(
            block_spans(pages, start, length, block_size=self.info.block_size)
        )
        if any(
            self._ranges_overlap(source.ranges, ranges)
            for source in self._sources.values()
        ):
            raise resource_error("KV interval still has a published version")
        if self.imports.dependencies(ranges):
            raise resource_error("KV interval still has an import destination")

    @staticmethod
    def _ranges_overlap(
        left: Mapping[int, tuple[int, int]],
        right: Sequence[tuple[int, int, int]],
    ) -> bool:
        """Whether any ``right`` span meets ``left``'s interval on its page.

        Intervals overlap only on the same page and only when their token
        ranges intersect.
        """
        return any(
            (other := left.get(page)) is not None
            and offset < other[0] + other[1]
            and other[0] < offset + count
            for page, offset, count in right
        )

    def _reap_sources(self) -> None:
        # A failed or cancelled retirement keeps its interval, so
        # ``retirement_ready`` re-raises the failure and ``close`` refuses.
        for buffer, source in tuple(self._sources.items()):
            if source.released and all(
                future.done()
                and not future.cancelled()
                and future.exception() is None
                for future in source.retirements
            ):
                del self._sources[buffer]

    def close(self) -> None:
        """Release every publication and close the backing cache.

        Imports are stopped first. The backing cache is closed only when no
        import, execution access or publication interval remains.

        Raises:
            WorkerError: A resource error when any of those still retains
                storage; the cache stays open in that case.
        """
        self.imports.stop()
        self.release_buffers(tuple(self._sources))
        self.imports.require_retired()

        if self._executions:
            raise resource_error(
                "KV cache still has executing producers or consumers"
            )
        if self._sources:
            raise resource_error(
                "KV cache still has unretired physical publications"
            )

        self.exports.clear()
        self.block_tables.close()
        self._publications.clear()
        self._destination_bases.clear()
        self._installed_bases.clear()
        self.cache.close()

    def _group_ranges(
        self,
        declared: Sequence[tuple[int, int]] | None,
    ) -> tuple[tuple[int, int], ...]:
        """Normalize physical cache-group ranges.

        The ranges must tile ``[0, num_blocks)`` exactly: no gaps, no
        overlap. Page ``0`` lies in whichever group covers it but is never
        allocatable (``page_ids``).

        Raises:
            WorkerError: ``invalid_descriptor`` when the ranges are empty,
                out of bounds, overlapping or incomplete.
        """
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
                raise invalid_descriptor(
                    f"KV group {group} has invalid physical page bounds"
                )
            for page in range(offset, end):
                if covered[page]:
                    raise invalid_descriptor(
                        "KV group physical page ranges overlap"
                    )
                covered[page] = True

        if not all(covered):
            raise invalid_descriptor(
                "KV group physical page ranges do not cover the request pool"
            )
        return ranges

    def validate_group(self, group: int) -> int:
        """Return ``group`` as an integer after checking its range.

        Raises:
            WorkerError: ``invalid_descriptor`` when ``group`` is outside
                ``[0, group_count)``.
        """
        value = int(group)
        if value < 0 or value >= self.group_count:
            raise invalid_descriptor(
                f"KV group {value} outside pool group count {self.group_count}"
            )
        return value

    def page_ids(self, group: int) -> range:
        """Return the allocatable page ids of one cache group.

        The sentinel page ``0`` is excluded.
        """
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
        """Validate physical page ids and return them as a tuple of ints.

        Real (non-zero) pages must be unique, in ``[1, num_blocks)`` and,
        when ``group`` is given, inside that group's range. The sentinel page
        ``0`` is accepted, possibly repeated, only with ``allow_sentinel``.
        Accepted tuples are memoized; the checks depend only on the pool
        layout fixed at construction.

        Raises:
            WorkerError: ``invalid_descriptor`` when any check fails.
        """
        # Resident scheduler tables already use immutable integer tuples. Check
        # their validated identity before normalizing every element again.
        pages = tuple(page_ids)
        key = (pages, bool(allow_sentinel), group)
        cached = self._validated_page_tuples.get(key)
        if cached is not None:
            return cached
        pages = tuple(int(page) for page in pages)
        key = (pages, bool(allow_sentinel), group)

        # Page 0 is the padding sentinel; uniqueness and group bounds apply
        # only to real pages.
        real_pages = tuple(page for page in pages if page != 0)
        if len(set(real_pages)) != len(real_pages):
            raise invalid_descriptor("KV allocation repeats a physical page")

        lower = 0 if allow_sentinel else 1
        upper = self.info.num_blocks
        if pages and (min(pages) < lower or max(pages) >= upper):
            raise invalid_descriptor(
                "KV allocation exceeds the fixed physical pool"
            )

        if group is not None:
            group_id = self.validate_group(group)
            offset, count = self.group_ranges[group_id]
            end = offset + count
            if any(page < offset or page >= end for page in real_pages):
                raise invalid_descriptor(
                    "KV allocation addresses another cache group"
                )

        # Bound the memo by clearing it wholesale.
        if len(self._validated_page_tuples) >= 16_384:
            self._validated_page_tuples.clear()
        self._validated_page_tuples[key] = pages
        return pages

    def zero_pages(self, group: int, page_ids: Iterable[int]) -> None:
        """Zero every layer and field for the selected physical KV pages.

        Raises:
            WorkerError: ``invalid_descriptor`` when the pages are invalid
                for ``group``; a resource error from ``require_reusable``
                when any selected page is still retained.
        """
        pages = self.validate_pages(page_ids, group=group)
        if not pages:
            return
        self.require_reusable(
            pages,
            group=group,
            start=0,
            length=len(pages) * self.info.block_size,
        )
        for name in self.layers:
            self.cache.zero_blocks(name, pages)

    def _layer_stacks(
        self, buffer: str
    ) -> tuple[tuple[int, torch.Tensor], ...]:
        """Pair each allocation run of `buffer` with its first logical layer.

        Runs cover the resident layers in order, so a run's first logical
        layer is this rank's layer offset plus the layers of earlier runs.
        """
        stacks = []
        layer = self.info.layer_offset
        for names, stack in self.cache.layer_stacks(buffer):
            stacks.append((layer, stack))
            layer += len(names)
        return tuple(stacks)

    def publish(
        self,
        *,
        request_pool_idx: int,
        group_id: int,
        visible_length: int,
        destination: str,
        buffer: BufferId,
        transports: Mapping[str, Transport],
        consumers: Sequence[int] = (),
    ) -> KvTransfer:
        """Export a visible KV extent under its exact buffer identity.

        Publications to one ``(request, destination)`` form a chain: this one
        exports only tokens ``[base_extent, visible_length)``, where the base
        is the latest committed publication to that destination. An empty
        suffix exports no tensors. The suffix interval is reserved before
        any view is exported. The returned ``KvTransfer`` becomes resident
        only when the batch commits it (``validate_publications`` then
        ``apply_publications``).

        `consumers` are the acknowledgment slots of the ranks that install it.

        Raises:
            WorkerError: ``invalid_descriptor`` when the slot has no installed
                table for ``group_id``, ``visible_length`` exceeds its
                allocated length or trails the destination base, or the
                reservation fails. A transport failure propagates after every
                exported locator and the reservation are released.
        """
        installed = self._destination_bases.get((buffer.owner, destination))
        base, base_extent = (None, 0) if installed is None else installed

        pages = self.block_tables.pages(request_pool_idx, group_id)
        visible = int(visible_length)
        if visible > self.block_tables.allocated_length(request_pool_idx):
            raise invalid_descriptor(
                "KV publication exceeds its scheduler block table"
            )
        if visible < base_extent:
            raise invalid_descriptor(
                "KV publication destination is ahead of its source"
            )

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
                spans = block_spans(
                    pages, base_extent, suffix, self.info.block_size
                )
                encoded = self.info.dtype == "float8_e4m3fn"
                fields = ("key", "value")
                # Each run of layers sharing one backing exports as a single
                # tensor per mechanism, so the descriptor's locator count
                # follows the cache's allocation runs and the rank's
                # mechanisms, never the model's depth, which keeps the
                # descriptor within its byte bound.
                for field in fields:
                    locations: list[Locator] = []
                    for layer, stack in self._layer_stacks(f"{field}.values"):
                        # Stacks are [layers, pages, page tokens, kv heads,
                        # head dim]; each span view is [tokens, layers, kv
                        # heads, head dim] over the run's layers. Unencoded
                        # views alias the live pages: ``require_writable``
                        # rejects writes into the reserved interval until it
                        # is released and retired.
                        views = tuple(
                            stack[:, page, start : start + count].permute(
                                1, 0, 2, 3
                            )
                            for page, start, count in spans
                        )
                        if encoded:
                            # Appending can enlarge a block's scale and
                            # re-encode its prefix. Freeze exported bytes so
                            # an immutable publication survives later
                            # numerical block updates.
                            views = (torch.cat(views, dim=0),)
                        # The offset places this run inside the global
                        # [suffix, total layers, total KV heads, head dim]
                        # transfer at the run's first layer and this rank's
                        # first KV head.
                        exported = publish_tensor(
                            transports,
                            views,
                            retain=partial(self.retain_publication, source),
                            offset=(0, layer, self.info.kv_head_offset, 0),
                            consumers=consumers,
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

                # FP8 exports additionally carry one scale row per
                # published page.
                if encoded:
                    locations = []
                    for field_index, field in enumerate(fields):
                        for layer, stack in self._layer_stacks(
                            f"{field}.scale"
                        ):
                            # Scale stacks are [layers, pages, 1, 1, 1] with
                            # one scale per page; published rows are [page,
                            # K/V, layers, head group], frozen like the values
                            # they encode. A rank's head group is its KV-head
                            # offset divided by its local KV-head count.
                            views = (
                                torch.cat(
                                    tuple(
                                        stack[:, page].reshape(1, 1, -1, 1)
                                        for page, _, _ in spans
                                    ),
                                    dim=0,
                                ),
                            )
                            exported = publish_tensor(
                                transports,
                                views,
                                retain=partial(self.retain_publication, source),
                                consumers=consumers,
                                offset=(
                                    0,
                                    field_index,
                                    layer,
                                    self.info.kv_head_offset
                                    // self.info.num_kv_heads,
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
                                self.info.total_kv_heads
                                // self.info.num_kv_heads,
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
            base=base,
            base_extent=base_extent,
            published_extent=visible,
            group_id=int(group_id),
            compute_dtype=str(self.compute_dtype).removeprefix("torch."),
            page_size=self.info.block_size,
        )
        return publication

    def publication(self, buffer: BufferId) -> KvTransfer:
        """Return the resident KV publication registered for ``buffer``.

        Raises:
            WorkerError: ``invalid_descriptor`` when none is resident.
        """
        try:
            return self._publications[buffer]
        except KeyError:
            raise invalid_descriptor(
                "KV publication buffer is not resident"
            ) from None

    def resident(self, buffer: BufferId) -> KvTransfer | None:
        """Look up a resident KV publication; absence is not an error."""
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
        """Verify that a request's allocation still covers a publication.

        ``buffer`` must belong to ``request_key``, ``group_id`` must be the
        publication's group, the published extent must lie within both
        ``visible_length`` and the slot's allocated length, and the slot must
        have an installed table for the group. ``publication`` skips
        the directory lookup when the caller already holds it.

        Returns:
            The publication.

        Raises:
            WorkerError: ``invalid_descriptor`` when any check fails or the
                publication is not resident.
        """
        publication = (
            self.publication(buffer) if publication is None else publication
        )
        if buffer.owner != request_key:
            raise invalid_descriptor(
                "KV conditioning buffer belongs to another request"
            )
        if (
            int(visible_length) < publication.published_extent
            or int(group_id) != publication.group_id
            or self.block_tables.allocated_length(request_pool_idx)
            < publication.published_extent
        ):
            raise invalid_descriptor(
                "KV conditioning allocation disagrees with its publication"
            )
        self.block_tables.pages(request_pool_idx, group_id)
        return publication

    def _validate_install(self, publication: KvTransfer) -> None:
        """Check lineage and transfer shape before destination access.

        A first installation into ``(request, destination)`` has no base and
        a zero base extent; a later one must name the currently installed
        base and extent. When the publication carries tensors, the first
        must have this worker's global
        ``[suffix, total layers, total KV heads, head dim]`` shape.

        Raises:
            WorkerError: ``invalid_descriptor`` when either check fails.
        """
        installed = self._installed_bases.get(
            (publication.source.owner, publication.destination)
        )
        if publication.base is None:
            if installed is not None or publication.base_extent != 0:
                raise invalid_descriptor("KV installation base is invalid")
        elif installed != (publication.base, publication.base_extent):
            raise invalid_descriptor(
                "KV installation base does not match destination"
            )
        if publication.tensors:
            suffix = publication.published_extent - publication.base_extent
            expected = (
                suffix,
                self.info.total_layers,
                self.info.total_kv_heads,
                self.info.head_dim,
            )
            if publication.tensors[0].shape != expected:
                raise invalid_descriptor(
                    "KV transfer shape does not match destination layers"
                )

    def prepare_install(
        self,
        publication: KvTransfer,
        *,
        request_pool_idx: int,
        page_ids: tuple[int, ...],
        allocated_length: int,
        initialized_pages: tuple[int, ...],
        transports: Mapping[str, Transport],
    ) -> CacheImport:
        """Reserve scheduler pages and start their bounded physical import.

        ``page_ids`` is the destination's page table, ``allocated_length``
        its token capacity, and ``initialized_pages`` the new pages the
        import zeroes before copying. Pages that hold the installed base must
        stay in place and must not be re-initialized. The copy runs in
        ``CacheImports``; ``install`` adopts it once complete.

        Raises:
            WorkerError: ``invalid_descriptor`` when lineage, shape, pages or
                capacity are invalid, imports are closed, or the publication
                source already has a registered import; a resource error
                when a destination interval is still retained or the import
                lane is closed or has no capacity.
        """
        self._validate_install(publication)
        group_id = publication.group_id
        pages = self.validate_pages(page_ids, group=group_id)
        initialized = self.validate_pages(initialized_pages, group=group_id)
        if (
            allocated_length > len(pages) * self.info.block_size
            or allocated_length < publication.published_extent
            or not set(initialized).issubset(pages)
        ):
            raise invalid_descriptor(
                "KV import exceeds its scheduler block table"
            )

        if publication.base_extent:
            base_pages = (
                publication.base_extent + self.info.block_size - 1
            ) // self.info.block_size
            # The installed base pages must stay identical and must not be
            # re-initialized by this import.
            installed_pages = self.block_tables.pages(
                request_pool_idx, group_id
            )[:base_pages]
            if pages[:base_pages] != installed_pages or set(
                initialized
            ).intersection(installed_pages):
                raise invalid_descriptor(
                    "KV import would replace its installed base pages"
                )

        return self.imports.reserve(
            publication,
            request_pool_idx=request_pool_idx,
            pages=pages,
            initialized_pages=initialized,
            transports=transports,
        )

    def install(
        self,
        *,
        installed_buffer: BufferId,
        write: CacheImport,
    ) -> KvTransfer:
        """Adopt a completed physical import under its source and base.

        The request's block table must be unchanged since
        ``prepare_install``. The slot's verified length becomes the published
        extent; the returned ``KvTransfer`` becomes resident when the batch
        commits (``apply_publications``).

        Raises:
            RuntimeError: When the import has not completed.
            WorkerError: ``invalid_descriptor`` when identities, lineage or
                the block table disagree, or the import was abandoned.
            Exception: The import's own failure, re-raised.
        """
        publication = write.publication
        if installed_buffer.owner != publication.source.owner:
            raise invalid_descriptor("installed KV buffer identity is invalid")

        self._validate_install(publication)
        request_pool_idx, group_id = (
            write.request_pool_idx,
            publication.group_id,
        )
        if (
            self.block_tables.pages(request_pool_idx, group_id) != write.pages
            or self.block_tables.allocated_length(request_pool_idx)
            < publication.published_extent
        ):
            raise invalid_descriptor(
                "KV installation scheduler block table changed"
            )

        self.imports.adopt(write)
        self.block_tables.set_verified(
            torch.tensor(
                (request_pool_idx,), device=self.block_tables.page_tables.device
            ),
            torch.tensor(
                (publication.published_extent,),
                device=self.block_tables.page_tables.device,
            ),
        )
        return publication

    def validate_publications(
        self,
        publications: Sequence[tuple[BufferId, KvTransfer]],
        installations: Sequence[tuple[BufferId, BufferId, KvTransfer]],
    ) -> None:
        """Validate touched KV versions before any group resource is visible.

        Checks the batch's publications and installations without changing
        the directory. Several entries for one ``(request, destination)``
        must chain in order.

        Raises:
            WorkerError: ``invalid_descriptor`` when a buffer identity does
                not match its transfer, a resident buffer names another
                transfer, or a base is not the current one.
        """
        # Transaction-local views start from resident state, so the batch is
        # validated as one consistent step.
        publications_by_buffer: dict[BufferId, KvTransfer] = {}
        destination_bases: dict[
            tuple[RequestKey, str], tuple[BufferId, int]
        ] = {}
        installed_bases: dict[tuple[RequestKey, str], tuple[BufferId, int]] = {}

        for buffer, publication in publications:
            if buffer != publication.source:
                raise invalid_descriptor(
                    "KV publication buffer identity is invalid"
                )
            existing = publications_by_buffer.get(
                buffer, self._publications.get(buffer)
            )
            if existing is not None and existing != publication:
                raise invalid_descriptor(
                    "KV publication conflicts with its buffer identity"
                )
            destination_key = (buffer.owner, publication.destination)
            current = destination_bases.get(
                destination_key, self._destination_bases.get(destination_key)
            )
            expected = (
                None
                if publication.base is None
                else (publication.base, publication.base_extent)
            )
            if current != expected:
                raise invalid_descriptor(
                    "KV publication base changed before publication"
                )
            publications_by_buffer[buffer] = publication
            destination_bases[destination_key] = (
                publication.source,
                publication.published_extent,
            )

        for source, installed_buffer, publication in installations:
            if (
                source != publication.source
                or installed_buffer.owner != source.owner
            ):
                raise invalid_descriptor(
                    "installed KV buffer identity is invalid"
                )
            destination_key = (installed_buffer.owner, publication.destination)
            current = installed_bases.get(
                destination_key, self._installed_bases.get(destination_key)
            )
            expected = (
                None
                if publication.base is None
                else (publication.base, publication.base_extent)
            )
            if current != expected:
                raise invalid_descriptor(
                    "KV installation base changed before publication"
                )
            publications_by_buffer[source] = publication
            publications_by_buffer[installed_buffer] = publication
            installed_bases[destination_key] = (
                publication.source,
                publication.published_extent,
            )

    def apply_publications(
        self,
        publications: Sequence[tuple[BufferId, KvTransfer]],
        installations: Sequence[tuple[BufferId, BufferId, KvTransfer]],
    ) -> None:
        """Apply a preflighted update without repeating fallible validation.

        The caller must call ``validate_publications`` before committing any
        owner and must not mutate this directory between preflight and
        application.
        """
        for buffer, publication in publications:
            self._publications[buffer] = publication
            self._destination_bases[(buffer.owner, publication.destination)] = (
                publication.source,
                publication.published_extent,
            )

        # An installation is resident under both its source identity and
        # the local installed identity.
        for source, installed_buffer, publication in installations:
            self._publications[source] = publication
            self._publications[installed_buffer] = publication
            self._installed_bases[
                (installed_buffer.owner, publication.destination)
            ] = (
                publication.source,
                publication.published_extent,
            )

    def release_calls(
        self, releases: Sequence[tuple[RequestKey, CallId]]
    ) -> tuple[BufferId, ...]:
        """Forget resident publications produced by the given calls.

        Returns the removed buffer identities for the caller to release
        (``release_buffers``). Locator registration and physical retirement
        belong to the export directory and its transports. Imported
        references may have no local registration; removing their semantic
        record does not release a remote publisher's storage.
        """
        identities = {(key, call_id) for key, call_id in releases}
        publications_by_buffer = tuple(
            buffer
            for buffer in self._publications
            if (buffer.owner, buffer.producer_call_id) in identities
        )
        for buffer in publications_by_buffer:
            del self._publications[buffer]
        return publications_by_buffer

    def drop(self, request_id: int) -> None:
        """Forget every resident publication and lineage base of a request.

        Physical registrations are retired separately by their owners.
        """
        selected = tuple(
            buffer
            for buffer in self._publications
            if int(buffer.owner.request_id) == int(request_id)
        )
        for buffer in selected:
            del self._publications[buffer]
        for table in (self._destination_bases, self._installed_bases):
            for key in tuple(
                key
                for key in table
                if int(key[0].request_id) == int(request_id)
            ):
                del table[key]
