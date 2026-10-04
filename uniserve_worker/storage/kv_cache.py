"""Request unit assignment, cache publications, imports, and retirement.

``KVCacheManager`` wraps one ``PrefixCache`` unit pool on a worker rank and
decides when each physical unit interval may be rewritten or reused. Unit
allocation belongs to the engine scheduler; this manager validates the unit
ids it assigns and owns the request block tables (``BlockTables``) that model
calls index. A cache group's logical page occupies ``units_per_page`` units
that each hold the page's tokens of some of the group's layers, so an
interval of a page's tokens is retained on every unit of the page. Three
kinds of owner retain unit intervals, each as ``unit -> (token offset, token
count)``:

- Execution accesses (``CacheAccess``): the units a model call reads or
  writes, retained until the batch's completion signal succeeds.
- Publications (``CacheExport``): an immutable token interval exported
  through transports, retained until the buffer is released and every
  physical registration has retired.
- Imports (``CacheImports``): scheduler-assigned destination units that an
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
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from threading import RLock

import torch

from uniserve.math import ceil_div
from uniserve.runtime import PrefixCache
from uniserve_worker._uniserve_ipc import Completion
from uniserve_worker.errors import invalid_descriptor, resource_error
from uniserve_worker.protocol.identity import BufferId, CallId, RequestKey
from uniserve_worker.protocol.transfer import (
    KvGroupTransfer,
    KvTransfer,
    Locator,
    TensorTransfer,
)
from uniserve_worker.protocol.worker_info import KVCacheInfo
from uniserve_worker.storage.block_tables import (
    BlockTables,
    GroupShape,
    GroupTable,
)
from uniserve_worker.storage.cache_imports import CacheImport, CacheImports
from uniserve_worker.transport.exports import ExportLocations, release_exports
from uniserve_worker.transport.interface import Transport
from uniserve_worker.transport.publication import publish_tensor

__all__ = ["KVCacheManager"]


@dataclass(slots=True)
class CacheExport:
    """An immutable token interval retained by its physical publications.

    Ranges map each physical unit to its token offset and token count. A
    publication covers every layer's K/V for these ranges, so appends outside
    the interval remain independent even when they share its final page.

    Attributes:
        buffer: Buffer identity the publication is registered under.
        ranges: Retained interval per physical unit.
        retirements: One future per physical registration, attached through
            ``KVCacheManager.retain_publication``.
        released: Whether semantic ownership has been revoked. The entry
            leaves the manager only when it is released and every retirement
            has succeeded.
    """

    buffer: BufferId
    ranges: dict[int, tuple[int, int]]
    retirements: tuple[Completion, ...] = ()
    released: bool = False


@dataclass(eq=False, slots=True)
class CacheAccess:
    """Physical unit intervals retained by one computation's completion.

    Every model access that shares one ``completion`` future joins one
    access; ``requests`` names their request keys and ``ranges`` keeps one
    interval per unit.
    """

    completion: Completion
    requests: set[RequestKey]
    ranges: dict[int, tuple[int, int]]


@dataclass(frozen=True, slots=True)
class GroupAxis:
    """Where this rank's layers of one group lie in its transfer layer axis.

    A published group tensor spans every layer of the group across pipeline
    stages, ``total`` layers in global cache-layer order; this rank's layers
    are the consecutive run starting at ``offset``.
    """

    offset: int
    total: int


class KVCacheManager:
    """Coordinate request ownership around one numerical unit pool.

    Physical unit ``0`` is the padding sentinel and is never allocatable.
    """

    def __init__(
        self,
        cache: PrefixCache,
        *,
        info: KVCacheInfo,
        group_layers: Sequence[Sequence[int]] | None = None,
        import_capacity: int = 1,
        request_pool_size: int = 1,
        table_width: int | None = None,
        staging_depth: int = 1,
    ) -> None:
        """Validate the unit pool against ``info`` and create its owners.

        Args:
            cache: Numerical K/V unit pool; closed by ``close``.
            info: Advertised unit count and group page shapes and placement.
            group_layers: Global cache-layer ids of every layer of each group
                across all pipeline stages, in ascending order; defaults to
                this rank's layers, the complete group without pipelining.
            import_capacity: Maximum copy tasks ``CacheImports`` admits at
                once.
            request_pool_size: Request slots of the block tables.
            table_width: Most pages one slot's group table may hold; defaults
                to every non-sentinel unit.
            staging_depth: Depth of the block tables' host staging rings.

        Raises:
            ValueError: When the pool does not match ``info``, a group's
                layers are not one consecutive run of its transfer layers, or
                ``staging_depth`` is below 1.
            WorkerError: ``invalid_descriptor`` when a block-table dimension
                is below 1.
        """
        self.cache, self.info = cache, info
        groups = cache.groups
        if cache.num_units != info.num_units or len(groups) != len(info.groups):
            raise ValueError(
                "cache backing must match the advertised unit pool"
            )
        for group, advertised in zip(groups, info.groups, strict=True):
            if (
                group.page_tokens != advertised.page_tokens
                or group.units_per_page != advertised.units_per_page
                or group.num_kv_heads != advertised.num_kv_heads
                or group.head_dim != advertised.head_dim
                or len(group.layers) != len(advertised.layer_ids)
            ):
                raise ValueError(
                    "cache backing must match the advertised group shapes"
                )

        # Every group's layers read one transfer layer axis; this rank's run
        # must be consecutive and in order within it.
        complete = (
            [tuple(group.layer_ids) for group in info.groups]
            if group_layers is None
            else [tuple(int(layer) for layer in axis) for axis in group_layers]
        )
        if len(complete) != len(info.groups):
            raise ValueError("every cache group requires its transfer layers")
        self.axes: tuple[GroupAxis, ...] = ()
        for axis, group in zip(complete, info.groups, strict=True):
            local = tuple(group.layer_ids)
            if local[0] not in axis:
                raise ValueError(
                    "cache group layers are missing from their transfer axis"
                )
            offset = axis.index(local[0])
            if axis[offset : offset + len(local)] != local:
                raise ValueError(
                    "cache group layers must be one consecutive transfer run"
                )
            self.axes += (GroupAxis(offset, len(axis)),)

        self.compute_dtypes = tuple(
            cache.config.layers[group.layers[0]].compute_dtype
            for group in groups
        )
        self.shapes = tuple(
            GroupShape(group.page_tokens, group.units_per_page, group.window)
            for group in groups
        )
        # The most tokens a unit holds in any group bounds a whole-unit span.
        self._unit_tokens = max(group.page_tokens for group in groups)
        # Reuse normalized unit tuples after their bounds have been
        # established by the validation path.
        self._validated_units: dict[
            tuple[tuple[int, ...], bool], tuple[int, ...]
        ] = {}

        # Transport locations of committed exports, updated by the batch
        # commit; publication intervals retained under their buffers.
        self.exports: dict[BufferId, ExportLocations] = {}
        self._sources: dict[BufferId, CacheExport] = {}

        # Execution accesses by completion signal, indexed by unit.
        # ``_execution_completed`` runs as an observer on the thread
        # that resolves the completion, so both maps are mutated under
        # ``_execution_lock``.
        self._executions: dict[Completion, CacheAccess] = {}
        self._execution_units: dict[int, set[CacheAccess]] = {}
        self._execution_lock = RLock()

        self.block_tables = BlockTables(
            groups=self.shapes,
            request_pool_size=request_pool_size,
            width=max(1, self.info.num_units - 1)
            if table_width is None
            else table_width,
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

    @property
    def row_units(self) -> int:
        """Return the units one page of every cache group occupies."""
        return sum(shape.units_per_page for shape in self.shapes)

    @property
    def token_capacity(self) -> int:
        """Return the most tokens one request's tables can cover at once.

        Every group holds whole pages of the same tokens. Page sizes are
        powers of two, so whole pages of the largest size fill every group
        exactly; the allocatable units (all but the sentinel) bound how many
        such spans fit.
        """
        largest = max(shape.page_tokens for shape in self.shapes)
        span_units = sum(
            largest // shape.page_tokens * shape.units_per_page
            for shape in self.shapes
        )
        return (self.info.num_units - 1) // span_units * largest

    def page_units(self, tokens: int) -> int:
        """Return the units of whole pages covering ``tokens`` in every group.

        At least one page per group, even for zero tokens.
        """
        return sum(
            ceil_div(max(1, tokens), shape.page_tokens) * shape.units_per_page
            for shape in self.shapes
        )

    @contextmanager
    def startup_units(self, count: int):
        """Borrow bounded scratch units before scheduler admission.

        Yields the first ``count`` allocatable units, reset on entry and
        again on exit, after the current stream drains on CUDA. The startup
        caller serializes this lease with other preparation. Serving
        allocation authority remains with the scheduler; no request or
        publication is introduced.

        Raises:
            RuntimeError: When any cache interval is still retained.
            ValueError: When ``count`` is negative or exceeds the allocatable
                units.
        """
        if self.has_pending_accesses:
            raise RuntimeError("startup scratch requires an idle KV pool")
        if not 0 <= count < self.info.num_units:
            raise ValueError("startup scratch exceeds KV capacity")
        units = tuple(range(1, count + 1))
        self.zero_units(units)
        try:
            yield units
        finally:
            if self.cache.device.type == "cuda":
                torch.cuda.current_stream(self.cache.device).synchronize()
            self.zero_units(units)

    @property
    def has_pending_accesses(self) -> bool:
        """Whether any publication, execution access or import is retained."""
        return (
            bool(self._sources) or bool(self._executions) or bool(self.imports)
        )

    def retain_execution(
        self,
        request: RequestKey,
        table: GroupTable,
        *,
        length: int,
        completion: Completion,
    ) -> None:
        """Retain a model access until its device work completes.

        The access covers the table's tokens from its start page up to
        ``length``. Model calls are ordered by the runner; this retention
        keeps independent import streams and unit reuse
        (``require_reusable``, ``write_dependencies``) away from these ranges
        while a producer or consumer kernel may still run. An access that
        reaches no held token retains nothing, and an already resolved
        ``completion`` retains nothing and re-raises its failure or
        cancellation.

        Raises:
            WorkerError: ``invalid_descriptor`` when the table's units are
                invalid or ``length`` exceeds its pages.
        """
        start = table.start_page * table.shape.page_tokens
        if length <= start:
            return
        self.validate_units(table.units)
        ranges = table.spans(start, length - start)

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

            # Each unit keeps one interval: the hull of every span retained
            # on it by this access.
            for unit, offset, count in ranges:
                previous = execution.ranges.get(unit)
                if previous is not None:
                    end = max(previous[0] + previous[1], offset + count)
                    offset = min(previous[0], offset)
                    count = end - offset
                execution.ranges[unit] = (offset, count)
                self._execution_units.setdefault(unit, set()).add(execution)

        if register:
            completion.add_done_callback(self._execution_completed)

    def _execution_completed(self, completion: Completion) -> None:
        # A failed or cancelled completion leaves its access registered: its
        # ranges stay blocked, ``retirement_ready`` re-raises the failure for
        # its requests, and ``close`` refuses to proceed.
        with self._execution_lock:
            if not completion.succeeded():
                return
            execution = self._executions.pop(completion, None)
            if execution is None:
                return

            for unit in execution.ranges:
                uses = self._execution_units[unit]
                uses.remove(execution)
                if not uses:
                    del self._execution_units[unit]

    def _execution_dependencies(
        self, ranges: Sequence[tuple[int, int, int]]
    ) -> tuple[Completion, ...]:
        """Return completions of accesses that overlap the given spans."""
        with self._execution_lock:
            return tuple(
                {
                    execution.completion
                    for unit, offset, count in ranges
                    for execution in self._execution_units.get(unit, ())
                    if self._ranges_overlap(
                        execution.ranges, ((unit, offset, count),)
                    )
                }
            )

    def unit_spans(
        self, units: Iterable[int]
    ) -> tuple[tuple[int, int, int], ...]:
        """Span every token each unit can hold, in any group.

        A whole-unit reset or import conflicts with any retained interval of
        the unit, whichever group recorded it.
        """
        return tuple((unit, 0, self._unit_tokens) for unit in units)

    def require_reusable(self, ranges: Sequence[tuple[int, int, int]]) -> None:
        """Authorize unit initialization or stream import before submission.

        ``ranges`` holds ``(unit, token offset, token count)`` spans. Applies
        ``require_writable`` and also rejects spans that an execution access
        still retains, including one whose completion failed or was
        cancelled. ``zero_units``, ``recycle_units`` and
        ``CacheImports.reserve`` call this before handing units to a writer.

        Raises:
            WorkerError: A resource error when a span overlaps a publication,
                an import destination or an execution access.
        """
        self._require_unretained(ranges)
        if self._execution_dependencies(ranges):
            raise resource_error(
                "KV interval still has an executing producer or consumer"
            )

    def reserve_publication(
        self, buffer: BufferId, ranges: Sequence[tuple[int, int, int]]
    ) -> CacheExport:
        """Retain the exact published spans before exporting any view.

        The caller attaches every registration's retirement signal with
        ``retain_publication`` and releases this reservation
        (``release_buffers``) if publication is abandoned before semantic
        visibility.

        Raises:
            WorkerError: ``invalid_descriptor`` when the spans are empty or
                ``buffer`` already has a retained interval.
        """
        self._reap_sources()
        if not ranges or buffer in self._sources:
            raise invalid_descriptor(
                "KV publication has an empty or already registered interval"
            )
        source = CacheExport(
            buffer,
            {unit: (offset, count) for unit, offset, count in ranges},
        )
        self._sources[source.buffer] = source
        return source

    def retain_publication(
        self, source: CacheExport, retirement: Completion
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

        # Free retires a publication, not the request's resident KV units.
        # Unrelated products can share that request while later computation
        # still reads its prefix. Only request retirement waits for all such
        # call kinds; physical unit reuse separately checks their ranges.
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
        self, ranges: Sequence[tuple[int, int, int]]
    ) -> tuple[Completion, ...]:
        """Return the completions that must succeed before writing spans.

        ``ranges`` holds ``(unit, token offset, token count)`` spans. These
        are the completions of overlapping execution accesses and the
        retirements of overlapping import destinations and publications.
        """
        if not self.has_pending_accesses:
            return ()
        self._reap_sources()
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
        self, table: GroupTable, *, start: int, length: int
    ) -> None:
        """Authorize a table interval before staging a kernel that writes it.

        Device-indexed attention kernels borrow raw cache views. Their caller
        must validate the scheduler's write interval here before dispatch; no
        device-to-host read of per-token addresses is needed in the kernel
        path. When no publication or import is retained, this returns without
        validating the units.

        Raises:
            WorkerError: A resource error when the interval overlaps a
                publication or an import destination; ``invalid_descriptor``
                when the interval exceeds the table.
        """
        # The runner orders model accesses. Only independent imports and
        # published immutable ranges add write conflicts at this boundary.
        if not self._sources and not self.imports:
            return
        self._require_unretained(table.spans(start, length))

    def _require_unretained(
        self, ranges: Sequence[tuple[int, int, int]]
    ) -> None:
        """Reject spans overlapping a publication or import destination."""
        if not self._sources and not self.imports:
            return
        self._reap_sources()
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
        """Whether any ``right`` span meets ``left``'s interval on its unit.

        Intervals overlap only on the same unit and only when their token
        ranges intersect.
        """
        return any(
            (other := left.get(unit)) is not None
            and offset < other[0] + other[1]
            and other[0] < offset + count
            for unit, offset, count in right
        )

    def _reap_sources(self) -> None:
        # A failed or cancelled retirement keeps its interval, so
        # ``retirement_ready`` re-raises the failure and ``close`` refuses.
        for buffer, source in tuple(self._sources.items()):
            if source.released and all(
                completion.succeeded() for completion in source.retirements
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

    def validate_group(self, group: int) -> int:
        """Return ``group`` as an integer after checking its range.

        Raises:
            WorkerError: ``invalid_descriptor`` when ``group`` is outside the
                advertised groups.
        """
        value = int(group)
        if value < 0 or value >= len(self.shapes):
            raise invalid_descriptor(
                f"KV group {value} outside pool group count {len(self.shapes)}"
            )
        return value

    def validate_units(
        self, unit_ids: Iterable[int], *, allow_sentinel: bool = False
    ) -> tuple[int, ...]:
        """Validate physical unit ids and return them as a tuple of ints.

        Real (non-zero) units must be unique and in ``[1, num_units)``. The
        sentinel unit ``0`` is accepted, possibly repeated, only with
        ``allow_sentinel``. Accepted tuples are memoized; the checks depend
        only on the pool size fixed at construction.

        Raises:
            WorkerError: ``invalid_descriptor`` when any check fails.
        """
        # Resident scheduler tables already use immutable integer tuples.
        # Check their validated identity before normalizing every element.
        units = tuple(unit_ids)
        key = (units, bool(allow_sentinel))
        cached = self._validated_units.get(key)
        if cached is not None:
            return cached
        units = tuple(int(unit) for unit in units)
        key = (units, bool(allow_sentinel))

        # Unit 0 is the padding sentinel; uniqueness applies only to real
        # units.
        real_units = tuple(unit for unit in units if unit != 0)
        if len(set(real_units)) != len(real_units):
            raise invalid_descriptor("KV allocation repeats a physical unit")
        lower = 0 if allow_sentinel else 1
        if units and (min(units) < lower or max(units) >= self.info.num_units):
            raise invalid_descriptor(
                "KV allocation exceeds the fixed physical pool"
            )

        # Bound the memo by clearing it wholesale.
        if len(self._validated_units) >= 16_384:
            self._validated_units.clear()
        self._validated_units[key] = units
        return units

    def zero_units(self, unit_ids: Iterable[int]) -> None:
        """Reset every column and field of the selected physical units.

        Raises:
            WorkerError: ``invalid_descriptor`` when the units are invalid;
                a resource error from ``require_reusable`` when any selected
                unit is still retained.
        """
        units = self.validate_units(unit_ids)
        if not units:
            return
        self.require_reusable(self.unit_spans(units))
        self.cache.zero_units(units)

    def recycle_units(self, unit_ids: Iterable[int]) -> None:
        """Prepare newly allocated units for their new owner.

        Resets only the unit state a new writer reads
        (`PrefixCache.recycle_units`); stale values stay unread.

        Raises:
            WorkerError: ``invalid_descriptor`` when the units are invalid;
                a resource error from ``require_reusable`` when any selected
                unit is still retained.
        """
        units = self.validate_units(unit_ids)
        if not units:
            return
        self.require_reusable(self.unit_spans(units))
        self.cache.recycle_units(units)

    def _published_start(self, group: int, base: int, visible: int) -> int:
        """Return the first token a group carries in a publication.

        A full-attention group carries the whole suffix after ``base``; a
        sliding-window group only the history a reader of ``visible`` needs.
        """
        window = self.shapes[group].window
        return base if window is None else max(base, visible - window)

    def publish(
        self,
        *,
        request_pool_idx: int,
        visible_length: int,
        destination: str,
        buffer: BufferId,
        transports: Mapping[str, Transport],
        consumers: Sequence[int] = (),
    ) -> KvTransfer:
        """Export a visible KV extent of every group under its buffer identity.

        Publications to one ``(request, destination)`` form a chain: this one
        exports only tokens after ``base_extent``, the latest committed
        publication to that destination, and a sliding-window group only its
        window before ``visible_length``. An empty suffix exports no tensors.
        The exported spans are reserved before any view is exported. The
        returned ``KvTransfer`` becomes resident only when the batch commits
        it (``validate_publications`` then ``apply_publications``).

        `consumers` are the acknowledgment slots of the ranks that install it.

        Raises:
            WorkerError: ``invalid_descriptor`` when the slot has no installed
                table for a group, ``visible_length`` exceeds its allocated
                length or trails the destination base, a group interval
                reaches retired pages, or the reservation fails. A transport
                failure propagates after every exported locator and the
                reservation are released.
        """
        installed = self._destination_bases.get((buffer.owner, destination))
        base, base_extent = (None, 0) if installed is None else installed

        visible = int(visible_length)
        tables = tuple(
            self.block_tables.table(request_pool_idx, group)
            for group in range(len(self.shapes))
        )
        if visible > self.block_tables.allocated_length(request_pool_idx):
            raise invalid_descriptor(
                "KV publication exceeds its scheduler block table"
            )
        if visible < base_extent:
            raise invalid_descriptor(
                "KV publication destination is ahead of its source"
            )

        starts = tuple(
            self._published_start(group, base_extent, visible)
            for group in range(len(tables))
        )
        spans = tuple(
            table.spans(start, visible - start)
            for table, start in zip(tables, starts, strict=True)
        )
        source = (
            self.reserve_publication(
                buffer, tuple(span for group in spans for span in group)
            )
            if visible > base_extent
            else None
        )

        locators: list[Locator] = []
        groups: list[KvGroupTransfer] = []
        try:
            if source is not None:
                for group, (table, start) in enumerate(
                    zip(tables, starts, strict=True)
                ):
                    tensors = self._publish_group(
                        group,
                        table,
                        start,
                        visible,
                        source=source,
                        transports=transports,
                        consumers=consumers,
                        locators=locators,
                    )
                    groups.append(
                        KvGroupTransfer(
                            start=start,
                            page_tokens=table.shape.page_tokens,
                            tensors=tensors,
                        )
                    )
        except BaseException:
            for locator in locators:
                transports[locator.backend].release(locator)
            self.release_buffers((buffer,))
            raise

        # Every group of one pool shares a compute dtype; the first names it.
        publication = KvTransfer(
            groups=tuple(groups),
            source=buffer,
            destination=destination,
            base=base,
            base_extent=base_extent,
            published_extent=visible,
            compute_dtype=str(self.compute_dtypes[0]).removeprefix("torch."),
        )
        return publication

    def _publish_group(
        self,
        group: int,
        table: GroupTable,
        start: int,
        visible: int,
        *,
        source: CacheExport,
        transports: Mapping[str, Transport],
        consumers: Sequence[int],
        locators: list[Locator],
    ) -> tuple[TensorTransfer, ...]:
        """Export one group's tokens ``[start, visible)`` of every layer.

        Returns the group's key, value and, with FP8, scale transfers; appends
        every exported locator to ``locators``.
        """
        count = visible - start
        if not count:
            return ()
        advertised = self.info.groups[group]
        axis = self.axes[group]
        columns = self.cache.planes.columns
        page_tokens = table.shape.page_tokens
        encoded = self.info.dtype == "float8_e4m3fn"
        # Pages the carried tokens touch, as (absolute page, offset, count).
        pages = []
        position = start
        while position < visible:
            page, offset = divmod(position, page_tokens)
            length = min(visible - position, page_tokens - offset)
            pages.append((page, offset, length))
            position += length

        tensors = []
        for field in ("key", "value"):
            planes = self.cache.planes_of(group, field)
            locations: list[Locator] = []
            # The layers in unit ``row`` of every page are one consecutive
            # run of ``columns`` layers, exported as one tensor per
            # mechanism: the descriptor's locator count follows the group's
            # units per page and the rank's mechanisms, never the model's
            # depth, which keeps it within its byte bound.
            for row in range(table.shape.units_per_page):
                units = table.row(row)
                # Planes are [columns, units, page tokens, heads, dim]; each
                # page view is [tokens, columns, heads, dim]. Unencoded views
                # alias the live units: ``require_writable`` rejects writes
                # into the reserved interval until it is released and
                # retired.
                views = tuple(
                    planes[
                        :,
                        units[page - table.start_page],
                        offset : offset + length,
                    ].permute(1, 0, 2, 3)
                    for page, offset, length in pages
                )
                if encoded:
                    # Appending can enlarge a unit's scale and re-encode its
                    # prefix. Freeze exported bytes so an immutable
                    # publication survives later numerical updates.
                    views = (torch.cat(views, dim=0),)
                # The offset places this run inside the group's
                # [tokens, layers, KV heads, head dim] transfer at the run's
                # first layer and this rank's first KV head.
                exported = publish_tensor(
                    transports,
                    views,
                    retain=partial(self.retain_publication, source),
                    offset=(
                        0,
                        axis.offset + row * columns,
                        advertised.kv_head_offset,
                        0,
                    ),
                    consumers=consumers,
                )
                locations.extend(exported)
                locators.extend(exported)
            tensors.append(
                TensorTransfer(
                    shape=(
                        count,
                        axis.total,
                        advertised.total_kv_heads,
                        advertised.head_dim,
                    ),
                    locations=tuple(locations),
                )
            )

        # FP8 exports additionally carry one scale row per touched page.
        if encoded:
            locations = []
            for field_index, field in enumerate(("key", "value")):
                scales = self.cache.scales_of(field)
                for row in range(table.shape.units_per_page):
                    units = table.row(row)
                    # Scale planes are [columns, units, 1, 1, 1] with one scale
                    # per unit and column; published rows are [page, K/V,
                    # layers, head group], frozen like the values they
                    # encode. A rank's head group is its KV-head offset
                    # divided by its local KV-head count.
                    views = (
                        torch.cat(
                            tuple(
                                scales[
                                    :, units[page - table.start_page]
                                ].reshape(1, 1, -1, 1)
                                for page, _, _ in pages
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
                            axis.offset + row * columns,
                            advertised.kv_head_offset
                            // advertised.num_kv_heads,
                        ),
                    )
                    locations.extend(exported)
                    locators.extend(exported)
            tensors.append(
                TensorTransfer(
                    shape=(
                        len(pages),
                        2,
                        axis.total,
                        advertised.total_kv_heads // advertised.num_kv_heads,
                    ),
                    locations=tuple(locations),
                )
            )
        return tuple(tensors)

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
        visible_length: int,
        publication: KvTransfer | None = None,
    ) -> KvTransfer:
        """Verify that a request's allocation still covers a publication.

        ``buffer`` must belong to ``request_key``, the published extent must
        lie within both ``visible_length`` and the slot's allocated length,
        and the slot must have an installed table for every group.
        ``publication`` skips the directory lookup when the caller already
        holds it.

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
            or self.block_tables.allocated_length(request_pool_idx)
            < publication.published_extent
        ):
            raise invalid_descriptor(
                "KV conditioning allocation disagrees with its publication"
            )
        for group in range(len(self.shapes)):
            self.block_tables.table(request_pool_idx, group)
        return publication

    def _validate_install(self, publication: KvTransfer) -> None:
        """Check lineage and every group's transfer shape before access.

        A first installation into ``(request, destination)`` has no base and
        a zero base extent; a later one must name the currently installed
        base and extent. A publication carrying tensors must carry one entry
        per group, each ``[tokens, group layers, KV heads, head dim]`` over
        this worker's transfer axes and starting where this worker's group
        needs it.

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
        if not publication.groups:
            return
        if len(publication.groups) != len(self.shapes):
            raise invalid_descriptor(
                "KV transfer groups do not match destination groups"
            )
        for group, value in enumerate(publication.groups):
            advertised = self.info.groups[group]
            expected_start = self._published_start(
                group, publication.base_extent, publication.published_extent
            )
            carried = publication.published_extent - value.start
            if value.start != expected_start or (
                value.tensors
                and value.tensors[0].shape
                != (
                    carried,
                    self.axes[group].total,
                    advertised.total_kv_heads,
                    advertised.head_dim,
                )
            ):
                raise invalid_descriptor(
                    "KV transfer shape does not match destination layers"
                )

    def prepare_install(
        self,
        publication: KvTransfer,
        *,
        request_pool_idx: int,
        tables: Sequence[GroupTable],
        initialized_units: tuple[int, ...],
        transports: Mapping[str, Transport],
    ) -> CacheImport:
        """Reserve scheduler units and start their bounded physical import.

        ``tables`` is the destination's table of every group and
        ``initialized_units`` the new units the import resets before copying.
        Units that hold the installed base must stay in place and must not
        be reset. The copy runs in ``CacheImports``; ``install`` adopts it
        once complete.

        Raises:
            WorkerError: ``invalid_descriptor`` when lineage, shape, units or
                capacity are invalid, imports are closed, or the publication
                source already has a registered import; a resource error
                when a destination interval is still retained or the import
                lane is closed or has no capacity.
        """
        self._validate_install(publication)
        if len(tables) != len(self.shapes):
            raise invalid_descriptor(
                "KV import requires a destination table per group"
            )
        initialized = self.validate_units(initialized_units)
        held = set()
        for table in tables:
            held.update(self.validate_units(table.units))
            if (
                table.allocated_tokens < publication.published_extent
                or table.start_page * table.shape.page_tokens
                > publication.published_extent
            ):
                raise invalid_descriptor(
                    "KV import exceeds its scheduler block table"
                )
        if not set(initialized).issubset(held):
            raise invalid_descriptor(
                "KV import resets units outside its block tables"
            )

        if publication.base_extent:
            # Units of the pages that hold the installed base must stay
            # identical and must not be reset by this import.
            for group, table in enumerate(tables):
                installed = self.block_tables.table(request_pool_idx, group)
                page_tokens = table.shape.page_tokens
                first = max(table.start_page, installed.start_page)
                last = min(
                    -(-publication.base_extent // page_tokens),
                    table.end_page,
                    installed.end_page,
                )
                per_page = table.shape.units_per_page
                for page in range(first, last):
                    new = table.units[
                        (page - table.start_page) * per_page : (
                            page - table.start_page + 1
                        )
                        * per_page
                    ]
                    old = installed.units[
                        (page - installed.start_page) * per_page : (
                            page - installed.start_page + 1
                        )
                        * per_page
                    ]
                    if new != old or set(new).intersection(initialized):
                        raise invalid_descriptor(
                            "KV import would replace its installed base units"
                        )

        return self.imports.reserve(
            publication,
            request_pool_idx=request_pool_idx,
            tables=tuple(tables),
            initialized_units=initialized,
            transports=transports,
        )

    def install(
        self,
        *,
        installed_buffer: BufferId,
        write: CacheImport,
    ) -> KvTransfer:
        """Adopt a completed physical import under its source and base.

        The request's block tables must be unchanged since
        ``prepare_install``. The slot's verified length becomes the published
        extent; the returned ``KvTransfer`` becomes resident when the batch
        commits (``apply_publications``).

        Raises:
            RuntimeError: When the import has not completed.
            WorkerError: ``invalid_descriptor`` when identities, lineage or
                the block tables disagree, or the import was abandoned.
            Exception: The import's own failure, re-raised.
        """
        publication = write.publication
        if installed_buffer.owner != publication.source.owner:
            raise invalid_descriptor("installed KV buffer identity is invalid")

        self._validate_install(publication)
        request_pool_idx = write.request_pool_idx
        if (
            tuple(
                self.block_tables.table(request_pool_idx, group)
                for group in range(len(self.shapes))
            )
            != write.tables
            or self.block_tables.allocated_length(request_pool_idx)
            < publication.published_extent
        ):
            raise invalid_descriptor(
                "KV installation scheduler block table changed"
            )

        self.imports.adopt(write)
        self.block_tables.set_verified(
            torch.tensor(
                (request_pool_idx,), device=self.block_tables.unit_tables.device
            ),
            torch.tensor(
                (publication.published_extent,),
                device=self.block_tables.unit_tables.device,
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
