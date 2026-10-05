"""KV backing tensors, numerical views, transfers, and block-table binding.

The engine scheduler assigns physical units. Native KVCacheManager retains
their token intervals through model execution, exports and imports, and
decides when writers may reuse them. BlockTables owns request assignments.
This module binds those owners to the same PrefixCache tensors that public
numerical computation uses.

A group's logical page occupies units_per_page physical units. Each holds
that page's tokens for a subset of layers, so an access retains its interval
on every unit. Incremental transfers preserve group intervals and their
installed bases while Python prepares tensor views and numerical copies.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

import torch

from uniserve.math import ceil_div
from uniserve.runtime import PrefixCache
from uniserve_worker._uniserve_ipc import Completion
from uniserve_worker._uniserve_ipc import KVCacheManager as NativeKVCacheManager
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol.identity import BufferId, CallId, RequestKey
from uniserve_worker.protocol.transfer import (
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
from uniserve_worker.storage.cache_imports import KVImport, KVImporter
from uniserve_worker.transport.exports import (
    ExportLocations,
    export_tensor,
    release_exports,
)
from uniserve_worker.transport.interface import Transport

__all__ = ["KVCacheManager"]


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
        host_buffer_depth: int = 1,
    ) -> None:
        """Validate the unit pool against ``info`` and create its owners.

        Args:
            cache: Numerical K/V unit pool; closed by ``close``.
            info: Advertised unit count and group page shapes and placement.
            group_layers: Global cache-layer ids of every layer of each group
                across all pipeline stages, in ascending order; defaults to
                this rank's layers, the complete group without pipelining.
            import_capacity: Maximum copy tasks ``KVImporter`` admits at
                once.
            request_pool_size: Request slots of the block tables.
            table_width: Most pages one slot's group table may hold; defaults
                to every non-sentinel unit.
            host_buffer_depth: Depth of the block tables' host buffer rings.

        Raises:
            ValueError: When the pool does not match ``info``, a group's
                layers are not one consecutive run of its transfer layers, or
                ``host_buffer_depth`` is below 1.
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
        # commit; export intervals retained under their buffers.
        self.exports: dict[BufferId, ExportLocations] = {}

        self.block_tables = BlockTables(
            groups=self.shapes,
            num_units=self.info.num_units,
            request_pool_size=request_pool_size,
            width=max(1, self.info.num_units - 1)
            if table_width is None
            else table_width,
            device=cache.device,
            host_buffer_depth=host_buffer_depth,
        )

        self._manager = NativeKVCacheManager(
            self.block_tables._tables,
            str(self.compute_dtypes[0]).removeprefix("torch."),
            self._export_group,
        )
        self.imports = KVImporter(self, capacity=import_capacity)

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
        export is introduced.

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
        """Whether model execution or transfer still retains cache units."""
        return self._manager.has_pending_accesses

    def retain_execution(
        self,
        request: RequestKey,
        table: GroupTable,
        *,
        length: int,
        completion: Completion,
    ) -> None:
        """Retain model accesses until their physical completion succeeds.

        The runner orders model calls. This retention excludes independent
        import streams and unit reuse while a kernel may still run.
        """
        start = table.start_page * table.shape.page_tokens
        if length <= start:
            return
        self.validate_units(table.units)
        self._manager.retain_execution(
            request, table.spans(start, length - start), completion
        )

    def unit_spans(
        self, units: Iterable[int]
    ) -> tuple[tuple[int, int, int], ...]:
        """Cover every token a unit can hold, regardless of its cache group."""
        return tuple((unit, 0, self._unit_tokens) for unit in units)

    def require_reusable(self, ranges: Sequence[tuple[int, int, int]]) -> None:
        """Reject reuse while any model or transfer still accesses the spans."""
        self._manager.require_reusable(ranges)

    def reserve_export(
        self, buffer: BufferId, ranges: Sequence[tuple[int, int, int]]
    ) -> BufferId:
        """Retain an immutable interval before exporting its tensor views.

        The buffer identifies the reservation. Attach each transport's
        retirement through retain_export and revoke it through
        release_buffers even if export fails.
        """
        self._manager.reserve_export(buffer, ranges)
        return buffer

    def retain_export(self, buffer: BufferId, retirement: Completion) -> None:
        """Retain the exported interval until its transport retires."""
        self._manager.retain_export(buffer, retirement)

    def release_buffers(self, buffers: Iterable[BufferId]) -> None:
        """Revoke new readers and preserve every pending physical access."""
        selected = tuple(buffers)
        release_exports(self.exports, selected)
        self.imports.release(selected)
        self._manager.release_exports(selected)

    def retirement_ready(
        self,
        *,
        buffers: Iterable[BufferId] = (),
        requests: Iterable[RequestKey] = (),
        retained: frozenset[BufferId] = frozenset(),
    ) -> bool:
        """Report physical retirement or raise a selected owner's failure.

        Buffer release waits for its transfers. Request retirement also
        waits for that request's model accesses and excludes retained buffers.
        """
        return self._manager.retirement_ready(buffers, requests, retained)

    def write_dependencies(
        self, ranges: Sequence[tuple[int, int, int]]
    ) -> tuple[Completion, ...]:
        """Return physical completions of every overlapping access."""
        return self._manager.write_dependencies(ranges)

    def require_writable(
        self, table: GroupTable, *, start: int, length: int
    ) -> None:
        """Reject writes overlapping independent imports or immutable exports.

        The runner orders model accesses on its stream. Transfers use other
        streams or processes, so their ranges must retire before a write.
        """
        if self._manager.has_transfers:
            self._manager.require_writable(table.spans(start, length))

    def close(self) -> None:
        """Drain imports and exports before releasing the backing cache.

        Unknown physical completion keeps the cache open and raises a
        resource error. The caller must retain it through worker shutdown.
        """
        self.imports.stop()
        self.release_buffers(self._manager.exported_buffers())
        self.imports.require_retired()
        self._manager.require_retired()

        self.exports.clear()
        self.block_tables.close()
        self._manager.clear_resident()
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

    def export(
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

        Exports to one ``(request, destination)`` form a chain: this one
        exports only tokens after ``base_extent``, the latest committed
        export to that destination, and a sliding-window group only its
        window before ``visible_length``. An empty suffix exports no tensors.
        The exported spans are reserved before any view is exported. The
        returned ``KvTransfer`` becomes resident only when the batch commits
        it (``validate_exports`` then ``apply_exports``).

        `consumers` are the acknowledgment slots of the ranks that install it.

        Raises:
            WorkerError: ``invalid_descriptor`` when the slot has no installed
                table for a group, ``visible_length`` exceeds its allocated
                length or trails the destination base, a group interval
                reaches retired pages, or the reservation fails. A transport
                failure propagates after every exported locator and the
                reservation are released.
        """
        return self._manager.export(
            request_pool_idx=request_pool_idx,
            visible_length=visible_length,
            destination=destination,
            buffer=buffer,
            transports=transports,
            consumers=consumers,
        )

    def _export_group(
        self,
        group: int,
        table: GroupTable,
        start: int,
        visible: int,
        *,
        source: BufferId,
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
                    # export survives later numerical updates.
                    views = (torch.cat(views, dim=0),)
                # The offset places this run inside the group's
                # [tokens, layers, KV heads, head dim] transfer at the run's
                # first layer and this rank's first KV head.
                exported = export_tensor(
                    transports,
                    views,
                    retain=partial(self.retain_export, source),
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
                    exported = export_tensor(
                        transports,
                        views,
                        retain=partial(self.retain_export, source),
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

    def get_export(self, buffer: BufferId) -> KvTransfer:
        """Return the resident KV export registered for ``buffer``.

        Raises:
            WorkerError: ``invalid_descriptor`` when none is resident.
        """
        resident = self.resident(buffer)
        if resident is None:
            raise invalid_descriptor("KV export buffer is not resident")
        return resident

    def resident(self, buffer: BufferId) -> KvTransfer | None:
        """Look up a resident KV export; absence is not an error."""
        return self._manager.resident(buffer)

    def validate_conditioning(
        self,
        request_key: RequestKey,
        buffer: BufferId,
        *,
        request_pool_idx: int,
        visible_length: int,
    ) -> KvTransfer:
        """Verify that a request's allocation still covers an export.

        ``buffer`` must belong to ``request_key``, the published extent must
        lie within both ``visible_length`` and the slot's allocated length,
        and the slot must have an installed table for every group.

        Returns:
            The export.

        Raises:
            WorkerError: ``invalid_descriptor`` when any check fails or the
                export is not resident.
        """
        export = self.get_export(buffer)
        if buffer.owner != request_key:
            raise invalid_descriptor(
                "KV conditioning buffer belongs to another request"
            )
        if (
            int(visible_length) < export.exported_extent
            or self.block_tables.allocated_length(request_pool_idx)
            < export.exported_extent
        ):
            raise invalid_descriptor(
                "KV conditioning allocation disagrees with its export"
            )
        for group in range(len(self.shapes)):
            self.block_tables.table(request_pool_idx, group)
        return export

    def prepare_install(
        self,
        export: KvTransfer,
        *,
        request_pool_idx: int,
        tables: Sequence[GroupTable],
        initialized_units: tuple[int, ...],
        transports: Mapping[str, Transport],
    ) -> KVImport:
        """Reserve scheduler units and start their bounded physical import.

        ``tables`` is the destination's table of every group and
        ``initialized_units`` the new units the import resets before copying.
        Units that hold the installed base must stay in place and must not
        be reset. The copy runs in ``KVImporter``; ``install`` adopts it
        once complete.

        Raises:
            WorkerError: ``invalid_descriptor`` when the base, shape, units or
                capacity are invalid, imports are closed, or the export
                source already has a registered import; a resource error
                when a destination interval is still retained or the import
                lane is closed or has no capacity.
        """
        return self.imports.reserve(
            export,
            request_pool_idx=request_pool_idx,
            tables=tuple(tables),
            initialized_units=initialized_units,
            transports=transports,
        )

    def install(
        self,
        *,
        installed_buffer: BufferId,
        write: KVImport,
    ) -> KvTransfer:
        """Adopt a completed physical import under its source and base.

        The request's block tables must be unchanged since
        ``prepare_install``. The slot's verified length becomes the imported
        extent; the returned ``KvTransfer`` becomes resident when the batch
        commits (``apply_exports``).

        Raises:
            RuntimeError: When the import has not completed.
            WorkerError: ``invalid_descriptor`` when the source, base or
                the block tables disagree, or the import was abandoned.
            Exception: The import's own failure, re-raised.
        """
        return self.imports.adopt(write, installed_buffer)

    def _set_imported_length(self, slot: int, extent: int) -> None:
        """Copy the adopted extent into the numerical block tables."""
        device = self.block_tables.unit_tables.device
        self.block_tables.set_verified(
            torch.tensor((slot,), device=device),
            torch.tensor((extent,), device=device),
        )

    def validate_exports(
        self,
        exports: Sequence[tuple[BufferId, KvTransfer]],
        installations: Sequence[tuple[BufferId, BufferId, KvTransfer]],
    ) -> None:
        """Validate touched KV versions before any group resource is visible.

        Checks the batch's exports and installations without changing
        the directory. Several entries for one ``(request, destination)``
        must chain in order.

        Raises:
            WorkerError: ``invalid_descriptor`` when a buffer identity does
                not match its transfer, a resident buffer names another
                transfer, or a base is not the current one.
        """
        self._manager.validate_exports(exports, installations)

    def apply_exports(
        self,
        exports: Sequence[tuple[BufferId, KvTransfer]],
        installations: Sequence[tuple[BufferId, BufferId, KvTransfer]],
    ) -> None:
        """Apply a preflighted update without repeating fallible validation.

        The caller must call ``validate_exports`` before committing any
        owner and must not mutate this directory between preflight and
        application.
        """
        self._manager.apply_exports(exports, installations)

    def release_calls(
        self, releases: Sequence[tuple[RequestKey, CallId]]
    ) -> tuple[BufferId, ...]:
        """Forget resident exports produced by the given calls.

        Returns the removed buffer identities for the caller to release
        (``release_buffers``). Locator registration and physical retirement
        belong to the export directory and its transports. Imported
        references may have no local registration; removing their semantic
        record does not release a remote publisher's storage.
        """
        return self._manager.release_calls(releases)

    def drop(self, request_id: int) -> None:
        """Forget every resident export and lineage base of a request.

        Physical registrations are retired separately by their owners.
        """
        self._manager.drop_request(request_id)
