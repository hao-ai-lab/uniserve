"""Device-resident execution image of scheduler-owned KV unit tables.

The engine scheduler owns request-slot and KV-unit assignment and sends each
batch's per-group unit tables in `Batch.block_tables`. `execution.prepare`
validates the unit ids against `KVCacheManager` and installs the tables here,
before the batch's forward calls read them. `BlockTables` holds the device
copy that the decode input kernels (`model_executor._decode_inputs`) index by
request slot, plus a host mirror used for change detection and for host-side
lookups such as attention input construction (`model_executor.attention`).

A cache group's logical page occupies ``units_per_page`` units, and the units
at one position of every page form one numerical block table (see
`uniserve.runtime.prefix_cache`). A slot's table of one group therefore
installs as ``units_per_page`` device rows, one per numerical table, each
holding the slot's units of pages ``start_page..`` at that position.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from uniserve.runtime.device import fill_cpu_ints
from uniserve.runtime.resources import close_resources
from uniserve.runtime.tensor_buffers import TensorBuffers
from uniserve.tensors import BufferConfig
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol.identity import RequestKey
from uniserve_worker.storage.host_buffers import HostBuffers

__all__ = ["BlockTables", "GroupShape", "GroupTable"]


@dataclass(frozen=True, slots=True)
class GroupShape:
    """The page shape of one cache group, as block tables index it.

    Attributes:
        page_tokens: Tokens per logical page.
        units_per_page: Units one logical page occupies; the group owns this
            many consecutive numerical tables.
        window: History tokens any reader of the group needs, or ``None``
            for full attention.
    """

    page_tokens: int
    units_per_page: int
    window: int | None = None


@dataclass(frozen=True, slots=True)
class GroupTable:
    """One request slot's installed units of one cache group.

    ``units`` holds the units of logical pages ``start_page..``, page-major;
    ``allocated_tokens`` is the absolute token extent the table covers.
    """

    shape: GroupShape
    start_page: int
    units: tuple[int, ...]
    allocated_tokens: int

    @property
    def end_page(self) -> int:
        """Return the logical page after the last installed one."""
        return self.start_page + len(self.units) // self.shape.units_per_page

    def row(self, index: int) -> tuple[int, ...]:
        """Return the units at page position ``index`` of every page."""
        return self.units[index :: self.shape.units_per_page]

    def spans(
        self, start: int, length: int
    ) -> tuple[tuple[int, int, int], ...]:
        """Map an absolute token interval to ``(unit, offset, count)`` spans.

        Every unit of each page the interval touches receives the page's
        token offset and count, since each holds those tokens of its own
        layers.

        Raises:
            WorkerError: ``invalid_descriptor`` when the interval reaches a
                retired page or lies past the installed pages.
        """
        page_tokens = self.shape.page_tokens
        per_page = self.shape.units_per_page
        if (
            start < 0
            or length < 0
            or (length and start < self.start_page * page_tokens)
            or start + length > self.end_page * page_tokens
        ):
            raise invalid_descriptor("KV token interval exceeds its unit table")
        spans = []
        while length:
            page, offset = divmod(start, page_tokens)
            count = min(length, page_tokens - offset)
            first = (page - self.start_page) * per_page
            spans.extend(
                (unit, offset, count)
                for unit in self.units[first : first + per_page]
            )
            start += count
            length -= count
        return tuple(spans)


class BlockTables:
    """Own request slots, per-table unit rows, start pages and KV lengths.

    Slot zero is permanently reserved for padding and CUDA-graph rows. A live
    slot's group table is installed from the scheduler's complete value and
    cleared by `release` before the slot can be reused. Cleared and unused
    entries hold unit ``0``, the KV pool's padding sentinel.

    Device tensors (``rows = request_pool_size + 1``):

    - ``unit_tables``: int32 ``[tables, rows, width]``, the units of each
      numerical table from the slot's start page of its group on.
    - ``start_pages``: int32 ``[groups, rows]``, each slot's first installed
      logical page per group.
    - ``verified_lengths``: int32 ``[rows]``, tokens whose KV is valid. The
      worker shares this tensor with `DecodeState` as its verified cache
      lengths.
    - ``alloced_lens``: int32 ``[rows]``, tokens every installed group of
      the slot covers.
    - ``table_shapes``: int32 ``[tables, 3]``, each numerical table's
      ``(group, page_tokens, window)``, with window ``-1`` for full
      attention; constant after construction.

    The device staging tensors are single-buffered and reused by every
    `install` and `release`. Nothing but the order of the CUDA stream current
    at each call separates one use from the next.
    """

    def __init__(
        self,
        *,
        groups: Sequence[GroupShape],
        request_pool_size: int,
        width: int,
        device: torch.device | str,
        staging_depth: int = 1,
    ) -> None:
        """Allocate device unit tables and bounded host staging.

        ``width`` is the most pages one slot's group table may hold.
        ``staging_depth`` is the number of pinned host staging generations
        per staging tensor; an update blocks only when it reuses a generation
        whose previous host-to-device copy has not completed.

        Raises:
            WorkerError: ``invalid_descriptor`` when a dimension is below 1.
            ValueError: When ``staging_depth`` is below 1.
        """
        # Alternative-prefix slots (for example CFG branch prefixes) owned by
        # each exact request epoch; see `retain_prefix`.
        self._prefix_slots: dict[RequestKey, set[int]] = {}
        self.groups = tuple(groups)
        self.request_pool_size = int(request_pool_size)
        self.width = int(width)
        if (
            not self.groups
            or min(self.request_pool_size, self.width) < 1
            or any(
                min(group.page_tokens, group.units_per_page) < 1
                for group in self.groups
            )
        ):
            raise invalid_descriptor(
                "request-to-token pool dimensions are invalid"
            )

        # Numerical tables, group-major: group g owns tables
        # [first_table[g], first_table[g] + units_per_page).
        self.first_table = []
        count = 0
        for group in self.groups:
            self.first_table.append(count)
            count += group.units_per_page
        self.table_count = count

        # One installation can touch every (slot, table) pair; the staging
        # tensors are sized for that worst case.
        self._row_capacity = self.request_pool_size * self.table_count
        self._entry_capacity = self.request_pool_size * len(self.groups)
        buffer_configs = self.buffers(
            groups=self.groups,
            request_pool_size=self.request_pool_size,
            width=self.width,
        )
        tensors = TensorBuffers.allocate(buffer_configs, device=device).view(
            buffer_configs
        )
        for name in (
            "unit_tables",
            "start_pages",
            "verified_lengths",
            "alloced_lens",
        ):
            tensors[name].fill_(0)
        tensors["table_shapes"].copy_(
            torch.tensor(
                [
                    (
                        group,
                        shape.page_tokens,
                        -1 if shape.window is None else shape.window,
                    )
                    for group, shape in enumerate(self.groups)
                    for _ in range(shape.units_per_page)
                ],
                dtype=torch.int32,
            )
        )

        self.unit_tables = tensors["unit_tables"]
        self.start_pages = tensors["start_pages"]
        self.verified_lengths = tensors["verified_lengths"]
        self.alloced_lens = tensors["alloced_lens"]
        self.table_shapes = tensors["table_shapes"]
        self._row_staging = tensors["_row_staging"]
        self._index_staging = tensors["_index_staging"]
        self._value_staging = tensors["_value_staging"]

        # Host mirrors of the device tables drive change detection in install.
        self._host_tables: dict[tuple[int, int], GroupTable] = {}
        self._host_alloced_lens: dict[int, int] = {}

        # Generation-safe pinned rings retain CPU sources until asynchronous
        # copies into the device staging tensors have completed.
        self._row_host = HostBuffers(
            (self._row_capacity, self.width),
            dtype=torch.int32,
            depth=staging_depth,
            device=self.unit_tables.device,
        )
        self._index_host = HostBuffers(
            (5, max(self._row_capacity, self._entry_capacity)),
            dtype=torch.int64,
            depth=staging_depth,
            device=self.unit_tables.device,
        )
        self._value_host = HostBuffers(
            (2, self._entry_capacity),
            dtype=torch.int32,
            depth=staging_depth,
            device=self.unit_tables.device,
        )

    @staticmethod
    def buffers(
        *, groups: Sequence[GroupShape], request_pool_size: int, width: int
    ) -> dict[str, BufferConfig]:
        """Describe the device tensors `__init__` allocates.

        `__init__` allocates exactly these configurations, and startup memory
        accounting (`bootstrap.report`) sums them, so every device tensor the
        class owns belongs here. Row zero of every slot-indexed tensor is the
        padding slot.

        Raises:
            WorkerError: ``invalid_descriptor`` when a dimension is below 1.
        """
        tables = sum(group.units_per_page for group in groups)
        if not groups or min(request_pool_size, width, tables) < 1:
            raise invalid_descriptor(
                "request-to-token pool dimensions are invalid"
            )
        rows = request_pool_size + 1
        row_capacity = request_pool_size * tables
        entry_capacity = request_pool_size * len(groups)
        return {
            # [table, slot, column]: unit ids per numerical table and slot.
            "unit_tables": BufferConfig((tables, rows, width), torch.int32),
            # [group, slot]: first installed logical page per group.
            "start_pages": BufferConfig((len(groups), rows), torch.int32),
            # [slot]: verified and allocated token lengths per request slot.
            "verified_lengths": BufferConfig((rows,), torch.int32),
            "alloced_lens": BufferConfig((rows,), torch.int32),
            # [table, 3]: (group, page tokens, window or -1) per table.
            "table_shapes": BufferConfig((tables, 3), torch.int32),
            # Device staging targets for one full installation. In `install`,
            # row 0 of ``_index_staging`` holds the tables and row 1 the slots
            # of changed unit rows, rows 2 and 3 the groups and slots of
            # changed start pages, and row 4 the slots whose allocated length
            # changed; ``_value_staging`` holds those start pages (row 0) and
            # lengths (row 1). `release` stages its slots in row 1.
            "_row_staging": BufferConfig((row_capacity, width), torch.int32),
            "_index_staging": BufferConfig(
                (5, max(row_capacity, entry_capacity)), torch.int64
            ),
            "_value_staging": BufferConfig((2, entry_capacity), torch.int32),
        }

    def install(
        self,
        tables: Sequence[tuple[int, int, int, Sequence[int], int]],
    ) -> None:
        """Install scheduler unit tables and allocated lengths on the device.

        Each entry is ``(slot, group, start_page, units, allocated_tokens)``
        and replaces that slot's table of the group: ``units`` lists the
        units of pages ``start_page..`` page-major. Only table rows, start
        pages and lengths that differ from the host mirror are copied. On
        CUDA the device writes are enqueued asynchronously on the current
        stream; the host blocks only when a pinned staging generation is
        reused before its previous copy has completed.

        This method checks table-local bounds only: slots in
        ``[1, request_pool_size]``, groups in range, positive unique units
        forming whole pages that fit the width, ``allocated_tokens`` within
        the installed pages, and no repeated ``(slot, group)``. The caller
        validates unit ids against the physical pool
        (`KVCacheManager.validate_units`). All entries are validated before
        any staging or device write, so a rejected update changes nothing.

        Raises:
            WorkerError: ``invalid_descriptor`` when any entry fails
                validation.
        """
        if not tables:
            return
        accepted: dict[tuple[int, int], GroupTable] = {}
        for raw_slot, raw_group, raw_start, raw_units, raw_allocated in tables:
            slot, group = int(raw_slot), int(raw_group)
            if not 1 <= slot <= self.request_pool_size or not (
                0 <= group < len(self.groups)
            ):
                raise invalid_descriptor("scheduler block table is invalid")
            shape = self.groups[group]
            table = GroupTable(
                shape,
                int(raw_start),
                tuple(int(unit) for unit in raw_units),
                int(raw_allocated),
            )
            pages, remainder = divmod(len(table.units), shape.units_per_page)
            if (
                table.start_page < 0
                or remainder
                or pages > self.width
                or any(unit < 1 for unit in table.units)
                or len(set(table.units)) != len(table.units)
                or not 0 <= table.allocated_tokens
                or table.allocated_tokens > table.end_page * shape.page_tokens
                or (slot, group) in accepted
            ):
                raise invalid_descriptor("scheduler block table is invalid")
            accepted[(slot, group)] = table

        # Collect the changed numerical rows and start pages; a group table
        # changes all of its rows together.
        rows: list[tuple[int, ...]] = []
        row_tables: list[int] = []
        row_slots: list[int] = []
        start_groups: list[int] = []
        start_slots: list[int] = []
        start_values: list[int] = []
        for (slot, group), table in accepted.items():
            previous = self._host_tables.get((slot, group))
            if previous is None or previous.units != table.units:
                for position in range(table.shape.units_per_page):
                    rows.append(table.row(position))
                    row_tables.append(self.first_table[group] + position)
                    row_slots.append(slot)
            if previous is None or previous.start_page != table.start_page:
                start_groups.append(group)
                start_slots.append(slot)
                start_values.append(table.start_page)

        # A slot's allocated length is the extent every installed group
        # covers after this update.
        mirror = dict(self._host_tables)
        mirror.update(accepted)
        updated = {slot for slot, _ in accepted}
        lengths: dict[int, int] = {}
        for (slot, _), table in mirror.items():
            if slot in updated:
                lengths[slot] = min(
                    lengths.get(slot, table.allocated_tokens),
                    table.allocated_tokens,
                )
        allocated = {
            slot: length
            for slot, length in lengths.items()
            if self._host_alloced_lens.get(slot) != length
        }

        non_blocking = self.unit_tables.device.type == "cuda"
        if rows or start_values or allocated:
            # Every index set of one installation shares one pinned
            # generation, so a batch does not consume several ring slots.
            index_slot, index_host = self._index_host.acquire()
            value_slot, value_host = self._value_host.acquire()
            index_host.zero_()
            fill_cpu_ints(index_host[0, : len(rows)], row_tables)
            fill_cpu_ints(index_host[1, : len(rows)], row_slots)
            fill_cpu_ints(index_host[2, : len(start_values)], start_groups)
            fill_cpu_ints(index_host[3, : len(start_values)], start_slots)
            fill_cpu_ints(index_host[4, : len(allocated)], tuple(allocated))
            fill_cpu_ints(value_host[0, : len(start_values)], start_values)
            fill_cpu_ints(
                value_host[1, : len(allocated)], tuple(allocated.values())
            )
            self._index_staging.copy_(index_host, non_blocking=non_blocking)
            self._value_staging.copy_(value_host, non_blocking=non_blocking)
            self._index_host.record_copy(index_slot)
            self._value_host.record_copy(value_slot)

        if rows:
            row_slot, row_host = self._row_host.acquire()
            staged = row_host[: len(rows)]
            staged.zero_()
            for index, units in enumerate(rows):
                fill_cpu_ints(staged[index, : len(units)], units)
            self._row_staging[: len(rows)].copy_(
                staged, non_blocking=non_blocking
            )
            self._row_host.record_copy(row_slot)
            # Scatter the staged rows into unit_tables[table, slot, :]. A
            # staged row is zero past its units, which clears a longer
            # previous row.
            self.unit_tables[
                self._index_staging[0, : len(rows)],
                self._index_staging[1, : len(rows)],
            ] = self._row_staging[: len(rows)]

        if start_values:
            self.start_pages[
                self._index_staging[2, : len(start_values)],
                self._index_staging[3, : len(start_values)],
            ] = self._value_staging[0, : len(start_values)]

        if allocated:
            self.alloced_lens.index_copy_(
                0,
                self._index_staging[4, : len(allocated)],
                self._value_staging[1, : len(allocated)],
            )

        # Mirror the accepted installation on the host.
        self._host_tables.update(accepted)
        self._host_alloced_lens.update(lengths)

    def table(self, request_pool_idx: int, group_id: int) -> GroupTable:
        """Return the installed table of one request slot and cache group.

        Reads the host mirror; no device access.

        Raises:
            WorkerError: ``invalid_descriptor`` when no table is installed.
        """
        try:
            return self._host_tables[(int(request_pool_idx), int(group_id))]
        except KeyError:
            raise invalid_descriptor(
                "request slot has no installed block table"
            ) from None

    def allocated_length(self, request_pool_idx: int) -> int:
        """Return the tokens every installed group of a slot covers, or 0."""
        return self._host_alloced_lens.get(int(request_pool_idx), 0)

    def set_verified(self, slots: torch.Tensor, lengths: torch.Tensor) -> None:
        """Update verified cache lengths without changing unit tables.

        ``slots`` and ``lengths`` are 1-D and row aligned; slots are not
        range-checked here. Each length must lie in
        ``[0, alloced_lens[slot]]``. On CUDA the bound is checked with
        ``torch._assert_async``, which avoids a host synchronization but
        reports a violation as an asynchronous device assertion; on CPU a
        violation raises ``invalid_descriptor`` before any write.

        Raises:
            WorkerError: ``invalid_descriptor`` when the inputs are not row
                aligned, or on CPU when a length is out of bounds.
        """
        slots = slots.to(device=self.unit_tables.device, dtype=torch.int64)
        lengths = lengths.to(device=self.unit_tables.device, dtype=torch.int32)
        if slots.ndim != 1 or lengths.shape != slots.shape:
            raise invalid_descriptor(
                "verified-length update is not row aligned"
            )
        allocated = self.alloced_lens.index_select(0, slots)
        bounds = torch.all((lengths >= 0) & (lengths <= allocated))
        if bounds.device.type == "cuda":
            torch._assert_async(
                bounds, "verified length exceeds allocated KV capacity"
            )
        elif not bool(bounds):
            raise invalid_descriptor(
                "verified length exceeds allocated KV capacity"
            )
        self.verified_lengths.index_copy_(0, slots, lengths)

    def close(self) -> None:
        """Release pinned staging sources and clear host bookkeeping.

        Waits for outstanding host-to-device staging copies and drops the
        pinned sources; call it while the CUDA streams those copies ran on
        still exist. The device tensors are not freed here.
        """
        close_resources(
            self._row_host.close,
            self._index_host.close,
            self._value_host.close,
        )
        self._prefix_slots.clear()
        self._host_tables.clear()
        self._host_alloced_lens.clear()

    def retain_prefix(self, request_key: RequestKey, slot: int) -> None:
        """Record that a request epoch uses another slot as a prefix row.

        `execution.prepare` calls this for forward rows whose slot differs
        from the request's own slot, such as a CFG branch prefix. The slot is
        cleared by `release_prefixes` for the same `RequestKey`.
        """
        self._prefix_slots.setdefault(request_key, set()).add(int(slot))

    def release_prefixes(
        self, request_key: RequestKey, slots: Sequence[int] | None = None
    ) -> None:
        """Clear alternative-prefix slots and stop tracking them.

        With ``slots``, clears exactly those slots (tracked or not); without
        it, clears every slot recorded for ``request_key``. The key's entry is
        dropped once it tracks no slot.

        Raises:
            WorkerError: ``invalid_descriptor`` when a slot is outside
                ``[1, request_pool_size]``; nothing is cleared then.
        """
        tracked = self._prefix_slots.get(request_key, set())
        selected = tuple(tracked) if slots is None else tuple(slots)
        self.release(selected)
        tracked.difference_update(selected)
        if not tracked:
            self._prefix_slots.pop(request_key, None)

    def release(self, slots: Sequence[int]) -> None:
        """Clear selected request slots for reuse by the scheduler.

        Zeroes the slots' unit rows in every table together with their start
        pages and verified and allocated lengths, and drops their host
        mirror entries. Duplicates are ignored. Alternative-prefix tracking
        is unchanged; see `release_prefixes`.

        Raises:
            WorkerError: ``invalid_descriptor`` when a slot is outside
                ``[1, request_pool_size]``.
        """
        values = tuple(dict.fromkeys(int(slot) for slot in slots))
        if not values:
            return
        if any(slot < 1 or slot > self.request_pool_size for slot in values):
            raise invalid_descriptor(
                "released request slot is outside capacity"
            )

        count = len(values)
        slot, host = self._index_host.acquire()
        fill_cpu_ints(host[1, :count], values)
        indices = self._index_staging[1, :count]
        indices.copy_(
            host[1, :count], non_blocking=self.unit_tables.device.type == "cuda"
        )
        self._index_host.record_copy(slot)

        self.unit_tables.index_fill_(1, indices, 0)
        self.start_pages.index_fill_(1, indices, 0)
        self.verified_lengths.index_fill_(0, indices, 0)
        self.alloced_lens.index_fill_(0, indices, 0)

        selected = set(values)
        for identity in tuple(self._host_tables):
            if identity[0] in selected:
                del self._host_tables[identity]
        for slot in selected:
            self._host_alloced_lens.pop(slot, None)
