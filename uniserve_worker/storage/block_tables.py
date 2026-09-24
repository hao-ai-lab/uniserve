"""Device-resident execution image of scheduler-owned block tables.

The engine scheduler owns request-slot and KV-page assignment and sends each
batch's block tables in `Batch.block_tables`. `execution.prepare` validates
the page ids against `KVCacheManager` and installs the tables here, before the
batch's forward calls read them. `BlockTables` holds the device copy that
the decode input kernels (`model_executor._decode_inputs`) index by request
slot, plus a host mirror used for change detection and for host-side lookups
such as attention input construction (`model_executor.attention`).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from uniserve.runtime.device import fill_cpu_ints
from uniserve.runtime.resources import close_resources
from uniserve.runtime.tensor_buffers import TensorBuffers
from uniserve.tensors import BufferConfig
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol.identity import RequestKey
from uniserve_worker.storage.host_buffers import HostBuffers

__all__ = ["BlockTables"]


class BlockTables:
    """Own request slots, group page tables, and physical KV lengths.

    Slot zero is permanently reserved for padding and CUDA-graph rows. A live
    slot is installed from the scheduler's complete block-table value and is
    cleared by `release` before it can be reused. Cleared and unused
    page-table entries hold page ``0``, the KV pool's padding sentinel.

    Device tensors (``rows = request_pool_size + 1``):

    - ``page_tables``: int32 ``[group_count, rows, max_blocks_per_request]``.
    - ``verified_lengths``: int32 ``[rows]``, tokens whose KV is valid. The
      worker shares this tensor with `DecodeState` as its verified cache
      lengths.
    - ``alloced_lens``: int32 ``[rows]``, tokens the installed pages can hold.

    The device staging tensors are single-buffered and reused by every
    `install` and `release`. Nothing but the order of the CUDA stream current
    at each call separates one use from the next.
    """

    def __init__(
        self,
        *,
        group_count: int,
        request_pool_size: int,
        max_blocks_per_request: int,
        block_size: int,
        device: torch.device | str,
        staging_depth: int = 1,
    ) -> None:
        """Allocate device block tables and bounded host staging.

        ``staging_depth`` is the number of pinned host staging generations per
        staging tensor; an update blocks only when it reuses a generation
        whose previous host-to-device copy has not completed.

        Raises:
            WorkerError: ``invalid_descriptor`` when any dimension is below 1.
            ValueError: When ``staging_depth`` is below 1.
        """
        # Alternative-prefix slots (for example CFG branch prefixes) owned by
        # each exact request epoch; see `retain_prefix`.
        self._prefix_slots: dict[RequestKey, set[int]] = {}
        self.group_count = int(group_count)
        self.request_pool_size = int(request_pool_size)
        self.max_blocks_per_request = int(max_blocks_per_request)
        self.block_size = int(block_size)
        if (
            min(
                self.group_count,
                self.request_pool_size,
                self.max_blocks_per_request,
                self.block_size,
            )
            < 1
        ):
            raise invalid_descriptor(
                "request-to-token pool dimensions are invalid"
            )

        # One installation can touch at most every (slot, group) pair; the
        # staging tensors are sized for that worst case.
        self._table_capacity = self.request_pool_size * self.group_count
        buffer_configs = self.buffers(
            group_count=self.group_count,
            request_pool_size=self.request_pool_size,
            max_blocks_per_request=self.max_blocks_per_request,
        )
        tensors = TensorBuffers.allocate(buffer_configs, device=device).view(
            buffer_configs
        )
        for name, value in {
            "page_tables": 0,
            "verified_lengths": 0,
            "alloced_lens": 0,
        }.items():
            tensors[name].fill_(value)

        self.page_tables = tensors["page_tables"]
        self.verified_lengths = tensors["verified_lengths"]
        self.alloced_lens = tensors["alloced_lens"]
        self._page_staging = tensors["_page_staging"]
        self._slot_staging = tensors["_slot_staging"]
        self._group_staging = tensors["_group_staging"]
        self._allocated_staging = tensors["_allocated_staging"]

        # Host mirrors of the device tables drive change detection in install.
        self._host_tables: dict[tuple[int, int], tuple[int, ...]] = {}
        self._host_alloced_lens: dict[int, int] = {}

        # Generation-safe pinned rings retain CPU sources until asynchronous
        # copies into all four device staging tensors have completed.
        self._page_host = HostBuffers(
            (self._table_capacity, self.max_blocks_per_request),
            dtype=torch.int32,
            depth=staging_depth,
            device=self.page_tables.device,
        )
        self._slot_host = HostBuffers(
            (2, self._table_capacity),
            dtype=torch.int64,
            depth=staging_depth,
            device=self.page_tables.device,
        )
        self._group_host = HostBuffers(
            self._table_capacity,
            dtype=torch.int64,
            depth=staging_depth,
            device=self.page_tables.device,
        )
        self._allocated_host = HostBuffers(
            self._table_capacity,
            dtype=torch.int32,
            depth=staging_depth,
            device=self.page_tables.device,
        )

    @staticmethod
    def buffers(
        *, group_count: int, request_pool_size: int, max_blocks_per_request: int
    ) -> dict[str, BufferConfig]:
        """Describe the device tensors `__init__` allocates.

        `__init__` allocates exactly these configurations, and startup memory
        accounting (`bootstrap.report`) sums them, so every device tensor the
        class owns belongs here. Row zero of every slot-indexed tensor is the
        padding slot.

        Raises:
            WorkerError: ``invalid_descriptor`` when any dimension is below 1.
        """
        if min(group_count, request_pool_size, max_blocks_per_request) < 1:
            raise invalid_descriptor(
                "request-to-token pool dimensions are invalid"
            )
        rows, tables = request_pool_size + 1, request_pool_size * group_count
        return {
            # [group, slot, block]: page ids per request slot and cache group.
            "page_tables": BufferConfig(
                (group_count, rows, max_blocks_per_request), torch.int32
            ),
            # [slot]: verified and allocated token lengths per request slot.
            "verified_lengths": BufferConfig((rows,), torch.int32),
            "alloced_lens": BufferConfig((rows,), torch.int32),
            # Device staging targets for one full installation batch. In
            # `install`, row 0 of ``_slot_staging`` holds slots of changed page
            # rows and row 1 holds slots whose allocated length changed;
            # `release` stages its slots in row 0.
            "_page_staging": BufferConfig(
                (tables, max_blocks_per_request), torch.int32
            ),
            "_slot_staging": BufferConfig((2, tables), torch.int64),
            "_group_staging": BufferConfig((tables,), torch.int64),
            "_allocated_staging": BufferConfig((tables,), torch.int32),
        }

    def install(
        self,
        tables: Sequence[tuple[int, int, Sequence[int], int]],
    ) -> None:
        """Install scheduler block tables and allocated lengths on the device.

        Each entry is ``(slot, group, pages, allocated_tokens)`` and replaces
        that slot's page row for the group. Only rows and lengths that differ
        from the host mirror are copied. On CUDA the device writes are
        enqueued asynchronously on the current stream; the host blocks only
        when a pinned staging generation is reused before its previous copy
        has completed.

        This method checks table-local bounds only: slots in
        ``[1, request_pool_size]``, groups in range, positive unique pages
        that fit the row, ``allocated_tokens`` within the pages' capacity, no
        repeated ``(slot, group)``, and one allocated length per slot across
        its groups. The caller validates page ids against the physical pool
        (`KVCacheManager.validate_pages`). All entries are validated before
        any staging or device write, so a rejected update changes nothing.

        Raises:
            WorkerError: ``invalid_descriptor`` when the update exceeds the
                staging capacity or any entry fails validation.
        """
        count = len(tables)
        if count == 0:
            return
        if count > self._table_capacity:
            raise invalid_descriptor(
                "block-table update exceeds staging capacity"
            )
        rows: list[tuple[int, ...]] = []
        slots: list[int] = []
        groups: list[int] = []
        allocated_by_slot: dict[int, int] = {}
        slot_allocations: dict[int, int] = {}
        identities: set[tuple[int, int]] = set()

        # Validate every entry and collect only the changes.
        for raw_slot, raw_group, raw_pages, raw_allocated in tables:
            slot = int(raw_slot)
            group = int(raw_group)
            pages = tuple(int(page) for page in raw_pages)
            allocated_tokens = int(raw_allocated)
            if (
                slot < 1
                or slot > self.request_pool_size
                or group < 0
                or group >= self.group_count
                or len(pages) > self.max_blocks_per_request
                or any(page < 1 for page in pages)
                or len(set(pages)) != len(pages)
                or allocated_tokens < 0
                or allocated_tokens > len(pages) * self.block_size
                or (slot, group) in identities
            ):
                raise invalid_descriptor("scheduler block table is invalid")
            # The allocated length is per slot; all of a slot's cache groups
            # must report the same value.
            previous = slot_allocations.setdefault(slot, allocated_tokens)
            if previous != allocated_tokens:
                raise invalid_descriptor(
                    "cache groups disagree on allocated length"
                )
            identities.add((slot, group))
            if self._host_tables.get((slot, group)) != pages:
                rows.append(pages)
                slots.append(slot)
                groups.append(group)
            if self._host_alloced_lens.get(slot) != allocated_tokens:
                allocated_by_slot[slot] = allocated_tokens

        changed_count = len(rows)
        allocated_slots = tuple(allocated_by_slot)
        allocated = tuple(allocated_by_slot.values())
        allocated_count = len(allocated_slots)
        non_blocking = self.page_tables.device.type == "cuda"
        if changed_count or allocated_count:
            # Page rows and allocated lengths have independent index sets, but
            # belong to one installation. Submit both from one pinned generation
            # so a batch does not consume two slots of the staging ring.
            slot_slot, slot_host = self._slot_host.acquire()
            slot_host.zero_()
            fill_cpu_ints(slot_host[0, :changed_count], slots)
            fill_cpu_ints(slot_host[1, :allocated_count], allocated_slots)
            self._slot_staging.copy_(slot_host, non_blocking=non_blocking)
            self._slot_host.record_copy(slot_slot)

        if changed_count:
            page_slot, page_host = self._page_host.acquire()
            group_slot, group_host = self._group_host.acquire()
            pages_host = page_host[:changed_count]
            pages_host.zero_()
            fill_cpu_ints(group_host[:changed_count], groups)
            for row, pages in enumerate(rows):
                fill_cpu_ints(pages_host[row, : len(pages)], pages)
            self._page_staging[:changed_count].copy_(
                pages_host, non_blocking=non_blocking
            )
            self._group_staging[:changed_count].copy_(
                group_host[:changed_count], non_blocking=non_blocking
            )
            self._page_host.record_copy(page_slot)
            self._group_host.record_copy(group_slot)
            # Scatter the staged rows into page_tables[group, slot, :]. A
            # staged row is zero past its page count, which clears any longer
            # previous row.
            self.page_tables[
                self._group_staging[:changed_count],
                self._slot_staging[0, :changed_count],
            ] = self._page_staging[:changed_count]

        if allocated_count:
            allocated_slot, allocated_host = self._allocated_host.acquire()
            fill_cpu_ints(allocated_host[:allocated_count], allocated)
            self._allocated_staging[:allocated_count].copy_(
                allocated_host[:allocated_count], non_blocking=non_blocking
            )
            self._allocated_host.record_copy(allocated_slot)
            self.alloced_lens.index_copy_(
                0,
                self._slot_staging[1, :allocated_count],
                self._allocated_staging[:allocated_count],
            )

        # Mirror the accepted installation on the host for change detection.
        for slot, group, pages, allocated_tokens in (
            (
                int(raw_slot),
                int(raw_group),
                tuple(int(page) for page in raw_pages),
                int(raw_allocated),
            )
            for raw_slot, raw_group, raw_pages, raw_allocated in tables
        ):
            self._host_tables[(slot, group)] = pages
            self._host_alloced_lens[slot] = allocated_tokens

    def pages(self, request_pool_idx: int, group_id: int) -> tuple[int, ...]:
        """Return the installed page ids for one request slot and cache group.

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
        """Return the installed token capacity of a slot, or 0 if none."""
        return self._host_alloced_lens.get(int(request_pool_idx), 0)

    def set_verified(self, slots: torch.Tensor, lengths: torch.Tensor) -> None:
        """Update verified cache lengths without changing page tables.

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
        slots = slots.to(device=self.page_tables.device, dtype=torch.int64)
        lengths = lengths.to(device=self.page_tables.device, dtype=torch.int32)
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
            self._page_host.close,
            self._slot_host.close,
            self._group_host.close,
            self._allocated_host.close,
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

        Zeroes the slots' page rows in every group together with their
        verified and allocated lengths, and drops their host mirror entries.
        Duplicates are ignored. Alternative-prefix tracking is unchanged; see
        `release_prefixes`.

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
        slot, host = self._slot_host.acquire()
        fill_cpu_ints(host[0, :count], values)
        indices = self._slot_staging[0, :count]
        indices.copy_(
            host[0, :count], non_blocking=self.page_tables.device.type == "cuda"
        )
        self._slot_host.record_copy(slot)

        self.page_tables.index_fill_(1, indices, 0)
        self.verified_lengths.index_fill_(0, indices, 0)
        self.alloced_lens.index_fill_(0, indices, 0)

        selected = set(values)
        for identity in tuple(self._host_tables):
            if identity[0] in selected:
                del self._host_tables[identity]
        for slot in selected:
            self._host_alloced_lens.pop(slot, None)
