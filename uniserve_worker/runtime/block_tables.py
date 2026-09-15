"""Device-resident execution image of scheduler-owned block tables."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from uniserve.runtime.device import fill_cpu_ints
from uniserve.runtime.resources import close_resources
from uniserve.runtime.tensor_buffers import TensorBuffers
from uniserve.tensors import BufferConfig
from uniserve_worker.runtime.staging_buffers import StagingBuffers

from ..foundation.errors import invalid_descriptor
from ..protocol.batch import RequestKey

__all__ = ["BlockTables"]


class BlockTables:
    """Own request slots, group page tables, and physical KV lengths.

    Slot zero is permanently reserved for padding and CUDA-graph rows. A live
    slot is installed from the scheduler's complete block-table value and is
    cleared before it can be reused.
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
        """Allocate device block tables and bounded host staging for atomic table updates."""

        # Slot zero is included in every device row allocation but remains
        # reserved for padding and graph replay rather than scheduler requests.
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
            raise invalid_descriptor("request-to-token pool geometry is invalid")

        self._table_capacity = self.request_pool_size * self.group_count
        buffer_configs = self.buffers(
            group_count=self.group_count,
            request_pool_size=self.request_pool_size,
            max_blocks_per_request=self.max_blocks_per_request,
        )
        tensors = TensorBuffers.allocate(buffer_configs, device=device).view(buffer_configs)
        for name, value in {"page_tables": 0, "verified_lengths": 0, "alloced_lens": 0}.items():
            tensors[name].fill_(value)
        self.page_tables = tensors["page_tables"]
        self.verified_lengths = tensors["verified_lengths"]
        self.alloced_lens = tensors["alloced_lens"]
        self._page_staging = tensors["_page_staging"]
        self._slot_staging = tensors["_slot_staging"]
        self._group_staging = tensors["_group_staging"]
        self._allocated_staging = tensors["_allocated_staging"]
        self._host_tables: dict[tuple[int, int], tuple[int, ...]] = {}
        self._host_alloced_lens: dict[int, int] = {}

        # Generation-safe pinned rings retain CPU sources until asynchronous
        # copies into all four device staging tensors have completed.
        self._page_host = StagingBuffers(
            (self._table_capacity, self.max_blocks_per_request),
            dtype=torch.int32,
            depth=staging_depth,
            device=self.page_tables.device,
        )
        self._slot_host = StagingBuffers(
            (2, self._table_capacity),
            dtype=torch.int64,
            depth=staging_depth,
            device=self.page_tables.device,
        )
        self._group_host = StagingBuffers(
            self._table_capacity,
            dtype=torch.int64,
            depth=staging_depth,
            device=self.page_tables.device,
        )
        self._allocated_host = StagingBuffers(
            self._table_capacity,
            dtype=torch.int32,
            depth=staging_depth,
            device=self.page_tables.device,
        )

    @staticmethod
    def buffers(
        *, group_count: int, request_pool_size: int, max_blocks_per_request: int
    ) -> dict[str, BufferConfig]:
        """Describe page tables and the full request/group installation workspace."""

        if min(group_count, request_pool_size, max_blocks_per_request) < 1:
            raise invalid_descriptor("request-to-token pool geometry is invalid")
        rows, tables = request_pool_size + 1, request_pool_size * group_count
        return {
            "page_tables": BufferConfig((group_count, rows, max_blocks_per_request), torch.int32),
            "verified_lengths": BufferConfig((rows,), torch.int32),
            "alloced_lens": BufferConfig((rows,), torch.int32),
            "_page_staging": BufferConfig((tables, max_blocks_per_request), torch.int32),
            "_slot_staging": BufferConfig((2, tables), torch.int64),
            "_group_staging": BufferConfig((tables,), torch.int64),
            "_allocated_staging": BufferConfig((tables,), torch.int32),
        }

    def install(
        self,
        tables: Sequence[tuple[int, int, Sequence[int], int]],
    ) -> None:
        """Atomically install validated request block tables and allocated lengths on the device."""

        count = len(tables)
        if count == 0:
            return
        if count > self._table_capacity:
            raise invalid_descriptor("block-table update exceeds staging capacity")
        rows: list[tuple[int, ...]] = []
        slots: list[int] = []
        groups: list[int] = []
        allocated_by_slot: dict[int, int] = {}
        slot_allocations: dict[int, int] = {}
        identities: set[tuple[int, int]] = set()
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
            previous = slot_allocations.setdefault(slot, allocated_tokens)
            if previous != allocated_tokens:
                raise invalid_descriptor("cache groups disagree on allocated length")
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
            self._page_staging[:changed_count].copy_(pages_host, non_blocking=non_blocking)
            self._group_staging[:changed_count].copy_(
                group_host[:changed_count], non_blocking=non_blocking
            )
            self._page_host.record_copy(page_slot)
            self._group_host.record_copy(group_slot)
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
        """Resolve the installed cache-page table for one request slot and cache group."""

        try:
            return self._host_tables[(int(request_pool_idx), int(group_id))]
        except KeyError:
            raise invalid_descriptor("request slot has no installed block table") from None

    def allocated_length(self, request_pool_idx: int) -> int:
        """Expose the token capacity currently installed for a request slot."""

        return self._host_alloced_lens.get(int(request_pool_idx), 0)

    def set_verified(self, slots: torch.Tensor, lengths: torch.Tensor) -> None:
        """Update verified cache lengths for selected request slots without changing page tables."""

        slots = slots.to(device=self.page_tables.device, dtype=torch.int64)
        lengths = lengths.to(device=self.page_tables.device, dtype=torch.int32)
        if slots.ndim != 1 or lengths.shape != slots.shape:
            raise invalid_descriptor("verified-length update is not row aligned")
        allocated = self.alloced_lens.index_select(0, slots)
        bounds = torch.all((lengths >= 0) & (lengths <= allocated))
        if bounds.device.type == "cuda":
            torch._assert_async(bounds, "verified length exceeds allocated KV capacity")
        elif not bool(bounds):
            raise invalid_descriptor("verified length exceeds allocated KV capacity")
        self.verified_lengths.index_copy_(0, slots, lengths)

    def close(self) -> None:
        """Retire pinned page-table sources before their borrowed streams are destroyed."""

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
        """Associate an alternative CFG prefix row with its exact request epoch."""

        self._prefix_slots.setdefault(request_key, set()).add(int(slot))

    def release_prefixes(self, request_key: RequestKey, slots: Sequence[int] | None = None) -> None:
        """Release selected alternative rows, or every row owned by a retiring epoch."""

        tracked = self._prefix_slots.get(request_key, set())
        selected = tuple(tracked) if slots is None else tuple(slots)
        self.release(selected)
        tracked.difference_update(selected)
        if not tracked:
            self._prefix_slots.pop(request_key, None)

    def release(self, slots: Sequence[int]) -> None:
        """Clear selected request slots and return them to the scheduler-owned free state."""

        values = tuple(dict.fromkeys(int(slot) for slot in slots))
        if not values:
            return
        if any(slot < 1 or slot > self.request_pool_size for slot in values):
            raise invalid_descriptor("released request slot is outside capacity")
        count = len(values)
        slot, host = self._slot_host.acquire()
        fill_cpu_ints(host[0, :count], values)
        indices = self._slot_staging[0, :count]
        indices.copy_(host[0, :count], non_blocking=self.page_tables.device.type == "cuda")
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


def page_spans(page_ids, start: int, length: int, page_size: int):
    """Map a logical token interval to scheduler-owned page/offset/count spans."""

    if page_size < 1 or start < 0 or length < 0 or start + length > len(page_ids) * page_size:
        raise ValueError("KV token interval exceeds its block table")
    spans = []
    while length:
        logical, offset = divmod(start, page_size)
        count = min(length, page_size - offset)
        spans.append((page_ids[logical], offset, count))
        start += count
        length -= count
    return tuple(spans)
