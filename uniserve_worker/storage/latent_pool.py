"""Fixed physical storage for scheduler-placed generation trajectories.

``LatentPool`` owns the device pages that hold a denoiser's samples while a
request's trajectory advances. Storage is two banks of the same pages,
``[2, num_pages, page_units, latent_width]``. A request slot's committed
trajectory lives in its active bank; a step writes the successor to the
other, inactive bank, and the batch commit makes it visible by flipping the
slot's active bank (``validate_updates`` then ``apply_updates``). Page zero
is a sentinel outside scheduler capacity.

The scheduler names each call's pages; the pool checks them against the
page ownership, step and generation it recorded. It also owns the physical
lifetimes around the banks: an import's pages, written in bank zero, stay
reserved until its transfers retire even if the import is abandoned, and a
publication keeps an immutable bank version reserved until every transport
reader retires, so a later step cannot overwrite pages a peer is still
reading.

``LatentPoolPlan`` chooses one of two geometries. A KV-conditioned image
denoiser's pages hold latent tokens, and each step is staged through the
pool's fixed ``step_buffer`` (``stage``, ``gather_current``,
``write_inactive``). A standalone denoiser's pages hold sample elements,
and the pool allocates no step buffer: media execution validates each call
with ``initial_bank`` or ``step_banks``, writes initial samples through
``bank_view``, and ``DiffusionRunner`` gathers and scatters each step's
pages through ``page_rows``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from uniserve.runtime.device import fill_cpu_ints
from uniserve_worker.errors import invalid_descriptor, resource_error
from uniserve_worker.protocol.batch import LatentParams
from uniserve_worker.protocol.identity import BufferId, RequestKey
from uniserve_worker.protocol.tensor import TensorRef
from uniserve_worker.storage.host_buffers import HostBuffers
from uniserve_worker.transport.exports import ExportLocations, release_exports
from uniserve_worker.transport.ticket import TransferTicket


@dataclass(slots=True)
class LatentUpdate:
    """A prepared trajectory publication or release for one physical slot.

    The pool validates these coordinates before any batch publication becomes
    visible, then applies the same values once every owner accepts its update.

    Attributes:
        request_pool_idx: One-based request slot the update applies to.
        params: The trajectory's pages, unit count and raster. ``None``
            means the call changes no trajectory, and the pool skips it.
        expected_generation: Generation of the committed trajectory a
            publication advances; zero initializes an empty slot.
        expected_step: Committed step a publication advances.
        generation: The successor's generation for a publication, or the
            committed generation being released for a release.
        step: The successor's step for a publication, or the committed step
            being released for a release.
        release: Release the slot's committed trajectory and pages instead of
            publishing a successor.
    """

    request_pool_idx: int
    params: LatentParams | None = None
    expected_generation: int = 0
    expected_step: int = 0
    generation: int = 0
    step: int = 0
    release: bool = False


@dataclass(frozen=True, slots=True)
class LatentStaging:
    """Fixed-address page-table and contiguous value views for one call.

    ``pages`` is a slice of the pool's ``page_table_buffer`` and ``value`` the
    same page range of its ``step_buffer``; ``LatentPool.stage`` locates a
    live staging's range from the storage offset of ``pages``.
    """

    page_table: tuple[int, ...]
    # Device page indices for this call, [len(page_table)] int64.
    pages: torch.Tensor
    # Contiguous staged values, [len(page_table) * page_units, latent_width];
    # rows past the call's latent units are the last page's padding.
    value: torch.Tensor


@dataclass(slots=True)
class LatentImport:
    """A reserved import range, held until its physical read retires.

    The pool keeps the import registered for its slot until every retained
    transfer has retired, even after adoption or abandonment; an abandoned
    import's pages are freed only then.
    """

    product: TensorRef
    request_pool_idx: int
    page_table: tuple[int, ...]
    # Per-page bank-zero views, padding excluded: [units_in_page, latent_width].
    spans: tuple[torch.Tensor, ...]
    # Reads writing ``spans``, attached through ``retain_transfer``.
    transfers: tuple[TransferTicket, ...] = ()
    # Set by ``adopt_import`` once the pages hold the slot's trajectory.
    adopted: bool = False
    # Set when the import is abandoned or its slot is cleared.
    released: bool = False


@dataclass(slots=True)
class LatentExport:
    """One page-bank version retained by its publication registrations.

    While the export is registered, the version is immutable: the pool refuses
    writes to ``page_table`` in ``bank`` and does not grant those pages as
    free to another slot. The export is dropped once ``released`` is set and
    every retirement has completed without an exception.
    """

    buffer: BufferId
    request_pool_idx: int
    bank: int
    page_table: tuple[int, ...]
    # Per-page views, padding excluded: [units_in_page, latent_width].
    spans: tuple[torch.Tensor, ...]
    # One future per transport registration, attached through
    # ``retain_publication``; each completes when that transport's readers
    # retire.
    retirements: tuple[Future[None], ...] = ()
    # Set by ``release_buffers``; ``retain_publication`` then refuses it.
    released: bool = False


class LatentPool:
    """Own two page banks, fixed step staging, and request-indexed visibility.

    Request slots are one-based, matching the scheduler's slot ids. Per slot
    the pool records the active bank, step, generation, unit count and
    raster of the committed trajectory; generation zero marks an empty slot.
    Per page it records the owning slot, zero for a free page.
    """

    def __init__(
        self,
        *,
        request_pool_size: int,
        num_pages: int,
        page_units: int,
        latent_width: int,
        dtype: torch.dtype,
        device: torch.device | str,
        staging: bool = True,
    ) -> None:
        """Allocate double-buffered latent pages and request-indexed progress.

        ``staging`` allocates the fixed step buffer through which ``stage``
        lends contiguous views. A consumer that gathers and scatters a
        trajectory's pages itself, naming them with ``initial_bank`` and
        ``step_banks``, allocates none.

        ``page_units`` counts latent rows per page and ``latent_width`` the
        values per row; ``capacity_units`` excludes the sentinel page.

        Raises:
            ValueError: ``request_pool_size``, ``page_units`` or
                ``latent_width`` is below one, ``num_pages`` is below two, or
                ``dtype`` is not a floating dtype.
        """
        if (
            min(int(request_pool_size), int(page_units), int(latent_width)) < 1
            or int(num_pages) < 2
        ):
            raise ValueError(
                "latent-pool shape must contain slots, pages, and elements"
            )
        if not dtype.is_floating_point:
            raise ValueError("latent-pool storage must use a floating dtype")

        self.request_pool_size = int(request_pool_size)
        self.num_pages = int(num_pages)
        self.page_units = int(page_units)
        self.latent_width = int(latent_width)
        self.dtype = dtype
        self.device = torch.device(device)
        self.capacity_units = (self.num_pages - 1) * self.page_units

        # Page zero remains outside scheduler capacity; the two banks alternate
        # source and destination roles across diffusion steps.
        # [2, num_pages, page_units, latent_width]
        self.storage = torch.zeros(
            (2, self.num_pages, self.page_units, self.latent_width),
            dtype=self.dtype,
            device=self.device,
        )
        # [capacity_units, latent_width], or no rows without pool staging.
        # The page at position ``p`` of ``page_table_buffer`` is staged in the
        # ``page_units`` rows starting at row ``p * page_units``.
        self.step_buffer = torch.empty(
            (self.capacity_units if staging else 0, self.latent_width),
            dtype=self.dtype,
            device=self.device,
        )
        # [num_pages - 1] int64: device page indices of every live staging.
        self.page_table_buffer = torch.empty(
            self.num_pages - 1,
            dtype=torch.int64,
            device=self.device,
        )
        # [request_pool_size + 1, 1] float32, indexed by request slot (row
        # zero unused): each slot's current network time, the fixed storage
        # its denoising rows read.
        self.timesteps = torch.empty(
            (self.request_pool_size + 1, 1),
            dtype=torch.float32,
            device=self.device,
        )

        # Host metadata is authoritative for ownership and generation checks.
        # Per-slot rows are indexed by the one-based slot id (row zero unused);
        # ``_owners`` holds each page's slot, zero when free.
        rows = self.request_pool_size + 1
        self._active = torch.zeros(rows, dtype=torch.int8)
        self._steps = torch.zeros(rows, dtype=torch.int32)
        self._generations = torch.zeros(rows, dtype=torch.int64)
        self._units = torch.zeros(rows, dtype=torch.int32)
        self._heights = torch.zeros(rows, dtype=torch.int32)
        self._widths = torch.zeros(rows, dtype=torch.int32)
        self._owners = torch.zeros(self.num_pages, dtype=torch.int32)
        self._slot_pages: list[tuple[int, ...]] = [() for _ in range(rows)]
        # Pending or adopted imports by slot, until their transfers retire.
        self._imports: dict[int, LatentImport] = {}
        # Transport registrations of this pool's publications by buffer. The
        # batch commit adds them; ``release_buffers`` and the executor revoke
        # them through ``uniserve_worker.transport.exports``.
        self.exports: dict[BufferId, ExportLocations] = {}
        # Reserved bank versions by publication buffer.
        self._sources: dict[BufferId, LatentExport] = {}
        # Released slots whose reset waits for their imports and sources.
        self._retiring_slots: set[int] = set()

        # One host source, pinned on CUDA, for the page-index copies of
        # ``_device_pages``. With depth one, the next copy waits on the host
        # for the previous one to finish before the source is overwritten.
        self._page_table_staging = HostBuffers(
            self.num_pages - 1, dtype=torch.int64, depth=1, device=self.device
        )

    @property
    def persistent_bytes(self) -> int:
        """Bytes held persistently on the execution device by this pool.

        Must equal ``latent_pool_capacity_bytes`` for the same arguments:
        ``Worker.__init__`` refuses a pool whose allocation disagrees with the
        bytes storage sizing charged for it, so adding a device tensor here
        requires the same change there.
        """
        tensors = (
            self.storage,
            self.step_buffer,
            self.page_table_buffer,
            self.timesteps,
        )
        return sum(
            int(value.numel()) * int(value.element_size()) for value in tensors
        )

    @contextmanager
    def startup_values(self, rows: int, units: int):
        """Borrow the step buffer for numerical startup before admission.

        Stages ``rows`` disjoint runs of consecutive pages starting at page
        one, each sized for ``units``, and yields one ``[units, latent_width]``
        view per row. The pages are staged only; no slot takes ownership.
        Requires a pool allocated with ``staging``.

        Raises:
            RuntimeError: A slot owns pages or an import or publication is
                registered.
            WorkerError: From ``stage`` when the runs are invalid or exceed
                the step buffer.
        """
        if any(self._slot_pages) or self._imports or self._sources:
            raise RuntimeError("startup scratch requires an idle latent pool")

        count = (units + self.page_units - 1) // self.page_units
        pages = tuple(
            tuple(range(1 + row * count, 1 + (row + 1) * count))
            for row in range(rows)
        )
        views = tuple(
            item.value[:units] for item in self.stage(pages, (units,) * rows)
        )
        try:
            yield views
        finally:
            # The buffer is borrowed again by the first admitted call, so
            # startup writes must be complete before control returns.
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()

    def stage(
        self,
        page_tables: Sequence[Sequence[int]],
        latent_units: Sequence[int],
        *,
        occupied: Sequence[LatentStaging] = (),
    ) -> tuple[LatentStaging, ...]:
        """Borrow disjoint contiguous views alongside the supplied live staging.

        Each returned ``LatentStaging`` views one range of
        ``page_table_buffer``, filled here with its device page indices, and
        the same page range of ``step_buffer``. ``occupied`` must list every
        staging whose numerical consumer has not ended: a live range left out
        can be overwritten. The caller retains every live view until its
        numerical consumer ends. Scratch ranges are selected from these actual
        views, so independent completion groups need no duplicate allocator or
        allocation handles.

        Page tables are checked for bounds, extent and overlap only; the
        consumers (``initialize``, ``gather_current``, ``write_inactive``)
        check slot ownership. Requires a pool allocated with ``staging``.

        Raises:
            WorkerError: From ``invalid_descriptor`` when the columns are empty
                or misaligned, a page table is invalid for its units, the
                tables overlap each other or ``occupied``, or they exceed the
                step buffer; from ``resource_error`` when no free range beside
                ``occupied`` holds them.
        """
        if not page_tables or len(page_tables) != len(latent_units):
            raise invalid_descriptor("latent staging columns are not aligned")

        canonical: list[tuple[int, ...]] = []
        total_pages = 0
        for page_table, units in zip(page_tables, latent_units, strict=True):
            pages = self._validate_page_table(page_table, int(units))
            canonical.append(pages)
            total_pages += len(pages)
        if total_pages > self.num_pages - 1:
            raise invalid_descriptor(
                "latent staging exceeds the fixed step buffer"
            )

        flattened = tuple(page for pages in canonical for page in pages)
        if len(set(flattened)) != len(flattened):
            raise invalid_descriptor("latent staging page tables overlap")
        if set(flattened).intersection(
            page for item in occupied for page in item.page_table
        ):
            raise invalid_descriptor(
                "latent staging page tables overlap live calls"
            )

        # First fit: place this batch in the lowest gap between occupied
        # ranges that holds it. A staging's page offset in
        # ``page_table_buffer`` is also its page offset in ``step_buffer``.
        ranges = sorted(
            (int(item.pages.storage_offset()), len(item.page_table))
            for item in occupied
        )
        page_offset = 0
        for start, count in ranges:
            if page_offset + total_pages <= start:
                break
            page_offset = max(page_offset, start + count)
        if page_offset + total_pages > self.num_pages - 1:
            raise resource_error(
                "live latent staging exceeds the fixed step buffer"
            )

        self._device_pages(flattened, offset=page_offset)
        result: list[LatentStaging] = []
        unit_offset = page_offset * self.page_units
        for pages in canonical:
            page_count = len(pages)
            padded_units = page_count * self.page_units
            result.append(
                LatentStaging(
                    page_table=pages,
                    pages=self.page_table_buffer[
                        page_offset : page_offset + page_count
                    ],
                    value=self.step_buffer[
                        unit_offset : unit_offset + padded_units
                    ],
                )
            )
            page_offset += page_count
            unit_offset += padded_units
        return tuple(result)

    def initialize(
        self,
        request_pool_idx: int,
        staging: LatentStaging,
        *,
        latent_units: int,
    ) -> None:
        """Write a transition result to the currently inactive bank.

        The slot must be empty and its pages free. The pages stay unowned and
        the trajectory invisible until the batch commit applies the slot's
        initializing update (``expected_generation`` zero).
        """
        slot = self._validate_slot(int(request_pool_idx))
        pages = self._validate_staging(staging, int(latent_units))
        self._require_empty(slot)
        self._require_page_owners(pages, 0)

        # An empty slot's active bank is zero, and the initializing update
        # flips it like any successor in ``apply_updates``, so a fresh
        # trajectory is written to bank one.
        self._require_writable(1, pages)
        self._write_pages(1, staging.pages, staging.value)

    @property
    def page_rows(self) -> torch.Tensor:
        """Every page of both banks as one row each.

        Row ``bank * num_pages + page`` holds ``page`` of ``bank``:
        [2 * num_pages, page_units * latent_width]. ``DiffusionRunner``
        gathers and scatters a step's pages through these rows. The view
        performs no ownership check.
        """
        return self.storage.view(2 * self.num_pages, -1)

    def bank_view(self, bank: int, page_table: Sequence[int]) -> torch.Tensor:
        """View a trajectory's pages of one bank as contiguous memory.

        The pages must be consecutive, as the pages a request slot owns in a
        consumer-staged pool are: [len(page_table) * page_units, latent_width].
        Only the bank, bounds and consecutiveness are checked; callers
        validate the trajectory first with ``initial_bank`` or
        ``step_banks``.
        """
        pages = tuple(int(page) for page in page_table)
        first = pages[0] if pages else 0
        if (
            int(bank) not in (0, 1)
            or not pages
            or pages != tuple(range(first, first + len(pages)))
            or first < 1
            or first + len(pages) > self.num_pages
        ):
            raise invalid_descriptor(
                "latent bank view requires consecutive pool pages"
            )
        return self.storage[int(bank), first : first + len(pages)].view(
            -1, self.latent_width
        )

    def initial_bank(
        self,
        request_pool_idx: int,
        page_table: Sequence[int],
        *,
        latent_units: int,
    ) -> int:
        """Name the bank a fresh trajectory's consumer writes.

        Applies the checks ``initialize`` does for a trajectory whose
        consumer writes its pages itself; the caller then publishes an
        initialization update with the batch.
        """
        slot = self._validate_slot(int(request_pool_idx))
        pages = self._validate_page_table(page_table, int(latent_units))
        self._require_empty(slot)
        self._require_page_owners(pages, 0)

        # A fresh trajectory is written to bank one, as in ``initialize``.
        self._require_writable(1, pages)
        return 1

    def step_banks(
        self,
        request_pool_idx: int,
        page_table: Sequence[int],
        *,
        step: int,
        generation: int,
        latent_units: int,
        height: int,
        width: int,
    ) -> tuple[int, int]:
        """Name the banks of one step over a committed trajectory.

        Applies the checks ``gather_current`` and ``write_inactive`` do, for a
        consumer that gathers and scatters the trajectory's pages itself.
        Returns the bank holding the committed trajectory and the inactive
        bank its successor is written to; the caller publishes the successor
        with the batch, which makes it the committed bank.
        """
        slot = self._validate_slot(int(request_pool_idx))
        pages = self._validate_page_table(page_table, int(latent_units))
        self._require_current(
            slot,
            step=int(step),
            generation=int(generation),
            latent_units=int(latent_units),
            height=int(height),
            width=int(width),
        )
        self._require_slot_pages(slot, pages)
        self._require_page_owners(pages, slot)

        bank = int(self._active[slot].item())
        self._require_writable(1 - bank, pages)
        return bank, 1 - bank

    def gather_current(
        self,
        request_pool_idx: int,
        staging: LatentStaging,
        *,
        step: int,
        generation: int,
        latent_units: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Gather one committed page table into its fixed contiguous view.

        Validates the call against the slot's committed trajectory, page
        table and page ownership, copies the active bank's pages, padding
        included, into ``staging.value`` on the current stream, and returns
        its ``[latent_units, latent_width]`` prefix.
        """
        slot = self._validate_slot(int(request_pool_idx))
        pages = self._validate_staging(staging, int(latent_units))
        self._require_current(
            slot,
            step=int(step),
            generation=int(generation),
            latent_units=int(latent_units),
            height=int(height),
            width=int(width),
        )
        self._require_slot_pages(slot, pages)
        self._require_page_owners(pages, slot)

        bank = int(self._active[slot].item())
        torch.index_select(
            self.storage[bank],
            0,
            staging.pages,
            out=staging.value.view(
                len(pages), self.page_units, self.latent_width
            ),
        )
        return staging.value[: int(latent_units)]

    def write_inactive(
        self,
        request_pool_idx: int,
        staging: LatentStaging,
        *,
        expected_step: int,
        expected_generation: int,
        latent_units: int,
        height: int,
        width: int,
    ) -> None:
        """Scatter a complete successor to the slot's hidden inactive bank.

        Validates the committed trajectory the successor advances as
        ``gather_current`` does, and refuses pages of the inactive bank that a
        publication still holds. The successor becomes visible only when the
        batch commit applies its update and flips the slot's active bank.
        """
        slot = self._validate_slot(int(request_pool_idx))
        pages = self._validate_staging(staging, int(latent_units))
        self._require_current(
            slot,
            step=int(expected_step),
            generation=int(expected_generation),
            latent_units=int(latent_units),
            height=int(height),
            width=int(width),
        )
        self._require_slot_pages(slot, pages)
        self._require_page_owners(pages, slot)

        bank = 1 - int(self._active[slot].item())
        self._require_writable(bank, pages)
        self._write_pages(bank, staging.pages, staging.value)

    def reserve_publication(
        self,
        product: TensorRef,
        *,
        request_pool_idx: int,
        page_table: Sequence[int],
        latent_units: int,
    ) -> LatentExport:
        """Retain the written successor bank before registering its spans.

        The successor is in the slot's inactive bank, not yet committed. Only
        the page table's bounds and the absence of another publication on
        those pages of that bank are checked, not page ownership, so the
        successor of an initializing call can be published before the commit
        that makes its pages owned. The caller attaches every transport
        retirement with ``retain_publication``. Failure before semantic
        visibility must release this reservation (``release_buffers``) as
        well as any physical registrations that were already created.
        """
        self._reap_sources()
        slot = self._validate_slot(request_pool_idx)
        pages = self._validate_page_table(page_table, latent_units)

        # The successor was written to the inactive bank and is not yet active.
        bank = 1 - int(self._active[slot].item())
        self._require_writable(bank, pages)
        return self._reserve_source(product, slot, bank, pages, latent_units)

    def reserve_current_publication(
        self,
        product: TensorRef,
        *,
        request_pool_idx: int,
        page_table: Sequence[int],
        generation: int,
        step: int,
        latent_units: int,
        height: int,
        width: int,
    ) -> LatentExport:
        """Retain a committed trajectory for an independently owned output.

        The trajectory must match the slot's committed step, generation,
        units, raster and pages exactly.

        Publication does not change the request's generation or step.
        Multiple products may retain the same immutable bank; all must retire
        before a later step can reuse it. The caller attaches each
        registration's future with ``retain_publication`` and releases the
        output on abandonment.
        """
        self._reap_sources()
        slot = self._validate_slot(request_pool_idx)
        pages = self._validate_page_table(page_table, latent_units)
        self._require_current(
            slot,
            generation=generation,
            step=step,
            latent_units=latent_units,
            height=height,
            width=width,
        )
        self._require_slot_pages(slot, pages)
        self._require_page_owners(pages, slot)
        return self._reserve_source(
            product, slot, int(self._active[slot].item()), pages, latent_units
        )

    def _reserve_source(
        self,
        product: TensorRef,
        slot: int,
        bank: int,
        pages: tuple[int, ...],
        latent_units: int,
    ) -> LatentExport:
        """Register one immutable bank version with its per-page value spans."""
        if product.buffer_id in self._sources:
            raise invalid_descriptor(
                "latent publication generation is already registered"
            )

        # Each span excludes its page's padding: [units_in_page, latent_width].
        source = LatentExport(
            product.buffer_id,
            slot,
            bank,
            pages,
            tuple(
                self.storage[
                    bank, page, : min(self.page_units, latent_units - offset)
                ]
                for page, offset in zip(
                    pages, range(0, latent_units, self.page_units), strict=True
                )
            ),
        )
        self._sources[source.buffer] = source
        return source

    def retain_publication(
        self, source: LatentExport, retirement: Future[None]
    ) -> None:
        """Keep the registered bank version until its physical readers retire.

        Raises ``invalid_descriptor`` when the reservation was released or is
        no longer registered.
        """
        if self._sources.get(source.buffer) is not source or source.released:
            raise invalid_descriptor(
                "latent publication reservation is no longer active"
            )
        source.retirements = (*source.retirements, retirement)

    def release_buffers(self, buffers: Sequence[BufferId]) -> None:
        """Revoke bank reservations while retaining every pending physical read.

        Revokes the named buffers' transport registrations and marks their
        bank versions released; each version stays reserved until its
        retirements complete. Buffers this pool never published are ignored.
        """
        release_exports(self.exports, buffers)
        for buffer in buffers:
            source = self._sources.get(buffer)
            if source is not None:
                source.released = True
        self._reap_sources()

    def write_dependencies(
        self, request_pool_idx: int, page_table: Sequence[int]
    ) -> tuple[Future[None], ...]:
        """Return the retirements that must precede writing the next bank.

        These are the retirements of every publication holding any of
        ``page_table`` in the slot's inactive bank, the bank its next
        preparation or step writes. Batch preparation adds them to the batch's
        storage dependencies.
        """
        self._reap_sources()
        slot = self._validate_slot(request_pool_idx)
        bank = 1 - int(self._active[slot].item())
        pages = set(page_table)
        return tuple(
            retirement
            for source in self._sources.values()
            if source.bank == bank and not pages.isdisjoint(source.page_table)
            for retirement in source.retirements
        )

    def _require_writable(self, bank: int, pages: Sequence[int]) -> None:
        """Reject writes that would clobber a published version's pages.

        Raises ``resource_error`` while any registered publication, released
        or not, holds one of ``pages`` in ``bank``.
        """
        self._reap_sources()
        selected = set(pages)
        if any(
            source.bank == bank and not selected.isdisjoint(source.page_table)
            for source in self._sources.values()
        ):
            raise resource_error(
                "latent page bank still has a published version"
            )

    def _reap_sources(self) -> None:
        """Drop fully retired sources and finish clearing their slots.

        A source whose retirement completed with an exception is kept, so its
        pages stay reserved; ``retirement_ready`` reports the failure to the
        owning request.
        """
        for buffer, source in tuple(self._sources.items()):
            if not source.released or any(
                not future.done() or future.exception() is not None
                for future in source.retirements
            ):
                continue
            del self._sources[buffer]
        for slot in tuple(self._retiring_slots):
            if slot not in self._imports and not any(
                source.request_pool_idx == slot
                for source in self._sources.values()
            ):
                self._retiring_slots.remove(slot)
                self._clear_slot(slot, self._slot_pages[slot])

    def stage_timestep(
        self, request_pool_idx: int, value: float
    ) -> torch.Tensor:
        """Place one step's network time in the slot's fixed storage.

        Returns the one-element view the slot's denoising rows and solver
        update read.
        """
        slot = self._validate_slot(int(request_pool_idx))
        row = self.timesteps[slot]
        row.fill_(float(value))
        return row

    def validate_updates(
        self,
        updates: Sequence[LatentUpdate],
    ) -> None:
        """Validate an entire lane's visibility changes without mutation.

        Updates without ``params`` are skipped, and a slot may appear once. A
        publication with ``expected_generation`` zero initializes an empty
        slot at step zero on free pages; any other publication must name the
        slot's committed trajectory and pages and advance both its step and
        generation. The batch's publications must not share pages. A release
        must name the slot's committed trajectory exactly. Nothing changes
        beyond reaping retired publications and the released slots they held.

        Raises:
            WorkerError: From ``invalid_descriptor`` for any violated rule.
        """
        publications = tuple(
            output
            for output in updates
            if output.params is not None and not output.release
        )
        releases = tuple(
            output
            for output in updates
            if output.params is not None and output.release
        )
        slots = (
            *(int(value.request_pool_idx) for value in publications),
            *(int(value.request_pool_idx) for value in releases),
        )
        if len(set(slots)) != len(slots):
            raise invalid_descriptor("latent commit repeats a request slot")

        claimed_pages: set[int] = set()
        for publication in publications:
            params = publication.params
            assert params is not None
            slot = self._validate_slot(int(publication.request_pool_idx))
            pages = self._validate_page_table(
                params.page_table, int(params.latent_units)
            )
            self._validate_metadata(
                generation=int(publication.generation),
                step=int(publication.step),
                latent_units=int(params.latent_units),
                height=int(params.height),
                width=int(params.width),
            )

            # An expected generation of zero marks slot initialization; every
            # later publication must advance the committed trajectory. The
            # initializing slot's own publications may already hold its
            # pages: ``reserve_publication`` can register the successor before
            # this commit.
            expected = int(publication.expected_generation)
            if expected == 0:
                if (
                    int(publication.expected_step) != 0
                    or int(publication.step) != 0
                ):
                    raise invalid_descriptor(
                        "latent initialization must publish step zero"
                    )
                self._require_empty(slot)
                self._require_page_owners(pages, 0, publication_slot=slot)
            else:
                self._require_current(
                    slot,
                    step=int(publication.expected_step),
                    generation=expected,
                    latent_units=int(params.latent_units),
                    height=int(params.height),
                    width=int(params.width),
                )
                self._require_slot_pages(slot, pages)
                self._require_page_owners(pages, slot)
                if int(publication.step) <= int(publication.expected_step):
                    raise invalid_descriptor(
                        "latent successor does not advance its step"
                    )
            if int(publication.generation) <= expected:
                raise invalid_descriptor(
                    "latent publication does not advance its generation"
                )
            if not claimed_pages.isdisjoint(pages):
                raise invalid_descriptor(
                    "latent commit publications overlap physical pages"
                )
            claimed_pages.update(pages)

        for release in releases:
            params = release.params
            assert params is not None
            slot = self._validate_slot(int(release.request_pool_idx))
            pages = self._validate_page_table(
                params.page_table, int(params.latent_units)
            )
            self._require_current(
                slot,
                step=int(release.step),
                generation=int(release.generation),
                latent_units=int(params.latent_units),
                height=int(params.height),
                width=int(params.width),
            )
            self._require_slot_pages(slot, pages)
            self._require_page_owners(pages, slot)

    def apply_updates(
        self,
        updates: Sequence[LatentUpdate],
    ) -> None:
        """Apply changes already accepted by :meth:`validate_updates`.

        Nothing is rechecked, so the caller must pass the same updates with no
        pool change in between. Initializations take their pages' ownership,
        every publication flips its slot's active bank, and releases clear
        their slots, deferred while readers remain.
        """
        publications = tuple(
            output
            for output in updates
            if output.params is not None and not output.release
        )
        releases = tuple(
            output
            for output in updates
            if output.params is not None and output.release
        )
        for publication in publications:
            params = publication.params
            assert params is not None
            slot = int(publication.request_pool_idx)
            pages = tuple(int(page) for page in params.page_table)
            if int(publication.expected_generation) == 0:
                for page in pages:
                    self._owners[page] = slot
                self._slot_pages[slot] = pages

            # The successor was written to the inactive bank; flip it active.
            bank = 1 - int(self._active[slot].item())
            self._active[slot] = bank
            self._steps[slot] = int(publication.step)
            self._generations[slot] = int(publication.generation)
            self._units[slot] = int(params.latent_units)
            self._heights[slot] = int(params.height)
            self._widths[slot] = int(params.width)

        for release in releases:
            params = release.params
            assert params is not None
            self._clear_slot(
                int(release.request_pool_idx),
                tuple(int(page) for page in params.page_table),
            )
        self._reap_imports()

    def reserve_import(
        self,
        product: TensorRef,
        *,
        request_pool_idx: int,
        page_table: Sequence[int],
        latent_units: int,
    ) -> LatentImport:
        """Own destination pages before granting a transfer access.

        The slot must be empty and the pages free; the slot owns them from
        this call. The returned first-axis spans exclude page padding. They
        address bank zero directly and remain invisible to computation until
        adoption, which makes bank zero the slot's active bank.
        """
        self._reap_imports()
        slot = self._validate_slot(int(request_pool_idx))
        self._require_empty(slot)
        pages = self._validate_page_table(page_table, int(latent_units))
        self._require_page_owners(pages, 0)

        # One span per page, padding excluded: [units_in_page, latent_width].
        spans = tuple(
            self.storage[
                0, page, : min(self.page_units, int(latent_units) - offset)
            ]
            for page, offset in zip(
                pages, range(0, int(latent_units), self.page_units), strict=True
            )
        )
        # Padding is outside every granted span, so its initialization cannot
        # race the independent transfer stream's payload writes.
        self.storage[0, pages[-1], int(spans[-1].shape[0]) :].zero_()
        write = LatentImport(product, slot, pages, spans)
        for page in pages:
            self._owners[page] = slot
        self._imports[slot] = write
        return write

    def retain_transfer(
        self, write: LatentImport, ticket: TransferTicket
    ) -> None:
        """Retain the physical copy even if its preparation is later abandoned.

        Raises ``invalid_descriptor`` when the import was released, replaced
        or already adopted.
        """
        self._require_import(write)
        if write.adopted:
            raise invalid_descriptor(
                "resident latent import cannot accept another read"
            )
        write.transfers = (*write.transfers, ticket)

    def adopt_import(
        self,
        write: LatentImport,
        *,
        generation: int,
        step: int,
        height: int,
        width: int,
    ) -> None:
        """Expose imported pages after their tickets order the consuming stream.

        The imported trajectory becomes the slot's committed trajectory in
        bank zero at ``generation`` and ``step``, with the unit count of its
        spans. Raises ``invalid_descriptor`` when the import is stale, already
        adopted or has no transfer, the metadata is invalid, or ``generation``
        differs from the product's; a transfer that is not ready, failed or
        closed raises from its ticket's ``result``.
        """
        self._require_import(write)
        if write.adopted or not write.transfers:
            raise invalid_descriptor("latent import cannot be adopted")
        # result() establishes the producer fence on the current execution
        # stream; readiness alone does not authorize use of the page contents.
        for transfer in write.transfers:
            transfer.result()

        units = sum(int(span.shape[0]) for span in write.spans)
        self._validate_metadata(
            generation=generation,
            step=step,
            latent_units=units,
            height=height,
            width=width,
        )
        if generation != write.product.generation:
            raise invalid_descriptor(
                "latent import generation disagrees with its product"
            )

        # Imported pages live in bank zero, which becomes the active bank
        # without a flip. The pages were owned at reservation.
        slot = write.request_pool_idx
        self._slot_pages[slot] = write.page_table
        self._active[slot] = 0
        self._steps[slot] = step
        self._generations[slot] = generation
        self._units[slot] = units
        self._heights[slot] = height
        self._widths[slot] = width
        write.adopted = True
        self._reap_imports()

    def abandon_import(self, write: LatentImport) -> None:
        """Revoke an unadopted import without reusing a still-written page.

        Cancels its transfers; the pages return to the free pool only after
        every transfer retires. Abandoning a released import does nothing;
        an adopted or stale import raises ``invalid_descriptor``.
        """
        if write.released:
            return
        self._require_import(write)
        if write.adopted:
            raise invalid_descriptor(
                "resident latent import cannot be abandoned"
            )

        write.released = True
        for transfer in write.transfers:
            transfer.cancel()
        self._reap_imports()

    def retirement_ready(self, requests: Sequence[RequestKey]) -> bool:
        """Require known copy completion before Finish returns request pages.

        Returns whether no import or publication of ``requests`` remains
        after reaping. Raises when a transfer of one of those requests has an
        unknown physical completion or a completed publication retirement of
        theirs failed.
        """
        # Failed physical access retains its range. Report the failure only to
        # its owner; independent requests can still reclaim or use other pages.
        for write in self._imports.values():
            if write.product.request_key in requests:
                for transfer in write.transfers:
                    transfer.retirement_ready()

        for source in self._sources.values():
            if source.buffer.owner in requests:
                for future in source.retirements:
                    if future.done():
                        future.result()

        self._reap_imports()
        self._reap_sources()
        return all(
            write.product.request_key not in requests
            for write in self._imports.values()
        ) and all(
            source.buffer.owner not in requests
            for source in self._sources.values()
        )

    def cancel_imports(self, requests: Sequence[RequestKey]) -> None:
        """Revoke unfinished admissions; resident trajectories keep their pages.

        Abandons every unadopted, unreleased import of ``requests``. Adopted
        trajectories stay until their slots are released.
        """
        for write in tuple(self._imports.values()):
            if (
                write.product.request_key in requests
                and not write.adopted
                and not write.released
            ):
                self.abandon_import(write)

    def _require_import(self, write: LatentImport) -> None:
        """Reject stale handles whose reservation was adopted or revoked."""
        if (
            self._imports.get(write.request_pool_idx) is not write
            or write.released
        ):
            raise invalid_descriptor(
                "latent import reservation is no longer writable"
            )

    def _reap_imports(self) -> None:
        """Drop finished imports and release the pages of abandoned ones.

        An adopted or released import is dropped only once every transfer
        has retired.
        """
        for slot, write in tuple(self._imports.items()):
            if not write.adopted and not write.released:
                continue
            if any(not transfer.retired() for transfer in write.transfers):
                continue
            del self._imports[slot]
            if write.released:
                self._clear_slot(slot, write.page_table)

    def release_slots(self, request_pool_indices: Sequence[int]) -> None:
        """Release all pages owned by exact request slots.

        A slot whose imports or publications are still registered is reset
        only once they retire. Raises ``invalid_descriptor`` when a slot is
        out of range or repeated.
        """
        slots = tuple(
            self._validate_slot(int(value)) for value in request_pool_indices
        )
        if len(set(slots)) != len(slots):
            raise invalid_descriptor("latent release repeats a request slot")
        for slot in slots:
            pages = self._slot_pages[slot]
            self._clear_slot(slot, pages)
        self._reap_imports()

    def close(self) -> None:
        """Release storage once the owning transport has drained its reads.

        Revokes every publication and releases every slot with an import or
        publication, then drops the device allocations. Raises
        ``resource_error`` when an import or publication has not retired.
        """
        self._page_table_staging.close()
        self.release_buffers(tuple(self._sources))
        self.release_slots(
            tuple(
                set(self._imports)
                | {source.request_pool_idx for source in self._sources.values()}
            )
        )
        self._reap_sources()
        if self._imports or self._sources:
            raise resource_error(
                "latent physical reads must retire before pool shutdown"
            )

        self.exports.clear()
        # Drop device allocations while keeping each attribute well-typed.
        for name, dtype in (
            ("storage", self.dtype),
            ("step_buffer", self.dtype),
            ("page_table_buffer", torch.int64),
            ("timesteps", torch.float32),
        ):
            setattr(self, name, torch.empty(0, dtype=dtype, device=self.device))

    def _write_pages(
        self, bank: int, pages: torch.Tensor, value: torch.Tensor
    ) -> None:
        """Scatter contiguous latent units into the selected physical page bank.

        ``pages`` is a device int64 page index per staged page, and ``value``
        the padded ``[pages.numel() * page_units, latent_width]`` rows.
        """
        self.storage[int(bank)].index_copy_(
            0,
            pages,
            value.view(int(pages.numel()), self.page_units, self.latent_width),
        )

    def _validate_staging(
        self, staging: LatentStaging, latent_units: int
    ) -> tuple[int, ...]:
        """Validate a staging's device, dtype, shape and extent for its units.

        Returns the staging's validated page table.
        """
        if (
            staging.pages.device != self.device
            or staging.pages.dtype != torch.int64
        ):
            raise invalid_descriptor(
                "latent page table is not in fixed device staging"
            )
        if (
            staging.value.device != self.device
            or staging.value.dtype != self.dtype
            or staging.value.ndim != 2
            or int(staging.value.shape[1]) != self.latent_width
        ):
            raise invalid_descriptor(
                "latent value is not in fixed device staging"
            )
        expected_pages = math.ceil(int(latent_units) / self.page_units)
        if (
            int(staging.pages.numel()) != expected_pages
            or int(staging.value.shape[0]) != expected_pages * self.page_units
        ):
            raise invalid_descriptor(
                "latent staging does not establish its logical extent"
            )
        return self._validate_page_table(staging.page_table, int(latent_units))

    def _validate_page_table(
        self, page_table: Sequence[int], latent_units: int
    ) -> tuple[int, ...]:
        """Validate page count, bounds and uniqueness for a latent payload.

        The table must hold exactly the pages ``latent_units`` needs, all
        distinct and within the usable pages, which exclude page zero.
        """
        units = int(latent_units)
        pages = tuple(int(page) for page in page_table)
        expected = math.ceil(units / self.page_units) if units > 0 else 0
        if (
            units < 1
            or len(pages) != expected
            or len(set(pages)) != len(pages)
            or any(page < 1 or page >= self.num_pages for page in pages)
        ):
            raise invalid_descriptor(
                "latent page table is outside physical pool bounds"
            )
        return pages

    def _require_page_owners(
        self,
        pages: Sequence[int] | torch.Tensor,
        owner: int,
        *,
        publication_slot: int | None = None,
    ) -> None:
        """Require every latent page to be owned by the expected request slot.

        ``owner`` zero requires free pages, which additionally must not be
        held by a publication of a slot other than ``publication_slot``.
        """
        canonical = (
            tuple(int(value) for value in pages.detach().cpu().tolist())
            if isinstance(pages, torch.Tensor)
            else tuple(int(value) for value in pages)
        )
        self._reap_sources()

        # Fresh pages (owner 0) must also be free of published bank versions.
        if owner == 0 and any(
            source.request_pool_idx != publication_slot
            and not set(canonical).isdisjoint(source.page_table)
            for source in self._sources.values()
        ):
            raise invalid_descriptor(
                "latent pages are owned by a published version"
            )
        if any(
            int(self._owners[page].item()) != int(owner) for page in canonical
        ):
            raise invalid_descriptor(
                "latent page table is not owned by its request slot"
            )

    def _require_current(
        self,
        slot: int,
        *,
        step: int,
        generation: int,
        latent_units: int,
        height: int,
        width: int,
    ) -> None:
        """Require the slot's committed trajectory to match the expected values.

        Compares step, generation, unit count and raster dimensions; page
        ownership is checked separately.
        """
        current = (
            int(self._steps[slot].item()),
            int(self._generations[slot].item()),
            int(self._units[slot].item()),
            int(self._heights[slot].item()),
            int(self._widths[slot].item()),
        )
        expected = (
            int(step),
            int(generation),
            int(latent_units),
            int(height),
            int(width),
        )
        if current != expected:
            raise invalid_descriptor(
                "latent allocation does not name the committed trajectory"
            )

    def _require_slot_pages(self, slot: int, pages: Sequence[int]) -> None:
        """Require a slot's committed page table to match the supplied pages."""
        if self._slot_pages[slot] != tuple(int(page) for page in pages):
            raise invalid_descriptor(
                "latent page table does not match its committed trajectory"
            )

    def _require_empty(self, slot: int) -> None:
        """Require a request slot to have no active latent trajectory.

        A registered import occupies the slot until it is reaped.
        """
        if (
            slot in self._imports
            or int(self._generations[slot].item()) != 0
            or int(self._units[slot].item()) != 0
        ):
            raise invalid_descriptor(
                "request slot already owns a committed trajectory"
            )

    def _validate_metadata(
        self,
        *,
        generation: int,
        step: int,
        latent_units: int,
        height: int,
        width: int,
    ) -> None:
        """Validate generation, step, unit count, and raster dimensions."""
        if (
            min(int(generation), int(latent_units), int(height), int(width)) < 1
            or int(step) < 0
        ):
            raise invalid_descriptor("latent publication metadata is invalid")
        if int(latent_units) > self.capacity_units:
            raise invalid_descriptor(
                "latent publication exceeds physical capacity"
            )

    def _clear_slot(self, slot: int, pages: Sequence[int]) -> None:
        """Release latent page ownership and reset all metadata for one slot.

        While an import or publication of the slot is registered, the slot is
        only marked retiring: its import is released, unadopted transfers are
        cancelled, and ``_reap_imports`` or ``_reap_sources`` finishes the
        reset once they retire.
        """
        write = self._imports.get(slot)
        if write is not None:
            write.released = True
            if not write.adopted:
                for transfer in write.transfers:
                    transfer.cancel()

        sources = tuple(
            source
            for source in self._sources.values()
            if source.request_pool_idx == slot
        )
        if write is not None or sources:
            # Physical readers may still hold the pages; defer the metadata
            # reset until every import and source of this slot retires.
            self._retiring_slots.add(slot)
            # A provisional producer can publish before the first lane commit.
            # Remember its pages so abandoning that lane cannot leak ownership.
            if not self._slot_pages[slot]:
                self._slot_pages[slot] = tuple(pages)
            return

        self._retiring_slots.discard(slot)
        for page in pages:
            self._owners[int(page)] = 0
        self._active[slot] = 0
        self._steps[slot] = 0
        self._generations[slot] = 0
        self._units[slot] = 0
        self._heights[slot] = 0
        self._widths[slot] = 0
        self._slot_pages[slot] = ()

    def _validate_slot(self, slot: int) -> int:
        """Validate a scheduler-visible, one-based latent request slot index."""
        if slot < 1 or slot > self.request_pool_size:
            raise invalid_descriptor(
                "latent request slot is outside physical capacity"
            )
        return slot

    def _device_pages(
        self, pages: Sequence[int], *, offset: int
    ) -> torch.Tensor:
        """Copy host page identifiers into reusable device index storage."""
        slot, host = self._page_table_staging.acquire()
        fill_cpu_ints(host, pages)

        # The copy may be asynchronous; the staging slot keeps the pinned host
        # source alive until the copy is known to be complete.
        target = self.page_table_buffer[offset : offset + len(pages)]
        target.copy_(
            host[: len(pages)], non_blocking=self.device.type == "cuda"
        )
        self._page_table_staging.record_copy(slot)
        return target


__all__ = [
    "LatentPool",
    "LatentStaging",
    "LatentImport",
]
