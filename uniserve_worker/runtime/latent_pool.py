"""Fixed physical storage for scheduler-placed generation trajectories."""

from __future__ import annotations

import math
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass

import torch

from ..execution.batch import BufferId, ProductRef, RequestKey
from ..foundation.errors import invalid_descriptor, resource_error
from ..transfer.tickets import TransferTicket
from .device import fill_cpu_ints


@dataclass(frozen=True, slots=True)
class LatentSnapshot:
    """One committed trajectory encoded for administrative recovery."""

    generation: int
    step: int
    latent_units: int
    height: int
    width: int
    value: torch.Tensor

    def __post_init__(self) -> None:
        """Validate committed latent pages, generation, step, units, and raster geometry."""

        if min(int(self.generation), int(self.latent_units), int(self.height), int(self.width)) < 1:
            raise ValueError("latent snapshot geometry is invalid")
        if int(self.step) < 0:
            raise ValueError("latent snapshot step is invalid")
        if (
            not self.value.is_floating_point()
            or self.value.ndim != 2
            or int(self.value.shape[0]) != int(self.latent_units)
        ):
            raise ValueError("latent snapshot tensor disagrees with its logical geometry")


@dataclass(frozen=True, slots=True)
class LatentPublication:
    """Validated visibility change applied at the lane commit point."""

    request_pool_idx: int
    page_table: tuple[int, ...]
    expected_generation: int
    expected_step: int
    generation: int
    step: int
    latent_units: int
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class LatentRelease:
    """Validated trajectory release applied at the lane commit point."""

    request_pool_idx: int
    page_table: tuple[int, ...]
    generation: int
    step: int
    latent_units: int
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class LatentStaging:
    """Fixed-address page-table and contiguous value views for one operation."""

    page_table: tuple[int, ...]
    pages: torch.Tensor
    value: torch.Tensor


@dataclass(slots=True)
class LatentWrite:
    """A reserved import range, held until its physical read retires."""

    product: ProductRef
    request_pool_idx: int
    page_table: tuple[int, ...]
    spans: tuple[torch.Tensor, ...]
    transfers: tuple[TransferTicket, ...] = ()
    adopted: bool = False
    released: bool = False


@dataclass(slots=True)
class LatentSource:
    """One immutable page-bank version retained by its publication registrations."""

    buffer: BufferId
    request_pool_idx: int
    bank: int
    page_table: tuple[int, ...]
    spans: tuple[torch.Tensor, ...]
    retirements: tuple[Future[None], ...] = ()
    released: bool = False


class LatentPool:
    """Own two page banks, fixed step staging, and request-indexed visibility."""

    def __init__(
        self,
        *,
        request_pool_size: int,
        num_pages: int,
        page_units: int,
        latent_width: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        """Allocate double-buffered latent pages and request-indexed progress metadata."""

        if (
            min(int(request_pool_size), int(page_units), int(latent_width)) < 1
            or int(num_pages) < 2
        ):
            raise ValueError("latent-pool geometry must contain slots, pages, and elements")
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
        self.storage = torch.zeros(
            (2, self.num_pages, self.page_units, self.latent_width),
            dtype=self.dtype,
            device=self.device,
        )
        self.step_buffer = torch.empty(
            (self.capacity_units, self.latent_width),
            dtype=self.dtype,
            device=self.device,
        )
        self.page_table_buffer = torch.empty(
            self.num_pages - 1,
            dtype=torch.int64,
            device=self.device,
        )
        self.timestep_pairs = torch.empty(
            (self.request_pool_size + 1, 2),
            dtype=torch.float32,
            device=self.device,
        )

        # Host metadata is authoritative for ownership and generation checks;
        # tensors keep it compact and cheap to query from scheduler paths.
        rows = self.request_pool_size + 1
        self._active = torch.zeros(rows, dtype=torch.int8)
        self._steps = torch.zeros(rows, dtype=torch.int32)
        self._generations = torch.zeros(rows, dtype=torch.int64)
        self._units = torch.zeros(rows, dtype=torch.int32)
        self._heights = torch.zeros(rows, dtype=torch.int32)
        self._widths = torch.zeros(rows, dtype=torch.int32)
        self._owners = torch.zeros(self.num_pages, dtype=torch.int32)
        self._slot_pages: list[tuple[int, ...]] = [() for _ in range(rows)]
        self._imports: dict[int, LatentWrite] = {}
        self._sources: dict[BufferId, LatentSource] = {}
        self._retiring_slots: set[int] = set()

        # Retain a pinned source for nonblocking page-index copies into the
        # fixed gather buffer used by one latent step at a time.
        self._page_table_host = torch.empty(
            self.num_pages - 1,
            dtype=torch.int64,
            device="cpu",
            pin_memory=self.device.type == "cuda",
        )

    @property
    def capacity_bytes(self) -> int:
        """Measure scheduler-visible latent payload capacity, excluding pool metadata."""

        return self.capacity_units * self.latent_width * self.storage.element_size()

    @property
    def persistent_bytes(self) -> int:
        """Bytes held persistently on the execution device by this pool."""

        tensors = (
            self.storage,
            self.step_buffer,
            self.page_table_buffer,
            self.timestep_pairs,
        )
        return sum(int(value.numel()) * int(value.element_size()) for value in tensors)

    def resident_byte_count(self) -> int:
        """Measure latent payload bytes owned by active request slots."""

        return int(self._units.sum().item()) * self.latent_width * self.storage.element_size()

    def stage(
        self,
        page_tables: Sequence[Sequence[int]],
        latent_units: Sequence[int],
    ) -> tuple[LatentStaging, ...]:
        """Stage exact page tables and return disjoint contiguous value views."""

        if not page_tables or len(page_tables) != len(latent_units):
            raise invalid_descriptor("latent staging columns are not aligned")
        canonical: list[tuple[int, ...]] = []
        total_pages = 0
        for page_table, units in zip(page_tables, latent_units, strict=True):
            pages = self._validate_page_table(page_table, int(units))
            canonical.append(pages)
            total_pages += len(pages)
        if total_pages > self.num_pages - 1:
            raise invalid_descriptor("latent staging exceeds the fixed step buffer")
        flattened = tuple(page for pages in canonical for page in pages)
        if len(set(flattened)) != len(flattened):
            raise invalid_descriptor("latent staging page tables overlap")
        fill_cpu_ints(self._page_table_host, flattened)
        self.page_table_buffer[:total_pages].copy_(
            self._page_table_host[:total_pages],
            non_blocking=self.device.type == "cuda",
        )
        result: list[LatentStaging] = []
        page_offset = 0
        unit_offset = 0
        for pages in canonical:
            page_count = len(pages)
            padded_units = page_count * self.page_units
            result.append(
                LatentStaging(
                    page_table=pages,
                    pages=self.page_table_buffer[page_offset : page_offset + page_count],
                    value=self.step_buffer[unit_offset : unit_offset + padded_units],
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
        """Write a transition result to the currently inactive bank."""

        slot = self._validate_slot(int(request_pool_idx))
        pages = self._validate_staging(staging, int(latent_units))
        self._require_empty(slot)
        self._require_page_owners(pages, 0)
        self._require_writable(1, pages)
        self._write_pages(1, staging.pages, staging.value)

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
        """Gather one committed page table into its fixed contiguous view."""

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
            out=staging.value.view(len(pages), self.page_units, self.latent_width),
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
        """Scatter a complete successor to pages hidden behind the inactive bank."""

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
        product: ProductRef,
        *,
        request_pool_idx: int,
        page_table: Sequence[int],
        latent_units: int,
    ) -> LatentSource:
        """Retain the written successor bank before registering its exact page spans.

        The caller attaches every transport retirement with retain_publication().
        Failure before semantic visibility must release this reservation as well
        as any physical registrations that were already created.
        """

        self._reap_sources()
        slot = self._validate_slot(request_pool_idx)
        pages = self._validate_page_table(page_table, latent_units)
        bank = 1 - int(self._active[slot].item())
        self._require_writable(bank, pages)
        return self._reserve_source(product, slot, bank, pages, latent_units)

    def reserve_current_publication(
        self,
        product: ProductRef,
        *,
        request_pool_idx: int,
        page_table: Sequence[int],
        generation: int,
        step: int,
        latent_units: int,
        height: int,
        width: int,
    ) -> LatentSource:
        """Retain an exact committed trajectory for an independently owned output.

        Publication does not change the request's generation or step. Multiple
        products may retain the same immutable bank; all must retire before a
        later step can reuse it. The caller attaches each registration's future
        with retain_publication() and releases the output on abandonment.
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
        product: ProductRef,
        slot: int,
        bank: int,
        pages: tuple[int, ...],
        latent_units: int,
    ) -> LatentSource:
        if product.buffer_id in self._sources:
            raise invalid_descriptor("latent publication generation is already registered")
        source = LatentSource(
            product.buffer_id,
            slot,
            bank,
            pages,
            tuple(
                self.storage[bank, page, : min(self.page_units, latent_units - offset)]
                for page, offset in zip(pages, range(0, latent_units, self.page_units), strict=True)
            ),
        )
        self._sources[source.buffer] = source
        return source

    def retain_publication(self, source: LatentSource, retirement: Future[None]) -> None:
        """Keep the registered page-bank version until its physical readers retire."""

        if self._sources.get(source.buffer) is not source or source.released:
            raise invalid_descriptor("latent publication reservation is no longer active")
        source.retirements = (*source.retirements, retirement)

    def release_buffers(self, buffers: Sequence[BufferId]) -> None:
        """Revoke bank reservations while retaining every pending physical publication."""

        for buffer in buffers:
            source = self._sources.get(buffer)
            if source is not None:
                source.released = True
        self._reap_sources()

    def write_dependencies(
        self, request_pool_idx: int, page_table: Sequence[int]
    ) -> tuple[Future[None], ...]:
        """Return the physical retirements that must precede reuse of the next bank."""

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
        self._reap_sources()
        selected = set(pages)
        if any(
            source.bank == bank and not selected.isdisjoint(source.page_table)
            for source in self._sources.values()
        ):
            raise resource_error("latent page bank still has a published version")

    def _reap_sources(self) -> None:
        for buffer, source in tuple(self._sources.items()):
            if not source.released or any(
                not future.done() or future.exception() is not None for future in source.retirements
            ):
                continue
            del self._sources[buffer]
        for slot in tuple(self._retiring_slots):
            if slot not in self._imports and not any(
                source.request_pool_idx == slot for source in self._sources.values()
            ):
                self._retiring_slots.remove(slot)
                self._clear_slot(slot, self._slot_pages[slot])

    def stage_timestep(
        self,
        request_pool_idx: int,
        current: float,
        following: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Place one analytical schedule pair in fixed request-indexed storage."""

        slot = self._validate_slot(int(request_pool_idx))
        row = self.timestep_pairs[slot]
        row[0].fill_(float(current))
        row[1].fill_(float(following))
        return row[:1], row[1:2]

    def validate_commit(
        self,
        publications: Sequence[LatentPublication],
        releases: Sequence[LatentRelease],
    ) -> None:
        """Validate an entire lane's visibility changes without mutation."""

        slots = (
            *(int(value.request_pool_idx) for value in publications),
            *(int(value.request_pool_idx) for value in releases),
        )
        if len(set(slots)) != len(slots):
            raise invalid_descriptor("latent commit repeats a request slot")
        claimed_pages: set[int] = set()
        for publication in publications:
            slot = self._validate_slot(int(publication.request_pool_idx))
            pages = self._validate_page_table(publication.page_table, int(publication.latent_units))
            self._validate_metadata(
                generation=int(publication.generation),
                step=int(publication.step),
                latent_units=int(publication.latent_units),
                height=int(publication.height),
                width=int(publication.width),
            )
            expected = int(publication.expected_generation)
            if expected == 0:
                if int(publication.expected_step) != 0 or int(publication.step) != 0:
                    raise invalid_descriptor("latent initialization must publish step zero")
                self._require_empty(slot)
                self._require_page_owners(pages, 0, publication_slot=slot)
            else:
                self._require_current(
                    slot,
                    step=int(publication.expected_step),
                    generation=expected,
                    latent_units=int(publication.latent_units),
                    height=int(publication.height),
                    width=int(publication.width),
                )
                self._require_slot_pages(slot, pages)
                self._require_page_owners(pages, slot)
                if int(publication.step) <= int(publication.expected_step):
                    raise invalid_descriptor("latent successor does not advance its step")
            if int(publication.generation) <= expected:
                raise invalid_descriptor("latent publication does not advance its generation")
            if not claimed_pages.isdisjoint(pages):
                raise invalid_descriptor("latent commit publications overlap physical pages")
            claimed_pages.update(pages)
        for release in releases:
            slot = self._validate_slot(int(release.request_pool_idx))
            pages = self._validate_page_table(release.page_table, int(release.latent_units))
            self._require_current(
                slot,
                step=int(release.step),
                generation=int(release.generation),
                latent_units=int(release.latent_units),
                height=int(release.height),
                width=int(release.width),
            )
            self._require_slot_pages(slot, pages)
            self._require_page_owners(pages, slot)

    def apply_commit(
        self,
        publications: Sequence[LatentPublication],
        releases: Sequence[LatentRelease],
    ) -> None:
        """Apply changes already accepted by :meth:`validate_commit`."""

        for publication in publications:
            slot = int(publication.request_pool_idx)
            pages = tuple(int(page) for page in publication.page_table)
            if int(publication.expected_generation) == 0:
                for page in pages:
                    self._owners[page] = slot
                self._slot_pages[slot] = pages
            bank = 1 - int(self._active[slot].item())
            self._active[slot] = bank
            self._steps[slot] = int(publication.step)
            self._generations[slot] = int(publication.generation)
            self._units[slot] = int(publication.latent_units)
            self._heights[slot] = int(publication.height)
            self._widths[slot] = int(publication.width)
        for release in releases:
            self._clear_slot(
                int(release.request_pool_idx), tuple(int(page) for page in release.page_table)
            )
        self._reap_imports()

    def snapshot(
        self,
        *,
        request_pool_idx: int,
        page_table: Sequence[int],
    ) -> LatentSnapshot:
        """Copy one committed trajectory to a device-independent tensor."""

        slot = self._validate_slot(int(request_pool_idx))
        generation = int(self._generations[slot].item())
        if generation < 1:
            raise invalid_descriptor("latent snapshot request has no committed trajectory")
        units = int(self._units[slot].item())
        pages = self._validate_page_table(page_table, units)
        self._require_slot_pages(slot, pages)
        self._require_page_owners(pages, slot)
        device_pages = self._device_pages(pages)
        gathered = torch.index_select(self.storage[int(self._active[slot].item())], 0, device_pages)
        value = gathered.reshape(-1, self.latent_width)[:units].detach().cpu().contiguous()
        return LatentSnapshot(
            generation=generation,
            step=int(self._steps[slot].item()),
            latent_units=units,
            height=int(self._heights[slot].item()),
            width=int(self._widths[slot].item()),
            value=value,
        )

    def restore(
        self,
        snapshot: LatentSnapshot,
        *,
        request_pool_idx: int,
        page_table: Sequence[int],
    ) -> None:
        """Install one validated snapshot directly into its committed bank."""

        slot = self._validate_slot(int(request_pool_idx))
        self._require_empty(slot)
        if int(snapshot.value.shape[1]) != self.latent_width:
            raise invalid_descriptor("latent snapshot width is incompatible with this pool")
        if snapshot.value.dtype != self.dtype:
            raise invalid_descriptor("latent snapshot dtype is incompatible with this pool")
        pages = self._validate_page_table(page_table, int(snapshot.latent_units))
        self._require_page_owners(pages, 0)
        padded_units = len(pages) * self.page_units
        target = self.step_buffer[:padded_units]
        target.zero_()
        target[: int(snapshot.latent_units)].copy_(
            snapshot.value,
            non_blocking=self.device.type == "cuda",
        )
        device_pages = self._device_pages(pages)
        self._write_pages(0, device_pages, target)
        for page in pages:
            self._owners[page] = slot
        self._slot_pages[slot] = pages
        self._active[slot] = 0
        self._steps[slot] = int(snapshot.step)
        self._generations[slot] = int(snapshot.generation)
        self._units[slot] = int(snapshot.latent_units)
        self._heights[slot] = int(snapshot.height)
        self._widths[slot] = int(snapshot.width)

    def reserve_import(
        self,
        product: ProductRef,
        *,
        request_pool_idx: int,
        page_table: Sequence[int],
        latent_units: int,
    ) -> LatentWrite:
        """Own destination pages before granting an asynchronous transfer access.

        The returned first-axis spans exclude page padding. They address bank
        zero directly and remain invisible to computation until adoption.
        """

        self._reap_imports()
        slot = self._validate_slot(int(request_pool_idx))
        self._require_empty(slot)
        pages = self._validate_page_table(page_table, int(latent_units))
        self._require_page_owners(pages, 0)
        spans = tuple(
            self.storage[0, page, : min(self.page_units, int(latent_units) - offset)]
            for page, offset in zip(
                pages, range(0, int(latent_units), self.page_units), strict=True
            )
        )
        # Padding is outside every granted span, so its initialization cannot
        # race the independent transfer stream's payload writes.
        self.storage[0, pages[-1], int(spans[-1].shape[0]) :].zero_()
        write = LatentWrite(product, slot, pages, spans)
        for page in pages:
            self._owners[page] = slot
        self._imports[slot] = write
        return write

    def retain_transfer(self, write: LatentWrite, ticket: TransferTicket) -> None:
        """Retain the physical copy even if its preparation is later abandoned."""

        self._require_import(write)
        if write.adopted:
            raise invalid_descriptor("resident latent import cannot accept another read")
        write.transfers = (*write.transfers, ticket)

    def adopt_import(
        self,
        write: LatentWrite,
        *,
        generation: int,
        step: int,
        height: int,
        width: int,
    ) -> None:
        """Expose imported pages after their ticket orders the consuming stream."""

        self._require_import(write)
        if write.adopted or not write.transfers:
            raise invalid_descriptor("latent import cannot be adopted")
        # result() establishes the producer fence on the current execution
        # stream; readiness alone does not authorize use of the page contents.
        for transfer in write.transfers:
            transfer.result()
        units = sum(int(span.shape[0]) for span in write.spans)
        self._validate_metadata(
            generation=generation, step=step, latent_units=units, height=height, width=width
        )
        if generation != write.product.generation:
            raise invalid_descriptor("latent import generation disagrees with its product")
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

    def abandon_import(self, write: LatentWrite) -> None:
        """Revoke an unadopted import without reusing a still-written page."""

        if write.released:
            return
        self._require_import(write)
        if write.adopted:
            raise invalid_descriptor("resident latent import cannot be abandoned")
        write.released = True
        for transfer in write.transfers:
            transfer.cancel()
        self._reap_imports()

    def retirement_ready(self, requests: Sequence[RequestKey]) -> bool:
        """Require known copy completion before Finish returns request pages."""

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
            write.product.request_key not in requests for write in self._imports.values()
        ) and all(source.buffer.owner not in requests for source in self._sources.values())

    def cancel_imports(self, requests: Sequence[RequestKey]) -> None:
        """Revoke unfinished admissions; resident trajectories retain execution ownership."""

        for write in tuple(self._imports.values()):
            if write.product.request_key in requests and not write.adopted and not write.released:
                self.abandon_import(write)

    def _require_import(self, write: LatentWrite) -> None:
        if self._imports.get(write.request_pool_idx) is not write or write.released:
            raise invalid_descriptor("latent import reservation is no longer writable")

    def _reap_imports(self) -> None:
        for slot, write in tuple(self._imports.items()):
            if not write.adopted and not write.released:
                continue
            if any(not transfer.retired() for transfer in write.transfers):
                continue
            del self._imports[slot]
            if write.released:
                self._clear_slot(slot, write.page_table)

    def release_slots(self, request_pool_indices: Sequence[int]) -> None:
        """Release all pages owned by exact request slots."""

        slots = tuple(self._validate_slot(int(value)) for value in request_pool_indices)
        if len(set(slots)) != len(slots):
            raise invalid_descriptor("latent release repeats a request slot")
        for slot in slots:
            pages = self._slot_pages[slot]
            self._clear_slot(slot, pages)
        self._reap_imports()

    def close(self) -> None:
        """Release storage after the owning transport has drained its physical reads."""

        self.release_buffers(tuple(self._sources))
        self.release_slots(
            tuple(
                set(self._imports) | {source.request_pool_idx for source in self._sources.values()}
            )
        )
        self._reap_sources()
        if self._imports or self._sources:
            raise resource_error("latent physical reads must retire before pool shutdown")
        for name, dtype in (
            ("storage", self.dtype),
            ("step_buffer", self.dtype),
            ("page_table_buffer", torch.int64),
            ("timestep_pairs", torch.float32),
        ):
            setattr(self, name, torch.empty(0, dtype=dtype, device=self.device))

    def _write_pages(self, bank: int, pages: torch.Tensor, value: torch.Tensor) -> None:
        """Scatter contiguous latent units into the selected physical page bank."""

        self.storage[int(bank)].index_copy_(
            0,
            pages,
            value.view(int(pages.numel()), self.page_units, self.latent_width),
        )

    def _validate_staging(self, staging: LatentStaging, latent_units: int) -> tuple[int, ...]:
        """Validate staged latent units, bank, pages, tensor shape, and dtype."""

        if staging.pages.device != self.device or staging.pages.dtype != torch.int64:
            raise invalid_descriptor("latent page table is not in fixed device staging")
        if (
            staging.value.device != self.device
            or staging.value.dtype != self.dtype
            or staging.value.ndim != 2
            or int(staging.value.shape[1]) != self.latent_width
        ):
            raise invalid_descriptor("latent value is not in fixed device staging")
        expected_pages = math.ceil(int(latent_units) / self.page_units)
        if (
            int(staging.pages.numel()) != expected_pages
            or int(staging.value.shape[0]) != expected_pages * self.page_units
        ):
            raise invalid_descriptor("latent staging does not establish its logical extent")
        return self._validate_page_table(staging.page_table, int(latent_units))

    def _validate_page_table(self, page_table: Sequence[int], latent_units: int) -> tuple[int, ...]:
        """Validate page count, bounds, uniqueness, and capacity for a latent payload."""

        units = int(latent_units)
        pages = tuple(int(page) for page in page_table)
        expected = math.ceil(units / self.page_units) if units > 0 else 0
        if (
            units < 1
            or len(pages) != expected
            or len(set(pages)) != len(pages)
            or any(page < 1 or page >= self.num_pages for page in pages)
        ):
            raise invalid_descriptor("latent page table is outside physical pool geometry")
        return pages

    def _require_page_owners(
        self,
        pages: Sequence[int] | torch.Tensor,
        owner: int,
        *,
        publication_slot: int | None = None,
    ) -> None:
        """Require every latent page to be owned by the expected request slot."""

        canonical = (
            tuple(int(value) for value in pages.detach().cpu().tolist())
            if isinstance(pages, torch.Tensor)
            else tuple(int(value) for value in pages)
        )
        self._reap_sources()
        if owner == 0 and any(
            source.request_pool_idx != publication_slot
            and not set(canonical).isdisjoint(source.page_table)
            for source in self._sources.values()
        ):
            raise invalid_descriptor("latent pages are owned by a published version")
        if any(int(self._owners[page].item()) != int(owner) for page in canonical):
            raise invalid_descriptor("latent page table is not owned by its request slot")

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
        """Validate slot ownership, generation, step, units, and raster geometry."""

        current = (
            int(self._steps[slot].item()),
            int(self._generations[slot].item()),
            int(self._units[slot].item()),
            int(self._heights[slot].item()),
            int(self._widths[slot].item()),
        )
        expected = (int(step), int(generation), int(latent_units), int(height), int(width))
        if current != expected:
            raise invalid_descriptor("latent allocation does not name the committed trajectory")

    def _require_slot_pages(self, slot: int, pages: Sequence[int]) -> None:
        """Require a slot's committed page table to match the supplied pages."""

        if self._slot_pages[slot] != tuple(int(page) for page in pages):
            raise invalid_descriptor("latent page table does not match its committed trajectory")

    def _require_empty(self, slot: int) -> None:
        """Require a request slot to have no active latent trajectory."""

        if (
            slot in self._imports
            or int(self._generations[slot].item()) != 0
            or int(self._units[slot].item()) != 0
        ):
            raise invalid_descriptor("request slot already owns a committed trajectory")

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

        if min(int(generation), int(latent_units), int(height), int(width)) < 1 or int(step) < 0:
            raise invalid_descriptor("latent publication metadata is invalid")
        if int(latent_units) > self.capacity_units:
            raise invalid_descriptor("latent publication exceeds physical capacity")

    def _clear_slot(self, slot: int, pages: Sequence[int]) -> None:
        """Release latent page ownership and reset all metadata for one slot."""

        write = self._imports.get(slot)
        if write is not None:
            write.released = True
            if not write.adopted:
                for transfer in write.transfers:
                    transfer.cancel()
        sources = tuple(
            source for source in self._sources.values() if source.request_pool_idx == slot
        )
        if write is not None or sources:
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
        """Validate a scheduler-visible latent request slot index."""

        if slot < 1 or slot > self.request_pool_size:
            raise invalid_descriptor("latent request slot is outside physical capacity")
        return slot

    def _device_pages(self, pages: Sequence[int]) -> torch.Tensor:
        """Copy host page identifiers into reusable device index storage."""

        fill_cpu_ints(self._page_table_host, pages)
        target = self.page_table_buffer[: len(pages)]
        target.copy_(self._page_table_host[: len(pages)], non_blocking=self.device.type == "cuda")
        return target


__all__ = [
    "LatentPool",
    "LatentPublication",
    "LatentRelease",
    "LatentSnapshot",
    "LatentStaging",
    "LatentWrite",
]
