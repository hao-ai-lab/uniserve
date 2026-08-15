"""Fixed physical storage for scheduler-placed generation trajectories."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .device import fill_cpu_ints
from ..foundation.errors import invalid_descriptor


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
    """Validated visibility change applied at the partition commit point."""

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
    """Validated trajectory release applied at the partition commit point."""

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
        rows = self.request_pool_size + 1
        self._active = torch.zeros(rows, dtype=torch.int8)
        self._steps = torch.zeros(rows, dtype=torch.int32)
        self._generations = torch.zeros(rows, dtype=torch.int64)
        self._units = torch.zeros(rows, dtype=torch.int32)
        self._heights = torch.zeros(rows, dtype=torch.int32)
        self._widths = torch.zeros(rows, dtype=torch.int32)
        self._owners = torch.zeros(self.num_pages, dtype=torch.int32)
        self._slot_pages: list[tuple[int, ...]] = [() for _ in range(rows)]
        self._page_table_host = torch.empty(
            self.num_pages - 1,
            dtype=torch.int64,
            device="cpu",
            pin_memory=self.device.type == "cuda",
        )

    @property
    def capacity_bytes(self) -> int:
        """Logical scheduler-visible trajectory capacity."""

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
        self._write_pages(1 - int(self._active[slot].item()), staging.pages, staging.value)

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
        """Validate an entire partition's visibility changes without mutation."""

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
                self._require_page_owners(pages, 0)
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

    def release_slots(self, request_pool_indices: Sequence[int]) -> None:
        """Release all pages owned by exact request slots."""

        slots = tuple(self._validate_slot(int(value)) for value in request_pool_indices)
        if len(set(slots)) != len(slots):
            raise invalid_descriptor("latent release repeats a request slot")
        for slot in slots:
            pages = self._slot_pages[slot]
            self._clear_slot(slot, pages)

    def close(self) -> None:
        for name, dtype in (
            ("storage", self.dtype),
            ("step_buffer", self.dtype),
            ("page_table_buffer", torch.int64),
            ("timestep_pairs", torch.float32),
        ):
            setattr(self, name, torch.empty(0, dtype=dtype, device=self.device))

    def _write_pages(self, bank: int, pages: torch.Tensor, value: torch.Tensor) -> None:
        self.storage[int(bank)].index_copy_(
            0,
            pages,
            value.view(int(pages.numel()), self.page_units, self.latent_width),
        )

    def _validate_staging(self, staging: LatentStaging, latent_units: int) -> tuple[int, ...]:
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

    def _require_page_owners(self, pages: Sequence[int] | torch.Tensor, owner: int) -> None:
        canonical = (
            tuple(int(value) for value in pages.detach().cpu().tolist())
            if isinstance(pages, torch.Tensor)
            else tuple(int(value) for value in pages)
        )
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
        current = (
            int(self._steps[slot].item()),
            int(self._generations[slot].item()),
            int(self._units[slot].item()),
            int(self._heights[slot].item()),
            int(self._widths[slot].item()),
        )
        expected = (int(step), int(generation), int(latent_units), int(height), int(width))
        if current != expected:
            raise invalid_descriptor("latent placement does not name the committed trajectory")

    def _require_slot_pages(self, slot: int, pages: Sequence[int]) -> None:
        if self._slot_pages[slot] != tuple(int(page) for page in pages):
            raise invalid_descriptor("latent page table does not match its committed trajectory")

    def _require_empty(self, slot: int) -> None:
        if int(self._generations[slot].item()) != 0 or int(self._units[slot].item()) != 0:
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
        if min(int(generation), int(latent_units), int(height), int(width)) < 1 or int(step) < 0:
            raise invalid_descriptor("latent publication metadata is invalid")
        if int(latent_units) > self.capacity_units:
            raise invalid_descriptor("latent publication exceeds physical capacity")

    def _clear_slot(self, slot: int, pages: Sequence[int]) -> None:
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
        if slot < 1 or slot > self.request_pool_size:
            raise invalid_descriptor("latent request slot is outside physical capacity")
        return slot

    def _device_pages(self, pages: Sequence[int]) -> torch.Tensor:
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
]
