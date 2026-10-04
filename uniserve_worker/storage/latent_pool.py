"""Numerical storage and views for the native latent pool.

The pool owns page assignments, trajectory visibility, imports and retirement.
These routines allocate fixed tensor backing and gather or scatter its values
on the calling stream.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from uniserve.runtime.device import fill_cpu_ints
from uniserve_worker._uniserve_ipc import (
    LatentExport,
    LatentImport,
    LatentPool,
    LatentUpdate,
)
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.storage.host_buffers import HostBuffers


@dataclass(frozen=True, slots=True)
class LatentStaging:
    """A fixed page-index view and its contiguous, padded latent values."""

    page_table: tuple[int, ...]
    # [len(page_table)] int64, in the pool's device page-index buffer.
    pages: torch.Tensor
    # [len(page_table) * page_units, latent_width], with final-page padding.
    value: torch.Tensor


def _allocate(
    request_pool_size: int,
    num_pages: int,
    page_units: int,
    latent_width: int,
    dtype: torch.dtype,
    device: torch.device,
    staging: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, HostBuffers]:
    capacity_units = (num_pages - 1) * page_units
    # Bank zero and bank one alternate across steps; page zero is padding.
    storage = torch.zeros(
        (2, num_pages, page_units, latent_width), dtype=dtype, device=device
    )
    step_buffer = torch.empty(
        (capacity_units if staging else 0, latent_width),
        dtype=dtype,
        device=device,
    )
    page_table_buffer = torch.empty(
        num_pages - 1, dtype=torch.int64, device=device
    )
    timesteps = torch.empty(
        (request_pool_size + 1, 1), dtype=torch.float32, device=device
    )
    page_staging = HostBuffers(
        num_pages - 1, dtype=torch.int64, depth=1, device=device
    )
    return storage, step_buffer, page_table_buffer, timesteps, page_staging


@contextmanager
def _startup_values(pool: LatentPool, rows: int, units: int):
    views = pool._startup_staging(rows, units)
    try:
        yield views
    finally:
        # The first admitted call can immediately borrow this scratch again.
        if pool.device.type == "cuda":
            torch.cuda.current_stream(pool.device).synchronize()


def _stage(
    page_tables: Sequence[Sequence[int]],
    page_offset: int,
    page_units: int,
    page_table_buffer: torch.Tensor,
    step_buffer: torch.Tensor,
    page_staging: HostBuffers,
) -> tuple[LatentStaging, ...]:
    pages = tuple(page for table in page_tables for page in table)
    slot, host = page_staging.acquire()
    fill_cpu_ints(host, pages)
    page_table_buffer[page_offset : page_offset + len(pages)].copy_(
        host[: len(pages)], non_blocking=page_table_buffer.is_cuda
    )
    page_staging.record_copy(slot)

    result = []
    for table in page_tables:
        end = page_offset + len(table)
        result.append(
            LatentStaging(
                tuple(table),
                page_table_buffer[page_offset:end],
                step_buffer[page_offset * page_units : end * page_units],
            )
        )
        page_offset = end
    return tuple(result)


def _check_staging(
    staging: LatentStaging, units: int, storage: torch.Tensor
) -> None:
    page_units, width = storage.shape[-2:]
    pages = (units + page_units - 1) // page_units
    if (
        staging.pages.device != storage.device
        or staging.pages.dtype != torch.int64
    ):
        raise invalid_descriptor(
            "latent page table is not in fixed device staging"
        )
    if (
        staging.value.device != storage.device
        or staging.value.dtype != storage.dtype
        or staging.value.ndim != 2
        or staging.value.shape[1] != width
    ):
        raise invalid_descriptor("latent value is not in fixed device staging")
    if (
        staging.pages.numel() != pages
        or staging.value.shape[0] != pages * page_units
    ):
        raise invalid_descriptor(
            "latent staging does not establish its logical extent"
        )


def _gather(
    storage: torch.Tensor, bank: int, staging: LatentStaging, units: int
) -> torch.Tensor:
    torch.index_select(
        storage[bank],
        0,
        staging.pages,
        out=staging.value.view(len(staging.page_table), *storage.shape[-2:]),
    )
    return staging.value[:units]


def _scatter(storage: torch.Tensor, bank: int, staging: LatentStaging) -> None:
    storage[bank].index_copy_(
        0,
        staging.pages,
        staging.value.view(len(staging.page_table), *storage.shape[-2:]),
    )


def _spans(
    storage: torch.Tensor, bank: int, pages: Sequence[int], units: int
) -> tuple[torch.Tensor, ...]:
    page_units = storage.shape[2]
    return tuple(
        storage[bank, page, : min(page_units, units - index * page_units)]
        for index, page in enumerate(pages)
    )


def _empty_storage(dtype: torch.dtype, device: torch.device):
    return (
        torch.empty(0, dtype=dtype, device=device),
        torch.empty(0, dtype=dtype, device=device),
        torch.empty(0, dtype=torch.int64, device=device),
        torch.empty(0, dtype=torch.float32, device=device),
    )


__all__ = [
    "LatentPool",
    "LatentStaging",
    "LatentImport",
    "LatentExport",
    "LatentUpdate",
]
