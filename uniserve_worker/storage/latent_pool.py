"""Numerical storage and views for the native latent pool.

The pool owns page assignments, trajectory visibility, imports and retirement.
These routines allocate fixed tensor backing and gather or scatter its values
on the calling stream.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager

import torch

from uniserve.runtime.device import fill_cpu_ints
from uniserve_worker._uniserve_ipc import (
    LatentBuffer,
    LatentExport,
    LatentImport,
    LatentPool,
    LatentUpdate,
)
from uniserve_worker.storage.host_buffers import HostBuffers


def _allocate(
    request_pool_size: int,
    num_pages: int,
    page_units: int,
    latent_width: int,
    dtype: torch.dtype,
    device: torch.device,
    with_workspace: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, HostBuffers]:
    capacity_units = (num_pages - 1) * page_units
    # Bank zero and bank one alternate across steps; page zero is padding.
    storage = torch.zeros(
        (2, num_pages, page_units, latent_width), dtype=dtype, device=device
    )
    step_buffer = torch.empty(
        (capacity_units if with_workspace else 0, latent_width),
        dtype=dtype,
        device=device,
    )
    page_table_buffer = torch.empty(
        num_pages - 1, dtype=torch.int64, device=device
    )
    timesteps = torch.empty(
        (request_pool_size + 1, 1), dtype=torch.float32, device=device
    )
    page_host = HostBuffers(
        num_pages - 1, dtype=torch.int64, depth=1, device=device
    )
    return storage, step_buffer, page_table_buffer, timesteps, page_host


@contextmanager
def _startup_values(pool: LatentPool, rows: int, units: int):
    views = pool._startup_buffers(rows, units)
    try:
        yield views
    finally:
        # The first admitted call can immediately borrow this scratch again.
        if pool.device.type == "cuda":
            torch.cuda.current_stream(pool.device).synchronize()


def _copy_pages(
    pages: Sequence[int],
    page_offset: int,
    page_table_buffer: torch.Tensor,
    page_host: HostBuffers,
) -> None:
    slot, host = page_host.acquire()
    fill_cpu_ints(host, pages)
    page_table_buffer[page_offset : page_offset + len(pages)].copy_(
        host[: len(pages)], non_blocking=page_table_buffer.is_cuda
    )
    page_host.record_copy(slot)


def _gather(
    storage: torch.Tensor,
    bank: int,
    pages: torch.Tensor,
    value: torch.Tensor,
    units: int,
) -> torch.Tensor:
    torch.index_select(
        storage[bank], 0, pages, out=value.view(-1, *storage.shape[-2:])
    )
    return value[:units]


def _scatter(
    storage: torch.Tensor, bank: int, pages: torch.Tensor, value: torch.Tensor
) -> None:
    storage[bank].index_copy_(0, pages, value.view(-1, *storage.shape[-2:]))


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
    "LatentBuffer",
    "LatentImport",
    "LatentExport",
    "LatentUpdate",
]
