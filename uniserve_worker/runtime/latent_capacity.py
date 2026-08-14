"""Exact byte geometry for the fixed paged latent owner."""

from __future__ import annotations


def latent_trajectory_bytes(
    latent_units: int,
    latent_width: int,
    dtype_bytes: int,
) -> int:
    """Bytes in one logical trajectory extent."""

    units = int(latent_units)
    width = int(latent_width)
    element_bytes = int(dtype_bytes)
    if units < 0 or width < 1 or element_bytes < 1:
        raise ValueError("latent trajectory geometry is invalid")
    return units * width * element_bytes


def latent_pool_capacity_bytes(
    *,
    request_pool_size: int,
    num_pages: int,
    page_units: int,
    latent_width: int,
    dtype_bytes: int,
) -> int:
    """Persistent device bytes owned by one fixed two-bank latent pool."""

    slots = int(request_pool_size)
    pages = int(num_pages)
    units = int(page_units)
    width = int(latent_width)
    element_bytes = int(dtype_bytes)
    if min(slots, units, width, element_bytes) < 1 or pages < 2:
        raise ValueError("latent pool geometry is invalid")
    usable_pages = pages - 1
    storage = 2 * pages * units * width * element_bytes
    step_buffer = usable_pages * units * width * element_bytes
    page_table = usable_pages * 8
    timestep_pairs = (slots + 1) * 2 * 4
    return storage + step_buffer + page_table + timestep_pairs


__all__ = ["latent_pool_capacity_bytes", "latent_trajectory_bytes"]
