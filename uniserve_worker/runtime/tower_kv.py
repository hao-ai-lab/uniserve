"""Tower-axis KV snapshot: ``reshard(Pinned(tower, primary) -> Pinned(tower, gen))``.

The understanding/text tower owns the authoritative conditioning KV; the
generation tower reads a *snapshot* of it once per image. This module realizes
that snapshot as the paged-KV form of a ``Pinned -> Pinned`` reshard over the
``tower`` axis transport:

* the per-element device move is the transport's ``copy_to`` (in-process NVLink
  peer copy or a data-plane publish/fetch),
* readiness barriers are the transport's ``record_ready``/``wait_ready`` keyed by
  coordinate, so the gen tower never reads partially-written und KV nor an
  in-flight copy,
* the destination is a *writable* paged cache with room to append the transient
  denoise K/V, allocated from the gen-tower KV residency.

A trivial tower (no transport, or src==dst coordinate) degrades to a same-device
copy into the scratch pool.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..foundation.sizing import ceil_div
from .paged_text_cache import PagedTextCache

__all__ = [
    "reshard_kv_snapshot",
    "wait_kv_snapshot_ready",
]

# Event recorded on the gen coordinate after the snapshot copy (B2); the consumer
# waits it before its first read of the replica.
_GEN_READY_ATTR = "_uniserve_gen_ready_event"


def reshard_kv_snapshot(
    cache: Any,
    *,
    target_pool: Any,
    allocate_blocks: Callable[[int], list[int]],
    num_layers: int,
    block_size: int,
    target_device: Any,
    transport: Any | None = None,
    src_coord: int = 0,
    dst_coord: int = 0,
) -> Any:
    """Snapshot ``cache``'s ``[0, length)`` KV prefix into a fresh writable cache.

    Realizes ``reshard(KV, Pinned(tower, src_coord) -> Pinned(tower, dst_coord))``
    for a paged KV span: every layer's prefix is copied from ``cache``'s pool into
    a newly allocated ``target_pool`` cache (leaving room to append transient
    denoise K/V). When a ``transport`` is supplied and the coordinates differ, the
    move goes through ``transport.copy_to`` with copy-start and copy-complete
    barriers; otherwise it is a same-device copy (trivial tower). Returns the
    destination cache (with the readiness event attached when cross-coordinate).
    """
    if cache is None:
        return cache
    source_pool = getattr(cache, "pool", None)
    source_blocks = list(getattr(cache, "block_ids", []) or [])
    length = int(cache.get_seq_length())
    initial_blocks = ceil_div(length, block_size)
    block_ids = allocate_blocks(initial_blocks)
    out = PagedTextCache(
        target_pool,
        block_ids,
        num_layers=num_layers,
        length=length,
        allocate_blocks=allocate_blocks,
    )

    active_transport = (
        transport if transport is not None and int(src_coord) != int(dst_coord) else None
    )
    if active_transport is not None:
        # B1: the gen coordinate's stream waits for the primary's pending KV
        # writes so the copy never reads partially-written conditioning KV.
        b1 = active_transport.record_ready(int(src_coord))
        active_transport.wait_ready(b1, int(dst_coord))
    if source_pool is not None and length > 0:
        for layer_idx in range(int(num_layers)):
            k, v = source_pool.read(layer_idx, source_blocks, start=0, length=length)
            if k is None or v is None:
                continue
            if active_transport is not None:
                k = active_transport.copy_to(k, coord=int(dst_coord))
                v = active_transport.copy_to(v, coord=int(dst_coord))
            else:
                k = k.to(target_device, non_blocking=True)
                v = v.to(target_device, non_blocking=True)
            target_pool.write(layer_idx, out.block_ids, start=0, k=k, v=v)
    if active_transport is not None:
        # B2: record copy completion on the gen coordinate; the consumer waits it.
        done = active_transport.record_ready(int(dst_coord))
        if done is not None:
            setattr(out, _GEN_READY_ATTR, done)
    return out


def wait_kv_snapshot_ready(
    cache: Any,
    *,
    transport: Any | None = None,
    coord: int = 0,
) -> None:
    """Wait the snapshot's B2 readiness event before the gen tower reads it."""
    event = getattr(cache, _GEN_READY_ATTR, None)
    if event is None:
        return
    if transport is not None:
        transport.wait_ready(event, int(coord))
