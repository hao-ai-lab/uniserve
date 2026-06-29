"""Worker-side data plane, Tier 1: the per-worker handle↔tensor registry over a
single register-once :class:`Transport`.

There is one :class:`TensorStore` per worker — it *is* the worker's data plane.
``publish`` registers a producer tensor with the worker's transport (once) and
mints a globally-unique handle id (the control-plane reference);
``locator_of`` returns the compact wire descriptor the host routes to a consumer;
``fetch`` resolves a handle (or a raw locator from another worker) to a local
tensor over the transport. The default transport is in-process (zero copy); a
cross-process transport (shm / cuda_ipc / mooncake) moves real bytes between
workers spanning devices/nodes.
"""
from __future__ import annotations

import itertools
import threading
from enum import StrEnum
from typing import TYPE_CHECKING

from ..foundation.errors import invalid_descriptor
from .transfer import LocalTransport, Locator, Transport

if TYPE_CHECKING:
    import torch

__all__ = ["TensorKind", "TensorStore"]

class TensorKind(StrEnum):
    """Tensor semantic classes, mirroring Rust ``executor::TensorKind``."""

    EMBEDDING = "embedding"
    KV_PAGES = "kv_pages"
    LOGITS = "logits"
    IMAGE = "image"
    VIDEO_FRAME = "video_frame"


class TensorStore:
    """Per-worker Tier-1 data plane over one register-once :class:`Transport`."""

    def __init__(self, *, id_base: int = 0, transport: Transport | None = None) -> None:
        # Handle ids are globally unique across the data plane. Each worker carves
        # a disjoint range via ``id_base``; the low bits are a per-store counter.
        self._counter = itertools.count(1)
        self._id_base = int(id_base)
        self.transport: Transport = transport if transport is not None else LocalTransport()
        self._store: dict[int, tuple[TensorKind, Locator]] = {}
        self._lock = threading.Lock()

    def session(self) -> str:
        return self.transport.session()

    def publish(self, tensor: "torch.Tensor", kind: TensorKind | str = TensorKind.LOGITS) -> int:
        """Register a tensor with the transport (once) and return its handle id."""
        locator = self.transport.publish(tensor)
        handle = self._id_base + next(self._counter)
        with self._lock:
            self._store[handle] = (TensorKind(str(kind)), locator)
        return handle

    def fetch(self, handle: int) -> "torch.Tensor":
        """Return the tensor a locally-published handle refers to."""
        with self._lock:
            entry = self._store.get(int(handle))
        if entry is None:
            raise invalid_descriptor(f"tensor handle {handle} is not registered (released or never published)")
        return self.transport.fetch(entry[1])

    def fetch_locator(self, locator: "bytes | Locator") -> "torch.Tensor":
        """Read-driven pull from a producer's locator (consumer side: the handle
        was minted on another worker and only the locator crossed the wire)."""
        loc = locator if isinstance(locator, Locator) else Locator.from_bytes(locator)
        return self.transport.fetch(loc)

    def push(self, tensor: "torch.Tensor", locator: "bytes | Locator") -> None:
        """Write-driven push into the remote buffer ``locator`` names (KV edge)."""
        loc = locator if isinstance(locator, Locator) else Locator.from_bytes(locator)
        self.transport.push(tensor, loc)

    def locator_of(self, handle: int) -> bytes | None:
        """The wire-ready opaque locator bytes for a handle, or ``None``."""
        with self._lock:
            entry = self._store.get(int(handle))
        return entry[1].to_bytes() if entry is not None else None

    def kind_of(self, handle: int) -> TensorKind | None:
        with self._lock:
            entry = self._store.get(int(handle))
        return entry[0] if entry is not None else None

    def release(self, handle: int) -> None:
        """Reclaim a handle's buffer; idempotent."""
        with self._lock:
            entry = self._store.pop(int(handle), None)
        if entry is not None:
            self.transport.release(entry[1])

    def release_many(self, handles: "list[int] | tuple[int, ...]") -> None:
        for handle in handles:
            self.release(handle)

    def close(self) -> None:
        self.transport.close()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __contains__(self, handle: int) -> bool:
        with self._lock:
            return int(handle) in self._store
