"""Data plane, Tier 2: the pluggable, register-once byte transport.

A :class:`Transport` is one per worker and follows the register-once,
reference-by-endpoint model. A producer registers a buffer once and hands out a
typed :class:`Locator` describing a bounded region; a consumer resolves that
locator over the selected transport.

Backends (one chosen per worker via :func:`make_transport`):

* ``local``    — same process, zero copy.
* ``shm``      — same node, host bytes via POSIX shared memory.
* ``cuda_ipc`` — same node, GPU↔GPU via CUDA IPC (torch's per-storage handle
  cache *is* register-once).
"""

from __future__ import annotations

import concurrent.futures
import ctypes
import errno
import mmap
import os
import queue
import selectors
import socket
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ..execution.batch import (
    CudaIpcTransfer,
    DeviceProductTransferValue,
    EncoderTransferValue,
    KvTransferValue,
    LatentTransferValue,
    LocalTransfer,
    PosixShmTransfer,
    ProductKind,
    TransferHandle,
    TransferKind,
    TransferLocator,
)
from ..foundation.errors import invalid_descriptor, resource_error, unsupported_setup

if TYPE_CHECKING:
    import torch

__all__ = [
    "Locator",
    "Transport",
    "TransferTicket",
    "LocalTransport",
    "ShmTransport",
    "CudaIpcTransport",
    "fetch_locator",
    "make_transport",
    "TransportKind",
    "TRANSPORTS",
    "decode_transfer_handle",
    "encode_transfer_handle",
]


class TransportKind(StrEnum):
    LOCAL = "local"
    SHM = "shm"
    CUDA_IPC = "cuda_ipc"


TRANSPORTS = tuple(kind.value for kind in TransportKind)


@dataclass(frozen=True)
class Locator:
    """Compact IPC reference into a registered region.

    Carried opaquely as an exact generation-tagged product reference from the
    producing operation, through the control plane, to the consuming operation,
    and resolved only by the consumer's transport.
    """

    transport: str
    endpoint: str
    nbytes: int
    dtype: str
    shape: tuple[int, ...]
    device: str
    handle: bytes = b""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        if self.transport == "local":
            transport = LocalTransfer(endpoint=self.endpoint, key=int(self.handle.decode()))
        elif self.transport == "shm":
            semaphore = self.meta.get("ready_semaphore")
            transport = PosixShmTransfer(
                name=self.handle.decode(),
                ready_header_bytes=int(self.meta.get("ready_header_bytes", 0)),
                ready_semaphore=None if semaphore is None else str(semaphore),
            )
        elif self.transport == "cuda_ipc":
            transport = CudaIpcTransfer(
                endpoint=self.endpoint,
                publication_id=str(self.meta["publication_id"]),
                storage_handle=self.handle,
                storage_size_bytes=int(self.meta["storage_size_bytes"]),
                storage_offset_bytes=int(self.meta["storage_offset_bytes"]),
                tensor_offset=int(self.meta["tensor_offset"]),
                tensor_stride=tuple(int(value) for value in self.meta["tensor_stride"]),
                ref_counter_handle=bytes(self.meta["ref_counter_handle"]),
                ref_counter_offset=int(self.meta["ref_counter_offset"]),
                event_handle=bytes(self.meta["event_handle"]),
                event_sync_required=bool(self.meta["event_sync_required"]),
                ready_event_handle=bytes(self.meta["ready_event_handle"]),
            )
        else:
            raise invalid_descriptor(f"unsupported locator transport {self.transport!r}")
        return TransferLocator(
            transport=transport,
            nbytes=self.nbytes,
            dtype=self.dtype,
            shape=self.shape,
            device=self.device,
        ).to_mapping()

    @staticmethod
    def from_mapping(raw: dict[str, Any]) -> "Locator":
        locator = TransferLocator.from_mapping(raw)
        transport = locator.transport
        if isinstance(transport, LocalTransfer):
            name = "local"
            endpoint = transport.endpoint
            handle = str(transport.key).encode()
            meta: dict[str, Any] = {}
        elif isinstance(transport, PosixShmTransfer):
            name = "shm"
            endpoint = "shm"
            handle = transport.name.encode()
            meta = {
                "ready_header_bytes": transport.ready_header_bytes,
                "ready_semaphore": transport.ready_semaphore,
            }
        else:
            name = "cuda_ipc"
            endpoint = transport.endpoint
            handle = transport.storage_handle
            meta = {
                "publication_id": transport.publication_id,
                "storage_size_bytes": transport.storage_size_bytes,
                "storage_offset_bytes": transport.storage_offset_bytes,
                "tensor_offset": transport.tensor_offset,
                "tensor_stride": transport.tensor_stride,
                "ref_counter_handle": transport.ref_counter_handle,
                "ref_counter_offset": transport.ref_counter_offset,
                "event_handle": transport.event_handle,
                "event_sync_required": transport.event_sync_required,
                "ready_event_handle": transport.ready_event_handle,
            }
        return Locator(
            transport=name,
            endpoint=endpoint,
            nbytes=locator.nbytes,
            dtype=locator.dtype,
            shape=locator.shape,
            device=locator.device,
            handle=handle,
            meta=meta,
        )


def encode_transfer_handle(
    kind: str,
    value: dict[str, object],
) -> TransferHandle:
    try:
        transfer_kind = TransferKind(kind)
    except ValueError:
        raise invalid_descriptor("transport entry kind is invalid")
    generation = int(value.get("generation", 0))
    if transfer_kind is TransferKind.ENCODER:
        typed = EncoderTransferValue(
            generation=generation,
            height=int(value["height"]),
            width=int(value["width"]),
            payload_kind=ProductKind(str(value["payload_kind"])).value,
            locator=TransferLocator.from_mapping(value["locator"]),
        )
    elif transfer_kind is TransferKind.DEVICE_PRODUCT:
        typed = DeviceProductTransferValue(
            generation=generation,
            height=int(value.get("height", 0)),
            width=int(value.get("width", 0)),
            value_range=str(value.get("value_range", "")),
            locator=TransferLocator.from_mapping(value["locator"]),
        )
    elif transfer_kind is TransferKind.LATENT:
        typed = LatentTransferValue(
            generation=generation,
            height=int(value["height"]),
            width=int(value["width"]),
            latent_units=int(value["latent_units"]),
            step=int(value["step"]),
            locator=TransferLocator.from_mapping(value["locator"]),
        )
    else:
        snapshot = value.get("snapshot")
        if not isinstance(snapshot, dict):
            raise invalid_descriptor("KV transfer snapshot is invalid")
        raw_locators = snapshot.get("locators")
        if not isinstance(raw_locators, (list, tuple)):
            raise invalid_descriptor("KV transfer locators are invalid")
        from ..execution.batch import Checkpoint

        raw_base = snapshot.get("base_version")
        typed = KvTransferValue(
            generation=generation,
            locators=tuple(TransferLocator.from_mapping(item) for item in raw_locators),
            source=Checkpoint.from_mapping(snapshot.get("source_version")),
            destination=str(snapshot.get("destination", "")),
            base=None if raw_base is None else Checkpoint.from_mapping(raw_base),
            base_extent=int(snapshot.get("base_extent", 0)),
            published_extent=int(snapshot.get("published_extent", 0)),
            group_id=int(snapshot.get("group_id", 0)),
            scale_identity=str(snapshot.get("scale_identity", "")),
        )
    return TransferHandle(value=typed)


def decode_transfer_handle(handle: TransferHandle) -> tuple[str, dict[str, object]]:
    typed = handle.value
    if isinstance(typed, EncoderTransferValue):
        value: dict[str, object] = {
            "generation": typed.generation,
            "height": typed.height,
            "width": typed.width,
            "payload_kind": typed.payload_kind,
            "locator": typed.locator.to_mapping(),
        }
    elif isinstance(typed, DeviceProductTransferValue):
        value = {
            "generation": typed.generation,
            "height": typed.height,
            "width": typed.width,
            "value_range": typed.value_range,
            "locator": typed.locator.to_mapping(),
        }
    elif isinstance(typed, LatentTransferValue):
        value = {
            "generation": typed.generation,
            "height": typed.height,
            "width": typed.width,
            "latent_units": typed.latent_units,
            "step": typed.step,
            "locator": typed.locator.to_mapping(),
        }
    else:
        value = {
            "generation": typed.generation,
            "snapshot": {
                "locators": [locator.to_mapping() for locator in typed.locators],
                "source_version": typed.source.to_mapping(),
                "destination": typed.destination,
                "base_version": None if typed.base is None else typed.base.to_mapping(),
                "base_extent": typed.base_extent,
                "published_extent": typed.published_extent,
                "group_id": typed.group_id,
                "scale_identity": typed.scale_identity,
            },
        }
    return handle.kind.value, value


def fetch_locator(transport: "Transport", locator: Locator) -> "torch.Tensor":
    """Resolve a live transport locator."""

    return transport.fetch(locator)


def _dtype_to_str(dtype: "torch.dtype") -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_str(name: str) -> "torch.dtype":
    import torch

    return getattr(torch, name)


def _nbytes(tensor: "torch.Tensor") -> int:
    return int(tensor.numel() * tensor.element_size())


class Transport(ABC):
    """Register-once one-sided transport. One instance per worker."""

    name: str = "transport"
    supports_async_publication: bool = False
    #: Whether a synchronous :meth:`fetch` observes producer completion by blocking
    #: the calling thread. Request threads must never resolve such a transport
    #: synchronously; they submit :meth:`fetch_async` tickets gated by :meth:`ready`.
    blocking_fetch: bool = False

    def endpoint(self) -> str:
        """This worker's stable transport endpoint identity."""
        return self.name

    @abstractmethod
    def publish(self, tensor: "torch.Tensor") -> Locator:
        """Register the tensor's buffer (once) and return a locator for it."""

    def publish_async(self, tensor: "torch.Tensor") -> Locator:
        """Enqueue publication without observing device completion on the caller."""

        if not self.supports_async_publication:
            raise unsupported_setup(
                f"{self.name} transport does not support asynchronous publication"
            )
        return self.publish(tensor)

    @abstractmethod
    def fetch(self, locator: Locator) -> "torch.Tensor":
        """Read-driven: materialize the located tensor on this worker."""

    def fetch_async(self, locator: Locator) -> "TransferTicket":
        """Submit a read without waiting for remote or device progress."""

        raise unsupported_setup(f"{self.name} transport does not support asynchronous reads")

    def ready(self, locator: Locator) -> bool:
        """Query producer readiness without waiting."""

        return True

    def release(self, locator: Locator) -> None:
        """Deregister the producer buffer behind ``locator`` (idempotent)."""

    def close(self) -> None:
        """Tear down the transport (engine, segments)."""

    def set_completion_wake(self, wake: Any) -> None:
        """Connect asynchronous ticket completion to the worker controller."""


class TransferTicket(ABC):
    """One bounded transfer whose readiness is query-only on request threads."""

    @abstractmethod
    def ready(self) -> bool:
        """Return whether the result can be obtained without waiting."""

    @abstractmethod
    def result(self) -> "torch.Tensor":
        """Return the completed value, rejecting observation before readiness."""

    @abstractmethod
    def add_done_callback(self, callback: Any) -> None:
        """Schedule a non-blocking owner notification after completion."""


class _ImmediateTransferTicket(TransferTicket):
    def __init__(self, value: "torch.Tensor") -> None:
        self._value = value

    def ready(self) -> bool:
        return True

    def result(self) -> "torch.Tensor":
        return self._value

    def add_done_callback(self, callback: Any) -> None:
        callback()


class _FutureTransferTicket(TransferTicket):
    def __init__(self, future: "concurrent.futures.Future[torch.Tensor]") -> None:
        self._future = future

    def ready(self) -> bool:
        return self._future.done()

    def result(self) -> "torch.Tensor":
        if not self.ready():
            raise RuntimeError("transfer ticket was observed before readiness")
        return self._future.result()

    def add_done_callback(self, callback: Any) -> None:
        self._future.add_done_callback(lambda _future: callback())


class _ByteCapacity:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("transfer byte capacity must be positive")
        self.used = 0
        self._lock = threading.Lock()

    def acquire(self, amount: int) -> None:
        value = int(amount)
        if value < 0:
            raise ValueError("transfer byte reservation must not be negative")
        with self._lock:
            projected = self.used + value
            if projected > self.capacity:
                raise resource_error(
                    f"transfer byte capacity is exhausted ({projected}>{self.capacity})"
                )
            self.used = projected

    def release(self, amount: int) -> None:
        value = int(amount)
        with self._lock:
            if value < 0 or value > self.used:
                raise RuntimeError("transfer byte release exceeds the live reservation")
            self.used -= value


class _NamedSemaphore:
    """Process-shared completion signal for a POSIX shared-memory publication."""

    _libc = ctypes.CDLL(None, use_errno=True)
    _libc.sem_open.restype = ctypes.c_void_p
    _failed = ctypes.c_void_p(-1).value

    def __init__(self, name: str, handle: int) -> None:
        self.name = name
        self._handle = ctypes.c_void_p(handle)
        self._closed = False

    @classmethod
    def create(cls) -> "_NamedSemaphore":
        name = f"/uniserve-{uuid.uuid4().hex}"
        handle = cls._libc.sem_open(
            name.encode(),
            os.O_CREAT | os.O_EXCL,
            0o600,
            0,
        )
        if handle == cls._failed:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), name)
        return cls(name, int(handle))

    @classmethod
    def open(cls, name: str) -> "_NamedSemaphore":
        handle = cls._libc.sem_open(name.encode(), 0, 0, 0)
        if handle == cls._failed:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), name)
        return cls(name, int(handle))

    def wait(self) -> None:
        while self._libc.sem_wait(self._handle) != 0:
            error = ctypes.get_errno()
            if error != errno.EINTR:
                raise OSError(error, os.strerror(error), self.name)

    def post(self) -> None:
        if self._libc.sem_post(self._handle) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), self.name)

    def close(self, *, unlink: bool = False) -> None:
        if not self._closed:
            if self._libc.sem_close(self._handle) != 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error), self.name)
            self._closed = True
        if unlink and self._libc.sem_unlink(self.name.encode()) != 0:
            error = ctypes.get_errno()
            if error != errno.ENOENT:
                raise OSError(error, os.strerror(error), self.name)


_SHM_LIBC = ctypes.CDLL(None, use_errno=True)
_SHM_LIBC.shm_open.restype = ctypes.c_int


def _open_shared_memory(name: str, size: int) -> mmap.mmap:
    canonical_name = name if name.startswith("/") else f"/{name}"
    descriptor = _SHM_LIBC.shm_open(canonical_name.encode(), os.O_RDONLY)
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), canonical_name)
    try:
        return mmap.mmap(
            descriptor,
            int(size),
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ,
        )
    finally:
        os.close(descriptor)


class _BoundedTransferPool:
    def __init__(
        self,
        *,
        workers: int,
        capacity: int,
        byte_capacity: int | _ByteCapacity,
        name: str,
    ) -> None:
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=name,
        )
        self._entries = threading.BoundedSemaphore(capacity)
        self._bytes = (
            byte_capacity
            if isinstance(byte_capacity, _ByteCapacity)
            else _ByteCapacity(byte_capacity)
        )
        self._completion_wake: Any = None

    def set_completion_wake(self, wake: Any) -> None:
        self._completion_wake = wake

    def submit(self, operation: Any, *args: Any, nbytes: int) -> TransferTicket:
        if not self._entries.acquire(blocking=False):
            raise resource_error("asynchronous transfer ticket capacity is exhausted")
        bytes_acquired = False
        try:
            self._bytes.acquire(nbytes)
            bytes_acquired = True
            future = self._executor.submit(operation, *args)
        except BaseException:
            if bytes_acquired:
                self._bytes.release(nbytes)
            self._entries.release()
            raise

        def release(_future: object) -> None:
            self._bytes.release(nbytes)
            self._entries.release()
            wake = self._completion_wake
            if wake is not None:
                wake()

        future.add_done_callback(release)
        return _FutureTransferTicket(future)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)


class LocalTransport(Transport):
    """Same process, zero copy. The locator is a counter into a local table."""

    name = "local"
    supports_async_publication = True

    def __init__(self, *, byte_capacity: int) -> None:
        self._table: dict[int, "torch.Tensor"] = {}
        self._next = 0
        self._lock = threading.Lock()
        self._endpoint = f"local:{uuid.uuid4().hex}"
        self._bytes = _ByteCapacity(byte_capacity)

    def endpoint(self) -> str:
        return self._endpoint

    def publish(self, tensor: "torch.Tensor") -> Locator:
        t = tensor.detach()
        nbytes = _nbytes(t)
        self._bytes.acquire(nbytes)
        with self._lock:
            key = self._next
            self._next += 1
            self._table[key] = t
        return Locator(
            transport="local",
            endpoint=self._endpoint,
            nbytes=nbytes,
            dtype=_dtype_to_str(t.dtype),
            shape=tuple(t.shape),
            device=str(t.device),
            handle=str(key).encode(),
        )

    def fetch(self, locator: Locator) -> "torch.Tensor":
        if locator.endpoint != self._endpoint:
            raise invalid_descriptor("local locator belongs to another transport endpoint")
        key = int(locator.handle.decode())
        with self._lock:
            t = self._table.get(key)
        if t is None:
            raise invalid_descriptor(f"local locator {key} not registered (released?)")
        return t

    def fetch_async(self, locator: Locator) -> TransferTicket:
        return _ImmediateTransferTicket(self.fetch(locator))

    def release(self, locator: Locator) -> None:
        if locator.endpoint != self._endpoint:
            return
        with self._lock:
            removed = self._table.pop(int(locator.handle.decode()), None)
        if removed is not None:
            self._bytes.release(locator.nbytes)

    def close(self) -> None:
        with self._lock:
            values = tuple(self._table.values())
            self._table.clear()
        for value in values:
            self._bytes.release(_nbytes(value))


class _ShmReadTicket(TransferTicket):
    """A shared-memory read owned by the bounded transfer executor."""

    def __init__(self, transport: "ShmTransport", locator: Locator) -> None:
        self._inner = transport._reads.submit(
            transport.fetch,
            locator,
            nbytes=locator.nbytes,
        )

    def ready(self) -> bool:
        return self._inner.ready()

    def result(self) -> "torch.Tensor":
        if not self._inner.ready():
            raise RuntimeError("transfer ticket was observed before readiness")
        return self._inner.result()

    def add_done_callback(self, callback: Any) -> None:
        self._inner.add_done_callback(callback)


class ShmTransport(Transport):
    """Same-node snapshot transport over producer-owned POSIX shared memory.

    Every publication copies the tensor value into a distinct bounded segment.
    Asynchronous CUDA publication exposes a readiness byte and named semaphore;
    consumers use read-only mappings, while the producer owns unlink lifetime.
    """

    name = "shm"
    supports_async_publication = True
    blocking_fetch = True
    _MAX_LIVE_SEGMENTS = 256

    def __init__(self, *, byte_capacity: int, ticket_capacity: int) -> None:
        from collections import OrderedDict

        self._segments: "OrderedDict[str, Any]" = OrderedDict()  # name -> SharedMemory (LRU)
        self._pending: dict[str, tuple[Any, Any, threading.Event]] = {}
        self._semaphores: dict[str, _NamedSemaphore] = {}
        self._release_pending: set[str] = set()
        self._publication_bytes: dict[str, int] = {}
        self._lock = threading.Lock()
        self._bytes = _ByteCapacity(byte_capacity)
        self._ticket_capacity = int(ticket_capacity)
        if self._ticket_capacity < 1:
            raise ValueError("shared-memory transfer ticket capacity must be positive")
        self._publication_queue: queue.Queue[
            tuple[str, Any, int, Any, Any, threading.Event] | None
        ] = queue.Queue()
        self._publication_control_rx, self._publication_control_tx = socket.socketpair()
        self._publication_control_rx.setblocking(False)
        self._publication_control_tx.setblocking(False)
        self._completion_wake: Any = None
        self._publication_worker = threading.Thread(
            target=self._complete_publications,
            name="uniserve-shm-publication",
            daemon=True,
        )
        self._publication_worker.start()
        self._reads = _BoundedTransferPool(
            workers=2,
            capacity=self._ticket_capacity,
            byte_capacity=self._bytes,
            name="uniserve-shm-read",
        )

    def set_completion_wake(self, wake: Any) -> None:
        self._completion_wake = wake
        self._reads.set_completion_wake(wake)

    def _complete_publications(self) -> None:
        import torch

        selector = selectors.DefaultSelector()
        selector.register(self._publication_control_rx, selectors.EVENT_READ)
        closing = False
        while not closing or len(selector.get_map()) > 1:
            for key, _events in selector.select():
                if key.fileobj is self._publication_control_rx:
                    while True:
                        try:
                            if not self._publication_control_rx.recv(4096):
                                closing = True
                                break
                        except BlockingIOError:
                            break
                    while True:
                        try:
                            item = self._publication_queue.get_nowait()
                        except queue.Empty:
                            break
                        if item is None:
                            closing = True
                            continue
                        signal = item[4]
                        selector.register(signal, selectors.EVENT_READ, item)
                    continue
                item = key.data
                if item is None:
                    raise RuntimeError("shared-memory publication selector lost its entry")
                name, shm, nbytes, host, signal, completed = item
                selector.unregister(signal)
                try:
                    signal.consume()
                except BaseException:
                    succeeded = False
                else:
                    succeeded = True
                try:
                    if succeeded:
                        raw = host.view(torch.uint8).reshape(-1)
                        shm.buf[1 : nbytes + 1] = bytes(raw.numpy())
                except BaseException:
                    succeeded = False
                shm.buf[0] = 1 if succeeded else 2
                semaphore = self._semaphores.get(name)
                if semaphore is not None:
                    semaphore.post()
                completed.set()
                with self._lock:
                    self._pending.pop(name, None)
                    release = name in self._release_pending
                    self._release_pending.discard(name)
                    if release:
                        self._segments.pop(name, None)
                        released_bytes = self._publication_bytes.pop(name, 0)
                if release:
                    self._bytes.release(released_bytes)
                    semaphore = self._semaphores.pop(name, None)
                    if semaphore is not None:
                        semaphore.close(unlink=True)
                    shm.close()
                    try:
                        shm.unlink()
                    except FileNotFoundError:
                        pass
                wake = self._completion_wake
                if wake is not None:
                    wake()
        selector.close()

    def _queue_publication(
        self,
        item: tuple[str, Any, int, Any, Any, threading.Event] | None,
    ) -> None:
        self._publication_queue.put(item)
        try:
            self._publication_control_tx.send(b"\x01")
        except BlockingIOError:
            pass

    def publish(self, tensor: "torch.Tensor") -> Locator:
        from multiprocessing import shared_memory

        import torch

        host = tensor.detach().to("cpu").contiguous()
        raw = host.view(torch.uint8).reshape(-1)
        nbytes = int(raw.numel())
        shm = shared_memory.SharedMemory(create=True, size=max(1, nbytes))
        shm_buffer = shm.buf
        if shm_buffer is None:
            shm.close()
            shm.unlink()
            raise RuntimeError("shared-memory segment has no writable buffer")
        memoryview(shm_buffer)[:nbytes] = bytes(raw.numpy())
        try:
            self._bytes.acquire(nbytes)
        except BaseException:
            shm.close()
            shm.unlink()
            raise
        with self._lock:
            if len(self._segments) >= self._MAX_LIVE_SEGMENTS:
                shm.close()
                shm.unlink()
                self._bytes.release(nbytes)
                raise resource_error("shared-memory transport publication capacity is exhausted")
            self._segments[shm.name] = shm
            self._publication_bytes[shm.name] = nbytes
        return Locator(
            transport="shm",
            endpoint=self.name,
            nbytes=nbytes,
            dtype=_dtype_to_str(tensor.dtype),
            shape=tuple(tensor.shape),
            device=str(tensor.device),
            handle=shm.name.encode(),
        )

    @staticmethod
    def publish_bytes(payload: bytes) -> Locator:
        """Publish bytes with ownership transferred to one external consumer."""

        from multiprocessing import resource_tracker, shared_memory

        value = bytes(payload)
        if not value:
            raise ValueError("shared-memory media publication must not be empty")
        shm = shared_memory.SharedMemory(create=True, size=len(value))
        try:
            shm.buf[: len(value)] = value
            name = shm.name
        finally:
            shm.close()
        resource_tracker.unregister(shm._name, "shared_memory")
        return Locator(
            transport="shm",
            endpoint="media",
            nbytes=len(value),
            dtype="uint8",
            shape=(len(value),),
            device="cpu",
            handle=name.encode(),
        )

    def publish_async(self, tensor: "torch.Tensor") -> Locator:
        if not tensor.is_cuda:
            return self.publish(tensor)
        from multiprocessing import shared_memory

        import torch

        source = tensor.detach().contiguous()
        nbytes = _nbytes(source)
        host = torch.empty(tuple(source.shape), dtype=source.dtype, device="cpu", pin_memory=True)
        host.copy_(source, non_blocking=True)
        from .._uniserve_ipc import StreamSignal

        signal = StreamSignal()
        shm = shared_memory.SharedMemory(create=True, size=max(1, nbytes + 1))
        try:
            semaphore = _NamedSemaphore.create()
        except BaseException:
            shm.close()
            shm.unlink()
            raise
        shm_buffer = shm.buf
        if shm_buffer is None:
            semaphore.close(unlink=True)
            shm.close()
            shm.unlink()
            raise RuntimeError("shared-memory segment has no writable buffer")
        shm_buffer[0] = 0
        try:
            self._bytes.acquire(nbytes)
        except BaseException:
            semaphore.close(unlink=True)
            shm.close()
            shm.unlink()
            raise

        completed = threading.Event()
        with self._lock:
            if len(self._segments) >= self._MAX_LIVE_SEGMENTS:
                semaphore.close(unlink=True)
                shm.close()
                shm.unlink()
                self._bytes.release(nbytes)
                raise resource_error("shared-memory transport publication capacity is exhausted")
            self._segments[shm.name] = shm
            self._publication_bytes[shm.name] = nbytes
            self._pending[shm.name] = (host, signal, completed)
            self._semaphores[shm.name] = semaphore
        try:
            signal.schedule(int(torch.cuda.current_stream(source.device).cuda_stream))
        except BaseException:
            with self._lock:
                self._pending.pop(shm.name, None)
                self._segments.pop(shm.name, None)
                self._publication_bytes.pop(shm.name, None)
                self._semaphores.pop(shm.name, None)
            self._bytes.release(nbytes)
            semaphore.close(unlink=True)
            shm.close()
            shm.unlink()
            raise
        self._queue_publication((shm.name, shm, nbytes, host, signal, completed))
        return Locator(
            transport="shm",
            endpoint=self.name,
            nbytes=nbytes,
            dtype=_dtype_to_str(source.dtype),
            shape=tuple(source.shape),
            device=str(source.device),
            handle=shm.name.encode(),
            meta={"ready_header_bytes": 1, "ready_semaphore": semaphore.name},
        )

    def fetch(self, locator: Locator) -> "torch.Tensor":
        import torch

        name = locator.handle.decode()
        header = int(locator.meta.get("ready_header_bytes", 0))
        shm = _open_shared_memory(name, header + int(locator.nbytes))
        try:
            if header and shm[0] == 0:
                self._await_publication(locator)
            if header and shm[0] != 1:
                raise unsupported_setup("shared-memory publication did not complete")
            buf = bytearray(shm[header : header + locator.nbytes])
        finally:
            shm.close()
        out = (
            torch.frombuffer(buf, dtype=torch.uint8)
            .view(_dtype_from_str(locator.dtype))
            .reshape(locator.shape)
            .clone()
        )
        if locator.device != "cpu" and torch.cuda.is_available():
            out = out.to(locator.device)
        return out

    def ready(self, locator: Locator) -> bool:
        header = int(locator.meta.get("ready_header_bytes", 0))
        if header == 0:
            return True
        try:
            shm = _open_shared_memory(locator.handle.decode(), header)
        except FileNotFoundError:
            return False
        try:
            return bool(shm[0] != 0)
        finally:
            shm.close()

    def _await_publication(self, locator: Locator) -> None:
        name = locator.handle.decode()
        with self._lock:
            pending = self._pending.get(name)
        if pending is not None:
            pending[2].wait()
            return
        semaphore_name = locator.meta.get("ready_semaphore")
        if not isinstance(semaphore_name, str) or not semaphore_name:
            raise invalid_descriptor("shared-memory publication has no completion signal")
        semaphore = _NamedSemaphore.open(semaphore_name)
        try:
            semaphore.wait()
            semaphore.post()
        finally:
            semaphore.close()

    def fetch_async(self, locator: Locator) -> TransferTicket:
        return _ShmReadTicket(self, locator)

    def release(self, locator: Locator) -> None:
        name = locator.handle.decode()
        with self._lock:
            pending = name in self._pending
            if pending:
                self._release_pending.add(name)
            shm = self._segments.pop(name, None)
            released_bytes = 0 if pending else self._publication_bytes.pop(name, 0)
        if pending:
            return
        if shm is not None:
            self._bytes.release(released_bytes)
            semaphore = self._semaphores.pop(name, None)
            if semaphore is not None:
                semaphore.close(unlink=True)
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

    def close(self) -> None:
        self._queue_publication(None)
        self._publication_worker.join()
        self._publication_control_rx.close()
        self._publication_control_tx.close()
        self._reads.close()
        with self._lock:
            segs = list(self._segments.items())
            self._segments.clear()
            publication_bytes = self._publication_bytes
            self._publication_bytes = {}
            semaphores = self._semaphores
            self._semaphores = {}
        for name, shm in segs:
            self._bytes.release(publication_bytes.get(name, 0))
            semaphore = semaphores.get(name)
            if semaphore is not None:
                semaphore.close(unlink=True)
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass


class CudaIpcTransport(Transport):
    """Same-node GPU↔GPU via CUDA IPC. torch's reduction machinery emits the
    ``cudaIpcMemHandle`` and caches it per storage, so re-publishing tensors that
    share an allocation reuses one handle (register-once). The consumer opens the
    handle into a view of the producer's VRAM and clones it out."""

    name = "cuda_ipc"
    supports_async_publication = True
    _MAX_LIVE_PUBLICATIONS = 256

    def __init__(self, *, byte_capacity: int) -> None:
        self._alive: dict[str, tuple["torch.Tensor", "torch.cuda.Event"]] = {}
        self._lock = threading.Lock()
        self._bytes = _ByteCapacity(byte_capacity)

    def publish(self, tensor: "torch.Tensor") -> Locator:
        if not tensor.is_cuda:
            raise invalid_descriptor("cuda_ipc transport requires a CUDA tensor")
        import torch
        from torch.multiprocessing.reductions import StorageWeakRef, shared_cache

        source = tensor.detach().contiguous()
        nbytes = _nbytes(source)
        self._bytes.acquire(nbytes)
        try:
            t = source.clone()
        except BaseException:
            self._bytes.release(nbytes)
            raise
        event = torch.cuda.Event(interprocess=True)
        event.record(torch.cuda.current_stream(source.device))
        publication_id = uuid.uuid4().hex
        with self._lock:
            if len(self._alive) >= self._MAX_LIVE_PUBLICATIONS:
                self._bytes.release(nbytes)
                raise resource_error("CUDA IPC publication capacity is exhausted")
            self._alive[publication_id] = (t, event)
        try:
            storage = t._typed_storage()
            (
                storage_device,
                storage_handle,
                storage_size_bytes,
                storage_offset_bytes,
                ref_counter_handle,
                ref_counter_offset,
                allocator_event_handle,
                event_sync_required,
            ) = storage._share_cuda_()
            shared_cache[storage_handle] = StorageWeakRef(storage)
        except BaseException:
            with self._lock:
                self._alive.pop(publication_id, None)
            self._bytes.release(nbytes)
            raise
        return Locator(
            transport="cuda_ipc",
            endpoint=self.name,
            nbytes=nbytes,
            dtype=_dtype_to_str(t.dtype),
            shape=tuple(t.shape),
            device=str(t.device),
            handle=bytes(storage_handle),
            meta={
                "publication_id": publication_id,
                "storage_device": int(storage_device),
                "storage_size_bytes": int(storage_size_bytes),
                "storage_offset_bytes": int(storage_offset_bytes),
                "tensor_offset": int(t.storage_offset()),
                "tensor_stride": tuple(int(value) for value in t.stride()),
                "ref_counter_handle": bytes(ref_counter_handle),
                "ref_counter_offset": int(ref_counter_offset),
                "event_handle": bytes(allocator_event_handle),
                "event_sync_required": bool(event_sync_required),
                "ready_event_handle": bytes(event.ipc_handle()),
            },
        )

    def _open(self, locator: Locator) -> "torch.Tensor":
        import torch
        from torch.multiprocessing.reductions import rebuild_cuda_tensor

        return rebuild_cuda_tensor(
            torch.Tensor,
            locator.shape,
            tuple(int(value) for value in locator.meta["tensor_stride"]),
            int(locator.meta["tensor_offset"]),
            torch.storage.TypedStorage,
            _dtype_from_str(locator.dtype),
            int(locator.meta.get("storage_device", torch.device(locator.device).index or 0)),
            locator.handle,
            int(locator.meta["storage_size_bytes"]),
            int(locator.meta["storage_offset_bytes"]),
            False,
            bytes(locator.meta["ref_counter_handle"]),
            int(locator.meta["ref_counter_offset"]),
            bytes(locator.meta["event_handle"]),
            bool(locator.meta["event_sync_required"]),
        )

    def fetch(self, locator: Locator) -> "torch.Tensor":
        import torch

        event_handle = locator.meta.get("ready_event_handle")
        if not isinstance(event_handle, bytes):
            raise invalid_descriptor("CUDA IPC locator has no producer event")
        event = torch.cuda.Event.from_ipc_handle(
            torch.device(locator.device),
            event_handle,
        )
        torch.cuda.current_stream(torch.device(locator.device)).wait_event(event)
        return self._open(locator).clone()

    def fetch_async(self, locator: Locator) -> TransferTicket:
        return _ImmediateTransferTicket(self.fetch(locator))

    def release(self, locator: Locator) -> None:
        publication_id = locator.meta.get("publication_id")
        if not isinstance(publication_id, str):
            return
        with self._lock:
            removed = self._alive.pop(publication_id, None)
        if removed is not None:
            self._bytes.release(locator.nbytes)

    def close(self) -> None:
        with self._lock:
            alive = tuple(self._alive.values())
            self._alive.clear()
        for tensor, _event in alive:
            self._bytes.release(_nbytes(tensor))


def make_transport(name: str | TransportKind, **cfg: Any) -> Transport:
    """Construct the worker's configured bounded product transport."""
    raw_name = (str(name or TransportKind.LOCAL)).strip()
    try:
        kind = TransportKind(raw_name)
    except ValueError as exc:
        raise invalid_descriptor(
            f"unknown transport {raw_name!r}; expected one of {TRANSPORTS}"
        ) from exc
    if kind is TransportKind.LOCAL:
        return LocalTransport(byte_capacity=int(cfg["byte_capacity"]))
    if kind is TransportKind.SHM:
        return ShmTransport(
            byte_capacity=int(cfg["byte_capacity"]),
            ticket_capacity=int(cfg["ticket_capacity"]),
        )
    if kind is TransportKind.CUDA_IPC:
        return CudaIpcTransport(byte_capacity=int(cfg["byte_capacity"]))
    raise AssertionError(f"unhandled transport kind {kind!r}")
