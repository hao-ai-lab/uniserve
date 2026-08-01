"""Data plane, Tier 2: the pluggable, register-once byte transport.

A :class:`Transport` is **one per worker** and follows the register-once +
reference-by-(session, addr, len) model: a producer registers a buffer once and
hands out a compact :class:`Locator` describing a (sub)region of it; a consumer
materializes that locator over the real transport. The locator rides the control
plane opaquely — the host never parses it.

Backends (one chosen per worker via :func:`make_transport`):

* ``local``    — same process, zero copy (non-disaggregated / tests).
* ``shm``      — same node, host bytes via POSIX shared memory.
* ``cuda_ipc`` — same node, GPU↔GPU via CUDA IPC (torch's per-storage handle
  cache *is* register-once).
* ``mooncake`` — same/cross node, one-sided RDMA via Mooncake's ``TransferEngine``
  (the cross-device/node workhorse: GPUDirect VRAM, NIC selection, same-node
  fallback). One engine per worker; ``register_memory`` is cached per pointer so
  each buffer is registered exactly once.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import pickle
import queue
import socket
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..foundation.errors import capability_mismatch, invalid_descriptor

if TYPE_CHECKING:
    import torch

__all__ = [
    "Locator",
    "Transport",
    "LocalTransport",
    "ShmTransport",
    "CudaIpcTransport",
    "MooncakeTransport",
    "fetch_locator",
    "make_transport",
    "TransportKind",
    "TRANSPORTS",
]


class TransportKind(StrEnum):
    LOCAL = "local"
    SHM = "shm"
    CUDA_IPC = "cuda_ipc"
    MOONCAKE = "mooncake"


TRANSPORTS = tuple(kind.value for kind in TransportKind)


@dataclass(frozen=True)
class Locator:
    """Compact, wire-ready reference into a registered region.

    Carried opaquely by the control plane (producer ``SeqResult`` → host →
    consumer ``ForwardOp``) and resolved only by the consumer's transport.
    """

    transport: str
    session: str
    nbytes: int
    dtype: str
    shape: tuple[int, ...]
    device: str
    addr: int = 0  # absolute addr within the registered region (mooncake)
    handle: bytes = b""  # transport-specific resolve info (cuda_ipc / shm / local)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        return pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def from_bytes(raw: bytes) -> "Locator":
        loc = pickle.loads(raw)
        if not isinstance(loc, Locator):
            raise invalid_descriptor("decoded object is not a Locator")
        return loc

    def to_wire(self) -> dict[str, Any]:
        return {
            "version": 1,
            "transport": self.transport,
            "session": self.session,
            "nbytes": self.nbytes,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "device": self.device,
            "addr": self.addr,
            "handle_b64": base64.b64encode(self.handle).decode("ascii"),
            "meta": self.meta,
        }

    @staticmethod
    def from_wire(raw: dict[str, Any]) -> "Locator":
        if int(raw.get("version", 1)) != 1:
            raise invalid_descriptor("unsupported locator wire version")
        return Locator(
            transport=str(raw["transport"]),
            session=str(raw["session"]),
            nbytes=int(raw["nbytes"]),
            dtype=str(raw["dtype"]),
            shape=tuple(int(v) for v in raw["shape"]),
            device=str(raw["device"]),
            addr=int(raw.get("addr", 0)),
            handle=base64.b64decode(str(raw.get("handle_b64", "")).encode("ascii")),
            meta=dict(raw.get("meta") or {}),
        )

    def to_wire_json(self) -> str:
        return json.dumps(self.to_wire(), separators=(",", ":"), sort_keys=True)

    @staticmethod
    def from_wire_json(raw: str) -> "Locator":
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise invalid_descriptor("locator wire value must be a JSON object")
        return Locator.from_wire(value)


def fetch_locator(transport: "Transport", locator: Locator) -> "torch.Tensor":
    """Resolve a live transport locator or its verified durable tensor fallback."""

    try:
        return transport.fetch(locator)
    except Exception as transport_error:
        descriptor = locator.meta.get("durable_snapshot")
        if not isinstance(descriptor, dict):
            raise
        try:
            return _load_durable_tensor(locator, descriptor)
        except Exception as snapshot_error:
            raise invalid_descriptor(
                "runtime locator and durable snapshot fallback are both unavailable: "
                f"runtime={transport_error}; snapshot={snapshot_error}"
            ) from snapshot_error


def _load_durable_tensor(locator: Locator, descriptor: dict[str, Any]) -> "torch.Tensor":
    import torch
    from safetensors.torch import load_file

    if set(descriptor) != {"format_version", "root", "object", "tensor"}:
        raise invalid_descriptor("durable locator descriptor has an invalid shape")
    if descriptor.get("format_version") != 1:
        raise invalid_descriptor("durable locator format is unsupported")
    root_text = descriptor.get("root")
    object_digest = descriptor.get("object")
    tensor_key = descriptor.get("tensor")
    if not isinstance(root_text, str) or not Path(root_text).is_absolute():
        raise invalid_descriptor("durable locator root must be absolute")
    if not isinstance(object_digest, str) or not _is_sha256(object_digest):
        raise invalid_descriptor("durable locator object digest is invalid")
    if not isinstance(tensor_key, str) or not tensor_key.startswith("assets."):
        raise invalid_descriptor("durable locator tensor key is invalid")
    object_root = Path(root_text) / "objects" / object_digest
    manifest_path = object_root / "manifest.json"
    tensor_path = object_root / "tensors.safetensors"
    if not object_root.is_dir() or not manifest_path.is_file() or not tensor_path.is_file():
        raise invalid_descriptor("durable locator object is incomplete")
    raw_manifest = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw_manifest)
    except json.JSONDecodeError as error:
        raise invalid_descriptor(f"durable locator manifest is invalid: {error}") from error
    if not isinstance(manifest, dict) or _canonical_json(manifest) != raw_manifest:
        raise invalid_descriptor("durable locator manifest is not canonical")
    if _snapshot_digest(raw_manifest, tensor_path) != object_digest:
        raise invalid_descriptor("durable locator object failed content verification")
    assets = manifest.get("assets")
    if (
        not isinstance(assets, list)
        or sum(isinstance(asset, dict) and asset.get("tensor") == tensor_key for asset in assets)
        != 1
    ):
        raise invalid_descriptor("durable locator asset is not declared exactly once")
    tensors = load_file(str(tensor_path), device="cpu")
    value = tensors.get(tensor_key)
    if value is None:
        raise invalid_descriptor("durable locator tensor is missing")
    expected_dtype = _dtype_from_str(locator.dtype)
    if (
        tuple(value.shape) != locator.shape
        or value.dtype != expected_dtype
        or _nbytes(value) != locator.nbytes
    ):
        raise invalid_descriptor("durable locator tensor metadata does not match its descriptor")
    return value.to(torch.device(locator.device))


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _snapshot_digest(manifest: bytes, tensor_path: Path) -> str:
    digest = hashlib.sha256(b"uniserve-worker-snapshot-v1\0")
    digest.update(len(manifest).to_bytes(8, "little"))
    digest.update(manifest)
    with tensor_path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


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

    def session(self) -> str:
        """This worker's transport session id (for the locator)."""
        return self.name

    @abstractmethod
    def publish(self, tensor: "torch.Tensor") -> Locator:
        """Register the tensor's buffer (once) and return a locator for it."""

    def publish_async(self, tensor: "torch.Tensor") -> Locator:
        """Enqueue publication without observing device completion on the caller."""

        if not self.supports_async_publication:
            raise capability_mismatch(
                f"{self.name} transport does not support asynchronous publication"
            )
        return self.publish(tensor)

    @abstractmethod
    def fetch(self, locator: Locator) -> "torch.Tensor":
        """Read-driven: materialize the located tensor on this worker."""

    def push(self, tensor: "torch.Tensor", locator: Locator) -> None:
        """Write-driven: write ``tensor`` to the remote buffer ``locator`` names.

        Used by the KV edge. Optional; transports that support only the read
        direction raise ``NotImplementedError``.
        """
        raise NotImplementedError(f"{self.name} transport does not support write-driven push")

    def release(self, locator: Locator) -> None:
        """Deregister the producer buffer behind ``locator`` (idempotent)."""

    def close(self) -> None:
        """Tear down the transport (engine, segments)."""


class LocalTransport(Transport):
    """Same process, zero copy. The locator is a counter into a local table."""

    name = "local"
    supports_async_publication = True

    def __init__(self) -> None:
        self._table: dict[int, "torch.Tensor"] = {}
        self._next = 0
        self._lock = threading.Lock()
        self._session = f"local:{uuid.uuid4().hex}"

    def session(self) -> str:
        return self._session

    def publish(self, tensor: "torch.Tensor") -> Locator:
        t = tensor.detach()
        with self._lock:
            key = self._next
            self._next += 1
            self._table[key] = t
        return Locator(
            transport="local",
            session=self._session,
            nbytes=_nbytes(t),
            dtype=_dtype_to_str(t.dtype),
            shape=tuple(t.shape),
            device=str(t.device),
            handle=str(key).encode(),
        )

    def fetch(self, locator: Locator) -> "torch.Tensor":
        if locator.session != self._session:
            raise invalid_descriptor("local locator belongs to another transport session")
        key = int(locator.handle.decode())
        with self._lock:
            t = self._table.get(key)
        if t is None:
            raise invalid_descriptor(f"local locator {key} not registered (released?)")
        return t

    def push(self, tensor: "torch.Tensor", locator: Locator) -> None:
        if locator.session != self._session:
            raise invalid_descriptor("local locator belongs to another transport session")
        key = int(locator.handle.decode())
        with self._lock:
            dst = self._table.get(key)
        if dst is None:
            raise invalid_descriptor(f"local locator {key} not registered")
        dst.copy_(tensor)

    def release(self, locator: Locator) -> None:
        if locator.session != self._session:
            return
        with self._lock:
            self._table.pop(int(locator.handle.decode()), None)

    def close(self) -> None:
        with self._lock:
            self._table.clear()


class ShmTransport(Transport):
    """Same-node host transport over POSIX shared memory.

    This is a *snapshot* transport: each ``publish`` copies the tensor's current
    bytes into a FRESH named segment. (register-once-by-pointer would be wrong
    here — a producer reuses a logits/scratch buffer across steps with new data
    each time, so a pointer-keyed cache returns stale bytes; the register-once
    optimization is for the live-buffer RDMA/cuda_ipc transports.) A bounded LRU
    of recent segments is kept alive so a consumer can still map them; older
    segments are unlinked once the producer is well past them, and ``release``
    reclaims promptly."""

    name = "shm"
    supports_async_publication = True
    _MAX_LIVE_SEGMENTS = 256

    def __init__(self) -> None:
        from collections import OrderedDict

        self._segments: "OrderedDict[str, Any]" = OrderedDict()  # name -> SharedMemory (LRU)
        self._pending: dict[str, tuple[Any, Any]] = {}
        self._release_pending: set[str] = set()
        self._lock = threading.Lock()
        self._publication_queue: queue.Queue[tuple[str, Any, int, Any, Any] | None] = queue.Queue()
        self._publication_worker = threading.Thread(
            target=self._complete_publications,
            name="uniserve-shm-publication",
            daemon=True,
        )
        self._publication_worker.start()

    def _complete_publications(self) -> None:
        import torch

        pending: list[tuple[str, Any, int, Any, Any]] = []
        closing = False
        while pending or not closing:
            try:
                item = self._publication_queue.get(timeout=0.001)
                if item is None:
                    closing = True
                else:
                    pending.append(item)
            except queue.Empty:
                pass
            deferred: list[tuple[str, Any, int, Any, Any]] = []
            for name, shm, nbytes, host, event in pending:
                try:
                    ready = bool(event.query())
                except BaseException:
                    ready = True
                    succeeded = False
                else:
                    succeeded = True
                if not ready:
                    deferred.append((name, shm, nbytes, host, event))
                    continue
                try:
                    if succeeded:
                        raw = host.view(torch.uint8).reshape(-1)
                        shm.buf[1 : nbytes + 1] = bytes(raw.numpy())
                except BaseException:
                    succeeded = False
                shm.buf[0] = 1 if succeeded else 2
                with self._lock:
                    self._pending.pop(name, None)
                    release = name in self._release_pending
                    self._release_pending.discard(name)
                    if release:
                        self._segments.pop(name, None)
                if release:
                    shm.close()
                    try:
                        shm.unlink()
                    except FileNotFoundError:
                        pass
            pending = deferred

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
        evicted: list[Any] = []
        with self._lock:
            self._segments[shm.name] = shm
            while len(self._segments) > self._MAX_LIVE_SEGMENTS:
                _, old = self._segments.popitem(last=False)
                evicted.append(old)
        for old in evicted:
            old.close()
            try:
                old.unlink()
            except FileNotFoundError:
                pass
        return Locator(
            transport="shm",
            session=self.name,
            nbytes=nbytes,
            dtype=_dtype_to_str(tensor.dtype),
            shape=tuple(tensor.shape),
            device=str(tensor.device),
            handle=shm.name.encode(),
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
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(source.device))
        shm = shared_memory.SharedMemory(create=True, size=max(1, nbytes + 1))
        shm_buffer = shm.buf
        if shm_buffer is None:
            raise RuntimeError("shared-memory segment has no writable buffer")
        shm_buffer[0] = 0

        evicted: list[Any] = []
        with self._lock:
            self._segments[shm.name] = shm
            self._pending[shm.name] = (host, event)
            while len(self._segments) > self._MAX_LIVE_SEGMENTS:
                name, old = self._segments.popitem(last=False)
                if name in self._pending:
                    self._segments[name] = old
                    break
                evicted.append(old)
        self._publication_queue.put((shm.name, shm, nbytes, host, event))
        for old in evicted:
            old.close()
            try:
                old.unlink()
            except FileNotFoundError:
                pass
        return Locator(
            transport="shm",
            session=self.name,
            nbytes=nbytes,
            dtype=_dtype_to_str(source.dtype),
            shape=tuple(source.shape),
            device=str(source.device),
            handle=shm.name.encode(),
            meta={"ready_header_bytes": 1},
        )

    def fetch(self, locator: Locator) -> "torch.Tensor":
        from multiprocessing import shared_memory

        import torch

        shm = shared_memory.SharedMemory(name=locator.handle.decode())
        try:
            shm_buffer = shm.buf
            if shm_buffer is None:
                raise RuntimeError("shared-memory segment has no readable buffer")
            header = int(locator.meta.get("ready_header_bytes", 0))
            while header and shm_buffer[0] == 0:
                time.sleep(0.0001)
            if header and shm_buffer[0] != 1:
                raise capability_mismatch("shared-memory publication did not complete")
            try:
                buf = bytearray(memoryview(shm_buffer)[header : header + locator.nbytes])
            finally:
                shm_buffer.release()
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

    def release(self, locator: Locator) -> None:
        name = locator.handle.decode()
        with self._lock:
            pending = name in self._pending
            if pending:
                self._release_pending.add(name)
            shm = self._segments.pop(name, None)
        if pending:
            return
        if shm is not None:
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

    def close(self) -> None:
        self._publication_queue.put(None)
        self._publication_worker.join()
        with self._lock:
            segs = list(self._segments.values())
            self._segments.clear()
        for shm in segs:
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

    def __init__(self) -> None:
        self._alive: dict[int, "torch.Tensor"] = {}  # data_ptr -> kept-alive tensor
        self._lock = threading.Lock()

    def publish(self, tensor: "torch.Tensor") -> Locator:
        from torch.multiprocessing.reductions import reduce_tensor

        if not tensor.is_cuda:
            raise invalid_descriptor("cuda_ipc transport requires a CUDA tensor")
        t = tensor.detach().contiguous()
        with self._lock:
            self._alive[int(t.data_ptr())] = t  # source storage must outlive the read
        rebuild, args = reduce_tensor(t)
        return Locator(
            transport="cuda_ipc",
            session=self.name,
            nbytes=_nbytes(t),
            dtype=_dtype_to_str(t.dtype),
            shape=tuple(t.shape),
            device=str(t.device),
            handle=pickle.dumps((rebuild, args), protocol=pickle.HIGHEST_PROTOCOL),
        )

    def _open(self, locator: Locator) -> "torch.Tensor":
        rebuild, args = pickle.loads(locator.handle)
        return rebuild(*args)  # view into the producer's VRAM

    def fetch(self, locator: Locator) -> "torch.Tensor":
        return self._open(locator).clone()

    def push(self, tensor: "torch.Tensor", locator: Locator) -> None:
        self._open(locator).copy_(tensor.detach())

    def release(self, locator: Locator) -> None:
        # The producer drops its kept-alive tensor when the host releases the
        # handle; we cannot match a specific ptr from the consumer side cheaply.
        return None

    def close(self) -> None:
        with self._lock:
            self._alive.clear()


class MooncakeTransport(Transport):
    """Same/cross-node one-sided RDMA via Mooncake's ``TransferEngine``.

    One engine per worker, initialized once; ``register_memory`` is cached per
    pointer so each buffer registers exactly once. Locators name (session, addr);
    ``fetch`` is a one-sided ``transfer_sync_read`` from the producer, ``push`` a
    ``transfer_sync_write`` to a remote buffer (the KV edge).
    """

    name = "mooncake"
    supports_async_publication = True

    def __init__(
        self,
        *,
        device_name: str = "",
        protocol: str = "rdma",
        hostname: str | None = None,
        metadata_server: str = "P2PHANDSHAKE",
    ) -> None:
        _ensure_mooncake_runtime()
        from mooncake.engine import TransferEngine

        self.engine = TransferEngine()
        host = hostname or _local_ip()
        rc = self.engine.initialize(host, metadata_server, protocol, device_name)
        if rc != 0:
            raise capability_mismatch(
                f"Mooncake TransferEngine.initialize failed (code {rc}); "
                f"protocol={protocol!r} device_name={device_name!r}"
            )
        port = self.engine.get_rpc_port()
        self._session = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
        self._registered: set[int] = set()  # register-once, by data_ptr
        self._alive: dict[int, "torch.Tensor"] = {}
        self._lock = threading.Lock()

    def session(self) -> str:
        return self._session

    def _register_once(self, ptr: int, nbytes: int) -> None:
        with self._lock:
            if ptr in self._registered:
                return
        rc = self.engine.register_memory(ptr, nbytes)
        if rc != 0:
            raise capability_mismatch(f"Mooncake register_memory failed (code {rc})")
        with self._lock:
            self._registered.add(ptr)

    def publish(self, tensor: "torch.Tensor") -> Locator:
        t = tensor.detach().contiguous()
        ptr = int(t.data_ptr())
        nbytes = _nbytes(t)
        self._register_once(ptr, nbytes)
        with self._lock:
            self._alive[ptr] = t  # keep the source buffer alive for remote reads
        return Locator(
            transport="mooncake",
            session=self._session,
            nbytes=nbytes,
            dtype=_dtype_to_str(t.dtype),
            shape=tuple(t.shape),
            device=str(t.device),
            addr=ptr,
        )

    def fetch(self, locator: Locator) -> "torch.Tensor":
        import torch

        dst = torch.empty(
            locator.shape, dtype=_dtype_from_str(locator.dtype), device=locator.device
        )
        dptr = int(dst.data_ptr())
        self._register_once(dptr, locator.nbytes)
        rc = self.engine.transfer_sync_read(locator.session, dptr, locator.addr, locator.nbytes)
        if rc != 0:
            raise capability_mismatch(f"Mooncake transfer_sync_read failed (code {rc})")
        return dst

    def push(self, tensor: "torch.Tensor", locator: Locator) -> None:
        t = tensor.contiguous()
        ptr = int(t.data_ptr())
        self._register_once(ptr, _nbytes(t))
        rc = self.engine.transfer_sync_write(locator.session, ptr, locator.addr, locator.nbytes)
        if rc != 0:
            raise capability_mismatch(f"Mooncake transfer_sync_write failed (code {rc})")

    def release(self, locator: Locator) -> None:
        with self._lock:
            self._alive.pop(int(locator.addr), None)

    def close(self) -> None:
        with self._lock:
            ptrs = list(self._registered)
            self._registered.clear()
            self._alive.clear()
        for ptr in ptrs:
            try:
                self.engine.unregister_memory(ptr)
            except Exception:
                pass


def _ensure_mooncake_runtime() -> None:
    """Preload the CUDA-12 runtime if Mooncake's prebuilt wheel needs it.

    The ``mooncake-transfer-engine`` wheel links ``libcudart.so.12``; on a
    CUDA-13 host ``dlopen`` it RTLD_GLOBAL from ``nvidia-cuda-runtime-cu12`` (if
    installed) so Mooncake's extension resolves it without ``LD_LIBRARY_PATH``.
    No-op if the runtime library is already resolvable on the host.
    """
    import ctypes
    import glob

    try:
        import nvidia.cuda_runtime

        roots = list(getattr(nvidia.cuda_runtime, "__path__", []))
    except Exception:
        return
    for root in roots:
        for so in glob.glob(os.path.join(root, "lib", "libcudart.so.12*")):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                return
            except OSError:
                continue


def _local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def make_transport(name: str | TransportKind, **cfg: Any) -> Transport:
    """Select the worker's single Tier-2 transport (mirrors Rust
    ``make_transfer_agent``)."""
    raw_name = (str(name or TransportKind.LOCAL)).strip()
    try:
        kind = TransportKind(raw_name)
    except ValueError as exc:
        raise invalid_descriptor(
            f"unknown transport {raw_name!r}; expected one of {TRANSPORTS}"
        ) from exc
    if kind is TransportKind.LOCAL:
        return LocalTransport()
    if kind is TransportKind.SHM:
        return ShmTransport()
    if kind is TransportKind.CUDA_IPC:
        return CudaIpcTransport()
    if kind is TransportKind.MOONCAKE:
        return MooncakeTransport(
            device_name=str(cfg["device_name"]),
            protocol=str(cfg["protocol"]),
            hostname=cfg.get("hostname"),
        )
    raise AssertionError(f"unhandled transport kind {kind!r}")
