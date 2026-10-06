"""Resolve in-process transport owners and expose their buffer registry."""

from __future__ import annotations

import threading
import weakref
from typing import TYPE_CHECKING

from uniserve_worker._uniserve_ipc import BufferRegistry as BufferRegistry
from uniserve_worker._uniserve_ipc import TransportBuffer as TransportBuffer

if TYPE_CHECKING:
    from uniserve_worker.transport.interface import Transport

# Weak values let local readers find an owner without extending its lifetime.
_endpoints: weakref.WeakValueDictionary[str, Transport] = (
    weakref.WeakValueDictionary()
)
_endpoint_lock = threading.Lock()
