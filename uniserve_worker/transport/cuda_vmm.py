"""CUDA virtual storage export and bounded peer transfers.

A device product is exported in one of two forms. Storage that already has
exportable physical backing is exported where it lies, which copies nothing.
Anything else is copied into a chunk of the device's `VmmPool`, whose single
handle covers every chunk exported from that device. Either way the locator
carries a shareable allocation handle, the byte offsets of the spans inside
that allocation, and a readiness fence when one can reach the consumer.

Readiness: a rank whose products cross hosts drains its stream on every
export and exports no fence; otherwise the locator carries an
interprocess event handle. Retirement: once the engine has retired a
export, the producer's source is released when its fence drains, and a
pool chunk returns once no named consumer is still reading it, as the
acknowledgment words in the chunk header show when `CudaVmmTransport.reap`
sweeps them.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from collections.abc import Sequence
from functools import cache
from itertools import groupby, repeat
from typing import TYPE_CHECKING, Any

from uniserve.profiling import profile_range
from uniserve.runtime import CUDAEvent, EventPool
from uniserve_worker._uniserve_ipc import Completion
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol.transfer import (
    DESCRIPTOR_HANDLE_BYTES,
    FABRIC_HANDLE_BYTES,
    CudaVmmTransfer,
    Locator,
    WorkerEndpoint,
)
from uniserve_worker.transport import descriptor_grants, vmm_pool
from uniserve_worker.transport.descriptor_grants import DescriptorGrants
from uniserve_worker.transport.endpoint import (
    BufferRegistry,
    TransportBuffer,
    _endpoint_lock,
    _endpoints,
)
from uniserve_worker.transport.interface import Transport
from uniserve_worker.transport.layout import (
    copy_pairs,
    dtype_name,
    export_views,
    read_destination,
    region_view,
    tensor_nbytes,
)
from uniserve_worker.transport.pool import (
    ReadReservation,
    TransferCapacity,
    TransferPool,
    chunk_word,
)
from uniserve_worker.transport.ticket import TransferTicket
from uniserve_worker.transport.vmm_pool import (
    ACK_WORD_BYTES,
    VmmPool,
)

if TYPE_CHECKING:
    import torch

_LOG = logging.getLogger(__name__)


@cache
def _can_access_peer(device: str, peer: str) -> bool:
    """Return the process-stable CUDA peer relation for two visible devices."""
    import torch

    return torch.cuda.can_device_access_peer(
        torch.device(device), torch.device(peer)
    )


class CudaVmmTransport(Transport):
    """Device exports a consumer imports by their shareable handle.

    A consumer maps the producer's allocation from the handle the locator
    carries and, for a pool chunk, writes its acknowledgment word in the
    chunk's header once its copies retire. Where the device exports fabric
    handles nothing connects back to the producing rank, which is what lets a
    device product cross hosts.

    Where it exports POSIX descriptors instead, the exported handle names an
    open file of the producing process and carries no meaning elsewhere, so a
    consumer asks this rank for it over the grant socket in
    `descriptor_grants`. Such a device cannot place an edge across hosts in
    any case, so that connection has none of the costs of the fabric case.
    """

    name = "cuda_vmm"

    def __init__(
        self,
        *,
        capacity: TransferCapacity,
        event_pool: EventPool,
        source: WorkerEndpoint | None = None,
        acknowledgment_slot: int = 0,
        cross_host_consumers: bool = False,
    ) -> None:
        from uniserve_kernels import peer_storage

        peer_storage.load()
        self.source = source or WorkerEndpoint.local()
        self._events = event_pool
        # A failed export whose device work could not be drained. The
        # drain's error is re-raised by later `export` calls and by `close`,
        # and the tuple keeps the source, fence and copied views referenced so
        # storage a device may still be accessing is never reused.
        self._failed_export: (
            tuple[
                BaseException,
                torch.Tensor | tuple[torch.Tensor, ...] | None,
                CUDAEvent | None,
                torch.Tensor | tuple[torch.Tensor, ...] | None,
            ]
            | None
        ) = None
        self.capacity = capacity
        # Grants exist only where the device exports descriptors, so the
        # socket is bound when the first such export needs it.
        self._grants: DescriptorGrants | None = None
        # One bounded pool per device, reserved when that device first
        # exports, so a rank reserves nothing on a device it never
        # exports from. The pool is bounded by this rank's transfer byte
        # budget, which is what that budget already governs.
        self._pools: dict[str, VmmPool] = {}
        # This rank's own word in every chunk header it reads.
        self._acknowledgment_slot = int(acknowledgment_slot)
        # Whether an export has to state readiness without an interprocess
        # event, which only a consumer on another host requires.
        self._cross_host_consumers = bool(cross_host_consumers)
        # What exporting costs this rank. A crossing export drains the
        # producer's stream, because a consumer on another host can wait on no
        # fence this rank records, and that wait is on the batch's critical
        # path. The payloads are counted with it, since the cost of a crossing
        # is only interpretable against what it carries.
        self._exported = 0
        self._payload_bytes = 0
        self._largest_payload = 0
        self._synchronize_seconds = 0.0
        self._longest_synchronize = 0.0
        # The source retires after its copy. The native pool separately holds
        # the exported chunk until remote readers acknowledge completion.
        self._buffers = BufferRegistry(capacity=256, event_pool=event_pool)
        self._reads = TransferPool(
            workers=2,
            capacity=capacity,
            name="uniserve-cuda-read",
            event_pool=event_pool,
        )

        with _endpoint_lock:
            _endpoints[self.endpoint()] = self

    def endpoint(self) -> str:
        return self._buffers.name

    def _descriptor_grants(self) -> DescriptorGrants:
        """Bind this address space's grant socket on first use."""
        if self._grants is None:
            self._grants = DescriptorGrants(self.endpoint())
        return self._grants

    def retirement(self, locator: Locator) -> Completion:
        return self._buffers.retirement(locator)

    def set_completion_wake(self, wake: Any) -> None:
        self._reads.set_completion_wake(wake)

    def reap(self) -> None:
        """Return chunks whose consumers have finished acknowledging.

        The producing rank sweeps here rather than waiting on a per-reader
        connection, so a consumer on another host retires a product the same
        way one on this host does: by writing its slot's word in the chunk it
        mapped.
        """
        for pool in tuple(self._pools.values()):
            pool.reap()

    def awaiting_acknowledgment(self) -> bool:
        return any(
            pool.awaiting_acknowledgment() for pool in self._pools.values()
        )

    def _pool(self, device: torch.device) -> VmmPool:
        """Return this device's pool, reserving it on first export."""
        key = str(device)
        pool = self._pools.get(key)
        if pool is None:
            pool = VmmPool(device, capacity_bytes=self.capacity.capacity)
            self._pools[key] = pool
        return pool

    def export(
        self,
        tensor: torch.Tensor | tuple[torch.Tensor, ...],
        *,
        offset: tuple[int, ...] | None = None,
        consumers: Sequence[int] = (),
    ) -> Locator:
        """Export an immutable source and retain capacity.

        Capacity is retained until its producer fence retires. A product that
        can neither be exported where it lies nor fit its device's pool raises
        `PoolExhaustedError`, and the caller exports it as bytes over the
        host mechanism instead.
        """
        import torch
        from uniserve_kernels.peer_storage import export_handle

        source, shape, offset = export_views(tensor, offset)
        spans = source if isinstance(source, tuple) else (source,)
        first = spans[0]
        if not first.is_cuda:
            raise invalid_descriptor(
                "cuda_vmm transport requires a CUDA tensor"
            )
        # The locator addresses every span as a byte offset into one exported
        # allocation and carries one stride for all of them.
        if any(
            span.untyped_storage().data_ptr()
            != first.untyped_storage().data_ptr()
            or span.stride() != first.stride()
            for span in spans
        ):
            raise invalid_descriptor(
                "CUDA VMM export spans require one allocation and stride"
            )
        if self._failed_export is not None:
            raise self._failed_export[0]

        self._events.reap()
        nbytes = tensor_nbytes(tensor)
        self.capacity.acquire(nbytes)

        event = None
        export = None
        descriptor = None
        copied_source = None
        pool = chunk = grants = None
        export_id = ""
        submitted = False
        try:
            # Storage that can be exported where it lies is exported where
            # it lies, wherever it is read. That copies nothing, and it is
            # what lets an export name a row whose bytes arrive later: an
            # encoded media unit is exported with the batch that reserves its
            # row and filled when the encode completes, so a snapshot taken
            # now would carry whatever the row held before the encoder wrote
            # it. Where the device exports a fabric handle, the handle this
            # produces reaches another host too; what does not reach is the
            # fence, and that is settled below.
            exported = export_handle(first)
            if exported is None:
                # Otherwise the product is materialized in this device's pool,
                # whose one handle covers every product exported from that
                # device. Only the export's logical spans are
                # materialized, never an enclosing allocator segment. A
                # product the pool cannot hold is the caller's to export over
                # the host mechanism; the pool reports the exhaustion once.
                pool = self._pool(first.device)
                chunk = pool.reserve(tensor_nbytes(tensor))
            if exported is not None:
                descriptor, storage_size, storage_offset = exported
            else:
                assert pool is not None and chunk is not None
                shared = chunk.storage.view(first.dtype).view(shape)
                # From here device copies may be in flight, so a failure must
                # drain the stream before the chunk returns to its pool.
                submitted = True
                for target, value in copy_pairs(source, shared):
                    target.copy_(value, non_blocking=True)
                # The asynchronous copies still read the original views, which
                # the export retains until its fence drains.
                copied_source = source
                source = shared
                spans = (shared,)
                first = shared
                descriptor = pool.handle
                storage_size = pool.capacity
                # A consumer reads the payload, which follows the chunk's
                # acknowledgment words, so the offset names the payload.
                storage_offset = chunk.payload_offset
            # An export hands its consumers an event when one can reach
            # all of them, which is when this rank has no consumer on another
            # host. A consumer elsewhere can wait on nothing this rank records:
            # an event handle is host-local, and imported VMM storage admits
            # no device-side wait on current drivers. So a rank whose products
            # cross hosts drains its stream on every export, and the
            # export carries no fence at all. The event recorded below
            # still gates the source's release on this rank.
            interprocess = not self._cross_host_consumers
            if not interprocess:
                started = time.perf_counter()
                with profile_range("vmm_export_synchronize"):
                    torch.cuda.current_stream(first.device).synchronize()
                waited = time.perf_counter() - started
                self._synchronize_seconds += waited
                self._longest_synchronize = max(
                    self._longest_synchronize, waited
                )
            self._exported += 1
            self._payload_bytes += nbytes
            self._largest_payload = max(self._largest_payload, nbytes)
            event = self._events.acquire(
                first.device, interprocess=interprocess
            )
            self._events.retain(event, first.device)
            self._events.record(event, first.device)
            # The chunk returns to its pool once every rank the head named as
            # a reader of this call's products has written its acknowledgment,
            # which a consumer on another host can do as well as one here.
            readers = tuple(int(slot) for slot in consumers)
            export_id = uuid.uuid4().hex
            grants = None
            if len(descriptor) == DESCRIPTOR_HANDLE_BYTES:
                # The handle is an open file of this process. A consumer can
                # only receive it over the grant socket, so it is registered
                # before the locator naming it leaves this rank.
                grants = self._descriptor_grants()
                grants.register(
                    export_id, int.from_bytes(descriptor, sys.byteorder)
                )
            export = TransportBuffer.cuda(
                source,
                event,
                nbytes,
                self.capacity,
                descriptor,
                copied_source,
                pool=(pool, chunk)
                if pool is not None and chunk is not None
                else None,
                consumers=readers if chunk is not None else (),
                grants=grants,
                export_id=export_id,
            )

            # Run-length encode first-axis span lengths so the importer can
            # rebuild every span view without a per-span locator entry.
            length_runs = tuple(
                (length, sum(1 for _ in values))
                for length, values in groupby(
                    int(span.shape[0]) for span in spans
                )
            )
            locator = Locator(
                source=self.source,
                transport=CudaVmmTransfer(
                    endpoint=self.endpoint(),
                    export_id=export_id,
                    storage_size_bytes=storage_size,
                    storage_offsets_bytes=tuple(
                        storage_offset + span.data_ptr() - first.data_ptr()
                        for span in spans
                    ),
                    span_lengths=tuple(length for length, _ in length_runs),
                    span_counts=tuple(count for _, count in length_runs),
                    tensor_stride=tuple(first.stride()),
                    ready_event_handle=(
                        bytes(event.ipc_handle()) if interprocess else b""
                    ),
                    # A fabric handle travels with the export and a
                    # consumer imports it directly, anywhere in the fabric
                    # domain. A descriptor travels only so a consumer can tell
                    # which kind it is; the usable one comes from the grant.
                    allocation_handle=descriptor,
                    # A consumer writes its own slot's word here once its reads
                    # retire. An export outside the pool carries no header.
                    acknowledgment_offset=(
                        chunk.offset if chunk is not None else -1
                    ),
                ),
                nbytes=nbytes,
                dtype=dtype_name(first.dtype),
                shape=shape,
                offset=offset,
                device=str(first.device),
            )
            self._buffers.register(locator, export)
            return locator
        except BaseException:
            if export is not None:
                export.retire(self._events)
            else:
                try:
                    # No usable producer fence exists on this failure path.
                    # Keep its allocation and quota if draining also fails.
                    if submitted or event is not None:
                        torch.cuda.current_stream(first.device).synchronize()
                    if event is not None:
                        self._events.defer_release((event,), source)
                except BaseException as error:
                    self._failed_export = (
                        error,
                        source,
                        event,
                        copied_source,
                    )
                    raise
                if grants is not None:
                    grants.release(export_id)
                if chunk is not None:
                    assert pool is not None
                    pool.release(chunk)
                elif (
                    descriptor is not None
                    and len(descriptor) == DESCRIPTOR_HANDLE_BYTES
                ):
                    os.close(int.from_bytes(descriptor, sys.byteorder))
                self.capacity.release(nbytes)
            raise

    def fetch(
        self,
        locator: Locator,
        *,
        device: torch.device,
        destination: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        region: tuple[slice, ...] | None = None,
        reservation: ReadReservation | None = None,
    ) -> TransferTicket:
        if not isinstance(locator.transport, CudaVmmTransfer):
            raise invalid_descriptor(
                "CUDA VMM read requires a CUDA VMM locator"
            )
        # A device product reaches another host as a fabric handle and not as
        # a process descriptor, which names an allocation only within the host
        # that exported it. The head refuses an edge whose devices cannot
        # export one, so this is the rank restating what it was given.
        if (
            locator.source.node != self.source.node
            and len(locator.transport.allocation_handle) != FABRIC_HANDLE_BYTES
        ):
            raise invalid_descriptor(
                "a device product crosses hosts only as a fabric handle"
            )
        if device.type != "cuda":
            raise invalid_descriptor(
                "CUDA VMM destination must be a CUDA device"
            )
        target = (
            None
            if destination is None
            else read_destination(locator, device, destination, region)
        )
        return self._reads.submit(
            self._read,
            locator,
            device,
            target,
            region,
            nbytes=locator.nbytes,
            destination=target,
            reservation=reservation,
        )

    def _read(
        self,
        ticket: TransferTicket,
        locator: Locator,
        device: torch.device,
        destination: torch.Tensor | tuple[torch.Tensor, ...] | None,
        region: tuple[slice, ...] | None,
    ) -> None:
        import torch
        from uniserve_kernels.peer_storage import import_handle

        handle = locator.transport
        assert isinstance(handle, CudaVmmTransfer)
        # A consumer in another process cannot fully check that a locator
        # still names an export the producer holds: a descriptor grant is
        # refused once withdrawn, but a fabric handle is imported unchecked.
        # The locator is the engine's word. The engine binds only locators the
        # producing rank reported to it, and frees a product's buffer only
        # once the batch consuming it has completed.
        mapped = None
        event = None
        import_device = device
        # A read in the producer's own address space needs no acknowledgment:
        # the export's own owner reclaims it.
        acknowledgment = None
        try:
            ticket._require_active()
            # An omitted destination is allocated here, on the read thread.
            destination = read_destination(locator, device, destination, region)

            with torch.cuda.device(device):
                if locator.source.address_space == self.source.address_space:
                    # Same address space: borrow the owner's registered tensor
                    # and producer fence directly; no IPC mapping is needed.
                    with _endpoint_lock:
                        owner = _endpoints.get(handle.endpoint)
                    if not isinstance(owner, CudaVmmTransport):
                        raise invalid_descriptor(
                            "CUDA export has no live local owner"
                        )
                    export = owner._buffers.acquire(locator, ticket)
                    mapped = export.tensor
                    event = export.event
                else:
                    destination_prototype = (
                        destination[0]
                        if isinstance(destination, tuple)
                        else destination
                    )
                    itemsize = destination_prototype.element_size()
                    if any(
                        offset % itemsize
                        for offset in handle.storage_offsets_bytes
                    ):
                        raise invalid_descriptor(
                            "CUDA VMM span offset is not element aligned"
                        )
                    # A VMM mapping can be granted only to a device that can
                    # access the producing allocation. Some hosts expose all
                    # GPUs to one process but grant peer access only inside
                    # smaller peer islands. Map such an allocation on its
                    # owning GPU, then let CUDA perform the cross-device copy
                    # into the destination. CUDA stages through host storage
                    # when no peer path exists. This decision follows the
                    # discovered CUDA topology, not a device model or SKU.
                    source_device = torch.device(locator.device)
                    if (
                        locator.source.node == self.source.node
                        and source_device.type == "cuda"
                        and source_device != device
                        and not _can_access_peer(
                            str(device), str(source_device)
                        )
                    ):
                        import_device = source_device
                    prototype = (
                        destination_prototype
                        if import_device == device
                        else torch.empty(
                            0,
                            dtype=destination_prototype.dtype,
                            device=import_device,
                        )
                    )
                    # A fabric handle is importable as exported. A descriptor
                    # names an open file of the producing process, so the
                    # usable one is received from that rank over its grant
                    # socket and closed once the allocation, which holds its
                    # own reference, has been imported.
                    granted = None
                    if len(handle.allocation_handle) == DESCRIPTOR_HANDLE_BYTES:
                        granted = descriptor_grants.fetch(
                            handle.endpoint, handle.export_id
                        )
                        exported = granted.to_bytes(
                            DESCRIPTOR_HANDLE_BYTES, sys.byteorder
                        )
                    else:
                        exported = handle.allocation_handle
                    try:
                        allocation = import_handle(
                            prototype,
                            exported,
                            handle.storage_size_bytes,
                        )
                    finally:
                        if granted is not None:
                            os.close(granted)

                    # One mapping owns every span; tensor views share its
                    # deleter.
                    lengths = (
                        length
                        for length, count in zip(
                            handle.span_lengths, handle.span_counts, strict=True
                        )
                        for length in repeat(length, count)
                    )
                    mapped = tuple(
                        allocation.as_strided(
                            (length, *locator.shape[1:]),
                            handle.tensor_stride,
                            byte_offset // itemsize,
                        )
                        for byte_offset, length in zip(
                            handle.storage_offsets_bytes, lengths, strict=True
                        )
                    )
                    # This rank's acknowledgment word inside the source
                    # chunk's header, written once the copies below complete.
                    # It shares the mapping's deleter, like the span views.
                    if handle.acknowledgment_offset >= 0:
                        acknowledgment = allocation.view(torch.uint8)[
                            handle.acknowledgment_offset
                            + self._acknowledgment_slot * ACK_WORD_BYTES :
                        ][:ACK_WORD_BYTES].view(torch.int32)
                    del allocation
                    # A producer whose products cross hosts drained its stream
                    # before exporting, and its locator carries no fence;
                    # otherwise the fence is an interprocess event of the
                    # producer's host.
                    if handle.ready_event_handle:
                        event = CUDAEvent.from_ipc_handle(
                            import_device, handle.ready_event_handle
                        )

                if region is not None:
                    mapped = region_view(mapped, region)
                if event is not None and import_device != device:
                    # CUDA does not permit the destination device's stream to
                    # wait on this imported source-device IPC event on every
                    # topology. Drain the producer on its owning device before
                    # submitting the host-staged cross-device copy.
                    event.synchronize()
                    event = None
                if acknowledgment is not None and import_device != device:
                    # The acknowledgment word belongs to the source-device
                    # mapping. Claim it before the staged copy, then export
                    # completion only after the destination copy has drained.
                    with torch.cuda.device(import_device):
                        acknowledgment.copy_(chunk_word(vmm_pool.CLAIMED))
                        torch.cuda.current_stream(import_device).synchronize()
                    self._reads.copy(ticket, mapped, destination, event, None)
                    with torch.cuda.device(import_device):
                        acknowledgment.copy_(chunk_word(vmm_pool.ACKNOWLEDGED))
                        torch.cuda.current_stream(import_device).synchronize()
                else:
                    self._reads.copy(
                        ticket, mapped, destination, event, acknowledgment
                    )
        except BaseException as error:
            # Failure visibility must not wait for the source's retirement
            # acknowledgement. Physical ownership remains with the backend.
            ticket._fail(error)
            raise

    def release(self, locator: Locator) -> Completion | None:
        if not isinstance(locator.transport, CudaVmmTransfer):
            raise invalid_descriptor(
                "CUDA VMM release requires a CUDA VMM locator"
            )
        return self._buffers.release(locator)

    def close(self) -> None:
        if self._exported:
            _LOG.info(
                "cuda_vmm transport retired: exported=%d crossing=%s "
                "payload_total=%d payload_mean=%d payload_max=%d "
                "synchronize_ms_total=%.3f synchronize_ms_max=%.3f "
                "synchronize_ms_mean=%.3f",
                self._exported,
                self._cross_host_consumers,
                self._payload_bytes,
                self._payload_bytes // self._exported,
                self._largest_payload,
                self._synchronize_seconds * 1e3,
                self._longest_synchronize * 1e3,
                self._synchronize_seconds * 1e3 / self._exported,
            )
        try:
            self._reads.close()
        finally:
            self._buffers.close()
            # Consumers of this rank's remaining chunks are gone with it, so
            # their acknowledgments will never arrive. The pools are released
            # whole, which is what closing the transport means for them.
            for pool in self._pools.values():
                pool.close()
            self._pools.clear()
            if self._grants is not None:
                self._grants.close()
                self._grants = None
            with _endpoint_lock:
                _endpoints.pop(self.endpoint(), None)
        if self._failed_export is not None:
            raise self._failed_export[0]
