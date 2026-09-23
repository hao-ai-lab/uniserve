"""CUDA virtual storage publication and bounded peer transfers."""

from __future__ import annotations

import concurrent.futures
import logging
import os
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from itertools import groupby, repeat
from typing import TYPE_CHECKING, Any

from uniserve.profiling import profile_range
from uniserve.runtime import EventPool
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
    Publications,
    _endpoint_lock,
    _endpoints,
)
from uniserve_worker.transport.interface import Transport
from uniserve_worker.transport.layout import (
    copy_pairs,
    dtype_name,
    publication_views,
    read_destination,
    region_view,
    tensor_nbytes,
)
from uniserve_worker.transport.pool import (
    TransferCapacity,
    TransferPool,
    chunk_word,
)
from uniserve_worker.transport.ticket import TransferTicket
from uniserve_worker.transport.vmm_pool import (
    ACK_WORD_BYTES,
    PoolChunk,
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


@dataclass(slots=True)
class _CudaSource:
    """Retain publication bytes through the producer's final device access."""

    tensor: torch.Tensor | tuple[torch.Tensor, ...]
    event: torch.cuda.Event
    nbytes: int
    capacity: TransferCapacity
    handle: bytes
    copied_source: torch.Tensor | tuple[torch.Tensor, ...] | None = None
    retirement: concurrent.futures.Future[None] | None = None
    #: Pool and chunk this publication occupies, when it came from a pool.
    pool: VmmPool | None = None
    chunk: PoolChunk | None = None
    #: Acknowledgment slots of the ranks that read this publication.
    consumers: tuple[int, ...] = ()
    #: Grant table this publication's descriptor is lent from, on a device
    #: that exports descriptors rather than fabric handles.
    grants: DescriptorGrants | None = None
    publication_id: str = ""

    def events_released(self) -> None:
        # A fabric handle is bytes the publication carried and this rank owns
        # nothing. A descriptor is an open file of this process, and a direct
        # export's belongs to the publication: the engine retires a
        # publication only once no named reader is still reading it, so the
        # grant and the descriptor end here together.
        #
        # A chunk's descriptor belongs to its pool, and a chunk outlives its
        # source by design -- the source was copied into it and is released
        # while consumers are still reading. Withdrawing that grant here would
        # refuse a reader that has not imported yet, so it is withdrawn where
        # the chunk returns to the pool instead.
        if (
            self.grants is not None
            and self.publication_id
            and self.pool is None
        ):
            self.grants.release(self.publication_id)
            os.close(int.from_bytes(self.handle, sys.byteorder))

        # This ends the source's lifetime, not the chunk's. A pool publication
        # was copied into its chunk, so the source is the producer's to reuse
        # as soon as its own fence drains, however long consumers keep reading
        # the chunk. The chunk returns separately, once they acknowledge it.
        self.capacity.release(self.nbytes)
        if self.retirement is not None:
            self.retirement.set_result(None)


class CudaVmmTransport(Transport):
    """Device publications a consumer imports by their shareable handle.

    A consumer maps the producer's allocation from the handle the locator
    carries and, for a pool chunk, writes its acknowledgment word in the
    chunk's header once its copies retire. Where the device exports fabric
    handles nothing connects back to the producing rank, which is what lets a
    device product cross hosts.

    Where it exports POSIX descriptors instead, the published handle names an
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
        self._failed_publication: (
            tuple[
                BaseException,
                torch.Tensor | tuple[torch.Tensor, ...] | None,
                torch.cuda.Event | None,
                torch.Tensor | tuple[torch.Tensor, ...] | None,
            ]
            | None
        ) = None
        self._bytes = capacity
        # Grants exist only where the device exports descriptors, so the
        # socket is bound when the first such publication needs it.
        self._grants: DescriptorGrants | None = None
        # One bounded pool per device, reserved when that device first
        # publishes, so a rank reserves nothing on a device it never
        # publishes from. The pool is bounded by this rank's transfer byte
        # budget, which is what that budget already governs.
        self._pools: dict[str, VmmPool] = {}
        # This rank's own word in every chunk header it reads.
        self._acknowledgment_slot = int(acknowledgment_slot)
        # Whether a publication has to state readiness without an interprocess
        # event, which only a consumer on another host requires.
        self._cross_host_consumers = bool(cross_host_consumers)
        # What publishing costs this rank. A crossing publication drains the
        # producer's stream, because a consumer on another host can wait on no
        # fence this rank records, and that wait is on the batch's critical
        # path. The payloads are counted with it, since the cost of a crossing
        # is only interpretable against what it carries.
        self._published = 0
        self._payload_bytes = 0
        self._largest_payload = 0
        self._synchronize_seconds = 0.0
        self._longest_synchronize = 0.0
        # Publications whose fence has drained but whose consumers have not all
        # acknowledged. Their chunks are held until reap() finds them settled.
        self._unacknowledged: list[_CudaSource] = []
        # A source is released when its own fence drains: a pool publication
        # was copied into its chunk, and the chunk is what waits for the
        # consumers, swept separately in reap().
        self._publications = Publications[_CudaSource](
            capacity=256,
            reclaim=self._reclaim,
            drain=self._drain,
            # A device publication's chunk is held by the pool until its
            # readers finish, which the transport's own sweep decides; the
            # publication itself owes nothing once the producer's work is
            # complete.
            settled=lambda source: True,
        )
        self._reads = TransferPool(
            workers=2,
            capacity=capacity,
            name="uniserve-cuda-read",
            event_pool=event_pool,
        )

        with _endpoint_lock:
            _endpoints[self.endpoint()] = self

    def endpoint(self) -> str:
        return self._publications.name

    def _descriptor_grants(self) -> DescriptorGrants:
        """Bind this address space's grant socket on first use."""
        if self._grants is None:
            self._grants = DescriptorGrants(self.endpoint())
        return self._grants

    def publication_retirement(
        self, locator: Locator
    ) -> concurrent.futures.Future[None]:
        return self._publications.retirement(locator)

    def set_completion_wake(self, wake: Any) -> None:
        self._reads.set_completion_wake(wake)

    def _release_chunk(self, source: _CudaSource) -> None:
        """Return one chunk to its pool and withdraw the grant that named it.

        The grant lends the pool's descriptor, which the pool keeps; what ends
        here is a consumer's right to ask for this chunk, which ends when the
        chunk can be handed out again.
        """
        assert source.pool is not None and source.chunk is not None
        if source.grants is not None and source.publication_id:
            source.grants.release(source.publication_id)
        source.pool.release(source.chunk)

    def _reclaim(
        self,
        source: _CudaSource,
        retirement: concurrent.futures.Future[None] | None = None,
    ) -> None:
        source.retirement = retirement
        if source.chunk is not None and source.pool is not None:
            if source.consumers:
                # A consumer may still be reading this chunk, and it will say
                # so by writing its word rather than by closing a connection.
                # Holding the chunk until then does not hold the source: that
                # was copied into the chunk and is released below.
                self._unacknowledged.append(source)
            else:
                # No other rank reads this product, so nothing can be waiting.
                self._release_chunk(source)
        self._events.defer_release(
            (source.event,), source, completed=source.events_released
        )

    def reap(self) -> None:
        """Return chunks whose consumers have finished acknowledging.

        The producing rank sweeps here rather than waiting on a per-reader
        connection, so a consumer on another host retires a product the same
        way one on this host does: by writing its slot's word in the chunk it
        mapped.
        """
        import torch

        held = [
            source
            for source in self._unacknowledged
            if source.pool is not None and source.chunk is not None
        ]
        if not held:
            return
        # Every held chunk's words are read in one transfer. Asking each chunk
        # separately would put one device-to-host synchronize per held
        # publication into every retirement pass.
        watched = [
            source.chunk.acknowledgments[list(source.consumers)]
            for source in held
            if source.chunk is not None
        ]
        observed = (
            torch.cat(watched).cpu().split([len(words) for words in watched])
        )

        waiting = []
        for source, words in zip(held, observed, strict=True):
            # A chunk returns once no named consumer is still reading it: one
            # that never claimed its word holds nothing, which is how a
            # product whose consuming call was never submitted retires.
            if bool((words != vmm_pool.CLAIMED).all()):
                self._release_chunk(source)
            else:
                waiting.append(source)
        self._unacknowledged = waiting

    def awaiting_acknowledgment(self) -> bool:
        return bool(self._unacknowledged)

    def _drain(self, source: _CudaSource) -> None:
        source.event.synchronize()
        self._events.reap()

    def _pool(self, device: torch.device) -> VmmPool:
        """Return this device's pool, reserving it on first publication."""
        key = str(device)
        pool = self._pools.get(key)
        if pool is None:
            pool = VmmPool(device, capacity_bytes=self._bytes.capacity)
            self._pools[key] = pool
        return pool

    def publish(
        self,
        tensor: torch.Tensor | tuple[torch.Tensor, ...],
        *,
        offset: tuple[int, ...] | None = None,
        consumers: Sequence[int] = (),
    ) -> Locator:
        """Export an immutable source and retain capacity.

        Capacity is retained until its producer fence retires. A product that
        can neither be exported where it lies nor fit its device's pool raises
        `PoolExhaustedError`, and the caller publishes it as bytes over the
        host mechanism instead.
        """
        import torch
        from uniserve_kernels.peer_storage import export_handle

        source, shape, offset = publication_views(tensor, offset)
        spans = source if isinstance(source, tuple) else (source,)
        first = spans[0]
        if not first.is_cuda:
            raise invalid_descriptor(
                "cuda_vmm transport requires a CUDA tensor"
            )
        if any(
            span.untyped_storage().data_ptr()
            != first.untyped_storage().data_ptr()
            or span.stride() != first.stride()
            for span in spans
        ):
            raise invalid_descriptor(
                "CUDA VMM publication spans require one allocation and stride"
            )
        if self._failed_publication is not None:
            raise self._failed_publication[0]

        self._events.reap()
        nbytes = tensor_nbytes(tensor)
        self._bytes.acquire(nbytes)

        event = None
        publication = None
        descriptor = None
        copied_source = None
        pool = chunk = grants = None
        publication_id = ""
        submitted = False
        try:
            # Storage that can be exported where it lies is published where
            # it lies, wherever it is read. That copies nothing, and it is
            # what lets a publication name a row whose bytes arrive later: an
            # encoded media unit is published with the batch that reserves its
            # row and filled when the encode completes, so a snapshot taken
            # now would carry whatever the row held before the encoder wrote
            # it. Where the device exports a fabric handle, the handle this
            # produces reaches another host too; what does not reach is the
            # fence, and that is settled below.
            exported = export_handle(first)
            if exported is None:
                # Otherwise the product is materialized in this device's pool,
                # whose one handle a consumer imports once however many
                # products it reads from that device. Only the publication's
                # logical spans are materialized, never an enclosing allocator
                # segment. A product the pool cannot hold is the caller's to
                # publish over the host mechanism; the pool reports the
                # exhaustion once.
                pool = self._pool(first.device)
                chunk = pool.reserve(tensor_nbytes(tensor))
            if exported is not None:
                descriptor, storage_size, storage_offset = exported
            else:
                assert pool is not None and chunk is not None
                shared = chunk.storage.view(first.dtype).view(shape)
                submitted = True
                for target, value in copy_pairs(source, shared):
                    target.copy_(value, non_blocking=True)
                copied_source = source
                source = shared
                spans = (shared,)
                first = shared
                descriptor = pool.handle
                storage_size = pool.capacity
                # A consumer reads the payload, which follows the chunk's
                # acknowledgment words, so the offset names the payload.
                storage_offset = chunk.payload_offset
            # A publication hands its consumers an event wherever one can
            # reach them, which is every consumer on this host. A consumer
            # elsewhere can wait on nothing this rank records: an event handle
            # is host-local, and imported VMM storage admits no device-side
            # wait on current drivers. So the producer drains its stream
            # instead, and the publication carries no fence at all.
            interprocess = not self._cross_host_consumers
            if not interprocess:
                started = time.perf_counter()
                with profile_range("vmm_publish_synchronize"):
                    torch.cuda.current_stream(first.device).synchronize()
                waited = time.perf_counter() - started
                self._synchronize_seconds += waited
                self._longest_synchronize = max(
                    self._longest_synchronize, waited
                )
            self._published += 1
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
            publication_id = uuid.uuid4().hex
            grants = None
            if len(descriptor) == DESCRIPTOR_HANDLE_BYTES:
                # The handle is an open file of this process. A consumer can
                # only receive it over the grant socket, so it is registered
                # before the locator naming it leaves this rank.
                grants = self._descriptor_grants()
                grants.register(
                    publication_id, int.from_bytes(descriptor, sys.byteorder)
                )
            publication = _CudaSource(
                source,
                event,
                nbytes,
                self._bytes,
                descriptor,
                copied_source,
                pool=pool if chunk is not None else None,
                chunk=chunk,
                consumers=readers if chunk is not None else (),
                grants=grants,
                publication_id=publication_id,
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
                    publication_id=publication_id,
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
                    # A fabric handle travels with the publication and a
                    # consumer imports it directly, anywhere in the fabric
                    # domain. A descriptor travels only so a consumer can tell
                    # which kind it is; the usable one comes from the grant.
                    allocation_handle=descriptor,
                    # A consumer writes its own slot's word here once its reads
                    # retire. A publication outside the pool carries no header.
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
            self._publications.publish(locator, publication)
            return locator
        except BaseException:
            if publication is not None:
                self._reclaim(publication)
            else:
                try:
                    # No usable producer fence exists on this failure path.
                    # Keep its allocation and quota if draining also fails.
                    if submitted or event is not None:
                        torch.cuda.current_stream(first.device).synchronize()
                    if event is not None:
                        self._events.defer_release((event,), source)
                except BaseException as error:
                    self._failed_publication = (
                        error,
                        source,
                        event,
                        copied_source,
                    )
                    raise
                if grants is not None:
                    grants.release(publication_id)
                if chunk is not None:
                    assert pool is not None
                    pool.release(chunk)
                elif (
                    descriptor is not None
                    and len(descriptor) == DESCRIPTOR_HANDLE_BYTES
                ):
                    os.close(int.from_bytes(descriptor, sys.byteorder))
                self._bytes.release(nbytes)
            raise

    def fetch(
        self,
        locator: Locator,
        *,
        device: torch.device,
        destination: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        region: tuple[slice, ...] | None = None,
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
        # No consumer can check that a locator still names a publication the
        # producer holds; the locator is the engine's word. The engine binds
        # only locators the producing rank reported to it, and frees a
        # product's buffer only once the batch consuming it has completed.
        mapped = None
        event = None
        import_device = device
        # A read in the producer's own address space needs no acknowledgment:
        # the publication's own owner reclaims it.
        acknowledgment = None
        try:
            ticket._require_active()
            destination = read_destination(locator, device, destination, region)
            with torch.cuda.device(device):
                if locator.source.address_space == self.source.address_space:
                    # Same address space: borrow the owner's registered tensor
                    # and producer fence directly; no IPC mapping is needed.
                    with _endpoint_lock:
                        owner = _endpoints.get(handle.endpoint)
                    if (
                        not isinstance(owner, CudaVmmTransport)
                        or owner.source != locator.source
                    ):
                        raise invalid_descriptor(
                            "CUDA publication has no live local owner"
                        )
                    publication = owner._publications.source(
                        locator, reading=True
                    )
                    mapped = publication.tensor
                    event = publication.event
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
                    # One mapping owns every span; tensor views share its
                    # deleter.
                    # A fabric handle is importable as published. A descriptor
                    # names an open file of the producing process, so the
                    # usable one is received from that rank over its grant
                    # socket and closed once the allocation, which holds its
                    # own reference, has been imported.
                    granted = None
                    if len(handle.allocation_handle) == DESCRIPTOR_HANDLE_BYTES:
                        granted = descriptor_grants.fetch(
                            handle.endpoint, handle.publication_id
                        )
                        exported = descriptor_grants.descriptor_bytes(granted)
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
                    # A pool publication was made readable by the producer's
                    # own synchronize; only an in-place one carries a fence,
                    # and that fence reaches this rank only on its own host.
                    if handle.ready_event_handle:
                        event = torch.cuda.Event.from_ipc_handle(
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
                    # mapping. Claim it before the staged copy, then publish
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
        finally:
            # An undrained read keeps its mapping and fence through the
            # ticket; a physically settled read drops them here.
            if not ticket._unretired:
                mapped = None
                event = None

    def release(
        self, locator: Locator
    ) -> concurrent.futures.Future[None] | None:
        if not isinstance(locator.transport, CudaVmmTransfer):
            raise invalid_descriptor(
                "CUDA VMM release requires a CUDA VMM locator"
            )
        retirement = self._publications.release(locator)
        if retirement is not None and not retirement.done():
            source = self._publications.source(locator)
            self._events.schedule_completion_wake(
                (
                    source.tensor[0]
                    if isinstance(source.tensor, tuple)
                    else source.tensor
                ).device,
                source.event,
            )
        return retirement

    def close(self) -> None:
        if self._published:
            _LOG.info(
                "cuda_vmm transport retired: published=%d crossing=%s "
                "payload_total=%d payload_mean=%d payload_max=%d "
                "synchronize_ms_total=%.3f synchronize_ms_max=%.3f "
                "synchronize_ms_mean=%.3f",
                self._published,
                self._cross_host_consumers,
                self._payload_bytes,
                self._payload_bytes // self._published,
                self._largest_payload,
                self._synchronize_seconds * 1e3,
                self._longest_synchronize * 1e3,
                self._synchronize_seconds * 1e3 / self._published,
            )
        try:
            self._reads.close()
        finally:
            self._publications.close()
            # Consumers of this rank's remaining chunks are gone with it, so
            # their acknowledgments will never arrive. The pools are released
            # whole, which is what closing the transport means for them.
            self._unacknowledged.clear()
            self._pools.clear()
            if self._grants is not None:
                self._grants.close()
                self._grants = None
            with _endpoint_lock:
                _endpoints.pop(self.endpoint(), None)
        if self._failed_publication is not None:
            raise self._failed_publication[0]
