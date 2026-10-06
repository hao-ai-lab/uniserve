"""Host products carried through rank-channel descriptors."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from uniserve import _slices
from uniserve.profiling import profile_range
from uniserve.runtime import EventPool
from uniserve_worker._uniserve_ipc import Completion
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol.transfer import (
    ChannelTransfer,
    Locator,
    WorkerEndpoint,
)
from uniserve_worker.transport.interface import Transport
from uniserve_worker.transport.layout import (
    copy_pairs,
    dtype_name,
    export_views,
    read_destination,
    region_view,
    resolve_dtype,
    tensor_nbytes,
)
from uniserve_worker.transport.pool import (
    ReadReservation,
    TransferCapacity,
    TransferPool,
)
from uniserve_worker.transport.ticket import TransferTicket

if TYPE_CHECKING:
    import torch

_LOG = logging.getLogger(__name__)


class ChannelTransport(Transport):
    """Host products carried on the rank channel's data path.

    Shared storage names a segment in one host's namespace, so it cannot serve
    a consumer on another host. This transport puts the product's bytes in its
    locator instead: they travel in the producing rank's result, into the
    head's custody, and out in the consuming rank's batch, reaching wherever
    the rank channel does.

    The producing rank owns nothing after export. The bytes are copied out
    of its storage while it exports, so the source is its own again as soon
    as the locator exists, and the head releases its copy when the buffer is
    freed. That is what the acknowledgment discipline reduces to when the head
    is the intermediary: it already holds the product exactly as long as some
    consumer may still be given it.
    """

    name = "channel"

    def __init__(
        self,
        *,
        capacity: TransferCapacity,
        event_pool: EventPool,
        source: WorkerEndpoint | None = None,
        host_slots: Sequence[int] = (),
    ) -> None:
        self.capacity = capacity
        self._events = event_pool
        self.source = source or WorkerEndpoint.local()
        # Slots of the ranks on this host, which shared storage reaches; the
        # channel carries a product only for a consumer elsewhere.
        self._host_slots = frozenset(int(slot) for slot in host_slots)
        self._endpoint = f"uniserve-channel-{uuid.uuid4().hex}"
        self._reads = TransferPool(
            workers=1,
            capacity=capacity,
            name="uniserve-channel-read",
            event_pool=event_pool,
        )
        # What carrying a product on the channel costs this rank. An export
        # blocks on its own stream and then copies the bytes out, and both are
        # on the batch's critical path, so each is counted separately from the
        # payload they move.
        self._exported = 0
        self._payload_bytes = 0
        self._largest_payload = 0
        self._synchronize_seconds = 0.0
        self._longest_synchronize = 0.0
        self._copy_seconds = 0.0
        self._fetched = 0
        self._fetch_seconds = 0.0

    def endpoint(self) -> str:
        return self._endpoint

    def serves(self, consumers: Sequence[int]) -> bool:
        return not consumers or any(
            int(slot) not in self._host_slots for slot in consumers
        )

    def set_completion_wake(self, wake: Any) -> None:
        self._reads.set_completion_wake(wake)

    def export(
        self,
        tensor: torch.Tensor | tuple[torch.Tensor, ...],
        *,
        offset: tuple[int, ...] | None = None,
        # The head holds the bytes for its consumers and releases them with
        # the buffer, so nothing here waits for an acknowledgment.
        consumers: Sequence[int] = (),
    ) -> Locator:
        """Copy the product into a locator that carries it."""
        import torch

        source, shape, offset = export_views(tensor, offset)
        spans = source if isinstance(source, tuple) else (source,)
        first = spans[0]
        nbytes = tensor_nbytes(source)
        self.capacity.acquire(nbytes)
        try:
            # One contiguous host buffer in physical tensor order. A device
            # product is copied through it, which is the same crossing a host
            # product would make to reach any consumer off this device.
            packed = torch.empty(shape, dtype=first.dtype, device="cpu")
            for target, value in copy_pairs(source, packed):
                target.copy_(value)
            if first.is_cuda:
                # The producer waits here for its own writes: the bytes leave
                # with the result, so nothing downstream can fence them. This
                # is the export's synchronize cost.
                started = time.perf_counter()
                with profile_range("channel_export_synchronize"):
                    torch.cuda.current_stream(first.device).synchronize()
                waited = time.perf_counter() - started
                self._synchronize_seconds += waited
                self._longest_synchronize = max(
                    self._longest_synchronize, waited
                )

            started = time.perf_counter()
            with profile_range("channel_export_payload"):
                payload = bytes(packed.flatten().view(torch.uint8).numpy())
            self._copy_seconds += time.perf_counter() - started
            self._exported += 1
            self._payload_bytes += nbytes
            self._largest_payload = max(self._largest_payload, nbytes)

            return Locator(
                source=self.source,
                transport=ChannelTransfer(
                    endpoint=self._endpoint,
                    # A byte view, so a dtype NumPy does not model travels as
                    # readily as one it does.
                    payload=payload,
                ),
                nbytes=nbytes,
                dtype=dtype_name(first.dtype),
                shape=shape,
                offset=offset,
                device=str(first.device),
            )
        finally:
            # The buffer is the only thing this rank held: the bytes
            # are in the locator by now, and the source is its own again.
            self.capacity.release(nbytes)

    def fetch(
        self,
        locator: Locator,
        *,
        device: torch.device,
        destination: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        region: tuple[slice, ...] | None = None,
        reservation: ReadReservation | None = None,
    ) -> TransferTicket:
        """Copy the locator's own bytes into a reserved destination."""
        import torch

        handle = locator.transport
        if not isinstance(handle, ChannelTransfer):
            raise invalid_descriptor("channel read requires a channel locator")
        target = read_destination(locator, device, destination, region)
        started = time.perf_counter()
        with profile_range("channel_fetch_payload"):
            payload = torch.frombuffer(
                bytearray(handle.payload), dtype=resolve_dtype(locator.dtype)
            ).reshape(locator.shape)
            # A device destination is filled by an asynchronous copy, which
            # reads pinned host storage; the bytes arrived pageable.
            if device.type == "cuda":
                carried = torch.empty_like(payload, pin_memory=True)
                carried.copy_(payload)
            else:
                carried = payload
        self._fetch_seconds += time.perf_counter() - started
        self._fetched += 1
        if region is not None:
            if not _slices.within(region, locator.shape):
                raise invalid_descriptor(
                    "read region exceeds the exported view"
                )
            carried = region_view(carried, region)
        return self._reads.submit(
            self._reads.copy,
            carried,
            target,
            None,
            nbytes=locator.nbytes,
            destination=target,
            reservation=reservation,
        )

    def release(self, locator: Locator) -> Completion | None:
        """Revoke an export the rank no longer owns anything of."""
        self._require_own(locator)
        return None

    def retirement(self, locator: Locator) -> Completion:
        """Expose completion, which export itself established.

        The product was copied out of the rank's storage while it exported,
        so there is nothing left to wait for and the source is reusable at
        once. The head holds the bytes from here, until the buffer is freed.
        """
        self._require_own(locator)
        settled: Completion = Completion()
        settled.set_result(None)
        return settled

    def _require_own(self, locator: Locator) -> ChannelTransfer:
        """Return this endpoint's handle, or refuse another's."""
        handle = locator.transport
        if (
            not isinstance(handle, ChannelTransfer)
            or handle.endpoint != self._endpoint
        ):
            raise invalid_descriptor(
                "channel export belongs to another endpoint"
            )
        return handle

    def close(self) -> None:
        """Report what the channel cost this rank, then release its reads."""
        if self._exported or self._fetched:
            mean = (
                self._payload_bytes // self._exported if self._exported else 0
            )
            _LOG.info(
                "channel transport retired: exported=%d payload_total=%d "
                "payload_mean=%d payload_max=%d synchronize_ms_total=%.3f "
                "synchronize_ms_max=%.3f copy_ms_total=%.3f fetched=%d "
                "fetch_ms_total=%.3f",
                self._exported,
                self._payload_bytes,
                mean,
                self._largest_payload,
                self._synchronize_seconds * 1e3,
                self._longest_synchronize * 1e3,
                self._copy_seconds * 1e3,
                self._fetched,
                self._fetch_seconds * 1e3,
            )
        self._reads.close()
