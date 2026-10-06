"""Host products carried through rank-channel descriptors."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

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
                # is included in the export's profiler range.
                with profile_range("channel_export_synchronize"):
                    torch.cuda.current_stream(first.device).synchronize()

            with profile_range("channel_export_payload"):
                payload = bytes(packed.flatten().view(torch.uint8).numpy())

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
        """Admit a read, then copy the locator's bytes on the read thread.

        Supplied destinations are checked before admission. Without one,
        allocation and shape errors fail the ticket. The native pool retains
        payload storage and read credits through destination-copy completion.
        """
        return self._reads.fetch_channel(
            locator,
            device=device,
            destination=destination,
            region=region,
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
        """Drain accepted reads and release their physical storage."""
        self._reads.close()


def _copy_payload(locator: Locator, device: torch.device) -> torch.Tensor:
    """Make a private tensor from channel bytes, pinned for device DMA."""
    import torch

    handle = cast(ChannelTransfer, locator.transport)
    with profile_range("channel_fetch_payload"):
        payload = torch.frombuffer(
            bytearray(handle.payload),
            dtype=resolve_dtype(locator.dtype),
        ).reshape(locator.shape)
        if device.type != "cuda":
            return payload

        # The copy stream needs pinned storage until its DMA completes.
        carried = torch.empty_like(payload, pin_memory=True)
        carried.copy_(payload)
        return carried
