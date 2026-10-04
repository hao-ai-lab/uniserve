"""Physical product publication and asynchronous read contracts.

A producer publishes a product through a `Transport` and receives a `Locator`
naming where its bytes are. The engine carries that locator to consumers,
whose own transport of the same kind reads from it into a destination and
returns a `TransferTicket`. The producer keeps the published storage
unwritten until the publication's retirement signal completes after the
engine releases it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar

from uniserve_worker._uniserve_ipc import Completion
from uniserve_worker.protocol.transfer import Locator, WorkerEndpoint
from uniserve_worker.transport.ticket import TransferTicket

if TYPE_CHECKING:
    import torch

    from uniserve_worker.transport.pool import (
        ReadReservation,
        TransferCapacity,
    )


class TransportKind(StrEnum):
    """Selects in-process, shared storage, device, or rank-channel transport."""

    LOCAL = "local"
    SHM = "shm"
    CUDA_VMM = "cuda_vmm"
    CHANNEL = "channel"


#: Every transport name a rank may bind, as `make_transports` accepts them.
TRANSPORTS = tuple(kind.value for kind in TransportKind)


class Transport(ABC):
    """Bounded physical publications and asynchronous reads per endpoint.

    ``capacity`` is the rank's byte and read-ticket budget, which every
    transport `make_transports` builds for the rank shares.
    """

    name: ClassVar[str]
    source: WorkerEndpoint
    capacity: TransferCapacity

    @abstractmethod
    def endpoint(self) -> str:
        """Return this instance's unique endpoint name.

        Every locator the instance publishes carries it: a release refuses a
        locator from another instance, and a `local` or `cuda_vmm` read in
        the publishing address space finds the owning instance by it.
        """

    @abstractmethod
    def publish(
        self,
        tensor: torch.Tensor | tuple[torch.Tensor, ...],
        *,
        offset: tuple[int, ...] | None = None,
        consumers: Sequence[int] = (),
    ) -> Locator:
        """Expose a descriptor and producer fence for an immutable version.

        The allocation owner must retain the published range, without writes,
        until publication_retirement() completes after release(). Keeping a
        tensor reference does not authorize reuse of an arena or page range.

        `consumers` are the acknowledgment slots of the ranks that read this
        publication, as the head stated them on the producing call. A
        mechanism that holds storage another process reads returns it once
        each has acknowledged; a mechanism whose consumers are in this process
        or hold their own copy has nothing to wait for and ignores them.

        Every implementation reserves the publication's bytes against the
        rank's shared `TransferCapacity` for as long as it holds the source,
        and raises `resource_error` when that budget is exhausted.
        """

    def serves(self, consumers: Sequence[int]) -> bool:
        """Whether this mechanism reaches the named consumers.

        A publication is made over each mechanism that reaches one of the
        acknowledgment slots the producing call names; a call that names none
        is published over every mechanism the rank binds. A mechanism that
        reaches every consumer wherever it runs serves any call.
        """
        del consumers
        return True

    @abstractmethod
    def fetch(
        self,
        locator: Locator,
        *,
        device: torch.device,
        destination: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        region: tuple[slice, ...] | None = None,
        reservation: ReadReservation | None = None,
    ) -> TransferTicket:
        """Read into one tensor or ordered first-axis spans on the device.

        Spans must be writable, disjoint and cover the exact requested region
        without dtype conversion. Backend layout restrictions are checked before
        submission. One ticket and one completion fence cover the whole read.
        An omitted destination lets the backend allocate one, or, for `local`,
        borrow the published views themselves. The read uses a ticket of
        ``reservation``, or takes one of ``capacity``'s without one, and
        raises `ReadBackpressureError` when none is free.
        """

    @abstractmethod
    def release(self, locator: Locator) -> Completion | None:
        """Revoke new reads of a publication and return its retirement.

        The returned signal completes once the owner may reuse the published
        storage. `None` means this instance holds no registration for the
        locator; `channel` always returns `None`, since it retains nothing
        after publishing.
        """

    @abstractmethod
    def publication_retirement(self, locator: Locator) -> Completion:
        """Observe physical ownership completion.

        The publication is not revoked.
        """

    def reap(self) -> None:
        """Release publications their consumers have finished acknowledging.

        A consumer acknowledges a product by writing into the storage it read,
        which reaches the producer with no local notification. A transport
        whose publications retire with their own producer has nothing to sweep.
        """

    def awaiting_acknowledgment(self) -> bool:
        """Report whether a retired publication still waits on a consumer.

        An acknowledgment arrives with no notification, so a producer with one
        outstanding sweeps on a short period rather than on its next event.
        """
        return False

    @abstractmethod
    def close(self) -> None:
        """Drain reads and release endpoint resources."""

    @abstractmethod
    def set_completion_wake(self, wake: Any) -> None:
        """Connect asynchronous readiness to the worker controller.

        `wake` is a zero-argument callable (or None), which the transport
        invokes when asynchronous work it owns, such as a read, completes.
        """
