"""Physical product publication and asynchronous read contracts."""

from __future__ import annotations

import concurrent.futures
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar

from uniserve_worker.protocol.transfer import Locator, WorkerEndpoint
from uniserve_worker.transport.ticket import TransferTicket

if TYPE_CHECKING:
    import torch


class TransportKind(StrEnum):
    """Selects in-process, shared storage, device, or rank-channel transport."""

    LOCAL = "local"
    SHM = "shm"
    CUDA_VMM = "cuda_vmm"
    CHANNEL = "channel"


TRANSPORTS = tuple(kind.value for kind in TransportKind)


class Transport(ABC):
    """Bounded physical publications and asynchronous reads per endpoint."""

    name: ClassVar[str]
    source: WorkerEndpoint

    @abstractmethod
    def endpoint(self) -> str:
        """Return the publishing address-space incarnation."""

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
    ) -> TransferTicket:
        """Read into one tensor or ordered first-axis spans on the device.

        Spans must be writable, disjoint and cover the exact requested region
        without dtype conversion. Backend layout restrictions are checked before
        submission. One ticket and one completion fence cover the whole read.
        An omitted destination lets the backend allocate or borrow.
        """

    @abstractmethod
    def release(
        self, locator: Locator
    ) -> concurrent.futures.Future[None] | None:
        """Retire a publication after its physical readers release ownership."""

    @abstractmethod
    def publication_retirement(
        self, locator: Locator
    ) -> concurrent.futures.Future[None]:
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
        """Connect asynchronous readiness to the worker controller."""
