"""Physical tensor transports backed by the native worker runtime."""

from enum import StrEnum

from uniserve_worker._uniserve_ipc import Transport as Transport


class TransportKind(StrEnum):
    """Selects process-local, shared storage, device, or rank-channel access."""

    LOCAL = "local"
    SHM = "shm"
    CUDA_VMM = "cuda_vmm"
    CHANNEL = "channel"


TRANSPORTS = tuple(kind.value for kind in TransportKind)
