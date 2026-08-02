"""System-owned tensor transfer authority."""

from __future__ import annotations

from ..foundation.errors import capability_mismatch
from .transfer import Transport, make_transport

__all__ = ["Mover"]


class Mover:
    """Own the single transport used for product and state transfers."""

    def __init__(
        self,
        *,
        transfer_backend: str,
        mooncake_device: str,
        mooncake_protocol: str,
        transfer_byte_capacity: int,
        cross_process: bool = False,
    ) -> None:
        backend = str(transfer_backend)
        if bool(cross_process) and backend in {"", "local"}:
            raise capability_mismatch(
                f"cross-process transfer requires a shared transport, got {backend!r}"
            )
        self._backend = backend
        self._mooncake_device = str(mooncake_device)
        self._mooncake_protocol = str(mooncake_protocol)
        self._transfer_byte_capacity = int(transfer_byte_capacity)
        if self._transfer_byte_capacity < 1:
            raise capability_mismatch("transfer byte capacity must be positive")
        self._transport: Transport | None = None

    @property
    def transport(self) -> Transport:
        if self._transport is None:
            self._transport = make_transport(
                self._backend,
                device_name=self._mooncake_device,
                protocol=self._mooncake_protocol,
                byte_capacity=self._transfer_byte_capacity,
            )
        return self._transport

    def close(self) -> None:
        if self._transport is not None:
            self._transport.close()
            self._transport = None
