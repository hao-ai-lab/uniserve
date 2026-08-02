"""Wire framing bounds for exact cross-stage product transport entries."""

from __future__ import annotations

TRANSFER_DESCRIPTOR_PREFIX = b"uniserve-transfer\0"
MAX_TRANSFER_DESCRIPTOR_BYTES = 64 * 1024


def is_transfer_descriptor(value: bytes) -> bool:
    return value.startswith(TRANSFER_DESCRIPTOR_PREFIX)


def validate_transfer_descriptor_frame(value: bytes) -> None:
    if not is_transfer_descriptor(value):
        raise ValueError("cross-stage product input has no transfer descriptor frame")
    if len(value) > MAX_TRANSFER_DESCRIPTOR_BYTES:
        raise ValueError("cross-stage product transfer descriptor exceeds its byte bound")


__all__ = [
    "MAX_TRANSFER_DESCRIPTOR_BYTES",
    "TRANSFER_DESCRIPTOR_PREFIX",
    "is_transfer_descriptor",
    "validate_transfer_descriptor_frame",
]
