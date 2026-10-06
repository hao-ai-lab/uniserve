"""Numerical media encoding and framed tensor rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.media.container import (
    AvMuxConfig,
    encode_video_unit,
    encoded_video_bytes,
    require_media_codecs,
)

if TYPE_CHECKING:
    import torch

    from uniserve_worker._uniserve_ipc import SharedRead

__all__ = [
    "AvMuxConfig",
    "encode_unit",
    "encoded_unit_bytes",
    "require_media_codecs",
]

# A framed media unit carries its own length because the product row that holds
# it is sized for the largest unit a request can produce. The prefix is a
# uint64 in the host's native byte order, written by `frame_encoded_unit` and
# read by `read_encoded_unit`.
_LENGTH_BYTES = 8


def encoded_unit_bytes(frames: int, height: int, width: int) -> int:
    """Storage bound for a serving codec unit, including its length prefix."""
    return _LENGTH_BYTES + encoded_video_bytes(frames, height, width)


def frame_encoded_unit(
    payload: bytes, destination: torch.Tensor
) -> torch.Tensor:
    """Write a framed unit and return its initialized prefix for export.

    ``destination`` is a 1-D uint8 row sized by `encoded_unit_bytes`. Bytes
    past the returned view are left untouched; the caller exports only that
    initialized span as the row's region.

    Raises:
        WorkerError: When ``payload`` does not fit in the row after the
            length prefix.
    """
    import torch

    capacity = int(destination.numel()) - _LENGTH_BYTES
    if len(payload) > capacity:
        raise invalid_descriptor(
            f"encoded media unit of {len(payload)} bytes exceeds the "
            f"{capacity} bytes reserved for it"
        )
    header = np.frombuffer(
        np.uint64(len(payload)).tobytes(), dtype=np.uint8
    ).copy()
    destination[:_LENGTH_BYTES].copy_(torch.from_numpy(header))
    body = np.frombuffer(payload, dtype=np.uint8).copy()
    destination[_LENGTH_BYTES : _LENGTH_BYTES + len(payload)].copy_(
        torch.from_numpy(body)
    )
    return destination[: _LENGTH_BYTES + len(payload)]


def read_encoded_unit(row: torch.Tensor) -> bytes:
    """Return the encoded media unit a framed CPU uint8 row carries.

    Raises:
        WorkerError: When the length prefix exceeds the row's capacity.
    """
    values = row.numpy()
    length = int(
        np.frombuffer(values[:_LENGTH_BYTES].tobytes(), dtype=np.uint64)[0]
    )
    if length > len(values) - _LENGTH_BYTES:
        raise invalid_descriptor("encoded media unit names an invalid length")
    return values[_LENGTH_BYTES : _LENGTH_BYTES + length].tobytes()


def encode_unit(config: AvMuxConfig, source: SharedRead | np.ndarray) -> bytes:
    """Encode RGB24 frames without copying borrowed bytes.

    A borrowed unit stays in its producer's shared-storage segment, mapped
    throughout the codec call.
    """
    raw = (
        source
        if isinstance(source, np.ndarray)
        else np.frombuffer(source, dtype=np.uint8)
    )
    raster = config.height * config.width * 3
    if raw.size % raster != 0:
        raise invalid_descriptor("video capture has invalid RGB24 dimensions")
    return encode_video_unit(
        config, raw.reshape(-1, config.height, config.width, 3)
    )


def host_array(value: torch.Tensor) -> np.ndarray:
    """Copy an imported CPU tensor for a codec that outlives its read lease."""
    import torch

    return value.detach().contiguous().view(torch.uint8).numpy().copy()
