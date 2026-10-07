"""Numerical media encoding and framed tensor rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from uniserve_worker._uniserve_ipc import (
    frame_encoded_unit as frame_encoded_unit,
)
from uniserve_worker._uniserve_ipc import (
    read_encoded_unit as read_encoded_unit,
)
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.media.container import (
    AvMuxConfig,
    encode_video_unit,
    encoded_video_bytes,
    require_media_codecs,
)

if TYPE_CHECKING:
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
