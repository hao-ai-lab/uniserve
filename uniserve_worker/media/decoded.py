"""Decoded reference pixel bounds independent of model preprocessing."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(slots=True)
class DecodedMemoryBudget:
    """Account retained decoded buffers and temporary pixel arrays for one bundle.

    Create one budget per request, not per source. Successful decodes retain their
    charge until the bundle is discarded; failed decodes restore the prior charge.
    This is a decoded-array budget, not a bound on codec internals or process RSS.
    """

    limit_bytes: int = 512 * 1024 * 1024
    used_bytes: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.used_bytes <= self.limit_bytes or self.limit_bytes < 1:
            raise ValueError("invalid decoded memory budget")

    def reserve(self, size: int) -> None:
        """Reserve bytes before allocating a retained or temporary decoded array."""
        if size < 0 or size > self.limit_bytes - self.used_bytes:
            raise ValueError("aggregate decoded memory budget exceeded")
        self.used_bytes += size

    def release(self, size: int) -> None:
        """Release an earlier reservation after its storage is no longer needed."""
        if not 0 <= size <= self.used_bytes:
            raise ValueError("invalid decoded memory release")
        self.used_bytes -= size


def decode_reference_image(payload: bytes, *, budget: DecodedMemoryBudget) -> np.ndarray:
    """Decode a single image to owned RGB uint8 [1, height, width, 3] pixels.

    Header dimensions and aggregate storage are checked before Pillow loads pixel
    data. Animated containers are not silently truncated to an image. Conversion
    follows RGB channel conversion, without target-model resizing or augmentation.
    """

    if not payload or len(payload) > 32 * 1024 * 1024:
        raise ValueError("encoded reference image is empty or exceeds 32 MiB")
    reserved = 0
    retained = 0
    try:
        with Image.open(
            io.BytesIO(payload), formats=("PNG", "JPEG", "WEBP", "BMP", "GIF")
        ) as image:
            width, height = image.size
            if min(width, height) < 1 or max(width, height) > 4096:
                raise ValueError("reference image dimensions exceed pixel bounds")
            pixels = width * height
            retained = 3 * pixels
            # Source pixels (up to four bytes/pixel), converted RGB, and the
            # owned NumPy RGB result coexist. Round temporary capacity upward.
            workspace = 8 * pixels
            budget.reserve(retained + workspace)
            reserved = retained + workspace
            # Seeking just the second frame detects animation without enumerating
            # or decoding the entire container. Some plugins load the first frame
            # while seeking, so this must happen after the memory reservation.
            try:
                image.seek(1)
            except EOFError:
                image.seek(0)
            else:
                raise ValueError("image reference must contain exactly one frame")
            image.load()
            with image.convert("RGB") as rgb:
                result = np.array(rgb, dtype=np.uint8, copy=True)[None]
        budget.release(workspace)
        return result
    except (OSError, Image.DecompressionBombError):
        if reserved:
            budget.release(reserved)
        raise ValueError("invalid encoded reference image") from None
    except BaseException:
        if reserved:
            budget.release(reserved)
        raise
