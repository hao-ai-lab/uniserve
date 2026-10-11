"""Request-level numbers of MiniMax-H3 that the model package reads.

The serving path sizes and presents every request in Rust
(``crates/server/src/serving/video``): the canvas and frame count, the size
every condition is prepared at, the Qwen3-VL vision grids and the tokenized
presentation. The rules are checked against
``tests/python/fixtures/minimax_h3_plan.json``, which is generated from the
diffusers reference pipeline. This module states only what Python
computation and tools use: the frame count a duration generates within the
API's range, the aspect ratio ``auto`` means, and the reference limits a
denoiser provisions its condition capacity for.
"""

from __future__ import annotations

import math

from .packing import FPS, align_num_frames

__all__ = [
    "DEFAULT_ASPECT_RATIO",
    "MAX_IMAGE_REFERENCES",
    "MAX_SECONDS",
    "MAX_VIDEO_REFERENCES",
    "MIN_SECONDS",
    "REFERENCE_IMAGE_SHORT_EDGE",
    "frame_count",
]

# The durations the request API admits, in seconds. The aligned frame count
# may run past the upper bound: 15 seconds is 360 frames, aligned up to 362.
MIN_SECONDS = 4.0
MAX_SECONDS = 15.0

# The aspect ratio ``auto`` names for t2va and ref2va requests.
DEFAULT_ASPECT_RATIO = (16, 9)

# The released checkpoint encodes an image reference at a 2048-pixel short
# edge, without an area cap.
REFERENCE_IMAGE_SHORT_EDGE = 2048

# Reference counts the released checkpoint documents for one request.
MAX_IMAGE_REFERENCES = 9
MAX_VIDEO_REFERENCES = 3


def frame_count(seconds: float) -> int:
    """Return the frames a duration generates.

    The duration times 24 rounds half to even and aligns up to the next
    ``17 * n + 5``: 4 seconds give 107 frames, 15 seconds 362.

    Raises:
        ValueError: ``seconds`` lies outside ``[4, 15]``.
    """
    if not math.isfinite(seconds) or not MIN_SECONDS <= seconds <= MAX_SECONDS:
        raise ValueError(
            f"duration must lie in [{MIN_SECONDS:g}, {MAX_SECONDS:g}] "
            f"seconds, got {seconds:g}"
        )
    return align_num_frames(round(seconds * FPS))
