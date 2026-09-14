"""Numerical input bounds selected by a model caller."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelLimits:
    """Maximum input token and video frame counts, independent of serving capacity.

    Models align these logical bounds to their mathematical packing. They do not
    describe concurrent requests, physical buffer placement, or pool sizes.
    """

    text_tokens: int
    video_frames: int

    def __post_init__(self) -> None:
        for name in ("text_tokens", "video_frames"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"model limit {name} must be a positive integer")
