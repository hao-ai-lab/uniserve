"""Frame extent and spatial dimensions of a numerical video."""

from dataclasses import dataclass

from . import image


@dataclass(frozen=True, slots=True)
class Config:
    num_frames: int
    frame: image.Config

    def __post_init__(self):
        if type(self.num_frames) is not int or self.num_frames < 1:
            raise ValueError("video frame count must be a positive integer")
        if not isinstance(self.frame, image.Config):
            raise TypeError("video frame dimensions must use image.Config")
