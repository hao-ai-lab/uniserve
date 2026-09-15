"""Spatial dimensions of one image or video frame."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Config:
    height: int
    width: int

    def __post_init__(self):
        if any(type(size) is not int or size < 1 for size in (self.height, self.width)):
            raise ValueError("image height and width must be positive integers")
