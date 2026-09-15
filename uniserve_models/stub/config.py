"""Immutable numerical configuration for the serving simulator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Config:
    patch_size: int = 16

    def __post_init__(self):
        if type(self.patch_size) is not int or self.patch_size < 1:
            raise ValueError(
                "simulation patches must have a positive pixel width"
            )
