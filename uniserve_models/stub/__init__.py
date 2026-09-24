"""Deterministic weightless model that stands in for a checkpoint.

A worker launched with ``no_model`` (which also requires ``allow_stub``)
builds ``Model`` and ``image_processor`` instead of loading a checkpoint, so
the serving path runs end to end without weights and yields predictable
tokens, cache writes and images. No checkpoint catalog registers ``Model``.
The Rust ``WorkerInfo`` validation accepts an empty checkpoint identity only
for model names under this package's module path, so the path is part of
that contract.
"""

from .config import Config
from .denoiser import Denoiser
from .inputs import DenoiserInput
from .model import (
    STUB_EOS_TOKEN_ID,
    STUB_IMG_START_TOKEN_ID,
    Model,
    entry_points,
)
from .processing import image_processor

__all__ = [
    "image_processor",
    "Config",
    "DenoiserInput",
    "Denoiser",
    "Model",
    "STUB_EOS_TOKEN_ID",
    "STUB_IMG_START_TOKEN_ID",
    "entry_points",
]
