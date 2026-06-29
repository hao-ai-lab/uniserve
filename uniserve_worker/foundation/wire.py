"""Shared wire-field validators for the worker IPC boundary.

The runner parses FlatBuffers-decoded dicts in several places; this module holds
the one canonical integer validator so the predicate (reject non-ints and bools,
optionally enforce a lower bound) lives in a single spot.
"""
from __future__ import annotations

from typing import Any

from .errors import invalid_descriptor

__all__ = [
    'wire_int',
]


def wire_int(value: Any, where: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise invalid_descriptor(f"{where} must be an integer")
    if minimum is not None and value < minimum:
        raise invalid_descriptor(f"{where} must be >= {minimum}")
    return value
