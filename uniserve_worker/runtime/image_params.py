"""Helpers for request image parameter validation."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..foundation.errors import invalid_descriptor

__all__ = [
    "required_image_height",
    "required_image_width",
]


def required_image_width(params: Mapping[str, Any] | None, *, owner: str = "image") -> int:
    return _required_positive_int(params, "width", owner=owner)


def required_image_height(params: Mapping[str, Any] | None, *, owner: str = "image") -> int:
    return _required_positive_int(params, "height", owner=owner)


def _required_positive_int(params: Mapping[str, Any] | None, key: str, *, owner: str) -> int:
    value = (params or {}).get(key)
    if value is None:
        raise invalid_descriptor(f"{owner}.{key} is required")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise invalid_descriptor(f"{owner}.{key} must be an integer") from exc
    if parsed <= 0:
        raise invalid_descriptor(f"{owner}.{key} must be positive")
    return parsed
