"""Primitive validation for decoded worker wire values.

The ``from_mapping`` constructors of the `uniserve_worker.protocol` records
use these helpers to type-check each mapping field; `worker_info` keeps local
variants instead. Every helper takes the decoded value and `where`, the path
of the field (`_enum` also takes the enum type first); the checks a helper
performs raise `invalid_descriptor` with that path in the message. Integer
and number helpers reject `bool`, although it subclasses `int`.

Most helpers test for the exact built-in type first and then fall back to
`isinstance`, so subclasses and other `Mapping` or `Sequence` implementations
are accepted as well.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, TypeVar, cast

from uniserve_worker.errors import invalid_descriptor

_E = TypeVar("_E", bound=StrEnum)


def _enum(kind: type[_E], value: object, where: str) -> _E:
    """Decode and validate one string-backed enum value for a wire field."""
    if type(value) is str:
        member = kind._value2member_map_.get(value)
        if member is not None:
            return cast(_E, member)
        raise invalid_descriptor(f"{where} has unknown value {value!r}")
    if not isinstance(value, str):
        raise invalid_descriptor(f"{where} must be a string")
    try:
        return kind(value)
    except ValueError:
        raise invalid_descriptor(
            f"{where} has unknown value {value!r}"
        ) from None


def _map(value: object, where: str) -> Mapping[str, Any]:
    """Require a decoded wire field to be a mapping."""
    if type(value) is dict:
        return value
    if not isinstance(value, Mapping):
        raise invalid_descriptor(f"{where} must be a map")
    return cast(Mapping[str, Any], value)


def _seq(value: object, where: str) -> Sequence[Any]:
    """Require a decoded wire field to be a non-string sequence."""
    kind = type(value)
    if kind is list or kind is tuple:
        return cast(Sequence[Any], value)
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise invalid_descriptor(f"{where} must be a list")
    return value


def _pair(value: object, where: str) -> Sequence[Any]:
    """Require a decoded wire field to contain exactly two values."""
    items = _seq(value, where)
    if len(items) != 2:
        raise invalid_descriptor(f"{where} must contain two values")
    return items


def _tagged(value: object, where: str) -> tuple[str, object]:
    """Split an adjacently tagged ``{"kind": ..., "value": ...}`` mapping.

    Returns the tag, which must be a string, and the payload unvalidated;
    the caller validates the payload according to the tag.
    """
    data = _map(value, where)
    return _str(data.get("kind"), f"{where}.kind"), data.get("value")


def _str(value: object, where: str) -> str:
    """Require a decoded wire field to contain text."""
    if not isinstance(value, str):
        raise invalid_descriptor(f"{where} must be a string")
    return value


def _bool(value: object, where: str) -> bool:
    """Require a decoded wire field to contain a boolean."""
    if value is True or value is False:
        return value
    raise invalid_descriptor(f"{where} must be a bool")


def _uint(value: object, where: str) -> int:
    """Decode a non-negative integer wire field."""
    if type(value) is int and value >= 0:
        return value
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise invalid_descriptor(f"{where} must be a non-negative integer")
    return value


def _int(value: object, where: str) -> int:
    """Decode a signed integer wire field while rejecting booleans.

    An `int` subclass is returned as a plain `int`.
    """
    if type(value) is int:
        return value
    if not isinstance(value, int) or isinstance(value, bool):
        raise invalid_descriptor(f"{where} must be an integer")
    return int(value)


def _optional_uint(value: object, where: str) -> int | None:
    """Decode an optional non-negative integer wire field."""
    return None if value is None else _uint(value, where)


def _float(value: object, where: str) -> float:
    """Decode a finite floating-point wire field while rejecting booleans."""
    kind = type(value)
    if kind is not float and kind is not int:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise invalid_descriptor(f"{where} must be a number")

    result = float(cast(int | float, value))
    if not math.isfinite(result):
        raise invalid_descriptor(f"{where} must be finite")
    return result


def _uints(value: object, where: str) -> tuple[int, ...]:
    """Decode a sequence of non-negative integer wire values."""
    items = _seq(value, where)
    # Fast path: copy directly when every item is an exact non-negative int.
    # Otherwise validate per item, which also accepts int subclasses other
    # than bool and names the index of the first invalid item.
    for item in items:
        if not (type(item) is int and item >= 0):
            return tuple(
                _uint(item, f"{where}[{index}]")
                for index, item in enumerate(items)
            )
    return tuple(items)


def _ints(value: object, where: str) -> tuple[int, ...]:
    """Decode a sequence of integer wire values."""
    items = _seq(value, where)
    if not all(
        isinstance(item, int) and not isinstance(item, bool) for item in items
    ):
        raise invalid_descriptor(f"{where} must contain integers")
    return tuple(int(item) for item in items)


def _bytes(value: object, where: str) -> bytes:
    """Decode a bytes-like wire payload to immutable bytes.

    Accepts `bytes`, `bytearray`, `memoryview`, or a non-string sequence of
    byte values, which `bytes` converts.
    """
    if type(value) is bytes:
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    return bytes(_uints(value, where))


def _nonnegative(value: int, where: str) -> None:
    """Validate that an already decoded integer field is non-negative."""
    _uint(value, where)
