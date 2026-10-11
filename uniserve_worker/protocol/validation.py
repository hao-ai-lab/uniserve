"""Primitive validation for decoded worker wire values.

The ``from_mapping`` parsers of the `uniserve_worker.protocol` records that
are read from a mapping (the launch descriptor's `CanvasSampling` and the
records nested in `WorkerInfo`) use these helpers to type-check each field,
and record constructors use `_nonnegative`; `worker_info` keeps local
variants instead. Every helper takes the decoded value and `where`, the path
of the field; the checks a helper performs raise `invalid_descriptor` with
that path in the message. Integer and number helpers reject `bool`, although
it subclasses `int`.

Most helpers test for the exact built-in type first and then fall back to
`isinstance`, so subclasses and other `Mapping` or `Sequence` implementations
are accepted as well.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

from uniserve_worker.errors import invalid_descriptor


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


def _uint(value: object, where: str) -> int:
    """Decode a non-negative integer wire field."""
    if type(value) is int and value >= 0:
        return value
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise invalid_descriptor(f"{where} must be a non-negative integer")
    return value


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


def _nonnegative(value: int, where: str) -> None:
    """Validate that an already decoded integer field is non-negative."""
    _uint(value, where)
