"""Canonical parsing helpers for worker environment variables.

Boolean flags use allowlist semantics: only ``{"1", "true", "yes", "on"}``
enable a flag; unrecognized values fall back to the caller-supplied ``default``.
"""

from __future__ import annotations

__all__ = ["flag_from_value", "int_from_value"]

_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
_FALSE_TOKENS = frozenset({"0", "false", "no", "off"})


def flag_from_value(raw: str | None, *, default: bool = False) -> bool:
    """Parse an already-fetched flag string.

    Unset (``None``) or empty -> ``default``; a recognized true/false token maps
    accordingly; any unrecognized value falls back to ``default``.
    """
    if raw is None:
        return default
    value = raw.strip().lower()
    if value == "":
        return default
    if value in _TRUE_TOKENS:
        return True
    if value in _FALSE_TOKENS:
        return False
    return default


def int_from_value(
    raw: str | None, *, default: int, strict: bool = False
) -> int:
    """Parse an already-fetched integer string.

    Unset (``None``) or empty -> ``default``. A present-but-unparseable value
    falls back to ``default`` (``strict=False``) or raises ``ValueError`` with a
    friendly message naming the offending value (``strict=True``).
    """
    if raw is None:
        return default
    value = raw.strip()
    if value == "":
        return default
    try:
        return int(value)
    except ValueError:
        if strict:
            raise ValueError(f"expected an integer, got {raw!r}") from None
        return default
