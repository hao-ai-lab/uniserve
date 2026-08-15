"""Canonical parsing helpers for worker environment variables.

Boolean flags use allowlist semantics: only ``{"1", "true", "yes", "on"}`` enable
a flag; unrecognized values fall back to the caller-supplied ``default``.
"""
from __future__ import annotations

import os

__all__ = [
    "flag_from_value",
    "env_flag",
    "int_from_value",
    "env_int",
    "env_optional_int",
    "env_optional_flag",
]

_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
_FALSE_TOKENS = frozenset({"0", "false", "no", "off"})
_NONE_SENTINELS = frozenset({"", "none", "null"})


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


def env_flag(name: str, *, default: bool = False) -> bool:
    """Parse the boolean env var ``name`` with allowlist semantics."""
    return flag_from_value(os.environ.get(name), default=default)


def int_from_value(raw: str | None, *, default: int, strict: bool = False) -> int:
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


def env_int(name: str, *, default: int, strict: bool = False) -> int:
    """Parse the integer env var ``name``; fall back to ``default``.

    With ``strict=True`` a present-but-malformed value raises ``ValueError``
    naming ``name`` rather than silently using ``default``.
    """
    try:
        return int_from_value(os.environ.get(name), default=default, strict=strict)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer: {exc}") from None


def env_optional_int(name: str, *, default: int | None = None, strict: bool = True) -> int | None:
    """Parse an optional integer env var.

    Unset/empty -> ``default`` (``None`` by default). A present-but-malformed
    value raises ``ValueError`` naming ``name`` (``strict``) or returns
    ``default``.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        if strict:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from None
        return default


def env_optional_flag(name: str) -> bool | None:
    """Parse a tri-state flag: unset/sentinel -> ``None``, else allowlist bool."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in _NONE_SENTINELS:
        return None
    return value in _TRUE_TOKENS
