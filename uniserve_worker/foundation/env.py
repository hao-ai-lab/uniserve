"""Canonical parsing helpers for worker environment variables.

Boolean flags use allowlist semantics: only ``{"1", "true", "yes", "on"}`` enable
a flag; unrecognized values fall back to the caller-supplied ``default``.
"""
from __future__ import annotations

import os

__all__ = [
    'DEFAULT_ATTENTION_BACKEND',
    'DEFAULT_FALLBACK_BACKEND',
    'DEFAULT_KV_DTYPE',
    'DEFAULT_COMPILE_BACKEND',
    'flag_from_value',
    'env_flag',
    'int_from_value',
    'env_int',
    'env_optional_int',
    'float_from_value',
    'env_optional_float',
    'optional_str_from_value',
    'env_optional_str',
    'env_str',
    'env_int_list',
    'env_optional_flag',
]

_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
_FALSE_TOKENS = frozenset({"0", "false", "no", "off"})

# Tokens that mean "explicitly unset" for an optional-string env var.
_NONE_SENTINELS = frozenset({"", "none", "null"})

# Default backend/dtype strings shared across worker subsystems.
DEFAULT_ATTENTION_BACKEND = "auto"
DEFAULT_FALLBACK_BACKEND = "torch_sdpa"
DEFAULT_KV_DTYPE = "bfloat16"
DEFAULT_COMPILE_BACKEND = "inductor"


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


def float_from_value(raw: str | None, *, default: float, strict: bool = False) -> float:
    """Parse an already-fetched float string (unset/empty -> ``default``)."""
    if raw is None:
        return default
    value = raw.strip()
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        if strict:
            raise ValueError(f"expected a float, got {raw!r}") from None
        return default


def env_optional_float(name: str, *, default: float | None = None, strict: bool = True) -> float | None:
    """Parse an optional float env var.

    Unset/empty -> ``default``. A present-but-malformed value raises
    ``ValueError`` naming ``name`` (``strict``) or returns ``default``.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError:
        if strict:
            raise ValueError(f"{name} must be a float, got {raw!r}") from None
        return default


def optional_str_from_value(raw: str | None, *, default: str | None = None) -> str | None:
    """Parse an already-fetched optional-string value.

    Unset (``None``) -> ``default``; a value in :data:`_NONE_SENTINELS`
    (case-insensitive, after stripping) -> ``None``; otherwise the stripped
    value.
    """
    if raw is None:
        return default
    value = raw.strip()
    if value.lower() in _NONE_SENTINELS:
        return None
    return value


def env_optional_str(name: str, *, default: str | None = None) -> str | None:
    """Parse the optional-string env var ``name`` (see :func:`optional_str_from_value`)."""
    return optional_str_from_value(os.environ.get(name), default=default)


def env_str(name: str, *, default: str) -> str:
    """Parse a required-with-default string env var.

    Unset or empty (after stripping) -> ``default``; otherwise the stripped
    value. Unlike :func:`env_optional_str` the sentinels are not special — use
    this for "name of the thing, defaulting to X" reads.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip()
    return value if value else default


def env_int_list(
    name: str,
    *,
    default: tuple[int, ...],
    minimum: int | None = None,
    strict: bool = False,
) -> tuple[int, ...]:
    """Parse a comma-separated integer list env var.

    Unset/empty -> ``default``. Tokens are split on commas, stripped, and
    parsed; blanks are skipped. When ``minimum`` is set, values below it are
    dropped. A present-but-malformed token raises ``ValueError`` naming ``name``
    (``strict``) or is skipped. An env value that yields no usable ints -> the
    caller decides via the returned empty tuple (callers typically fall back to
    ``default`` themselves; this helper returns the parsed tuple, empty if none).
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    out: list[int] = []
    for token in raw.split(","):
        tok = token.strip()
        if tok == "":
            continue
        try:
            value = int(tok)
        except ValueError:
            if strict:
                raise ValueError(f"{name} must be a comma-separated integer list, got {tok!r}") from None
            continue
        if minimum is not None and value < minimum:
            continue
        out.append(value)
    return tuple(out) if out else default


def env_optional_flag(name: str) -> bool | None:
    """Parse a tri-state flag: unset/sentinel -> ``None``, else allowlist bool."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in _NONE_SENTINELS:
        return None
    return value in _TRUE_TOKENS
