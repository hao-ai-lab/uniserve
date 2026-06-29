"""Attention backend registry."""
from __future__ import annotations

import logging
import threading
from types import MappingProxyType

from .base import AttentionBackend, AttentionCapabilities

__all__ = [
    "AttentionBackend",
    "AttentionCapabilities",
    "register_attention_backend",
    "get_attention_backend",
    "has_attention_backend",
    "list_attention_backends",
    "normalize_attention_backend_name",
]

_log = logging.getLogger("uniserve.attention.registry")

# Registration happens at module-import time, but the registry is process-global
# mutable state; guard mutation/lookup so concurrent importers/registrants cannot
# race on the shared dict.
_LOCK = threading.Lock()

_BACKENDS: dict[str, AttentionBackend] = {}
# Snapshot of the import-time registered backends, captured once the package
# finishes its availability-gated registration; used only to restore the
# default set in ``_reset_for_testing``.
_DEFAULT_BACKENDS: dict[str, AttentionBackend] = {}
_ALIASES = MappingProxyType({
    "sdpa": "torch_sdpa",
    "torch": "torch_sdpa",
    "fa4": "fa4_cute",
    "flash_attn_4": "fa4_cute",
    "flashattention4": "fa4_cute",
    "flash": "flash_attn",
    "flashattention": "flash_attn",
})


def register_attention_backend(name: str, backend: AttentionBackend) -> AttentionBackend:
    key = name.lower()
    with _LOCK:
        existing = _BACKENDS.get(key)
        if existing is not None:
            # Idempotent re-registration of the same backend class is a no-op.
            if type(existing) is type(backend):
                return existing
            raise ValueError(
                f"attention backend {name!r} already registered with a different backend"
            )
        _BACKENDS[key] = backend
    return backend


def get_attention_backend(name: str = "torch_sdpa") -> AttentionBackend:
    name = normalize_attention_backend_name(name)
    with _LOCK:
        try:
            return _BACKENDS[name]
        except KeyError as exc:
            raise ValueError(f"unknown attention backend {name!r}") from exc


def has_attention_backend(name: str) -> bool:
    with _LOCK:
        return normalize_attention_backend_name(name) in _BACKENDS


def list_attention_backends() -> tuple[str, ...]:
    with _LOCK:
        return tuple(sorted(_BACKENDS))


def normalize_attention_backend_name(name: str | None) -> str:
    raw = (name or "auto").lower().replace("-", "_")
    return _ALIASES.get(raw, raw)


def _snapshot_default_backends_for_testing() -> None:
    """Test-only; not part of the public API.

    Record the currently registered backends as the default set restored by
    ``_reset_for_testing``. The package calls this once after its import-time,
    availability-gated registration completes.
    """

    with _LOCK:
        _DEFAULT_BACKENDS.clear()
        _DEFAULT_BACKENDS.update(_BACKENDS)


def _reset_for_testing() -> None:
    """Test-only; not part of the public API.

    Clear the process-global registry state for deterministic test isolation,
    then restore the default-registered backend set so the registry remains
    usable. ``_BACKENDS`` is reset to the snapshot captured at import time (see
    ``_snapshot_default_backends_for_testing``), so callers do not need to
    re-register. The snapshot is preferred over re-running registration because
    optional backends self-register behind availability gates, which a plain
    module re-import would not reproduce.
    """

    with _LOCK:
        _BACKENDS.clear()
        _BACKENDS.update(_DEFAULT_BACKENDS)
