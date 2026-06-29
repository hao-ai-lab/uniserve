"""Visible-end multimodal attention provider surface.

This package owns the UniServe-facing API and the visible-end helpers.  The
FlashAttention CUTE runtime itself is an optional external provider dependency;
when it is absent this module stays importable and reports itself unavailable.
"""
from __future__ import annotations

from ._prefix_bounds import compute_prefix_bounds, compute_prefix_bounds_varlen

_IMPORT_ERROR: Exception | None = None

try:  # pragma: no cover - optional CUDA/CUTE runtime.
    from ._fa4_runtime import flash_attn_fwd, hybrid_multimodal_mask
except Exception as exc:  # pragma: no cover
    _IMPORT_ERROR = exc
    flash_attn_fwd = None
    hybrid_multimodal_mask = None


def available() -> bool:
    return flash_attn_fwd is not None


def import_error() -> Exception | None:
    return _IMPORT_ERROR


def require_available() -> None:
    if available():
        return
    detail = f": {_IMPORT_ERROR}" if _IMPORT_ERROR is not None else ""
    raise RuntimeError(
        "uniserve_kernel.mm_attn_varlen is not available. Install/build the "
        "uniserve-kernel provider pack with its CUTE runtime dependencies"
        f"{detail}"
    )


__all__ = [
    "available",
    "compute_prefix_bounds",
    "compute_prefix_bounds_varlen",
    "flash_attn_fwd",
    "hybrid_multimodal_mask",
    "import_error",
    "require_available",
]
