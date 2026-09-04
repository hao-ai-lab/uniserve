"""Exposes jagged visible-end FlashAttention with optional CuTe runtime loading."""

from __future__ import annotations

from .prefix_bounds import compute_prefix_bounds, compute_prefix_bounds_varlen

_IMPORT_ERROR: BaseException | None = None

try:  # pragma: no cover - optional CUDA/CUTE runtime.
    from .runtime import flash_attn_fwd, hybrid_multimodal_mask
except Exception as exc:  # pragma: no cover
    _IMPORT_ERROR = exc
    flash_attn_fwd = None
    hybrid_multimodal_mask = None


def available() -> bool:
    """Report whether the jagged FlashAttention runtime loaded successfully."""

    return flash_attn_fwd is not None


def import_error() -> BaseException | None:
    """Return the exception that prevented runtime loading, when present."""

    return _IMPORT_ERROR


def require_available() -> None:
    """Require the optional runtime and raise an actionable error when absent."""

    if available():
        return
    detail = f": {_IMPORT_ERROR}" if _IMPORT_ERROR is not None else ""
    raise RuntimeError(
        "uniserve_kernel.flash_attn_jagged is unavailable. Install "
        "uniserve-kernel[flash_attn_jagged] with flash-attn-4[cu13] and its "
        "CUTE dependencies"
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
