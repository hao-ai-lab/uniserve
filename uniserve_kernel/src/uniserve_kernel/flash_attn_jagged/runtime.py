"""Loads the optional CuTe runtime used by jagged visible-end attention."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path


def _ensure_upstream_cute() -> None:
    """Add the flash-attn-4 CuTe provider to the shared package search path."""

    try:
        from flash_attn.cute import utils as _utils  # noqa: F401

        return
    except ImportError:
        pass

    try:
        dist = metadata.distribution("flash-attn-4")
    except metadata.PackageNotFoundError as exc:
        raise ImportError(
            "FlashAttention CUTE runtime is unavailable. Install flash-attn-4[cu13]."
        ) from exc

    interface = Path(dist.locate_file("flash_attn/cute/interface.py"))
    if not interface.is_file():
        raise ImportError(
            "FlashAttention CUTE runtime is unavailable. The installed "
            "flash-attn-4 package does not contain flash_attn.cute.interface."
        )

    import flash_attn as flash_attn_pkg

    package_path = getattr(flash_attn_pkg, "__path__", None)
    if package_path is None:
        raise ImportError("flash_attn is not a package and cannot expose flash_attn.cute")
    provider_path = str(interface.parent.parent)
    existing = [str(path) for path in package_path]
    if provider_path not in existing:
        flash_attn_pkg.__path__ = [*existing, provider_path]


try:  # pragma: no cover - optional runtime dependency.
    _ensure_upstream_cute()
    from .cute.interface import _flash_attn_fwd as flash_attn_fwd
    from .visible_end import hybrid_multimodal_mask
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "FlashAttention CUTE runtime is unavailable. Install flash-attn-4[cu13] "
        "with its CUTE runtime dependencies."
    ) from exc

__all__ = ["flash_attn_fwd", "hybrid_multimodal_mask"]
