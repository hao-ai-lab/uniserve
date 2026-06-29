"""Loader for the optional FlashAttention CUTE visible-end runtime."""
from __future__ import annotations

try:  # pragma: no cover - optional runtime dependency.
    from flash_attn.cute.interface import _flash_attn_fwd as flash_attn_fwd  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "FlashAttention CUTE runtime is unavailable. Install a provider package "
        "that exposes flash_attn.cute.interface._flash_attn_fwd."
    ) from exc

from ._visible_end_mask import hybrid_multimodal_mask

__all__ = ["flash_attn_fwd", "hybrid_multimodal_mask"]
