"""FlashInfer workspace and backend knobs resolved at worker launch."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["FlashInferTuningConfig"]


@dataclass(frozen=True)
class FlashInferTuningConfig:
    """Controls FlashInfer workspace size, kernel families, split-KV policy, and fast planning."""

    workspace_size: int = 512 * 1024 * 1024
    use_tensor_core: bool | None = None
    decode_backend: str = "fa2"
    prefill_backend: str = "auto"
    decode_split_tile_size: int | None = None
    prefill_split_tile_size: int | None = None
    disable_split_kv: bool = False
    fast_decode_plan: bool = True
