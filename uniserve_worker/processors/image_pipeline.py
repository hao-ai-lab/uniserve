"""Family-owned multimodal input preparation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "ImageInputPipeline",
    "PreparedImageInput",
]


@dataclass(frozen=True)
class PreparedImageInput:
    pixels: Any | None = None
    grid: Any | None = None
    metadata: dict[str, Any] | None = None


class ImageInputPipeline:
    """Adapts concrete processor helpers to domain-shaped prepared inputs."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def prepare_understanding(self, op: dict[str, Any]) -> PreparedImageInput:
        helper = getattr(self.processor, "prepare_from_b64", None)
        if callable(helper) and op.get("image_b64") is not None:
            value = helper(op["image_b64"])
            return _coerce_prepared(value)
        helper = getattr(self.processor, "understanding_patches", None)
        if callable(helper):
            return _coerce_prepared(helper(op))
        return PreparedImageInput(metadata=dict(op))

    def prepare_latent_encode(self, op: dict[str, Any]) -> PreparedImageInput:
        helper = getattr(self.processor, "prepare_latent_encode", None)
        if callable(helper):
            return _coerce_prepared(helper(op))
        return self.prepare_understanding(op)

    def prepare_vision_encode(self, op: dict[str, Any]) -> PreparedImageInput:
        helper = getattr(self.processor, "prepare_vision_encode", None)
        if callable(helper):
            return _coerce_prepared(helper(op))
        return self.prepare_understanding(op)


def _coerce_prepared(value: Any) -> PreparedImageInput:
    if isinstance(value, PreparedImageInput):
        return value
    if isinstance(value, dict):
        return PreparedImageInput(
            pixels=value.get("pixels"),
            grid=value.get("grid"),
            metadata=dict(value),
        )
    if isinstance(value, tuple):
        pixels = value[0] if len(value) > 0 else None
        grid = value[1] if len(value) > 1 else None
        return PreparedImageInput(pixels=pixels, grid=grid)
    return PreparedImageInput(pixels=value)
