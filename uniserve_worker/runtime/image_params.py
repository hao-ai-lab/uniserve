"""Helpers for request image parameter validation."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..foundation.errors import invalid_descriptor

__all__ = [
    "TextImageGenerationParams",
    "parse_text_image_generation_params",
    "required_image_height",
    "required_image_width",
]


@dataclass(frozen=True)
class TextImageGenerationParams:
    """Validated wire parameters for text-conditioned image denoise models."""

    width: int
    height: int
    steps: int
    cfg_text: float
    cfg_img: float
    cfg_interval: tuple[float, float]
    cfg_norm: str
    cfg_renorm_min: float
    timestep_shift: float
    retain_images: bool
    seed: int | None


def parse_text_image_generation_params(
    params: Mapping[str, Any] | None,
    *,
    cfg: Mapping[str, Any] | None = None,
    owner: str = "image",
    timestep_shift_default: float | None = None,
    retain_images_default: bool = True,
) -> TextImageGenerationParams:
    """Parse the shared text-image generation request contract."""

    raw = params or {}
    cfg_raw = cfg or {}
    steps = _required_positive_int(raw, "steps", owner=owner)
    cfg_interval = _as_float_pair(
        _cfg_or_required(cfg_raw, raw, ("interval",), "cfg_interval", owner=owner),
        "cfg_interval",
        owner=owner,
    )
    timestep_shift_value = raw.get("timestep_shift", timestep_shift_default)
    if timestep_shift_value is None:
        raise invalid_descriptor(f"{owner}.timestep_shift is required")
    return TextImageGenerationParams(
        width=required_image_width(raw, owner=owner),
        height=required_image_height(raw, owner=owner),
        steps=steps,
        cfg_text=float(
            _cfg_or_required(cfg_raw, raw, ("text_scale",), "cfg_text_scale", owner=owner)
        ),
        cfg_img=float(_cfg_or_required(cfg_raw, raw, ("img_scale",), "cfg_img_scale", owner=owner)),
        cfg_interval=cfg_interval,
        cfg_norm=str(
            _cfg_or_required(
                cfg_raw,
                raw,
                ("renorm_type", "renorm"),
                "cfg_renorm_type",
                owner=owner,
            )
        ),
        cfg_renorm_min=float(
            _cfg_or_required(cfg_raw, raw, ("renorm_min",), "cfg_renorm_min", owner=owner)
        ),
        timestep_shift=float(timestep_shift_value),
        retain_images=bool(raw.get("retain_images", retain_images_default)),
        seed=_optional_int(raw.get("seed"), "seed", owner=owner),
    )


def required_image_width(params: Mapping[str, Any] | None, *, owner: str = "image") -> int:
    return _required_positive_int(params, "width", owner=owner)


def required_image_height(params: Mapping[str, Any] | None, *, owner: str = "image") -> int:
    return _required_positive_int(params, "height", owner=owner)


def _required_positive_int(params: Mapping[str, Any] | None, key: str, *, owner: str) -> int:
    value = (params or {}).get(key)
    if value is None:
        raise invalid_descriptor(f"{owner}.{key} is required")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise invalid_descriptor(f"{owner}.{key} must be an integer") from exc
    if parsed <= 0:
        raise invalid_descriptor(f"{owner}.{key} must be positive")
    return parsed


def _cfg_or_required(
    cfg: Mapping[str, Any],
    params: Mapping[str, Any],
    cfg_keys: tuple[str, ...],
    image_key: str,
    *,
    owner: str,
) -> Any:
    for key in cfg_keys:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    return _required_value(params, image_key, owner=owner)


def _required_value(params: Mapping[str, Any], key: str, *, owner: str) -> Any:
    value = params.get(key)
    if value is None:
        raise invalid_descriptor(f"{owner}.{key} is required")
    return value


def _as_float_pair(value: Any, key: str, *, owner: str) -> tuple[float, float]:
    try:
        pair = tuple(value)
    except TypeError as exc:
        raise invalid_descriptor(f"{owner}.{key} must contain exactly two values") from exc
    if len(pair) != 2:
        raise invalid_descriptor(f"{owner}.{key} must contain exactly two values")
    return float(pair[0]), float(pair[1])


def _optional_int(value: Any, key: str, *, owner: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise invalid_descriptor(f"{owner}.{key} must be an integer") from exc
