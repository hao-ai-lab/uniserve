"""Read architecture sidecars and checkpoint shapes before numerical construction."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..foundation.errors import unsupported_setup
from ..loader.source import WeightSourceSet
from ..models.bagel import BagelConfig
from ..models.minimax_h3.config import validate_h3_config
from ..models.sensenova.config import NeoChatConfig


def read_config(path: Path) -> dict[str, Any]:
    """Read a required JSON object; malformed metadata never reaches a model."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint metadata {path.name!r} must contain an object")
    return value


def bagel_config(
    config: dict[str, Any], root: Path, sources: tuple[WeightSourceSet, ...]
) -> BagelConfig:
    """Resolve BAGEL's tower configuration and learned positional-grid extent."""

    raw = dict(config)
    for field, filename in (
        ("llm_config", "llm_config.json"),
        ("vit_config", "vit_config.json"),
        ("vae_config", "vae_config.json"),
    ):
        if field not in raw:
            path = root / filename
            if not path.is_file():
                raise unsupported_setup(f"BAGEL checkpoint is missing {filename!r} for {field!r}")
            raw[field] = read_config(path)
    primary = next(source for source in sources if source.source_name == "primary")
    positions = primary.preview_shape("latent_pos_embed.pos_embed")[0]
    size = math.isqrt(positions)
    if size * size != positions:
        raise unsupported_setup(f"BAGEL latent position count {positions} is not square")
    raw["max_latent_size"] = size
    return BagelConfig.from_mapping(raw)


def sensenova_config(
    config: dict[str, Any], root: Path, sources: tuple[WeightSourceSet, ...]
) -> NeoChatConfig:
    """Normalize SenseNova's serialized tower configuration before construction."""

    return NeoChatConfig.from_dict(config)


def h3_metadata(root: Path) -> dict[str, dict[str, Any]]:
    """Read and validate the full FastH3 numerical contract without model weights."""

    if not (root / "fastvideo_inference.json").is_file():
        raise ValueError(
            "MiniMax H3 requires fastvideo_inference.json from the full FastH3 VSA "
            "checkpoint; base partitions and adapter-only checkpoints are unsupported"
        )
    metadata = {
        name: read_config(root / path)
        for name, path in (
            ("inference", "fastvideo_inference.json"),
            ("transformer", "transformer/config.json"),
            ("text_encoder", "text_encoder/config.json"),
            ("audio_vae", "audio_vae/config.json"),
            ("video_vae", "vae/config.json"),
            ("scheduler", "scheduler/scheduler_config.json"),
            ("audio_scheduler", "audio_scheduler/scheduler_config.json"),
        )
    }
    validate_h3_config(metadata)
    return metadata


def h3_config(
    config: dict[str, Any], root: Path, sources: tuple[WeightSourceSet, ...]
) -> dict[str, Any]:
    """Supply resolved numerical component configuration to H3 construction."""

    return {**config, **h3_metadata(root)}
