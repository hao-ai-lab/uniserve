"""Read architecture sidecars and checkpoint shapes before numerical construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from uniserve.loading.source import WeightSourceSet
from uniserve_models.bagel import BagelConfig
from uniserve_models.bagel import read_config as read_bagel_config
from uniserve_models.minimax_h3.config import H3Config
from uniserve_models.minimax_h3.config import read_config as read_h3_config
from uniserve_models.sensenova.config import NeoChatConfig
from uniserve_models.sensenova.config import read_config as read_neo_config


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
                raise ValueError(f"BAGEL checkpoint is missing {filename!r} for {field!r}")
            raw[field] = read_config(path)
    primary = next(source for source in sources if source.source_name == "primary")
    return read_bagel_config(
        raw, latent_position_shape=primary.preview_shape("latent_pos_embed.pos_embed")
    )


def sensenova_config(
    config: dict[str, Any], root: Path, sources: tuple[WeightSourceSet, ...]
) -> NeoChatConfig:
    """Normalize SenseNova's serialized tower configuration before construction."""

    return read_neo_config(config)


def h3_metadata(root: Path) -> H3Config:
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
    return read_h3_config(metadata)


def h3_config(config: dict[str, Any], root: Path, sources: tuple[WeightSourceSet, ...]) -> H3Config:
    """Supply resolved numerical component configuration to H3 construction."""

    return h3_metadata(root)
