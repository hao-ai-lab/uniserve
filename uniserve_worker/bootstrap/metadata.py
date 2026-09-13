"""Read architecture sidecars and checkpoint shapes before numerical construction."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from ..foundation.errors import unsupported_setup
from ..loader.source import WeightSourceSet
from ..models.bagel import BagelConfig
from ..models.minimax_h3.config import h3_contract, validate_h3_config
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


def h3_manifest(root: Path, contract_path: Path | None = None) -> dict[str, Any]:
    """Read an embedded recipe or a sidecar explicitly bound to this local export.

    Sidecars cannot override embedded manifests. Digests are exporter identity
    declarations, not independently recomputed checkpoint hashes.
    """

    selected = contract_path or os.environ.get("UNISERVE_H3_CONTRACT")
    embedded = root / "fastvideo_inference.json"
    path = Path(selected) if selected else embedded
    if not path.is_file():
        raise ValueError(
            "MiniMax H3 requires fastvideo_inference.json or an explicit UNISERVE_H3_CONTRACT sidecar"
        )
    manifest = read_config(path)
    if selected:
        if embedded.is_file():
            if read_config(embedded) != manifest:
                raise ValueError("sidecar contract disagrees with fastvideo_inference.json")
        elif manifest.get("checkpoint_root") != str(root.resolve()):
            raise ValueError("sidecar checkpoint_root must equal the resolved checkpoint root")
    return manifest


def resolve_h3_contract(root: Path, contract_path: Path | None = None) -> dict[str, Any]:
    """Resolve numerical settings and separately report snapshot provenance."""

    contract = h3_contract(h3_manifest(root, contract_path))
    contract["revision"] = root.name if root.parent.name == "snapshots" else None
    return contract


def h3_metadata(root: Path) -> dict[str, dict[str, Any]]:
    """Read and validate FastH3 numerical metadata without model weights."""

    manifest = h3_manifest(root)
    h3_contract(manifest)
    metadata = {
        name: read_config(root / path)
        for name, path in (
            ("transformer", "transformer/config.json"),
            ("text_encoder", "text_encoder/config.json"),
            ("audio_vae", "audio_vae/config.json"),
            ("video_vae", "vae/config.json"),
            ("scheduler", "scheduler/scheduler_config.json"),
            ("audio_scheduler", "audio_scheduler/scheduler_config.json"),
        )
    }
    metadata["inference"] = manifest
    validate_h3_config(metadata)
    return metadata


def h3_config(
    config: dict[str, Any], root: Path, sources: tuple[WeightSourceSet, ...]
) -> dict[str, Any]:
    """Supply resolved numerical component configuration to H3 construction."""

    return {**config, **h3_metadata(root)}
