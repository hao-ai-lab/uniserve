"""Normalize the fixed FastH3 checkpoint architecture and diffusion recipe."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from . import audio_vae, output, video_vae
from .encoder import TextEncoderConfig

FASTH3_LADDER = (1000, 750, 500, 250)
FASTH3_SHIFTS = (12.0, 3.0)
FASTH3_8_STEP_LADDER = (1000, 875, 750, 625, 500, 375, 250, 125)
FASTH3_8_STEP_SHIFTS = (10.0, 3.0)
FASTH3_TIME_SCALE = 1000.0

FASTH3_MODEL_ID = "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"
FASTH3_REVISION = "5ea076f35b84da4c3c82217112fa733d8eea2ae1"
FASTH3_8_STEP_MODEL_ID = "FastVideo/FastVideo-FastH3-8-Step-V2"
FASTH3_8_STEP_REVISION = "3da2ddfe1954d9cda4c05b643dc0f26007a655c5"

FASTH3_VARIANTS = {
    FASTH3_MODEL_ID: {
        "revision": FASTH3_REVISION,
        "checkpoint_content_sha256": (
            "b36987515e4c75fa4c7aaa632a7842c829ea141b235358a54d782b51230497b3"
        ),
        "checkpoint_metadata_sha256": (
            "dcad0fbee2a7c7e75e53435f4fd98fccf3138844883874edf057962ab48fa428"
        ),
        "fastvideo_commit": "48a047c05ff4138f20cfa33351499c6ec5945f5d",
        "transformer_forwards": 4,
        "num_inference_steps": 5,
        "dmd_denoising_steps": [999, 749, 500, 250],
        "vsa_sparsity": 0.9,
        "ladder": FASTH3_LADDER,
        "shifts": FASTH3_SHIFTS,
    },
    FASTH3_8_STEP_MODEL_ID: {
        "revision": FASTH3_8_STEP_REVISION,
        "checkpoint_content_sha256": (
            "516323fa396fa5dff4e82669d4e9a08a5791692a3d3b98ff6bc3de3fc6a33d11"
        ),
        "checkpoint_metadata_sha256": (
            "ca9f2d609c05742ba465d24989981ec02cca26acb6ca2f163dc0f6dc8d11c27b"
        ),
        "fastvideo_commit": "24bbe7fddd05ca6f2c34b3dbed06ac1c75b72086",
        "transformer_forwards": 8,
        "num_inference_steps": 9,
        "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125],
        "vsa_sparsity": 0.8,
        "ladder": FASTH3_8_STEP_LADDER,
        "shifts": FASTH3_8_STEP_SHIFTS,
    },
}


def _validate_manifest(manifest: Mapping[str, object]) -> Mapping[str, object]:
    """Validate a supported full VSA checkpoint before allocating components.

    The manifest's training indices do not select the inference schedule. The
    numerical ladder is owned by the pinned deployment contract for each
    checkpoint variant.
    """
    model_id = manifest.get("model_id")
    variant = FASTH3_VARIANTS.get(model_id)
    if variant is None:
        raise ValueError(
            f"unsupported FastH3 checkpoint: model_id must be one of "
            f"{tuple(FASTH3_VARIANTS)}, got {model_id!r}"
        )
    expected = {
        "schema_version": "fasth3-inference-contract-v1",
        "model_id": model_id,
        "task": "t2av",
        "guidance_scale": 1.0,
        "attention_backend": "VIDEO_SPARSE_ATTN_H3",
        "vsa_tile_size": 64,
        **{
            name: variant[name]
            for name in (
                "checkpoint_content_sha256",
                "checkpoint_metadata_sha256",
                "fastvideo_commit",
                "transformer_forwards",
                "num_inference_steps",
                "dmd_denoising_steps",
                "vsa_sparsity",
            )
        },
    }
    for name, value in expected.items():
        if (
            type(manifest.get(name)) is not type(value)
            or manifest.get(name) != value
        ):
            raise ValueError(
                f"unsupported FastH3 checkpoint: {name} must be {value!r}, "
                f"got {manifest.get(name)!r}; "
                f"use {model_id}@{variant['revision']}"
            )
    return variant


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    """Define H3 widths, layers, attention, experts, modulation, and sparse-video layout."""  # noqa: E501

    hidden_size: int = 5376
    num_attention_heads: int = 56
    head_dim: int = 128
    num_hidden_layers: int = 50
    num_refiner_layers: int = 2
    intermediate_size: int = 14336
    video_channels: int = 24
    audio_channels: int = 32
    text_dim: int = 5120
    frequency_dim: int = 256
    time_hidden_dim: int = 5376
    time_dim: int = 2688
    rope_frequency_dim: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    vsa_sparsity: float = 0.9

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "num_attention_heads",
            "head_dim",
            "num_hidden_layers",
            "num_refiner_layers",
            "intermediate_size",
            "video_channels",
            "audio_channels",
            "text_dim",
            "frequency_dim",
            "time_hidden_dim",
            "time_dim",
            "rope_frequency_dim",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(
                    f"H3 transformer {name} must be a positive integer"
                )
        for name in ("rope_theta", "norm_eps", "qk_norm_eps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"H3 transformer {name} must be finite and positive"
                )
        if (
            not math.isfinite(self.vsa_sparsity)
            or not 0 <= self.vsa_sparsity < 1
        ):
            raise ValueError("H3 transformer VSA sparsity must lie in [0, 1)")
        if (
            self.frequency_dim % 2
            or self.rope_frequency_dim * 6 > self.head_dim
        ):
            raise ValueError(
                "H3 transformer rotary dimensions must fit its attention "
                "num_attention_heads"
            )


@dataclass(frozen=True, slots=True)
class DiffusionConfig:
    """A checkpoint-owned FastH3 clean-sample Euler recipe."""

    ladder: tuple[int, ...] = FASTH3_LADDER
    video_shift: float = FASTH3_SHIFTS[0]
    audio_shift: float = FASTH3_SHIFTS[1]
    time_scale: float = FASTH3_TIME_SCALE

    def __post_init__(self) -> None:
        if (
            not isinstance(self.ladder, tuple)
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in self.ladder
            )
            or self.ladder not in {FASTH3_LADDER, FASTH3_8_STEP_LADDER}
        ):
            raise ValueError("FastH3 requires a supported checkpoint ladder")
        expected_shifts = (
            FASTH3_SHIFTS
            if self.ladder == FASTH3_LADDER
            else FASTH3_8_STEP_SHIFTS
        )
        if (self.video_shift, self.audio_shift) != expected_shifts or (
            self.time_scale != FASTH3_TIME_SCALE
        ):
            raise ValueError(
                "FastH3 requires its trained video/audio shifts and time scale"
            )


@dataclass(frozen=True, slots=True)
class Config:
    """Compose the fixed FastH3 networks and their diffusion mathematics."""

    text_encoder: TextEncoderConfig = TextEncoderConfig()
    denoiser: TransformerConfig = TransformerConfig()
    video_decoder: video_vae.Config = video_vae.Config()
    audio_decoder: audio_vae.Config = audio_vae.Config()
    diffusion: DiffusionConfig = DiffusionConfig()
    output: output.Config = output.Config()

    def __post_init__(self) -> None:
        if self.text_encoder.hidden_size != self.denoiser.text_dim:
            raise ValueError(
                "H3 text features must match denoiser conditioning width"
            )
        if self.video_decoder.latent_channels != self.denoiser.video_channels:
            raise ValueError("H3 video latent channels must match the denoiser")
        if self.audio_decoder.latent_channels != self.denoiser.audio_channels:
            raise ValueError("H3 audio latent channels must match the denoiser")
        # Packing, sparse attention, native reconstruction and checkpoint
        # identity implement this architecture. Typed configs do not imply
        # arbitrary variants.
        for name, expected in (
            ("text_encoder", TextEncoderConfig()),
            ("denoiser", TransformerConfig()),
            ("video_decoder", video_vae.Config()),
            ("audio_decoder", audio_vae.Config()),
            ("output", output.Config()),
        ):
            actual = getattr(self, name)
            for field in fields(expected):
                if field.name in {
                    "latents_mean",
                    "latents_std",
                    "vsa_sparsity",
                }:
                    continue
                value = getattr(actual, field.name)
                supported = getattr(expected, field.name)
                if value != supported:
                    raise ValueError(
                        f"FastH3 {name}.{field.name} must be {supported!r}, "
                        f"got {value!r}"
                    )
        expected_sparsity = (
            0.9 if self.diffusion.ladder == FASTH3_LADDER else 0.8
        )
        if self.denoiser.vsa_sparsity != expected_sparsity:
            raise ValueError(
                "FastH3 denoiser VSA sparsity must match its checkpoint ladder"
            )


# Checkpoint field names differ from the mathematical modules' established
# names.
# The reader and native weight-name enumeration use this single correspondence.
TRANSFORMER_FIELDS = {
    "num_attention_heads": "num_attention_heads",
    "attention_head_dim": "head_dim",
    "hidden_size": "hidden_size",
    "num_layers": "num_hidden_layers",
    "num_refiner_layers": "num_refiner_layers",
    "ffn_dim": "intermediate_size",
    "in_channels": "video_channels",
    "audio_in_channels": "audio_channels",
    "text_dim": "text_dim",
    "freq_dim": "frequency_dim",
    "time_embed_hidden_dim": "time_hidden_dim",
    "time_embed_dim": "time_dim",
    "rope_freq_dim": "rope_frequency_dim",
    "rope_theta": "rope_theta",
    "norm_eps": "norm_eps",
    "qk_norm_eps": "qk_norm_eps",
}
TEXT_FIELDS = {
    "vocab_size": "vocab_size",
    "hidden_size": "hidden_size",
    "intermediate_size": "intermediate_size",
    "num_hidden_layers": "num_checkpoint_layers",
    "num_attention_heads": "num_attention_heads",
    "num_key_value_heads": "num_key_value_heads",
    "head_dim": "head_dim",
    "rope_theta": "rope_theta",
    "rms_norm_eps": "rms_norm_eps",
    "max_position_embeddings": "max_position_embeddings",
}


def _normalize(metadata: Mapping[str, Mapping[str, Any]]) -> Config:
    """Normalize the seven checkpoint sidecars without allocating model resources."""  # noqa: E501
    for name in (
        "inference",
        "transformer",
        "text_encoder",
        "video_vae",
        "audio_vae",
        "scheduler",
        "audio_scheduler",
    ):
        if not isinstance(metadata.get(name), Mapping):
            raise ValueError(f"FastH3 requires {name} metadata")
    variant = _validate_manifest(metadata["inference"])
    transformer = metadata["transformer"]
    missing = set(TRANSFORMER_FIELDS) - transformer.keys()
    if missing:
        raise ValueError(
            f"FastH3 transformer is missing fields: "
            f"{', '.join(sorted(missing))}"
        )
    if transformer.get("patch_size") != [1, 2, 2]:
        raise ValueError("FastH3 transformer patch_size must be [1, 2, 2]")
    if transformer.get("final_norm_eps") != transformer.get("norm_eps"):
        raise ValueError(
            "FastH3 transformer final_norm_eps must equal norm_eps"
        )
    denoiser = TransformerConfig(
        vsa_sparsity=variant["vsa_sparsity"],
        **{
            target: transformer[source]
            for source, target in TRANSFORMER_FIELDS.items()
        },
    )

    text = metadata["text_encoder"].get("text_config")
    if not isinstance(text, dict):
        raise ValueError("FastH3 text encoder requires text_config")
    for name, expected in (("hidden_act", "silu"), ("attention_bias", False)):
        if text.get(name) != expected:
            raise ValueError(f"unsupported FastH3 text encoder {name}")
    if metadata["text_encoder"].get("tie_word_embeddings", False):
        raise ValueError(
            "FastH3 checkpoint requires an independent vocabulary head"
        )
    if text.get("rope_scaling") != {
        "mrope_interleaved": True,
        "mrope_section": [24, 20, 20],
        "rope_type": "default",
    }:
        raise ValueError("unsupported FastH3 text encoder rope_scaling")
    # These omitted visual blocks remain part of exact checkpoint name matching.
    vision = metadata["text_encoder"].get("vision_config")
    if (
        not isinstance(vision, dict)
        or vision.get("depth") != 27
        or vision.get("deepstack_visual_indexes") != [8, 16, 24]
    ):
        raise ValueError("unsupported FastH3 checkpoint visual layer layout")
    missing = set(TEXT_FIELDS) - text.keys()
    if missing:
        raise ValueError(
            f"FastH3 text_encoder.text_config is missing fields: "
            f"{', '.join(sorted(missing))}"
        )
    encoder = TextEncoderConfig(
        **{target: text[source] for source, target in TEXT_FIELDS.items()}
    )

    video_values = {}
    for field in fields(video_vae.Config):
        if field.name not in metadata["video_vae"]:
            raise ValueError(f"FastH3 video_vae is missing field {field.name}")
        value = metadata["video_vae"][field.name]
        if isinstance(field.default, tuple):
            if not isinstance(value, (tuple, list)):
                raise ValueError(
                    f"FastH3 video_vae.{field.name} must be a sequence"
                )
            value = tuple(value)
        video_values[field.name] = value

    audio_values = {}
    for field in fields(audio_vae.Config):
        if field.name not in metadata["audio_vae"]:
            raise ValueError(f"FastH3 audio_vae is missing field {field.name}")
        value = metadata["audio_vae"][field.name]
        if isinstance(field.default, tuple) and not isinstance(
            value, (tuple, list)
        ):
            raise ValueError(
                f"FastH3 audio_vae.{field.name} must be a sequence"
            )
        if field.name == "resblock_dilation_sizes":
            if any(not isinstance(row, (tuple, list)) for row in value):
                raise ValueError(
                    "FastH3 audio_vae.resblock_dilation_sizes "
                    "must contain sequences"
                )
            value = tuple(tuple(row) for row in value)
        elif isinstance(field.default, tuple):
            value = tuple(value)
        audio_values[field.name] = value

    for name in ("scheduler", "audio_scheduler"):
        if "shift" not in metadata[name]:
            raise ValueError(f"FastH3 {name} is missing field shift")

    return Config(
        text_encoder=encoder,
        denoiser=denoiser,
        video_decoder=video_vae.Config(**video_values),
        audio_decoder=audio_vae.Config(**audio_values),
        diffusion=DiffusionConfig(
            ladder=variant["ladder"],
            video_shift=metadata["scheduler"]["shift"],
            audio_shift=metadata["audio_scheduler"]["shift"],
        ),
    )


def read_config(root: Path, io) -> Config:
    """Read all architecture sidecars before any numerical module construction."""  # noqa: E501
    metadata = {}
    for name, relative in (
        ("inference", "fastvideo_inference.json"),
        ("transformer", "transformer/config.json"),
        ("text_encoder", "text_encoder/config.json"),
        ("audio_vae", "audio_vae/config.json"),
        ("video_vae", "vae/config.json"),
        ("scheduler", "scheduler/scheduler_config.json"),
        ("audio_scheduler", "audio_scheduler/scheduler_config.json"),
    ):
        metadata[name] = json.loads(
            (root / relative).read_text(encoding="utf-8")
        )
    if metadata["audio_vae"].get("sampling_rate") != 32000:
        raise ValueError(
            "FastH3 audio output requires a 32000 Hz sampling clock"
        )
    return _normalize(metadata)


# Checkpoint headers needed to resolve architecture before module selection.
config_sources = ()
