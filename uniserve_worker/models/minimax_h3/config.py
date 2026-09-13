"""Validated H3 component formats and fixed precision preset values."""

import math
import re
from dataclasses import dataclass
from typing import Mapping

from ...nn.quant.config import LinearPrecision

SUPPORTED_PRECISIONS: Mapping[str, tuple[LinearPrecision, ...]] = {
    "transformer.attention": ("bf16", "fp8", "nvfp4"),
    "transformer.mlp": ("bf16", "fp8", "mxfp8", "nvfp4"),
    "text_encoder": ("bf16", "fp8", "nvfp4"),
    "video_vae": ("fp16", "bf16", "nvfp4"),
}

PRECISION_PRESETS: Mapping[str, Mapping[str, LinearPrecision]] = {
    "quality": {
        "transformer.attention": "bf16",
        "transformer.mlp": "bf16",
        "text_encoder": "bf16",
        "video_vae": "fp16",
    },
    "balanced": {
        "transformer.attention": "bf16",
        "transformer.mlp": "bf16",
        "text_encoder": "bf16",
        "video_vae": "nvfp4",
    },
    "performance": {
        "transformer.attention": "bf16",
        "transformer.mlp": "fp8",
        "text_encoder": "bf16",
        "video_vae": "nvfp4",
    },
    "maximum": {
        "transformer.attention": "nvfp4",
        "transformer.mlp": "mxfp8",
        "text_encoder": "fp8",
        "video_vae": "nvfp4",
    },
}

PRECISION_SHORTHANDS: Mapping[str, Mapping[str, LinearPrecision]] = {
    "bf16": PRECISION_PRESETS["quality"],
    "fp8": {
        "transformer.attention": "fp8",
        "transformer.mlp": "fp8",
        "text_encoder": "bf16",
        "video_vae": "fp16",
    },
    "mxfp8": {
        "transformer.attention": "bf16",
        "transformer.mlp": "mxfp8",
        "text_encoder": "bf16",
        "video_vae": "fp16",
    },
    "nvfp4": {
        "transformer.attention": "nvfp4",
        "transformer.mlp": "nvfp4",
        "text_encoder": "nvfp4",
        "video_vae": "nvfp4",
    },
}


FASTH3_LADDER = (1000, 750, 500, 250)
FASTH3_SHIFTS = (12.0, 3.0)
FASTH3_TIME_SCALE = 1000.0

FASTH3_MODEL_ID = "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"
FASTH3_REVISION = "5ea076f35b84da4c3c82217112fa733d8eea2ae1"
FASTH3_8STEP_MODEL_ID = "FastVideo/FastVideo-FastH3-8-step-Preview-v1-VSA80-DataFree-Shift10"
FASTH3_8STEP_RELEASE = {
    "checkpoint_content_sha256": "516323fa396fa5dff4e82669d4e9a08a5791692a3d3b98ff6bc3de3fc6a33d11",
    "checkpoint_metadata_sha256": "ca9f2d609c05742ba465d24989981ec02cca26acb6ca2f163dc0f6dc8d11c27b",
    "fastvideo_commit": "24bbe7fddd05ca6f2c34b3dbed06ac1c75b72086",
    "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125],
    "video_scheduler_shift": 10.0,
    "audio_scheduler_shift": 3.0,
    "attention_backend": "VIDEO_SPARSE_ATTN_H3",
    "vsa_sparsity": 0.8,
    "sequence_parallel_size": 4,
}


def h3_contract(manifest: Mapping[str, object]) -> dict[str, object]:
    """Validate the supported full VSA checkpoint before allocating components.

    The pinned four-step recipe uses five uniformly spaced grid points rather
    than its training indices. Other exports select explicit DMD indices and
    scheduler shifts. Parsing files and establishing provenance belong to bootstrap.
    """

    expected = {
        "schema_version": "fasth3-inference-contract-v1",
        "model_id": FASTH3_MODEL_ID,
        "checkpoint_content_sha256": "b36987515e4c75fa4c7aaa632a7842c829ea141b235358a54d782b51230497b3",
        "checkpoint_metadata_sha256": "dcad0fbee2a7c7e75e53435f4fd98fccf3138844883874edf057962ab48fa428",
        "fastvideo_commit": "48a047c05ff4138f20cfa33351499c6ec5945f5d",
        "task": "t2av",
        "transformer_forwards": 4,
        "num_inference_steps": 5,
        "dmd_denoising_steps": [999, 749, 500, 250],
        "guidance_scale": 1.0,
        "attention_backend": "VIDEO_SPARSE_ATTN_H3",
        "vsa_tile_size": 64,
        "vsa_sparsity": 0.9,
    }
    pinned = manifest.get("model_id") == FASTH3_MODEL_ID
    if pinned:
        for name, value in expected.items():
            if type(manifest.get(name)) is not type(value) or manifest.get(name) != value:
                raise ValueError(
                    f"unsupported FastH3 checkpoint: {name} must be {value!r}, "
                    f"got {manifest.get(name)!r}; use {FASTH3_MODEL_ID}@{FASTH3_REVISION}"
                )
    else:
        _validate_manifest(manifest)
        if manifest["model_id"] == FASTH3_8STEP_MODEL_ID:
            for name, value in FASTH3_8STEP_RELEASE.items():
                if manifest.get(name) != value:
                    raise ValueError(f"published eight-step checkpoint {name} must be {value!r}")
    ladder = FASTH3_LADDER if pinned else tuple(manifest["dmd_denoising_steps"])
    shifts = (
        FASTH3_SHIFTS
        if pinned
        else (manifest["video_scheduler_shift"], manifest["audio_scheduler_shift"])
    )
    return {
        "family": "minimax-h3",
        "variant": "fasth3",
        "model_id": manifest["model_id"],
        "checkpoint_content_sha256": manifest["checkpoint_content_sha256"],
        "attention": "vsa",
        "sparsity": manifest["vsa_sparsity"],
        "attention_backend": manifest["attention_backend"],
        "tile_size": manifest["vsa_tile_size"],
        "guidance_scale": manifest["guidance_scale"],
        "sequence_parallel_size": manifest.get("sequence_parallel_size", 4),
        "checkpoint_metadata_sha256": manifest["checkpoint_metadata_sha256"],
        "fastvideo_commit": manifest["fastvideo_commit"],
        "ladder": list(ladder),
        "tasks": ["t2va"],
        "inference_grid": [*(value / FASTH3_TIME_SCALE for value in ladder), 0.0],
        "sigma_shifts": list(shifts),
        "denoise_steps": len(ladder),
        "width": 1344,
        "height": 768,
        "fps": 24,
        "audio_rate": 32000,
        "precision_presets": list(PRECISION_PRESETS),
    }


def _validate_manifest(manifest: Mapping[str, object]) -> None:
    """Reject incomplete explicit distilled recipes before numerical construction."""

    for key, value in {
        "schema_version": "fasth3-inference-contract-v1",
        "task": "t2av",
        "guidance_scale": 1.0,
        "vsa_tile_size": 64,
    }.items():
        if manifest.get(key) != value:
            raise ValueError(f"{key} must be {value!r}")
    if type(manifest.get("guidance_scale")) not in (float, int):
        raise ValueError("guidance_scale must be numeric")
    if type(manifest.get("vsa_tile_size")) is not int:
        raise ValueError("vsa_tile_size must be an integer")
    if not isinstance(manifest.get("model_id"), str) or not manifest["model_id"].strip():
        raise ValueError("model_id must be a nonempty identity")
    for key, length in (
        ("checkpoint_content_sha256", 64),
        ("checkpoint_metadata_sha256", 64),
        ("fastvideo_commit", 40),
    ):
        if not isinstance(manifest.get(key), str) or not re.fullmatch(
            rf"[0-9a-f]{{{length}}}", manifest[key]
        ):
            raise ValueError(f"{key} must be a lowercase hexadecimal digest")
    if manifest.get("attention_backend") not in {"VIDEO_SPARSE_ATTN", "VIDEO_SPARSE_ATTN_H3"}:
        raise ValueError("attention_backend must select a supported VSA backend")
    for key in ("video_scheduler_shift", "audio_scheduler_shift", "vsa_sparsity"):
        value = manifest.get(key)
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        if key == "vsa_sparsity":
            if not 0 <= value < 1:
                raise ValueError("vsa_sparsity must be in [0, 1)")
        elif value <= 0:
            raise ValueError(f"{key} must be positive")
    ladder = manifest.get("dmd_denoising_steps")
    if (
        not isinstance(ladder, list)
        or not ladder
        or any(type(value) is not int or not 0 < value <= 1000 for value in ladder)
        or any(left <= right for left, right in zip(ladder, ladder[1:]))
    ):
        raise ValueError(
            "dmd_denoising_steps must be strictly descending integer indices in (0, 1000]"
        )
    for key, expected_count in (
        ("transformer_forwards", len(ladder)),
        ("num_inference_steps", len(ladder) + 1),
    ):
        if type(manifest.get(key)) is not int or manifest[key] != expected_count:
            raise ValueError(f"{key} disagrees with dmd_denoising_steps")
    if (
        type(manifest.get("sequence_parallel_size")) is not int
        or manifest["sequence_parallel_size"] < 1
    ):
        raise ValueError("sequence_parallel_size must be a positive integer")


@dataclass(frozen=True, slots=True)
class H3TransformerConfig:
    """Defines H3 multimodal width, layer, attention, expert, modulation, and sparse-video geometry."""

    hidden_size: int = 5376
    heads: int = 56
    head_dim: int = 128
    layers: int = 50
    refiner_layers: int = 2
    ffn_dim: int = 14336
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


def validate_h3_config(metadata: Mapping[str, dict[str, object]]) -> None:
    """Validate parsed component dimensions and fixed diffusion mathematics."""

    from .encoder import H3TextEncoderConfig

    contract = h3_contract(metadata["inference"])
    transformer = metadata["transformer"]
    transformer_config = H3TransformerConfig()
    expected_transformer = {
        "num_attention_heads": transformer_config.heads,
        "attention_head_dim": transformer_config.head_dim,
        "hidden_size": transformer_config.hidden_size,
        "num_layers": transformer_config.layers,
        "num_refiner_layers": transformer_config.refiner_layers,
        "ffn_dim": transformer_config.ffn_dim,
        "in_channels": transformer_config.video_channels,
        "audio_in_channels": transformer_config.audio_channels,
        "patch_size": [1, 2, 2],
        "text_dim": transformer_config.text_dim,
        "freq_dim": transformer_config.frequency_dim,
        "time_embed_hidden_dim": transformer_config.time_hidden_dim,
        "time_embed_dim": transformer_config.time_dim,
        "rope_freq_dim": transformer_config.rope_frequency_dim,
        "rope_theta": transformer_config.rope_theta,
        "norm_eps": transformer_config.norm_eps,
        "qk_norm_eps": transformer_config.qk_norm_eps,
        "final_norm_eps": transformer_config.norm_eps,
    }
    for field, expected in expected_transformer.items():
        if transformer.get(field) != expected:
            raise ValueError(
                f"FastH3 transformer {field} must be {expected!r}, got {transformer.get(field)!r}"
            )

    encoder = metadata["text_encoder"].get("text_config")
    if not isinstance(encoder, dict):
        raise ValueError("FastH3 text encoder has no Qwen3-VL text configuration")
    encoder_config = H3TextEncoderConfig()
    expected_encoder = {
        "vocab_size": encoder_config.vocab_size,
        "hidden_size": encoder_config.hidden_size,
        "intermediate_size": encoder_config.intermediate_size,
        "num_hidden_layers": encoder_config.checkpoint_layers,
        "num_attention_heads": encoder_config.heads,
        "num_key_value_heads": encoder_config.kv_heads,
        "head_dim": encoder_config.head_dim,
        "rope_theta": encoder_config.rope_theta,
        "rms_norm_eps": encoder_config.norm_eps,
    }
    for field, expected in expected_encoder.items():
        if encoder.get(field) != expected:
            raise ValueError(
                f"FastH3 text encoder {field} must be {expected!r}, got {encoder.get(field)!r}"
            )

    for component, expected_shift in zip(
        ("scheduler", "audio_scheduler"), contract["sigma_shifts"], strict=True
    ):
        scheduler = metadata[component]
        if scheduler.get("shift") != expected_shift:
            raise ValueError(
                f"FastH3 {component} shift must be {expected_shift:g}, got {scheduler.get('shift')!r}"
            )
