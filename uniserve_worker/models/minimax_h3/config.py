"""Validated H3 component formats and fixed precision preset values."""

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


def h3_contract(manifest: Mapping[str, object]) -> dict[str, object]:
    """Validate the supported full VSA checkpoint before allocating components.

    The manifest's training indices do not select the inference schedule. The
    pinned FastVideo basic_fasth3 recipe uses five uniformly spaced grid points;
    its explicit DMD-index override is a different numerical protocol.
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
    for name, value in expected.items():
        if type(manifest.get(name)) is not type(value) or manifest.get(name) != value:
            raise ValueError(
                f"unsupported FastH3 checkpoint: {name} must be {value!r}, "
                f"got {manifest.get(name)!r}; use {FASTH3_MODEL_ID}@{FASTH3_REVISION}"
            )
    return {
        "family": "minimax-h3",
        "variant": "fasth3",
        "model_id": manifest["model_id"],
        "checkpoint_content_sha256": manifest["checkpoint_content_sha256"],
        "attention": "vsa",
        "sparsity": 0.9,
        "tasks": ["t2va"],
        "inference_grid": [1.0, 0.75, 0.5, 0.25, 0.0],
        "sigma_shifts": list(FASTH3_SHIFTS),
        "denoise_steps": len(FASTH3_LADDER),
        "width": 1344,
        "height": 768,
        "fps": 24,
        "audio_rate": 32000,
        "precision_presets": list(PRECISION_PRESETS),
    }


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

    h3_contract(metadata["inference"])
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

    for component, expected_shift in (("scheduler", 12.0), ("audio_scheduler", 3.0)):
        scheduler = metadata[component]
        if scheduler.get("shift") != expected_shift:
            raise ValueError(
                f"FastH3 {component} shift must be {expected_shift:g}, got {scheduler.get('shift')!r}"
            )
