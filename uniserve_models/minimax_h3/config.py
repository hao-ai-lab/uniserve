"""Validated H3 component formats and fixed precision preset values."""

import math
from dataclasses import dataclass, fields
from typing import Any, Mapping

from uniserve.nn.quant.config import LinearPrecision
from uniserve_models.minimax_h3.audio_vae import AudioDecoderConfig
from uniserve_models.minimax_h3.encoder import H3TextEncoderConfig
from uniserve_models.minimax_h3.video_vae import VideoDecoderConfig

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


def _validate_manifest(manifest: Mapping[str, object]) -> None:
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

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "heads",
            "head_dim",
            "layers",
            "refiner_layers",
            "ffn_dim",
            "video_channels",
            "audio_channels",
            "text_dim",
            "frequency_dim",
            "time_hidden_dim",
            "time_dim",
            "rope_frequency_dim",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"H3 transformer {name} must be a positive integer")
        for name in ("rope_theta", "norm_eps", "qk_norm_eps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"H3 transformer {name} must be finite and positive")
        if self.frequency_dim % 2 or self.rope_frequency_dim * 6 > self.head_dim:
            raise ValueError("H3 transformer rotary dimensions must fit its attention heads")


@dataclass(frozen=True, slots=True)
class H3DiffusionConfig:
    """The supported four-evaluation clean-sample Euler recipe."""

    ladder: tuple[int, ...] = FASTH3_LADDER
    video_shift: float = FASTH3_SHIFTS[0]
    audio_shift: float = FASTH3_SHIFTS[1]
    time_scale: float = FASTH3_TIME_SCALE

    def __post_init__(self) -> None:
        if (
            not isinstance(self.ladder, tuple)
            or any(not isinstance(value, int) or isinstance(value, bool) for value in self.ladder)
            or self.ladder != FASTH3_LADDER
        ):
            raise ValueError("FastH3 requires its fixed four-evaluation ladder")
        if (
            self.video_shift,
            self.audio_shift,
        ) != FASTH3_SHIFTS or self.time_scale != FASTH3_TIME_SCALE:
            raise ValueError("FastH3 requires its trained video/audio shifts and time scale")


@dataclass(frozen=True, slots=True)
class H3Config:
    """Compose the fixed FastH3 networks and their diffusion mathematics."""

    text_encoder: H3TextEncoderConfig = H3TextEncoderConfig()
    denoiser: H3TransformerConfig = H3TransformerConfig()
    video_decoder: VideoDecoderConfig = VideoDecoderConfig()
    audio_decoder: AudioDecoderConfig = AudioDecoderConfig()
    diffusion: H3DiffusionConfig = H3DiffusionConfig()

    def __post_init__(self) -> None:
        if self.text_encoder.hidden_size != self.denoiser.text_dim:
            raise ValueError("H3 text features must match denoiser conditioning width")
        if self.video_decoder.latent_channels != self.denoiser.video_channels:
            raise ValueError("H3 video latent channels must match the denoiser")
        if self.audio_decoder.latent_channels != self.denoiser.audio_channels:
            raise ValueError("H3 audio latent channels must match the denoiser")
        # Packing, sparse attention, native reconstruction and checkpoint identity
        # implement this architecture. Typed configs do not imply arbitrary variants.
        for name, expected in (
            ("text_encoder", H3TextEncoderConfig()),
            ("denoiser", H3TransformerConfig()),
            ("video_decoder", VideoDecoderConfig()),
            ("audio_decoder", AudioDecoderConfig()),
        ):
            actual = getattr(self, name)
            for field in fields(expected):
                if field.name in {"latents_mean", "latents_std"}:
                    continue
                value = getattr(actual, field.name)
                supported = getattr(expected, field.name)
                if value != supported:
                    raise ValueError(
                        f"FastH3 {name}.{field.name} must be {supported!r}, got {value!r}"
                    )


# Checkpoint field names differ from the mathematical modules' established names.
# The reader and native weight-name enumeration use this single correspondence.
TRANSFORMER_FIELDS = {
    "num_attention_heads": "heads",
    "attention_head_dim": "head_dim",
    "hidden_size": "hidden_size",
    "num_layers": "layers",
    "num_refiner_layers": "refiner_layers",
    "ffn_dim": "ffn_dim",
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
    "num_hidden_layers": "checkpoint_layers",
    "num_attention_heads": "heads",
    "num_key_value_heads": "kv_heads",
    "head_dim": "head_dim",
    "rope_theta": "rope_theta",
    "rms_norm_eps": "norm_eps",
    "max_position_embeddings": "max_position_embeddings",
}


def read_config(metadata: Mapping[str, Mapping[str, Any]]) -> H3Config:
    """Normalize the seven checkpoint sidecars without allocating model resources."""

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
    _validate_manifest(metadata["inference"])
    transformer = metadata["transformer"]
    missing = set(TRANSFORMER_FIELDS) - transformer.keys()
    if missing:
        raise ValueError(f"FastH3 transformer is missing fields: {', '.join(sorted(missing))}")
    if transformer.get("patch_size") != [1, 2, 2]:
        raise ValueError("FastH3 transformer patch_size must be [1, 2, 2]")
    if transformer.get("final_norm_eps") != transformer.get("norm_eps"):
        raise ValueError("FastH3 transformer final_norm_eps must equal norm_eps")
    denoiser = H3TransformerConfig(
        **{target: transformer[source] for source, target in TRANSFORMER_FIELDS.items()}
    )
    text = metadata["text_encoder"].get("text_config")
    if not isinstance(text, dict):
        raise ValueError("FastH3 text encoder requires text_config")
    for name, expected in (("hidden_act", "silu"), ("attention_bias", False)):
        if text.get(name) != expected:
            raise ValueError(f"unsupported FastH3 text encoder {name}")
    if metadata["text_encoder"].get("tie_word_embeddings", False):
        raise ValueError("FastH3 checkpoint requires an independent vocabulary head")
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
            f"FastH3 text_encoder.text_config is missing fields: {', '.join(sorted(missing))}"
        )
    encoder = H3TextEncoderConfig(
        **{target: text[source] for source, target in TEXT_FIELDS.items()}
    )
    video_values = {}
    for field in fields(VideoDecoderConfig):
        if field.name in {"width", "height", "fps"}:
            continue
        if field.name not in metadata["video_vae"]:
            raise ValueError(f"FastH3 video_vae is missing field {field.name}")
        value = metadata["video_vae"][field.name]
        if isinstance(field.default, tuple):
            if not isinstance(value, (tuple, list)):
                raise ValueError(f"FastH3 video_vae.{field.name} must be a sequence")
            value = tuple(value)
        video_values[field.name] = value
    audio_values = {}
    for field in fields(AudioDecoderConfig):
        if field.name not in metadata["audio_vae"]:
            raise ValueError(f"FastH3 audio_vae is missing field {field.name}")
        value = metadata["audio_vae"][field.name]
        if isinstance(field.default, tuple) and not isinstance(value, (tuple, list)):
            raise ValueError(f"FastH3 audio_vae.{field.name} must be a sequence")
        if field.name == "resblock_dilation_sizes":
            if any(not isinstance(row, (tuple, list)) for row in value):
                raise ValueError("FastH3 audio_vae.resblock_dilation_sizes must contain sequences")
            value = tuple(tuple(row) for row in value)
        elif isinstance(field.default, tuple):
            value = tuple(value)
        audio_values[field.name] = value
    for name in ("scheduler", "audio_scheduler"):
        if "shift" not in metadata[name]:
            raise ValueError(f"FastH3 {name} is missing field shift")
    return H3Config(
        text_encoder=encoder,
        denoiser=denoiser,
        video_decoder=VideoDecoderConfig(**video_values),
        audio_decoder=AudioDecoderConfig(**audio_values),
        diffusion=H3DiffusionConfig(
            video_shift=metadata["scheduler"]["shift"],
            audio_shift=metadata["audio_scheduler"]["shift"],
        ),
    )
