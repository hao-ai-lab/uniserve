"""Normalize the fixed FastH3 checkpoint architecture and diffusion recipe.

``read_config`` reads the export's seven JSON sidecars and ``_normalize``
turns them into one typed ``Config`` before any module is constructed. The
architecture is fixed: ``Config`` rejects any network or output field that
differs from its default. Only the DMD ladder, the scheduler shifts, the VSA
sparsity and the latent normalization statistics may differ between
supported exports.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from . import audio_vae, output, video_vae
from .encoder import TextEncoderConfig

# Inference contract schema a FastH3 export publishes in its manifest.
INFERENCE_SCHEMA = "fasth3-inference-contract-v1"

# DMD rungs are unshifted noise levels on the 1000-step training clock.
TRAINING_CLOCK = 1000.0


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_positive_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _inference_contract(
    manifest: Mapping[str, object],
    shifts: Mapping[str, float],
) -> tuple[tuple[int, ...], float]:
    """Validate a FastH3 inference contract and return its rungs and sparsity.

    The export's trained DMD rungs are the denoising ladder: each rung is an
    unshifted noise level on the training clock, and a uniform grid over the
    same number of points is not a substitute for them. `shifts` holds the
    video and audio scheduler shifts, which a contract may restate and must
    then agree with.

    Raises:
        ValueError: The manifest is not a text-to-video-and-audio contract
            this implementation serves, naming the offending field.
    """
    for name, required in (
        ("schema_version", INFERENCE_SCHEMA),
        ("task", "t2av"),
        ("attention_backend", "VIDEO_SPARSE_ATTN_H3"),
    ):
        if manifest.get(name) != required:
            raise ValueError(
                f"unsupported FastH3 checkpoint: {name} must be "
                f"{required!r}, got {manifest.get(name)!r}"
            )
    # The model has no unconditional branch and the sparse-attention kernel
    # tiles 64 rows.
    for name, expected in (("guidance_scale", 1.0), ("vsa_tile_size", 64)):
        value = manifest.get(name)
        if not _is_positive_number(value) or value != expected:
            raise ValueError(
                f"unsupported FastH3 checkpoint: {name} must be "
                f"{expected!r}, got {value!r}"
            )

    rungs = manifest.get("dmd_denoising_steps")
    if (
        not isinstance(rungs, list)
        or not rungs
        or any(
            not _is_integer(rung) or not 0 < rung <= TRAINING_CLOCK
            for rung in rungs
        )
        or any(left <= right for left, right in zip(rungs, rungs[1:]))
    ):
        raise ValueError(
            "unsupported FastH3 checkpoint: dmd_denoising_steps must be "
            "strictly decreasing integers in (0, 1000], got "
            f"{rungs!r}"
        )
    # `num_inference_steps` counts sigma-grid points, including the clean
    # endpoint the solver reaches after the last rung.
    for name, expected in (
        ("transformer_forwards", len(rungs)),
        ("num_inference_steps", len(rungs) + 1),
    ):
        value = manifest.get(name)
        if not _is_integer(value) or value != expected:
            raise ValueError(
                f"unsupported FastH3 checkpoint: {name} must be {expected} "
                f"for {len(rungs)} DMD rungs, got {value!r}"
            )

    sparsity = manifest.get("vsa_sparsity")
    if (
        not isinstance(sparsity, int | float)
        or isinstance(sparsity, bool)
        or not math.isfinite(sparsity)
        or not 0 <= sparsity < 1
    ):
        raise ValueError(
            "unsupported FastH3 checkpoint: vsa_sparsity must lie in [0, 1), "
            f"got {sparsity!r}"
        )

    for modality in ("video", "audio"):
        name = f"{modality}_scheduler_shift"
        if name in manifest and manifest[name] != shifts[modality]:
            raise ValueError(
                f"unsupported FastH3 checkpoint: {name}={manifest[name]!r} "
                f"disagrees with the {modality} scheduler shift "
                f"{shifts[modality]!r}"
            )
    return tuple(rungs), float(sparsity)


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
        # Each of the three rotary axes (time, height, width) rotates
        # 2 * rope_frequency_dim channels of a head; the rest stay unrotated.
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
    """A checkpoint-owned FastH3 clean-sample Euler recipe.

    `ladder` holds the trained DMD rungs on the `time_scale` clock; each
    modality shifts them once with its own scheduler shift. The defaults are
    the four-rung Preview export's recipe.
    """

    ladder: tuple[int, ...] = (999, 749, 500, 250)
    video_shift: float = 12.0
    audio_shift: float = 3.0
    time_scale: float = TRAINING_CLOCK

    def __post_init__(self) -> None:
        if (
            not isinstance(self.ladder, tuple)
            or not self.ladder
            or any(
                not _is_integer(rung) or not 0 < rung <= self.time_scale
                for rung in self.ladder
            )
            or any(
                left <= right
                for left, right in zip(self.ladder, self.ladder[1:])
            )
        ):
            raise ValueError(
                "FastH3 ladder must be strictly decreasing integer rungs "
                "within its time scale"
            )
        for name in ("video_shift", "audio_shift", "time_scale"):
            if not _is_positive_number(getattr(self, name)):
                raise ValueError(f"FastH3 {name} must be finite and positive")


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
        # arbitrary variants. The latent statistics and VSA sparsity come
        # from each checkpoint and are exempt, as is ``diffusion``.
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


# Checkpoint config field names mapped to the typed config fields they fill.
# ``_normalize`` reads the checkpoint through them, and ``weights`` maps back
# through them to construct the native modules whose parameter names it
# enumerates.
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
    for name in ("scheduler", "audio_scheduler"):
        if "shift" not in metadata[name]:
            raise ValueError(f"FastH3 {name} is missing field shift")
    shifts = {
        "video": metadata["scheduler"]["shift"],
        "audio": metadata["audio_scheduler"]["shift"],
    }
    ladder, sparsity = _inference_contract(metadata["inference"], shifts)
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
        vsa_sparsity=sparsity,
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
    # The text encoder checkpoint also holds a vision tower that H3 never
    # runs. ``weights`` enumerates checkpoint tensor names from this fixed
    # vision layout, so the checkpoint must match it.
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

    return Config(
        text_encoder=encoder,
        denoiser=denoiser,
        video_decoder=video_vae.Config(**video_values),
        audio_decoder=audio_vae.Config(**audio_values),
        diffusion=DiffusionConfig(
            ladder=ladder,
            video_shift=shifts["video"],
            audio_shift=shifts["audio"],
        ),
    )


def read_config(root: Path, io) -> Config:
    """Read all architecture sidecars before any numerical module construction.

    ``root`` is the checkpoint directory, whose sidecars the loader has
    already fetched; ``io`` is part of the package interface and unused here.
    An unreadable sidecar raises ``OSError``; invalid JSON or an unsupported
    export raises ``ValueError``.
    """  # noqa: E501
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


# Checkpoint tensor headers needed to resolve architecture before module
# selection. H3's architecture comes entirely from JSON sidecars.
config_sources = ()
