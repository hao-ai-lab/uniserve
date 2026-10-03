"""Normalize MiniMax-H3 checkpoints into one typed configuration.

``read_config`` recognizes the checkpoint layout (``checkpoint.detect``),
reads the JSON sidecars of every component, validates the FastVideo
inference contract where one exists, and returns one immutable ``Config``
before any module is constructed. The network architecture is fixed:
``Config`` rejects any network or output field that differs from its
default. What a checkpoint may vary is each denoiser's grids and attention
(``DenoiserConfig``), the PDD output heads of a parallel-decoding
student, the VSA sparsity and the latent normalization statistics.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias

from uniserve.diffusion import BlockGrid, FixedGrid, RungGrid, UniformGrid
from uniserve.media import image
from uniserve.nn.functional import Rounding
from uniserve_models import qwen3_vl

from . import audio_vae, output, video_vae
from .checkpoint import (
    DENOISER_DIRECTORIES,
    DENOISER_TASKS,
    Kind,
    Layout,
    detect,
)
from .encoder import TextEncoderConfig
from .packing import CANVAS_MULTIPLE

# DMD rungs are unshifted noise levels on the 1000-step training clock.
TRAINING_CLOCK = 1000.0

# A diffusers root evaluates the released sigma grid of 50 points: 49 network
# evaluations and the clean endpoint.
UNIFORM_GRID_POINTS = 50

# The released (adapt_shape_v1) canvas rule: a 768-pixel short edge, an area
# cap of the 16:9 canvas, sides on the packing's 32-pixel grid, and aspect
# ratios from 1:4 to 4:1. Request planning and the denoisers' canvases both
# resolve through ``canvas``.
CANVAS_SHORT_EDGE = 768
CANVAS_MAX_PIXELS = 768 * 1344
MIN_ASPECT_RATIO, MAX_ASPECT_RATIO = 1 / 4, 4

# Width:height ratios a text- or reference-conditioned request may name.
NAMED_ASPECT_RATIOS = ((21, 9), (16, 9), (4, 3), (1, 1), (3, 4), (9, 16))


def canvas(aspect_width: float, aspect_height: float) -> image.Config:
    """Resolve a display aspect ratio into the released canvas.

    The short edge starts at 768, the area is capped at ``768 * 1344`` and
    both sides are then rounded to the nearest multiple of 32, so the final
    area may slightly exceed the cap. Only the ratio of the arguments
    matters: an aspect ratio or a keyframe's displayed dimensions.

    Raises:
        ValueError: The ratio lies outside 1:4 to 4:1.
    """
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError("an H3 aspect ratio must be positive")
    ratio = aspect_width / aspect_height
    if not MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO:
        raise ValueError(
            "H3 generates aspect ratios from 1:4 to 4:1, got "
            f"{aspect_width}:{aspect_height}"
        )
    if ratio >= 1:
        width, height = CANVAS_SHORT_EDGE * ratio, float(CANVAS_SHORT_EDGE)
    else:
        width, height = float(CANVAS_SHORT_EDGE), CANVAS_SHORT_EDGE / ratio
    area = width * height
    if area > CANVAS_MAX_PIXELS:
        # ``** 0.5`` rather than ``math.sqrt`` reproduces the reference's
        # arithmetic.
        scale = (CANVAS_MAX_PIXELS / area) ** 0.5
        width, height = width * scale, height * scale
    return image.Config(
        max(CANVAS_MULTIPLE, round(height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


# FastH3 DMD exports are trained on, and generate only, 12 buckets: the named
# aspect ratios at the 768p and 480p resolutions. The 768p buckets are the
# canvas rule's canvases; the 480p buckets, (height, width) below in the same
# ratio order, are the training table's own rather than a rescaled rule (21:9
# is 992x416, not 960x416). A deployment prepares a subset of them.
DMD_CANVASES = (
    *(canvas(*ratio) for ratio in NAMED_ASPECT_RATIOS),
    image.Config(416, 992),
    image.Config(480, 832),
    image.Config(480, 640),
    image.Config(480, 480),
    image.Config(640, 480),
    image.Config(832, 480),
)

# A FastVideo PDD student's packed sequence capacity in rows; the checkpoint
# owns this bound and serving options may only tighten it.
PDD_MAX_SEQUENCE_ROWS = 131_072


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_positive_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    """Define H3 widths, layers, attention, modulation and output heads.

    ``output_heads`` is 1 for an ordinary DiT and the fine-grid interval
    count for a parallel-decoding (PDD) student, whose output projections
    hold one head-major prediction per interval. ``rounding`` is the
    checkpoint's elementwise recipe for the block epilogues (modulation,
    gated residuals, Q/K rotation and SwiGLU gating): the released base and
    component checkpoints are defined by references that round after every
    eager BF16 operation; FastH3 exports keep UniServe's single-rounding
    kernels, whose served results their release fixed.
    """

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
    output_heads: int = 1
    rounding: Rounding = Rounding.STEPWISE

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
            "output_heads",
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
        if not isinstance(self.rounding, Rounding):
            raise ValueError("H3 transformer rounding must be a Rounding")
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


def _modality_grids(
    grid: Callable[..., FixedGrid], shifts: Mapping[str, float]
) -> Mapping[str, FixedGrid]:
    """Build one ``grid`` per modality, each at its scheduler shift."""
    return MappingProxyType(
        {name: grid(shift=shifts[name]) for name in ("video", "audio")}
    )


@dataclass(frozen=True, slots=True)
class DenseAttention:
    """Full bidirectional attention over the packed sequence."""


@dataclass(frozen=True, slots=True)
class SparseAttention:
    """Video sparse attention over ``tile``-row spatiotemporal tiles.

    Each video query tile keeps ``ceil((1 - sparsity) n)`` of the ``n`` key
    tiles of the generated video. Without ``reference_keep`` the generated
    video is the only region, packed in 64-row tiles (``packing.TilePacking``,
    the FastH3 DMD students). With it, every reference video is a region of
    its own, of which a query keeps the fraction ``reference_keep``
    (``packing.RegionPacking``, the ``p2_multi_region`` policy of
    reference-conditioned students).
    """

    tile: int
    sparsity: float
    reference_keep: float | None = None

    def __post_init__(self) -> None:
        if self.tile not in (64, 128):
            raise ValueError("H3 sparse attention tiles hold 64 or 128 rows")
        for name in ("sparsity",) + (
            () if self.reference_keep is None else ("reference_keep",)
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError(f"H3 sparse attention {name} must be finite")
        if not 0 <= self.sparsity < 1:
            raise ValueError("H3 VSA sparsity must lie in [0, 1)")
        if self.reference_keep is not None and not 0 < self.reference_keep <= 1:
            raise ValueError("H3 reference keep rate must lie in (0, 1]")
        if self.reference_keep is None and self.tile != 64:
            raise ValueError("single-region H3 sparse attention tiles 64 rows")


Attention: TypeAlias = DenseAttention | SparseAttention


@dataclass(frozen=True, slots=True)
class DenoiserConfig:
    """One DiT a deployment can place, with the requests it serves.

    Attributes:
        transformer: The network dimensions.
        grids: The checkpoint's video and audio grids, which differ only in
            their shifts; requests cannot change them.
        attention: Dense or sparse attention.
        tasks: The tasks this DiT serves, in canonical order.
        canvases: The only canvases the checkpoint generates, or None when it
            generates every canvas of the canvas rules.
        max_sequence_rows: The checkpoint's packed sequence capacity, or None
            when only memory and serving options bound it.
    """

    transformer: TransformerConfig
    grids: Mapping[str, FixedGrid]
    attention: Attention
    tasks: tuple[str, ...]
    canvases: tuple[image.Config, ...] | None
    max_sequence_rows: int | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.tasks, tuple)
            or not self.tasks
            or any(
                task not in ("t2va", "fl2va", "ref2va") for task in self.tasks
            )
            or len(set(self.tasks)) != len(self.tasks)
        ):
            raise ValueError("H3 denoiser tasks must name distinct H3 tasks")
        if self.canvases is not None and (
            not isinstance(self.canvases, tuple)
            or not self.canvases
            or any(
                not isinstance(value, image.Config) for value in self.canvases
            )
        ):
            raise ValueError("H3 denoiser canvases must be image.Config values")
        if self.max_sequence_rows is not None and (
            not _is_integer(self.max_sequence_rows)
            or self.max_sequence_rows < 1
        ):
            raise ValueError("H3 sequence capacity must be a positive integer")
        if (
            set(self.grids) != {"video", "audio"}
            or any(
                not isinstance(grid, FixedGrid) for grid in self.grids.values()
            )
            or replace(self.grids["audio"], shift=self.grids["video"].shift)
            != self.grids["video"]
        ):
            raise ValueError(
                "H3 video and audio grids must differ only in their shifts"
            )
        heads = self.transformer.output_heads
        grid = self.grids["video"]
        pdd = isinstance(grid, BlockGrid)
        if pdd != (heads > 1) or (pdd and heads != grid.intervals):
            raise ValueError(
                "H3 output heads must equal the PDD interval count, and only "
                "a PDD student has more than one"
            )


@dataclass(frozen=True, slots=True)
class Config:
    """Compose the fixed H3 networks and the checkpoint's denoisers.

    ``denoisers`` maps each denoising component the checkpoint holds
    (``denoiser`` for the ``transformer`` partition, ``reference_denoiser``
    for ``transformer_ref``) to its configuration; a deployment places one.
    """

    text_encoder: TextEncoderConfig
    denoisers: Mapping[str, DenoiserConfig]
    video_vae: video_vae.Config
    audio_vae: audio_vae.Config
    output: output.Config = output.Config()

    def __post_init__(self) -> None:
        if not isinstance(self.denoisers, Mapping) or not self.denoisers:
            raise ValueError("H3 checkpoints hold at least one denoiser")
        if set(self.denoisers) - set(DENOISER_DIRECTORIES):
            raise ValueError(
                "H3 denoisers must be named denoiser or reference_denoiser"
            )
        object.__setattr__(
            self, "denoisers", MappingProxyType(dict(self.denoisers))
        )
        for denoiser in self.denoisers.values():
            transformer = denoiser.transformer
            if self.text_encoder.hidden_size != transformer.text_dim:
                raise ValueError(
                    "H3 text features must match denoiser conditioning width"
                )
            if self.video_vae.latent_channels != transformer.video_channels:
                raise ValueError(
                    "H3 video latent channels must match the denoiser"
                )
            if self.audio_vae.latent_channels != transformer.audio_channels:
                raise ValueError(
                    "H3 audio latent channels must match the denoiser"
                )
        # Packing, attention, native reconstruction and checkpoint identity
        # implement this architecture; typed configs do not imply arbitrary
        # variants. The latent statistics come from each checkpoint and are
        # exempt, as are the PDD output heads and the checkpoint family's
        # rounding recipe.
        expected_networks: tuple[tuple[str, object, object], ...] = (
            ("text_encoder", self.text_encoder, TextEncoderConfig()),
            ("video_vae", self.video_vae, video_vae.Config()),
            ("audio_vae", self.audio_vae, audio_vae.Config()),
            ("output", self.output, output.Config()),
            *(
                (f"{name}.transformer", value.transformer, TransformerConfig())
                for name, value in self.denoisers.items()
            ),
        )
        for name, actual, expected in expected_networks:
            for field in fields(expected):  # type: ignore[arg-type]
                if field.name in {
                    "latents_mean",
                    "latents_std",
                    "output_heads",
                    "rounding",
                }:
                    continue
                value = getattr(actual, field.name)
                supported = getattr(expected, field.name)
                if value != supported:
                    raise ValueError(
                        f"H3 {name}.{field.name} must be {supported!r}, "
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


def _reject(field: str, expected: object, value: object) -> ValueError:
    return ValueError(
        f"unsupported MiniMax-H3 checkpoint: {field} must be {expected!r}, "
        f"got {value!r}"
    )


def _check_contract_fields(
    contract: Mapping[str, Any], required: Mapping[str, object]
) -> None:
    for name, expected in required.items():
        if contract.get(name) != expected:
            raise _reject(name, expected, contract.get(name))


def _contract_shifts(
    contract: Mapping[str, Any], shifts: Mapping[str, float]
) -> None:
    """Require a contract that restates a scheduler shift to agree with it."""
    for modality in ("video", "audio"):
        name = f"{modality}_scheduler_shift"
        if name in contract and contract[name] != shifts[modality]:
            raise ValueError(
                f"unsupported MiniMax-H3 checkpoint: {name}="
                f"{contract[name]!r} disagrees with the {modality} scheduler "
                f"shift {shifts[modality]!r}"
            )


def _sparsity(contract: Mapping[str, Any]) -> float:
    value = contract.get("vsa_sparsity")
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0 <= value < 1
    ):
        raise ValueError(
            "unsupported MiniMax-H3 checkpoint: vsa_sparsity must lie in "
            f"[0, 1), got {value!r}"
        )
    return float(value)


def _dmd_denoiser(
    contract: Mapping[str, Any],
    transformer: TransformerConfig,
    shifts: Mapping[str, float],
) -> DenoiserConfig:
    """Validate a FastH3 DMD contract and describe its denoiser.

    The export's trained DMD rungs are the denoising ladder: a uniform grid
    over the same number of points is not a substitute for them.

    Raises:
        ValueError: The contract is not a text-to-video-and-audio DMD export
            this implementation serves, naming the offending field.
    """
    _check_contract_fields(
        contract, {"task": "t2av", "attention_backend": "VIDEO_SPARSE_ATTN_H3"}
    )
    # The model has no unconditional branch and the sparse-attention kernel
    # tiles 64 rows.
    for name, expected in (("guidance_scale", 1.0), ("vsa_tile_size", 64)):
        value = contract.get(name)
        if not _is_positive_number(value) or value != expected:
            raise _reject(name, expected, value)

    rungs = contract.get("dmd_denoising_steps")
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
            "unsupported MiniMax-H3 checkpoint: dmd_denoising_steps must be "
            "strictly decreasing integers in (0, 1000], got "
            f"{rungs!r}"
        )
    # `num_inference_steps` counts sigma-grid points, including the clean
    # endpoint the solver reaches after the last rung.
    for name, expected in (
        ("transformer_forwards", len(rungs)),
        ("num_inference_steps", len(rungs) + 1),
    ):
        value = contract.get(name)
        if not _is_integer(value) or value != expected:
            raise ValueError(
                f"unsupported MiniMax-H3 checkpoint: {name} must be "
                f"{expected} for {len(rungs)} DMD rungs, got {value!r}"
            )
    sparsity = _sparsity(contract)
    _contract_shifts(contract, shifts)
    return DenoiserConfig(
        transformer=transformer,
        grids=_modality_grids(
            partial(RungGrid, tuple(rungs), clock=TRAINING_CLOCK), shifts
        ),
        attention=SparseAttention(tile=64, sparsity=sparsity),
        tasks=("t2va",),
        canvases=DMD_CANVASES,
        max_sequence_rows=None,
    )


def _pdd_denoiser(
    contract: Mapping[str, Any],
    transformer: TransformerConfig,
    shifts: Mapping[str, float],
    component: str,
) -> DenoiserConfig:
    """Validate a FastVideo PDD contract and describe its denoiser.

    Raises:
        ValueError: The contract is not a reference-conditioned PDD student
            this implementation serves, naming the offending field.
    """
    _check_contract_fields(
        contract,
        {
            "model_type": "ref2va",
            "attention_backend": "VIDEO_SPARSE_ATTN_H3",
            "conditioning": "fixed_ordered_references_target_only_flow",
            "vsa_ref_policy": "p2_multi_region",
        },
    )
    if component != "reference_denoiser":
        raise _reject(
            "transformer_component",
            "transformer_ref",
            contract.get("transformer_component"),
        )
    for name, expected in (("guidance_scale", 1.0), ("vsa_tile_size", 128)):
        value = contract.get(name)
        if not _is_positive_number(value) or value != expected:
            raise _reject(name, expected, value)

    intervals = contract.get("pdd_steps")
    if not _is_integer(intervals) or intervals < 2:
        raise ValueError(
            "unsupported MiniMax-H3 checkpoint: pdd_steps must be an integer "
            f">= 2, got {intervals!r}"
        )
    if transformer.output_heads != intervals:
        raise ValueError(
            "unsupported MiniMax-H3 checkpoint: the transformer's pdd_steps "
            f"{transformer.output_heads} disagrees with the contract's "
            f"{intervals}"
        )
    nodes = contract.get("pdd_step_indices")
    if not isinstance(nodes, list) or any(
        not _is_integer(node) for node in nodes
    ):
        raise ValueError(
            "unsupported MiniMax-H3 checkpoint: pdd_step_indices must list "
            f"integers, got {nodes!r}"
        )
    # FastVideo's `num_inference_steps` counts network evaluations, one per
    # block of the fine grid.
    for name in ("transformer_forwards", "num_inference_steps"):
        value = contract.get(name)
        if not _is_integer(value) or value != len(nodes) - 1:
            raise ValueError(
                f"unsupported MiniMax-H3 checkpoint: {name} must be "
                f"{len(nodes) - 1} for {len(nodes) - 1} PDD blocks, "
                f"got {value!r}"
            )
    max_t = contract.get("grid_max_t")
    if not _is_positive_number(max_t) or max_t > 1:
        raise _reject("grid_max_t", "a value in (0, 1]", max_t)
    keep = contract.get("vsa_ref_keep_rate")
    if not _is_positive_number(keep) or keep > 1:
        raise _reject("vsa_ref_keep_rate", "a value in (0, 1]", keep)
    sparsity = _sparsity(contract)
    _contract_shifts(contract, shifts)
    return DenoiserConfig(
        transformer=transformer,
        grids=_modality_grids(
            partial(BlockGrid, intervals, tuple(nodes), max_t=float(max_t)),
            shifts,
        ),
        attention=SparseAttention(
            tile=128, sparsity=sparsity, reference_keep=float(keep)
        ),
        tasks=DENOISER_TASKS[component],
        canvases=None,
        max_sequence_rows=PDD_MAX_SEQUENCE_ROWS,
    )


def _transformer(
    values: Mapping[str, Any], *, heads: int, rounding: Rounding
) -> TransformerConfig:
    missing = set(TRANSFORMER_FIELDS) - values.keys()
    if missing:
        raise ValueError(
            "MiniMax-H3 transformer is missing fields: "
            f"{', '.join(sorted(missing))}"
        )
    if values.get("patch_size") != [1, 2, 2]:
        raise ValueError("MiniMax-H3 transformer patch_size must be [1, 2, 2]")
    if values.get("final_norm_eps") != values.get("norm_eps"):
        raise ValueError(
            "MiniMax-H3 transformer final_norm_eps must equal norm_eps"
        )
    return TransformerConfig(
        output_heads=heads,
        rounding=rounding,
        **{
            target: values[source]
            for source, target in TRANSFORMER_FIELDS.items()
        },
    )


def _text_encoder(metadata: Mapping[str, Any]) -> TextEncoderConfig:
    text = metadata.get("text_config")
    if not isinstance(text, dict):
        raise ValueError("MiniMax-H3 text encoder requires text_config")
    for name, expected in (("hidden_act", "silu"), ("attention_bias", False)):
        if text.get(name) != expected:
            raise ValueError(f"unsupported MiniMax-H3 text encoder {name}")
    if metadata.get("tie_word_embeddings", False):
        raise ValueError(
            "MiniMax-H3 checkpoint requires an independent vocabulary head"
        )
    # The language layers rotate by interleaved M-RoPE; its sections are
    # the checkpoint's.
    rope = text.get("rope_scaling")
    if (
        not isinstance(rope, dict)
        or rope.get("rope_type") != "default"
        or rope.get("mrope_interleaved") is not True
        or not isinstance(rope.get("mrope_section"), list)
    ):
        raise ValueError("unsupported MiniMax-H3 text encoder rope_scaling")
    vision = metadata.get("vision_config")
    if not isinstance(vision, dict):
        raise ValueError("MiniMax-H3 text encoder requires vision_config")
    missing = set(TEXT_FIELDS) - text.keys()
    if missing:
        raise ValueError(
            "MiniMax-H3 text_encoder.text_config is missing fields: "
            f"{', '.join(sorted(missing))}"
        )
    missing = {"image_token_id", "video_token_id"} - metadata.keys()
    if missing:
        raise ValueError(
            "MiniMax-H3 text_encoder config is missing fields: "
            f"{', '.join(sorted(missing))}"
        )
    return TextEncoderConfig(
        **{target: text[source] for source, target in TEXT_FIELDS.items()},
        mrope_sections=tuple(rope["mrope_section"]),
        image_token_id=metadata["image_token_id"],
        video_token_id=metadata["video_token_id"],
        vision=qwen3_vl.read_vision_config(vision),
    )


def _video_vae(values: Mapping[str, Any]) -> video_vae.Config:
    result = {}
    for field in fields(video_vae.Config):
        if field.name not in values:
            raise ValueError(
                f"MiniMax-H3 video_vae is missing field {field.name}"
            )
        value = values[field.name]
        if isinstance(field.default, tuple):
            if not isinstance(value, (tuple, list)):
                raise ValueError(
                    f"MiniMax-H3 video_vae.{field.name} must be a sequence"
                )
            value = tuple(value)
        result[field.name] = value
    return video_vae.Config(**result)


def _audio_vae(values: Mapping[str, Any]) -> audio_vae.Config:
    result = {}
    for field in fields(audio_vae.Config):
        if field.name not in values:
            raise ValueError(
                f"MiniMax-H3 audio_vae is missing field {field.name}"
            )
        value = values[field.name]
        if isinstance(field.default, tuple) and not isinstance(
            value, (tuple, list)
        ):
            raise ValueError(
                f"MiniMax-H3 audio_vae.{field.name} must be a sequence"
            )
        if field.name == "resblock_dilation_sizes":
            if any(not isinstance(row, (tuple, list)) for row in value):
                raise ValueError(
                    "MiniMax-H3 audio_vae.resblock_dilation_sizes "
                    "must contain sequences"
                )
            value = tuple(tuple(row) for row in value)
        elif isinstance(field.default, tuple):
            value = tuple(value)
        result[field.name] = value
    return audio_vae.Config(**result)


def _shifts(metadata: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    shifts = {}
    for modality, name in (
        ("video", "scheduler"),
        ("audio", "audio_scheduler"),
    ):
        if "shift" not in metadata[name]:
            raise ValueError(f"MiniMax-H3 {name} is missing field shift")
        shifts[modality] = metadata[name]["shift"]
    return shifts


def normalize(
    layout: Layout, metadata: Mapping[str, Mapping[str, Any]]
) -> Config:
    """Normalize recognized checkpoint sidecars without allocating resources.

    ``metadata`` holds the parsed JSON of ``text_encoder``, ``video_vae``,
    ``audio_vae``, ``scheduler``, ``audio_scheduler`` and one entry per
    denoising component the layout holds (its ``config.json``).
    """
    for name in (
        "text_encoder",
        "video_vae",
        "audio_vae",
        "scheduler",
        "audio_scheduler",
        *layout.denoisers,
    ):
        if not isinstance(metadata.get(name), Mapping):
            raise ValueError(f"MiniMax-H3 requires {name} metadata")
    if metadata["audio_vae"].get("sampling_rate") != 32000:
        raise ValueError(
            "MiniMax-H3 audio output requires a 32000 Hz sampling clock"
        )
    shifts = _shifts(metadata)
    denoisers: dict[str, DenoiserConfig] = {}
    for component in layout.denoisers:
        values = metadata[component]
        heads = values.get("pdd_steps", 1)
        if not _is_integer(heads) or heads < 1:
            raise ValueError(
                "unsupported MiniMax-H3 checkpoint: transformer pdd_steps "
                f"must be a positive integer, got {heads!r}"
            )
        transformer = _transformer(
            values,
            heads=heads,
            rounding=Rounding.ONCE
            if layout.kind is Kind.FASTVIDEO_EXPORT
            else Rounding.STEPWISE,
        )
        if layout.kind is Kind.FASTVIDEO_EXPORT:
            assert layout.contract is not None
            denoisers[component] = _dmd_denoiser(
                layout.contract, transformer, shifts
            )
        elif layout.kind is Kind.COMPONENT_EXPORT:
            assert layout.contract is not None
            denoisers[component] = _pdd_denoiser(
                layout.contract, transformer, shifts, component
            )
        else:
            if heads != 1:
                raise ValueError(
                    "unsupported MiniMax-H3 checkpoint: a diffusers root DiT "
                    "has one output head"
                )
            denoisers[component] = DenoiserConfig(
                transformer=transformer,
                grids=_modality_grids(
                    partial(UniformGrid, UNIFORM_GRID_POINTS), shifts
                ),
                attention=DenseAttention(),
                tasks=DENOISER_TASKS[component],
                canvases=None,
                max_sequence_rows=None,
            )
    return Config(
        text_encoder=_text_encoder(metadata["text_encoder"]),
        denoisers=denoisers,
        video_vae=_video_vae(metadata["video_vae"]),
        audio_vae=_audio_vae(metadata["audio_vae"]),
    )


# Component sidecars, relative to the directory that holds the component. A
# component export holds its schedulers and draws the others from its base.
_SIDECARS = {
    "text_encoder": "text_encoder/config.json",
    "video_vae": "vae/config.json",
    "audio_vae": "audio_vae/config.json",
    "scheduler": "scheduler/scheduler_config.json",
    "audio_scheduler": "audio_scheduler/scheduler_config.json",
}
_EXPORT_SIDECARS = frozenset({"scheduler", "audio_scheduler"})


def read_config(root: Path, io, *, sources, base: Path | None = None) -> Config:
    """Read all architecture sidecars before any numerical module construction.

    ``root`` is the checkpoint directory, whose sidecars the loader has
    already fetched; ``io`` and ``sources`` (the resolved ``config_sources``,
    of which H3 declares none) are part of the package interface. ``base``
    is the directory of the base checkpoint a component export pins
    (``checkpoint.base_checkpoint``), which the loader resolved and verified;
    the export's text encoder and VAE sidecars are read from it, and its
    schedulers and DiT partition from ``root``. Every other layout has no
    base.

    An unreadable sidecar raises ``OSError``; invalid JSON, an unsupported
    checkpoint, or a base that does not match the layout raises
    ``ValueError``.
    """
    layout = detect(root)
    if (layout.kind is Kind.COMPONENT_EXPORT) != (base is not None):
        raise ValueError(
            "a MiniMax-H3 component export, and only one, reads its other "
            "components from its pinned base checkpoint"
        )
    metadata: dict[str, Any] = {}
    for name, relative in _SIDECARS.items():
        directory = root if base is None or name in _EXPORT_SIDECARS else base
        metadata[name] = json.loads(
            (directory / relative).read_text(encoding="utf-8")
        )
    for component, directory in layout.denoisers.items():
        metadata[component] = json.loads(
            (root / directory / "config.json").read_text(encoding="utf-8")
        )
    return normalize(layout, metadata)


# Checkpoint tensor headers needed to resolve architecture before module
# selection. H3's architecture comes entirely from JSON sidecars.
config_sources = ()
