"""H3's retained Qwen3-VL language layers and vision tower for conditioning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from uniserve_models import qwen3, qwen3_vl

# The Qwen3-VL-32B vision tower of the H3 text encoder checkpoint.
_VISION = qwen3_vl.VisionConfig(
    depth=27,
    hidden_size=1_152,
    intermediate_size=4_304,
    num_heads=16,
    hidden_act="gelu_pytorch_tanh",
    in_channels=3,
    patch_size=16,
    temporal_patch_size=2,
    spatial_merge_size=2,
    out_hidden_size=5_120,
    num_position_embeddings=2_304,
    deepstack_visual_indexes=(8, 16, 24),
)


@dataclass(frozen=True, slots=True)
class TextEncoderConfig:
    """Define the H3 text encoder's vocabulary and tensor dimensions.

    ``num_checkpoint_layers`` is the checkpoint's decoder depth; the encoder
    runs only the first ``num_retained_layers`` of them. ``read_config``
    fills the language fields from the checkpoint's ``text_config``, while
    ``num_retained_layers`` always keeps its default. ``mrope_sections``
    (the interleaved M-RoPE widths of the temporal, height and width axes),
    the image and video placeholder tokens, ``vision`` (the vision tower)
    and ``pixels`` (its processor's pixel normalization) default to the H3
    checkpoint's Qwen3-VL-32B values.
    """

    vocab_size: int = 151_936
    hidden_size: int = 5_120
    intermediate_size: int = 25_600
    num_checkpoint_layers: int = 64
    num_retained_layers: int = 50
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    rope_theta: float = 5_000_000.0
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 262_144
    mrope_sections: tuple[int, ...] = (24, 20, 20)
    image_token_id: int = 151_655
    video_token_id: int = 151_656
    vision: qwen3_vl.VisionConfig = _VISION
    pixels: qwen3_vl.PixelConfig = qwen3_vl.PixelConfig()

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_checkpoint_layers",
            "num_retained_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(f"H3 text {name} must be a positive integer")
        if self.num_retained_layers > self.num_checkpoint_layers:
            raise ValueError(
                "H3 retained layers must lie within the checkpoint"
            )
        if (
            self.num_attention_heads % self.num_key_value_heads
            or self.head_dim % 2
        ):
            raise ValueError(
                "H3 text requires divisible GQA and even rotary dimensions"
            )
        if any(
            not math.isfinite(value) or value <= 0
            for value in (self.rope_theta, self.rms_norm_eps)
        ):
            raise ValueError(
                "H3 text rotary base and normalization epsilon must be positive"
            )
        if (
            not isinstance(self.mrope_sections, tuple)
            or len(self.mrope_sections) != 3
            or sum(self.mrope_sections) != self.head_dim // 2
        ):
            raise ValueError(
                "H3 text M-RoPE sections must split half the head width "
                "over three axes"
            )
        if any(
            type(token) is not int or not 0 <= token < self.vocab_size
            for token in (self.image_token_id, self.video_token_id)
        ):
            raise ValueError("H3 vision placeholders must be vocabulary tokens")
        # The vision token width is checked against the language width where
        # both towers compose (``qwen3_vl.TextEncoder``).
        if not isinstance(self.vision, qwen3_vl.VisionConfig):
            raise ValueError("H3 vision requires a Qwen3-VL vision config")


class TextEncoder(qwen3_vl.TextEncoder):
    """Return the retained Qwen3-VL layer's hidden state without final norm.

    ``vision.encode`` encodes keyframes, reference images and reference
    video blocks; ``encode`` reads the prompt with those tokens spliced in
    (``qwen3_vl.TextEncoder``). Vision tokens use BF16, the checkpoint's
    representation.
    """

    def __init__(self, config: TextEncoderConfig):
        network = qwen3.Transformer(
            qwen3.Config(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_hidden_layers=config.num_checkpoint_layers,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                hidden_act="silu",
                rms_norm_eps=config.rms_norm_eps,
                rope_theta=config.rope_theta,
                rope_scaling=None,
                max_position_embeddings=config.max_position_embeddings,
                attention_bias=False,
                tie_word_embeddings=False,
                num_experts=0,
                num_experts_per_tok=1,
                moe_intermediate_size=config.intermediate_size,
                mrope_sections=config.mrope_sections,
                norm_topk_prob=False,
                decoder_sparse_step=1,
                mlp_only_layers=(),
            )
        )
        # H3 conditions on the retained layer's raw hidden state; the Qwen
        # final norm is not part of the checkpoint's conditioning path.
        network.norm = nn.Identity()
        super().__init__(
            network,
            tuple(range(config.num_retained_layers)),
            config.vision,
            pixels=config.pixels,
            image_token_id=config.image_token_id,
            video_token_id=config.video_token_id,
            dtype=torch.bfloat16,
        )
        self.config = config
