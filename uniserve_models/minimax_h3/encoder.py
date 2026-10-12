"""H3's retained Qwen3-VL language layers and vision tower for conditioning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from uniserve_models import qwen3, qwen3_vl

# H3 conditions on the hidden state after the first 50 of the text encoder
# checkpoint's 64 decoder layers; the checkpoint's configuration does not
# state it.
RETAINED_LAYERS = 50


@dataclass(frozen=True, slots=True)
class TextEncoderConfig:
    """Define the H3 text encoder's vocabulary and tensor dimensions.

    ``num_checkpoint_layers`` is the checkpoint's decoder depth; the encoder
    runs only the first ``num_retained_layers`` of them. ``mrope_sections``
    are the interleaved M-RoPE widths of the temporal, height and width
    axes, ``vision`` is the vision tower and ``pixels`` the pixel
    normalization of the checkpoint's processor.
    """

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_checkpoint_layers: int
    num_retained_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    rms_norm_eps: float
    max_position_embeddings: int
    mrope_sections: tuple[int, ...]
    image_token_id: int
    video_token_id: int
    vision: qwen3_vl.VisionConfig
    pixels: qwen3_vl.PixelConfig

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
        # both towers compose (``qwen3_vl.Encoder``).
        if not isinstance(self.vision, qwen3_vl.VisionConfig):
            raise ValueError("H3 vision requires a Qwen3-VL vision config")


def text_encoder(config: TextEncoderConfig) -> qwen3_vl.Encoder:
    """Build H3's text encoder: the retained Qwen3-VL layers and vision tower.

    ``vision.encode`` encodes keyframes, reference images and reference
    video blocks; ``encode`` reads the prompt with those tokens spliced in
    and returns the retained layers' hidden state without the final norm.
    Vision tokens use BF16, the checkpoint's representation.
    """
    network = qwen3_vl.Transformer(
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
    return qwen3_vl.Encoder(
        network,
        tuple(range(config.num_retained_layers)),
        config.vision,
        pixels=config.pixels,
        image_token_id=config.image_token_id,
        video_token_id=config.video_token_id,
        dtype=torch.bfloat16,
    )
