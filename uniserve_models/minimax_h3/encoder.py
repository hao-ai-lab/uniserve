"""H3's retained Qwen language layers for text conditioning."""

from __future__ import annotations

import math
from dataclasses import dataclass

from torch import nn

from uniserve.model import TextEncoder as BaseTextEncoder
from uniserve_models import qwen3


@dataclass(frozen=True, slots=True)
class TextEncoderConfig:
    """Define the H3 text encoder's vocabulary and tensor dimensions.

    ``num_checkpoint_layers`` is the checkpoint's decoder depth; the encoder
    runs only the first ``num_retained_layers`` of them. ``read_config``
    fills every other field from the checkpoint's ``text_config``, while
    ``num_retained_layers`` always keeps its default.
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


class TextEncoder(BaseTextEncoder):
    """Return the configured Qwen checkpoint hidden state without final norm."""

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
                max_position_embeddings=config.max_position_embeddings,
                attention_bias=False,
                tie_word_embeddings=False,
                num_experts=0,
                num_experts_per_tok=1,
                moe_intermediate_size=config.intermediate_size,
                norm_topk_prob=False,
            )
        )
        # H3 conditions on the retained layer's raw hidden state; the Qwen
        # final norm is not part of the checkpoint's conditioning path.
        network.norm = nn.Identity()
        super().__init__(network, tuple(range(config.num_retained_layers)))
        self.config = config
