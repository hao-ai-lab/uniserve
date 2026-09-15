"""Immutable BAGEL architecture and checkpoint metadata normalization."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from uniserve import loading
from uniserve_models import siglip

from . import vae
from .weights import checkpoint_sources


@dataclass(frozen=True)
class TransformerConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    head_dim: int
    qk_norm: bool
    max_position_embeddings: int

    def __post_init__(self):
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "vocab_size",
            "head_dim",
            "max_position_embeddings",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"BAGEL {name} must be a positive integer")

        if (
            self.num_attention_heads % self.num_key_value_heads
            or self.head_dim % 2
        ):
            raise ValueError(
                "BAGEL requires compatible GQA heads and even rotary dimensions"
            )
        if type(self.qk_norm) is not bool:
            raise ValueError("BAGEL QK normalization must be boolean")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (self.rms_norm_eps, self.rope_theta)
        ):
            raise ValueError(
                "BAGEL normalization epsilon and rotary theta must be "
                "finite and positive"
            )


@dataclass(frozen=True)
class Config:
    text: TransformerConfig
    vision: siglip.Config
    vae: vae.Config
    start_of_image_id: int
    end_of_image_id: int
    latent_patch_size: int
    max_latent_size: int
    timestep_shift: float
    connector_act: str

    def __post_init__(self):
        for value in (self.start_of_image_id, self.end_of_image_id):
            if type(value) is not int or not 0 <= value < self.text.vocab_size:
                raise ValueError(
                    "BAGEL image marker IDs must lie within its vocabulary"
                )
        if self.start_of_image_id == self.end_of_image_id:
            raise ValueError("BAGEL image markers must be distinct")

        if any(
            type(value) is not int or value < 1
            for value in (self.latent_patch_size, self.max_latent_size)
        ):
            raise ValueError(
                "BAGEL latent patches and learned grid size must be positive"
            )
        if not math.isfinite(self.timestep_shift) or self.timestep_shift <= 0:
            raise ValueError("BAGEL timestep shift must be finite and positive")


def read_config(root: Path, io: loading.Config) -> Config:
    """Read checkpoint metadata into typed configs before module construction.

    Tower configs live either inline in ``config.json`` or in per-tower files.
    The learned latent position grid is read from the checkpoint header, so its
    table must be a square grid whose width matches the text hidden size.
    """
    raw = json.loads((root / "config.json").read_text())

    towers = []
    for name in ("llm", "vit", "vae"):
        towers.append(
            raw[f"{name}_config"]
            if f"{name}_config" in raw
            else json.loads((root / f"{name}_config.json").read_text())
        )
    text, vision, latent = towers

    heads, hidden = text["num_attention_heads"], text["hidden_size"]
    if (
        type(heads) is not int
        or heads < 1
        or ("head_dim" not in text and hidden % heads)
    ):
        raise ValueError(
            "BAGEL checkpoint requires compatible text width and heads"
        )

    with checkpoint_sources[0].resolve(root, io=io).open(io=io) as reader:
        shape = reader.get("latent_pos_embed.pos_embed").shape
    side = math.isqrt(shape[0])
    if len(shape) != 2 or shape[1] != hidden or side * side != shape[0]:
        raise ValueError(
            "BAGEL latent position table must be a square grid at text width"
        )

    return Config(
        TransformerConfig(
            hidden,
            text["intermediate_size"],
            text["num_hidden_layers"],
            heads,
            text["num_key_value_heads"],
            text["vocab_size"],
            text.get("rms_norm_eps", 1e-6),
            text.get("rope_theta", 1_000_000.0),
            text.get("head_dim", hidden // heads),
            text.get("qk_norm", True),
            text.get("max_position_embeddings", 32768),
        ),
        siglip.Config(
            vision.get("patch_size", 14),
            vision.get("image_size", 980),
            vision.get("num_channels", 3),
            siglip.TransformerConfig(
                vision.get("hidden_size", 1152),
                vision.get("num_attention_heads", 16),
                vision.get("intermediate_size", 4304),
                vision.get("num_hidden_layers", 27) - 1,
                vision.get("layer_norm_eps", 1e-6),
            ),
        ),
        vae.Config(
            latent.get("resolution", 256),
            latent.get("in_channels", 3),
            latent.get("downsample", 8),
            latent.get("ch", 128),
            latent.get("out_ch", 3),
            tuple(latent.get("ch_mult", (1, 2, 4, 4))),
            latent.get("num_res_blocks", 2),
            latent.get("z_channels", 16),
            latent.get("scale_factor", 0.3611),
            latent.get("shift_factor", 0.1159),
        ),
        raw.get("start_of_image_id", 151652),
        raw.get("end_of_image_id", 151653),
        raw.get("latent_patch_size", 2),
        side,
        raw.get("timestep_shift", 1.0),
        raw.get("connector_act", "gelu_pytorch_tanh"),
    )


# Checkpoint headers needed to resolve architecture before module selection.
config_sources = checkpoint_sources[:1]
