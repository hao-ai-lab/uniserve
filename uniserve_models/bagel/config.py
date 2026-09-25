"""Immutable BAGEL architecture and checkpoint metadata normalization."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uniserve import loading
from uniserve.loading import checkpoint
from uniserve_models import siglip

from . import vae
from .weights import checkpoint_sources


def _finite(value: object) -> bool:
    """Return whether ``value`` is a finite real number other than a bool."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


@dataclass(frozen=True)
class TransformerConfig:
    """Language backbone dimensions shared by the text and flow experts.

    ``read_config`` builds it from the ``llm`` tower config. Both experts of
    every ``TransformerLayer`` use these widths, and one attention module per
    layer covers the tokens of both.
    """

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
            not _finite(value) or value <= 0
            for value in (self.rms_norm_eps, self.rope_theta)
        ):
            raise ValueError(
                "BAGEL normalization epsilon and rotary theta must be "
                "finite and positive"
            )


@dataclass(frozen=True)
class Config:
    """Complete BAGEL architecture composed from its tower configs.

    Attributes:
        text: Shared MoT backbone dimensions.
        vision: SigLIP tower feeding understanding inputs.
        vae: FLUX autoencoder mapping pixels to latents.
        start_of_image_id: Token that opens each framed image sequence.
        end_of_image_id: Token that closes each framed image sequence.
        latent_patch_size: Autoencoder latent positions per flow token along
            each spatial axis.
        max_latent_size: Side of the learned square latent position grid, in
            flow tokens; ``read_config`` takes it from the checkpoint table.
        timestep_shift: Schedule shift ``Denoiser.make_schedules`` applies
            when the caller passes none.
        connector_act: Activation name of the vision connector MLP.
    """

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
        if not _finite(self.timestep_shift) or self.timestep_shift <= 0:
            raise ValueError("BAGEL timestep shift must be finite and positive")
        if not isinstance(self.connector_act, str):
            raise ValueError("BAGEL connector activation must be a name")


def _required(text: Mapping[str, Any], name: str) -> Any:
    """Return a required ``llm_config`` field for its config to validate.

    The value is raw checkpoint JSON; the config dataclass that receives it
    checks its type.
    """
    if name not in text:
        raise ValueError(f"BAGEL llm_config requires {name!r}")
    return text[name]


def read_config(
    root: Path,
    io: loading.Config,
    *,
    sources: Mapping[str, checkpoint.Source],
) -> Config:
    """Read checkpoint metadata into typed configs before module construction.

    Tower configs live either inline in ``config.json`` or in per-tower files;
    absent optional fields take fixed defaults. The learned latent position
    grid is read from a tensor header of the primary source rather than from
    ``config.json``, so its table must be a square grid whose width matches
    the text hidden size.

    Args:
        root: Local checkpoint directory containing ``config.json``.
        io: Checkpoint IO policy for opening the primary source.
        sources: The resolved ``config_sources`` by name; only the primary
            source's tensor metadata is read, so a header-only source of a
            dummy load suffices.

    Raises:
        FileNotFoundError: If ``config.json`` or a tower config that is not
            inline is missing.
        ValueError: If a tower config is not an object; if a required text
            dimension or the ``latent_pos_embed.pos_embed`` tensor is
            absent; if the text width or head count is not a positive
            integer or, without an explicit ``head_dim``, the head count does
            not divide the width; if the latent position table is not a
            square grid at text width; or if a config's ``__post_init__``
            rejects a value. Malformed JSON and checkpoint metadata also
            raise ``ValueError``.
    """
    raw = json.loads((root / "config.json").read_text())

    towers = []
    for name in ("llm", "vit", "vae"):
        tower = (
            raw[f"{name}_config"]
            if f"{name}_config" in raw
            else json.loads((root / f"{name}_config.json").read_text())
        )
        if not isinstance(tower, Mapping):
            raise ValueError(f"BAGEL {name}_config must be an object")
        towers.append(tower)
    text, vision, latent = towers

    heads = _required(text, "num_attention_heads")
    hidden = _required(text, "hidden_size")
    # Validate both before the default head_dim divides one by the other.
    if (
        type(heads) is not int
        or heads < 1
        or type(hidden) is not int
        or hidden < 1
        or ("head_dim" not in text and hidden % heads)
    ):
        raise ValueError(
            "BAGEL checkpoint requires compatible text width and heads"
        )

    with sources[checkpoint_sources[0].name].open(io=io) as reader:
        if "latent_pos_embed.pos_embed" not in reader.names():
            raise ValueError(
                "BAGEL primary checkpoint lacks latent_pos_embed.pos_embed"
            )
        shape = reader.get("latent_pos_embed.pos_embed").shape
    if (
        len(shape) != 2
        or shape[1] != hidden
        or math.isqrt(shape[0]) ** 2 != shape[0]
    ):
        raise ValueError(
            "BAGEL latent position table must be a square grid at text width"
        )
    side = math.isqrt(shape[0])

    # BAGEL runs one fewer SigLIP layer than its tower config declares, so
    # the count must be an integer before the subtraction.
    vision_layers = vision.get("num_hidden_layers", 27)
    if type(vision_layers) is not int:
        raise ValueError("BAGEL vit_config num_hidden_layers must be integer")
    channel_multipliers = latent.get("ch_mult", (1, 2, 4, 4))
    if not isinstance(channel_multipliers, (list, tuple)):
        raise ValueError("BAGEL vae_config ch_mult must be a list")

    return Config(
        TransformerConfig(
            hidden,
            _required(text, "intermediate_size"),
            _required(text, "num_hidden_layers"),
            heads,
            _required(text, "num_key_value_heads"),
            _required(text, "vocab_size"),
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
                vision_layers - 1,
                vision.get("layer_norm_eps", 1e-6),
            ),
        ),
        vae.Config(
            latent.get("resolution", 256),
            latent.get("in_channels", 3),
            latent.get("downsample", 8),
            latent.get("ch", 128),
            latent.get("out_ch", 3),
            tuple(channel_multipliers),
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


# ``read_config`` reads the latent position table from the primary source's
# tensor header, so the loader resolves that source before module selection:
# downloaded for a Hub checkpoint, or header-only for a dummy Hub load.
config_sources = checkpoint_sources[:1]
