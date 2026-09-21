"""SenseNova U1 capability composition with one shared MoT backbone."""

from __future__ import annotations

from types import MappingProxyType

import torch
from torch import nn

from uniserve.model import (
    CausalLM,
    ComponentEntry,
    DEFAULT_COMPONENT,
    EntryPoint,
    ImageDecoder,
    PatchEncoder,
)
from uniserve.nn.linear import VocabParallelHead
from uniserve.nn.vae.patch import RGBDecoder

from . import vision
from .config import Config
from .denoiser import Denoiser
from .transformer import Transformer


class Model(nn.Module):
    """Compose text, denoising, and vision over one shared MoT backbone.

    The text LM and the image denoiser share the same transformer; the vision
    encoder feeds NEO patch features into it, and the image decoder unfolds
    predicted patches back into RGB pixels.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        backbone = Transformer(config.text)
        self.text = CausalLM(
            backbone,
            VocabParallelHead(config.text.hidden_size, config.text.vocab_size),
        )
        if config.text.tie_word_embeddings:
            self.text.lm_head.weight = backbone.embedding.weight
        self.denoiser = Denoiser(config, backbone)

        self.vision_encoder = PatchEncoder(
            vision.Encoder(config.vision),
            nn.Identity(),
            patch_size=config.vision.patch_size,
            downsample=round(1 / config.vision.downsample_ratio),
            output_size=config.text.hidden_size,
            output_dtype=torch.bfloat16,
        )
        self.image_decoder = ImageDecoder(RGBDecoder(self.denoiser.patch_size))


def entry_points(config: Config):
    return MappingProxyType(
        {
            DEFAULT_COMPONENT: ComponentEntry(
                "",
                (
                    EntryPoint("text.forward", groups=("tp", "sp", "pp")),
                    EntryPoint("text.embed_input_ids", "first", ("tp",)),
                    EntryPoint("text.compute_logits", "last", ("tp",)),
                    EntryPoint("denoiser.forward", groups=("tp", "sp", "pp")),
                    EntryPoint("vision_encoder.encode"),
                    EntryPoint("image_decoder.decode"),
                ),
            ),
        }
    )
