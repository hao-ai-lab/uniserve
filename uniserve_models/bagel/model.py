"""BAGEL capability composition with one shared MoT backbone."""

from __future__ import annotations

from types import MappingProxyType

import torch
from torch import nn

from uniserve.model import (
    DEFAULT_COMPONENT,
    CausalLM,
    ComponentEntry,
    EntryPoint,
    ImageDecoder,
    PatchEncoder,
)
from uniserve.nn.linear import VocabParallelHead
from uniserve.nn.vae.patch import PatchAutoencoder

from . import vae
from .config import Config
from .denoiser import Denoiser
from .transformer import Transformer
from .vision import Encoder


class Model(nn.Module):
    """Compose text, denoising, vision, and latent codec over one backbone.

    The text LM and the image denoiser share the same MoT transformer; the
    vision encoder feeds SigLIP features into it, and the latent codec maps
    between pixels and the VAE latents the denoiser predicts. ``text.backbone``
    and ``denoiser.backbone`` are one module instance, as are
    ``latent_encoder`` and ``image_decoder.decoder``, so each shared part
    holds one set of weights.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        backbone = Transformer(config.text)
        self.text = CausalLM(
            backbone,
            VocabParallelHead(config.text.hidden_size, config.text.vocab_size),
        )
        self.denoiser = Denoiser(config, backbone)

        # BAGEL's ``Encoder`` applies its own connector and language-width
        # positions, so the wrapper's connector is the identity; downsample=1
        # keeps one feature per SigLIP patch.
        self.vision_encoder = PatchEncoder(
            Encoder(config),
            nn.Identity(),
            patch_size=config.vision.patch_size,
            downsample=1,
            output_size=config.text.hidden_size,
            output_dtype=torch.bfloat16,
        )

        autoencoder = vae.Model(config.vae)
        self.latent_encoder = PatchAutoencoder(
            autoencoder.encoder,
            autoencoder.decoder,
            autoencoder.posterior,
            patch_size=config.latent_patch_size,
            latent_channels=config.vae.latent_channels,
            latent_dtype=torch.bfloat16,
            downsample=config.vae.downsample * config.latent_patch_size,
            scale=config.vae.scale_factor,
            shift=config.vae.shift_factor,
        )
        self.image_decoder = ImageDecoder(self.latent_encoder)


def entry_points(config: Config):
    """Declare the one BAGEL component and the methods serving ranks call.

    Token embedding runs on the first pipeline stage, the only one pipeline
    binding keeps the embedding on, and logits on the last, the only one it
    keeps the vocabulary head on; both join only the tensor-parallel group.
    The backbone forwards join the tensor, sequence and pipeline groups, and
    the media methods join no communication group.
    """
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
                    EntryPoint("latent_encoder.encode"),
                    EntryPoint("image_decoder.decode"),
                ),
            ),
        }
    )
