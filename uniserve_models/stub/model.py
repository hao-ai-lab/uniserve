"""Deterministic text, vision and image module composition."""

from __future__ import annotations

from types import MappingProxyType

import torch
from torch import nn

from uniserve.distributed import Communicator
from uniserve.model import (
    CausalLM,
    ComponentEntry,
    DEFAULT_COMPONENT,
    EntryPoint,
    ImageDecoder,
    PatchEncoder,
    TransformerDecoder,
    VocabShard,
)
from uniserve.nn.attention import Attention, PagedInput, SegmentedInput
from uniserve.nn.functional import patchify
from uniserve.nn.vae.layers import DiagonalGaussian
from uniserve.nn.vae.patch import PatchAutoencoder, RGBDecoder

from .config import Config
from .denoiser import Denoiser

# Token IDs mirror the Qwen special-token vocabulary so the simulated model
# can be served with a real Qwen tokenizer.
STUB_EOS_TOKEN_ID = 151645
STUB_IMG_START_TOKEN_ID = 151670

_VOCAB_SIZE = STUB_IMG_START_TOKEN_ID + 1

_HIDDEN_SIZE = 4


class _Embedding(nn.Module):
    """Represent token IDs exactly using three base-128 BF16 digits."""

    embedding_dim = _HIDDEN_SIZE
    num_embeddings = _VOCAB_SIZE

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones((), dtype=torch.bfloat16), requires_grad=False
        )

    def forward(self, tokens):
        # Three base-128 digits recover the ID losslessly below 128**3, plus a
        # constant bias digit, so the head can invert features back to tokens.
        return (
            torch.stack(
                (
                    tokens % 128,
                    tokens // 128 % 128,
                    tokens // 16384,
                    torch.ones_like(tokens),
                ),
                dim=-1,
            ).to(self.weight.dtype)
            * self.weight
        )


class _Head(nn.Module):
    """Project the fixed multimodal successor cycle from numerical features."""

    def __init__(self):
        super().__init__()
        self.vocab = VocabShard(
            _VOCAB_SIZE, slice(0, _VOCAB_SIZE), _VOCAB_SIZE, Communicator()
        )

    def forward(self, hidden):
        # Invert the embedding's base-128 digits back into the input token ID.
        digits = hidden[..., :3].long()
        tokens = digits[..., 0] + 128 * digits[..., 1] + 16384 * digits[..., 2]

        # The simulated vocabulary cycles deterministically: 1000 -> 1001 ->
        # image start -> 1002 .. 1007 -> EOS. Any other input maps to 1000.
        targets = torch.full_like(tokens, 1000)
        targets = torch.where(tokens == 1000, 1001, targets)
        targets = torch.where(tokens == 1001, STUB_IMG_START_TOKEN_ID, targets)
        targets = torch.where(tokens == STUB_IMG_START_TOKEN_ID, 1002, targets)
        targets = torch.where(
            (tokens >= 1002) & (tokens < 1007), tokens + 1, targets
        )
        targets = torch.where(tokens == 1007, STUB_EOS_TOKEN_ID, targets)

        # ±16 logits make argmax sampling pick the cycle's successor token.
        logits = hidden.new_full((*tokens.shape, _VOCAB_SIZE), -16.0)
        return logits.scatter_(1, targets.reshape(-1, 1), 16.0)


class _Layer(nn.Module):
    """Preserve features while publishing zero scalar K/V at supplied indices."""  # noqa: E501

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(
            torch.zeros((), dtype=torch.bfloat16), requires_grad=False
        )
        self.attention = Attention(
            1, 1, 1, cache_name="backbone.layers.0.attention"
        )

    def forward(self, hidden, residual, positions, attention):
        values = hidden.new_zeros((hidden.shape[0], 1, 1)) + self.scale
        if (
            isinstance(attention, (PagedInput, SegmentedInput))
            and attention.write_indices is not None
        ):
            self.attention.update_cache(
                values, values, indices=attention.write_indices
            )
        return hidden, torch.zeros_like(
            hidden
        ) if residual is None else residual


class _Vision(nn.Module):
    """Reduce each patch to its pixel mean, broadcast to the hidden width."""

    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, pixels, grids, grid_shapes):
        patches = (
            patchify(pixels, patch_size=self.patch_size)
            if pixels.ndim == 4
            else pixels
        )
        return (
            patches.reshape(-1, patches.shape[-1])
            .mean(-1, keepdim=True)
            .expand(-1, _HIDDEN_SIZE)
        )


class _Scale(nn.Module):
    """Identity pixel scaling through a unit weight, for the latent decoder."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones((), dtype=torch.bfloat16), requires_grad=False
        )

    def forward(self, pixels):
        return pixels * self.weight


class _Moments(_Scale):
    """Emit (mean, zero log-variance) pairs for a deterministic latent."""

    def forward(self, pixels):
        means = super().forward(pixels)
        return torch.cat((means, torch.zeros_like(means)), dim=1)


class Model(CausalLM):
    """Compose deterministic text, vision, latent and image capabilities."""

    def __init__(self, config: Config = Config()):
        backbone = TransformerDecoder(
            _Embedding(), nn.ModuleDict({"0": _Layer()}), nn.Identity()
        )
        super().__init__(backbone, _Head())
        self.config = config

        # The denoiser shares the language backbone's single cache layer so
        # prefill and diffusion publish to the same scalar attention cache.
        self.denoiser = Denoiser(config, backbone)
        self.vision_encoder = PatchEncoder(
            _Vision(config.patch_size),
            nn.Identity(),
            patch_size=config.patch_size,
            downsample=1,
            output_size=_HIDDEN_SIZE,
            output_dtype=torch.bfloat16,
        )
        self.latent_encoder = PatchAutoencoder(
            _Moments(),
            _Scale(),
            DiagonalGaussian(sample=False),
            patch_size=config.patch_size,
            latent_channels=3,
            latent_dtype=torch.bfloat16,
            downsample=config.patch_size,
            scale=1.0,
            shift=0.0,
        )
        self.image_decoder = ImageDecoder(RGBDecoder(config.patch_size))


def entry_points(config: Config):
    """Declare the numerical methods serving ranks may invoke on this model."""
    return MappingProxyType(
        {
            DEFAULT_COMPONENT: ComponentEntry(
                "",
                (
                    EntryPoint("forward"),
                    EntryPoint("embed_input_ids"),
                    EntryPoint("compute_logits"),
                    EntryPoint("denoiser.forward"),
                    EntryPoint("latent_encoder.encode"),
                    EntryPoint("image_decoder.decode"),
                ),
            ),
            # The patch encoder is its own component, as it is in a model that
            # encodes images separately from its language backbone, so a
            # placement can hold the two on different ranks.
            "vision_encoder": ComponentEntry(
                "vision_encoder",
                (EntryPoint("encode"),),
            ),
        }
    )
