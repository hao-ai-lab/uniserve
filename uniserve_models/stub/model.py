"""Deterministic text, vision and image module composition.

Every capability computes a closed-form result, so outputs are exactly
predictable. Text features are a lossless digit encoding of each token ID
and the head maps each ID to a fixed successor (see ``_Head``). The single
backbone layer passes features through and writes zero scalar K/V. Vision
features are per-patch pixel means, the latent encoder's latents are the
pixels themselves cast to BF16 and patchified, and the denoiser predicts zero.
"""

from __future__ import annotations

from types import MappingProxyType

import torch
from torch import nn

from uniserve.distributed import Communicator
from uniserve.model import (
    DEFAULT_COMPONENT,
    CausalLM,
    ComponentEntry,
    EntryPoint,
    ImageDecoder,
    PatchEncoder,
    TransformerDecoder,
    VocabShard,
)
from uniserve.nn.attention import Attention
from uniserve.nn.functional import patchify
from uniserve.nn.vae import (
    DiagonalGaussian,
    LatentDecoder,
    LatentEncoder,
    PatchAutoencoder,
    RGBDecoder,
    ScaleShift,
)

from .config import Config
from .denoiser import Denoiser

# Token IDs mirror the Qwen special-token vocabulary so the simulated model
# can be served with a real Qwen tokenizer.
STUB_EOS_TOKEN_ID = 151645
STUB_IMG_START_TOKEN_ID = 151670

_VOCAB_SIZE = STUB_IMG_START_TOKEN_ID + 1

# Three token-ID digits plus a constant digit; see ``_Embedding``.
_HIDDEN_SIZE = 4


class _Embedding(nn.Module):
    """Represent token IDs exactly using three base-128 BF16 digits."""

    # ``TransformerDecoder`` reads these as its hidden and vocabulary sizes.
    embedding_dim = _HIDDEN_SIZE
    num_embeddings = _VOCAB_SIZE

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones((), dtype=torch.bfloat16), requires_grad=False
        )

    def forward(self, tokens):
        # [tokens] -> [tokens, 4]. Three base-128 digits recover the ID
        # losslessly below 128**3, and every digit there is an integer below
        # 128, which BF16 represents exactly. A constant fourth digit fills
        # the hidden width. ``_Head`` inverts the digits back to the ID.
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
        # ``hidden`` is [rows, hidden], so logits are [rows, vocab].
        logits = hidden.new_full((*tokens.shape, _VOCAB_SIZE), -16.0)
        return logits.scatter_(1, targets.reshape(-1, 1), 16.0)


class _Layer(nn.Module):
    """Preserve features while publishing zero scalar K/V at supplied indices."""  # noqa: E501

    def __init__(self):
        super().__init__()

        # The layer's only parameter: ``TransformerDecoder.cache_config``
        # takes the cache dtype from a layer's first parameter, so this
        # scalar makes the K/V cache BF16.
        self.scale = nn.Parameter(
            torch.zeros((), dtype=torch.bfloat16), requires_grad=False
        )
        self.attention = Attention(
            1, 1, 1, cache_name="backbone.layers.0.attention"
        )

    def forward(self, hidden, residual, positions, attention):
        # The layer's table entry supplies the write addresses; an entry
        # without them publishes nothing.
        values = hidden.new_zeros((hidden.shape[0], 1, 1)) + self.scale
        self.attention.update_cache(values, values, attention)

        # A zero residual keeps the decoder's final ``norm(hidden +
        # residual)`` equal to the embedding digits ``_Head`` decodes.
        return hidden, torch.zeros_like(
            hidden
        ) if residual is None else residual


class _Vision(nn.Module):
    """Reduce each patch to its pixel mean, broadcast to the hidden width."""

    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, pixels, grids, grid_shapes):
        # Patch rows, or NCHW pixels patchified here, reduce to
        # [total_patches, _HIDDEN_SIZE].
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
    """Identity pixel scaling through a unit weight, for the latent decoder.

    ``PatchAutoencoder`` casts its inputs to the dtype of the encoder's or
    decoder's first parameter, so the unit weight makes the codec run in
    BF16.
    """

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
        identity = ScaleShift(scale=1.0, shift=0.0)
        self.latent_encoder = PatchAutoencoder(
            LatentEncoder(
                _Moments(),
                normalization=identity,
                posterior=DiagonalGaussian(sample=False),
            ),
            LatentDecoder(
                _Scale(),
                normalization=identity,
                latent_shape=(None, 3, None, None),
            ),
            patch_size=config.patch_size,
            latent_channels=3,
            latent_dtype=torch.bfloat16,
            downsample=config.patch_size,
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
