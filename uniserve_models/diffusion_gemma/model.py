"""DiffusionGemma capability composition over one shared text backbone."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import torch
from torch import nn

from uniserve.model import (
    DEFAULT_COMPONENT,
    CausalLM,
    ComponentEntry,
    EntryPoint,
    PatchEncoder,
    SelfConditioning,
    TokenDenoiser,
)
from uniserve.nn.linear import VocabParallelHead
from uniserve.nn.mlp import GatedMLP
from uniserve.nn.norm import RMSNorm

from . import vision
from .config import Config
from .transformer import Backbone


class Model(nn.Module):
    """Compose the causal prompt pass, the canvas denoiser and the vision tower.

    ``text`` is the causal language model that encodes prompts into the K/V
    cache; ``denoiser`` refines token canvases that read that cache. Both
    share one ``Backbone`` instance and one vocabulary head whose weight is
    the token embedding and whose logits are soft-capped. ``vision_encoder``
    turns preprocessed images into soft tokens at the text width, which
    callers substitute for the image placeholder embeddings of a prompt.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        text = config.text

        backbone = Backbone(text)
        head = VocabParallelHead(
            text.hidden_size,
            text.vocab_size,
            softcap=text.final_logit_softcapping,
        )
        # The checkpoint stores one token embedding; the head shares that
        # Parameter. Construction precedes pipeline binding, so both modules
        # are still present.
        embedding = backbone.embedding
        assert embedding is not None
        head.weight = embedding.weight
        self.text = CausalLM(backbone, head)

        size, eps = text.hidden_size, text.rms_norm_eps
        self.denoiser = TokenDenoiser(
            backbone,
            head,
            SelfConditioning(
                RMSNorm(size, eps),
                GatedMLP(
                    size,
                    text.intermediate_size,
                    activation="gelu_pytorch_tanh",
                ),
                RMSNorm(size, eps, elementwise_affine=False),
            ),
        )

        self.vision_encoder = PatchEncoder(
            vision.Encoder(config.vision),
            vision.Embedder(config.vision, text.hidden_size),
            patch_size=config.vision.patch_size,
            downsample=config.vision.pooling_kernel_size,
            output_size=text.hidden_size,
            output_dtype=torch.bfloat16,
        )


def entry_points(config: Config) -> Mapping[str, ComponentEntry]:
    """Declare the single DiffusionGemma component and its callable methods.

    Token embedding runs on the first pipeline stage and logits on the last;
    both join only the tensor-parallel group. The prompt and canvas passes
    join the tensor, sequence and pipeline groups; the vision encoder joins
    none.
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
                    EntryPoint("denoiser.compute_logits", "last", ("tp",)),
                    EntryPoint("vision_encoder.encode"),
                ),
            ),
        }
    )
