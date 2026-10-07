"""Shared causal-language-model composition.

And explicit vocabulary projection.
"""

import torch
from torch import nn

from .inputs import TextInput
from .logits import Logits, project_logits
from .transformer import TransformerDecoder


class CausalLM(nn.Module):
    """Compose embedding replacement, a decoder, and a vocabulary head."""

    # Pipeline binding drops the head outside the last stage.
    lm_head: nn.Module | None

    def __init__(self, backbone: TransformerDecoder, lm_head: nn.Module):
        super().__init__()
        self.backbone, self.lm_head = backbone, lm_head

    @property
    def cache_config(self):
        return self.backbone.cache_config

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.embed_input_ids(input_ids)

    def forward(self, inputs: TextInput) -> torch.Tensor:
        return self.backbone(
            self._embeddings(inputs),
            inputs.positions,
            inputs.attention,
            routes=inputs.routes,
        )

    def fill_cache(self, inputs: TextInput) -> None:
        """Write the K/V cache of ``inputs`` without evaluating any output.

        The cache holds exactly what ``forward`` writes; the final layer
        stops at its cache write (``TransformerDecoder.fill_cache``).
        """
        self.backbone.fill_cache(
            self._embeddings(inputs),
            inputs.positions,
            inputs.attention,
            routes=inputs.routes,
        )

    def _embeddings(self, inputs: TextInput) -> torch.Tensor | None:
        """Token embeddings with any replacement, on the first stage."""
        if self.backbone._pipeline.rank != 0:
            return None
        hidden = self.embed_input_ids(inputs.input_ids.reshape(-1))
        if inputs.embeddings is not None:
            hidden = inputs.embeddings.apply(hidden)
        return hidden

    def compute_logits(
        self, hidden: torch.Tensor, *, token_indices: torch.Tensor
    ) -> Logits | None:
        """Project caller-selected token rows on the final pipeline stage.

        Empty selections remain empty. Vocabulary columns stay local until the
        caller requests Logits.gather; no sampling or request state is consumed.
        """
        pipeline = self.backbone._pipeline
        if pipeline.rank != pipeline.size - 1:
            return None

        # The last stage retains the head, which exposes its vocabulary shard.
        head = self.lm_head
        assert head is not None
        return project_logits(head, hidden, token_indices)
