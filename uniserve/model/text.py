"""Shared causal-language-model composition.

And explicit vocabulary projection.
"""

import torch
from torch import nn

from .inputs import TextInput
from .logits import Logits
from .transformer import TransformerDecoder


class CausalLM(nn.Module):
    """Compose embedding replacement, a decoder, and a vocabulary head."""

    def __init__(self, backbone: TransformerDecoder, lm_head: nn.Module):
        super().__init__()
        self.backbone, self.lm_head = backbone, lm_head

    @property
    def cache_config(self):
        return self.backbone.cache_config

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.embed_input_ids(input_ids)

    def forward(self, inputs: TextInput) -> torch.Tensor:
        hidden = None
        if self.backbone._pipeline.rank == 0:
            hidden = self.embed_input_ids(inputs.input_ids.reshape(-1))
            if inputs.embeddings is not None:
                replacement = inputs.embeddings
                hidden = torch.where(
                    replacement.mask.reshape(-1, 1),
                    replacement.values.to(hidden.dtype),
                    hidden,
                )

        return self.backbone(
            hidden, inputs.positions, inputs.attention, routes=inputs.routes
        )

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
        if hidden.ndim != 2 or token_indices.ndim != 1:
            raise ValueError(
                "logits require packed hidden rows and one-dimensional token "
                "indices"
            )
        if token_indices.dtype not in {torch.int32, torch.int64}:
            raise ValueError("token indices must be integers")

        selected = hidden.index_select(0, token_indices)
        return Logits(self.lm_head(selected), self.lm_head.vocab)
