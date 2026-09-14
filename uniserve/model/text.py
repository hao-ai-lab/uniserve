"""Shared token embedding, decoder invocation, and vocabulary projection."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol, cast

import torch

from uniserve.model.batch import TextBatch, TextOutput
from uniserve.model.tensors import TensorViews, TokenSelection, VocabularyPartition
from uniserve.nn.decoder.base import Decoder
from uniserve.nn.logits import project_outputs
from uniserve.nn.parallel_pipeline import LayerPipeline
from uniserve.nn.parallel_sequence import SequencePartition
from uniserve.nn.vocab_parallel_embedding import ParallelLMHead


@dataclass(frozen=True, slots=True)
class TextSize:
    """Token extent, logical rows, and optional text-result selection.

    No selection describes backbone activations. A selection describes hidden
    or vocabulary rows after projection; inactive selected rows may have zero
    query tokens. Encoder calls require a positive unselected token extent.
    """

    tokens: int
    rows: int = 1
    selection: TokenSelection | None = None

    def __post_init__(self) -> None:
        if self.tokens < 0 or self.rows < 1 or (self.tokens == 0 and self.selection is None):
            raise ValueError("text shape requires nonnegative tokens and positive sequence extents")


class _TextModules(Protocol):
    """Read-only access to the registered modules used by shared text computation."""

    @property
    def text_backbone(self) -> Decoder: ...

    @property
    def lm_head(self) -> ParallelLMHead | None: ...

    @property
    def vocabulary(self) -> VocabularyPartition: ...

    @property
    def text_pipeline(self: _TextModules) -> LayerPipeline | None: ...

    def embed_input_ids(self: _TextModules, input_ids: torch.Tensor) -> torch.Tensor: ...


class TextMixin:
    """Text computation composed from one backbone and its vocabulary head.

    Properties may expose already-registered modules so text and diffusion can
    share parameters without registering or loading a second backbone.
    """

    @property
    @abstractmethod
    def text_backbone(self) -> Decoder:
        """Borrow the decoder registered by the concrete numerical composition."""

    @property
    @abstractmethod
    def vocabulary(self) -> VocabularyPartition:
        """Describe the logical vocabulary columns produced by this partition."""

    @property
    def dtype(self) -> torch.dtype:
        """Expose the composed backbone's numerical activation representation."""

        return cast(_TextModules, self).text_backbone.dtype

    @property
    def text_pipeline(self) -> LayerPipeline | None:
        return cast(_TextModules, self).text_backbone.pipeline

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedding = cast(_TextModules, self).text_backbone.embed_tokens
        if embedding is None:
            raise ValueError("token embedding belongs to the first pipeline stage")
        return embedding(input_ids)

    @torch.inference_mode()
    def forward(
        self, batch: TextBatch, *, constants: TensorViews, scratch: TensorViews
    ) -> torch.Tensor:
        """Replace selected embedding rows and evaluate the resident decoder layers."""

        pipeline = self.text_pipeline
        hidden = None
        if pipeline is None or pipeline.first:
            hidden = self.embed_input_ids(batch.input_ids.reshape(-1))
            if batch.inputs_embeds is not None:
                assert batch.embedding_mask is not None
                hidden = torch.where(
                    batch.embedding_mask.reshape(-1, 1),
                    batch.inputs_embeds.to(dtype=hidden.dtype),
                    hidden,
                )
        return cast(_TextModules, self).text_backbone(
            hidden, batch.attention, positions=batch.positions
        )

    def compute_logits(self, hidden_states: torch.Tensor, batch: TextBatch) -> TextOutput:
        """Return each sequence's selected hidden or local vocabulary rows."""

        modules = cast(_TextModules, self)
        pipeline = self.text_pipeline
        rows = batch.input_ids.numel()
        if pipeline is not None and not pipeline.last:
            rows = SequencePartition(rows, modules.text_backbone.sequence).count
        # Captured calls may retain padding beyond the live token rows.
        if (
            hidden_states.ndim != 2
            or hidden_states.shape[0] < rows
            or hidden_states.shape[1] != modules.text_backbone.hidden_size
            or hidden_states.dtype != self.dtype
        ):
            raise ValueError("text hidden states disagree with the backbone shape or dtype")
        return project_outputs(
            hidden_states,
            batch,
            modules.lm_head,
            pipeline=self.text_pipeline,
            vocabulary=modules.vocabulary,
        )
