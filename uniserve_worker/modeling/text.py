"""Shared token embedding, decoder invocation, and vocabulary projection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import torch

from ..nn.decoder.base import Decoder
from ..nn.logits import project_outputs
from ..nn.parallel_pipeline import LayerPipeline
from ..nn.parallel_sequence import SequencePartition
from ..nn.vocab_parallel_embedding import ParallelLMHead
from ..transfer.layout import TensorRegion
from .batch import TextBatch, TextOutput
from .components import Call
from .geometry import Shape, TextShape
from .resources import TensorNeeds, TensorSchema
from .tensors import TensorViews, TokenSelection, VocabularyPartition

if TYPE_CHECKING:
    from .model import Model


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
    def dtype(self) -> torch.dtype:
        """Expose the composed backbone's numerical activation representation."""

        return cast(_TextModules, self).text_backbone.dtype

    def tensor_specs(self, call: Call, shape: Shape) -> TensorNeeds:
        """Declare backbone, selected hidden, or local vocabulary result rows."""

        if call is not Call.TEXT:
            return cast("Model", super()).tensor_specs(call, shape)
        if not isinstance(shape, TextShape):
            raise ValueError("text computation requires token and selection geometry")
        model = cast("Model", self)
        if shape.selection in {None, TokenSelection.HIDDEN}:
            region = None
            pipeline = self.text_pipeline
            if shape.selection is None and pipeline is not None and not pipeline.last:
                partition = SequencePartition(
                    shape.tokens, cast(_TextModules, self).text_backbone.sequence
                )
                region = TensorRegion((partition.start, 0), (partition.count, model.hidden_size))
            return TensorNeeds(
                outputs={
                    "hidden_states": TensorSchema(
                        (shape.tokens, model.hidden_size),
                        self.dtype,
                        region=region,
                        variable_axes=(0,) if shape.selection is None else (),
                    )
                }
            )
        vocabulary = cast(_TextModules, self).vocabulary
        rows = shape.rows if shape.selection is TokenSelection.LAST_LOGITS else shape.tokens
        columns = vocabulary.width * len(vocabulary.backend_order)
        region = (
            TensorRegion((0, vocabulary.rank * vocabulary.width), (rows, vocabulary.width))
            if len(vocabulary.backend_order) > 1
            else None
        )
        return TensorNeeds(
            outputs={
                "logits": TensorSchema(
                    (rows, columns),
                    self.dtype,
                    region=region,
                    variable_axes=(0,) if shape.selection is TokenSelection.LAST_LOGITS else (),
                )
            }
        )

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
        return project_outputs(
            hidden_states,
            batch,
            modules.lm_head,
            pipeline=self.text_pipeline,
            vocabulary=modules.vocabulary,
        )
