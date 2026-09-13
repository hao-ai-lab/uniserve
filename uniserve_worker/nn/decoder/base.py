"""Shared embedding and mathematical partitioning for transformer decoders."""

from __future__ import annotations

from abc import abstractmethod
from typing import cast

import torch
from torch import nn

from ...modeling.tensors import AttentionMetadata
from ..layer import LayerConfig
from ..parallel_pipeline import LayerPipeline
from ..parallel_sequence import SequencePartition
from ..row_pipeline import RowStage, RowTensors, RowTensorSegments, run_row_pipeline
from ..vocab_parallel_embedding import VocabParallelEmbedding


class Decoder(nn.Module):
    """Compose first-stage embeddings and the decoder's logical layer partition.

    Concrete decoders supply their attention, residual, normalization, and
    position equations. Parameter names stay relative to the existing backbone.
    """

    layers: nn.ModuleDict

    @property
    def dtype(self) -> torch.dtype:
        """Return the activation representation, independent of quantized projections.

        Resident layers retain an input normalization weight in the arithmetic
        dtype also used to receive pipeline activations.
        """

        first = next(iter(self.layers.values()))
        normalization = cast(nn.Module, first.input_layernorm)
        return cast(torch.Tensor, normalization.weight).dtype

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        layer_count: int,
        *,
        layer_config: LayerConfig,
        padding_idx: int | None = None,
        init_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.pipeline = LayerPipeline(layer_config.pipeline, layer_count)
        self.sequence = layer_config.sequence
        self.hidden_size = hidden_size
        self.embed_tokens = (
            VocabParallelEmbedding(
                vocab_size,
                hidden_size,
                padding_idx,
                layer_config=layer_config,
                init_weights=init_embeddings,
            )
            if self.pipeline.first
            else None
        )

    def receive(
        self,
        inputs_embeds: torch.Tensor | None,
        token_count: int,
        *,
        reference: torch.Tensor,
        partition: SequencePartition | None,
        residual: bool = False,
    ) -> RowTensors:
        """Supply this stage's activation and, when needed, its separate residual.

        The first stage consumes embeddings; later stages receive the preceding
        stage's numerical states. ``reference`` fixes the receiving activation
        dtype independently of quantized projection weights. Sequence slicing
        happens before communication so PP peers exchange matching local rows.
        """

        if self.pipeline.first:
            if inputs_embeds is None:
                raise ValueError("the first decoder stage requires input embeddings")
            hidden = inputs_embeds if partition is None else partition.local(inputs_embeds)
            return (hidden,)

        rows = token_count if partition is None else partition.count
        hidden = reference.new_empty((rows, self.hidden_size))
        values = (hidden, torch.empty_like(hidden)) if residual else (hidden,)
        self.pipeline.receive_activation(*values)
        return values

    def run_layers(
        self, values: RowTensors, stages: tuple[RowStage[RowTensorSegments], ...]
    ) -> RowTensors:
        """Traverse resident layers and pass their states to the next PP stage.

        The row pipeline preserves projection/attention dependencies across
        layers. Residual tensors remain separate throughout traversal and PP
        transfer; the last stage applies its model's final normalization.
        """

        values = run_row_pipeline(RowTensorSegments.complete(values), stages).materialize()
        self.pipeline.send_activation(*values)
        return values

    @abstractmethod
    def forward(
        self,
        inputs_embeds: torch.Tensor | None,
        context: AttentionMetadata,
        *,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate resident layers using the supplied numerical attention geometry."""

        raise NotImplementedError
