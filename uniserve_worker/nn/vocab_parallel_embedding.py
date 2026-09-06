"""Vocabulary-sharded embedding and language-model projection with shared padding rules.

Vocabulary rows are padded once to satisfy kernel alignment and even tensor-parallel
partitioning. Loaders preserve that layout, and both embedding and output projection
zero synthetic rows so padded token ids cannot affect model results.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import LayerConfig
from .linear import ColumnParallelLinear

__all__ = [
    "pad_vocab_size",
    "zero_vocab_padding",
    "VocabParallelEmbedding",
    "ParallelLMHead",
]

# Vocab is padded to a multiple of this for TP sharding / kernel alignment.
_VOCAB_PAD_MULTIPLE = 64


def pad_vocab_size(vocab_size: int, *, pad_to: int = _VOCAB_PAD_MULTIPLE, tp_size: int = 1) -> int:
    """Round vocabulary rows to a kernel-aligned size divisible across TP ranks."""

    vocab_size = int(vocab_size)
    pad_to = max(1, int(pad_to))
    tp_size = max(1, int(tp_size))
    multiple = math.lcm(pad_to, tp_size)
    return ((vocab_size + multiple - 1) // multiple) * multiple


def zero_vocab_padding(
    real_vocab_size: int,
    partition_start: int,
    partition_size: int,
    *tensors: torch.Tensor,
) -> None:
    """Zero synthetic rows in one tensor-parallel vocabulary partition.

    Apply the same boundary after initialization and checkpoint loading so every
    tensor sharing the vocabulary layout treats padding identically.
    """

    start = max(0, int(real_vocab_size) - int(partition_start))
    if start < int(partition_size):
        for tensor in tensors:
            tensor.data[start:].zero_()


class VocabParallelEmbedding(nn.Module):
    """Embeds local vocabulary shards and reduces masked results across tensor-parallel ranks."""

    weight: nn.Parameter  # Embedding rows owned by this tensor-parallel rank.

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        *,
        layer_config: LayerConfig,
        pad_vocab_size_to: int = _VOCAB_PAD_MULTIPLE,
        init_weights: bool = True,
    ) -> None:
        """Allocate this rank's padded vocabulary interval and checkpoint-load metadata."""

        super().__init__()
        self.tp_group = layer_config.tensor_group()
        parallel = layer_config.parallel
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.padding_idx = None if padding_idx is None else int(padding_idx)
        self.padded_num_embeddings = pad_vocab_size(
            self.num_embeddings,
            pad_to=pad_vocab_size_to,
            tp_size=parallel.size,
        )
        self.num_embeddings_per_partition = self.padded_num_embeddings // int(parallel.size)
        self.vocab_start_index = int(parallel.rank) * self.num_embeddings_per_partition
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, self.embedding_dim)
        )
        from ..loader.weight_loaders import set_vocab_layout

        set_vocab_layout(
            self.weight,
            real_size=self.num_embeddings,
            start=self.vocab_start_index,
            end=self.vocab_end_index,
        )
        self.reset_parameters(init_weights=init_weights)

    def reset_parameters(self, *, init_weights: bool = True) -> None:
        """Optionally initialize real rows and always zero non-vocabulary padding."""

        if init_weights:
            nn.init.normal_(self.weight)
        self._zero_padding_rows()

    def _zero_padding_rows(self) -> None:
        """Zero synthetic vocabulary rows in this rank's embedding partition."""

        zero_vocab_padding(
            self.num_embeddings,
            self.vocab_start_index,
            self.num_embeddings_per_partition,
            self.weight,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed locally owned ids and sum masked shard outputs across TP ranks."""

        mask = (
            (input_ids < self.vocab_start_index)
            | (input_ids >= self.vocab_end_index)
            | (input_ids >= self.num_embeddings)
        )
        local_ids = (input_ids - self.vocab_start_index).masked_fill(mask, 0)
        out = F.embedding(local_ids, self.weight)
        out = out.masked_fill(mask.unsqueeze(-1), 0)
        return self.tp_group.all_reduce(out)


class ParallelLMHead(ColumnParallelLinear):
    """LM head whose output vocabulary is sharded across tensor-parallel ranks."""

    def __init__(
        self,
        input_size: int,
        vocab_size: int,
        *,
        layer_config: LayerConfig,
        bias: bool = False,
        gather_output: bool = True,
        pad_vocab_size_to: int = _VOCAB_PAD_MULTIPLE,
        prefix: str = "",
    ) -> None:
        """Shard a padded vocabulary projection and configure optional global-logit gathering."""

        self.tp_group = layer_config.tensor_group()
        parallel = layer_config.parallel
        self.vocab_size = int(vocab_size)
        self.padded_vocab_size = pad_vocab_size(
            self.vocab_size,
            pad_to=pad_vocab_size_to,
            tp_size=parallel.size,
        )
        self.gather_output = bool(gather_output)
        super().__init__(
            input_size,
            self.padded_vocab_size,
            layer_config=layer_config,
            bias=bias,
            prefix=prefix,
        )
        self.vocab_start_index = int(parallel.rank) * self.output_size
        self.vocab_end_index = self.vocab_start_index + self.output_size
        from ..loader.weight_loaders import set_vocab_layout

        set_vocab_layout(
            self.weight,
            real_size=self.vocab_size,
            start=self.vocab_start_index,
            end=self.vocab_end_index,
        )
        if self.bias is not None:
            set_vocab_layout(
                self.bias,
                real_size=self.vocab_size,
                start=self.vocab_start_index,
                end=self.vocab_end_index,
            )
        self._zero_padding_rows()

    def _zero_padding_rows(self) -> None:
        """Zero vocabulary-padding rows so gathered logits cannot expose synthetic tokens."""

        zero_vocab_padding(
            self.vocab_size,
            self.vocab_start_index,
            self.output_size,
            self.weight,
        )
        if self.bias is not None:
            zero_vocab_padding(
                self.vocab_size,
                self.vocab_start_index,
                self.output_size,
                self.bias,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Project local vocabulary logits and optionally gather the unpadded vocabulary."""

        local_logits = super().forward(x)
        if not self.gather_output:
            return local_logits
        logits = self.tp_group.all_gather(local_logits, -1)
        return logits[..., : self.vocab_size]
