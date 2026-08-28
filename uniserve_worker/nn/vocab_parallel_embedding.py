"""Vocab-parallel embedding and LM head seams.

The classes are identity-shaped at tensor-parallel size 1 except for optional
vocab padding, and they fail loudly for tp>1 unless the required collectives
are initialized.  This keeps the load-time sharding seam usable before a full
distributed runtime is made default.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..execution.forward_batch import MeshView
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
    """Zero the trailing vocab-padding rows of each per-partition tensor.

    The single home for the padding-zero rule shared by the embedding table and
    the LM head: rows beyond ``real_vocab_size`` (this rank's slice of the padded
    vocab) hold no real vocabulary and must read as zero. Tensors whose
    partition lies entirely within the real vocab are left untouched.
    """
    start = max(0, int(real_vocab_size) - int(partition_start))
    if start < int(partition_size):
        for tensor in tensors:
            tensor.data[start:].zero_()


class VocabParallelEmbedding(nn.Module):
    weight: nn.Parameter

    """Embedding table sharded on the vocab axis."""

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
        super().__init__()
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
        # Weight-init policy (H7) is explicit/injectable here. ``LinearBase`` is
        # load-only (leaves weights uninitialized so a missing-weight bug surfaces
        # as garbage); this embedding defaults to a random dummy init for the
        # load-time sharding seam's usability, but callers can pass
        # ``init_weights=False`` to opt into the same load-only policy. The
        # padding rows are always zeroed (a correctness requirement, not init).
        if init_weights:
            nn.init.normal_(self.weight)
        self._zero_padding_rows()

    def _zero_padding_rows(self) -> None:
        zero_vocab_padding(
            self.num_embeddings,
            self.vocab_start_index,
            self.num_embeddings_per_partition,
            self.weight,
        )

    def forward(self, input_ids: torch.Tensor, mesh: MeshView) -> torch.Tensor:
        mask = (
            (input_ids < self.vocab_start_index)
            | (input_ids >= self.vocab_end_index)
            | (input_ids >= self.num_embeddings)
        )
        local_ids = (input_ids - self.vocab_start_index).masked_fill(mask, 0)
        out = F.embedding(local_ids, self.weight)
        out = out.masked_fill(mask.unsqueeze(-1), 0)
        return mesh.all_reduce(out, "tp")


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

    def forward(self, x: torch.Tensor, mesh: MeshView) -> torch.Tensor:  # type: ignore[override]
        local_logits = super().forward(x)
        if not self.gather_output:
            return local_logits
        logits = mesh.all_gather(local_logits, "tp", -1)
        return logits[..., : self.vocab_size]
