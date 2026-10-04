"""Numerical vocabulary shards and explicit global logit gathering."""

from dataclasses import dataclass

import torch
from torch import nn

from uniserve._slices import within
from uniserve.distributed import Communicator


@dataclass(frozen=True, slots=True)
class VocabShard:
    """Own a logical interval of the padded vocabulary, borrowing its group.

    size excludes padding. local_slice indexes the padded vocabulary; gather
    removes columns beyond size after restoring logical communicator order.
    """

    size: int
    local_slice: slice
    padded_size: int
    group: Communicator

    def __post_init__(self):
        if (
            not 0 < self.size <= self.padded_size
            or self.padded_size % self.group.size
        ):
            raise ValueError(
                "vocabulary padding must cover real tokens and divide its group"
            )
        if not within((self.local_slice,), (self.padded_size,)):
            raise ValueError(
                "local vocabulary slice exceeds the padded vocabulary"
            )
        width = self.padded_size // self.group.size
        if self.local_slice != slice(
            self.group.rank * width, (self.group.rank + 1) * width
        ):
            raise ValueError(
                "vocabulary slice must follow the logical communicator "
                "membership"
            )


@dataclass(frozen=True, slots=True)
class Logits:
    values: torch.Tensor
    vocab: VocabShard

    def __post_init__(self):
        if (
            self.values.ndim < 1
            or self.values.shape[-1]
            != self.vocab.local_slice.stop - self.vocab.local_slice.start
        ):
            raise ValueError(
                "local logits must match their vocabulary shard width"
            )

    def gather(self, *, out: torch.Tensor | None = None) -> torch.Tensor:
        """Gather vocabulary columns in logical order and discard padding."""
        shape = (*self.values.shape[:-1], self.vocab.size)
        if out is not None and (
            out.shape != shape
            or out.dtype != self.values.dtype
            or out.device != self.values.device
        ):
            raise ValueError(
                "global logits output must match unpadded shape, dtype and "
                "device"
            )

        # Without padding, the gather writes the caller's storage directly.
        target = out if self.vocab.padded_size == self.vocab.size else None
        result = self.vocab.group.all_gather(self.values, dim=-1, out=target)[
            ..., : self.vocab.size
        ]
        return result if out is None else out.copy_(result)


def project_logits(
    head: nn.Module, hidden: torch.Tensor, token_indices: torch.Tensor
) -> Logits:
    """Project caller-selected hidden rows through a vocabulary head.

    ``hidden`` holds packed rows ``[tokens, hidden]`` and ``token_indices``
    the integer rows to project; empty selections remain empty. The head
    must expose its ``VocabShard`` as ``head.vocab``. Vocabulary columns stay
    local until the caller requests ``Logits.gather``.
    """
    if hidden.ndim != 2 or token_indices.ndim != 1:
        raise ValueError(
            "logits require packed hidden rows and one-dimensional token "
            "indices"
        )
    if token_indices.dtype not in {torch.int32, torch.int64}:
        raise ValueError("token indices must be integers")

    vocab = head.vocab
    if not isinstance(vocab, VocabShard):
        raise TypeError("the vocabulary head must expose a VocabShard")
    return Logits(head(hidden.index_select(0, token_indices)), vocab)
