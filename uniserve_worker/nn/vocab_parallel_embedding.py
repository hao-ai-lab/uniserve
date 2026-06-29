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

from .linear import ColumnParallelLinear
from .mesh import DeviceMesh, get_current_mesh
from .placement import Partial, Replicate, Shard, Sharding, reshard
from .quant.load_state import set_allow_shape_mismatch, set_weight_loader

__all__ = [
    'pad_vocab_size',
    'zero_vocab_padding',
    'VocabParallelEmbedding',
    'ParallelLMHead',
]

# Vocab is padded to a multiple of this for TP sharding / kernel alignment.
_VOCAB_PAD_MULTIPLE = 64

# Reusable placement transitions on the tensor-parallel axis.
_TP_PARTIAL = Sharding((Partial("tp"),))
_TP_REPLICATE = Sharding((Replicate("tp"),))
_TP_SHARD_LAST = Sharding((Shard("tp", -1),))


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
    """Embedding table sharded on the vocab axis."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        *,
        mesh: DeviceMesh | None = None,
        pad_vocab_size_to: int = _VOCAB_PAD_MULTIPLE,
        init_weights: bool = True,
    ) -> None:
        super().__init__()
        mesh = mesh or get_current_mesh()
        self.mesh = mesh
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.padding_idx = None if padding_idx is None else int(padding_idx)
        self.padded_num_embeddings = pad_vocab_size(
            self.num_embeddings,
            pad_to=pad_vocab_size_to,
            tp_size=mesh.tp_size,
        )
        self.num_embeddings_per_partition = self.padded_num_embeddings // int(mesh.tp_size)
        self.vocab_start_index = int(mesh.tp_rank) * self.num_embeddings_per_partition
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, self.embedding_dim)
        )
        set_weight_loader(self.weight, _vocab_weight_loader(self))
        # The vocab partition loader (`_load_vocab_partition`) places this rank's
        # slice directly from the global vocab range, so it needs no ShardPlan.
        set_allow_shape_mismatch(self.weight, True)
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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.mesh.tp_size <= 1:
            return F.embedding(input_ids, self.weight)
        mask = (
            (input_ids < self.vocab_start_index)
            | (input_ids >= self.vocab_end_index)
            | (input_ids >= self.num_embeddings)
        )
        local_ids = (input_ids - self.vocab_start_index).masked_fill(mask, 0)
        out = F.embedding(local_ids, self.weight)
        out = out.masked_fill(mask.unsqueeze(-1), 0)
        return reshard(out, _TP_PARTIAL, _TP_REPLICATE, self.mesh, owner="VocabParallelEmbedding")


class ParallelLMHead(ColumnParallelLinear):
    """LM head whose output vocabulary is sharded across tensor-parallel ranks."""

    def __init__(
        self,
        input_size: int,
        vocab_size: int,
        *,
        bias: bool = False,
        gather_output: bool = True,
        pad_vocab_size_to: int = _VOCAB_PAD_MULTIPLE,
        mesh: DeviceMesh | None = None,
        prefix: str = "",
    ) -> None:
        mesh = mesh or get_current_mesh()
        self.vocab_size = int(vocab_size)
        self.padded_vocab_size = pad_vocab_size(
            self.vocab_size,
            pad_to=pad_vocab_size_to,
            tp_size=mesh.tp_size,
        )
        self.gather_output = bool(gather_output)
        super().__init__(
            input_size,
            self.padded_vocab_size,
            bias=bias,
            prefix=prefix,
            mesh=mesh,
        )
        self.vocab_start_index = int(mesh.tp_rank) * self.output_size
        self.vocab_end_index = self.vocab_start_index + self.output_size
        set_weight_loader(self.weight, _vocab_weight_loader(self))
        set_allow_shape_mismatch(self.weight, True)
        if self.bias is not None:
            set_weight_loader(self.bias, _vocab_weight_loader(self))
            set_allow_shape_mismatch(self.bias, True)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_logits = super().forward(x)
        if self.mesh.tp_size <= 1:
            return local_logits[..., : self.vocab_size]
        if not self.gather_output:
            return local_logits
        logits = reshard(local_logits, _TP_SHARD_LAST, _TP_REPLICATE, self.mesh, owner="ParallelLMHead")
        return logits[..., : self.vocab_size]


def _vocab_weight_loader(module: nn.Module):
    def load(param: nn.Parameter, loaded_weight: torch.Tensor, *, shard_id=None) -> None:
        del shard_id
        vocab_size = (
            int(module.num_embeddings)
            if hasattr(module, "num_embeddings")
            else int(module.vocab_size)
        )
        _load_vocab_partition(
            param,
            loaded_weight,
            vocab_size=vocab_size,
            start=int(module.vocab_start_index),
            end=int(module.vocab_end_index),
        )
        zero_padding = getattr(module, "_zero_padding_rows", None)
        if callable(zero_padding):
            zero_padding()

    return load


def _load_vocab_partition(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    *,
    vocab_size: int,
    start: int,
    end: int,
) -> None:
    target = param.data
    loaded = loaded_weight.to(device=target.device, dtype=target.dtype)
    if loaded.shape == target.shape:
        target.copy_(loaded)
        return
    if loaded.ndim != target.ndim or loaded.shape[1:] != target.shape[1:]:
        raise ValueError(f"loaded vocab tensor shape {loaded.shape} != target {target.shape}")
    target.zero_()
    copy_start = max(0, int(start))
    copy_end = min(int(end), int(vocab_size), int(loaded.shape[0]))
    if copy_end <= copy_start:
        return
    target[copy_start - start : copy_end - start].copy_(loaded[copy_start:copy_end])
