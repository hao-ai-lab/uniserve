"""Numerical attention layers borrowing context-bound operators and prefix
state.
"""  # noqa: D205

from __future__ import annotations

import math

import torch
from torch import nn

from uniserve.distributed import Communicator, DeviceMesh
from uniserve.distributed.tokens import HeadExchange, TokenShard
from uniserve.nn import _binding, functional
from uniserve.nn.functional._tensors import result

from ._parallel import ParallelAttention
from .config import AttentionParallelConfig
from .inputs import AttentionBatch, DenseInput, PagedInput, SegmentedInput


class Attention(nn.Module):
    """Apply attention to local projections using the declared sequence indices.

    Head counts describe the full architecture. Mathematical parallel binding
    assigns local query/KV heads, token ownership and the global cache-head IDs.
    Cache state and mutable kernel resources are borrowed from the active call.

    ``window`` bounds the visible history in tokens; ``None`` reads the whole
    history. Queries align to the end of their key sequence, so a paged query
    token ``i`` sits at ``prefix + i`` and a sequence's query tokens are its
    final keys. A causal query at absolute position ``q`` reads keys
    ``[max(q - window, 0), q]``. A non-causal query reads every query token
    of its sequence, a block attending to itself in both directions, and the
    keys before them from ``max(q - window, 0)`` on. A segmented input's
    queries read the fixed prefix interval ``[max(P - window, 0), P)`` of
    their prefix length ``P`` together with their declared current keys,
    which the window does not bound.
    """

    # Recorded by parallelize_ once the layer is bound to its partition.
    _parallel_mesh: DeviceMesh
    _attention_parallel: AttentionParallelConfig

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        scale: float | None = None,
        cache_name: str | None = None,
        window: int | None = None,
    ):
        super().__init__()
        if (
            min(num_heads, num_kv_heads, head_dim) < 1
            or num_heads % num_kv_heads
        ):
            raise ValueError(
                "attention requires positive compatible query and KV heads"
            )
        if window is not None and (type(window) is not int or window < 0):
            raise ValueError(
                "attention windows must be nonnegative token counts"
            )
        self.window = window
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5 if scale is None else scale
        if not math.isfinite(self.scale):
            raise ValueError("attention scale must be finite")
        if cache_name is not None and (
            not isinstance(cache_name, str) or not cache_name
        ):
            raise ValueError("cache names must identify a numerical layer path")
        self.cache_name = cache_name
        self.local_heads = num_heads
        self.local_kv_heads = num_kv_heads
        self.head_indices = tuple(range(num_kv_heads))
        self.exchange = HeadExchange(Communicator())
        self.context_parallel: ParallelAttention | None = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention: AttentionBatch,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply attention to ``[tokens, heads, head_dim]`` projections.

        ``attention`` supplies this call's inputs per cache table; the layer
        reads its own table's entry. Returns a tensor shaped like ``q``,
        written into ``out`` when given. Under Ulysses the token shard is
        exchanged for this rank's head shard before compute and restored to
        token layout afterwards.
        """
        if out is not None and (
            out.shape != q.shape
            or out.dtype != q.dtype
            or out.device != q.device
        ):
            raise ValueError(
                "attention output must match the local query representation"
            )

        group = self.exchange.group
        operator = _binding.attention.get().get(id(self))
        batch = attention.entry(None if operator is None else operator.table)
        if self.context_parallel is not None:
            if operator is None:
                raise RuntimeError(
                    "context attention requires an active ExecutionContext"
                )
            destination = torch.empty_like(q) if out is None else out
            return operator(q, k, v, batch, scale=self.scale, out=destination)
        if group.size == 1:
            return self._compute(operator, q, k, v, batch, out)

        if isinstance(batch, DenseInput) or q.ndim != 3:
            raise ValueError(
                "Ulysses attention requires explicit packed sequence lengths"
            )
        sequences = batch
        if sequences.queries.num_tokens is None:
            if operator is None:
                raise RuntimeError(
                    "device-only Ulysses lengths require an active "
                    "ExecutionContext"
                )
            sequences = operator.sequence_inputs(sequences)
        num_tokens = sequences.queries.num_tokens
        if num_tokens is None:
            raise RuntimeError(
                "Ulysses token partitions require host sequence lengths"
            )
        partition = TokenShard(num_tokens, group)
        if partition.num_tokens == 0:
            return torch.empty_like(q) if out is None else out

        # Token shard -> this rank's head shard, over the padded common layout.
        storage = _binding.attention_storage.get().get(id(self))
        query, key, value = (
            self.exchange.heads(
                partition.pad(tensor), storage=storage, role=role
            )[: partition.num_tokens]
            for role, tensor in zip(
                ("query", "key", "value"), (q, k, v), strict=True
            )
        )
        attended = self._compute(operator, query, key, value, sequences, None)

        # Head shard -> token shard. The exchange spans the padded physical
        # token count, so a short compute result is zero-filled first.
        physical_tokens = partition.capacity * group.size
        if attended.shape[0] != physical_tokens:
            padded = attended.new_zeros((physical_tokens, *attended.shape[1:]))
            padded[: partition.num_tokens].copy_(attended)
            attended = padded
        attended = self.exchange.tokens(attended, storage=storage)[
            : partition.count
        ]
        return result(attended, out)

    def _compute(self, operator, q, k, v, batch, out):
        if q.shape[1] != self.local_heads or k.shape[-1] != self.head_dim:
            raise ValueError(
                "attention projections do not match the bound head partition"
            )

        if operator is None:
            if isinstance(batch, SegmentedInput) or (
                isinstance(batch, PagedInput)
                and batch.write_indices is not None
            ):
                raise RuntimeError(
                    "cached attention requires an active ExecutionContext"
                )
            return functional.attention(
                q, k, v, batch, scale=self.scale, window=self.window, out=out
            )
        destination = torch.empty_like(q) if out is None else out
        return operator(q, k, v, batch, scale=self.scale, out=destination)

    def update_cache(
        self, k: torch.Tensor, v: torch.Tensor, attention: AttentionBatch
    ) -> None:
        """Write this call's K/V at its table's addresses, without attending.

        The layer's own table entry supplies the physical write addresses; an
        entry without write addresses publishes nothing. No request is
        scheduled or committed. With Ulysses, each rank supplies its local
        token interval and TP-local heads, and the same exchange as forward
        produces this rank's cache heads.
        """
        operator = _binding.attention.get().get(id(self))
        entry = attention.entry(None if operator is None else operator.table)
        indices = getattr(entry, "write_indices", None)
        if indices is None:
            return
        if operator is None:
            raise RuntimeError(
                "cache writes require an active ExecutionContext"
            )

        if self.context_parallel is not None:
            operator.update_cache(k, v, indices=indices)
            return

        if self.exchange.group.size > 1:
            partition = TokenShard(indices.numel(), self.exchange.group)
            if partition.num_tokens == 0:
                return
            storage = _binding.attention_storage.get().get(id(self))
            k, v = (
                self.exchange.heads(
                    partition.pad(tensor), storage=storage, role=role
                )[: partition.num_tokens]
                for role, tensor in (("key", k), ("value", v))
            )

        operator.update_cache(k, v, indices=indices)
