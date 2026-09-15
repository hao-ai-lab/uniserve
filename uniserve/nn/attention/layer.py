"""Numerical attention layers borrowing context-bound operators and prefix state."""

from __future__ import annotations

import math

import torch
from torch import nn

from uniserve.distributed import Communicator
from uniserve.distributed._tokens import _HeadExchange, _TokenShard
from uniserve.nn import _binding, functional

from .inputs import AttentionInput, DenseInput, PagedInput, SegmentedInput


class Attention(nn.Module):
    """Apply attention to local projections using the declared sequence indices.

    Head counts describe the full architecture. Mathematical parallel binding
    assigns local query/KV heads, token ownership and the global cache-head IDs.
    Cache state and mutable kernel resources are borrowed from the active call.
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        scale: float | None = None,
        cache_name: str | None = None,
    ):
        super().__init__()
        if min(num_heads, num_kv_heads, head_dim) < 1 or num_heads % num_kv_heads:
            raise ValueError("attention requires positive compatible query and KV heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5 if scale is None else scale
        if not math.isfinite(self.scale):
            raise ValueError("attention scale must be finite")
        if cache_name is not None and (not isinstance(cache_name, str) or not cache_name):
            raise ValueError("cache names must identify a numerical layer path")
        self.cache_name = cache_name
        self._local_heads = num_heads
        self._local_kv_heads = num_kv_heads
        self._head_indices = tuple(range(num_kv_heads))
        self._exchange = _HeadExchange(Communicator())
        self._context = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch: AttentionInput,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if out is not None and (
            out.shape != q.shape or out.dtype != q.dtype or out.device != q.device
        ):
            raise ValueError("attention output must match the local query representation")
        group = self._exchange.group
        operator = _binding.attention.get().get(id(self))
        if self._context is not None:
            if operator is None:
                raise RuntimeError("context attention requires an active ExecutionContext")
            destination = torch.empty_like(q) if out is None else out
            return operator(q, k, v, batch, scale=self.scale, out=destination)
        if group.size == 1:
            return self._compute(operator, q, k, v, batch, out)
        if isinstance(batch, DenseInput) or q.ndim != 3:
            raise ValueError("Ulysses attention requires explicit packed sequence lengths")
        if batch.queries.num_tokens is None:
            if operator is None:
                raise RuntimeError("device-only Ulysses lengths require an active ExecutionContext")
            batch = operator.sequence_inputs(batch)
        partition = _TokenShard(batch.queries.num_tokens, group)
        if partition.num_tokens == 0:
            return torch.empty_like(q) if out is None else out
        storage = _binding.attention_storage.get().get(id(self))
        query, key, value = (
            self._exchange.heads(partition.pad(tensor), storage=storage, role=role)[
                : partition.num_tokens
            ]
            for role, tensor in zip(("query", "key", "value"), (q, k, v), strict=True)
        )
        result = self._compute(operator, query, key, value, batch, None)
        physical_tokens = partition.capacity * group.size
        if result.shape[0] != physical_tokens:
            padded = result.new_zeros((physical_tokens, *result.shape[1:]))
            padded[: partition.num_tokens].copy_(result)
            result = padded
        result = self._exchange.tokens(result, storage=storage)[: partition.count]
        return functional._result(result, out)

    def _compute(self, operator, q, k, v, batch, out):
        if q.shape[1] != self._local_heads or k.shape[-1] != self.head_dim:
            raise ValueError("attention projections do not match the bound head partition")
        if operator is None:
            if isinstance(batch, SegmentedInput) or (
                isinstance(batch, PagedInput) and batch.write_indices is not None
            ):
                raise RuntimeError("cached attention requires an active ExecutionContext")
            return functional.attention(q, k, v, batch, scale=self.scale, out=out)
        destination = torch.empty_like(q) if out is None else out
        return operator(q, k, v, batch, scale=self.scale, out=destination)

    def update_cache(self, k: torch.Tensor, v: torch.Tensor, *, indices: torch.Tensor) -> None:
        """Write one global token view without scheduling or committing a request.

        With Ulysses, each rank supplies its local token interval and TP-local
        heads. The same exchange as forward produces this rank's cache heads.
        """

        operator = _binding.attention.get().get(id(self))
        if operator is None:
            raise RuntimeError("cache writes require an active ExecutionContext")
        if self._context is not None:
            operator.update_cache(k, v, indices=indices)
            return
        if self._exchange.group.size > 1:
            partition = _TokenShard(indices.numel(), self._exchange.group)
            if partition.num_tokens == 0:
                return
            storage = _binding.attention_storage.get().get(id(self))
            k, v = (
                self._exchange.heads(partition.pad(tensor), storage=storage, role=role)[
                    : partition.num_tokens
                ]
                for role, tensor in (("key", k), ("value", v))
            )
        operator.update_cache(k, v, indices=indices)
