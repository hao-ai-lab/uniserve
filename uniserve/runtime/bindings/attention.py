"""Attention operators, plans and exchange backing for one layer call site."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import prod

import torch

from uniserve.model.inputs import TextSize
from uniserve.nn import _binding
from uniserve.nn.attention.inputs import DenseInput

from ..backends import attention as attention_backend
from . import capturing


@dataclass(frozen=True)
class ExchangeBuffers:
    """Flat byte backing for one layer's Ulysses head-exchange tensors."""

    tensors: Mapping[str, torch.Tensor]

    def view(self, name, shape, like, *, offset=0):
        backing = self.tensors[name]
        # Offsets and extents are byte addresses into the flat uint8 backing.
        start, size = (
            offset * like.element_size(),
            prod(shape) * like.element_size(),
        )
        if (
            start < 0
            or start + size > backing.numel()
            or backing.device != like.device
        ):
            raise ValueError("attention exchange exceeds its prepared capacity")
        return backing.narrow(0, start, size).view(like.dtype).view(shape)


class AttentionBinding:
    """Specialize one attention call site.

    Specialize one attention call site's operators, plans, and bound
    metadata.
    """

    def __init__(
        self,
        module,
        backend,
        cache,
        size,
        device,
        dtype,
        allocate,
        context_transport,
    ):
        self.module, self.backend, self.cache, self.size = (
            module,
            backend,
            cache,
            size,
        )
        self.device, self.dtype, self.allocate = device, dtype, allocate
        self.operators = {}
        self._bound = set()
        self._bound_batches = {}
        self.batch = None
        # The caller's batch whose metadata bind() planned for the next call.
        self._fresh = None
        self.context_transport = context_transport
        self.context_plans = {}
        self.cache_batches = {}

    def _context_plan(self, batch, dtype):
        from ._context import _ContextPlan

        key = dtype, _ContextPlan.signature(batch)
        if key not in self.context_plans:
            if capturing(self.device):
                raise RuntimeError(
                    "prepare context attention metadata before capture"
                )
            self.context_plans[key] = _ContextPlan(
                self.module,
                batch,
                cache=self.cache,
                transport=self.context_transport,
                allocate=self.allocate,
                dtype=dtype,
            )
        return self.context_plans[key]

    def prepare(self, dtype, size):
        if self.size is not None and (
            size.num_tokens > self.size.num_tokens
            or size.batch_size > self.size.batch_size
        ):
            raise ValueError(
                "attention exceeds the prepared token or batch capacity"
            )

        previous = self.operators.get(dtype)
        if previous is not None and (
            previous.size.num_tokens >= size.num_tokens
            and previous.size.batch_size >= size.batch_size
        ):
            return previous

        if capturing(self.device):
            raise RuntimeError(
                "attention shape and dtype must be prepared before capture"
            )

        if self.size is not None:
            size = TextSize(
                max(size.num_tokens, self.size.num_tokens),
                max(size.batch_size, self.size.batch_size),
            )

        options = {
            "num_heads": self.module.local_heads,
            "num_kv_heads": self.module.local_kv_heads,
            "head_dim": self.module.head_dim,
            "dtype": dtype,
            "size": size,
            "cache": self.cache,
        }
        provider = attention_backend.resolve(self.backend, device=self.device)
        requirements = provider.workspace_buffers(**options)
        operator = provider.prepare(
            **options, workspace=self.allocate(requirements, self.device)
        )
        self.operators[dtype] = operator
        self._bound.discard(dtype)
        return operator

    def partitions_tokens(self):
        """Report whether token partitions read exact host lengths."""
        return (
            self.module.context_parallel is not None
            or self.module.exchange.group.size > 1
        )

    def reads_host_lengths(self, batch):
        """Report whether binding ``batch`` reads its exact host lengths."""
        return self.partitions_tokens() or any(
            operator.requires_host_lengths(batch)
            for operator in self.operators.values()
        )

    def sequence_inputs(self, batch):
        """Resolve lengths needed to construct mathematical token partitions."""
        if not self.partitions_tokens():
            return batch
        from ..backends.attention._sequences import host_lengths

        return host_lengths(
            batch, prepared=self.batch if capturing(self.device) else None
        )

    def bind(self, batch, *, source=None):
        """Plan numerical metadata for every prepared dtype.

        Graph capture uses these plans. The next eager call that receives
        ``source`` (the caller's batch, defaulting to ``batch``) also uses
        them; later calls plan again, since device columns can change.
        """
        from ..backends.attention._sequences import host_lengths

        self._fresh = batch if source is None else source
        batch = self.sequence_inputs(batch)
        self.batch = batch

        if not isinstance(batch, DenseInput):
            if batch.queries.num_tokens is not None:
                self.prepare(
                    self.dtype,
                    TextSize(
                        batch.queries.num_tokens, batch.queries.batch_size
                    ),
                )
            elif self.size is not None:
                self.prepare(self.dtype, self.size)

        for dtype, operator in self.operators.items():
            numerical = (
                batch
                if self.module.context_parallel is None
                else self._context_plan(batch, dtype).refresh(batch)
            )
            if operator.requires_host_lengths(numerical):
                numerical = host_lengths(numerical)
            operator.bind(numerical)
            self._bound_batches[dtype] = numerical

        self._bound.update(self.operators)

    def __call__(self, q, k, v, batch, *, scale, out):
        from ..backends.attention._sequences import host_lengths

        # A call on the batch just bound reuses that binding's plans and host
        # lengths; the binding serves only this one call.
        fresh = self._fresh is not None and self._fresh is batch
        self._fresh = None
        batch = self.batch if fresh else self.sequence_inputs(batch)
        plan = None
        storage = _binding.attention_storage.get().get(id(self.module))
        if self.module.context_parallel is not None:
            plan = self._context_plan(batch, q.dtype)
            # Plan construction may have grown exchange backing for a longer
            # key domain than the root's query capacity.
            storage = _binding.attention_storage.get().get(id(self.module))
            q, k, v, batch = plan.inputs(q, k, v, batch, storage=storage)

        size = (
            TextSize(
                q.shape[0] if q.ndim == 3 else q.shape[0] * q.shape[2],
                1 if q.ndim == 3 else q.shape[0],
            )
            if isinstance(batch, DenseInput)
            else TextSize(q.shape[0], batch.queries.batch_size)
        )
        operator = self.prepare(q.dtype, size)

        # Numerical metadata can be passed directly in eager code. Providers
        # requiring host planning are bound explicitly before CUDA capture,
        # and a freshly bound eager call uses the same plans.
        if capturing(q.device) or (fresh and q.dtype in self._bound):
            if q.dtype not in self._bound:
                raise RuntimeError(
                    "bind numerical attention metadata before capture"
                )
            if operator.requires_host_lengths(batch):
                batch = host_lengths(
                    batch, prepared=self._bound_batches[q.dtype]
                )
        else:
            # Eager callers may pass fresh lengths or mutate borrowed columns
            # without an explicit bind call. A previous plan for this dtype
            # does not describe the current batch.
            if operator.requires_host_lengths(batch):
                batch = host_lengths(batch)
            operator.bind(batch)
            self._bound.add(q.dtype)
            self._bound_batches[q.dtype] = batch

        result = operator(
            q,
            k,
            v,
            batch,
            scale=scale,
            out=out if plan is None else torch.empty_like(q),
        )
        return (
            result
            if plan is None
            else plan.restore(result, storage=storage, out=out)
        )

    def update_cache(self, k, v, *, indices):
        """Write current K/V into the bound prefix cache at physical indices."""
        if self.cache is None:
            raise RuntimeError("this attention layer has no bound prefix state")

        if self.module.context_parallel is not None:
            from uniserve.nn.attention import SequenceLengths, VarlenInput

            count = indices.numel()
            if count not in self.cache_batches:
                if capturing(indices.device):
                    raise RuntimeError(
                        "prepare context cache writes before capture"
                    )
                lengths = SequenceLengths.from_lengths(
                    (count,), device=indices.device
                )
                self.cache_batches[count] = VarlenInput(
                    lengths, lengths, (False,)
                )

            plan = self._context_plan(self.cache_batches[count], k.dtype)
            storage = _binding.attention_storage.get().get(id(self.module))
            k, v = plan.exchange(k, v, storage=storage)

        self.cache.update(k, v, indices=indices)

    def close(self):
        for operator in self.operators.values():
            operator.close()
        self.operators.clear()
        self._bound_batches.clear()
        self.context_plans.clear()
        self.cache_batches.clear()
        self.batch = self.cache = self._fresh = None
