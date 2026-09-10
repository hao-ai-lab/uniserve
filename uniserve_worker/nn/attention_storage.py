"""Borrow execution-owned storage for attention head and row transfers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from math import prod

import torch

from .mesh import Communicator


@dataclass(frozen=True)
class AttentionExchangeStorage:
    """Disjoint byte capacities borrowed by one serialized execution domain.

    Q/K/V buffers may be reused once attention finishes reading them. Output
    exchange buffers remain live while the next layer prepares its Q/K/V.
    The runtime retains the backing allocations through all graph lifetimes.
    """

    buffers: Mapping[str, torch.Tensor]

    def view(
        self,
        name: str,
        shape: tuple[int, ...],
        like: torch.Tensor,
        *,
        offset: int = 0,
    ) -> torch.Tensor:
        """Borrow a contiguous view; offset is measured in elements of like.dtype."""

        storage = self.buffers[name]
        elements = prod(shape)
        begin = offset * like.element_size()
        size = elements * like.element_size()
        if (
            storage.device != like.device
            or offset < 0
            or any(extent < 0 for extent in shape)
            or begin + size > storage.numel()
        ):
            raise ValueError("attention transfer exceeds its execution-owned capacity")
        return storage.narrow(0, begin, size).view(like.dtype).view(shape)


_ACTIVE_STORAGE: ContextVar[Mapping[Communicator, AttentionExchangeStorage]] = ContextVar(
    "attention_exchange_storage", default={}
)


@contextmanager
def attention_exchange_scope(
    storage: Mapping[Communicator, AttentionExchangeStorage],
) -> Iterator[None]:
    """Bind an execution domain's reusable transfer buffers while enqueueing work."""

    token = _ACTIVE_STORAGE.set(storage)
    try:
        yield
    finally:
        _ACTIVE_STORAGE.reset(token)


def attention_exchange_storage(group: Communicator) -> AttentionExchangeStorage | None:
    """Return storage assigned to the current execution domain and communicator."""

    return _ACTIVE_STORAGE.get().get(group)
