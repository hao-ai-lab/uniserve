"""Execution binding of backend plans during numerical attention calls."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uniserve.attention.video_sparse_provider import SparseAttentionProvider

# Captured kernels retain the selected plan's addresses. The caller owns the
# plan and must keep it alive until the graph and its result readers retire.
_CALL: ContextVar[tuple[int | None, bool]] = ContextVar("attention_call", default=(None, False))
_SPARSE: ContextVar[SparseAttentionProvider | None] = ContextVar("sparse_attention", default=None)


@contextmanager
def sparse_attention_scope(provider: SparseAttentionProvider | None) -> Iterator[None]:
    """Borrow one serialized execution domain's sparse plans while enqueueing.

    The owner retains the provider until its graphs and readers retire. Nested
    scopes restore the enclosing binding even when numerical execution fails.
    """

    token = _SPARSE.set(provider)
    try:
        yield
    finally:
        _SPARSE.reset(token)


def sparse_attention_provider() -> SparseAttentionProvider:
    """Return the active sparse provider; numerical calls cannot allocate owners."""

    provider = _SPARSE.get()
    if provider is None:
        raise RuntimeError("sparse attention requires an execution-bound provider")
    return provider


@contextmanager
def attention_scope(binding: int | None, *, capture: bool) -> Iterator[None]:
    """Select a backend plan without adding execution fields to model inputs."""

    token = _CALL.set((binding, capture))
    try:
        yield
    finally:
        _CALL.reset(token)


def attention_binding() -> int | None:
    return _CALL.get()[0]


def capturing_attention() -> bool:
    return _CALL.get()[1]
