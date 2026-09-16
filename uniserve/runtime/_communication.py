"""Stream-scoped collective bindings for model communicators."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol

import torch


class StreamCollectives(Protocol):
    """Ordered collectives for one computation binding.

    Ordered collectives and deferred transfers for one computation binding.
    """

    def all_reduce(self, value: torch.Tensor, op: str = "sum") -> None: ...
    def all_gather(self, output: torch.Tensor, value: torch.Tensor) -> None: ...
    def start_all_gather(
        self, output: torch.Tensor, value: torch.Tensor
    ) -> Any: ...
    def start_all_to_all(
        self, output: torch.Tensor, value: torch.Tensor
    ) -> Any: ...
    def all_to_all(
        self,
        output: torch.Tensor,
        value: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
    ) -> None: ...
    def gather(
        self, outputs: list[torch.Tensor] | None, value: torch.Tensor, root: int
    ) -> None: ...
    def broadcast(self, value: torch.Tensor, root: int) -> None: ...
    def reduce_scatter(
        self, output: torch.Tensor, value: torch.Tensor
    ) -> None: ...

    def send(self, value: torch.Tensor, peer: int) -> None: ...
    def recv(self, value: torch.Tensor, peer: int) -> None: ...
    def send_recv(
        self, output: torch.Tensor, value: torch.Tensor, dst: int, src: int
    ) -> None: ...

    def close(self) -> None: ...


_STREAM_COLLECTIVES: ContextVar[Mapping[str, StreamCollectives] | None] = (
    ContextVar("stream_collectives", default=None)
)


@contextmanager
def stream_collective_scope(
    bindings: Mapping[str, StreamCollectives] | None,
) -> Iterator[None]:
    """Select communication resources for one numerical invocation.

    ``None`` selects initialized process-group communication on an ordinary
    device stream. A mapping selects explicit stream bindings and must cover
    every used communicator, including when that mapping is empty.
    """
    token = _STREAM_COLLECTIVES.set(bindings)
    try:
        yield
    finally:
        _STREAM_COLLECTIVES.reset(token)


def stream_collectives(group_name: str) -> StreamCollectives | None:
    """Resolve a stream provider.

    Resolve a stream provider, rejecting incomplete explicit execution
    bindings.

    An unscoped numerical caller may use the initialized process group. Once a
    caller selects an explicit stream scope, every used group must belong to it;
    falling back would enqueue communication on a different execution stream.
    """
    bindings = _STREAM_COLLECTIVES.get()
    if bindings is None:
        return None
    try:
        return bindings[group_name]
    except KeyError:
        raise RuntimeError(
            f"stream scope has no binding for communicator {group_name!r}"
        ) from None
