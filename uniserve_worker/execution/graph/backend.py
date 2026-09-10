"""Graph ownership independent of model inputs and execution geometry."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable
from typing import Generic, TypeVar

T = TypeVar("T")


class GraphExecutionError(RuntimeError):
    """A configured CUDA graph could not execute safely."""


class CudaGraphBackend(ABC, Generic[T]):
    """Own recorded computations and their borrowed outputs.

    Callers stage inputs and order replay on the bound execution context. They
    must complete GPU use before discarding or closing graphs. Capture warms
    the callable and restores its entry state without advancing business state.
    """

    @abstractmethod
    def capture_one(
        self,
        key: Hashable,
        forward: Callable[[], T],
        *,
        keepalive: tuple[object, ...] = (),
        restore: Callable[[], None] | None = None,
    ) -> None: ...

    @abstractmethod
    def contains(self, key: Hashable) -> bool: ...

    @abstractmethod
    def replay(self, key: Hashable) -> T: ...

    @abstractmethod
    def discard(self, key: Hashable) -> None: ...

    @abstractmethod
    def close(self) -> None: ...
