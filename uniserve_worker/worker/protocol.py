"""Typed process-boundary contract implemented by assembled workers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..batch import Batch, CompletionReport, SnapshotRef
from ..capabilities import WorkerCapabilities


@runtime_checkable
class Worker(Protocol):
    @property
    def capabilities(self) -> WorkerCapabilities: ...

    def warmup(self) -> None: ...

    def execute(self, batch: Batch) -> CompletionReport: ...

    def drop_session(self, session_id: int) -> None: ...

    def copy_kv(self, copies: tuple[tuple[int, int], ...]) -> None: ...

    def release_products(self, handles: tuple[int, ...]) -> None: ...

    def resource_pressure(self) -> list[dict[str, object]]: ...

    def snapshot_session(self, session_id: int) -> SnapshotRef: ...

    def restore_session(self, reference: SnapshotRef) -> None: ...
__all__ = ["Worker"]
