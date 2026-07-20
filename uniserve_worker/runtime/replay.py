"""Terminal operation replay records."""

from __future__ import annotations

import copy
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Protocol

from ..foundation.errors import invalid_descriptor

__all__ = ["ReplayStore"]


class OperationIdentity(Protocol):
    session_id: int
    epoch: int
    op_id: int
    digest: str


class ReplayStore:
    """Bounded terminal-result store keyed by logical operation identity."""

    def __init__(self, capacity: int = 1024) -> None:
        if int(capacity) <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self._records: OrderedDict[tuple[int, int, int], tuple[str, dict[str, Any]]] = OrderedDict()

    @staticmethod
    def _key(operation: OperationIdentity) -> tuple[int, int, int]:
        return (int(operation.session_id), int(operation.epoch), int(operation.op_id))

    def lookup(self, operations: Sequence[OperationIdentity]) -> dict[str, Any] | None:
        records: list[dict[str, Any]] = []
        missing = 0
        for operation in operations:
            key = self._key(operation)
            record = self._records.get(key)
            if record is None:
                missing += 1
                continue
            recorded_digest, result = record
            if recorded_digest != operation.digest:
                raise invalid_descriptor(
                    f"operation {operation.op_id} conflicts with its committed digest"
                )
            self._records.move_to_end(key)
            records.append(result)
        if missing == len(operations):
            return None
        if missing:
            raise invalid_descriptor("execute batch mixes committed and uncommitted operations")
        first = records[0]
        if any(result != first for result in records[1:]):
            raise invalid_descriptor(
                "execute batch replay records do not share one terminal result"
            )
        return copy.deepcopy(first)

    def commit(
        self,
        operations: Sequence[OperationIdentity],
        result: dict[str, Any],
    ) -> None:
        for operation in operations:
            key = self._key(operation)
            self._records[key] = (str(operation.digest), copy.deepcopy(result))
            self._records.move_to_end(key)
        while len(self._records) > self.capacity:
            self._records.popitem(last=False)
