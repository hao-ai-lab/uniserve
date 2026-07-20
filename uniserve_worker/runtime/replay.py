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
        self._records: OrderedDict[tuple[int, int, int], tuple[str, int, Any]] = OrderedDict()

    @staticmethod
    def _key(operation: OperationIdentity) -> tuple[int, int, int]:
        return (int(operation.session_id), int(operation.epoch), int(operation.op_id))

    @staticmethod
    def _clone_result(result: Any) -> Any:
        if isinstance(result, dict):
            return copy.deepcopy(result)
        return result

    def lookup(
        self,
        operations: Sequence[OperationIdentity],
        *,
        step_id: int,
    ) -> dict[str, Any] | None:
        records: list[Any] = []
        recorded_steps: list[int] = []
        missing = 0
        for operation in operations:
            key = self._key(operation)
            record = self._records.get(key)
            if record is None:
                missing += 1
                continue
            recorded_digest, recorded_step, result = record
            if recorded_digest != operation.digest:
                raise invalid_descriptor(
                    f"operation {operation.op_id} conflicts with its committed digest"
                )
            self._records.move_to_end(key)
            recorded_steps.append(recorded_step)
            records.append(result)
        if missing == len(operations):
            return None
        if missing:
            raise invalid_descriptor("execute batch mixes committed and uncommitted operations")
        del step_id
        if len(set(recorded_steps)) != 1:
            raise invalid_descriptor("execute batch replay records come from different submissions")
        return {
            "step_id": recorded_steps[0],
            "per_seq": [self._clone_result(result) for result in records],
        }

    def commit(
        self,
        operations: Sequence[OperationIdentity],
        result: dict[str, Any],
    ) -> None:
        per_seq = result.get("per_seq")
        if not isinstance(per_seq, list) or len(per_seq) != len(operations):
            raise invalid_descriptor("terminal result does not align with its operations")
        for operation, sequence_result in zip(operations, per_seq, strict=True):
            key = self._key(operation)
            self._records[key] = (
                str(operation.digest),
                int(result["step_id"]),
                self._clone_result(sequence_result),
            )
            self._records.move_to_end(key)
        while len(self._records) > self.capacity:
            self._records.popitem(last=False)
