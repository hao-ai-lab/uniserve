"""Terminal operation records for deterministic at-least-once delivery."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import RLock

from ..batch import ExecutionResult, OperationEnvelope, OperationResult
from ..foundation.errors import invalid_descriptor


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    session_id: int
    epoch: int
    op_id: int
    digest: str
    step_id: int
    result: OperationResult


class ReplayStore:
    def __init__(self, capacity: int = 1024) -> None:
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self._records: OrderedDict[tuple[int, int, int], tuple[str, int, OperationResult]] = (
            OrderedDict()
        )
        self._lock = RLock()

    @staticmethod
    def _key(operation: OperationEnvelope) -> tuple[int, int, int]:
        return operation.session_id, operation.epoch, operation.op_id

    def lookup(
        self,
        operations: Sequence[OperationEnvelope],
    ) -> ExecutionResult | None:
        with self._lock:
            found: list[tuple[int, OperationResult]] = []
            missing = 0
            for operation in operations:
                record = self._records.get(self._key(operation))
                if record is None:
                    missing += 1
                    continue
                digest, step_id, result = record
                if digest != operation.digest:
                    raise invalid_descriptor(
                        f"operation {operation.op_id} conflicts with its committed digest"
                    )
                result.validate_for(operation)
                self._records.move_to_end(self._key(operation))
                found.append((step_id, result))
            if missing == len(operations):
                return None
            if missing:
                raise invalid_descriptor(
                    "execution batch mixes committed and uncommitted operations"
                )
            steps = {step_id for step_id, _result in found}
            if len(steps) != 1:
                raise invalid_descriptor(
                    "execution batch replay records belong to different submissions"
                )
            return ExecutionResult(
                step_id=next(iter(steps)),
                operations=tuple(result for _step, result in found),
            )

    def commit_atomic(
        self,
        operations: Sequence[OperationEnvelope],
        result: ExecutionResult,
        commit_state: Callable[[Callable[[], None]], None],
    ) -> None:
        result_operations = result.operations
        if len(result_operations) != len(operations):
            raise invalid_descriptor("terminal result does not align with its operations")
        result.validate_for(_batch_for_validation(result.step_id, operations))
        with self._lock:
            next_records = OrderedDict(self._records)
            for operation, operation_result in zip(operations, result_operations, strict=True):
                key = self._key(operation)
                existing = next_records.get(key)
                if existing is not None and existing[0] != operation.digest:
                    raise invalid_descriptor(
                        f"operation {operation.op_id} conflicts with its committed digest"
                    )
                next_records[key] = (
                    operation.digest,
                    result.step_id,
                    operation_result,
                )
                next_records.move_to_end(key)
            while len(next_records) > self.capacity:
                next_records.popitem(last=False)
            previous = self._records
            published = False

            def publish() -> None:
                nonlocal published
                published = True
                self._records = next_records

            try:
                commit_state(publish)
            except BaseException:
                if published:
                    self._records = previous
                raise

    def drop_session(self, session_id: int) -> None:
        with self._lock:
            keys = [key for key in self._records if key[0] == int(session_id)]
            for key in keys:
                del self._records[key]

    def rewrite_results(
        self,
        session_ids: set[int],
        mapper: Callable[[OperationResult], OperationResult],
    ) -> None:
        requested = {int(value) for value in session_ids}
        with self._lock:
            for key, (digest, step_id, result) in tuple(self._records.items()):
                if key[0] in requested:
                    mapped = mapper(result)
                    if (
                        mapped.session_id,
                        mapped.epoch,
                        mapped.op_id,
                        mapped.base_version,
                        mapped.result_version,
                    ) != (
                        result.session_id,
                        result.epoch,
                        result.op_id,
                        result.base_version,
                        result.result_version,
                    ):
                        raise invalid_descriptor("replay result rewrite changed operation identity")
                    self._records[key] = (digest, step_id, mapped)

    def snapshot_records(self, session_ids: set[int]) -> tuple[ReplayRecord, ...]:
        requested = {int(value) for value in session_ids}
        with self._lock:
            return tuple(
                ReplayRecord(
                    session_id=key[0],
                    epoch=key[1],
                    op_id=key[2],
                    digest=value[0],
                    step_id=value[1],
                    result=value[2],
                )
                for key, value in self._records.items()
                if key[0] in requested
            )

    def restore_records(
        self,
        session_ids: set[int],
        records: Sequence[ReplayRecord],
    ) -> None:
        requested = {int(value) for value in session_ids}
        staged: list[tuple[tuple[int, int, int], tuple[str, int, OperationResult]]] = []
        for record in records:
            key = (record.session_id, record.epoch, record.op_id)
            if record.session_id not in requested:
                raise invalid_descriptor("replay snapshot contains an undeclared session")
            if (
                record.result.session_id,
                record.result.epoch,
                record.result.op_id,
            ) != key:
                raise invalid_descriptor("replay snapshot result identity does not match its key")
            if record.step_id < 0 or len(record.digest) != 64:
                raise invalid_descriptor("replay snapshot record is incomplete")
            staged.append((key, (record.digest, record.step_id, record.result)))
        with self._lock:
            next_records = OrderedDict(
                (key, value) for key, value in self._records.items() if key[0] not in requested
            )
            for key, value in staged:
                if key in next_records:
                    raise invalid_descriptor("replay snapshot repeats an operation identity")
                next_records[key] = value
            while len(next_records) > self.capacity:
                next_records.popitem(last=False)
            self._records = next_records


def _batch_for_validation(
    step_id: int,
    operations: Sequence[OperationEnvelope],
):
    from ..batch import Batch

    return Batch(
        step_id=step_id,
        admissions=(),
        projections=(),
        operations=tuple(operations),
    )


__all__ = ["ReplayRecord", "ReplayStore"]
