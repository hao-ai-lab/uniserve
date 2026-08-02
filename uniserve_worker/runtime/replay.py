"""Terminal completion records for deterministic at-least-once delivery."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import RLock

from ..batch import (
    BatchPartition,
    CompletionRecord,
    CompletionReport,
    Operation,
    PartitionCompletion,
    RegistrationAck,
)
from ..foundation.errors import invalid_descriptor


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    session_id: int
    epoch: int
    op_id: int
    digest: str
    step_id: int
    result: CompletionRecord
    registration_visible: bool


@dataclass(frozen=True, slots=True)
class _StoredReplay:
    digest: str
    step_id: int
    result: CompletionRecord
    registration_visible: bool


class ReplayStore:
    def __init__(self, capacity: int = 1024) -> None:
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self._records: OrderedDict[tuple[int, int, int], _StoredReplay] = OrderedDict()
        self._lock = RLock()

    @staticmethod
    def _key(operation: Operation) -> tuple[int, int, int]:
        key = operation.request_key
        return key.session_id, key.epoch, operation.op_id

    def lookup(
        self,
        partitions: Sequence[BatchPartition],
    ) -> CompletionReport | None:
        operations = tuple(
            operation for partition in partitions for operation in partition.operations
        )
        with self._lock:
            found: list[_StoredReplay] = []
            missing = 0
            for operation in operations:
                record = self._records.get(self._key(operation))
                if record is None:
                    missing += 1
                    continue
                if record.digest != operation.plan_digest:
                    raise invalid_descriptor(
                        f"operation {operation.op_id} conflicts with its committed digest"
                    )
                self._records.move_to_end(self._key(operation))
                found.append(record)
            if missing == len(operations):
                return None
            if missing:
                raise invalid_descriptor(
                    "execution batch mixes committed and uncommitted operations"
                )
            steps = {record.step_id for record in found}
            if len(steps) != 1:
                raise invalid_descriptor(
                    "execution batch replay records belong to different submissions"
                )
            by_identity = {
                (record.result.request_key, int(record.result.op_id)): record
                for record in found
            }
            return CompletionReport(
                step_id=next(iter(steps)),
                partitions=tuple(
                    PartitionCompletion(
                        partition_id=partition.partition_id,
                        completions=tuple(
                            by_identity[(operation.request_key, int(operation.op_id))].result
                            for operation in partition.operations
                        ),
                        registration=RegistrationAck(
                            visible=all(
                                by_identity[
                                    (operation.request_key, int(operation.op_id))
                                ].registration_visible
                                for operation in partition.operations
                            )
                        ),
                    )
                    for partition in partitions
                ),
            )

    def commit_atomic(
        self,
        operations: Sequence[Operation],
        result: CompletionReport,
        commit_state: Callable[[Callable[[], None]], None],
    ) -> None:
        completions = result.completions
        if len(completions) != len(operations):
            raise invalid_descriptor("terminal report does not align with its operations")
        with self._lock:
            next_records = OrderedDict(self._records)
            for operation, completion in zip(operations, completions, strict=True):
                if (
                    completion.request_key != operation.request_key
                    or completion.op_id != operation.op_id
                ):
                    raise invalid_descriptor("terminal completion identity does not match operation")
                key = self._key(operation)
                existing = next_records.get(key)
                if existing is not None and existing.digest != operation.plan_digest:
                    raise invalid_descriptor(
                        f"operation {operation.op_id} conflicts with its committed digest"
                    )
                partition = next(
                    (
                        partition
                        for partition in result.partitions
                        if completion in partition.completions
                    ),
                    None,
                )
                if partition is None:
                    raise invalid_descriptor(
                        "terminal completion has no partition registration result"
                    )
                next_records[key] = _StoredReplay(
                    digest=operation.plan_digest,
                    step_id=result.step_id,
                    result=completion,
                    registration_visible=partition.registration.visible,
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
        mapper: Callable[[CompletionRecord], CompletionRecord],
    ) -> None:
        requested = {int(value) for value in session_ids}
        with self._lock:
            for key, record in tuple(self._records.items()):
                if key[0] in requested:
                    mapped = mapper(record.result)
                    if (mapped.request_key, mapped.op_id) != (
                        record.result.request_key,
                        record.result.op_id,
                    ):
                        raise invalid_descriptor("replay result rewrite changed operation identity")
                    self._records[key] = _StoredReplay(
                        digest=record.digest,
                        step_id=record.step_id,
                        result=mapped,
                        registration_visible=record.registration_visible,
                    )

    def snapshot_records(self, session_ids: set[int]) -> tuple[ReplayRecord, ...]:
        requested = {int(value) for value in session_ids}
        with self._lock:
            return tuple(
                ReplayRecord(
                    session_id=key[0],
                    epoch=key[1],
                    op_id=key[2],
                    digest=value.digest,
                    step_id=value.step_id,
                    result=value.result,
                    registration_visible=value.registration_visible,
                )
                for key, value in self._records.items()
                if key[0] in requested and _snapshot_ready(value.result)
            )

    def restore_records(
        self,
        session_ids: set[int],
        records: Sequence[ReplayRecord],
    ) -> None:
        requested = {int(value) for value in session_ids}
        staged: list[tuple[tuple[int, int, int], _StoredReplay]] = []
        for record in records:
            key = (record.session_id, record.epoch, record.op_id)
            if record.session_id not in requested:
                raise invalid_descriptor("replay snapshot contains an undeclared session")
            if (
                record.result.request_key.session_id,
                record.result.request_key.epoch,
                record.result.op_id,
            ) != key:
                raise invalid_descriptor("replay snapshot result identity does not match its key")
            if record.step_id < 0 or len(record.digest) != 64:
                raise invalid_descriptor("replay snapshot record is incomplete")
            staged.append(
                (
                    key,
                    _StoredReplay(
                        digest=record.digest,
                        step_id=record.step_id,
                        result=record.result,
                        registration_visible=record.registration_visible,
                    ),
                )
            )
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


def _snapshot_ready(record: CompletionRecord) -> bool:
    lengths = record.logical_lengths
    span = record.token_span
    timing = record.timing_counters
    scalar_values = (
        record.selected_point,
        lengths.token_len,
        lengths.kv_visible_len,
        lengths.latent_len,
        lengths.kv_reserved_len,
        lengths.kv_initialized_len,
        lengths.kv_committed_len,
        lengths.kv_published_len,
        span.base,
        span.len,
        timing.queued_us,
        timing.device_us,
        timing.copy_us,
        timing.host_us,
    )
    return (
        type(record.semantic_digest) is str
        and all(type(value) is int for value in scalar_values)
        and all(type(value) is int for value in record.committed_tokens)
        and all(type(value) is int for value in record.product_generations)
    )


__all__ = ["ReplayRecord", "ReplayStore"]
