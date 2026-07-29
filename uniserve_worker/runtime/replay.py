"""Terminal completion records for deterministic at-least-once delivery."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import RLock

from ..batch import CompletionRecord, CompletionReport, Operation, RegistrationAck
from ..foundation.errors import invalid_descriptor


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    session_id: int
    epoch: int
    op_id: int
    digest: str
    step_id: int
    result: CompletionRecord


class ReplayStore:
    def __init__(self, capacity: int = 1024) -> None:
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self._records: OrderedDict[tuple[int, int, int], tuple[str, int, CompletionRecord]] = (
            OrderedDict()
        )
        self._lock = RLock()

    @staticmethod
    def _key(operation: Operation) -> tuple[int, int, int]:
        key = operation.request_key
        return key.session_id, key.epoch, operation.op_id

    def lookup(
        self,
        operations: Sequence[Operation],
    ) -> CompletionReport | None:
        with self._lock:
            found: list[tuple[int, CompletionRecord]] = []
            missing = 0
            for operation in operations:
                record = self._records.get(self._key(operation))
                if record is None:
                    missing += 1
                    continue
                digest, step_id, result = record
                if digest != operation.plan_digest:
                    raise invalid_descriptor(
                        f"operation {operation.op_id} conflicts with its committed digest"
                    )
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
            return CompletionReport(
                step_id=next(iter(steps)),
                completions=tuple(result for _step, result in found),
                registration=RegistrationAck(visible=True),
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
            # Validation-only pass: no mutation of ``self._records``. ``pending``
            # mirrors the digest a key would carry after earlier operations in the
            # same batch, so an intra-batch digest conflict is detected exactly as
            # the previous copy-then-mutate loop did (which read from the growing
            # working copy). ``overwrites_existing`` records whether any key is
            # already committed -- an overwrite+move of a pre-existing entry, whose
            # exact position restoration on the error path is the one case the
            # cheap delta cannot reverse in place.
            pending: dict[tuple[int, int, int], str] = {}
            overwrites_existing = False
            for operation, completion in zip(operations, completions, strict=True):
                if (
                    completion.request_key != operation.request_key
                    or completion.op_id != operation.op_id
                ):
                    raise invalid_descriptor("terminal completion identity does not match operation")
                key = self._key(operation)
                if key in pending:
                    prior_digest: str | None = pending[key]
                elif key in self._records:
                    prior_digest = self._records[key][0]
                    overwrites_existing = True
                else:
                    prior_digest = None
                if prior_digest is not None and prior_digest != operation.plan_digest:
                    raise invalid_descriptor(
                        f"operation {operation.op_id} conflicts with its committed digest"
                    )
                pending[key] = operation.plan_digest

            if overwrites_existing:
                # Rare fallback (never reached on the decode/commit hot path, where
                # ``lookup`` has already deduplicated any committed operation): a
                # pre-existing key would be overwritten and moved to the end, and
                # restoring its exact prior position is intricate. Snapshot the
                # whole dict for this call only; correctness over micro-cost here.
                next_records = OrderedDict(self._records)
                for operation, completion in zip(operations, completions, strict=True):
                    key = self._key(operation)
                    next_records[key] = (operation.plan_digest, result.step_id, completion)
                    next_records.move_to_end(key)
                while len(next_records) > self.capacity:
                    next_records.popitem(last=False)
                previous = self._records
                published = False

                def publish_snapshot() -> None:
                    nonlocal published
                    published = True
                    self._records = next_records

                try:
                    commit_state(publish_snapshot)
                except BaseException:
                    if published:
                        self._records = previous
                    raise
                return

            # Hot path: every key is new. Apply additions and LRU evictions in
            # place inside ``publish`` -- no full copy -- recording only the delta
            # needed to undo them if a later commit step raises after publishing.
            added_keys: list[tuple[int, int, int]] = []
            evicted: list[tuple[tuple[int, int, int], tuple[str, int, CompletionRecord]]] = []
            published = False

            def publish() -> None:
                nonlocal published
                published = True
                records = self._records
                for operation, completion in zip(operations, completions, strict=True):
                    key = self._key(operation)
                    value = (operation.plan_digest, result.step_id, completion)
                    if key in records:
                        # A key repeated within this batch: it was appended by an
                        # earlier operation above, so it is a self-add. Update the
                        # value and re-append; the undo removes it regardless.
                        records[key] = value
                        records.move_to_end(key)
                    else:
                        records[key] = value
                        added_keys.append(key)
                while len(records) > self.capacity:
                    evicted.append(records.popitem(last=False))

            try:
                commit_state(publish)
            except BaseException:
                if published:
                    # Undo exactly: drop every newly-added key (some may already
                    # have been evicted -- pop tolerates that), then re-insert the
                    # evicted entries at the FRONT in reverse-eviction order so the
                    # original front ordering is reproduced. Evicted entries that
                    # were themselves self-adds must not reappear.
                    added_set = set(added_keys)
                    records = self._records
                    for key in added_keys:
                        records.pop(key, None)
                    for key, value in reversed(evicted):
                        if key in added_set:
                            continue
                        records[key] = value
                        records.move_to_end(key, last=False)
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
            for key, (digest, step_id, result) in tuple(self._records.items()):
                if key[0] in requested:
                    mapped = mapper(result)
                    if (mapped.request_key, mapped.op_id) != (result.request_key, result.op_id):
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
        staged: list[tuple[tuple[int, int, int], tuple[str, int, CompletionRecord]]] = []
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


__all__ = ["ReplayRecord", "ReplayStore"]
