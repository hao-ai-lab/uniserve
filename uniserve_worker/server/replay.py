"""Atomic in-flight joins and immutable completed execution replay."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Final

from ..batch import (
    Admission,
    Batch,
    CompletionReport,
    Control,
    Operation,
    PartitionCompletion,
)
from ..foundation.errors import invalid_descriptor, resource_error
from .completion import finalize_completion_report, partition_completion_ready

_OperationKey = tuple[int, int, int]


def _operation_key(operation: Operation) -> _OperationKey:
    request = operation.request_key
    return int(request.session_id), int(request.epoch), int(operation.op_id)


@dataclass(frozen=True, slots=True)
class _PartitionIdentity:
    partition_id: int
    submission_group: int
    collective_seq: int
    domain: object
    route: int
    execution: object
    attention: object
    shape_class: int
    operations: tuple[Operation, ...]


@dataclass(frozen=True, slots=True)
class _SubmissionIdentity:
    step_id: int
    admissions: tuple[Admission, ...]
    controls: tuple[Control, ...]
    partitions: tuple[_PartitionIdentity, ...]


def _submission_identity(batch: Batch) -> _SubmissionIdentity:
    return _SubmissionIdentity(
        step_id=int(batch.step_id),
        admissions=batch.admissions,
        controls=batch.controls,
        partitions=tuple(
            _PartitionIdentity(
                partition_id=int(partition.partition_id),
                submission_group=int(partition.submission_group),
                collective_seq=int(partition.collective_seq),
                domain=partition.domain,
                route=int(partition.route),
                execution=partition.execution,
                attention=partition.attention,
                shape_class=int(partition.shape_class),
                operations=partition.operations,
            )
            for partition in batch.partitions
        ),
    )


def _report_partition_order(batch: Batch) -> tuple[int, ...]:
    return tuple(int(partition.partition_id) for partition in batch.partitions)


def _report_operation_keys(batch: Batch) -> tuple[tuple[_OperationKey, ...], ...]:
    return tuple(
        tuple(_operation_key(operation) for operation in partition.operations)
        for partition in batch.partitions
    )


def _validate_report_shape(
    report: CompletionReport,
    *,
    step_id: int,
    partition_order: tuple[int, ...],
    partition_operation_keys: tuple[tuple[_OperationKey, ...], ...],
) -> None:
    if int(report.step_id) != step_id:
        raise invalid_descriptor("terminal report step identity does not match its submission")
    actual_partition_order = tuple(int(partition.partition_id) for partition in report.partitions)
    if actual_partition_order != partition_order:
        raise invalid_descriptor("terminal report partition identity does not match its submission")
    for partition, expected_keys in zip(
        report.partitions,
        partition_operation_keys,
        strict=True,
    ):
        actual_keys = tuple(
            (
                int(completion.request_key.session_id),
                int(completion.request_key.epoch),
                int(completion.op_id),
            )
            for completion in partition.completions
        )
        if actual_keys != expected_keys:
            raise invalid_descriptor("terminal report operations do not align with their partition")
        expected = set(expected_keys)
        for product in partition.products:
            reference = product.product
            product_key = (
                int(reference.request_key.session_id),
                int(reference.request_key.epoch),
                int(reference.producer_op_id),
            )
            if product_key not in expected:
                raise invalid_descriptor(
                    "terminal product does not belong to its completion partition"
                )


def _materialize_partition(step_id: int, partition: PartitionCompletion) -> PartitionCompletion:
    if not partition_completion_ready(partition):
        raise RuntimeError("completion partition was materialized before query readiness")
    finalized = finalize_completion_report(
        CompletionReport(step_id=step_id, partitions=(partition,))
    )
    if len(finalized.partitions) != 1:
        raise RuntimeError("completion materialization changed partition cardinality")
    # The protocol decoder is the canonical host-data boundary. Deferred values,
    # device objects, events, leases, and callbacks cannot cross this conversion.
    host = CompletionReport.from_wire(finalized.to_wire())
    if len(host.partitions) != 1:
        raise RuntimeError("host completion conversion changed partition cardinality")
    materialized = host.partitions[0]
    if any(type(product.payload) is not bytes for product in materialized.products):
        raise RuntimeError("materialized completion contains a non-byte product payload")
    return materialized


@dataclass(frozen=True, slots=True)
class _CompletedSubmission:
    token: int
    identity: _SubmissionIdentity
    operation_keys: tuple[_OperationKey, ...]
    operation_digests: tuple[str, ...]
    partition_order: tuple[int, ...]
    report: CompletionReport

    @property
    def complete(self) -> bool:
        return True

    def advance(self) -> None:
        return None

    def materialized_partitions(self) -> tuple[PartitionCompletion, ...]:
        return self.report.partitions


class _InFlightSubmission:
    __slots__ = (
        "token",
        "identity",
        "operation_keys",
        "operation_digests",
        "partition_order",
        "partition_operation_keys",
        "_coordinator",
        "_source",
        "_raw_report",
        "_materialized",
        "_completed",
        "_failure",
        "_lock",
    )

    def __init__(
        self,
        *,
        token: int,
        identity: _SubmissionIdentity,
        operation_keys: tuple[_OperationKey, ...],
        operation_digests: tuple[str, ...],
        partition_order: tuple[int, ...],
        partition_operation_keys: tuple[tuple[_OperationKey, ...], ...],
        coordinator: ReplayCoordinator,
    ) -> None:
        self.token = int(token)
        self.identity = identity
        self.operation_keys = operation_keys
        self.operation_digests = operation_digests
        self.partition_order = partition_order
        self.partition_operation_keys = partition_operation_keys
        self._coordinator = coordinator
        self._source: object | None = None
        self._raw_report: CompletionReport | None = None
        self._materialized: dict[int, PartitionCompletion] = {}
        self._completed: _CompletedSubmission | None = None
        self._failure: BaseException | None = None
        self._lock = RLock()

    @property
    def complete(self) -> bool:
        return self._completed is not None

    @property
    def source(self) -> object | None:
        return self._source

    def attach(self, source: object) -> None:
        with self._lock:
            if self._failure is not None:
                raise self._failure
            if self._source is not None:
                raise RuntimeError("in-flight submission already has an execution source")
            if not isinstance(source, CompletionReport):
                ready = getattr(source, "ready", None)
                resolve = getattr(source, "resolve", None)
                if not callable(ready) or not callable(resolve):
                    raise invalid_descriptor(
                        "in-flight execution source has no readiness and resolution contract"
                    )
            self._source = source

    def fail(self, error: BaseException) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = error

    def advance(self) -> None:
        with self._lock:
            if self._failure is not None:
                raise self._failure
            if self._completed is not None:
                return
            source = self._source
            if source is None:
                return
            try:
                report = self._raw_report
                if report is None:
                    if isinstance(source, CompletionReport):
                        report = source
                    else:
                        ready = getattr(source, "ready")
                        if not bool(ready()):
                            return
                        result = getattr(source, "resolve")()
                        if not isinstance(result, CompletionReport):
                            raise RuntimeError(
                                "in-flight execution resolved to an invalid completion report"
                            )
                        report = result
                    _validate_report_shape(
                        report,
                        step_id=self.identity.step_id,
                        partition_order=self.partition_order,
                        partition_operation_keys=self.partition_operation_keys,
                    )
                    self._raw_report = report

                for partition in report.partitions:
                    partition_id = int(partition.partition_id)
                    if partition_id in self._materialized:
                        continue
                    if partition_completion_ready(partition):
                        self._materialized[partition_id] = _materialize_partition(
                            self.identity.step_id,
                            partition,
                        )

                if len(self._materialized) != len(self.partition_order):
                    return
                ordered = tuple(
                    self._materialized[partition_id] for partition_id in self.partition_order
                )
                host_report = CompletionReport(
                    step_id=self.identity.step_id,
                    partitions=ordered,
                )
                _validate_report_shape(
                    host_report,
                    step_id=self.identity.step_id,
                    partition_order=self.partition_order,
                    partition_operation_keys=self.partition_operation_keys,
                )
                self._completed = self._coordinator._publish(self, host_report)
                self._raw_report = None
                self._source = None
            except BaseException as error:
                self._failure = error
                self._coordinator._abort_submission(self)
                raise

    def materialized_partitions(self) -> tuple[PartitionCompletion, ...]:
        with self._lock:
            if self._completed is not None:
                return self._completed.report.partitions
            return tuple(
                self._materialized[partition_id]
                for partition_id in self.partition_order
                if partition_id in self._materialized
            )

    def current(self) -> _InFlightSubmission | _CompletedSubmission:
        with self._lock:
            return self if self._completed is None else self._completed


class CompletionDelivery:
    """One response cursor over a shared execution submission."""

    __slots__ = ("_submission", "_sent_partitions")

    def __init__(self, submission: _InFlightSubmission | _CompletedSubmission) -> None:
        self._submission = submission
        self._sent_partitions: set[int] = set()

    @property
    def submission_token(self) -> int:
        return int(self._submission.token)

    @property
    def step_id(self) -> int:
        return int(self._submission.identity.step_id)

    @property
    def session_ids(self) -> frozenset[int]:
        return frozenset(key[0] for key in self._submission.operation_keys)

    @property
    def source(self) -> object | None:
        submission = self._submission
        return submission.source if isinstance(submission, _InFlightSubmission) else None

    def _current(self) -> _InFlightSubmission | _CompletedSubmission:
        submission = self._submission
        if isinstance(submission, _InFlightSubmission):
            submission.advance()
            current: _InFlightSubmission | _CompletedSubmission = submission.current()
            if current is not submission:
                self._submission = current
            return current
        return submission

    def ready(self) -> bool:
        current = self._current()
        return any(
            int(partition.partition_id) not in self._sent_partitions
            for partition in current.materialized_partitions()
        )

    def take_ready(self) -> CompletionReport:
        current = self._current()
        partitions = tuple(
            partition
            for partition in current.materialized_partitions()
            if int(partition.partition_id) not in self._sent_partitions
        )
        if not partitions:
            raise RuntimeError("completion delivery has no query-ready partition")
        self._sent_partitions.update(int(partition.partition_id) for partition in partitions)
        return CompletionReport(step_id=self.step_id, partitions=partitions)

    def pending(self) -> bool:
        current = self._current()
        if not current.complete:
            return True
        return any(
            int(partition.partition_id) not in self._sent_partitions
            for partition in current.materialized_partitions()
        )


@dataclass(frozen=True, slots=True)
class ReplayRegistration:
    delivery: CompletionDelivery
    execute: bool
    outcome: str
    _submission: _InFlightSubmission | None


class ReplayCoordinator:
    """Own operation registration, shared completion progress, and host replay."""

    _EXECUTE: Final[str] = "execute"
    _INFLIGHT: Final[str] = "inflight_join"
    _COMPLETED: Final[str] = "completed_replay"

    def __init__(self, *, in_flight_capacity: int, completed_capacity: int) -> None:
        if int(in_flight_capacity) < 1:
            raise ValueError("in-flight replay capacity must be positive")
        if int(completed_capacity) < 1:
            raise ValueError("completed replay capacity must be positive")
        self.in_flight_capacity = int(in_flight_capacity)
        self.completed_capacity = int(completed_capacity)
        self._in_flight: dict[_OperationKey, _InFlightSubmission] = {}
        self._in_flight_steps: dict[int, _InFlightSubmission] = {}
        self._completed_by_operation: dict[_OperationKey, _CompletedSubmission] = {}
        self._completed_submissions: OrderedDict[int, _CompletedSubmission] = OrderedDict()
        self._completed_operations = 0
        self._ended_epochs: set[tuple[int, int]] = set()
        self._next_token = 1
        self._lock = RLock()

    @property
    def in_flight_operations(self) -> int:
        with self._lock:
            return len(self._in_flight)

    @property
    def completed_operations(self) -> int:
        with self._lock:
            return self._completed_operations

    def register(self, batch: Batch) -> ReplayRegistration:
        operations = batch.operations
        if not operations:
            raise invalid_descriptor("replay registration requires at least one operation")
        keys = tuple(_operation_key(operation) for operation in operations)
        digests = tuple(str(operation.plan_digest) for operation in operations)
        identity = _submission_identity(batch)
        with self._lock:
            in_flight: list[_InFlightSubmission] = []
            completed: list[_CompletedSubmission] = []
            missing = 0
            for key, digest in zip(keys, digests, strict=True):
                pending = self._in_flight.get(key)
                if pending is not None:
                    expected = pending.operation_digests[pending.operation_keys.index(key)]
                    if expected != digest:
                        raise invalid_descriptor(
                            f"operation {key[2]} conflicts with its registered plan digest"
                        )
                    in_flight.append(pending)
                    continue
                terminal = self._completed_by_operation.get(key)
                if terminal is not None:
                    expected = terminal.operation_digests[terminal.operation_keys.index(key)]
                    if expected != digest:
                        raise invalid_descriptor(
                            f"operation {key[2]} conflicts with its completed plan digest"
                        )
                    completed.append(terminal)
                    continue
                missing += 1

            if missing == len(keys):
                if len(keys) > self.completed_capacity:
                    raise resource_error(
                        "completed replay capacity cannot hold this atomic submission"
                    )
                if len(self._in_flight) + len(keys) > self.in_flight_capacity:
                    raise resource_error("in-flight duplicate table capacity is exhausted")
                existing_step = self._in_flight_steps.get(int(batch.step_id))
                if existing_step is not None:
                    raise invalid_descriptor(
                        f"execution step {batch.step_id} already names an in-flight submission"
                    )
                token = self._next_token
                self._next_token += 1
                submission = _InFlightSubmission(
                    token=token,
                    identity=identity,
                    operation_keys=keys,
                    operation_digests=digests,
                    partition_order=_report_partition_order(batch),
                    partition_operation_keys=_report_operation_keys(batch),
                    coordinator=self,
                )
                for key in keys:
                    self._in_flight[key] = submission
                self._in_flight_steps[int(batch.step_id)] = submission
                return ReplayRegistration(
                    delivery=CompletionDelivery(submission),
                    execute=True,
                    outcome=self._EXECUTE,
                    _submission=submission,
                )

            if missing:
                raise invalid_descriptor(
                    "execution batch mixes registered and unregistered operation identities"
                )
            if in_flight and completed:
                raise invalid_descriptor(
                    "execution batch mixes in-flight and completed operation identities"
                )
            if in_flight:
                submission = in_flight[0]
                if any(candidate is not submission for candidate in in_flight):
                    raise invalid_descriptor(
                        "execution batch joins operations from different in-flight submissions"
                    )
                if submission.identity != identity or submission.operation_keys != keys:
                    raise invalid_descriptor(
                        "execution batch conflicts with its registered submission identity"
                    )
                return ReplayRegistration(
                    delivery=CompletionDelivery(submission),
                    execute=False,
                    outcome=self._INFLIGHT,
                    _submission=None,
                )
            if completed:
                completed_submission = completed[0]
                if any(candidate is not completed_submission for candidate in completed):
                    raise invalid_descriptor(
                        "execution batch joins operations from different completed submissions"
                    )
                if (
                    completed_submission.identity != identity
                    or completed_submission.operation_keys != keys
                ):
                    raise invalid_descriptor(
                        "execution batch conflicts with its completed submission identity"
                    )
                self._completed_submissions.move_to_end(completed_submission.token)
                return ReplayRegistration(
                    delivery=CompletionDelivery(completed_submission),
                    execute=False,
                    outcome=self._COMPLETED,
                    _submission=None,
                )
            raise RuntimeError("replay registration reached an incomplete state")

    def attach(self, registration: ReplayRegistration, source: object) -> None:
        submission = registration._submission
        if not registration.execute or submission is None:
            raise RuntimeError("only a newly registered submission accepts an execution source")
        submission.attach(source)

    def abort(self, registration: ReplayRegistration, error: BaseException) -> None:
        submission = registration._submission
        if submission is None:
            return
        submission.fail(error)
        self._abort_submission(submission)

    def _abort_submission(self, submission: _InFlightSubmission) -> None:
        with self._lock:
            for key in submission.operation_keys:
                if self._in_flight.get(key) is submission:
                    del self._in_flight[key]
            if self._in_flight_steps.get(submission.identity.step_id) is submission:
                del self._in_flight_steps[submission.identity.step_id]

    def _publish(
        self,
        submission: _InFlightSubmission,
        report: CompletionReport,
    ) -> _CompletedSubmission:
        with self._lock:
            if any(self._in_flight.get(key) is not submission for key in submission.operation_keys):
                raise RuntimeError("in-flight submission lost an operation registration")
            completed = _CompletedSubmission(
                token=submission.token,
                identity=submission.identity,
                operation_keys=submission.operation_keys,
                operation_digests=submission.operation_digests,
                partition_order=submission.partition_order,
                report=report,
            )
            next_submissions = OrderedDict(self._completed_submissions)
            next_by_operation = dict(self._completed_by_operation)
            next_count = self._completed_operations
            next_submissions[completed.token] = completed
            next_count += len(completed.operation_keys)
            for key in completed.operation_keys:
                next_by_operation[key] = completed
            while next_count > self.completed_capacity:
                _token, evicted = next_submissions.popitem(last=False)
                next_count -= len(evicted.operation_keys)
                for key in evicted.operation_keys:
                    if next_by_operation.get(key) is evicted:
                        del next_by_operation[key]
            for key in submission.operation_keys:
                del self._in_flight[key]
            if self._in_flight_steps.get(submission.identity.step_id) is submission:
                del self._in_flight_steps[submission.identity.step_id]
            self._completed_submissions = next_submissions
            self._completed_by_operation = next_by_operation
            self._completed_operations = next_count
            self._retain_referenced_ended_epochs()
            return completed

    def ensure_session_idle(self, session_id: int) -> None:
        target = int(session_id)
        with self._lock:
            if any(key[0] == target for key in self._in_flight):
                raise resource_error(
                    f"session {target} still has an in-flight completion submission"
                )

    def drop_session(self, session_id: int) -> None:
        target = int(session_id)
        self.ensure_session_idle(target)
        with self._lock:
            self._ended_epochs.update(
                (key[0], key[1])
                for submission in self._completed_submissions.values()
                for key in submission.operation_keys
                if key[0] == target
            )
            tokens = tuple(
                token
                for token, submission in self._completed_submissions.items()
                if all(
                    (key[0], key[1]) in self._ended_epochs
                    for key in submission.operation_keys
                )
            )
            for token in tokens:
                submission = self._completed_submissions.pop(token)
                self._completed_operations -= len(submission.operation_keys)
                for key in submission.operation_keys:
                    if self._completed_by_operation.get(key) is submission:
                        del self._completed_by_operation[key]
            self._retain_referenced_ended_epochs()

    def _retain_referenced_ended_epochs(self) -> None:
        referenced = {
            (key[0], key[1])
            for submission in self._completed_submissions.values()
            for key in submission.operation_keys
        }
        self._ended_epochs.intersection_update(referenced)


__all__ = [
    "CompletionDelivery",
    "ReplayCoordinator",
    "ReplayRegistration",
]
