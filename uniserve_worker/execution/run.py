"""Execution-run ownership and partial-result readers."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable

from ..foundation.errors import WorkerError, classify, invalid_descriptor
from .batch import Batch, CompletionReport, PartitionCompletion
from .output import _completion_payload_ready, _record_ready, finalize_completion_report
from .rows import PreparedExecution

_OperationKey = tuple[int, int, int]
_EpochKey = tuple[int, int]

__all__ = ["BatchReader", "BatchRun", "PartitionRun", "ReplayWindow"]


def _operation_key(operation: object) -> _OperationKey:
    request = getattr(operation, "request_key")
    return (int(request.session_id), int(request.epoch), int(getattr(operation, "op_id")))


def _batch_lineage(batch: Batch) -> tuple[frozenset[int], frozenset[_EpochKey]]:
    request_keys = (
        *(admission.request_key for admission in batch.admissions),
        *(operation.request_key for operation in batch.operations),
        *(control.request_key for control in batch.controls),
    )
    epochs = frozenset((int(key.session_id), int(key.epoch)) for key in request_keys)
    return frozenset(session for session, _epoch in epochs), epochs


def _validate_report(run: BatchRun, report: CompletionReport) -> None:
    if int(report.step_id) != run.step_id:
        raise invalid_descriptor("terminal report step identity does not match its submission")
    if tuple(int(partition.partition_id) for partition in report.partitions) != run.partition_order:
        raise invalid_descriptor("terminal report partition identity does not match its submission")
    for partition, expected_keys in zip(
        report.partitions, run.partition_operation_keys, strict=True
    ):
        actual_keys = tuple(
            (
                int(record.request_key.session_id),
                int(record.request_key.epoch),
                int(record.op_id),
            )
            for record in partition.completions
        )
        if actual_keys != expected_keys:
            raise invalid_descriptor("terminal report operations do not align with their partition")
        expected = set(expected_keys)
        for product in partition.products:
            reference = product.product
            key = (
                int(reference.request_key.session_id),
                int(reference.request_key.epoch),
                int(reference.producer_op_id),
            )
            if key not in expected:
                raise invalid_descriptor("terminal product does not belong to its completion partition")


class PartitionRun:
    """One partition from device launch through immutable host publication."""

    __slots__ = (
        "partition_id",
        "state",
        "result",
        "_raw",
        "_record_cursor",
        "_product_cursor",
    )

    def __init__(self, partition_id: int) -> None:
        self.partition_id = int(partition_id)
        self.state = "PREPARED"
        self.result: PartitionCompletion | None = None
        self._raw: PartitionCompletion | None = None
        self._record_cursor = 0
        self._product_cursor = 0

    def launch(self, partition: PartitionCompletion) -> None:
        if self.state != "PREPARED" or self._raw is not None:
            raise RuntimeError("partition execution source was bound more than once")
        if int(partition.partition_id) != self.partition_id:
            raise RuntimeError("partition execution source has the wrong identity")
        self._raw = partition
        self.state = "LAUNCHED"

    def device_ready(self) -> bool:
        if self.result is not None:
            return True
        raw = self._raw
        if raw is None:
            return False
        while self._record_cursor < len(raw.completions) and _record_ready(
            raw.completions[self._record_cursor]
        ):
            self._record_cursor += 1
        while self._product_cursor < len(raw.products) and _completion_payload_ready(
            raw.products[self._product_cursor].payload
        ):
            self._product_cursor += 1
        return self._record_cursor == len(raw.completions) and self._product_cursor == len(
            raw.products
        )

    @property
    def successors_ready(self) -> bool:
        raw = self._raw
        publication = None if raw is None else raw.publication
        return publication is not None and publication.successors_ready

    def finish(self, step_id: int) -> bool:
        if self.result is not None:
            return True
        if not self.device_ready():
            return False
        raw = self._raw
        if raw is None:
            raise RuntimeError("ready partition lost its execution source")
        report = finalize_completion_report(
            CompletionReport(step_id=int(step_id), partitions=(raw,))
        )
        if len(report.partitions) != 1:
            raise RuntimeError("completion materialization changed partition cardinality")
        result = report.partitions[0]
        if any(
            type(token) is not int
            for record in result.completions
            for token in record.committed_tokens
        ):
            raise RuntimeError("materialized completion carries an unresolved committed token")
        if any(type(product.payload) is not bytes for product in result.products):
            raise RuntimeError("materialized completion contains a non-byte product payload")
        self.result = result
        self._raw = None
        self.state = "FINISHED"
        return True

    def abort(self) -> None:
        if self.result is None:
            raw = self._raw
            publication = None if raw is None else raw.publication
            if publication is not None:
                publication.cancel()
            self._raw = None
            self.state = "ABORTED"


class BatchRun:
    """One exact-once batch lifecycle retained through replay eviction."""

    __slots__ = (
        "step_id",
        "batch",
        "session_ids",
        "epochs",
        "partition_order",
        "partition_operation_keys",
        "partitions",
        "weight",
        "state",
        "error",
        "report",
        "active_readers",
        "_source",
        "_on_successors_ready",
        "_on_ready",
        "_on_terminal",
        "_successors_notified",
        "_terminal_notified",
    )

    def __init__(
        self,
        batch: Batch,
        *,
        on_successors_ready: Callable[[BatchRun], None],
        on_ready: Callable[[BatchRun], None],
        on_terminal: Callable[[BatchRun], None],
    ) -> None:
        sessions, epochs = _batch_lineage(batch)
        self.step_id = int(batch.step_id)
        self.batch = batch
        self.session_ids = sessions
        self.epochs = epochs
        self.partition_order = tuple(int(partition.partition_id) for partition in batch.partitions)
        self.partition_operation_keys = tuple(
            tuple(_operation_key(operation) for operation in partition.operations)
            for partition in batch.partitions
        )
        self.partitions = tuple(PartitionRun(value) for value in self.partition_order)
        self.weight = max(1, len(batch.operations))
        self.state = "QUEUED"
        self.error: WorkerError | None = None
        self.report: CompletionReport | None = None
        self.active_readers = 0
        self._source: CompletionReport | PreparedExecution | None = None
        self._on_successors_ready = on_successors_ready
        self._on_ready = on_ready
        self._on_terminal = on_terminal
        self._successors_notified = False
        self._terminal_notified = False

    @property
    def complete(self) -> bool:
        return self.state == "TERMINAL"

    @property
    def source(self) -> CompletionReport | PreparedExecution | None:
        return self._source

    def attach(self, source: CompletionReport | PreparedExecution) -> None:
        if self.state != "QUEUED" or self._source is not None:
            raise RuntimeError("batch run already has an execution source")
        self._source = source
        self.state = "RUNNING"

    def fail(self, error: BaseException, *, context: str = "execute") -> None:
        if self.complete:
            return
        source = self._source
        self.error = (
            error
            if isinstance(error, WorkerError)
            else source.record_failure(error)
            if isinstance(source, PreparedExecution)
            else classify(error, context=context)
        )
        for partition in self.partitions:
            partition.abort()
        self._source = None
        self.state = "TERMINAL"
        self._notify_terminal()

    def advance_execution(self) -> bool:
        if self.complete or (
            self.partitions
            and all(partition.state != "PREPARED" for partition in self.partitions)
        ):
            return True
        source = self._source
        if source is None:
            return False
        try:
            if isinstance(source, CompletionReport):
                report = source
            else:
                if not source.ready():
                    return False
                report = source.resolve()
            _validate_report(self, report)
            for partition, raw in zip(self.partitions, report.partitions, strict=True):
                partition.launch(raw)
            self._source = None
            self.state = "FINISHING"
            if all(partition.successors_ready for partition in self.partitions):
                self._notify_successors_ready()
            if not self.partitions:
                self.report = CompletionReport(step_id=self.step_id, partitions=())
                self.state = "TERMINAL"
                self._notify_terminal()
            return True
        except BaseException as error:
            self.fail(error, context="execute")
            return True

    def advance(self) -> None:
        if self.complete or not self.advance_execution() or self.complete:
            return
        try:
            published = len(self.published_partitions())
            for partition in self.partitions:
                partition.finish(self.step_id)
            if len(self.published_partitions()) != published:
                self._on_ready(self)
            if any(partition.result is None for partition in self.partitions):
                return
            report = CompletionReport(
                step_id=self.step_id,
                partitions=tuple(
                    partition.result for partition in self.partitions if partition.result is not None
                ),
            )
            _validate_report(self, report)
            self.report = report
            self.state = "TERMINAL"
            self._notify_terminal()
        except BaseException as error:
            self.fail(error, context="completion materialization")

    def published_partitions(self) -> tuple[PartitionCompletion, ...]:
        if self.report is not None:
            return self.report.partitions
        return tuple(partition.result for partition in self.partitions if partition.result is not None)

    def _notify_terminal(self) -> None:
        if self._terminal_notified:
            return
        self._terminal_notified = True
        self._on_terminal(self)

    def _notify_successors_ready(self) -> None:
        if self._successors_notified:
            return
        self._successors_notified = True
        self._on_successors_ready(self)


class BatchReader:
    """Independent wire cursor over the immutable results of one batch run."""

    __slots__ = ("run", "_sent", "_empty_sent", "_error_sent", "_closed", "_on_close")

    def __init__(self, run: BatchRun, on_close: Callable[[BatchReader], None]) -> None:
        self.run = run
        self._sent: set[int] = set()
        self._empty_sent = False
        self._error_sent = False
        self._closed = False
        self._on_close = on_close

    @property
    def step_id(self) -> int:
        return self.run.step_id

    @property
    def session_ids(self) -> frozenset[int]:
        return self.run.session_ids

    @property
    def error(self) -> WorkerError | None:
        return self.run.error

    @property
    def complete(self) -> bool:
        return self.run.complete

    def ready(self) -> bool:
        self.run.advance()
        if self.run.error is not None:
            return not self._error_sent
        if any(
            int(partition.partition_id) not in self._sent
            for partition in self.run.published_partitions()
        ):
            return True
        return bool(self.run.complete and not self.run.partition_order and not self._empty_sent)

    def take_ready(self) -> CompletionReport:
        self.run.advance()
        if self.run.error is not None:
            raise RuntimeError("terminal error must be consumed through take_error")
        partitions = tuple(
            partition
            for partition in self.run.published_partitions()
            if int(partition.partition_id) not in self._sent
        )
        if partitions:
            self._sent.update(int(partition.partition_id) for partition in partitions)
            return CompletionReport(step_id=self.run.step_id, partitions=partitions)
        if self.run.complete and not self.run.partition_order and not self._empty_sent:
            self._empty_sent = True
            return CompletionReport(step_id=self.run.step_id, partitions=())
        raise RuntimeError("batch reader has no query-ready partition")

    def take_error(self) -> WorkerError:
        error = self.run.error
        if error is None or self._error_sent:
            raise RuntimeError("batch reader has no unread terminal error")
        self._error_sent = True
        return error

    def pending(self) -> bool:
        if self.run.error is not None:
            return not self._error_sent
        if not self.run.complete:
            return True
        if not self.run.partition_order:
            return not self._empty_sent
        return any(
            int(partition.partition_id) not in self._sent
            for partition in self.run.published_partitions()
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._on_close(self)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class ReplayWindow:
    """Weighted LRU ownership for terminal batch runs."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("replay window capacity must be positive")
        self._runs: OrderedDict[int, BatchRun] = OrderedDict()
        self._weight = 0

    def take(self, step_id: int) -> BatchRun | None:
        run = self._runs.pop(int(step_id), None)
        if run is not None:
            self._weight -= run.weight
        return run

    def touch(self, step_id: int) -> None:
        if int(step_id) in self._runs:
            self._runs.move_to_end(int(step_id))

    def put(self, run: BatchRun) -> tuple[BatchRun, ...]:
        prior = self._runs.pop(run.step_id, None)
        if prior is not None:
            self._weight -= prior.weight
        self._runs[run.step_id] = run
        self._weight += run.weight
        evicted: list[BatchRun] = []
        while self._weight > self.capacity:
            _step_id, victim = self._runs.popitem(last=False)
            self._weight -= victim.weight
            evicted.append(victim)
        return tuple(evicted)

    def remove(self, step_id: int) -> BatchRun | None:
        return self.take(step_id)
