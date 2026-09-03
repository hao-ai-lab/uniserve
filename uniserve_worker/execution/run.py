"""Execution-run ownership and partial-result readers."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable

from ..foundation.errors import WorkerError, classify, invalid_descriptor
from .batch import LaneResult, Run, RunResult, TransferHandle
from .output import _completion_payload_ready, _record_ready, finalize_run_result
from .rows import PreparedExecution

_OperationKey = tuple[int, int, int]
_EpochKey = tuple[int, int]

__all__ = ["LaneRun", "RunReader", "WorkerRun", "ReplayWindow"]


def _operation_key(operation: object) -> _OperationKey:
    request = getattr(operation, "request_key")
    return (int(request.request_id), int(request.epoch), int(getattr(operation, "op_id")))


def _run_lineage(batch: Run) -> tuple[frozenset[int], frozenset[_EpochKey]]:
    request_keys = (
        *(admission.request_key for admission in batch.admissions),
        *(operation.request_key for operation in batch.operations),
        *(command.request_key for command in batch.commands),
    )
    epochs = frozenset((int(key.request_id), int(key.epoch)) for key in request_keys)
    return frozenset(request for request, _epoch in epochs), epochs


def _validate_report(run: WorkerRun, report: RunResult) -> None:
    if int(report.batch_id) != run.batch_id:
        raise invalid_descriptor("result batch identity does not match its submission")
    if int(report.run_id) != run.run_id:
        raise invalid_descriptor("result run identity does not match its submission")
    if tuple(int(lane.lane_id) for lane in report.lanes) != run.lane_order:
        raise invalid_descriptor("terminal report lane identity does not match its submission")
    for lane, expected_keys in zip(
        report.lanes, run.lane_operation_keys, strict=True
    ):
        actual_keys = tuple(
            (
                int(record.request_key.request_id),
                int(record.request_key.epoch),
                int(record.op_id),
            )
            for record in lane.completions
        )
        if actual_keys != expected_keys:
            raise invalid_descriptor("terminal report operations do not align with their lane")
        expected = set(expected_keys)
        for product in lane.products:
            reference = product.product
            key = (
                int(reference.request_key.request_id),
                int(reference.request_key.epoch),
                int(reference.producer_op_id),
            )
            if key not in expected:
                raise invalid_descriptor("terminal product does not belong to its completion lane")


class LaneRun:
    """One lane from device launch through immutable host publication."""

    __slots__ = (
        "lane_id",
        "state",
        "result",
        "_raw",
        "_record_cursor",
        "_product_cursor",
    )

    def __init__(self, lane_id: int) -> None:
        self.lane_id = int(lane_id)
        self.state = "PREPARED"
        self.result: LaneResult | None = None
        self._raw: LaneResult | None = None
        self._record_cursor = 0
        self._product_cursor = 0

    def launch(self, lane: LaneResult) -> None:
        if self.state != "PREPARED" or self._raw is not None:
            raise RuntimeError("lane execution source was bound more than once")
        if int(lane.lane_id) != self.lane_id:
            raise RuntimeError("lane execution source has the wrong identity")
        self._raw = lane
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

    def finish(self, batch_id: int, run_id: int) -> bool:
        if self.result is not None:
            return True
        if not self.device_ready():
            return False
        raw = self._raw
        if raw is None:
            raise RuntimeError("ready lane lost its execution source")
        report = finalize_run_result(
            RunResult(batch_id=int(batch_id), run_id=int(run_id), lanes=(raw,), done=True)
        )
        if len(report.lanes) != 1:
            raise RuntimeError("completion materialization changed lane cardinality")
        result = report.lanes[0]
        if any(
            type(token) is not int
            for record in result.completions
            for token in record.committed_tokens
        ):
            raise RuntimeError("materialized completion carries an unresolved committed token")
        if any(
            not isinstance(product.payload, (bytes, TransferHandle))
            for product in result.products
        ):
            raise RuntimeError("materialized completion contains an unresolved product payload")
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


class WorkerRun:
    """One exact-once physical run retained through replay eviction."""

    __slots__ = (
        "batch_id",
        "run_id",
        "run",
        "request_ids",
        "epochs",
        "lane_order",
        "lane_operation_keys",
        "lanes",
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
        run: Run,
        *,
        on_successors_ready: Callable[[WorkerRun], None],
        on_ready: Callable[[WorkerRun], None],
        on_terminal: Callable[[WorkerRun], None],
    ) -> None:
        requests, epochs = _run_lineage(run)
        self.batch_id = int(run.batch_id)
        self.run_id = int(run.run_id)
        self.run = run
        self.request_ids = requests
        self.epochs = epochs
        self.lane_order = tuple(int(lane.lane_id) for lane in run.lanes)
        self.lane_operation_keys = tuple(
            tuple(_operation_key(operation) for operation in lane.operations)
            for lane in run.lanes
        )
        self.lanes = tuple(LaneRun(value) for value in self.lane_order)
        self.weight = max(1, len(run.operations))
        self.state = "QUEUED"
        self.error: WorkerError | None = None
        self.report: RunResult | None = None
        self.active_readers = 0
        self._source: RunResult | PreparedExecution | None = None
        self._on_successors_ready = on_successors_ready
        self._on_ready = on_ready
        self._on_terminal = on_terminal
        self._successors_notified = False
        self._terminal_notified = False

    @property
    def complete(self) -> bool:
        return self.state == "TERMINAL"

    @property
    def source(self) -> RunResult | PreparedExecution | None:
        return self._source

    def attach(self, source: RunResult | PreparedExecution) -> None:
        if self.state != "QUEUED" or self._source is not None:
            raise RuntimeError("run already has an execution source")
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
        for lane in self.lanes:
            lane.abort()
        self._source = None
        self.state = "TERMINAL"
        self._notify_terminal()

    def advance_execution(self) -> bool:
        if self.complete or (
            self.lanes
            and all(lane.state != "PREPARED" for lane in self.lanes)
        ):
            return True
        source = self._source
        if source is None:
            return False
        try:
            if isinstance(source, RunResult):
                report = source
            else:
                if not source.ready():
                    return False
                report = source.resolve()
            _validate_report(self, report)
            for lane, raw in zip(self.lanes, report.lanes, strict=True):
                lane.launch(raw)
            self._source = None
            self.state = "FINISHING"
            if all(lane.successors_ready for lane in self.lanes):
                self._notify_successors_ready()
            if not self.lanes:
                self.report = RunResult(
                    batch_id=self.batch_id, run_id=self.run_id, lanes=(), done=True
                )
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
            published = len(self.published_lanes())
            for lane in self.lanes:
                lane.finish(self.batch_id, self.run_id)
            if len(self.published_lanes()) != published:
                self._on_ready(self)
            if any(lane.result is None for lane in self.lanes):
                return
            report = RunResult(
                batch_id=self.batch_id,
                run_id=self.run_id,
                lanes=tuple(
                    lane.result for lane in self.lanes if lane.result is not None
                ),
                done=True,
            )
            _validate_report(self, report)
            self.report = report
            self.state = "TERMINAL"
            self._notify_terminal()
        except BaseException as error:
            self.fail(error, context="completion materialization")

    def published_lanes(self) -> tuple[LaneResult, ...]:
        if self.report is not None:
            return self.report.lanes
        return tuple(lane.result for lane in self.lanes if lane.result is not None)

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


class RunReader:
    """Independent wire cursor over the immutable results of one physical run."""

    __slots__ = ("run", "_sent", "_empty_sent", "_error_sent", "_closed", "_on_close")

    def __init__(self, run: WorkerRun, on_close: Callable[[RunReader], None]) -> None:
        self.run = run
        self._sent: set[int] = set()
        self._empty_sent = False
        self._error_sent = False
        self._closed = False
        self._on_close = on_close

    @property
    def run_id(self) -> int:
        return self.run.run_id

    @property
    def request_ids(self) -> frozenset[int]:
        return self.run.request_ids

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
            int(lane.lane_id) not in self._sent
            for lane in self.run.published_lanes()
        ):
            return True
        return bool(self.run.complete and not self.run.lane_order and not self._empty_sent)

    def take_ready(self) -> RunResult:
        self.run.advance()
        if self.run.error is not None:
            raise RuntimeError("terminal error must be consumed through take_error")
        lanes = tuple(
            lane
            for lane in self.run.published_lanes()
            if int(lane.lane_id) not in self._sent
        )
        if lanes:
            self._sent.update(int(lane.lane_id) for lane in lanes)
            done = self.run.complete and len(self._sent) == len(self.run.lane_order)
            return RunResult(
                batch_id=self.run.batch_id,
                run_id=self.run.run_id,
                lanes=lanes,
                done=done,
            )
        if self.run.complete and not self.run.lane_order and not self._empty_sent:
            self._empty_sent = True
            return RunResult(
                batch_id=self.run.batch_id, run_id=self.run.run_id, lanes=(), done=True
            )
        raise RuntimeError("run reader has no query-ready lane")

    def take_error(self) -> WorkerError:
        error = self.run.error
        if error is None or self._error_sent:
            raise RuntimeError("run reader has no unread terminal error")
        self._error_sent = True
        return error

    def pending(self) -> bool:
        if self.run.error is not None:
            return not self._error_sent
        if not self.run.complete:
            return True
        if not self.run.lane_order:
            return not self._empty_sent
        return any(
            int(lane.lane_id) not in self._sent
            for lane in self.run.published_lanes()
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
        self._runs: OrderedDict[int, WorkerRun] = OrderedDict()
        self._weight = 0

    def take(self, run_id: int) -> WorkerRun | None:
        run = self._runs.pop(int(run_id), None)
        if run is not None:
            self._weight -= run.weight
        return run

    def touch(self, run_id: int) -> None:
        if int(run_id) in self._runs:
            self._runs.move_to_end(int(run_id))

    def put(self, run: WorkerRun) -> tuple[WorkerRun, ...]:
        prior = self._runs.pop(run.run_id, None)
        if prior is not None:
            self._weight -= prior.weight
        self._runs[run.run_id] = run
        self._weight += run.weight
        evicted: list[WorkerRun] = []
        while self._weight > self.capacity:
            _run_id, victim = self._runs.popitem(last=False)
            self._weight -= victim.weight
            evicted.append(victim)
        return tuple(evicted)

    def remove(self, run_id: int) -> WorkerRun | None:
        return self.take(run_id)
