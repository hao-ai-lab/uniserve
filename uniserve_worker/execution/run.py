"""In-flight execution ownership and incremental result delivery."""

from __future__ import annotations

from collections.abc import Callable

from uniserve_worker.execution.batch import (
    Finish,
    Free,
    LaneResult,
    ModelOutput,
    Retire,
    Run,
    RunResult,
    TransferHandle,
)
from uniserve_worker.execution.output import (
    _completion_payload_ready,
    _record_ready,
    finalize_run_result,
)
from uniserve_worker.execution.rows import PreparedExecution
from uniserve_worker.foundation.errors import WorkerError, classify, invalid_descriptor
from uniserve_worker.foundation.resources import close_resources

_OperationKey = tuple[int, int, int]

__all__ = ["LaneRun", "WorkerRun"]


def _operation_key(operation: object) -> _OperationKey:
    """Extract a stable request-and-operation identity from an operation-like value."""

    request = getattr(operation, "request_key")
    return (int(request.request_id), int(request.epoch), int(getattr(operation, "op_id")))


def _request_ids(batch: Run) -> frozenset[int]:
    return frozenset(
        key.request_id
        for key in (
            *(admission.request_key for admission in batch.admissions),
            *(operation.request_key for operation in batch.operations),
            *(command.request_key for command in batch.commands),
        )
    )


def _validate_report(run: WorkerRun, report: RunResult) -> None:
    """Validate that a run report covers each submitted lane and operation exactly once."""

    if int(report.batch_id) != run.batch_id:
        raise invalid_descriptor("result batch identity does not match its submission")
    if int(report.run_id) != run.run_id:
        raise invalid_descriptor("result run identity does not match its submission")
    if tuple(int(lane.lane_id) for lane in report.lanes) != run.lane_order:
        raise invalid_descriptor("terminal report lane identity does not match its submission")
    for lane, expected_keys in zip(report.lanes, run.lane_operation_keys, strict=True):
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
    """Pending outputs for one logical result lane, materialized when ready."""

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
        """Attach the lane’s pending result and reject duplicate execution."""

        if self.state != "PREPARED" or self._raw is not None:
            raise RuntimeError("lane execution source was bound more than once")
        if int(lane.lane_id) != self.lane_id:
            raise RuntimeError("lane execution source has the wrong identity")
        self._raw = lane
        self.state = "LAUNCHED"

    def device_ready(self) -> bool:
        """Return whether the lane has launched and every device-side completion is query-ready."""

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
        """Indicate whether this lane's request publication is visible to dependent runs."""

        raw = self._raw
        publication = None if raw is None else raw.publication
        return publication is not None and publication.successors_ready

    def finish(self, batch_id: int, run_id: int) -> bool:
        """Finalize a ready lane, publish its request transition, and return its immutable result."""

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
        for record in result.completions:
            if not isinstance(record, ModelOutput):
                raise RuntimeError("materialized completion still has a pending output")
            if any(type(token) is not int for token in record.committed_tokens):
                raise RuntimeError("materialized completion carries an unresolved committed token")
        if any(
            not isinstance(product.payload, (bytes, TransferHandle)) for product in result.products
        ):
            raise RuntimeError("materialized completion contains an unresolved product payload")
        self.result = result
        self._raw = None
        self.state = "FINISHED"
        return True

    def abort(self) -> None:
        """Cancel publication and abandon unfinished outputs for this lane."""

        if self.result is None:
            raw = self._raw
            publication = None if raw is None else raw.publication
            actions = []
            if publication is not None:
                actions.append(publication.cancel)
            if raw is not None:
                actions.extend(
                    record.abandon
                    for record in raw.completions
                    if not isinstance(record, ModelOutput)
                )
            self._raw = None
            self.state = "ABORTED"
            close_resources(*actions)


class WorkerRun:
    """Own one execution and its undelivered Submit/Poll results until retirement."""

    __slots__ = (
        "batch_id",
        "run_id",
        "run",
        "request_ids",
        "lane_order",
        "lane_operation_keys",
        "lanes",
        "state",
        "error",
        "report",
        "requires_command_ack",
        "_retirement",
        "_source",
        "_on_successors_ready",
        "_on_ready",
        "_successors_notified",
        "_terminal_notified",
        "_sent",
        "_terminal_sent",
        "awaiting_poll",
    )

    def __init__(
        self,
        run: Run,
        *,
        on_successors_ready: Callable[[WorkerRun], None],
        on_ready: Callable[[WorkerRun], None],
    ) -> None:
        self.batch_id = int(run.batch_id)
        self.run_id = int(run.run_id)
        self.run = run
        self.request_ids = _request_ids(run)
        self.lane_order = tuple(int(lane.lane_id) for lane in run.lanes)
        self.lane_operation_keys = tuple(
            tuple(_operation_key(operation) for operation in lane.operations) for lane in run.lanes
        )
        self.lanes = tuple(LaneRun(value) for value in self.lane_order)

        self.state = "QUEUED"
        self.error: WorkerError | None = None
        self.report: RunResult | None = None
        self.requires_command_ack = any(
            isinstance(command, (Free, Finish, Retire)) for command in run.commands
        )
        self._retirement: Callable[[], bool] | None = None
        self._source: RunResult | PreparedExecution | None = None

        self._on_successors_ready = on_successors_ready
        self._on_ready = on_ready
        self._successors_notified = False
        self._terminal_notified = False

        self._sent: set[int] = set()
        self._terminal_sent = False
        self.awaiting_poll = False

    @property
    def complete(self) -> bool:
        """Indicate whether every lane has reached an immutable terminal result."""

        return self.state == "TERMINAL"

    @property
    def source(self) -> RunResult | PreparedExecution | None:
        """Expose the prepared execution or precomputed result attached to this run."""

        return self._source

    def attach(self, source: RunResult | PreparedExecution) -> None:
        """Attach either a prepared execution or an already materialized run result."""

        if self.state != "QUEUED" or self._source is not None:
            raise RuntimeError("run already has an execution source")
        self._source = source
        self.state = "RUNNING"

    def fail(self, error: BaseException, *, context: str = "execute") -> None:
        """Classify a run failure, abort active lanes, and deliver its error once."""

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
        try:
            self.close()
        except BaseException as cleanup_error:
            self.error.add_note(f"run cleanup failed: {cleanup_error!r}")
        self.state = "TERMINAL"
        self._notify_terminal()

    def close(self) -> None:
        """Abandon owned preparation and result leases before releasing run references."""

        source, self._source = self._source, None
        actions = [lane.abort for lane in self.lanes]
        if isinstance(source, PreparedExecution):
            actions.insert(0, source.abandon)
        elif isinstance(source, RunResult):
            for lane in source.lanes:
                if lane.publication is not None:
                    actions.append(lane.publication.cancel)
                actions.extend(
                    record.abandon
                    for record in lane.completions
                    if not isinstance(record, ModelOutput)
                )
        close_resources(*actions)

    def advance_execution(self) -> bool:
        """Launch a prepared batch once its input transfers and predicates are ready."""

        if self.complete or self.state == "FINISHING":
            return True
        source = self._source
        if source is None:
            return False
        try:
            if isinstance(source, RunResult):
                report = source
            else:
                if not source.advance():
                    return False
                report = source.resolve()
            _validate_report(self, report)
            for lane, raw in zip(self.lanes, report.lanes, strict=True):
                lane.launch(raw)
            self._retirement = report.retirement
            self._source = None
            self.state = "FINISHING"
            if self.lanes and all(lane.successors_ready for lane in self.lanes):
                self._notify_successors_ready()
            return True
        except BaseException as error:
            self.fail(error, context="execute")
            return True

    def advance(self) -> None:
        """Advance execution and finalize every lane whose device results are ready."""

        if not self.advance_execution():
            return
        # Advancing may have completed the run with an execution error.
        if self.complete:
            return
        try:
            published = len(self.published_lanes())
            for lane in self.lanes:
                lane.finish(self.batch_id, self.run_id)
            if len(self.published_lanes()) != published:
                self._on_ready(self)
            if any(lane.result is None for lane in self.lanes):
                return

            # Free/Finish/Retire acknowledge only after outstanding readers release storage.
            if self._retirement is not None:
                if not self._retirement():
                    return
                self._retirement = None

            report = RunResult(
                batch_id=self.batch_id,
                run_id=self.run_id,
                lanes=tuple(lane.result for lane in self.lanes if lane.result is not None),
                done=True,
            )
            _validate_report(self, report)
            self.report = report
            self.state = "TERMINAL"
            self._notify_terminal()
        except BaseException as error:
            self.fail(error, context="completion materialization")

    def published_lanes(self) -> tuple[LaneResult, ...]:
        """List lane results already safe to expose to the scheduler."""

        if self.report is not None:
            return self.report.lanes
        return tuple(lane.result for lane in self.lanes if lane.result is not None)

    def _notify_terminal(self) -> None:
        """Invoke the terminal callback once when a run reaches terminal state."""

        if self._terminal_notified:
            return
        self._terminal_notified = True
        self._notify_successors_ready()
        self._on_ready(self)

    def _notify_successors_ready(self) -> None:
        """Notify dependents once the run's produced values become readable."""

        if self._successors_notified:
            return
        self._successors_notified = True
        self._on_successors_ready(self)

    def ready(self) -> bool:
        """Return whether the requested run delivery position can yield a report or terminal error."""

        self.advance()
        if self.error is not None:
            return not self._terminal_sent
        if any(int(lane.lane_id) not in self._sent for lane in self.published_lanes()):
            return True
        return self.complete and not self._terminal_sent

    def take_ready(self) -> RunResult:
        """Return the next report at the delivery position and advance past completed lanes."""

        self.advance()
        if self.error is not None:
            raise RuntimeError("terminal error must be consumed through take_error")
        lanes = tuple(
            lane for lane in self.published_lanes() if int(lane.lane_id) not in self._sent
        )
        if lanes:
            # A fragment's aggregate counters belong to one computation entry.
            # Entry participants can differ even within the same physical run.
            entries = {int(lane.lane_id): lane.operations[0].entry for lane in self.run.lanes}
            entry = entries[int(lanes[0].lane_id)]
            lanes = tuple(lane for lane in lanes if entries[int(lane.lane_id)] == entry)
            self._sent.update(int(lane.lane_id) for lane in lanes)
            done = (
                self.complete
                and len(self._sent) == len(self.lane_order)
                and not self.requires_command_ack
            )
            self._terminal_sent = done
            return RunResult(
                batch_id=self.batch_id,
                run_id=self.run_id,
                lanes=lanes,
                done=done,
            )
        if self.complete and not self._terminal_sent:
            self._terminal_sent = True
            return RunResult(batch_id=self.batch_id, run_id=self.run_id, lanes=(), done=True)
        raise RuntimeError("run has no query-ready lane")

    def take_error(self) -> WorkerError:
        """Return the run’s terminal error after consuming this run."""

        error = self.error
        if error is None or self._terminal_sent:
            raise RuntimeError("run has no unread terminal error")
        self._terminal_sent = True
        return error

    def pending(self) -> bool:
        """Indicate whether the delivery position still owes a result or terminal acknowledgement."""

        return not self._terminal_sent
