"""WorkerServer: IPC dispatch and lifecycle around one execution worker.

Owns the host↔worker IPC endpoint, request dispatch, capability gating,
typed-error classification, metrics, and deferred result delivery. Model
materialization and execution choices are resolved before this module starts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping

from ..contracts.caps import CONTROL_KINDS, validate_caps, validate_forward_result
from ..contracts.outputs import FinalizableSeqResult
from ..foundation.errors import (
    WorkerError,
    classify,
    scheduler_bug,
    should_capture_trace,
)
from ..worker.protocol import ResultPolicy, Worker
from .control_plane import ControlPlane
from .execution_pipeline import ExecutionPipeline, PendingResult
from .metrics import MetricsService
from .profiler import WorkerProfiler

if TYPE_CHECKING:
    from .process import WorkerIpcTransport

__all__ = ["WorkerServer", "dispatch"]

logger = logging.getLogger(__name__)

# Control op kinds (vs get_caps/execute/drop_request/shutdown). Controls absent
# from the worker's declared ``supported_controls`` raise UnsupportedControl.
# Vocabulary is owned by ``CONTROL_KINDS`` so dispatch and caps validation stay aligned.
RESPONSE_OPTIONAL_FIELDS = (
    "call_id",
    "caps",
    "result",
    "metrics",
    "pressure",
    "message",
    "code",
    "retryable",
    "fatal",
)

# Optional per-seq result keys in wire completion order. ``op_id`` and
# ``op_kind`` are stamped by the runtime, not by output dataclasses.
SEQ_RESULT_FIELDS = (
    "op_kind",
    "sampled_token_id",
    "sampled_token_ids",
    "denoise_done",
    "num_steps_done",
    "image_png_b64",
    "image_hw",
    "sampled_logprob",
    "top_logprobs",
    "prompt_logprobs",
    "encoder_handle",
    "num_tokens",
    "num_accepted_tokens",
    "op_id",
    "logits_handle",
    "locator",
)

METRIC_MAP_FIELDS = (
    "op_kind_counts",
    "op_kind_us",
    "control_ok",
    "control_err",
    "error_counts",
)


def _worker_caps(worker: Worker) -> dict:
    return worker.caps().to_wire()


def _complete_seq_result(payload: dict) -> dict:
    payload = _finalize_seq_result(payload)
    response = {"req_id": payload.get("req_id")}
    for key in SEQ_RESULT_FIELDS:
        response[key] = payload.get(key)
    response["denoise_done"] = bool(response["denoise_done"])
    return response


def _complete_result(payload: dict) -> dict:
    _finalize_result_inplace(payload)
    return {
        "step_id": payload.get("step_id"),
        "per_seq": [_complete_seq_result(dict(item)) for item in payload.get("per_seq") or []],
        "worker_exec_us": payload.get("worker_exec_us"),
        "forward_stats": payload.get("forward_stats"),
    }


def _complete_metrics(payload: dict) -> dict:
    response: dict[str, Any] = {
        "executes": int(payload.get("executes") or 0),
        "ops_total": int(payload.get("ops_total") or 0),
        "exec_us_total": int(payload.get("exec_us_total") or 0),
        "last_exec_us": int(payload.get("last_exec_us") or 0),
    }
    for key in METRIC_MAP_FIELDS:
        response[key] = dict(payload.get(key) or {})
    if isinstance(payload.get("forward"), dict):
        forward = dict(payload["forward"])
        response["forward"] = forward
        for key in (
            "cuda_graph_captures",
            "cuda_graph_replays",
            "cuda_graph_misses",
            "cuda_graph_fallbacks",
            "cuda_graph_unpadded_tokens",
            "cuda_graph_padded_tokens",
        ):
            response[key] = int(forward.get(key) or 0)
        response["cuda_graph_runtime_mode_counts"] = dict(
            forward.get("cuda_graph_runtime_mode_counts") or {}
        )
    else:
        for key in (
            "cuda_graph_captures",
            "cuda_graph_replays",
            "cuda_graph_misses",
            "cuda_graph_fallbacks",
            "cuda_graph_unpadded_tokens",
            "cuda_graph_padded_tokens",
        ):
            response[key] = int(payload.get(key) or 0)
        response["cuda_graph_runtime_mode_counts"] = dict(
            payload.get("cuda_graph_runtime_mode_counts") or {}
        )
    return response


def _complete_response(payload: dict) -> dict:
    response = {"kind": payload["kind"]}
    for key in RESPONSE_OPTIONAL_FIELDS:
        response[key] = payload.get(key)
    if isinstance(response["result"], dict):
        response["result"] = _complete_result(response["result"])
    if isinstance(response["metrics"], dict):
        response["metrics"] = _complete_metrics(response["metrics"])
    return response


def _handle_get_caps(
    worker: Worker,
    supported_controls: set[str],
    request: Mapping[str, Any],
    metrics: "MetricsService | None",
) -> dict:
    return {"kind": "caps", "caps": _worker_caps(worker)}


def _handle_get_metrics(
    worker: Worker,
    supported_controls: set[str],
    request: Mapping[str, Any],
    metrics: "MetricsService | None",
) -> dict:
    return {"kind": "metrics", "metrics": (metrics.snapshot() if metrics is not None else {})}


def _handle_get_pressure(
    worker: Worker,
    supported_controls: set[str],
    request: Mapping[str, Any],
    metrics: "MetricsService | None",
) -> dict:
    return {"kind": "pressure", "pressure": worker.resource_pressure()}


def _handle_execute(
    worker: Worker,
    supported_controls: set[str],
    request: Mapping[str, Any],
    metrics: "MetricsService | None",
) -> dict:
    # execute is owned exclusively by WorkerServer.handle(), which performs the
    # metrics / forward-result validation / op_id stamping / deferred-CPU
    # handling. dispatch() must never run a second, weaker execute path, so
    # reaching here is a routing bug.
    raise scheduler_bug("execute must be dispatched through WorkerServer.handle()")


def _handle_drop_request(
    worker: Worker,
    supported_controls: set[str],
    request: Mapping[str, Any],
    metrics: "MetricsService | None",
) -> dict:
    worker.drop_request(request["req_id"])
    return {"kind": "ok"}


# Request-kind -> handler. Control kinds are not listed here; they fall through
# to the control path below, gated on the worker's declared supported_controls.
HANDLERS = {
    "get_caps": _handle_get_caps,
    "get_metrics": _handle_get_metrics,
    "get_pressure": _handle_get_pressure,
    "execute": _handle_execute,
    "drop_request": _handle_drop_request,
}


def _dispatch_control(
    worker: Worker,
    supported_controls: set[str],
    kind: str,
    request: Mapping[str, Any],
) -> dict:
    return ControlPlane(worker, supported_controls).handle(kind, request)


def dispatch(
    worker: Worker,
    supported_controls: set[str],
    request: Mapping[str, Any],
    metrics: MetricsService | None = None,
) -> dict:
    """Pure request -> response dispatch (no ring I/O, no shutdown).

    Raises a typed ``WorkerError`` for unsupported controls / unknown kinds and
    lets worker exceptions propagate to the caller's classifier.
    """
    kind = request.get("kind")
    handler = HANDLERS.get(kind) if isinstance(kind, str) else None
    if handler is not None:
        return handler(worker, supported_controls, request, metrics)
    if isinstance(kind, str) and kind in CONTROL_KINDS:
        return _dispatch_control(
            worker,
            supported_controls,
            kind,
            request,
        )
    raise scheduler_bug(f"unknown request kind: {kind!r}")


def _finalize_seq_result(payload):
    if isinstance(payload, FinalizableSeqResult):
        payload = payload.finalize()
    if not isinstance(payload, dict):
        payload = dict(payload)
    return payload


def _finalize_result_inplace(result: dict) -> None:
    per_seq = result.get("per_seq")
    if not isinstance(per_seq, list):
        return
    for idx, item in enumerate(per_seq):
        per_seq[idx] = _finalize_seq_result(item)


def _result_has_deferred(result: dict | None) -> bool:
    if not isinstance(result, dict):
        return False
    per_seq = result.get("per_seq")
    return isinstance(per_seq, list) and any(
        isinstance(item, FinalizableSeqResult) for item in per_seq
    )


def _result_ready(response: dict | PendingResult) -> bool:
    pending = response if isinstance(response, PendingResult) else PendingResult(response=response)
    return pending.ready()


def _add_forward_component_us(
    response: dict,
    component: str,
    duration_ns: int,
) -> None:
    result = response.get("result")
    if not isinstance(result, dict):
        return
    stats = result.get("forward_stats")
    if not isinstance(stats, dict):
        return
    component_us = stats.setdefault("component_us", {})
    if not isinstance(component_us, dict):
        return
    key = str(component)
    component_us[key] = int(component_us.get(key) or 0) + int(duration_ns) // 1000


def _add_deferred_cuda_ready_component(response: dict) -> None:
    result = response.get("result")
    if not isinstance(result, dict):
        return
    per_seq = result.get("per_seq")
    if not isinstance(per_seq, list):
        return
    seen: set[int] = set()
    total_us = 0
    for item in per_seq:
        elapsed = getattr(item, "cuda_ready_elapsed_us", None)
        if not callable(elapsed):
            continue
        key_fn = getattr(item, "cuda_ready_group_key", None)
        key = int(key_fn()) if callable(key_fn) else id(item)
        if key in seen:
            continue
        seen.add(key)
        value = elapsed()
        if isinstance(value, int) and value > 0:
            total_us += int(value)
    if total_us > 0:
        _add_forward_component_us(
            response,
            "worker_cuda_ready_elapsed",
            total_us * 1000,
        )


class WorkerServer:
    """Transport, dispatch, and metrics shell around one assembled worker."""

    def __init__(
        self,
        worker: Worker,
        ipc_endpoint: WorkerIpcTransport,
        metrics: MetricsService | None = None,
    ):
        self.worker = worker
        self.ipc_endpoint = ipc_endpoint
        self.metrics = metrics or MetricsService()
        contract = worker.contract
        self.caps = validate_caps(
            contract.capabilities,
            owner=type(worker).__name__,
        ).to_wire()
        self.pipeline_depth = max(1, int(self.caps["pipeline_depth"]))
        self.allowed_ops = frozenset(self.caps["supported_ops"])
        self.supported_controls = set(self.caps.get("supported_controls") or [])
        self.control_plane = ControlPlane(
            self.worker,
            self.supported_controls,
        )
        self.allow_deferred_results = contract.result_policy is ResultPolicy.DEFER_WHEN_AVAILABLE
        self.profiler = WorkerProfiler.from_env()
        self.execution_pipeline = ExecutionPipeline()

    def handle(self, request, *, allow_deferred: bool = False) -> dict:
        """Dispatch one decoded request, classifying failures and recording
        metrics. Returns the response dict (the ring write stays in `serve`)."""

        return self.pending(request, allow_deferred=allow_deferred).response

    def pending(self, request, *, allow_deferred: bool = False) -> PendingResult:
        """Dispatch one decoded request and retain pending-finalization state."""

        request_kind = request.get("kind")
        try:
            if request_kind == "execute":
                return self._execute_pipeline(
                    request,
                    allow_deferred=allow_deferred,
                )
            if request_kind == "get_caps":
                response = {"kind": "caps", "caps": dict(self.caps)}
            elif request_kind in CONTROL_KINDS:
                response = self.control_plane.handle(request_kind, request)
            else:
                response = dispatch(
                    self.worker,
                    self.supported_controls,
                    request,
                    self.metrics,
                )
            if request_kind in CONTROL_KINDS:
                self.metrics.record_control(request_kind, True)
            return self.execution_pipeline.immediate(response)
        except WorkerError as error:
            if request_kind in CONTROL_KINDS:
                self.metrics.record_control(request_kind, False)
            self.metrics.record_error(error.code)
            # Log per-request context locally; wire responses omit diagnostic ids.
            log = logger.exception if should_capture_trace(error.code) else logger.warning
            log(
                "worker request %r failed: %s [code=%s req_id=%s op_id=%s op_kind=%s details=%s]",
                request_kind,
                error.message,
                error.code,
                error.req_id,
                error.op_id,
                error.op_kind,
                error.details,
            )
            return self.execution_pipeline.immediate(error.to_wire())
        except Exception as error:  # noqa: BLE001 — classify everything else
            logger.exception(
                "worker request %r raised an unclassified error",
                request_kind,
            )
            worker_error = classify(error, context=request_kind)
            self.metrics.record_error(worker_error.code)
            return self.execution_pipeline.immediate(worker_error.to_wire())

    def _execute_pipeline(
        self,
        request: dict,
        *,
        allow_deferred: bool,
    ) -> PendingResult:
        """Run one execute batch: time -> execute -> validate -> annotate -> record.

        Returns a pending response. A deferred result skips eager validation here;
        the pending object retains the source batch so send-time finalization can
        validate after the late CPU copy completes. Worker exceptions propagate to
        ``pending``'s classifier unchanged.
        """
        batch = request.get("batch") or {}
        operations = batch.get("ops") or []
        operation_kinds = [operation.get("kind", "?") for operation in operations]
        for operation_kind in operation_kinds:
            if operation_kind not in self.allowed_ops:
                raise scheduler_bug(
                    f"worker received operation {operation_kind!r} outside its "
                    f"capability set {sorted(self.allowed_ops)}"
                )
        started_ns = self.metrics.now_ns()
        result = self._run_execute(
            request,
            allow_deferred=allow_deferred,
        )
        wait_started_ns = self.metrics.now_ns()
        pending = self.execution_pipeline.pending(
            result=result,
            batch=batch,
            wait_start_ns=wait_started_ns,
        )
        has_deferred = _result_has_deferred(result)
        if isinstance(result, dict) and not has_deferred:
            validate_forward_result(
                result,
                batch,
                owner=type(self.worker).__name__,
            )
        duration_ns = self.metrics.now_ns() - started_ns
        self._record_execute_metrics(
            duration_ns,
            operation_kinds,
            result,
        )
        if isinstance(result, dict):
            self._annotate_execute_result(
                result,
                duration_ns,
                operations,
            )
        return pending

    def _run_execute(
        self,
        request: dict,
        *,
        allow_deferred: bool,
    ) -> dict:
        with self.profiler.step("uniserve.worker.execute"):
            return self.worker.execute(
                request["batch"],
                defer_text_cpu_results=allow_deferred,
            )

    def _record_execute_metrics(
        self,
        duration_ns: int,
        operation_kinds: list,
        result,
    ) -> None:
        self.metrics.record_execute(duration_ns, operation_kinds)
        if isinstance(result, dict):
            self.metrics.record_forward_stats(result.get("forward_stats"))

    def _annotate_execute_result(
        self,
        result: dict,
        duration_ns: int,
        operations: list,
    ) -> None:
        # Stamp worker compute time and echo the submitted operation identity.
        result["worker_exec_us"] = duration_ns // 1000
        for operation, sequence_result in zip(
            operations,
            result.get("per_seq") or [],
        ):
            if isinstance(sequence_result, dict):
                sequence_result["op_kind"] = operation.get("kind")
                operation_id = operation.get("op_id")
                if operation_id is not None:
                    sequence_result["op_id"] = operation_id

    def _prepare_response_for_send(self, pending: PendingResult) -> dict:
        response = pending.response
        result = response.get("result")
        if isinstance(result, dict):
            _finalize_result_inplace(result)
            if isinstance(pending.batch, dict):
                validate_forward_result(
                    result,
                    pending.batch,
                    owner=type(self.worker).__name__,
                )
                for operation, sequence_result in zip(
                    pending.batch.get("ops") or [],
                    result.get("per_seq") or [],
                ):
                    if isinstance(sequence_result, dict):
                        sequence_result["op_kind"] = operation.get("kind")
                        operation_id = operation.get("op_id")
                        if operation_id is not None:
                            sequence_result["op_id"] = operation_id
        return response

    def _respond(
        self,
        pending_response: dict | PendingResult,
    ) -> None:
        pending = (
            pending_response
            if isinstance(pending_response, PendingResult)
            else PendingResult(response=pending_response)
        )
        response = pending.response
        call_id = response.get("call_id")
        wait_started_ns = pending.wait_start_ns
        if isinstance(wait_started_ns, int):
            wait_ns = self.metrics.now_ns() - wait_started_ns
            _add_forward_component_us(
                response,
                "worker_deferred_wait",
                wait_ns,
            )
            self.metrics.record_pipeline("deferred_wait", wait_ns)
            _add_deferred_cuda_ready_component(response)
        finalize_started_ns = self.metrics.now_ns()
        try:
            response = self._prepare_response_for_send(pending)
        except Exception as error:  # noqa: BLE001 - late finalization failures.
            logger.exception("worker response finalization failed")
            worker_error = classify(error, context="respond")
            self.metrics.record_error(worker_error.code)
            response = worker_error.to_wire()
            if call_id is not None:
                response["call_id"] = call_id
        # Split deferred-D2H materialize wall time from the ring write for metrics.
        finalized_ns = self.metrics.now_ns()
        _add_forward_component_us(
            response,
            "worker_result_finalize",
            finalized_ns - finalize_started_ns,
        )
        self.ipc_endpoint.respond(_complete_response(response))
        self.metrics.record_pipeline(
            "finalize",
            finalized_ns - finalize_started_ns,
        )
        self.metrics.record_pipeline(
            "encode_send",
            self.metrics.now_ns() - finalized_ns,
        )

    def serve(self) -> None:
        """Run the depth-D pipelined request loop."""

        from .process import WorkerServeLoop

        WorkerServeLoop(self, self.ipc_endpoint).run()

    def result_ready(
        self,
        response: dict | PendingResult,
    ) -> bool:
        return _result_ready(response)

    def respond_pending(
        self,
        response: dict | PendingResult,
    ) -> None:
        self._respond(response)
