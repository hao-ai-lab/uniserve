"""WorkerRuntime: the reusable worker shell.

Owns the host↔worker IPC endpoint, request dispatch, capability gating,
typed-error classification, and metrics. Model-specific work lives behind a
caps/execute/drop_request adapter; for runner-backed models that adapter is the
generic runner driver, so models never touch IPC or control dispatch.
"""
from __future__ import annotations

import inspect
import logging
from typing import Any, Mapping, Protocol, runtime_checkable

from ..contracts.caps import CONTROL_KINDS, Caps, validate_caps, validate_forward_result
from ..contracts.outputs import FinalizableSeqResult
from ..foundation.errors import (
    WorkerError,
    classify,
    scheduler_bug,
    should_capture_trace,
)
from .control_plane import ControlPlane
from .execution_pipeline import ExecutionPipeline, PendingResult
from .metrics import MetricsService
from .profiler import WorkerProfiler
from .worker_kind import FULL as WORKER_KIND_FULL
from .worker_kind import UND as WORKER_KIND_UND
from .worker_kind import restrict_supported_ops

__all__ = ["WorkerRuntime", "WorkerDriver", "dispatch"]

logger = logging.getLogger(__name__)

# Control op kinds (vs get_caps/execute/drop_request/shutdown). Controls absent
# from the driver's declared ``supported_controls`` raise UnsupportedControl.
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


def _driver_caps(driver: "WorkerDriver") -> dict:
    return driver.caps().to_wire()


def _complete_seq_result(raw: dict) -> dict:
    raw = _finalize_seq_result(raw)
    out = {"req_id": raw.get("req_id")}
    for key in SEQ_RESULT_FIELDS:
        out[key] = raw.get(key)
    out["denoise_done"] = bool(out["denoise_done"])
    return out


def _complete_result(raw: dict) -> dict:
    _finalize_result_inplace(raw)
    return {
        "step_id": raw.get("step_id"),
        "per_seq": [_complete_seq_result(dict(item)) for item in raw.get("per_seq") or []],
        "worker_exec_us": raw.get("worker_exec_us"),
        "forward_stats": raw.get("forward_stats"),
    }


def _complete_metrics(raw: dict) -> dict:
    out: dict[str, Any] = {
        "executes": int(raw.get("executes") or 0),
        "ops_total": int(raw.get("ops_total") or 0),
        "exec_us_total": int(raw.get("exec_us_total") or 0),
        "last_exec_us": int(raw.get("last_exec_us") or 0),
    }
    for key in METRIC_MAP_FIELDS:
        out[key] = dict(raw.get(key) or {})
    if isinstance(raw.get("forward"), dict):
        forward = dict(raw["forward"])
        out["forward"] = forward
        for key in (
            "cuda_graph_captures",
            "cuda_graph_replays",
            "cuda_graph_misses",
            "cuda_graph_fallbacks",
            "cuda_graph_unpadded_tokens",
            "cuda_graph_padded_tokens",
        ):
            out[key] = int(forward.get(key) or 0)
        out["cuda_graph_runtime_mode_counts"] = dict(
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
            out[key] = int(raw.get(key) or 0)
        out["cuda_graph_runtime_mode_counts"] = dict(
            raw.get("cuda_graph_runtime_mode_counts") or {}
        )
    return out


def _complete_response(raw: dict) -> dict:
    out = {"kind": raw["kind"]}
    for key in RESPONSE_OPTIONAL_FIELDS:
        out[key] = raw.get(key)
    if isinstance(out["result"], dict):
        out["result"] = _complete_result(out["result"])
    if isinstance(out["metrics"], dict):
        out["metrics"] = _complete_metrics(out["metrics"])
    return out


def _handle_get_caps(
    driver: "WorkerDriver",
    supported_controls: set[str],
    req: Mapping[str, Any],
    metrics: "MetricsService | None",
) -> dict:
    return {"kind": "caps", "caps": _driver_caps(driver)}


def _handle_get_metrics(
    driver: "WorkerDriver",
    supported_controls: set[str],
    req: Mapping[str, Any],
    metrics: "MetricsService | None",
) -> dict:
    return {"kind": "metrics", "metrics": (metrics.snapshot() if metrics is not None else {})}


def _handle_get_pressure(
    driver: "WorkerDriver",
    supported_controls: set[str],
    req: Mapping[str, Any],
    metrics: "MetricsService | None",
) -> dict:
    return {"kind": "pressure", "pressure": driver.resource_pressure()}


def _handle_execute(
    driver: "WorkerDriver",
    supported_controls: set[str],
    req: Mapping[str, Any],
    metrics: "MetricsService | None",
) -> dict:
    # execute is owned exclusively by WorkerRuntime.handle(), which performs the
    # metrics / forward-result validation / op_id stamping / deferred-CPU
    # handling. dispatch() must never run a second, weaker execute path, so
    # reaching here is a routing bug.
    raise scheduler_bug("execute must be dispatched through WorkerRuntime.handle()")


def _handle_drop_request(
    driver: "WorkerDriver",
    supported_controls: set[str],
    req: Mapping[str, Any],
    metrics: "MetricsService | None",
) -> dict:
    driver.drop_request(req["req_id"])
    return {"kind": "ok"}


# Request-kind -> handler. Control kinds are not listed here; they fall through
# to the control path below, gated on the driver's declared supported_controls.
HANDLERS = {
    "get_caps": _handle_get_caps,
    "get_metrics": _handle_get_metrics,
    "get_pressure": _handle_get_pressure,
    "execute": _handle_execute,
    "drop_request": _handle_drop_request,
}


def _dispatch_control(
    driver: "WorkerDriver",
    supported_controls: set[str],
    kind: str,
    req: Mapping[str, Any],
) -> dict:
    return ControlPlane(driver, supported_controls).handle(kind, req)


def dispatch(
    driver: "WorkerDriver",
    supported_controls: set[str],
    req: Mapping[str, Any],
    metrics: MetricsService | None = None,
) -> dict:
    """Pure request -> response dispatch (no ring I/O, no shutdown).

    Raises a typed ``WorkerError`` for unsupported controls / unknown kinds and
    lets driver exceptions propagate to the caller's classifier.
    """
    kind = req.get("kind")
    handler = HANDLERS.get(kind) if isinstance(kind, str) else None
    if handler is not None:
        return handler(driver, supported_controls, req, metrics)
    if isinstance(kind, str) and kind in CONTROL_KINDS:
        return _dispatch_control(driver, supported_controls, kind, req)
    raise scheduler_bug(f"unknown request kind: {kind!r}")


def _driver_accepts_deferred_text_cpu_results(driver: "WorkerDriver") -> bool:
    execute = driver.execute
    try:
        signature = inspect.signature(execute)
    except (TypeError, ValueError):
        return False
    return "defer_text_cpu_results" in signature.parameters


def _driver_execute(
    driver,
    batch: dict,
    *,
    defer_text_cpu_results: bool,
    accepts_deferred_text_cpu_results: bool,
) -> dict:
    if accepts_deferred_text_cpu_results:
        return driver.execute(batch, defer_text_cpu_results=defer_text_cpu_results)
    return driver.execute(batch)


def _finalize_seq_result(raw):
    if isinstance(raw, FinalizableSeqResult):
        raw = raw.finalize()
    if not isinstance(raw, dict):
        raw = dict(raw)
    return raw


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
    return isinstance(per_seq, list) and any(isinstance(item, FinalizableSeqResult) for item in per_seq)


def _result_ready(resp: dict | PendingResult) -> bool:
    pending = resp if isinstance(resp, PendingResult) else PendingResult(response=resp)
    return pending.ready()


def _add_forward_component_us(resp: dict, component: str, dur_ns: int) -> None:
    result = resp.get("result")
    if not isinstance(result, dict):
        return
    stats = result.get("forward_stats")
    if not isinstance(stats, dict):
        return
    component_us = stats.setdefault("component_us", {})
    if not isinstance(component_us, dict):
        return
    key = str(component)
    component_us[key] = int(component_us.get(key) or 0) + int(dur_ns) // 1000


def _add_deferred_cuda_ready_component(resp: dict) -> None:
    result = resp.get("result")
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
        _add_forward_component_us(resp, "worker_cuda_ready_elapsed", total_us * 1000)


@runtime_checkable
class WorkerDriver(Protocol):
    """The adapter contract the runtime shell drives.

    ``caps``/``execute``/``drop_request`` are always required. Control methods
    have no-op defaults on ``BaseWorkerDriver`` and are invoked directly after
    caps validation proves the worker advertises the control.
    """

    def caps(self) -> Caps: ...

    def execute(self, batch: dict) -> dict: ...

    def drop_request(self, req_id: int) -> None: ...

    def resource_pressure(self) -> list[dict[str, Any]]: ...

    def copy_blocks(self, copies: Any) -> None: ...

    def load_lora(self, lora_id: int, lora_path: str) -> None: ...

    def unload_lora(self, lora_id: int) -> None: ...

    def free_encoder(self, handles: Any) -> None: ...

    def reset_prefix_cache(self) -> None: ...

    def sleep(self) -> None: ...

    def wake_up(self) -> None: ...


class WorkerRuntime:
    """Transport + dispatch + services shell around one worker adapter."""

    def __init__(
        self,
        driver: WorkerDriver,
        server,
        metrics: MetricsService | None = None,
        *,
        worker_kind: str = WORKER_KIND_FULL,
        pipeline_depth: int = 1,
    ):
        self.driver = driver
        self.server = server
        self.metrics = metrics or MetricsService()
        self.worker_kind = worker_kind
        # Number of forwards the serve loop launches before finalizing the oldest.
        # Bounded by the host's in-flight cap; when the host stops submitting,
        # ``try_recv`` returns None and the pipeline drains.
        self.pipeline_depth = max(1, int(pipeline_depth))
        self.caps = validate_caps(_driver_caps(driver), owner=driver.__class__.__name__).to_wire()
        # A peeled stage advertises only its OpKind subset so the host's
        # StageRouter routes the right ops to it; the gate in ``_execute_pipeline``
        # then rejects anything outside it. ``full`` uses the model's full op set
        # and the gate is inert.
        declared_ops = list(self.caps.get("supported_ops") or [])
        if worker_kind != WORKER_KIND_FULL:
            restricted = restrict_supported_ops(worker_kind, declared_ops)
            self.caps["supported_ops"] = restricted
            self.allowed_ops: frozenset[str] = frozenset(restricted)
        else:
            self.allowed_ops = frozenset(declared_ops)
        self.supported_controls = set(self.caps.get("supported_controls") or [])
        self.control_plane = ControlPlane(self.driver, self.supported_controls)
        self._driver_accepts_deferred_text_cpu_results = _driver_accepts_deferred_text_cpu_results(
            driver
        )
        # The Mode-A und worker must read each decode's sampled token synchronously
        # to detect the image-start trigger and publish the conditioning KV; the
        # deferred-CPU-results pipeline hides the token until a later finalize, so
        # the und worker runs text results synchronously.
        self._defer_text_results = worker_kind != WORKER_KIND_UND
        self.profiler = WorkerProfiler.from_env()
        self.execution_pipeline = ExecutionPipeline()

    def handle(self, req, *, allow_deferred: bool = False) -> dict:
        """Dispatch one decoded request, classifying failures and recording
        metrics. Returns the response dict (the ring write stays in `serve`)."""

        return self.pending(req, allow_deferred=allow_deferred).response

    def pending(self, req, *, allow_deferred: bool = False) -> PendingResult:
        """Dispatch one decoded request and retain pending-finalization state."""

        kind = req.get("kind")
        try:
            if kind == "execute":
                return self._execute_pipeline(req, allow_deferred=allow_deferred)
            if kind == "get_caps":
                resp = {"kind": "caps", "caps": dict(self.caps)}
            elif kind in CONTROL_KINDS:
                resp = self.control_plane.handle(kind, req)
            else:
                resp = dispatch(self.driver, self.supported_controls, req, self.metrics)
            if kind in CONTROL_KINDS:
                self.metrics.record_control(kind, True)
            return self.execution_pipeline.immediate(resp)
        except WorkerError as err:
            if kind in CONTROL_KINDS:
                self.metrics.record_control(kind, False)
            self.metrics.record_error(err.code)
            # Log per-request context locally; wire responses omit diagnostic ids.
            log = logger.exception if should_capture_trace(err.code) else logger.warning
            log(
                "worker request %r failed: %s [code=%s req_id=%s op_id=%s op_kind=%s details=%s]",
                kind, err.message, err.code, err.req_id, err.op_id, err.op_kind, err.details,
            )
            return self.execution_pipeline.immediate(err.to_wire())
        except Exception as exc:  # noqa: BLE001 — classify everything else
            logger.exception("worker request %r raised an unclassified error", kind)
            werr = classify(exc, context=kind)
            self.metrics.record_error(werr.code)
            return self.execution_pipeline.immediate(werr.to_wire())

    def _execute_pipeline(self, req: dict, *, allow_deferred: bool) -> PendingResult:
        """Run one execute batch: time -> execute -> validate -> annotate -> record.

        Returns a pending response. A deferred result skips eager validation here;
        the pending object retains the source batch so send-time finalization can
        validate after the late CPU copy completes. Driver exceptions propagate to
        ``pending``'s classifier unchanged.
        """
        batch = req.get("batch") or {}
        ops = batch.get("ops") or []
        op_kinds = [op.get("kind", "?") for op in ops]
        # Routing guard: a peeled stage must only receive ops in its declared
        # OpKind subset. ``full`` skips this (subset is the model's full op set).
        if self.worker_kind != WORKER_KIND_FULL:
            for kind in op_kinds:
                if kind not in self.allowed_ops:
                    raise scheduler_bug(
                        f"worker_kind={self.worker_kind!r} received op kind {kind!r} "
                        f"outside its OpKind subset {sorted(self.allowed_ops)}"
                    )
        t0 = self.metrics.now_ns()
        result = self._run_execute(req, allow_deferred=allow_deferred)
        wait_start = self.metrics.now_ns()
        pending = self.execution_pipeline.pending(
            result=result,
            batch=batch,
            wait_start_ns=wait_start,
        )
        has_deferred = _result_has_deferred(result)
        if isinstance(result, dict) and not has_deferred:
            validate_forward_result(result, batch, owner=self.driver.__class__.__name__)
        dur = self.metrics.now_ns() - t0
        self._record_execute_metrics(dur, op_kinds, result)
        if isinstance(result, dict):
            self._annotate_execute_result(result, dur, ops)
        return pending

    def _run_execute(self, req: dict, *, allow_deferred: bool) -> dict:
        with self.profiler.step("uniserve.worker.execute"):
            return _driver_execute(
                self.driver,
                req["batch"],
                defer_text_cpu_results=allow_deferred,
                accepts_deferred_text_cpu_results=(
                    self._driver_accepts_deferred_text_cpu_results
                ),
            )

    def _record_execute_metrics(self, dur: int, op_kinds: list, result) -> None:
        self.metrics.record_execute(dur, op_kinds)
        if isinstance(result, dict):
            self.metrics.record_forward_stats(result.get("forward_stats"))

    def _annotate_execute_result(self, result: dict, dur: int, ops: list) -> None:
        # Stamp worker compute time and echo the submitted operation identity.
        result["worker_exec_us"] = dur // 1000
        for op, sr in zip(ops, result.get("per_seq") or []):
            if isinstance(sr, dict):
                sr["op_kind"] = op.get("kind")
                oid = op.get("op_id")
                if oid is not None:
                    sr["op_id"] = oid

    def _prepare_response_for_send(self, pending: PendingResult) -> dict:
        resp = pending.response
        result = resp.get("result")
        if isinstance(result, dict):
            _finalize_result_inplace(result)
            if isinstance(pending.batch, dict):
                validate_forward_result(
                    result,
                    pending.batch,
                    owner=self.driver.__class__.__name__,
                )
                for op, sr in zip(pending.batch.get("ops") or [], result.get("per_seq") or []):
                    if isinstance(sr, dict):
                        sr["op_kind"] = op.get("kind")
                        oid = op.get("op_id")
                        if oid is not None:
                            sr["op_id"] = oid
        return resp

    def _respond(self, pending: dict | PendingResult) -> None:
        pending = pending if isinstance(pending, PendingResult) else PendingResult(response=pending)
        resp = pending.response
        call_id = resp.get("call_id")
        wait_start = pending.wait_start_ns
        if isinstance(wait_start, int):
            wait_ns = self.metrics.now_ns() - wait_start
            _add_forward_component_us(resp, "worker_deferred_wait", wait_ns)
            self.metrics.record_pipeline("deferred_wait", wait_ns)
            _add_deferred_cuda_ready_component(resp)
        t0 = self.metrics.now_ns()
        try:
            resp = self._prepare_response_for_send(pending)
        except Exception as exc:  # noqa: BLE001 - late CPU-copy/validation failures.
            logger.exception("worker response finalization failed")
            werr = classify(exc, context="respond")
            self.metrics.record_error(werr.code)
            resp = werr.to_wire()
            if call_id is not None:
                resp["call_id"] = call_id
        # Split deferred-D2H materialize wall time from the ring write for metrics.
        t1 = self.metrics.now_ns()
        _add_forward_component_us(resp, "worker_result_finalize", t1 - t0)
        self.server.respond(_complete_response(resp))
        self.metrics.record_pipeline("finalize", t1 - t0)
        self.metrics.record_pipeline("encode_send", self.metrics.now_ns() - t1)

    def serve(self) -> None:
        """Run the depth-D pipelined request loop."""

        from .process import WorkerProcess

        WorkerProcess(self, self.server).run()

    def result_ready(self, resp: dict | PendingResult) -> bool:
        return _result_ready(resp)

    def respond_pending(self, resp: dict | PendingResult) -> None:
        self._respond(resp)
