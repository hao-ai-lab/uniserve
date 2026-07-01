"""WorkerRuntime: the reusable worker shell.

Owns the host↔worker IPC endpoint, request dispatch, capability gating,
typed-error classification, and metrics. Model-specific work lives behind a
caps/execute/drop_request adapter; for runner-backed models that adapter is the
generic runner driver, so models never touch IPC or control dispatch.
"""
from __future__ import annotations

import inspect
import logging
from collections import deque
from typing import Any, Mapping, Protocol, runtime_checkable

from ..contracts.caps import CONTROL_KINDS, Caps, validate_caps, validate_forward_result
from ..foundation.errors import (
    WorkerError,
    classify,
    scheduler_bug,
    should_capture_trace,
    unsupported_control,
)
from .controls import CONTROL_SPECS
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

# Optional per-seq result keys in wire completion order. ``op_id`` is stamped
# by the runtime in ``WorkerRuntime.handle``, not by output dataclasses.
SEQ_RESULT_FIELDS = (
    "sampled_token_id",
    "sampled_token_ids",
    "denoise_done",
    "num_steps_done",
    "image_png_b64",
    "image_hw",
    "sampled_logprob",
    "top_logprobs",
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
    out = {
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
    if kind not in supported_controls:
        raise unsupported_control(kind)
    spec = CONTROL_SPECS[kind]
    fn = getattr(driver, spec.method)
    fn(**spec.build_kwargs(kind, req))
    return {"kind": "ok"}


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
    if kind in CONTROL_KINDS:
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


@runtime_checkable
class FinalizableSeqResult(Protocol):
    def finalize(self) -> Mapping[str, Any]: ...


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
        self._driver_accepts_deferred_text_cpu_results = _driver_accepts_deferred_text_cpu_results(
            driver
        )
        # The Mode-A und worker must read each decode's sampled token synchronously
        # to detect the image-start trigger and publish the conditioning KV; the
        # deferred-CPU-results pipeline hides the token until a later finalize, so
        # the und worker runs text results synchronously.
        self._defer_text_results = worker_kind != WORKER_KIND_UND
        self.profiler = WorkerProfiler.from_env()

    def handle(self, req, *, allow_deferred: bool = False) -> dict:
        """Dispatch one decoded request, classifying failures and recording
        metrics. Returns the response dict (the ring write stays in `serve`)."""
        kind = req.get("kind")
        try:
            if kind == "execute":
                return self._execute_pipeline(req, allow_deferred=allow_deferred)
            if kind == "get_caps":
                resp = {"kind": "caps", "caps": dict(self.caps)}
            else:
                resp = dispatch(self.driver, self.supported_controls, req, self.metrics)
            if kind in CONTROL_KINDS:
                self.metrics.record_control(kind, True)
            return resp
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
            return err.to_wire()
        except Exception as exc:  # noqa: BLE001 — classify everything else
            logger.exception("worker request %r raised an unclassified error", kind)
            werr = classify(exc, context=kind)
            self.metrics.record_error(werr.code)
            return werr.to_wire()

    def _execute_pipeline(self, req: dict, *, allow_deferred: bool) -> dict:
        """Run one execute batch: time -> execute -> validate -> annotate -> record.

        Returns the ``{"kind": "result", ...}`` response. A deferred (CPU-finalize)
        result skips eager validation here and carries its source batch on
        ``_deferred_batch`` so the response path validates/annotates it after the
        late CPU copy completes (see ``_prepare_response_for_send``). Driver
        exceptions propagate to ``handle``'s classifier unchanged.
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
        resp = {"kind": "result", "result": result}
        has_deferred = _result_has_deferred(result)
        if isinstance(result, dict) and not has_deferred:
            validate_forward_result(result, batch, owner=self.driver.__class__.__name__)
        dur = self.metrics.now_ns() - t0
        self._record_execute_metrics(dur, op_kinds, result)
        if isinstance(result, dict):
            self._annotate_execute_result(result, dur, ops)
            if has_deferred:
                resp["_deferred_batch"] = batch
        return resp

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
        # Stamp worker compute time and echo per-op op_id (per_seq aligns with ops).
        result["worker_exec_us"] = dur // 1000
        for op, sr in zip(ops, result.get("per_seq") or []):
            oid = op.get("op_id")
            if oid is not None and isinstance(sr, dict):
                sr["op_id"] = oid

    def _prepare_response_for_send(self, resp: dict) -> dict:
        result = resp.get("result")
        if isinstance(result, dict):
            _finalize_result_inplace(result)
            batch = resp.pop("_deferred_batch", None)
            if isinstance(batch, dict):
                validate_forward_result(
                    result,
                    batch,
                    owner=self.driver.__class__.__name__,
                )
                for op, sr in zip(batch.get("ops") or [], result.get("per_seq") or []):
                    oid = op.get("op_id")
                    if oid is not None and isinstance(sr, dict):
                        sr["op_id"] = oid
        return resp

    def _respond(self, resp: dict) -> None:
        call_id = resp.get("call_id")
        t0 = self.metrics.now_ns()
        try:
            resp = self._prepare_response_for_send(resp)
        except Exception as exc:  # noqa: BLE001 - late CPU-copy/validation failures.
            logger.exception("worker response finalization failed")
            werr = classify(exc, context="respond")
            self.metrics.record_error(werr.code)
            resp = werr.to_wire()
            if call_id is not None:
                resp["call_id"] = call_id
        # Split deferred-D2H materialize wall time from the ring write for metrics.
        t1 = self.metrics.now_ns()
        self.server.respond(_complete_response(resp))
        self.metrics.record_pipeline("finalize", t1 - t0)
        self.metrics.record_pipeline("encode_send", self.metrics.now_ns() - t1)

    def serve(self) -> None:
        """Run the depth-D pipelined request loop."""

        _PipelineServeLoop(self).run()


class _PipelineServeLoop:
    """FIFO pipelined receive/dispatch/finalize loop for ``WorkerRuntime``.

    The runtime owns dispatch/classification/metrics; this helper owns only the
    transport scheduling policy: non-blocking refill while in-flight work exists,
    FIFO finalization, and blocking receive when idle.
    """

    def __init__(self, runtime: WorkerRuntime) -> None:
        self.runtime = runtime
        self.inflight: deque[tuple[int | None, dict]] = deque()
        self.shutdown_resp: dict | None = None
        self.draining = False

    def run(self) -> None:
        try:
            while True:
                self._refill_nonblocking()
                if self._finalize_oldest():
                    continue
                if self._finish_shutdown_if_drained():
                    break
                if self._receive_idle_request():
                    break
        finally:
            self.runtime.profiler.close()

    def _refill_nonblocking(self) -> None:
        while not self.draining and len(self.inflight) < self.runtime.pipeline_depth:
            req = self._recv_nonblocking()
            if req is None:
                return
            if self._capture_shutdown(req):
                return
            self.inflight.append(self._dispatch(req))

    def _finalize_oldest(self) -> bool:
        if not self.inflight:
            return False
        self._finalize_and_send(self.inflight.popleft())
        return True

    def _finish_shutdown_if_drained(self) -> bool:
        if not self.draining:
            return False
        self.runtime._respond(self.shutdown_resp or {"kind": "ok"})
        return True

    def _receive_idle_request(self) -> bool:
        req = self._recv_blocking()
        if self._capture_shutdown(req):
            self.runtime._respond(self.shutdown_resp or {"kind": "ok"})
            return True
        self.inflight.append(self._dispatch(req))
        return False

    def _capture_shutdown(self, req: dict) -> bool:
        if req.get("kind") != "shutdown":
            return False
        self.shutdown_resp = self._shutdown_response(req)
        self.draining = True
        return True

    def _recv_nonblocking(self) -> dict | None:
        """Non-blocking receive; ``None`` when no request is queued.

        Falls back to ``None`` for transports without ``try_recv`` (the loop then
        degrades to depth-1 blocking recv, still correct, just unpipelined)."""
        try_recv = getattr(self.runtime.server, "try_recv", None)
        if not callable(try_recv):
            return None
        t0 = self.runtime.metrics.now_ns()
        req = try_recv()
        self.runtime.metrics.record_pipeline("recv", self.runtime.metrics.now_ns() - t0)
        return req

    def _recv_blocking(self) -> dict:
        """Block until a request arrives. Only called with the pipeline drained."""
        t0 = self.runtime.metrics.now_ns()
        req = self.runtime.server.recv()
        self.runtime.metrics.record_pipeline("idle", self.runtime.metrics.now_ns() - t0)
        return req

    def _dispatch(self, req: dict) -> tuple[int | None, dict]:
        """Decode + launch one request, returning its (call_id, response).

        For an execute this launches the GPU forward (CUDA-async) and returns a
        response that may carry deferred per-seq results; for a control it runs
        the control inline. Finalize/send happens later, in receive order."""
        call_id = req.get("call_id")
        t0 = self.runtime.metrics.now_ns()
        resp = self.runtime.handle(req, allow_deferred=self.runtime._defer_text_results)
        self.runtime.metrics.record_pipeline("dispatch", self.runtime.metrics.now_ns() - t0)
        if call_id is not None:
            resp["call_id"] = call_id
        return call_id, resp

    def _finalize_and_send(self, item: tuple[int | None, dict]) -> None:
        self.runtime._respond(item[1])

    @staticmethod
    def _shutdown_response(req: dict) -> dict:
        resp: dict[str, Any] = {"kind": "ok"}
        call_id = req.get("call_id")
        if call_id is not None:
            resp["call_id"] = call_id
        return resp
