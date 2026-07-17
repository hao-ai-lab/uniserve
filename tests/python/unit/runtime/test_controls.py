"""Control dispatch and typed-error conformance.

Runs against the GPU-free StubWorker through the real dispatch path so the
worker's reject-before-execute behavior and the typed error taxonomy are
exercised exactly as in production.
"""

from __future__ import annotations

import pytest

from uniserve_worker.foundation.errors import ErrorCode, WorkerError, classify
from uniserve_worker.server.app import CONTROL_KINDS, WorkerServer, dispatch
from uniserve_worker.server.metrics import MetricsService
from uniserve_worker.server.stub import StubWorker

pytestmark = pytest.mark.unit


def _worker():
    return StubWorker(block_size=256)


def _supported(worker):
    return set(worker.caps().supported_controls)


def test_declared_controls_return_ok():
    worker = _worker()
    supported = _supported(worker)
    assert supported, "stub should declare some controls"
    for ctrl in supported:
        req = {"kind": ctrl, "copies": [], "free_handles": [1], "lora_id": 1, "lora_path": "/x"}
        assert dispatch(worker, supported, req) == {"kind": "ok"}, ctrl


class _RecordingControlWorker:
    """Worker that records the arguments each control hot path receives.

    `test_declared_controls_return_ok` only proves dispatch acknowledges a
    control against the no-op stub; it does not prove the untrusted wire fields
    are parsed and forwarded to the right method with the right arity. This spy
    captures the actual worker call so the wire-field plumbing for
    copy_blocks/load_lora/free_encoder is exercised end to end.
    """

    SUPPORTED_CONTROLS = ["copy_blocks", "load_lora", "unload_lora", "free_encoder"]

    def __init__(self):
        self.calls: list[tuple] = []

    def caps(self):
        return {"supported_controls": list(self.SUPPORTED_CONTROLS)}

    def copy_blocks(self, copies):
        self.calls.append(("copy_blocks", copies))

    def load_lora(self, lora_id, lora_path):
        self.calls.append(("load_lora", lora_id, lora_path))

    def unload_lora(self, lora_id):
        self.calls.append(("unload_lora", lora_id))

    def free_encoder(self, handles):
        self.calls.append(("free_encoder", handles))


def test_control_hot_paths_forward_parsed_wire_fields():
    worker = _RecordingControlWorker()
    supported = set(worker.SUPPORTED_CONTROLS)

    copies = [{"src": 1, "dst": 2}]
    assert dispatch(worker, supported, {"kind": "copy_blocks", "copies": copies}) == {"kind": "ok"}
    assert dispatch(
        worker, supported, {"kind": "load_lora", "lora_id": 7, "lora_path": "/adapters/a"}
    ) == {"kind": "ok"}
    assert dispatch(worker, supported, {"kind": "unload_lora", "lora_id": 7}) == {"kind": "ok"}
    assert dispatch(worker, supported, {"kind": "free_encoder", "free_handles": [11, 22]}) == {
        "kind": "ok"
    }

    # The worker methods were actually invoked with the wire fields parsed into
    # the documented positional shape (copies/handles as lists, scalars passed
    # through), not merely acknowledged.
    assert worker.calls == [
        ("copy_blocks", copies),
        ("load_lora", 7, "/adapters/a"),
        ("unload_lora", 7),
        ("free_encoder", [11, 22]),
    ]


def test_control_hot_paths_reject_missing_required_fields():
    worker = _RecordingControlWorker()
    supported = set(worker.SUPPORTED_CONTROLS)

    # load_lora's scalar wire fields are required: a missing field must fail
    # before the worker is invoked with a typed InvalidDescriptor, not silently
    # pass None into the control method.
    for req in (
        {"kind": "load_lora", "lora_path": "/adapters/a"},  # missing lora_id
        {"kind": "load_lora", "lora_id": 7},  # missing lora_path
        {"kind": "unload_lora"},  # missing lora_id
    ):
        with pytest.raises(WorkerError) as excinfo:
            dispatch(worker, supported, req)
        assert excinfo.value.code == ErrorCode.INVALID_DESCRIPTOR
    assert worker.calls == []

    # List-valued control fields default to an empty list when absent rather
    # than erroring, and reject a non-list payload with InvalidDescriptor.
    assert dispatch(worker, supported, {"kind": "copy_blocks"}) == {"kind": "ok"}
    assert worker.calls[-1] == ("copy_blocks", [])
    with pytest.raises(WorkerError) as excinfo:
        dispatch(worker, supported, {"kind": "free_encoder", "free_handles": 5})
    assert excinfo.value.code == ErrorCode.INVALID_DESCRIPTOR


def test_undeclared_controls_rejected_typed():
    worker = _worker()
    supported = _supported(worker)
    undeclared = [c for c in CONTROL_KINDS if c not in supported]
    assert "sleep" in undeclared and "wake_up" in undeclared
    for ctrl in undeclared:
        with pytest.raises(WorkerError) as excinfo:
            dispatch(worker, supported, {"kind": ctrl})
        err = excinfo.value
        assert err.code == ErrorCode.UNSUPPORTED_CONTROL
        assert err.fatal is False and err.retryable is False
        wire = err.to_wire()
        assert wire["kind"] == "error" and wire["code"] == "UnsupportedControl"
        assert wire["fatal"] is False


def test_unknown_kind_is_scheduler_bug():
    worker = _worker()
    with pytest.raises(WorkerError) as excinfo:
        dispatch(worker, _supported(worker), {"kind": "frobnicate"})
    assert excinfo.value.code == ErrorCode.SCHEDULER_BUG


def test_core_kinds_roundtrip():
    worker = _worker()
    supported = _supported(worker)
    assert dispatch(worker, supported, {"kind": "get_caps"})["kind"] == "caps"
    assert dispatch(worker, supported, {"kind": "drop_request", "req_id": 1}) == {"kind": "ok"}
    batch = {
        "step_id": 1,
        "new_reqs": [{"req_id": 1}],
        "ops": [
            {
                "req_id": 1,
                "kind": "prefill_und",
                "modality": "und",
                "pos_range": [0, 3],
                "token_ids": [1, 2, 3],
            }
        ],
    }
    # execute is owned by handle() (the single instrumented path the host uses);
    # dispatch() no longer carries a weaker shadow execute.
    runtime = WorkerServer(worker, ipc_endpoint=None)
    resp = runtime.handle({"kind": "execute", "batch": batch})
    assert resp["kind"] == "result"
    assert resp["result"]["per_seq"][0]["req_id"] == 1
    assert resp["result"]["per_seq"][0]["op_kind"] == "prefill_und"


def test_observability_kinds_are_read_only_and_scalar():
    worker = _worker()
    metrics = MetricsService()
    metrics.record_execute(12_000, ["prefill_und", "decode_und"])
    metrics.record_forward_stats(
        {
            "mode_counts": {"extend": 1},
            "mode_tokens": {"extend": 3},
            "mode_us": {"extend": 7},
            "component_us": {"text_model_forward": 6},
            "attention_launches": 2,
            "attention_us": 5,
            "attention_backend_counts": {"torch_sdpa": 2},
            "operator_launches": 3,
            "operator_us": 4,
            "operator_counts": {"rms_norm:eager": 2, "silu_and_mul:eager": 1},
            "cuda_graph_replays": 3,
            "cuda_graph_padded_tokens": 2,
            "cuda_graph_unpadded_tokens": 6,
            "cuda_graph_runtime_mode_counts": {"decode": 3},
            "flashinfer_decode_plan_calls": 5,
            "flashinfer_decode_plan_reuses": 11,
            "flashinfer_decode_graph_plan_calls": 2,
            "spec_verify_rows": 2,
            "spec_verify_draft_tokens": 6,
            "spec_verify_accepted_tokens": 4,
            "spec_verify_rejected_tokens": 2,
            "spec_verify_committed_tokens": 6,
            "spec_verify_path_counts": {"greedy_device": 1, "sglang_target_only": 1},
        }
    )

    metric_resp = dispatch(worker, _supported(worker), {"kind": "get_metrics"}, metrics)
    assert metric_resp["kind"] == "metrics"
    assert metric_resp["metrics"]["executes"] == 1
    assert metric_resp["metrics"]["op_kind_counts"]["prefill_und"] == 1
    assert metric_resp["metrics"]["forward"]["mode_tokens"]["extend"] == 3
    assert metric_resp["metrics"]["forward"]["component_us"]["text_model_forward"] == 6
    assert metric_resp["metrics"]["forward"]["attention_backend_counts"]["torch_sdpa"] == 2
    assert metric_resp["metrics"]["forward"]["operator_launches"] == 3
    assert metric_resp["metrics"]["forward"]["operator_counts"]["rms_norm:eager"] == 2
    assert metric_resp["metrics"]["cuda_graph_replays"] == 3
    assert metric_resp["metrics"]["cuda_graph_padded_tokens"] == 2
    assert metric_resp["metrics"]["forward"]["cuda_graph_runtime_mode_counts"]["decode"] == 3
    assert metric_resp["metrics"]["forward"]["flashinfer_decode_plan_calls"] == 5
    assert metric_resp["metrics"]["forward"]["flashinfer_decode_plan_reuses"] == 11
    assert metric_resp["metrics"]["forward"]["flashinfer_decode_graph_plan_calls"] == 2
    assert metric_resp["metrics"]["forward"]["spec_verify_accepted_tokens"] == 4
    assert metric_resp["metrics"]["forward"]["spec_verify_rejected_tokens"] == 2
    assert metric_resp["metrics"]["forward"]["spec_verify_committed_tokens"] == 6
    assert metric_resp["metrics"]["forward"]["spec_verify_path_counts"]["greedy_device"] == 1

    pressure_resp = dispatch(worker, _supported(worker), {"kind": "get_pressure"}, metrics)
    assert pressure_resp["kind"] == "pressure"
    assert isinstance(pressure_resp["pressure"], list)

    # Read-only contract: get_metrics/get_pressure must observe state, never
    # mutate it. A direct snapshot taken after the read-only dispatches must be
    # byte-identical to one taken before, so no counter advanced as a side
    # effect of being read.
    before = metrics.snapshot()
    dispatch(worker, _supported(worker), {"kind": "get_metrics"}, metrics)
    dispatch(worker, _supported(worker), {"kind": "get_pressure"}, metrics)
    assert metrics.snapshot() == before

    # The response must hand back a defensive copy, not the live counters:
    # mutating the returned payload cannot leak back into the service.
    leaked = dispatch(worker, _supported(worker), {"kind": "get_metrics"}, metrics)["metrics"]
    leaked["executes"] = 999_999
    leaked["op_kind_counts"]["prefill_und"] = 999_999
    leaked["forward"]["mode_tokens"]["extend"] = 999_999
    assert metrics.snapshot() == before


def test_classify_taxonomy_mapping():
    assert classify(NotImplementedError("x")).code == ErrorCode.UNSUPPORTED_OPERATION
    assert classify(KeyError("token_ids")).code == ErrorCode.INVALID_DESCRIPTOR
    assert classify(ValueError("bad")).code == ErrorCode.INVALID_DESCRIPTOR
    assert classify(IndexError("oob")).code == ErrorCode.INVALID_DESCRIPTOR
    assert classify(RuntimeError("CUDA error: out of memory")).code == ErrorCode.WORKER_OOM
    assert classify(AssertionError("inv")).code == ErrorCode.INVARIANT_VIOLATION
    assert classify(RuntimeError("kaboom")).code == ErrorCode.MODEL_EXECUTION_ERROR


def test_classify_passes_through_and_enriches():
    err = WorkerError(code=ErrorCode.USER_INPUT_ERROR, message="m")
    assert classify(err) is err
    # enrich missing context ids without overwriting
    enriched = classify(
        WorkerError(code=ErrorCode.MODEL_EXECUTION_ERROR, message="m"),
        req_id=7,
        op_kind="decode_und",
    )
    assert enriched.req_id == 7 and enriched.op_kind == "decode_und"


def test_worker_error_wire_shape():
    err = WorkerError(
        code=ErrorCode.MODEL_EXECUTION_ERROR,
        message="boom",
        fatal=False,
        req_id=3,
        op_kind="decode_und",
    )
    wire = err.to_wire()
    assert wire["kind"] == "error"
    assert wire["code"] == "ModelExecutionError"
    assert wire["fatal"] is False
    # Rich context stays local (logging/metrics); only the Invariant-A-audited
    # fields modeled on the Rust WorkerResponse cross the wire.
    assert err.req_id == 3 and err.op_kind == "decode_und"
    assert set(wire) == {"kind", "code", "message", "retryable", "fatal"}


class _FifoServer:
    """Single-FIFO request source mirroring the iceoryx2 receive side."""

    def __init__(self, requests):
        from collections import deque

        self.requests = deque(requests)
        self.responses = []
        self.recv_count = 0
        self.try_recv_count = 0

    def recv(self):
        self.recv_count += 1
        return self.requests.popleft()

    def try_recv(self):
        self.try_recv_count += 1
        return self.requests.popleft() if self.requests else None

    def respond(self, response):
        self.responses.append(response)


def _execute_req(
    call_id, step_id, req_id, *, kind="decode_und", token_ids=(5,), pos_range=(0, 1), **op
):
    return {
        "kind": "execute",
        "call_id": call_id,
        "batch": {
            "step_id": step_id,
            "new_reqs": [{"req_id": req_id}],
            "ops": [
                {
                    "req_id": req_id,
                    "kind": kind,
                    "modality": "und",
                    "pos_range": list(pos_range),
                    "token_ids": list(token_ids),
                    **op,
                }
            ],
        },
    }


def test_worker_server_responds_in_receive_order():
    # Mixed controls respond strictly in call_id order; already-queued requests
    # are taken via the non-blocking try_recv, so the blocking recv is reserved
    # for the idle path (plan Phase 2 recv-ahead).
    server = _FifoServer(
        [
            {"kind": "get_caps", "call_id": 1},
            {"kind": "get_metrics", "call_id": 2},
            {"kind": "shutdown", "call_id": 3},
        ]
    )

    WorkerServer(_worker(), server).serve()

    assert [resp["call_id"] for resp in server.responses] == [1, 2, 3]
    assert [resp["kind"] for resp in server.responses] == ["caps", "metrics", "ok"]
    assert server.recv_count == 0
    assert server.try_recv_count >= 3


class _DeferredSeq:
    def __init__(self, events: list[str], req_id: int, token: int, *, ready_after: int = 0) -> None:
        self.events = events
        self.req_id = req_id
        self.token = token
        self.ready_after = ready_after
        self.ready_checks = 0

    def ready(self) -> bool:
        self.ready_checks += 1
        self.events.append(f"ready:{self.req_id}:{self.ready_checks}")
        return self.ready_checks > self.ready_after

    def finalize(self) -> dict:
        self.events.append(f"finalize:{self.req_id}")
        return {"req_id": self.req_id, "sampled_token_id": self.token}


class _DeferredOverlapWorker(StubWorker):
    def __init__(
        self,
        events: list[str],
        *,
        defer_steps: set[int] | None = None,
        ready_after: dict[int, int] | None = None,
        pipeline_depth: int = 1,
    ) -> None:
        super().__init__(
            block_size=256,
            pipeline_depth=pipeline_depth,
        )
        self.events = events
        self.defer_steps = defer_steps or {1}
        self.ready_after = ready_after or {}

    def execute(self, batch, *, defer_text_cpu_results=False):
        req_id = int(batch["ops"][0]["req_id"])
        step_id = int(batch["step_id"])
        self.events.append(f"execute_step:{step_id}:defer={defer_text_cpu_results}")
        if step_id in self.defer_steps and defer_text_cpu_results:
            return {
                "step_id": step_id,
                "per_seq": [
                    _DeferredSeq(
                        self.events,
                        req_id,
                        101,
                        ready_after=int(self.ready_after.get(step_id, 0)),
                    )
                ],
                "forward_stats": {"component_us": {}},
            }
        return super().execute(batch)


def test_worker_server_depth1_finalizes_each_before_next_dispatch():
    # At depth 1 the oldest forward's deferred D2H is finalized before the next
    # forward is launched (no overlap window).
    events: list[str] = []
    server = _FifoServer(
        [
            _execute_req(1, 1, 1),
            _execute_req(2, 2, 1, pos_range=(1, 2), token_ids=(0,), token_source="last_sampled"),
            {"kind": "shutdown", "call_id": 3},
        ]
    )

    WorkerServer(
        _DeferredOverlapWorker(
            events,
            defer_steps={1},
            pipeline_depth=1,
        ),
        server,
    ).serve()

    assert events.index("finalize:1") < events.index("execute_step:2:defer=True")
    assert [resp["call_id"] for resp in server.responses] == [1, 2, 3]
    assert server.responses[0]["result"]["per_seq"][0]["sampled_token_id"] == 101
    component_us = server.responses[0]["result"]["forward_stats"]["component_us"]
    assert "worker_deferred_wait" in component_us
    assert "worker_result_finalize" in component_us


def test_worker_server_depth3_overlaps_multiple_deferred_finalizes():
    # At depth 3 the third forward launches before either earlier deferred D2H
    # finalizes; immediately-ready deferred results still leave in dispatch order.
    events: list[str] = []
    server = _FifoServer(
        [
            _execute_req(1, 1, 1),
            _execute_req(2, 2, 2, kind="prefill_und", pos_range=(0, 2), token_ids=(7, 8)),
            _execute_req(3, 3, 3, kind="prefill_und", pos_range=(0, 3), token_ids=(9, 10, 11)),
            {"kind": "shutdown", "call_id": 4},
        ]
    )

    WorkerServer(
        _DeferredOverlapWorker(
            events,
            defer_steps={1, 2, 3},
            pipeline_depth=3,
        ),
        server,
    ).serve()

    assert events.index("execute_step:3:defer=True") < events.index("finalize:1")
    assert events.index("execute_step:3:defer=True") < events.index("finalize:2")
    assert [resp["call_id"] for resp in server.responses] == [1, 2, 3, 4]


def test_worker_server_sends_ready_work_before_blocked_deferred_execute():
    events: list[str] = []
    server = _FifoServer(
        [
            _execute_req(1, 1, 1),
            {"kind": "get_metrics", "call_id": 2},
            _execute_req(3, 2, 2, kind="prefill_und", pos_range=(0, 2), token_ids=(7, 8)),
            {"kind": "shutdown", "call_id": 4},
        ]
    )

    WorkerServer(
        _DeferredOverlapWorker(
            events,
            defer_steps={1},
            ready_after={1: 3},
            pipeline_depth=4,
        ),
        server,
    ).serve()

    assert [resp["call_id"] for resp in server.responses] == [2, 3, 1, 4]
    assert [resp["kind"] for resp in server.responses] == ["metrics", "result", "result", "ok"]
    assert server.responses[2]["result"]["per_seq"][0]["sampled_token_id"] == 101
