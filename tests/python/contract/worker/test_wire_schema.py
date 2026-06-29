"""Python-side wire schema pins for the Rust worker protocol."""
from __future__ import annotations

from pathlib import Path

import pytest

from uniserve_worker.contracts.caps import validate_caps
from uniserve_worker.foundation.errors import ErrorCode, WorkerError
from uniserve_worker.server.stub import StubEngine

pytestmark = pytest.mark.contract


def test_caps_error_and_flatbuffers_schema_are_pinned():
    caps = validate_caps(StubEngine().caps(), owner="StubEngine").to_wire()
    assert set(caps) == {
        "block_size",
        "num_blocks",
        "num_layers",
        "scratch_capacity_tokens",
        "supported_ops",
        "max_latent_size",
        "latent_downsample",
        "bytes_per_token",
        "groups",
        "kv_dtype",
        "attention_backend",
        "quantization",
        "rank",
        "pipeline_depth",
        "encoder_cache_budget",
        "supported_controls",
        "adapter_mode",
        "execution_constraints",
        "resource_classes",
    }
    # pipeline_depth is a transport-level pinned field; the wire form must always
    # be a concrete positive int (defaulting to 1), never None/optional.
    assert isinstance(caps["pipeline_depth"], int)
    assert not isinstance(caps["pipeline_depth"], bool)
    assert caps["pipeline_depth"] >= 1

    error = WorkerError(
        code=ErrorCode.CAPABILITY_MISMATCH,
        message="declared capability is missing",
    ).to_wire()
    assert set(error) == {"kind", "code", "message", "retryable", "fatal"}
    assert error["code"] == "CapabilityMismatch"

    schema = Path("crates/protocol/worker-ipc-core/schema/worker.fbs").read_text()
    assert "namespace uniserve.wire;" in schema
    assert "enum OpKind" in schema
    assert "table WorkerRequest" in schema
    assert "table WorkerResponse" in schema
    assert "root_type WorkerRequest;" in schema
    # Pin the transport-framing + caps wire elements the IPC bridge relies on so a
    # rename of the request/response correlation id or pipeline-depth field is
    # caught here, where there is otherwise no behavioural IPC test.
    assert "table EngineCaps" in schema
    assert "pipeline_depth:uint;" in schema
    assert "call_id:ulong = null;" in schema


def test_native_ipc_transport_bridge_surface_is_pinned():
    """Pin the Python IPC transport bridge contract.

    There is no behavioural IPC/shm-ring test on the Python side (the native
    transport lives in Rust), so at minimum pin that the
    ``uniserve_worker.server.ipc.Server`` bridge exists and exposes its expected
    recv/respond transport surface. We do not construct a Server here because
    that allocates real OS resources (shared memory / sockets); this is purely a
    surface contract pin.
    """
    try:
        from uniserve_worker.server import ipc
    except ImportError:
        pytest.skip("uniserve_worker.server.ipc native extension is not built")

    server = ipc.Server
    for method in ("recv", "respond", "try_recv"):
        assert callable(getattr(server, method)), f"Server.{method} missing"
