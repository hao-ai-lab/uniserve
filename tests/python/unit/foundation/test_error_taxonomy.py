"""Canonical worker error classification and snapshot behavior."""

from __future__ import annotations

import pytest

from uniserve.runtime import EventPoolError
from uniserve_worker.errors import WorkerError, WorkerErrorCode, classify
from uniserve_worker.protocol.identity import CallId

pytestmark = pytest.mark.unit


# Message variants that the production text/heuristic matcher actually
# treats as CUDA OOM (case-insensitive substring match against "out of
# storage", "cuda oom", "cublas_status_alloc_failed"). The cuBLAS
# allocation-failure string is the load-bearing "incl. cuBLAS variant" case.
OOM_MESSAGE_VARIANTS = [
    "CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 39.59 GiB total)",
    "CUDA error: out of memory",
    "RuntimeError: CUDA out of memory",
    "out of memory",
    "OUT OF MEMORY",  # matcher lowercases the message
    "CUDA OOM",
    "cuda oom while allocating attention workspace",
    "CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`",
    "cublas_status_alloc_failed",
]


@pytest.mark.parametrize("message", OOM_MESSAGE_VARIANTS)
def test_runtime_error_with_oom_marker_classifies_as_resource_error(message):
    err = classify(RuntimeError(message))

    assert err.code == WorkerErrorCode.RESOURCE_ERROR
    assert err.code == "ResourceError"
    assert err.fatal is False


def test_out_of_memory_error_type_classifies_as_oom_without_message_marker():
    # The structural path: any exception whose class-name contains
    # "OutOfMemory" is OOM even when the message carries no OOM text token.
    class OutOfMemoryError(RuntimeError):
        pass

    err = classify(OutOfMemoryError("workspace allocation request denied"))

    assert err.code == WorkerErrorCode.RESOURCE_ERROR
    assert err.fatal is False


def test_oom_detection_walks_class_hierarchy_for_subclasses():
    # Detection walks the MRO, so a subclass of an OutOfMemory* type is still
    # OOM even though its own leaf name does not contain the token.
    class OutOfMemoryError(RuntimeError):
        pass

    # The private test type deliberately follows the error taxonomy.
    class DeviceAllocFailure(OutOfMemoryError):  # noqa: N818
        pass

    err = classify(DeviceAllocFailure("alloc denied"))

    assert err.code == WorkerErrorCode.RESOURCE_ERROR


def test_generic_runtime_error_classifies_as_compute_error():
    err = classify(RuntimeError("kaboom: tensor shape mismatch in layer 3"))

    assert err.code == WorkerErrorCode.COMPUTE_ERROR
    assert err.code == "ComputeError"
    assert err.fatal is False


def test_event_pool_error_classifies_as_fatal_invariant_violation():
    err = classify(
        EventPoolError("device event reference accounting is invalid")
    )

    assert err.code == WorkerErrorCode.INVARIANT_VIOLATION
    assert err.fatal is True


@pytest.mark.parametrize(
    "message",
    [
        # cuDNN alloc failure is NOT matched as OOM
        "CUDNN_STATUS_ALLOC_FAILED",
        "cudnn allocation failed",
        "cudaMalloc returned an error",
        "failed to allocate device buffer",
        "allocation failure in pinned host pool",
    ],
)
def test_non_matching_allocation_messages_are_compute_error(message):
    err = classify(RuntimeError(message))

    assert err.code == WorkerErrorCode.COMPUTE_ERROR


def test_fatal_cuda_marker_takes_precedence_over_oom_in_same_message():
    # Rule ordering is load-bearing: a context-corrupting CUDA fault is FATAL
    # even when the same message also mentions "out of memory".
    err = classify(
        RuntimeError(
            "an illegal memory access was encountered; CUDA out of memory"
        )
    )

    assert err.code == WorkerErrorCode.FATAL_WORKER_FAILURE
    assert err.fatal is True


def test_to_mapping_emits_canonical_error_context():
    err = WorkerError(
        code=WorkerErrorCode.RESOURCE_ERROR,
        message="CUDA out of memory",
        fatal=False,
        req_id=42,
        call_id=CallId(7, 0),
        call_kind="decode_und",
        phase="run",
        route="language",
        calls=((5, 42, 3, CallId(7, 0)),),
        details={"device": 0},
    )

    snapshot = err.to_mapping()

    assert snapshot["code"] == "ResourceError"
    assert snapshot["message"] == "CUDA out of memory"
    assert snapshot["fatal"] is False
    assert snapshot["phase"] == "run"
    assert snapshot["route"] == "language"
    assert snapshot["calls"] == [
        {
            "request_key": {
                "engine_id": 5,
                "request_id": 42,
                "request_epoch": 3,
            },
            "call_id": {"batch_id": 7, "request_index": 0},
        }
    ]
