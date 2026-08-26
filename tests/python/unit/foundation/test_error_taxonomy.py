"""Canonical worker error classification and wire behavior."""
from __future__ import annotations

import pytest

from uniserve_worker.foundation.errors import ErrorCode, WorkerError, classify

pytestmark = pytest.mark.unit


# Message variants that the production text/heuristic matcher actually treats as
# CUDA OOM (case-insensitive substring match against "out of memory", "cuda oom",
# "cublas_status_alloc_failed"). The cuBLAS allocation-failure string is the
# load-bearing "incl. cuBLAS variant" case.
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

    assert err.code == ErrorCode.RESOURCE_ERROR
    assert err.code == "ResourceError"
    assert err.retryable is True
    assert err.fatal is False


def test_out_of_memory_error_type_classifies_as_oom_without_message_marker():
    # The structural path: any exception whose class-name contains
    # "OutOfMemory" is OOM even when the message carries no OOM text token.
    class OutOfMemoryError(RuntimeError):
        pass

    err = classify(OutOfMemoryError("workspace allocation request denied"))

    assert err.code == ErrorCode.RESOURCE_ERROR
    assert err.retryable is True
    assert err.fatal is False


def test_oom_detection_walks_class_hierarchy_for_subclasses():
    # Detection walks the MRO, so a subclass of an OutOfMemory* type is still
    # OOM even though its own leaf name does not contain the token.
    class OutOfMemoryError(RuntimeError):
        pass

    class DeviceAllocFailure(OutOfMemoryError):
        pass

    err = classify(DeviceAllocFailure("alloc denied"))

    assert err.code == ErrorCode.RESOURCE_ERROR


def test_generic_runtime_error_classifies_as_compute_error():
    err = classify(RuntimeError("kaboom: tensor shape mismatch in layer 3"))

    assert err.code == ErrorCode.COMPUTE_ERROR
    assert err.code == "ComputeError"
    assert err.retryable is False
    assert err.fatal is False


@pytest.mark.parametrize(
    "message",
    [
        "CUDNN_STATUS_ALLOC_FAILED",  # cuDNN alloc failure is NOT matched as OOM
        "cudnn allocation failed",
        "cudaMalloc returned an error",
        "failed to allocate device buffer",
        "allocation failure in pinned host pool",
    ],
)
def test_non_matching_allocation_messages_are_compute_error(message):
    err = classify(RuntimeError(message))

    assert err.code == ErrorCode.COMPUTE_ERROR
    assert err.retryable is False


def test_fatal_cuda_marker_takes_precedence_over_oom_in_same_message():
    # Rule ordering is load-bearing: a context-corrupting CUDA fault is FATAL
    # even when the same message also mentions "out of memory".
    err = classify(
        RuntimeError("an illegal memory access was encountered; CUDA out of memory")
    )

    assert err.code == ErrorCode.FATAL_WORKER_FAILURE
    assert err.fatal is True
    assert err.retryable is False


def test_to_wire_emits_canonical_error_context():
    err = WorkerError(
        code=ErrorCode.RESOURCE_ERROR,
        message="CUDA out of memory",
        retryable=True,
        fatal=False,
        req_id=42,
        op_id=7,
        op_kind="decode_und",
        phase="run",
        route="language",
        operations=((42, 3, 7),),
        details={"device": 0},
    )

    wire = err.to_mapping()

    assert wire["kind"] == "error"
    assert wire["code"] == "ResourceError"
    assert wire["message"] == "CUDA out of memory"
    assert wire["retryable"] is True
    assert wire["fatal"] is False
    assert wire["phase"] == "run"
    assert wire["route"] == "language"
    assert wire["operations"] == [{"session_id": 42, "epoch": 3, "op_id": 7}]


def test_to_wire_coerces_truthy_flags_to_bool():
    # to_wire normalizes retryable/fatal through bool(): non-bool truthy/falsey
    # inputs surface on the wire as real booleans.
    wire = WorkerError(code="ResourceError", message="m", retryable=1, fatal=0).to_mapping()

    assert wire["retryable"] is True
    assert wire["fatal"] is False
