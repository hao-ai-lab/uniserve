"""OOM-variant classification matrix and the to_wire scalar contract.

Focuses on behavior that ``tests/python/unit/runtime/test_controls.py`` does not
already cover: the matrix of realistic CUDA out-of-memory message variants (incl.
the cuBLAS allocation-failure string) classifying as ``WORKER_OOM`` with its
retryable/non-fatal policy, the type-hierarchy OOM path (``OutOfMemoryError``
subclasses), the negative cases that must *not* be treated as OOM, and the exact
scalar key set ``WorkerError.to_wire`` emits.
"""
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
def test_runtime_error_with_oom_marker_classifies_as_worker_oom(message):
    err = classify(RuntimeError(message))

    assert err.code == ErrorCode.WORKER_OOM
    assert err.code == "WorkerOOM"


@pytest.mark.parametrize("message", OOM_MESSAGE_VARIANTS)
def test_worker_oom_is_retryable_and_non_fatal(message):
    # WORKER_OOM is the retryable-but-non-fatal class: the host may resubmit the
    # offending op without tearing the worker process down.
    err = classify(RuntimeError(message))

    assert err.retryable is True
    assert err.fatal is False


def test_out_of_memory_error_type_classifies_as_oom_without_message_marker():
    # The structural path: any exception whose class-name contains
    # "OutOfMemory" is OOM even when the message carries no OOM text token.
    class OutOfMemoryError(RuntimeError):
        pass

    err = classify(OutOfMemoryError("workspace allocation request denied"))

    assert err.code == ErrorCode.WORKER_OOM
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

    assert err.code == ErrorCode.WORKER_OOM


def test_generic_runtime_error_classifies_as_model_execution_error():
    # A RuntimeError with no OOM/fatal-CUDA marker falls through to the default
    # taxonomy class, which is neither retryable nor fatal.
    err = classify(RuntimeError("kaboom: tensor shape mismatch in layer 3"))

    assert err.code == ErrorCode.MODEL_EXECUTION_ERROR
    assert err.code == "ModelExecutionError"
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
def test_non_matching_allocation_messages_are_model_execution_error(message):
    # Allocation-failure phrasings that do not contain a recognized OOM token
    # are deliberately NOT promoted to WORKER_OOM; they classify as the generic
    # model-execution class. Locks the matcher's actual token set.
    err = classify(RuntimeError(message))

    assert err.code == ErrorCode.MODEL_EXECUTION_ERROR
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


def test_to_wire_emits_only_the_five_scalar_fields():
    # to_wire models the Rust WorkerResponse: only kind/code/message/retryable/
    # fatal cross the wire. Rich local context (req_id/op_id/op_kind/details/
    # cleanup) must NOT appear.
    err = WorkerError(
        code=ErrorCode.WORKER_OOM,
        message="CUDA out of memory",
        retryable=True,
        fatal=False,
        cleanup=True,
        req_id=42,
        op_id=7,
        op_kind="decode_und",
        details={"device": 0},
    )

    wire = err.to_wire()

    assert set(wire) == {"kind", "code", "message", "retryable", "fatal"}
    assert wire["kind"] == "error"
    assert wire["code"] == "WorkerOOM"
    assert wire["message"] == "CUDA out of memory"
    assert wire["retryable"] is True
    assert wire["fatal"] is False


def test_to_wire_code_is_plain_str_not_enum():
    # ``code`` crosses as a plain str so the wire bytes are independent of
    # whether the WorkerError was built from an ErrorCode member or a raw string.
    from_enum = WorkerError(code=ErrorCode.WORKER_OOM, message="m").to_wire()
    from_str = WorkerError(code="WorkerOOM", message="m").to_wire()

    assert from_enum["code"] == "WorkerOOM"
    assert type(from_enum["code"]) is str
    assert from_enum == from_str


def test_to_wire_coerces_truthy_flags_to_bool():
    # to_wire normalizes retryable/fatal through bool(): non-bool truthy/falsey
    # inputs surface on the wire as real booleans.
    wire = WorkerError(code="WorkerOOM", message="m", retryable=1, fatal=0).to_wire()

    assert wire["retryable"] is True
    assert wire["fatal"] is False
