"""CUDA driver results and captured-kernel context ownership."""

from __future__ import annotations

from typing import Any

import torch


class CUDAError(RuntimeError):
    """A CUDA resource or call could not satisfy its requirements."""

    def __init__(self, message: str, *, code: int | None = None):
        super().__init__(message)
        self.code = code
        self.message = message


def driver() -> Any:
    """Import the CUDA driver bindings.

    Import and return the CUDA driver bindings required for explicit CUDA
    resources.
    """
    try:
        from cuda.bindings import (
            driver,  # pyright: ignore[reportAttributeAccessIssue]
        )
    except (
        ImportError
    ) as error:  # pragma: no cover - CUDA configurations install cuda-python.
        raise CUDAError("CUDA driver calls require cuda-python") from error
    return driver


def cuda_status(result: tuple[Any, ...], call: str) -> None:
    """Validate a CUDA driver result.

    Validate a CUDA driver result, raising CUDAError with the decoded
    failure.
    """
    cu = driver()
    if result[0] == cu.CUresult.CUDA_SUCCESS:
        return

    name_result = cu.cuGetErrorName(result[0])
    name = (
        str(name_result[1])
        if name_result[0] == cu.CUresult.CUDA_SUCCESS
        else str(result[0])
    )
    message_result = cu.cuGetErrorString(result[0])
    message = (
        str(message_result[1])
        if message_result[0] == cu.CUresult.CUDA_SUCCESS
        else name
    )
    raise CUDAError(f"{call} failed: {name}: {message}", code=int(result[0]))


def cuda_value(result: tuple[Any, ...], call: str) -> Any:
    """Extract the sole value from a successful CUDA driver result."""
    cuda_status(result, call)
    return result[1]


def create_sibling_stream(
    stream: torch.cuda.Stream, purpose: str
) -> tuple[Any, torch.cuda.ExternalStream]:
    """Create a driver stream in ``stream``'s CUDA context.

    The new stream shares the context of ``stream``, including its SM
    partition when that is a green context, at the same priority and without
    implicit synchronization against the legacy default stream. A driver
    stream is never one of PyTorch's pooled streams, so it is distinct from
    every stream the pool hands out. Returns the raw handle, which the caller
    destroys with ``destroy_stream``, and the wrapped stream PyTorch
    dispatches onto. ``purpose`` names the stream in failures.
    """
    with torch.cuda.device(stream.device):
        cu = driver()
        origin = cu.CUstream(stream.cuda_stream)
        flags = int(cu.CUstream_flags.CU_STREAM_NON_BLOCKING)
        green = cuda_value(
            cu.cuStreamGetGreenCtx(origin), f"query {purpose} stream context"
        )
        if int(green):
            raw = cuda_value(
                cu.cuGreenCtxStreamCreate(green, flags, stream.priority),
                f"create partitioned {purpose} stream",
            )
        else:
            context = cuda_value(
                cu.cuStreamGetCtx(origin), f"query {purpose} stream context"
            )
            cuda_status(
                cu.cuCtxPushCurrent(context), f"enter {purpose} context"
            )
            try:
                raw = cuda_value(
                    cu.cuStreamCreateWithPriority(flags, stream.priority),
                    f"create {purpose} stream",
                )
            finally:
                cuda_status(cu.cuCtxPopCurrent(), f"leave {purpose} context")

    return raw, torch.cuda.ExternalStream(int(raw), device=stream.device)


def destroy_stream(raw: Any, purpose: str) -> None:
    """Destroy a driver stream created by ``create_sibling_stream``."""
    cuda_status(driver().cuStreamDestroy(raw), f"destroy {purpose} stream")


def verify_graph_context(
    graph: torch.cuda.CUDAGraph, contexts: frozenset[int]
) -> int:
    """Verify kernels belong to the execution context's actual device bindings.

    Returns the number of captured kernel nodes; raises CUDAError when any
    kernel was captured against a CUDA context outside ``contexts``.
    """
    if not contexts:
        raise ValueError("CUDA graph verification requires its bound contexts")

    cu = driver()
    raw_graph = graph.raw_cuda_graph()
    nodes_result = cu.cuGraphGetNodes(cu.CUgraph(raw_graph), 1 << 20)
    cuda_status(nodes_result, "enumerate CUDA graph nodes")
    nodes = nodes_result[1]
    count = int(nodes_result[2])

    kernels = 0
    for node in nodes[:count]:
        node_type = cuda_value(
            cu.cuGraphNodeGetType(node), "query CUDA graph node type"
        )
        if node_type != cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            continue

        params = cuda_value(
            cu.cuGraphKernelNodeGetParams(node), "query kernel node context"
        )
        if int(params.ctx) not in contexts:
            name_result = (
                cu.cuFuncGetName(params.func)
                if int(params.func)
                else cu.cuKernelGetName(params.kern)
            )
            name = (
                name_result[1]
                if name_result[0] == cu.CUresult.CUDA_SUCCESS
                else "unknown"
            )
            raise CUDAError(
                f"captured compute node {name!r} escaped its owning context: "
                f"actual={int(params.ctx):#x}, "
                f"expected={sorted(hex(value) for value in contexts)}"
            )
        kernels += 1

    # Empty token partitions and copy-only call kinds are valid graphs.
    # Their lack of kernels does not violate device-context containment.
    return kernels
