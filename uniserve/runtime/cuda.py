"""CUDA driver results and captured-kernel context ownership."""

from __future__ import annotations

from typing import Any

import torch


class CUDAError(RuntimeError):
    """A CUDA resource or operation could not satisfy its requirements."""

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
        raise CUDAError("CUDA driver operations require cuda-python") from error
    return driver


def cuda_status(result: tuple[Any, ...], operation: str) -> None:
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
    raise CUDAError(
        f"{operation} failed: {name}: {message}", code=int(result[0])
    )


def cuda_value(result: tuple[Any, ...], operation: str) -> Any:
    """Extract the sole value from a successful CUDA driver result."""
    cuda_status(result, operation)
    return result[1]


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

    # Empty token partitions and copy-only computations are valid graphs.
    # Their lack of kernels does not violate device-context containment.
    return kernels
