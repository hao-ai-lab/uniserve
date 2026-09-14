"""CUDA driver results and captured-kernel context ownership."""

from __future__ import annotations

from typing import Any

import torch


class CudaError(RuntimeError):
    """A CUDA resource or operation could not satisfy its required contract."""


def driver() -> Any:
    """Import and return the CUDA driver bindings required for explicit CUDA resources."""

    try:
        from cuda.bindings import driver  # pyright: ignore[reportAttributeAccessIssue]
    except ImportError as error:  # pragma: no cover - CUDA configurations install cuda-python.
        raise CudaError("CUDA driver operations require cuda-python") from error
    return driver


def cuda_status(result: tuple[Any, ...], operation: str) -> None:
    """Validate a CUDA driver result and return its remaining values."""

    cu = driver()
    if result[0] == cu.CUresult.CUDA_SUCCESS:
        return
    name_result = cu.cuGetErrorName(result[0])
    name = str(name_result[1]) if name_result[0] == cu.CUresult.CUDA_SUCCESS else str(result[0])
    raise CudaError(f"{operation} failed: {name}")


def cuda_value(result: tuple[Any, ...], operation: str) -> Any:
    """Extract the sole value from a successful CUDA driver result."""

    cuda_status(result, operation)
    return result[1]


def verify_graph_context(graph: torch.cuda.CUDAGraph, expected_context: int | None) -> int:
    """Prove every kernel node in a captured graph belongs to the supplied CUDA context."""

    if expected_context is None:
        return 0
    cu = driver()
    raw_graph = graph.raw_cuda_graph()
    nodes_result = cu.cuGraphGetNodes(cu.CUgraph(raw_graph), 1 << 20)
    cuda_status(nodes_result, "enumerate CUDA graph nodes")
    nodes = nodes_result[1]
    count = int(nodes_result[2])
    kernels = 0
    for node in nodes[:count]:
        node_type = cuda_value(cu.cuGraphNodeGetType(node), "query CUDA graph node type")
        if node_type != cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            continue
        params = cuda_value(cu.cuGraphKernelNodeGetParams(node), "query kernel node context")
        if int(params.ctx) != int(expected_context):
            name_result = (
                cu.cuFuncGetName(params.func)
                if int(params.func)
                else cu.cuKernelGetName(params.kern)
            )
            name = name_result[1] if name_result[0] == cu.CUresult.CUDA_SUCCESS else "unknown"
            raise CudaError(
                f"captured compute node {name!r} escaped its owning context: "
                f"actual={int(params.ctx):#x}, expected={int(expected_context):#x}"
            )
        kernels += 1
    if kernels == 0:
        raise CudaError("captured CUDA graph contains no compute node")
    return kernels
