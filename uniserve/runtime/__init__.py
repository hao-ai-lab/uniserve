"""Storage and executable ownership for direct numerical calls."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cuda import CUDAError
    from .cuda_graph import CUDAGraph, CUDAGraphError
    from .events import EventPool, EventPoolError
    from .execution import ExecutionContext, Scratch
    from .microbatches import Microbatches
    from .prefix_cache import PrefixCache
    from .process_groups import (
        ProcessGroups,
        Rendezvous,
        initialize_process_groups,
    )
    from .stream import CUDAStream, partition_streams
    from .tensor_buffers import TensorBuffers

__all__ = [
    "TensorBuffers",
    "PrefixCache",
    "ProcessGroups",
    "Rendezvous",
    "initialize_process_groups",
    "ExecutionContext",
    "Scratch",
    "Microbatches",
    "EventPool",
    "EventPoolError",
    "CUDAGraph",
    "CUDAGraphError",
    "CUDAError",
    "CUDAStream",
    "partition_streams",
]


def __getattr__(name):
    # Numerical modules can import borrowed bindings without recursively
    # constructing the storage/resource owner dependency graph.
    modules = {
        "TensorBuffers": "tensor_buffers",
        "PrefixCache": "prefix_cache",
        "ProcessGroups": "process_groups",
        "Rendezvous": "process_groups",
        "initialize_process_groups": "process_groups",
        "ExecutionContext": "execution",
        "Scratch": "execution",
        "Microbatches": "microbatches",
        "EventPool": "events",
        "EventPoolError": "events",
        "CUDAGraph": "cuda_graph",
        "CUDAGraphError": "cuda_graph",
        "CUDAError": "cuda",
        "CUDAStream": "stream",
        "partition_streams": "stream",
    }
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{modules[name]}"), name)
    globals()[name] = value
    return value
