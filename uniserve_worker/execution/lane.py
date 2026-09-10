"""Immutable CUDA execution-lane resources."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import torch

from uniserve_worker.config import LaneConfig
from uniserve_worker.execution.batch import Domain


class ExecutionLaneError(RuntimeError):
    """A worker_config-static execution lane could not be realized exactly."""


@dataclass(slots=True)
class _GreenContext:
    """Owns one CUDA Green Context, its stream, resource partition, and component identity."""

    lane: LaneConfig
    device: torch.device
    sm_count: int
    green: Any
    context: Any
    raw_stream: Any
    stream: torch.cuda.ExternalStream
    component_path: tuple[int, ...]


class ExecutionLaneRuntime:
    """Own streams, buffers, graphs, and events for one physical lane."""

    def __init__(
        self,
        *,
        lane: LaneConfig | None,
        device: torch.device,
        stream: torch.cuda.Stream | None,
        sm_count: int,
        buffer: Any,
        graphs: Any,
        green: _GreenContext | None = None,
        event_slots: int = 2,
    ) -> None:
        """Own one lane's stream, context, staging buffers, graph catalog, and output events."""

        if int(event_slots) < 1:
            raise ValueError("execution lane requires a positive event bound")
        self.lane = lane
        self.device = device
        self.stream = stream
        self.sm_count = int(sm_count)
        self.buffer = buffer
        self.graphs = graphs
        self._green = green
        self._closed = False
        self._event_cursor = 0
        if device.type == "cuda":
            self._ingress_events = tuple(
                torch.cuda.Event(blocking=False) for _ in range(int(event_slots))
            )
            self._output_events = tuple(
                torch.cuda.Event(blocking=False) for _ in range(int(event_slots))
            )
        else:
            self._ingress_events = ()
            self._output_events = ()

    @property
    def full_device(self) -> bool:
        """Whether kernels may address the device's complete physical SM domain."""

        return self._green is None

    @property
    def lane_id(self) -> str | None:
        """Expose the scheduler lane identifier, or ``None`` for the default stream."""

        return None if self.lane is None else self.lane.lane_id

    @property
    def domains(self) -> tuple[Domain, ...]:
        """List domains admitted by the bound lane, or all domains on the default stream."""

        return tuple(Domain) if self.lane is None else self.lane.domains

    def verify_stream(self) -> None:
        """Verify that the active CUDA stream and context still match the lane binding."""

        if self._green is None:
            return
        cu = _driver()
        associated = _cuda_value(
            cu.cuStreamGetGreenCtx(self._green.raw_stream), "query stream lane"
        )
        if int(associated) != int(self._green.green):
            raise ExecutionLaneError("lane origin stream is detached from its Green Context")
        resource = _cuda_value(
            cu.cuGreenCtxGetDevResource(
                self._green.green,
                cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
            ),
            "query Green Context SM resource",
        )
        if int(resource.sm.smCount) != self.sm_count:
            raise ExecutionLaneError("lane Green Context SM resource changed after startup")

    def order_after(self, producer: torch.cuda.Stream) -> None:
        """Make the lane stream wait for work enqueued on a producer stream."""

        if self.stream is None or int(producer.cuda_stream) == int(self.stream.cuda_stream):
            return
        event = self._ingress_events[self._event_cursor % len(self._ingress_events)]
        event.record(producer)
        self.stream.wait_event(event)

    def record_output(self) -> torch.cuda.Event | None:
        """Record an event after all currently enqueued lane output work."""

        if self.stream is None:
            return None
        event = self._output_events[self._event_cursor % len(self._output_events)]
        self._event_cursor += 1
        event.record(self.stream)
        return event

    def close(self) -> None:
        """Release the lane stream and CUDA Green Context resources."""

        if self._closed:
            return
        self._closed = True
        if self.stream is not None:
            self.stream.synchronize()
        self.graphs.close()
        self.buffer.close()
        self.buffer = None
        self.graphs = None
        self._ingress_events = ()
        self._output_events = ()
        gc.collect()
        if self._green is None:
            return
        cu = _driver()
        _cuda_status(cu.cuStreamDestroy(self._green.raw_stream), "destroy lane stream")
        _cuda_status(cu.cuGreenCtxDestroy(self._green.green), "destroy Green Context")
        self.stream = None


def create_green_contexts(
    lanes: tuple[LaneConfig, ...],
    device: torch.device,
) -> tuple[_GreenContext, ...]:
    """Realize exact, disjoint Green Context resources from one split tree."""

    if not lanes:
        return ()
    if device.type != "cuda":
        raise ExecutionLaneError("execution lanes require a CUDA device")
    if len({lane.lane_id for lane in lanes}) != len(lanes):
        raise ExecutionLaneError("lane ids must be unique")
    bound_domains: set[Domain] = set()
    for lane in lanes:
        overlap = bound_domains.intersection(lane.domains)
        if overlap:
            names = ", ".join(sorted(value.value for value in overlap))
            raise ExecutionLaneError(f"execution domains have multiple lane bindings: {names}")
        bound_domains.update(lane.domains)

    torch.cuda.init()
    index = device.index if device.index is not None else torch.cuda.current_device()
    cu = _driver()
    _cuda_status(cu.cuInit(0), "initialize CUDA driver")
    cuda_device = _cuda_value(cu.cuDeviceGet(index), "resolve CUDA device")
    full = _cuda_value(
        cu.cuDeviceGetDevResource(cuda_device, cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM),
        "query device SM resource",
    )
    requested = sum(int(lane.sm_budget) for lane in lanes)
    available = int(full.sm.smCount)
    if requested > available:
        raise ExecutionLaneError(
            f"lane SM budgets require {requested} SMs but the device exposes {available}"
        )

    aggregate, _unassigned = _split_one(cu, full, requested)
    aggregate_green = _green_from_resources(cu, cuda_device, (aggregate,))
    intermediate = [aggregate_green]
    current_resource = _green_resource(cu, aggregate_green)
    realized: list[_GreenContext] = []
    try:
        for ordinal, lane in enumerate(lanes):
            if ordinal + 1 == len(lanes):
                lane_resource = current_resource
                remainder = None
            else:
                lane_resource, remainder = _split_one(cu, current_resource, int(lane.sm_budget))
            lane_green = _green_from_resources(cu, cuda_device, (lane_resource,))
            try:
                context = _cuda_value(cu.cuCtxFromGreenCtx(lane_green), "resolve lane context")
                raw_stream = _cuda_value(
                    cu.cuGreenCtxStreamCreate(
                        lane_green,
                        int(cu.CUstream_flags.CU_STREAM_NON_BLOCKING),
                        0,
                    ),
                    "create lane origin stream",
                )
                resolved = _green_resource(cu, lane_green)
                sm_count = int(resolved.sm.smCount)
                if sm_count != int(lane.sm_budget):
                    raise ExecutionLaneError(
                        f"lane {lane.lane_id!r} resolved {sm_count} SMs, expected {lane.sm_budget}"
                    )
                associated = _cuda_value(
                    cu.cuStreamGetGreenCtx(raw_stream),
                    "query lane origin stream",
                )
                if int(associated) != int(lane_green):
                    raise ExecutionLaneError("lane origin stream has the wrong Green Context")
                realized.append(
                    _GreenContext(
                        lane=lane,
                        device=device,
                        sm_count=sm_count,
                        green=lane_green,
                        context=context,
                        raw_stream=raw_stream,
                        stream=torch.cuda.ExternalStream(int(raw_stream), device=device),
                        component_path=tuple(range(ordinal + 1)),
                    )
                )
            except Exception:
                _cuda_status(cu.cuGreenCtxDestroy(lane_green), "destroy rejected lane context")
                raise
            if remainder is not None:
                remainder_green = _green_from_resources(cu, cuda_device, (remainder,))
                intermediate.append(remainder_green)
                current_resource = _green_resource(cu, remainder_green)
        if sum(item.sm_count for item in realized) != requested:
            raise ExecutionLaneError("lane SM resources do not form the configured disjoint total")
        return tuple(realized)
    except Exception:
        for item in reversed(realized):
            _cuda_status(cu.cuStreamDestroy(item.raw_stream), "destroy rejected lane stream")
            _cuda_status(cu.cuGreenCtxDestroy(item.green), "destroy rejected lane context")
        raise
    finally:
        for parent in reversed(intermediate):
            _cuda_status(cu.cuGreenCtxDestroy(parent), "destroy lane split context")


def verify_graph_context(graph: torch.cuda.CUDAGraph, expected_context: int | None) -> int:
    """Prove every kernel node in a captured graph belongs to its lane."""

    if expected_context is None:
        return 0
    cu = _driver()
    raw_graph = graph.raw_cuda_graph()
    nodes_result = cu.cuGraphGetNodes(cu.CUgraph(raw_graph), 1 << 20)
    _cuda_status(nodes_result, "enumerate CUDA graph nodes")
    nodes = nodes_result[1]
    count = int(nodes_result[2])
    kernels = 0
    for node in nodes[:count]:
        node_type = _cuda_value(cu.cuGraphNodeGetType(node), "query CUDA graph node type")
        if node_type != cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            continue
        params = _cuda_value(cu.cuGraphKernelNodeGetParams(node), "query kernel node context")
        if int(params.ctx) != int(expected_context):
            raise ExecutionLaneError("captured compute node escaped its owning context")
        kernels += 1
    if kernels == 0:
        raise ExecutionLaneError("captured CUDA graph contains no compute node")
    return kernels


def _split_one(cu: Any, resource: Any, count: int) -> tuple[Any, Any]:
    """Split one resource between green and default execution lanes."""

    result = cu.cuDevSmResourceSplitByCount(1, resource, 0, int(count))
    _cuda_status(result, "split SM resource")
    groups, group_count, remainder = result[1], int(result[2]), result[3]
    if group_count != 1 or not groups:
        raise ExecutionLaneError("CUDA could not realize the requested SM resource")
    group = groups[0]
    if int(group.sm.smCount) != int(count):
        raise ExecutionLaneError(
            f"CUDA rounded an exact SM request from {count} to {int(group.sm.smCount)}"
        )
    return group, remainder


def _green_from_resources(cu: Any, device: Any, resources: tuple[Any, ...]) -> Any:
    """Construct a green context from split device resources when supported."""

    descriptor = _cuda_value(
        cu.cuDevResourceGenerateDesc(list(resources), len(resources)),
        "generate Green Context resource descriptor",
    )
    return _cuda_value(
        cu.cuGreenCtxCreate(
            descriptor,
            device,
            int(cu.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM),
        ),
        "create Green Context",
    )


def _green_resource(cu: Any, green: Any) -> Any:
    """Extract the green-partition handle returned by a CUDA split operation."""

    return _cuda_value(
        cu.cuGreenCtxGetDevResource(green, cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM),
        "query Green Context SM resource",
    )


def _driver() -> Any:
    """Import and return the CUDA driver bindings required for green contexts."""

    try:
        from cuda.bindings import driver  # pyright: ignore[reportAttributeAccessIssue]
    except ImportError as error:  # pragma: no cover - CUDA configurations install cuda-python.
        raise ExecutionLaneError("CUDA execution lanes require cuda-python") from error
    return driver


def _cuda_status(result: tuple[Any, ...], operation: str) -> None:
    """Validate a CUDA driver result and return its remaining values."""

    cu = _driver()
    if result[0] == cu.CUresult.CUDA_SUCCESS:
        return
    name_result = cu.cuGetErrorName(result[0])
    name = str(name_result[1]) if name_result[0] == cu.CUresult.CUDA_SUCCESS else str(result[0])
    raise ExecutionLaneError(f"{operation} failed: {name}")


def _cuda_value(result: tuple[Any, ...], operation: str) -> Any:
    """Extract the sole value from a successful CUDA driver result."""

    _cuda_status(result, operation)
    return result[1]


__all__ = [
    "ExecutionLaneError",
    "ExecutionLaneRuntime",
    "LaneConfig",
    "create_green_contexts",
    "verify_graph_context",
]
