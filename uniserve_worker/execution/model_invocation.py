"""Direct staged execution of concrete model phases."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
from torch import nn

from uniserve_worker.batch import Domain
from uniserve_worker.execution.cuda_graph import CudaGraphRunner, GraphExecutionError
from uniserve_worker.execution.forward_batch import (
    EmptyKvView,
    EmptyMeshView,
    ForwardBatch,
    ForwardOutput,
    KvView,
    MeshView,
    ModelPhase,
    TokenSelection,
)
from uniserve_worker.execution.trace import (
    ExecutionPhase,
    ExecutionTrace,
    OperationTrace,
)
from uniserve_worker.foundation.errors import (
    ComputeError,
    ErrorCode,
    InputError,
    ResourceError,
    WorkerError,
    classify,
)

from .forward_batch import AttnPlan
from .input_buffers import InputBuffers
from .lane import ExecutionPartitionRuntime, LaneConfig, create_green_contexts

logger = logging.getLogger(__name__)


class RunPath(StrEnum):
    EAGER = "eager"
    GRAPH_CAPTURE = "graph_capture"
    GRAPH_REPLAY = "graph_replay"
    GRAPH_FALLBACK = "graph_fallback"


@dataclass(frozen=True, slots=True)
class RunObservation:
    route: str
    row_count: int
    row_kind_counts: tuple[tuple[str, int], ...]
    path: RunPath
    duration_us: int
    graph_unpadded_tokens: int
    graph_padded_tokens: int


class _ConcretePhase(nn.Module):
    """Invoke the one concrete method family implemented by a loaded model."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        batch: ForwardBatch,
    ) -> ForwardOutput:
        model: Any = self.model
        if batch.phase in {ModelPhase.TEXT, ModelPhase.DENOISE}:
            hidden = model(input_ids, positions, batch)
            if not isinstance(hidden, torch.Tensor):
                raise TypeError("model text/denoise forward must return a tensor")
            result = model.project(hidden, batch)
        elif batch.phase is ModelPhase.ENCODE_VISION:
            result = model.encode(batch.encode_pixels, batch)
        elif batch.phase is ModelPhase.ENCODE_LATENT:
            result = model.encode_latent(batch.encode_pixels, batch)
        elif batch.phase is ModelPhase.DECODE_LATENT:
            result = model.decode_latent(batch.decode_latents, batch)
        else:
            raise TypeError(f"unsupported model phase {batch.phase.value!r}")
        if not isinstance(result, ForwardOutput):
            raise TypeError("concrete model phase must return ForwardOutput")
        return result


class _ModelInvocation:
    """Own fixed input buffers and invoke concrete model methods."""

    def __init__(
        self,
        model: nn.Module,
        graph_runner: CudaGraphRunner | None,
        trace: ExecutionTrace,
        *,
        max_rows: int,
        max_tokens: int,
        max_text_tokens: int | None = None,
        max_blocks_per_row: int,
        hidden_size: int,
        devices: tuple[torch.device | str, ...],
        lanes: tuple[LaneConfig, ...] = (),
        max_inflight: int = 1,
        graph_factory: Callable[
            [torch.device, LaneConfig | None, torch.cuda.Stream | None, int | None],
            CudaGraphRunner,
        ]
        | None = None,
    ) -> None:
        if type(model).forward is nn.Module.forward:
            raise TypeError("runner model must implement forward(input_ids, positions, batch)")
        canonical = tuple(dict.fromkeys(str(torch.device(device)) for device in devices))
        if not canonical:
            raise ValueError("model runner requires an execution device")
        self.model = model
        self._phase = _ConcretePhase(model)
        self.trace = trace
        self.uses_lanes = bool(lanes)
        self._partitions: dict[tuple[str, Domain], ExecutionPartitionRuntime] = {}
        self._owned_partitions: list[ExecutionPartitionRuntime] = []

        def make_buffer(device: str) -> InputBuffers:
            return InputBuffers(
                max_rows=max_rows,
                max_tokens=max_tokens,
                max_text_tokens=max_text_tokens,
                max_blocks_per_row=max_blocks_per_row,
                hidden_size=hidden_size,
                device=device,
            )

        if lanes:
            if len(canonical) != 1:
                raise ValueError("Green Context lanes require one physical CUDA device")
            if graph_factory is None:
                raise ValueError("lane deployment requires partition-local graph runners")
            device = torch.device(canonical[0])
            for green in create_green_contexts(lanes, device):
                with torch.cuda.stream(green.stream):
                    buffer = make_buffer(str(device))
                graphs = graph_factory(device, green.lane, green.stream, int(green.context))
                event_slots = int(green.lane.max_inflight or max_inflight) + 1
                partition = ExecutionPartitionRuntime(
                    lane=green.lane,
                    device=device,
                    stream=green.stream,
                    sm_count=green.sm_count,
                    buffer=buffer,
                    graphs=graphs,
                    green=green,
                    event_slots=event_slots,
                )
                partition.verify_stream()
                self._owned_partitions.append(partition)
                for domain in green.lane.domains:
                    self._partitions[(str(device), domain)] = partition
            missing = set(Domain).difference(domain for lane in lanes for domain in lane.domains)
            if missing:
                names = ", ".join(sorted(domain.value for domain in missing))
                raise ValueError(f"lane deployment has no binding for domains: {names}")
        else:
            if graph_factory is None and len(canonical) != 1:
                raise ValueError("multiple execution devices require partition-local graph runners")
            for device_name in canonical:
                device = torch.device(device_name)
                stream: torch.cuda.Stream | None = None
                context = nullcontext() if stream is None else torch.cuda.stream(stream)
                with context:
                    buffer = make_buffer(device_name)
                graphs = (
                    graph_factory(device, None, stream, None)
                    if graph_factory is not None
                    else _claim_graph_runner(graph_runner, stream, None)
                )
                sm_count = (
                    int(torch.cuda.get_device_properties(device).multi_processor_count)
                    if device.type == "cuda"
                    else 0
                )
                partition = ExecutionPartitionRuntime(
                    lane=None,
                    device=device,
                    stream=stream,
                    sm_count=sm_count,
                    buffer=buffer,
                    graphs=graphs,
                    event_slots=int(max_inflight) + 1,
                )
                self._owned_partitions.append(partition)
                for domain in Domain:
                    self._partitions[(device_name, domain)] = partition
        self._last_observation: RunObservation | None = None
        self._last_request_pool_indices: torch.Tensor | None = None
        self._last_output_event: torch.cuda.Event | None = None

    @property
    def last_observation(self) -> RunObservation | None:
        return self._last_observation

    @property
    def last_request_pool_indices(self) -> torch.Tensor | None:
        return self._last_request_pool_indices

    @property
    def last_output_event(self) -> torch.cuda.Event | None:
        return self._last_output_event

    @property
    def buffers(self) -> tuple[InputBuffers, ...]:
        return tuple(partition.buffer for partition in self._owned_partitions)

    @property
    def partitions(self) -> tuple[ExecutionPartitionRuntime, ...]:
        return tuple(self._owned_partitions)

    def partition_identity(self, device: torch.device | str, domain: Domain) -> int:
        partition = self._partitions.get((str(torch.device(device)), domain))
        if partition is None:
            raise InputError(
                f"model runner has no {domain.value!r} execution partition for {device}",
                phase="partition_selection",
                route=domain.value,
            )
        return id(partition)

    def complete_startup(self) -> None:
        for partition in self._owned_partitions:
            lane_id = partition.lane_id or "default"
            logger.info("verifying CUDA graph catalog lane=%s", lane_id)
            try:
                partition.graphs.complete_startup()
            except GraphExecutionError as error:
                raise GraphExecutionError(
                    f"execution partition {lane_id} failed startup"
                ) from error
            partition.verify_stream()
            logger.info("verified CUDA graph catalog lane=%s", lane_id)
        signature = tuple(
            (
                partition.lane_id,
                partition.sm_count,
                tuple(domain.value for domain in partition.domains),
                partition.graphs.startup_signature,
            )
            for partition in self._owned_partitions
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            logger.info("verifying tensor-parallel execution partition agreement")
            gathered: list[object] = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, signature)
            if any(value != signature for value in gathered):
                raise GraphExecutionError("tensor-parallel execution partitions disagree")
            logger.info("verified tensor-parallel execution partition agreement")

    def close(self) -> None:
        self._last_request_pool_indices = None
        self._last_output_event = None
        for partition in reversed(self._owned_partitions):
            partition.close()
        self._partitions.clear()
        self._owned_partitions.clear()

    def synchronize(self) -> None:
        for partition in self._owned_partitions:
            if partition.stream is not None:
                partition.stream.synchronize()

    def run(
        self,
        tasks: tuple[Any, ...],
        *,
        device: torch.device | str,
        kv: KvView | EmptyKvView,
        attention: AttnPlan,
        mesh: MeshView | EmptyMeshView,
        graph_shape: tuple[object, ...],
        graph_eligible: bool,
        domain: Domain | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if not tasks:
            raise ValueError("model runner received an empty call")
        self._last_request_pool_indices = None
        self._last_output_event = None
        started = time.perf_counter_ns()
        target = torch.device(device)
        phases = frozenset(task.phase for task in tasks)
        if phases <= {ModelPhase.TEXT, ModelPhase.DENOISE}:
            phase = ModelPhase.DENOISE if ModelPhase.DENOISE in phases else ModelPhase.TEXT
        elif len(phases) == 1:
            phase = next(iter(phases))
        else:
            raise ValueError("one model call cannot mix unrelated execution phases")
        operations = tuple(
            OperationTrace(
                task.operation.request_key.session_id,
                task.operation.request_key.epoch,
                task.operation.op_id,
                _base_version(task.operation),
            )
            for task in tasks
        )
        task_domains = frozenset(
            task_domain
            for task in tasks
            for task_domain in (getattr(task.operation, "domain", None),)
            if task_domain is not None
        )
        if domain is None and len(task_domains) > 1:
            raise ValueError("one unbound model call must contain one execution domain")
        resolved_domain = domain or (
            next(iter(task_domains))
            if task_domains
            else Domain.FLOW
            if phase is ModelPhase.DENOISE
            else Domain.PREFILL
        )
        partition = self._partitions.get((str(target), resolved_domain))
        if partition is None:
            raise InputError(
                f"model runner has no {resolved_domain.value!r} execution partition for {target}",
                phase="input_staging",
                route=phase.value,
                operations=tuple((item.session_id, item.epoch, item.op_id) for item in operations),
            )
        buffers = partition.buffer
        counts = _kind_counts(tasks)
        try:
            if partition.stream is not None:
                partition.order_after(torch.cuda.current_stream(target))
            stream_context = (
                nullcontext() if partition.stream is None else torch.cuda.stream(partition.stream)
            )
            with stream_context:
                batch = buffers.stage(
                    phase=phase,
                    row_count=len(tasks),
                    request_pool_indices=tuple(task.request.request_pool_idx for task in tasks),
                    token_row_indices=tuple(
                        index for index, task in enumerate(tasks) if task.token_ids is not None
                    ),
                    token_ids=tuple(task.token_ids for task in tasks if task.token_ids is not None),
                    token_embeddings=tuple(
                        task.token_embeddings for task in tasks if task.token_ids is not None
                    ),
                    token_embedding_masks=tuple(
                        task.token_embedding_mask for task in tasks if task.token_ids is not None
                    ),
                    token_positions=tuple(
                        task.positions for task in tasks if task.token_ids is not None
                    ),
                    token_selections=tuple(
                        task.selection for task in tasks if task.token_ids is not None
                    ),
                    flow_row_indices=tuple(
                        index
                        for index, task in enumerate(tasks)
                        if task.latent is not None and task.image_tokens > 0
                    ),
                    flow_positions=tuple(
                        task.positions
                        for task in tasks
                        if task.latent is not None and task.image_tokens > 0
                    ),
                    flow_timesteps=tuple(
                        task.timestep
                        for task in tasks
                        if task.latent is not None and task.image_tokens > 0
                    ),
                    flow_latents=tuple(
                        task.latent
                        for task in tasks
                        if task.latent is not None and task.image_tokens > 0
                    ),
                    flow_conditioning=tuple(
                        task.flow_conditioning
                        for task in tasks
                        if task.latent is not None and task.image_tokens > 0
                    ),
                    flow_image_tokens=tuple(
                        task.image_tokens
                        for task in tasks
                        if task.latent is not None and task.image_tokens > 0
                    ),
                    flow_heights=tuple(
                        task.image_height
                        for task in tasks
                        if task.latent is not None and task.image_tokens > 0
                    ),
                    flow_widths=tuple(
                        task.image_width
                        for task in tasks
                        if task.latent is not None and task.image_tokens > 0
                    ),
                    encode_pixels=tuple(
                        task.encode_pixels for task in tasks if task.encode_pixels is not None
                    ),
                    encode_grids=tuple(
                        task.encode_grid for task in tasks if task.encode_pixels is not None
                    ),
                    encode_grid_shapes=tuple(
                        task.encode_grid_shape for task in tasks if task.encode_pixels is not None
                    ),
                    decode_latents=tuple(
                        task.latent
                        for task in tasks
                        if task.latent is not None and task.image_tokens == 0
                    ),
                    decode_heights=tuple(
                        task.image_height
                        for task in tasks
                        if task.latent is not None and task.image_tokens == 0
                    ),
                    decode_widths=tuple(
                        task.image_width
                        for task in tasks
                        if task.latent is not None and task.image_tokens == 0
                    ),
                    kv=kv,
                    attention=attention,
                    mesh=mesh,
                )
                self._last_request_pool_indices = batch.request_pool_indices
        except Exception as error:
            self.trace.emit(
                ExecutionPhase.ROUTE_EXECUTION,
                operations,
                duration_us=(time.perf_counter_ns() - started) // 1000,
                route=phase.value,
                row_kind_counts=counts,
                error=error,
            )
            raise _input_failure(error, phase, operations) from error

        calls = 0
        weights = tasks[0].weights
        if any(
            task.weights.digest != weights.digest or task.weights.version != weights.version
            for task in tasks
        ):
            raise ValueError("one model call cannot mix immutable weight sets")

        def invoke(value: ForwardBatch) -> ForwardOutput:
            nonlocal calls
            calls += 1
            ids = buffers.input_ids[:0] if value.input_ids is None else value.input_ids
            positions = buffers.positions[0, :0] if value.positions is None else value.positions
            if weights.version == 0:
                result = self._phase(ids, positions, value)
            else:
                result = torch.func.functional_call(
                    self._phase,
                    {f"model.{name}": tensor for name, tensor in weights.tensors.items()},
                    (ids, positions, value),
                    strict=True,
                )
            return result

        try:
            self.trace.emit(
                ExecutionPhase.ROUTE_EXECUTION,
                operations,
                route=phase.value,
                row_kind_counts=counts,
            )
            with torch.inference_mode():
                graph_run = partition.graphs.execute(
                    (phase.value, *graph_shape),
                    batch,
                    invoke,
                    eligible=graph_eligible,
                    borrow_output=all(
                        task.token_ids is not None
                        and int(task.token_ids.numel()) == 1
                        and task.selection is TokenSelection.LAST_LOGITS
                        for task in tasks
                    ),
                )
            output = graph_run.output
            path = RunPath(graph_run.path)
            output.validate_for(batch)
            _validate_outputs(output.values, tasks, target)
            self._last_output_event = partition.record_output()
        except Exception as error:
            self.trace.emit(
                ExecutionPhase.FORWARD_COMPLETION,
                operations,
                duration_us=(time.perf_counter_ns() - started) // 1000,
                route=phase.value,
                row_kind_counts=counts,
                error=error,
            )
            raise _execution_failure(error, phase, operations) from error

        expected_calls = {
            RunPath.EAGER: 1,
            RunPath.GRAPH_CAPTURE: 2,
            RunPath.GRAPH_REPLAY: 0,
        }
        if path in expected_calls and calls != expected_calls[path]:
            raise ComputeError(
                f"model execution path {path.value!r} made {calls} forward calls",
                phase="graph_execution",
                route=phase.value,
                operations=tuple((item.session_id, item.epoch, item.op_id) for item in operations),
            )
        duration_us = (time.perf_counter_ns() - started) // 1000
        self._last_observation = RunObservation(
            route=phase.value,
            row_count=len(tasks),
            row_kind_counts=tuple(sorted(counts.items())),
            path=path,
            duration_us=duration_us,
            graph_unpadded_tokens=graph_run.row_count if path is not RunPath.EAGER else 0,
            graph_padded_tokens=(
                graph_run.padded_row_count - graph_run.row_count if path is not RunPath.EAGER else 0
            ),
        )
        self.trace.emit(
            ExecutionPhase.FORWARD_COMPLETION,
            operations,
            duration_us=duration_us,
            route=phase.value,
            row_kind_counts=counts,
            execution_path=path.value,
        )
        return output.values


def _kind_counts(tasks: tuple[Any, ...]) -> dict[str, int]:
    result: dict[str, int] = {}
    for task in tasks:
        result[task.kind] = result.get(task.kind, 0) + 1
    return result


def _claim_graph_runner(
    graph_runner: CudaGraphRunner | None,
    stream: torch.cuda.Stream | None,
    expected_context: int | None,
) -> CudaGraphRunner:
    if graph_runner is None:
        raise ValueError("model runner requires a graph runner")
    graph_runner.bind_partition(stream, expected_context)
    return graph_runner


def _base_version(operation: Any) -> int:
    point = operation.parent.point
    value = getattr(point, "point_index", 0)
    return int(value)


def _validate_outputs(
    values: tuple[torch.Tensor, ...],
    tasks: tuple[Any, ...],
    device: torch.device,
) -> None:
    for value, task in zip(values, tasks, strict=True):
        if value.device != device:
            raise ValueError(f"model output is on {value.device}, expected {device}")
        if not value.is_floating_point():
            raise ValueError("raw neural outputs must use a floating dtype")
        if task.token_ids is not None and value.ndim < 2:
            raise ValueError("token output must retain token and feature dimensions")
        if task.latent is not None and task.image_tokens > 0 and value.shape != task.latent.shape:
            raise ValueError("flow prediction shape does not match its latent")
        if task.encode_pixels is not None and value.numel() == 0:
            raise ValueError("encoder output must not be empty")
        if task.latent is not None and task.image_tokens == 0:
            if value.ndim < 2 or tuple(value.shape[-2:]) != (
                task.image_height,
                task.image_width,
            ):
                raise ValueError("decoded tensor does not match declared image geometry")


def _input_failure(
    error: BaseException,
    phase: ModelPhase,
    operations: tuple[OperationTrace, ...],
) -> InputError:
    if isinstance(error, InputError):
        return error
    return InputError(
        str(error) or type(error).__name__,
        phase="input_staging",
        route=phase.value,
        operations=tuple((item.session_id, item.epoch, item.op_id) for item in operations),
    )


def _execution_failure(
    error: BaseException,
    phase: ModelPhase,
    operations: tuple[OperationTrace, ...],
) -> WorkerError:
    if isinstance(error, WorkerError):
        return error
    classified = classify(error)
    identities = tuple((item.session_id, item.epoch, item.op_id) for item in operations)
    if isinstance(error, GraphExecutionError) or classified.code in {
        ErrorCode.RESOURCE_ERROR,
        ErrorCode.FATAL_WORKER_FAILURE,
    }:
        return ResourceError(
            str(error) or type(error).__name__,
            phase="graph_or_device",
            route=phase.value,
            operations=identities,
            retryable=classified.retryable,
            fatal=classified.fatal,
        )
    return ComputeError(
        str(error) or type(error).__name__,
        phase="neural_execution",
        route=phase.value,
        operations=identities,
    )


__all__ = ["RunObservation", "RunPath"]
