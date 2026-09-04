"""Direct staged execution of concrete model phases."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from enum import StrEnum

import torch

from uniserve_worker.execution.batch import Domain, Operation
from uniserve_worker.execution.cuda_graph import (
    CudaGraphRunner,
    GraphExecutionError,
    GraphGreedyOutput,
)
from uniserve_worker.execution.forward_batch import (
    ForwardBatch,
    ForwardOutput,
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
    InputError,
    ResourceError,
    WorkerError,
    WorkerErrorCode,
    classify,
)
from uniserve_worker.models.runtime import ExecutionModel, WorkerDeployment

from .input_buffers import InputBuffers
from .lane import ExecutionLaneRuntime, LaneConfig, create_green_contexts
from .rows import ForwardRow

logger = logging.getLogger(__name__)


class RunPath(StrEnum):
    """Identifies eager, graph-capture, graph-replay, and graph-fallback execution paths."""

    EAGER = "eager"
    GRAPH_CAPTURE = "graph_capture"
    GRAPH_REPLAY = "graph_replay"
    GRAPH_FALLBACK = "graph_fallback"


@dataclass(frozen=True, slots=True)
class RunObservation:
    """Records route, row composition, graph padding, path, and duration for one model invocation."""

    route: str
    row_count: int
    row_kind_counts: tuple[tuple[str, int], ...]
    path: RunPath
    duration_us: int
    graph_unpadded_tokens: int
    graph_padded_tokens: int


@dataclass(frozen=True, slots=True)
class ForwardResult:
    """Pairs a model output with the route and execution-path observation that produced it."""

    values: tuple[torch.Tensor, ...]
    request_pool_indices: torch.Tensor
    path: RunPath
    output_event: torch.cuda.Event | None
    observation: RunObservation
    greedy: GraphGreedyOutput | None


def _invoke(
    model: ExecutionModel,
    ids: torch.Tensor,
    positions: torch.Tensor,
    batch: ForwardBatch,
) -> ForwardOutput:
    """Call the model with staged identifiers, positions, and execution context."""

    if batch.phase in {ModelPhase.TEXT, ModelPhase.DENOISE}:
        hidden = model.forward(ids, positions, batch)
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("model text/denoise forward must return a tensor")
        result = model.project(hidden, batch)
    elif batch.phase is ModelPhase.ENCODE_VISION:
        result = model.encode(batch.encode_pixels, batch)
    elif batch.phase is ModelPhase.ENCODE_LATENT:
        result = model.encoder_latent(batch.encode_pixels, batch)
    elif batch.phase is ModelPhase.DECODE_LATENT:
        result = model.decode_latent(batch.decode_latents, batch)
    else:
        raise TypeError(f"unsupported model phase {batch.phase.value!r}")
    if not isinstance(result, ForwardOutput):
        raise TypeError("concrete model phase must return ForwardOutput")
    return result


class ModelRunner:
    """Own lane_runtime-local GPU resources and execute one physical model call."""

    def __init__(
        self,
        model: ExecutionModel,
        deployment: WorkerDeployment,
        trace: ExecutionTrace,
        *,
        max_rows: int,
        max_tokens: int,
        max_text_tokens: int | None = None,
        max_blocks_per_row: int,
        hidden_size: int,
        devices: tuple[torch.device | str, ...],
        graph_factory: Callable[
            [torch.device, LaneConfig | None, torch.cuda.Stream | None, int | None],
            CudaGraphRunner,
        ],
        lanes: tuple[LaneConfig, ...] = (),
        max_inflight: int = 1,
    ) -> None:
        """Construct device-and-lane runtimes around one pure execution model."""

        canonical = tuple(dict.fromkeys(str(torch.device(device)) for device in devices))
        if not canonical:
            raise ValueError("model runner requires an execution device")
        deployed = {
            str(torch.device(deployment.device)),
            str(torch.device(deployment.generation_device or deployment.device)),
        }
        if set(canonical) != deployed:
            raise ValueError("model runner devices must match the worker deployment")
        self.model = model
        self.trace = trace
        self.uses_lanes = bool(lanes)
        self._execution_lanes: dict[tuple[str, Domain], ExecutionLaneRuntime] = {}
        self._owned_lanes: list[ExecutionLaneRuntime] = []

        def make_buffer(device: str) -> InputBuffers:
            """Allocate fixed-capacity staging storage for one execution device."""

            return InputBuffers(
                max_rows=max_rows,
                max_tokens=max_tokens,
                max_text_tokens=max_text_tokens,
                max_blocks_per_row=max_blocks_per_row,
                hidden_size=hidden_size,
                device=device,
                max_inflight=max_inflight,
            )

        if lanes:
            if len(canonical) != 1:
                raise ValueError("Green Context lanes require one physical CUDA device")
            device = torch.device(canonical[0])
            for green in create_green_contexts(lanes, device):
                with torch.cuda.stream(green.stream):
                    buffer = make_buffer(str(device))
                graphs = graph_factory(device, green.lane, green.stream, int(green.context))
                event_slots = int(green.lane.max_inflight or max_inflight) + 1
                lane_runtime = ExecutionLaneRuntime(
                    lane=green.lane,
                    device=device,
                    stream=green.stream,
                    sm_count=green.sm_count,
                    buffer=buffer,
                    graphs=graphs,
                    green=green,
                    event_slots=event_slots,
                )
                lane_runtime.verify_stream()
                self._owned_lanes.append(lane_runtime)
                for domain in green.lane.domains:
                    self._execution_lanes[(str(device), domain)] = lane_runtime
            missing = set(Domain).difference(domain for lane in lanes for domain in lane.domains)
            if missing:
                names = ", ".join(sorted(domain.value for domain in missing))
                raise ValueError(f"lane deployment has no binding for domains: {names}")
        else:
            for device_name in canonical:
                device = torch.device(device_name)
                stream: torch.cuda.Stream | None = None
                context = nullcontext() if stream is None else torch.cuda.stream(stream)
                with context:
                    buffer = make_buffer(device_name)
                graphs = graph_factory(device, None, stream, None)
                sm_count = (
                    int(torch.cuda.get_device_properties(device).multi_processor_count)
                    if device.type == "cuda"
                    else 0
                )
                lane_runtime = ExecutionLaneRuntime(
                    lane=None,
                    device=device,
                    stream=stream,
                    sm_count=sm_count,
                    buffer=buffer,
                    graphs=graphs,
                    event_slots=int(max_inflight) + 1,
                )
                self._owned_lanes.append(lane_runtime)
                for domain in Domain:
                    self._execution_lanes[(device_name, domain)] = lane_runtime

    @property
    def execution_lanes(self) -> tuple[ExecutionLaneRuntime, ...]:
        """Expose lane runtimes in deterministic ownership order."""

        return tuple(self._owned_lanes)

    def complete_startup(self) -> None:
        """Freeze attention bindings and model state after warmup completes."""

        for lane_runtime in self._owned_lanes:
            lane_id = lane_runtime.lane_id or "default"
            logger.info("verifying CUDA graph catalog lane=%s", lane_id)
            try:
                lane_runtime.graphs.complete_startup()
            except GraphExecutionError as error:
                raise GraphExecutionError(
                    f"execution lane {lane_id} failed startup"
                ) from error
            lane_runtime.verify_stream()
            logger.info("verified CUDA graph catalog lane=%s", lane_id)
        signature = tuple(
            (
                lane_runtime.lane_id,
                lane_runtime.sm_count,
                tuple(domain.value for domain in lane_runtime.domains),
                lane_runtime.graphs.startup_signature,
            )
            for lane_runtime in self._owned_lanes
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            logger.info("verifying tensor-parallel execution lane agreement")
            gathered: list[object] = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, signature)
            if any(value != signature for value in gathered):
                raise GraphExecutionError("tensor-parallel execution lanes disagree")
            logger.info("verified tensor-parallel execution lane agreement")

    def invalidate_graphs(self, weight_version: int) -> None:
        """Drop captures whose embedded parameters predate the supplied weight version."""

        for lane_runtime in self._owned_lanes:
            lane_runtime.graphs.invalidate(weight_version)

    def close(self) -> None:
        """Release lane streams, graph bindings, and runner-owned staging state."""

        for lane_runtime in reversed(self._owned_lanes):
            lane_runtime.close()
        self._execution_lanes.clear()
        self._owned_lanes.clear()

    def synchronize(self) -> None:
        """Wait for every non-default lane stream owned by this runner."""

        for lane_runtime in self._owned_lanes:
            if lane_runtime.stream is not None:
                lane_runtime.stream.synchronize()

    def run(
        self,
        rows: tuple[ForwardRow, ...],
        *,
        device: torch.device | str,
        attention: dict[str, object],
        mesh: MeshView,
        graph_shape: tuple[object, ...],
        graph_eligible: bool,
        domain: Domain,
    ) -> ForwardResult:
        """Stage forward rows, choose eager or CUDA graph execution, invoke the model, and validate outputs."""

        tasks = rows
        if not tasks:
            raise ValueError("model runner received an empty call")
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
                task.operation.request_key.authority_id,
                task.operation.request_key.request_id,
                task.operation.request_key.epoch,
                task.operation.op_id,
                _base_version(task.operation),
            )
            for task in tasks
        )
        lane_runtime = self._execution_lanes.get((str(target), domain))
        if lane_runtime is None:
            raise InputError(
                f"model runner has no {domain.value!r} execution lane for {target}",
                phase="input_staging",
                route=phase.value,
                operations=tuple(
                    (item.authority_id, item.request_id, item.epoch, item.op_id)
                    for item in operations
                ),
            )
        buffers = lane_runtime.buffer
        counts = _kind_counts(tasks)
        try:
            if lane_runtime.stream is not None:
                lane_runtime.order_after(torch.cuda.current_stream(target))
            stream_context = (
                nullcontext() if lane_runtime.stream is None else torch.cuda.stream(lane_runtime.stream)
            )
            with stream_context:
                batch = buffers.stage(
                    phase=phase,
                    row_count=len(tasks),
                    request_pool_indices=tuple(task.request_pool_idx for task in tasks),
                    decode_force_finish=(
                        tuple(bool(task.decode_force_finish) for task in tasks)
                        if all(
                            task.decode_predicate is not None and task.decode_predicate_tagged
                            for task in tasks
                        )
                        else ()
                    ),
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
                    attention=attention,
                    mesh=mesh,
                )
                request_pool_indices = batch.request_pool_indices
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
        if any(task.weights.version != weights.version for task in tasks):
            raise ValueError("one model call cannot mix immutable weight sets")

        def invoke(value: ForwardBatch) -> ForwardOutput:
            """Invoke the model while counting eager or captured forward executions."""

            nonlocal calls
            calls += 1
            ids = buffers.input_ids[:0] if value.input_ids is None else value.input_ids
            positions = buffers.positions[0, :0] if value.positions is None else value.positions
            result = _invoke(self.model, ids, positions, value)
            return result

        try:
            self.trace.emit(
                ExecutionPhase.ROUTE_EXECUTION,
                operations,
                route=phase.value,
                row_kind_counts=counts,
            )
            with torch.inference_mode():
                graph_run = lane_runtime.graphs.execute(
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
            greedy = graph_run.greedy
            path = RunPath(graph_run.path)
            output.validate_for(batch)
            _validate_outputs(output.values, tasks, target)
            output_event = lane_runtime.record_output()
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
                operations=tuple(
                    (item.authority_id, item.request_id, item.epoch, item.op_id)
                    for item in operations
                ),
            )
        duration_us = (time.perf_counter_ns() - started) // 1000
        observation = RunObservation(
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
        return ForwardResult(
            values=output.values,
            request_pool_indices=request_pool_indices,
            path=path,
            output_event=output_event,
            observation=observation,
            greedy=greedy,
        )


def _kind_counts(tasks: tuple[ForwardRow, ...]) -> dict[str, int]:
    """Count forward rows by stable operation-kind label."""

    result: dict[str, int] = {}
    for task in tasks:
        result[task.kind] = result.get(task.kind, 0) + 1
    return result


def _base_version(operation: Operation) -> int:
    """Return an operation's base weight generation or the default generation."""

    point = operation.parent.point
    value = getattr(point, "point_index", 0)
    return int(value)


def _validate_outputs(
    values: tuple[torch.Tensor, ...],
    tasks: tuple[ForwardRow, ...],
    device: torch.device,
) -> None:
    """Validate one model output tensor per task on the expected device."""

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
    """Classify invalid model inputs with phase and operation trace context."""

    if isinstance(error, InputError):
        return error
    return InputError(
        str(error) or type(error).__name__,
        phase="input_staging",
        route=phase.value,
        operations=tuple(
            (item.authority_id, item.request_id, item.epoch, item.op_id) for item in operations
        ),
    )


def _execution_failure(
    error: BaseException,
    phase: ModelPhase,
    operations: tuple[OperationTrace, ...],
) -> WorkerError:
    """Classify a model failure and attach the active phase and operation traces."""

    if isinstance(error, WorkerError):
        return error
    classified = classify(error)
    identities = tuple(
        (item.authority_id, item.request_id, item.epoch, item.op_id) for item in operations
    )
    if isinstance(error, GraphExecutionError) or classified.code in {
        WorkerErrorCode.RESOURCE_ERROR,
        WorkerErrorCode.FATAL_WORKER_FAILURE,
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


__all__ = ["ForwardResult", "ModelRunner", "RunObservation", "RunPath"]
