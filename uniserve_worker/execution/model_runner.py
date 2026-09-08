"""Direct staged execution of concrete model phases."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, TypeVar, cast

import torch

from uniserve_worker.config import WorkerConfig
from uniserve_worker.execution.batch import Domain, ImageParams, Operation, StaticDim
from uniserve_worker.execution.cuda_graph import (
    CudaGraphRunner,
    FlowCapture,
    GraphEntry,
    GraphExecutionError,
    GraphGreedyOutput,
    MixedCapture,
)
from uniserve_worker.execution.forward_batch import (
    ForwardBatch,
    ForwardOutput,
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
    invalid_descriptor,
)
from uniserve_worker.models.runtime import ExecutionModel
from uniserve_worker.nn.diffusion.schedule import DiffusionSchedule
from uniserve_worker.runtime.device import HostStagingRing, canonical_device, fill_cpu_ints
from uniserve_worker.runtime.device_products import device_product_storage

from .bounded_storage import BoundedTensorStorage
from .input_buffers import AttentionInputs, InputBuffers, InputGeometry

if TYPE_CHECKING:
    from ..runtime.distributed import DistributedEnvironment
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


@dataclass(frozen=True, slots=True)
class EntryResult:
    """Borrowed tensor results and the shared execution observation for an entry call."""

    values: tuple[torch.Tensor, ...]
    observation: RunObservation


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


def capture_image_parameters(
    cfg_branches: int,
    *,
    steps: int,
    height: int,
    width: int,
) -> ImageParams:
    """Build deterministic bounded image-generation parameters for startup warmup."""

    scales = {
        1: (1.0, 1.0),
        2: (4.0, 1.0),
        3: (4.0, 2.0),
    }
    try:
        text_scale, image_scale = scales[int(cfg_branches)]
    except KeyError as error:
        raise invalid_descriptor(
            "flow CFG branch geometry exceeds the concrete branch set"
        ) from error
    return ImageParams(
        steps=int(steps),
        cfg_text_scale=text_scale,
        cfg_img_scale=image_scale,
        height=int(height),
        width=int(width),
        seed=0,
    )


GeometryT = TypeVar("GeometryT")


class ModelRunner:
    """Own lane_runtime-local GPU resources and execute one physical model call."""

    def __init__(
        self,
        model: ExecutionModel,
        worker_config: WorkerConfig,
        trace: ExecutionTrace,
        *,
        environment: DistributedEnvironment | None = None,
        schedule: DiffusionSchedule | None = None,
    ) -> None:
        """Bind loaded compute modules to one public execution resource owner."""

        self.model = model
        self.schedule = schedule
        self.worker_config = worker_config
        self.trace = trace
        self.uses_lanes = False
        self.flow_captures: tuple[FlowCapture, ...] = ()
        self.prefill_tokens: tuple[int, ...] = ()
        self.prefill_rows: tuple[int, ...] = ()
        self._mixed_qualification: dict[MixedCapture, bool] = {}
        self._startup_complete = False
        self._execution_lanes: dict[tuple[str, Domain], ExecutionLaneRuntime] = {}
        self._owned_lanes: list[ExecutionLaneRuntime] = []
        self._geometry_cache: dict[Hashable, object] = {}
        self._module_graphs: dict[str, GraphEntry[torch.Tensor]] = {}
        self._capture_stream: torch.cuda.Stream | None = None
        self._text_staging: HostStagingRing | None = None
        self._text_tokens: torch.Tensor | None = None
        if (
            "text_encoder" in model.entry_outputs
            and getattr(model, "text_encoder", None) is not None
        ):
            self._text_staging = HostStagingRing(
                model.text_max_tokens, dtype=torch.int64, depth=2, device=worker_config.device
            )
            self._text_tokens = torch.empty(
                model.text_max_tokens, dtype=torch.int64, device=worker_config.device
            )
        self._preparation_stream = (
            torch.cuda.Stream(device=worker_config.device)
            if bool(model.resource_geometry.request_tensors)
            else None
        )

        self.scratch: BoundedTensorStorage | None = None
        self.context_workspace = None
        if model.scratch_schema:
            self.scratch = BoundedTensorStorage.allocate(
                model.scratch_schema, worker_config.device, environment=environment
            )
        context = model.context_geometry
        if context is not None:
            if environment is None:
                raise ValueError("context attention requires its distributed resource owner")
            self.context_workspace = environment.attention_context(context)

    def prepare_geometry(self, key: Hashable, build: Callable[[], GeometryT]) -> GeometryT:
        """Own immutable model metadata until every dependent execution is retired."""

        if key not in self._geometry_cache:
            self._geometry_cache[key] = build()
        return cast(GeometryT, self._geometry_cache[key])

    def warmup_modules(self, storage: tuple[BoundedTensorStorage, ...]) -> None:
        """Execute declared first-use inputs and retain their immutable shape metadata."""

        inputs = self.model.warmup_inputs
        if inputs is None:
            return
        for workload in inputs(storage, self.scratch, self.context_workspace, self.schedule):
            if workload.geometry is not None:
                key, metadata = workload.geometry
                if key in self._geometry_cache and self._geometry_cache[key] is not metadata:
                    raise ValueError("warmup geometry key already owns different metadata")
                self._geometry_cache[key] = metadata
            self.run_module(workload.name, *workload.inputs)
        self.synchronize()

    def stage_text_tokens(self, tokens: tuple[int, ...]) -> torch.Tensor:
        """Stage one text encoder input on the caller's stream without a world broadcast.

        The input is borrowed until the next staging call. Host sources remain
        retained through the copy fence independently of subsequent model work.
        """

        staging, storage = self._text_staging, self._text_tokens
        if staging is None or storage is None:
            raise InputError("rank has no text encoder input storage")
        if not 1 <= len(tokens) <= storage.numel():
            raise InputError("text encoder input exceeds its token capacity")
        index, host = staging.acquire()
        fill_cpu_ints(host, tokens)
        target = storage[: len(tokens)]
        target.copy_(host[: len(tokens)], non_blocking=storage.is_cuda)
        staging.release(index)
        return target.view(1, -1)

    def run_entry(
        self, name: str, *inputs: torch.Tensor, operations: tuple[OperationTrace, ...] = ()
    ) -> EntryResult:
        """Execute an entry and validate its declared logical Tensor results."""

        schemas = self.model.entry_outputs.get(name)
        if schemas is None:
            raise InputError(f"model does not declare computation entry {name!r}")

        def validate(values: tuple[torch.Tensor, ...]) -> None:
            if len(values) != len(schemas):
                raise ComputeError(f"entry {name!r} returned the wrong number of tensor results")
            for value, schema in zip(values, schemas, strict=True):
                dtype, _ = device_product_storage(schema.dtype)
                if str(value.dtype).removeprefix("torch.") != dtype:
                    raise ComputeError(
                        f"entry {name!r} changed tensor result {schema.name!r} dtype"
                    )
                dims = schema.shape_bound.dims
                if value.ndim != len(dims) or any(
                    extent != dim.extent
                    if isinstance(dim, StaticDim)
                    else not 0 < extent <= dim.bound
                    for extent, dim in zip(value.shape, dims, strict=True)
                ):
                    raise ComputeError(
                        f"entry {name!r} exceeded tensor result {schema.name!r} shape"
                    )

        return self._run_module(name, inputs, operations, validate)

    def run_module(
        self, name: str, *inputs: object, operations: tuple[OperationTrace, ...] = ()
    ) -> EntryResult:
        """Execute bound numerical modules over Tensor views and typed metadata.

        Numerical returns may be rank-local intermediate values. Logical product
        publication belongs to the caller and uses the entry's result declaration.
        Captured inputs remain Tensors with stable storage; eager calls may also
        receive mathematical metadata, explicit views and solver parameters.
        """

        return self._run_module(name, inputs, operations)

    @torch.inference_mode()
    def _run_module(
        self,
        name: str,
        inputs: tuple[object, ...],
        operations: tuple[OperationTrace, ...],
        validate: Callable[[tuple[torch.Tensor, ...]], None] | None = None,
    ) -> EntryResult:
        """Own eager/captured selection and observations for numerical calls."""

        try:
            module = self.model.get_submodule(name)
        except AttributeError as error:
            raise InputError(f"rank does not own computation entry {name!r}") from error
        started = time.perf_counter_ns()
        path = RunPath.GRAPH_REPLAY if name in self.model.capture_inputs else RunPath.EAGER
        try:
            if path is RunPath.GRAPH_REPLAY:
                if any(not isinstance(value, torch.Tensor) for value in inputs):
                    raise InputError("captured module arguments must be Tensors")
                output = self.run_captured(name, *cast(tuple[torch.Tensor, ...], inputs))
            else:
                output = module(*inputs)
            values = (output,) if isinstance(output, torch.Tensor) else output
            if not isinstance(values, tuple) or any(
                not isinstance(value, torch.Tensor) for value in values
            ):
                raise ComputeError(f"entry {name!r} returned a non-tensor result")
            if any(value.device != canonical_device(self.worker_config.device) for value in values):
                raise ComputeError(f"entry {name!r} returned a tensor on another device")
            if validate is not None:
                validate(values)
            elapsed = (time.perf_counter_ns() - started) // 1000
            observation = RunObservation(name, 1, ((name, 1),), path, elapsed, 0, 0)
            self.trace.emit(
                ExecutionPhase.FORWARD_COMPLETION,
                operations,
                duration_us=elapsed,
                route=name,
                row_kind_counts={name: 1},
                execution_path=path.value,
            )
            return EntryResult(values, observation)
        except Exception as error:
            self.trace.emit(
                ExecutionPhase.FORWARD_COMPLETION,
                operations,
                duration_us=(time.perf_counter_ns() - started) // 1000,
                route=name,
                error=error,
                execution_path=path.value,
            )
            raise

    @property
    def mixed_captures(self) -> tuple[MixedCapture, ...]:
        return tuple(self._mixed_qualification)

    def allows_mixed(self, shape: MixedCapture) -> bool:
        """Allow configured startup calls, then only measured eligible service shapes."""

        eligible = self._mixed_qualification.get(shape)
        return eligible is not None and (not self._startup_complete or eligible)

    def qualify_mixed(self, shape: MixedCapture, eligible: bool) -> bool:
        """Record successful output and service qualification before admission."""

        if self._startup_complete or shape not in self._mixed_qualification:
            raise GraphExecutionError("mixed qualification is outside configured startup")
        qualified = self._mixed_qualification[shape] or eligible
        self._mixed_qualification[shape] = qualified
        return qualified

    def bind_packed(
        self,
        *,
        geometry: InputGeometry,
        graph_factory: Callable[
            [torch.device, LaneConfig | None, torch.cuda.Stream | None, int | None],
            CudaGraphRunner,
        ],
        lanes: tuple[LaneConfig, ...] = (),
        max_inflight: int = 1,
        flow_captures: tuple[FlowCapture, ...] = (),
        mixed_captures: tuple[MixedCapture, ...] = (),
        prefill_tokens: tuple[int, ...] = (),
        prefill_rows: tuple[int, ...] = (),
    ) -> None:
        """Construct device-and-lane runtimes around one pure execution model."""

        self.flow_captures = flow_captures
        self.prefill_tokens = prefill_tokens
        self.prefill_rows = prefill_rows
        self._mixed_qualification = dict.fromkeys(mixed_captures, False)
        worker_config = self.worker_config
        canonical = tuple(
            dict.fromkeys(
                (
                    str(torch.device(worker_config.device)),
                    str(torch.device(worker_config.generation_device or worker_config.device)),
                )
            )
        )
        if self._owned_lanes:
            raise RuntimeError("packed execution resources are already bound")
        self.uses_lanes = bool(lanes)

        def make_buffer(device: str) -> InputBuffers:
            """Allocate fixed-capacity staging storage for one execution device."""

            return InputBuffers(
                geometry=geometry,
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
                raise ValueError(f"lane worker_config has no binding for domains: {names}")
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

    @torch.inference_mode()
    def capture_modules(self) -> None:
        """Warm and capture loaded modules with their declared fixed inputs.

        Names identify local submodules bound during model loading, not methods
        selected by requests. Each executable has independent graph storage;
        replay runs on the caller's stream and borrows its output until reuse.
        """

        if not self.model.capture_inputs:
            return
        if self._startup_complete or self._module_graphs:
            raise GraphExecutionError("module capture requires an uncaptured startup binding")
        device = torch.device(self.worker_config.device)
        current = torch.cuda.current_stream(device)
        stream = torch.cuda.Stream(device=device)
        self._capture_stream = stream
        for name, schemas in self.model.capture_inputs.items():
            module = self.model.get_submodule(name)
            if any(schema.memory != "device" for schema in schemas):
                raise ValueError("captured module inputs require device storage")
            inputs = tuple(
                torch.full(
                    schema.shape,
                    0 if schema.fill is None else schema.fill,
                    dtype=schema.dtype,
                    device=device,
                )
                for schema in schemas
            )

            def invoke() -> torch.Tensor:
                value = module(*inputs)
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"captured module {name!r} must return a tensor")
                return value

            invoke()
            stream.wait_stream(current)
            entry = GraphEntry.capture(invoke, inputs=inputs, stream=stream)
            self._module_graphs[name] = entry
            with torch.cuda.stream(stream):
                entry.replay()
            current.wait_stream(stream)

    @torch.inference_mode()
    def run_captured(self, name: str, *inputs: torch.Tensor) -> torch.Tensor:
        """Replay a resident module on the caller's stream with stable input storage.

        Callers must finish consuming the borrowed output before another replay
        of this module, weight invalidation, or runner shutdown.
        """

        entry = self._module_graphs.get(name)
        if entry is None:
            raise GraphExecutionError(f"module {name!r} has no resident capture")
        return entry.replay(*inputs)

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
                raise GraphExecutionError(f"execution lane {lane_id} failed startup") from error
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

        self._startup_complete = True

    def invalidate_graphs(self, weight_version: int) -> None:
        """Drop captures whose embedded parameters predate the supplied weight version."""

        for entry in self._module_graphs.values():
            entry.close()
        self._module_graphs.clear()
        for lane_runtime in self._owned_lanes:
            lane_runtime.graphs.invalidate(weight_version)

    def close(self) -> None:
        """Release lane streams, graph bindings, and runner-owned staging state."""

        for entry in self._module_graphs.values():
            entry.close()
        self._module_graphs.clear()
        self._capture_stream = None
        if self._text_staging is not None:
            self._text_staging.close()
            self._text_staging = None
        self._text_tokens = None
        for lane_runtime in reversed(self._owned_lanes):
            lane_runtime.close()
        self._geometry_cache.clear()
        self.context_workspace = None
        self.scratch = None
        self._mixed_qualification.clear()
        self._execution_lanes.clear()
        self._owned_lanes.clear()

    def synchronize(self) -> None:
        """Drain compute and preparation streams before releasing resident resources."""

        if self._preparation_stream is not None:
            self._preparation_stream.synchronize()
            torch.cuda.current_stream(self.worker_config.device).synchronize()
        if self._capture_stream is not None:
            self._capture_stream.synchronize()

        for lane_runtime in self._owned_lanes:
            if lane_runtime.stream is not None:
                lane_runtime.stream.synchronize()

    @contextmanager
    def preparing_inputs(
        self, transfers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    ) -> Iterator[None]:
        """Overlap reserved input copies with computation, then join device order.

        Sources and destinations must remain reserved through this operation's
        physical completion. The final stream wait also covers failed compute,
        so completion and request retirement cannot overtake the input copies.
        """

        stream = self._preparation_stream
        if stream is None:
            raise RuntimeError("input preparation requires an assigned execution stream")
        try:
            with torch.cuda.stream(stream):
                for destination, source in transfers:
                    if destination.shape != source.shape or destination.dtype != source.dtype:
                        raise ValueError("prepared input must match destination shape and dtype")
                    destination.copy_(source, non_blocking=True)
            yield
        finally:
            torch.cuda.current_stream(self.worker_config.device).wait_stream(stream)

    def run(
        self,
        rows: tuple[ForwardRow, ...],
        *,
        device: torch.device | str,
        attention: AttentionInputs,
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
        weights = tasks[0].weights
        if any(task.weights.version != weights.version for task in tasks):
            raise ValueError("one model call cannot mix immutable weight sets")
        try:
            if lane_runtime.stream is not None:
                lane_runtime.order_after(torch.cuda.current_stream(target))
            stream_context = (
                nullcontext()
                if lane_runtime.stream is None
                else torch.cuda.stream(lane_runtime.stream)
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
                )
                request_pool_indices = batch.request_pool_indices
        except Exception as error:
            output_event = lane_runtime.record_output()
            if output_event is not None:
                torch.cuda.current_stream(target).wait_event(output_event)
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

        def invoke(value: ForwardBatch) -> ForwardOutput:
            """Invoke the model while counting eager or captured forward executions."""

            nonlocal calls
            calls += 1
            ids = buffers.input_ids[:0] if value.input_ids is None else value.input_ids
            positions = buffers.positions[0, :0] if value.positions is None else value.positions
            result = _invoke(self.model, ids, positions, value)
            return result

        output_event = None
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
                    graph_run.padded_row_count - graph_run.row_count
                    if path is not RunPath.EAGER
                    else 0
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
        except Exception as error:
            # A failed model can leave kernels on a lane stream. Its caller
            # retires storage behind the current stream's output fence, so join
            # every submitted lane access before reporting the failure.
            if output_event is None:
                output_event = lane_runtime.record_output()
            if output_event is not None:
                torch.cuda.current_stream(target).wait_event(output_event)
            self.trace.emit(
                ExecutionPhase.FORWARD_COMPLETION,
                operations,
                duration_us=(time.perf_counter_ns() - started) // 1000,
                route=phase.value,
                row_kind_counts=counts,
                error=error,
            )
            raise _execution_failure(error, phase, operations) from error


def _kind_counts(tasks: tuple[ForwardRow, ...]) -> dict[str, int]:
    """Count forward rows by stable operation-kind label."""

    result: dict[str, int] = {}
    for task in tasks:
        result[task.kind] = result.get(task.kind, 0) + 1
    return result


def _base_version(operation: Operation) -> int:
    """Return an operation's base weight generation or the default generation."""

    point = None if operation.parent is None else operation.parent.point
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
