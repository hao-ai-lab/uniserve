"""Direct staged execution of concrete model phases."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Hashable, Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, TypeVar, cast

import torch

from uniserve_worker.config import WorkerConfig
from uniserve_worker.execution.batch import (
    Domain,
    ImageParams,
    OpCode,
    Operation,
    RequestKey,
    RunLane,
    StaticDim,
)
from uniserve_worker.execution.forward_batch import (
    ForwardBatch,
    ForwardOutput,
    ModelPhase,
    TokenSelection,
)
from uniserve_worker.execution.runners.packed import (
    FlowCapture,
    GraphExecutionError,
    GraphGreedyOutput,
    MixedCapture,
    PackedRunner,
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
from uniserve_worker.foundation.resources import close_resources
from uniserve_worker.models.generation import GenerationPipeline
from uniserve_worker.models.inputs import ImageProcessor, PatchTransform
from uniserve_worker.models.runtime import ExecutionModel
from uniserve_worker.nn.attention_storage import attention_exchange_scope
from uniserve_worker.nn.collective import collective_scope
from uniserve_worker.nn.diffusion.schedule import DiffusionSchedule
from uniserve_worker.runtime.device import HostStagingRing, canonical_device, fill_cpu_ints
from uniserve_worker.runtime.device_products import device_product_storage

from ..nn.mesh import Communicator
from .bounded_storage import BoundedTensorStorage
from .graph.full import FullCudaGraphBackend
from .input_buffers import AttentionInputs, InputBuffers, InputGeometry
from .runners.denoise import DenoiseRunner
from .runners.module import ModuleRunner, TensorOutput

if TYPE_CHECKING:
    from ..runtime.cache_pool import CachePool
    from ..runtime.distributed import DistributedEnvironment
    from ..runtime.req_to_token_pool import ReqToTokenPool
    from ..runtime.runtime_states import RuntimeStates
    from .forward_batch import AttentionSelection
from .lane import ExecutionLaneRuntime, LaneConfig, create_green_contexts
from .rows import ForwardRow, LaneState

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

    output: ForwardOutput
    request_pool_indices: torch.Tensor
    path: RunPath
    output_event: torch.cuda.Event | None
    observation: RunObservation
    greedy: GraphGreedyOutput | None

    def materialize_values(self) -> tuple[torch.Tensor, ...]:
        """Order the consumer stream and collectively materialize raw outputs.

        A consumer using ``greedy`` directly can retain vocabulary sharding.
        Other consumers receive the complete unpadded logits through this
        boundary. Every TP member must make the same consumption decision.
        """

        if self.output_event is not None:
            torch.cuda.current_stream(self.request_pool_indices.device).wait_event(
                self.output_event
            )
        return self.output.materialize().values


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
        *,
        environment: DistributedEnvironment | None = None,
        attention: AttentionSelection | None = None,
        schedule: DiffusionSchedule | None = None,
    ) -> None:
        """Bind loaded compute modules to one public execution resource owner."""

        from ..backends.attention import resolve_attention_selection
        from ..nn.attention import RadixAttention, bind_dense_attention_modules

        self.attention = attention or resolve_attention_selection(
            worker_config.attention_backend or "auto",
            tuning=worker_config.flashinfer,
            block_size=worker_config.block_size,
        )
        bind_dense_attention_modules(model, self.attention)
        self.model = model
        self._environment = environment
        self.schedule = schedule
        self.worker_config = worker_config
        self.uses_lanes = False
        self.flow_captures: tuple[FlowCapture, ...] = ()
        self.flow_cfg_branches: tuple[int, ...] = ()
        self._mixed_qualification: dict[MixedCapture, bool] = {}
        self._startup_complete = False
        self._packed_bindings: dict[tuple[str, Domain], PackedRunner] = {}
        self._packed: list[PackedRunner] = []
        self._owned_lanes: list[ExecutionLaneRuntime] = []
        self._geometry_cache: OrderedDict[Hashable, object] = OrderedDict()
        self.modules: dict[str, ModuleRunner] = {}
        self.denoise: DenoiseRunner | None = None
        self._capture_stream: torch.cuda.Stream | None = None
        self._text_staging: HostStagingRing | None = None
        self._text_tokens: torch.Tensor | None = None
        self._preparation_stream: torch.cuda.Stream | None = None
        self._sum_reductions = {}
        self._attention_exchange_storage = {}
        self.scratch: BoundedTensorStorage | None = None
        self.context_workspace = None
        self._closed = False
        try:
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
                if model.resource_geometry.request_tensors
                and torch.device(worker_config.device).type == "cuda"
                else None
            )
            self._sum_reductions = environment.sum_reductions() if environment is not None else {}
            exchange_modules = tuple(
                module
                for module in model.modules()
                if isinstance(module, RadixAttention)
                and module.exchange.ulysses_group.world_size > 1
            )
            if exchange_modules:
                from ..bootstrap.capacity import packed_input_geometry
                from ..runtime.attention_storage import allocate_attention_exchange_storage

                if environment is None:
                    raise ValueError("sequence attention requires its distributed storage owner")
                geometry = packed_input_geometry(model, worker_config)
                dtype = getattr(torch, worker_config.model_dtype.removeprefix("torch."))
                for lane in worker_config.lanes or (None,):
                    lane_id = None if lane is None else lane.lane_id
                    self._attention_exchange_storage[lane_id] = allocate_attention_exchange_storage(
                        exchange_modules,
                        environment,
                        max_tokens=geometry.max_tokens,
                        dtype=dtype,
                        scope=("attention", id(self), lane_id),
                    )

            if model.scratch_schema:
                self.scratch = BoundedTensorStorage.allocate(
                    model.scratch_schema, worker_config.device, environment=environment
                )
            context = model.context_geometry
            if context is not None:
                if environment is None:
                    raise ValueError("context attention requires its distributed resource owner")
                self.context_workspace = environment.attention_context(context)

            model.bind_execution(self)
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                error.add_note(f"Resource cleanup also failed: {cleanup_error!r}")
            raise

    def graph_backend(self, *, shared_pool: bool = False) -> FullCudaGraphBackend | None:
        """Construct full capture for an already-bound numerical entry."""

        device = canonical_device(self.worker_config.device)
        if self.worker_config.graph_policy == "off" or device.type != "cuda":
            return None
        if self._capture_stream is None:
            self._capture_stream = torch.cuda.Stream(device=device)
        return FullCudaGraphBackend(
            device=device,
            stream=self._capture_stream,
            pool=torch.cuda.graph_pool_handle() if shared_pool else None,
        )

    def bind_module(
        self,
        name: str,
        forward: Callable[..., TensorOutput],
        *,
        inputs: tuple[torch.Tensor, ...] | None = None,
        groups: tuple[Communicator, ...] = (),
    ) -> ModuleRunner:
        """Associate a protocol entry with the rank's actual tensor computation."""

        if name in self.modules:
            raise ValueError(f"numerical entry {name!r} is already bound")
        module = ModuleRunner(
            forward,
            device=canonical_device(self.worker_config.device),
            backend=self.graph_backend(),
            inputs=inputs,
            groups=groups,
        )
        self.modules[name] = module
        return module

    def prepare_geometry(self, key: Hashable, build: Callable[[], GeometryT]) -> GeometryT:
        """Own immutable model metadata until every dependent execution is retired."""

        if key not in self._geometry_cache:
            if len(self._geometry_cache) >= max(2, self.worker_config.max_request_pool_size):
                torch.cuda.current_stream(self.worker_config.device).synchronize()
                retired = next(iter(self._geometry_cache))
                if self.denoise is not None:
                    self.denoise.discard_geometry(retired)
                del self._geometry_cache[retired]
            self._geometry_cache[key] = build()
        self._geometry_cache.move_to_end(key)
        return cast(GeometryT, self._geometry_cache[key])

    def warmup(self, storage: tuple[BoundedTensorStorage, ...]) -> None:
        """Prepare model/provider geometry through its numerical execution owners."""

        started = time.perf_counter()
        logger.info("starting numerical eager warmup device=%s", self.worker_config.device)
        with torch.inference_mode(), collective_scope(self._sum_reductions):
            self.model.warmup_execution(self, storage)
        self.synchronize()
        logger.info("completed numerical eager warmup seconds=%.3f", time.perf_counter() - started)

    def packed_forward(self, packed: PackedRunner, batch: ForwardBatch) -> ForwardOutput:
        """Use the same numerical and communication binding for startup and serving."""

        lane = packed.lane
        inputs = packed.inputs
        assert lane is not None and inputs is not None
        ids = inputs.input_ids[:0] if batch.input_ids is None else batch.input_ids
        positions = inputs.positions[0, :0] if batch.positions is None else batch.positions
        reductions = self._sum_reductions if lane.full_device else {}
        exchanges = self._attention_exchange_storage.get(lane.lane_id, {})
        with collective_scope(reductions), attention_exchange_scope(exchanges):
            return _invoke(self.model, ids, positions, batch)

    @torch.inference_mode()
    def capture(self, *, tokenizer, latents) -> None:
        """Prepare configured packed inputs in dependency order before runtime warmup."""

        from functools import partial

        from ..execution.batch import OpCode
        from .runners.decode import prepare_decode
        from .runners.flow import FlowRunner
        from .runners.packed import PrefillCapture
        from .runners.prefill import prepare_prefill

        processor = self.model.image_processor
        transform = None if processor is None else processor.vit
        patch_size = int(transform.patch_size) if isinstance(transform, PatchTransform) else None

        for phase in ("prefill", "decode", "flow"):
            started = time.perf_counter()
            captures_before = sum(packed.captures for packed in self._packed)
            logger.info("starting packed preparation phase=%s", phase)
            for packed in self._packed:
                lane = packed.lane
                assert lane is not None and packed.inputs is not None
                if lane.device != canonical_device(self.worker_config.device):
                    continue
                context = nullcontext() if lane.stream is None else torch.cuda.stream(lane.stream)
                if lane.stream is not None:
                    lane.order_after(torch.cuda.current_stream(lane.device))
                with context:
                    forward = partial(self.packed_forward, packed)
                    if (
                        phase == "prefill"
                        and Domain.PREFILL in lane.domains
                        and OpCode.AR_EXTEND in self.model.supported_work
                    ):
                        shapes = (
                            packed.prefill_shapes
                            if packed.enabled and packed.prefill_enabled
                            else (PrefillCapture(1, 1, 1),)
                        )
                        prepare_prefill(
                            packed,
                            packed.inputs,
                            forward,
                            shapes,
                            packed=self.model.tensorized_mixed,
                        )
                    elif (
                        phase == "decode"
                        and Domain.DECODE in lane.domains
                        and OpCode.AR_DECODE in self.model.supported_work
                    ):
                        prepare_decode(
                            packed, packed.inputs, forward, packed=self.model.tensorized_mixed
                        )
                    elif (
                        phase == "flow"
                        and Domain.FLOW in lane.domains
                        and self.model.generation is not None
                        and latents is not None
                    ):
                        flow = FlowRunner(
                            packed,
                            packed.inputs,
                            forward,
                            generation=self.model.generation,
                            latents=latents,
                            tokenizer=tokenizer,
                            patch_size=patch_size,
                            tensorized=self.model.tensorized_mixed,
                        )
                        if packed.enabled and packed.prefill_enabled and self.flow_captures:
                            flow.capture(
                                self.flow_captures, self.mixed_captures, self.qualify_mixed
                            )
                        else:
                            import math

                            generation = self.model.generation
                            capacity = min(
                                generation.max_latent_tokens,
                                generation.max_vae_grid_tokens,
                                latents.capacity_units,
                            )
                            side = max(1, math.isqrt(capacity)) * generation.latent_downsample
                            representative = tuple(
                                next(
                                    (
                                        shape
                                        for shape in reversed(self.flow_captures)
                                        if shape.cfg_branches == branches
                                    ),
                                    FlowCapture(1, side, side, branches),
                                )
                                for branches in self.flow_cfg_branches
                            )
                            flow.warmup(representative, self.mixed_captures, self.qualify_mixed)
            logger.info(
                "completed packed preparation phase=%s captured_shapes=%d seconds=%.3f",
                phase,
                sum(packed.captures for packed in self._packed) - captures_before,
                time.perf_counter() - started,
            )
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
        self,
        name: str,
        *inputs: torch.Tensor,
        output_indices: tuple[int, ...] | None = None,
    ) -> EntryResult:
        """Execute an entry and validate the logical results selected by its request.

        The entry schema declares available products. A request can select a
        subset, such as conditioning alone for text or conditioning plus image
        presentation tags. Results must match the selected schemas in order.
        Omitting the selection requires every declared result.
        """

        schemas = self.model.entry_outputs.get(name)
        if schemas is None:
            raise InputError(f"model does not declare computation entry {name!r}")
        if output_indices is not None:
            if (
                not output_indices
                or len(set(output_indices)) != len(output_indices)
                or any(index < 0 or index >= len(schemas) for index in output_indices)
            ):
                raise InputError(f"entry {name!r} selected undeclared tensor results")
            schemas = tuple(schemas[index] for index in output_indices)

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

        return self._run_module(name, inputs, validate)

    def run_module(self, name: str, *inputs: object) -> EntryResult:
        """Execute bound numerical modules over Tensor views and typed metadata.

        Numerical returns may be rank-local intermediate values. Logical product
        publication belongs to the caller and uses the entry's result declaration.
        Captured inputs remain Tensors with stable storage; eager calls may also
        receive mathematical metadata, explicit views and solver parameters.
        """

        return self._run_module(name, inputs)

    @torch.inference_mode()
    def _run_module(
        self,
        name: str,
        inputs: tuple[object, ...],
        validate: Callable[[tuple[torch.Tensor, ...]], None] | None = None,
    ) -> EntryResult:
        """Own eager/captured selection and observations for numerical calls."""

        module = self.modules.get(name)
        if module is None:
            raise InputError(f"rank does not own computation entry {name!r}")
        if any(not isinstance(value, torch.Tensor) for value in inputs):
            raise InputError("module arguments must be tensors")
        tensors = cast(tuple[torch.Tensor, ...], inputs)
        return self._observe_module(name, lambda: module.run(*tensors), validate)

    def run_denoising(
        self,
        tensors: object,
        metadata: object,
        step: int,
        count: int,
        schedule: DiffusionSchedule,
        *,
        slot: Hashable,
        geometry: Hashable,
    ) -> EntryResult:
        denoise = self.denoise
        if denoise is None:
            raise InputError("rank does not own denoising computation")
        if count != 1:
            raise InputError("denoising calls evaluate exactly one scheduled step")
        return self._observe_module(
            "denoiser",
            lambda: denoise.run(tensors, metadata, step, schedule, slot=slot, geometry=geometry),
        )

    @torch.inference_mode()
    def _observe_module(
        self,
        name: str,
        run: Callable[[], tuple[TensorOutput, str]],
        validate: Callable[[tuple[torch.Tensor, ...]], None] | None = None,
    ) -> EntryResult:
        started = time.perf_counter_ns()
        path = RunPath.EAGER
        with collective_scope(self._sum_reductions):
            output, execution_path = run()
            path = RunPath(execution_path)
        if path is RunPath.GRAPH_CAPTURE:
            logger.info(
                "completed first-use capture entry=%s device=%s seconds=%.3f",
                name,
                self.worker_config.device,
                (time.perf_counter_ns() - started) / 1e9,
            )
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
        return EntryResult(values, observation)

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

    def configure_packed(
        self,
        *,
        geometry: InputGeometry,
        cache_pool,
        latent_pool,
        decode_predicates: torch.Tensor,
        max_operations: int,
        request_slots: int,
        max_tokens: int,
        latent_capacity_units: int,
        decode_context_blocks: int,
        variants,
        max_inflight: int,
    ) -> None:
        """Resolve each lane's physical catalog from its bound capacities."""

        from ..bootstrap.capacity import device_total_bytes
        from ..config import DEFAULT_PREFILL_GRAPH_ROW_BUCKETS, graph_memory_budget_bytes
        from ..nn.diffusion.cfg import build_flow_cfg_plan
        from .batch import OpCode
        from .runners.packed import (
            PrefillCapture,
            select_flow_captures,
            select_mixed_captures,
            select_prefill_captures,
        )

        worker_config = self.worker_config
        packed_model = self.model
        attention = self.attention
        flow = packed_model.generation
        max_rows = min(
            max_operations,
            request_slots,
        )

        # Derive fixed staging and graph catalogs from the intersection of model,
        # lane, cache, latent, and scheduler capacities.
        decode_lane = next(
            (lane for lane in worker_config.lanes if Domain.DECODE in lane.domains),
            None,
        )
        prefill_lane = next(
            (lane for lane in worker_config.lanes if Domain.PREFILL in lane.domains),
            None,
        )
        flow_lane = next(
            (lane for lane in worker_config.lanes if Domain.FLOW in lane.domains),
            None,
        )
        decode_max_operations = min(
            max_rows,
            max_rows if decode_lane is None else int(decode_lane.max_batch_operations or max_rows),
        )
        prefill_max_tokens = min(
            int(max_tokens),
            (
                int(max_tokens)
                if prefill_lane is None
                else int(prefill_lane.max_batch_tokens or max_tokens)
            ),
        )
        flow_max_operations = min(
            max_rows,
            max_rows if flow_lane is None else int(flow_lane.max_batch_operations or max_rows),
        )
        decode_graph_batch_sizes = tuple(
            value
            for value in worker_config.decode_graph_batch_sizes
            if 0 < int(value) <= decode_max_operations and int(value) < int(cache_pool.num_pages)
        )
        prefill_capacity = min(
            int(max_tokens),
            int(packed_model.text_max_tokens),
            max(0, int(cache_pool.num_pages) - 1) * int(worker_config.block_size),
        )
        prefill_graph_token_sizes = tuple(
            value
            for value in worker_config.prefill_graph_token_sizes
            if 0 < int(value) <= min(prefill_capacity, prefill_max_tokens)
        )
        prefill_graph_row_sizes = DEFAULT_PREFILL_GRAPH_ROW_BUCKETS
        flow_cfg_branches: tuple[int, ...] = ()
        if flow is not None:
            branch_counts: list[int] = []
            for cfg_branches in range(1, int(flow.max_cfg_branches) + 1):
                image = capture_image_parameters(
                    cfg_branches,
                    steps=1,
                    height=16,
                    width=16,
                )
                guide = build_flow_cfg_plan(
                    cfg_text_scale=float(image.cfg_text_scale),
                    cfg_img_scale=float(image.cfg_img_scale),
                    recipe=flow.cfg_recipe,
                    renorm=image.cfg_renorm_type,
                    renorm_min=float(image.cfg_renorm_min),
                    use_cfg=True,
                )
                if len(guide.branches) != cfg_branches:
                    raise invalid_descriptor(
                        "flow CFG startup parameters do not realize their branch geometry"
                    )
                branch_counts.append(cfg_branches)
            flow_cfg_branches = tuple(branch_counts)
        flow_graph_buckets = (
            select_flow_captures(
                worker_config.flow_graph_shapes,
                worker_config.flow_graph_batch_sizes,
                flow_cfg_branches,
                max_operations=flow_max_operations,
                max_tokens=(geometry.max_tokens),
                per_image_capacity=int(latent_capacity_units),
                latent_capacity=int(latent_pool.capacity_units),
                physical_tokens=flow.physical_tokens,
                image_tokens=flow.image_tokens,
            )
            if flow is not None and latent_pool is not None
            else ()
        )
        shared_mixed_limit = (
            max_rows
            if not worker_config.lanes
            else max(
                (
                    min(
                        max_rows,
                        int(lane.max_batch_operations or max_rows),
                    )
                    for lane in worker_config.lanes
                    if {Domain.DECODE, Domain.FLOW} <= set(lane.domains)
                ),
                default=0,
            )
        )
        mixed_text_batch_sizes = (
            ()
            if (
                not flow_graph_buckets
                or not packed_model.tensorized_mixed
                or (
                    worker_config.lanes
                    and not any(
                        {Domain.DECODE, Domain.FLOW} <= set(lane.domains)
                        for lane in worker_config.lanes
                    )
                )
                or not {
                    OpCode.AR_DECODE,
                    OpCode.DIFFUSION_STEP,
                }.issubset(variants)
            )
            else tuple(
                range(
                    1,
                    min(
                        decode_max_operations,
                        shared_mixed_limit - 1,
                        max(int(batch_size) for batch_size in worker_config.flow_graph_batch_sizes),
                    )
                    + 1,
                )
            )
        )
        mixed_flow_graph_buckets = select_mixed_captures(flow_graph_buckets, mixed_text_batch_sizes)
        self.flow_cfg_branches = flow_cfg_branches
        graph_budget = graph_memory_budget_bytes(device_total_bytes(worker_config.device))

        def packed_factory(
            binding: ExecutionLaneRuntime,
            inputs: InputBuffers,
            expected_context: int | None,
        ) -> PackedRunner:
            """Construct a lane-scoped graph catalog within its row, token, and memory bounds."""

            device = binding.device
            lane = binding.lane
            stream = binding.stream
            domains = binding.domains
            owns_model_compute = str(device) == str(torch.device(worker_config.device))

            # Intersect global graph buckets with this physical lane's advertised capacity.
            lane_max_operations = (
                max_rows if lane is None else int(lane.max_batch_operations or max_rows)
            )
            lane_max_tokens = (
                prefill_capacity if lane is None else int(lane.max_batch_tokens or prefill_capacity)
            )
            lane_decode_buckets = (
                tuple(
                    value for value in decode_graph_batch_sizes if int(value) <= lane_max_operations
                )
                if owns_model_compute and Domain.DECODE in domains
                else ()
            )
            lane_prefill_buckets = (
                tuple(value for value in prefill_graph_token_sizes if int(value) <= lane_max_tokens)
                if owns_model_compute and Domain.PREFILL in domains
                else ()
            )
            lane_prefill_row_sizes = (
                prefill_graph_row_sizes if owns_model_compute and Domain.PREFILL in domains else ()
            )
            lane_prefill_catalog = (
                tuple(PrefillCapture(value, 1, 1) for value in lane_prefill_buckets)
                if packed_model.tensorized_mixed
                else select_prefill_captures(
                    lane_prefill_buckets,
                    lane_prefill_row_sizes,
                    max_rows=lane_max_operations,
                    max_tokens=lane_max_tokens,
                )
            )
            backend: FullCudaGraphBackend[tuple[ForwardOutput, GraphGreedyOutput | None]] | None = (
                FullCudaGraphBackend(
                    device=device,
                    stream=stream or torch.cuda.Stream(device=device),
                    pool=torch.cuda.graph_pool_handle(),
                    expected_context=expected_context,
                )
                if worker_config.graph_policy != "off" and device.type == "cuda"
                else None
            )
            return PackedRunner(
                inputs=inputs,
                lane=binding,
                backend=backend,
                enabled=worker_config.graph_policy != "off",
                prefill_enabled=worker_config.prefill_cuda_graph,
                cache=packed_model.cache_geometry,
                cache_pool=cache_pool,
                attention=attention,
                block_size=worker_config.block_size,
                memory_budget_bytes=graph_budget,
                decode_batch_sizes=lane_decode_buckets,
                decode_predicates=(
                    decode_predicates if owns_model_compute and Domain.DECODE in domains else None
                ),
                decode_context_blocks=decode_context_blocks,
                packed_context_blocks=geometry.max_blocks_per_row,
                prefill_token_sizes=(() if packed_model.tensorized_mixed else lane_prefill_buckets),
                prefill_row_sizes=lane_prefill_row_sizes,
                stream=stream,
                prefill_shapes=lane_prefill_catalog,
            )

        self.bind_packed(
            geometry=geometry,
            lanes=worker_config.lanes,
            max_inflight=max_inflight,
            packed_factory=packed_factory,
            flow_captures=flow_graph_buckets,
            mixed_captures=mixed_flow_graph_buckets,
        )

    def bind_packed(
        self,
        *,
        geometry: InputGeometry,
        packed_factory: Callable[
            [ExecutionLaneRuntime, InputBuffers, int | None],
            PackedRunner,
        ],
        lanes: tuple[LaneConfig, ...] = (),
        max_inflight: int = 1,
        flow_captures: tuple[FlowCapture, ...] = (),
        mixed_captures: tuple[MixedCapture, ...] = (),
    ) -> None:
        """Construct device-and-lane runtimes around one pure execution model."""

        self.flow_captures = flow_captures
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

        startup = ExitStack()
        bindings: list[tuple[ExecutionLaneRuntime, int | None]] = []
        packed_runners: list[PackedRunner] = []
        try:
            if lanes:
                if len(canonical) != 1:
                    raise ValueError("Green Context lanes require one physical CUDA device")
                missing = set(Domain).difference(
                    domain for lane in lanes for domain in lane.domains
                )
                if missing:
                    names = ", ".join(sorted(domain.value for domain in missing))
                    raise ValueError(f"lane configuration has no binding for domains: {names}")
                device = torch.device(canonical[0])
                greens = create_green_contexts(lanes, device)
                # Register every acquired context before allocating events or inputs.
                for green in greens:
                    startup.callback(green.close)
                for green in greens:
                    binding = ExecutionLaneRuntime(
                        lane=green.lane,
                        device=device,
                        stream=green.stream,
                        sm_count=green.sm_count,
                        green=green,
                        event_slots=int(green.lane.max_inflight or max_inflight) + 1,
                    )
                    binding.verify_stream()
                    bindings.append((binding, int(green.context)))
            else:
                for device_name in canonical:
                    device = torch.device(device_name)
                    binding = ExecutionLaneRuntime(
                        lane=None,
                        device=device,
                        stream=None,
                        sm_count=(
                            int(torch.cuda.get_device_properties(device).multi_processor_count)
                            if device.type == "cuda"
                            else 0
                        ),
                        event_slots=int(max_inflight) + 1,
                    )
                    bindings.append((binding, None))

            for binding, expected_context in bindings:
                context = (
                    nullcontext() if binding.stream is None else torch.cuda.stream(binding.stream)
                )
                with context:
                    inputs = make_buffer(str(binding.device))
                startup.callback(inputs.close)
                packed = packed_factory(binding, inputs, expected_context)
                startup.callback(packed.close)
                if expected_context is not None and self._environment is not None:
                    assert binding.stream is not None
                    packed.collectives = self._environment.stream_collectives(binding.stream)
                packed_runners.append(packed)
        except BaseException as error:
            try:
                self.synchronize()
                for binding, _context in bindings:
                    if binding.stream is not None:
                        binding.stream.synchronize()
            except BaseException as cleanup_error:
                error.add_note(f"Resource synchronization also failed: {cleanup_error!r}")
            try:
                startup.close()
            except BaseException as cleanup_error:
                error.add_note(f"Resource cleanup also failed: {cleanup_error!r}")
            raise
        # Packed runners own staging; physical lanes outlive all borrowing runners.
        self._owned_lanes.extend(binding for binding, _context in bindings)
        self._packed.extend(packed_runners)
        for packed in packed_runners:
            assert packed.lane is not None
            for domain in packed.lane.domains:
                self._packed_bindings[(str(packed.lane.device), domain)] = packed
        startup.pop_all()

    @torch.inference_mode()
    def prepare_fixed_modules(self) -> None:
        """Warm up and capture fixed module inputs during worker startup."""

        for name, module in self.modules.items():
            if not module.fixed:
                continue
            started = time.perf_counter()
            with collective_scope(self._sum_reductions):
                if module.backend is None:
                    module.warmup()
                else:
                    module.capture()
            logger.info(
                "prepared fixed module entry=%s mode=%s seconds=%.3f",
                name,
                "eager" if module.backend is None else "capture",
                time.perf_counter() - started,
            )

    def complete_startup(self) -> None:
        """Freeze attention bindings and model state after warmup completes."""

        for packed in self._packed:
            lane_runtime = packed.lane
            assert lane_runtime is not None
            lane_id = lane_runtime.lane_id or "default"
            logger.info("verifying CUDA graph catalog lane=%s", lane_id)
            try:
                packed.complete_startup()
            except GraphExecutionError as error:
                raise GraphExecutionError(f"execution lane {lane_id} failed startup") from error
            lane_runtime.verify_stream()
            logger.info("verified CUDA graph catalog lane=%s", lane_id)
        signature = tuple(
            (
                lane_runtime.lane_id,
                lane_runtime.sm_count,
                tuple(domain.value for domain in lane_runtime.domains),
                packed.startup_signature,
            )
            for packed in self._packed
            if (lane_runtime := packed.lane) is not None
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            logger.info("verifying tensor-parallel execution lane agreement")
            gathered: list[object] = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, signature)
            if any(value != signature for value in gathered):
                raise GraphExecutionError("tensor-parallel execution lanes disagree")
            logger.info("verified tensor-parallel execution lane agreement")

        self._startup_complete = True

    def close(self) -> None:
        """Drain computation, then release graphs and staging before physical lanes."""

        if self._closed:
            return
        self._closed = True
        actions: list[Callable[[], object]] = [self.synchronize]
        actions.extend(module.close for module in self.modules.values())
        if self.denoise is not None:
            actions.append(self.denoise.close)
        if self._text_staging is not None:
            actions.append(self._text_staging.close)
        actions.extend(packed.close for packed in reversed(self._packed))
        actions.extend(lane.close for lane in reversed(self._owned_lanes))
        actions.extend(
            reduction.close for reduction in reversed(tuple(self._sum_reductions.values()))
        )
        try:
            close_resources(*actions)
        finally:
            self.modules.clear()
            self.denoise = None
            self._capture_stream = None
            self._preparation_stream = None
            self._text_staging = None
            self._text_tokens = None
            self._packed.clear()
            self._sum_reductions.clear()
            self._geometry_cache.clear()
            self.context_workspace = None
            self.scratch = None
            self._attention_exchange_storage.clear()
            self._mixed_qualification.clear()
            self._packed_bindings.clear()
            self._owned_lanes.clear()

    def synchronize(self) -> None:
        """Drain each compute and preparation stream before releasing resident resources."""

        actions: list[Callable[[], object]] = []
        if canonical_device(self.worker_config.device).type == "cuda":
            actions.append(torch.cuda.current_stream(self.worker_config.device).synchronize)
        if self._preparation_stream is not None:
            actions.append(self._preparation_stream.synchronize)
        if self._capture_stream is not None:
            actions.append(self._capture_stream.synchronize)
        actions.extend(
            lane.stream.synchronize for lane in self._owned_lanes if lane.stream is not None
        )
        close_resources(*actions)

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
            if torch.device(self.worker_config.device).type == "cuda":
                raise RuntimeError("input preparation requires an assigned execution stream")
            # Host copies are synchronous; CPU execution has no stream to join.
            for destination, source in transfers:
                if destination.shape != source.shape or destination.dtype != source.dtype:
                    raise ValueError("prepared input must match destination shape and dtype")
                destination.copy_(source)
            yield
            return
        try:
            with torch.cuda.stream(stream):
                for destination, source in transfers:
                    if destination.shape != source.shape or destination.dtype != source.dtype:
                        raise ValueError("prepared input must match destination shape and dtype")
                    destination.copy_(source, non_blocking=True)
            yield
        finally:
            torch.cuda.current_stream(self.worker_config.device).wait_stream(stream)

    def generation(self) -> GenerationPipeline:
        """Require the model's diffusion generation contract for the active operation."""

        value = self.model.generation
        if not isinstance(value, GenerationPipeline):
            raise invalid_descriptor("operation requires model generation behavior")
        return value

    def image_processor(self) -> ImageProcessor:
        """Require the model's image preprocessing contract for the active operation."""

        value = self.model.image_processor
        if not isinstance(value, ImageProcessor):
            raise invalid_descriptor("operation requires model image processing")
        return value

    def operation_device(self, operation: Operation) -> torch.device:
        """Return the model or generation device assigned to an operation kind."""

        if operation.kind in {
            OpCode.DIFFUSION_PREPARE,
            OpCode.DIFFUSION_STEP,
            OpCode.DIFFUSION_DECODE,
            OpCode.MEDIA_APPEND,
            OpCode.DIFFUSION_FINALIZE,
        }:
            return canonical_device(
                self.worker_config.generation_device or self.worker_config.device
            )
        return canonical_device(self.worker_config.device)

    def phase_device(self, phase: ModelPhase) -> torch.device:
        """Resolve the construction-time device assignment for a numerical phase."""

        config = self.worker_config
        if phase in {ModelPhase.ENCODE_LATENT, ModelPhase.DECODE_LATENT}:
            return canonical_device(config.generation_device or config.device)
        return canonical_device(config.device)

    def plan_launches(self, lanes: tuple[RunLane, ...]) -> tuple[RunLane, ...]:
        """Resolve compatible mixed computation without merging logical operation identities."""

        decode = next((lane for lane in lanes if lane.domain is Domain.DECODE), None)
        flow = next((lane for lane in lanes if lane.domain is Domain.FLOW), None)
        if (
            decode is not None
            and flow is not None
            and self.model.tensorized_mixed
            and {operation.kind for lane in (decode, flow) for operation in lane.operations}
            == {OpCode.AR_DECODE, OpCode.DIFFUSION_STEP}
            and self.allows_mixed(self._mixed_bucket((decode, flow)))
        ):
            launch_id = min(decode.launch_id, flow.launch_id)
            lanes = tuple(
                replace(lane, launch_id=launch_id) if lane is decode or lane is flow else lane
                for lane in lanes
            )
        return tuple(lanes)

    def validate_launches(self, lanes: tuple[RunLane, ...]) -> None:
        """Validate submitted mixed launches against the model's qualified geometry."""

        groups: dict[int, list[RunLane]] = defaultdict(list)
        for lane in lanes:
            groups[lane.launch_id].append(lane)
        for group_lanes in groups.values():
            if len(group_lanes) < 2:
                continue
            variants = {operation.kind for lane in group_lanes for operation in lane.operations}
            if not self.model.tensorized_mixed or variants != {
                OpCode.AR_DECODE,
                OpCode.DIFFUSION_STEP,
            }:
                raise invalid_descriptor(
                    "tensorized mixed submission exceeds the supported mixed buckets"
                )
            bucket = self._mixed_bucket(tuple(group_lanes))
            if not self.allows_mixed(bucket):
                raise invalid_descriptor(
                    "tensorized mixed submission has no exact qualified bucket"
                )

    def _mixed_bucket(self, lanes: tuple[RunLane, ...]) -> MixedCapture:
        """Resolve a shared captured-graph bucket for a compatible mixed lane group."""

        decode_rows = sum(
            operation.kind is OpCode.AR_DECODE for lane in lanes for operation in lane.operations
        )
        flow_operations = tuple(
            operation
            for lane in lanes
            for operation in lane.operations
            if operation.kind is OpCode.DIFFUSION_STEP
        )
        latent_params = {
            (params.request_key, int(params.op_id)): params
            for lane in lanes
            for params in lane.latent_params
        }
        branch_counts: dict[tuple[RequestKey, int], int] = defaultdict(int)
        generation = self.model.generation
        if flow_operations and generation is None:
            raise invalid_descriptor("tensorized mixed flow has no generation runtime")
        for lane in lanes:
            for index, operation in enumerate(lane.operations):
                params = latent_params.get((operation.request_key, int(operation.op_id)))
                query_len = (
                    None
                    if params is None or generation is None
                    else generation.physical_tokens(int(params.height), int(params.width))
                )
                branch_counts[(operation.request_key, int(operation.op_id))] = min(
                    0 if generation is None else int(generation.max_cfg_branches),
                    sum(
                        row.operation_index == index
                        and (query_len is None or int(row.query_len) == int(query_len))
                        for row in lane.forward_rows
                    ),
                )
        geometries = {
            (
                int(latent_params[(operation.request_key, int(operation.op_id))].height),
                int(latent_params[(operation.request_key, int(operation.op_id))].width),
                branch_counts[(operation.request_key, int(operation.op_id))],
            )
            for operation in flow_operations
            if (operation.request_key, int(operation.op_id)) in latent_params
        }
        if len(geometries) != 1 or len(latent_params) != len(flow_operations):
            raise invalid_descriptor("tensorized mixed flow rows disagree on physical geometry")
        height, width, cfg_branches = next(iter(geometries))
        return MixedCapture(
            decode_rows=decode_rows,
            flow_rows=len(flow_operations),
            height=height,
            width=width,
            cfg_branches=cfg_branches,
        )

    def run_wave(
        self,
        tasks: tuple[tuple[ForwardRow, LaneState], ...],
        *,
        cache: CachePool | None,
        tables: ReqToTokenPool | None,
        states: RuntimeStates | None,
    ) -> tuple[torch.Tensor, ...]:
        """Combine compatible bound computations and return values in logical row order.

        Logical launch boundaries remain authoritative. Physical compatibility is
        resolved here, where staging, streams, model geometry, and weights meet.
        """

        grouped: dict[tuple[object, ...], list[tuple[int, ForwardRow, LaneState]]] = defaultdict(
            list
        )
        for index, (task, scope) in enumerate(tasks):
            target = self.phase_device(task.phase)
            packed = self._packed_bindings.get((str(target), scope.lane.domain))
            if packed is None:
                raise invalid_descriptor(
                    f"execution has no {scope.lane.domain.value!r} binding for {target}"
                )
            phase = (
                "textual"
                if task.phase in {ModelPhase.TEXT, ModelPhase.DENOISE}
                and self.model.tensorized_mixed
                else task.phase.value
            )
            shape = ()
            if not (self.model.tensorized_mixed and not self.uses_lanes):
                if task.encode_pixels is not None:
                    shape = tuple(int(value) for value in task.encode_pixels.shape)
                elif task.latent is not None:
                    shape = (task.image_height, task.image_width)
            key = (scope.lane.launch_id, packed, phase, str(target), shape)
            grouped[key].append((index, task, scope))

        values: list[torch.Tensor | None] = [None] * len(tasks)
        events: list[tuple[torch.device, torch.cuda.Event]] = []
        for group in grouped.values():
            rows = tuple(task for _index, task, _scope in group)
            if len({task.kind for task in rows}) > 1 and not self.model.tensorized_mixed:
                raise invalid_descriptor("tensorized mixed submission is outside the model limits")
            target = self.phase_device(rows[0].phase)
            for _index, _task, scope in group:
                scope.completion.register_device(target)
            result = self.run_forward_group(
                rows, group[0][2], cache=cache, tables=tables, states=states
            )
            output = result.materialize_values()
            if result.output_event is not None:
                events.append((target, result.output_event))
            for (index, _task, _scope), value in zip(group, output, strict=True):
                values[index] = value
        for device, event in events:
            torch.cuda.current_stream(device).wait_event(event)
        return tuple(cast(torch.Tensor, value) for value in values)

    def run_forward_group(
        self,
        tasks: tuple[ForwardRow, ...],
        scope: LaneState,
        *,
        cache: CachePool | None,
        tables: ReqToTokenPool | None,
        states: RuntimeStates | None,
    ) -> ForwardResult:
        """Prepare attention and execute one logical forward group on its binding."""

        from .attention import columns, dense_columns

        target = self.phase_device(tasks[0].phase)
        scope.completion.register_device(target)
        textual = all(task.phase in {ModelPhase.TEXT, ModelPhase.DENOISE} for task in tasks)
        attention = (
            columns(
                tasks, cache=cache, tables=tables, states=states, packed=self.model.tensorized_mixed
            )
            if textual
            else dense_columns(len(tasks), tuple(task.query_tokens for task in tasks))
        )
        result = self.run(
            tasks,
            device=target,
            attention=attention,
            graph_eligible=scope.graph_eligible and textual,
            domain=scope.lane.domain,
        )
        if int(result.request_pool_indices.numel()) != len(tasks):
            raise RuntimeError("model runner returned without aligned request slots")
        for index, task in enumerate(tasks):
            task.request_pool_index = result.request_pool_indices[index : index + 1]
        scope.observations.append(result.observation)
        return result

    def run(
        self,
        rows: tuple[ForwardRow, ...],
        *,
        device: torch.device | str,
        attention: AttentionInputs,
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
            (
                task.operation.request_key.authority_id,
                task.operation.request_key.request_id,
                task.operation.request_key.epoch,
                task.operation.op_id,
            )
            for task in tasks
        )
        packed = self._packed_bindings.get((str(target), domain))
        if packed is None:
            raise InputError(
                f"model runner has no {domain.value!r} execution lane for {target}",
                phase="input_staging",
                route=phase.value,
                operations=operations,
            )
        lane_runtime = packed.lane
        buffers = packed.inputs
        assert lane_runtime is not None and buffers is not None
        counts = _kind_counts(tasks)
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
                        cast(torch.Tensor, task.positions)
                        for task in tasks
                        if task.token_ids is not None
                    ),
                    token_selections=tuple(
                        cast(TokenSelection, task.selection)
                        for task in tasks
                        if task.token_ids is not None
                    ),
                    flow_row_indices=tuple(
                        index
                        for index, task in enumerate(tasks)
                        if task.latent is not None and task.image_tokens > 0
                    ),
                    flow_positions=tuple(
                        cast(torch.Tensor, task.positions)
                        for task in tasks
                        if task.latent is not None and task.image_tokens > 0
                    ),
                    flow_timesteps=tuple(
                        cast(torch.Tensor, task.timestep)
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
            raise _input_failure(error, phase, operations) from error

        def invoke(value: ForwardBatch) -> ForwardOutput:
            return self.packed_forward(packed, value)

        output_event = None
        try:
            with torch.inference_mode():
                graph_run = packed.run(
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
            return ForwardResult(
                output=output,
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
            raise _execution_failure(error, phase, operations) from error


def _kind_counts(tasks: tuple[ForwardRow, ...]) -> dict[str, int]:
    """Count forward rows by stable operation-kind label."""

    result: dict[str, int] = {}
    for task in tasks:
        result[task.kind] = result.get(task.kind, 0) + 1
    return result


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
    operations: tuple[tuple[int, int, int, int], ...],
) -> InputError:
    """Classify invalid model inputs with their phase and operation identities."""

    if isinstance(error, InputError):
        return error
    return InputError(
        str(error) or type(error).__name__,
        phase="input_staging",
        route=phase.value,
        operations=operations,
    )


def _execution_failure(
    error: BaseException,
    phase: ModelPhase,
    operations: tuple[tuple[int, int, int, int], ...],
) -> WorkerError:
    """Classify a model failure and attach the active phase and operation identities."""

    if isinstance(error, WorkerError):
        return error
    classified = classify(error)
    if isinstance(error, GraphExecutionError) or classified.code in {
        WorkerErrorCode.RESOURCE_ERROR,
        WorkerErrorCode.FATAL_WORKER_FAILURE,
    }:
        return ResourceError(
            str(error) or type(error).__name__,
            phase="graph_or_device",
            route=phase.value,
            operations=operations,
            retryable=classified.retryable,
            fatal=classified.fatal,
        )
    return ComputeError(
        str(error) or type(error).__name__,
        phase="neural_execution",
        route=phase.value,
        operations=operations,
    )


__all__ = ["ForwardResult", "ModelRunner", "RunObservation", "RunPath"]
