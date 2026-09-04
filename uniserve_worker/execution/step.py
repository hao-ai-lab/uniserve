"""Candidate preparation and resource-specific publication for execution batches."""

from __future__ import annotations

import logging
import math
import time
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from functools import partial
from typing import Any, cast

import torch

from uniserve_worker.execution.batch import (
    ArResult,
    AttentionRegime,
    Checkpoint,
    CompletionState,
    DeviceSelected,
    DiffusionResult,
    Domain,
    DType,
    EncoderResult,
    ErrorCode,
    FinishFlags,
    FixedCheckpoint,
    Free,
    LaneResult,
    LatentPlacement,
    LogicalLengths,
    ModelOutput,
    Operation,
    OpStatus,
    ProductKind,
    ProductPayload,
    ProductRef,
    RegistrationAck,
    RequestKey,
    Run,
    RunKind,
    RunLane,
    RunResult,
    ShapeBound,
    StorageClass,
    TimingCounters,
    TokenSpan,
    TransferHandle,
    TransferResult,
    WorkerForwardStats,
    decode_sampling_state_bytes,
    decode_token_product_bytes,
)
from uniserve_worker.execution.forward_batch import (
    AttentionSelection,
    ModelPhase,
    RouteMeshView,
)
from uniserve_worker.execution.graph_bucket import GraphBucket
from uniserve_worker.execution.output import (
    ImagePayload,
    LogprobPayload,
    OutputBuffer,
    OutputPool,
    OutputRecord,
    PendingOutput,
    TokenCapture,
    TransferPayload,
)
from uniserve_worker.execution.trace import (
    ExecutionPhase,
    ExecutionTrace,
    OperationTrace,
)
from uniserve_worker.foundation.errors import (
    WorkerError,
    WorkerErrorCode,
    classify,
    invalid_descriptor,
    should_capture_trace,
    unsupported_operation,
    unsupported_setup,
)
from uniserve_worker.loader.weight_set import WeightSet
from uniserve_worker.models.generation import (
    GenerationPipeline,
)
from uniserve_worker.models.inputs import ImageProcessor
from uniserve_worker.models.runtime import (
    ExecutionModel,
    WorkerDeployment,
)
from uniserve_worker.models.video import DecodeKind, VideoRunner
from uniserve_worker.nn.diffusion.cfg import build_flow_cfg_plan
from uniserve_worker.nn.diffusion.integrator import euler_step
from uniserve_worker.nn.diffusion.schedule import (
    x_pred_to_velocity,
)
from uniserve_worker.nn.mesh import DeviceMesh
from uniserve_worker.profiling import profile_range
from uniserve_worker.runtime.cache_pool import CachePool
from uniserve_worker.runtime.cpu import CpuPool
from uniserve_worker.runtime.device import canonical_device
from uniserve_worker.runtime.device_events import DeviceEventPool
from uniserve_worker.runtime.device_products import (
    DeviceProductMetadata,
    DeviceProductRead,
    DeviceProducts,
    DeviceProductWrite,
    ImageRange,
)
from uniserve_worker.runtime.encoder_cache import (
    EncoderCache,
    EncoderMetadata,
    EncoderRead,
)
from uniserve_worker.runtime.latent_pool import (
    LatentPool,
    LatentSnapshot,
)
from uniserve_worker.runtime.req_to_token_pool import ReqToTokenPool
from uniserve_worker.runtime.request import (
    RequestDraft,
    RequestPool,
    RequestRuntime,
    SpeculativeCommit,
)
from uniserve_worker.runtime.runtime_states import RuntimeStates
from uniserve_worker.transfer.connector import CachePublication, CachePublications
from uniserve_worker.transfer.tickets import Locator, Transport, decode_transfer_handle

from .attention import columns as _attention_columns
from .attention import dense_columns as _dense_attention_columns
from .model_runner import ForwardResult, ModelRunner, RunObservation, RunPath
from .resources import ExecutionResources
from .rows import (
    DecodeRuntimePublication,
    ForwardRow,
    LaneLayout,
    LaneState,
    LatentExecution,
    OperationIdentity,
    OperationState,
    Outcome,
    PreparedExecution,
    PreparedPredicateBatch,
    PreparedTransferInput,
    SampleWork,
    dependencies_ready,
)
from .sample import (
    copy_runtime_scalar as _copy_runtime_scalar,
)
from .sample import (
    sample as _sample_task_batch,
)

logger = logging.getLogger(__name__)

SAMPLING_COMPLETION_FIELDS = 4
TOKEN_CONTINUATION_BIT = 1 << 31
TOKEN_VALUE_MASK = TOKEN_CONTINUATION_BIT - 1
_GENERATION_WORK_VARIANTS = frozenset(
    {
        RunKind.DIFFUSION_PREPARE,
        RunKind.DIFFUSION_STEP,
        RunKind.DIFFUSION_DECODE,
        RunKind.DIFFUSION_FINALIZE,
    }
)
_MIN_MIXED_SERVICE_SPEEDUP = 1.03


def create_execution_resources(
    *,
    runner: ModelRunner | None,
    model: ExecutionModel,
    deployment: WorkerDeployment,
    attention: AttentionSelection | None,
    requests: RequestPool,
    runtime_states: RuntimeStates | None,
    cache_pool: CachePool | None,
    req_to_token_pool: ReqToTokenPool | None,
    latent_pool: LatentPool | None,
    device_products: DeviceProducts,
    encoder_cache: EncoderCache,
    device_events: DeviceEventPool,
    outputs: OutputPool,
    cpu_tasks: CpuPool,
    weights: WeightSet,
    mesh: DeviceMesh,
    transport: Transport | None,
    tokenizer: Any | None,
    model_name: str,
    weight_version: int,
    allowed_work_variants: frozenset[RunKind],
    mixed_buckets: tuple[GraphBucket, ...],
    trace: ExecutionTrace,
    media_mux: Any | None = None,
    media_output_ring: Any | None = None,
) -> ExecutionResources:
    """Compose validated model, runtime-store, lane, transport, tracing, and media ownership into one execution root."""

    if not allowed_work_variants:
        raise ValueError("execution step must accept at least one work variant")
    if not model_name:
        raise unsupported_setup("execution model name is empty")
    if weights.version != weight_version:
        raise unsupported_setup("base-weight version does not match its weight set")
    unsupported = allowed_work_variants - model.supported_work
    if unsupported:
        raise unsupported_setup(
            "execution work set exceeds the model implementation: "
            f"{sorted(value.value for value in unsupported)!r}"
        )
    kv_resources = (runner, runtime_states, cache_pool, req_to_token_pool)
    if any(resource is None for resource in kv_resources) != all(
        resource is None for resource in kv_resources
    ):
        raise unsupported_setup("packed-forward resources must be allocated as one set")
    if model.resource_geometry.kv != (cache_pool is not None):
        raise unsupported_setup("execution resources disagree with model KV ownership")
    if cache_pool is not None and attention is None:
        raise unsupported_setup("packed-forward execution requires attention selection")
    media_model = isinstance(model, VideoRunner)
    if media_model and mesh.coord("sp") == 0 and (
        media_mux is None or media_output_ring is None
    ):
        raise unsupported_setup("rank-zero video execution requires mux and output-ring resources")
    if not media_model and (media_mux is not None or media_output_ring is not None):
        raise unsupported_setup("packed-forward execution cannot own media output resources")
    device = canonical_device(deployment.device)
    generation_device = (
        device
        if deployment.generation_device is None
        else canonical_device(deployment.generation_device)
    )
    return ExecutionResources(
        runner=runner,
        model=model,
        deployment=deployment,
        attention=attention,
        requests=requests,
        runtime_states=runtime_states,
        cache_pool=cache_pool,
        req_to_token_pool=req_to_token_pool,
        cache_publications=(
            CachePublications(cache_pool, req_to_token_pool)
            if cache_pool is not None and req_to_token_pool is not None
            else None
        ),
        latent_pool=latent_pool,
        _media_mux=media_mux,
        _media_output_ring=media_output_ring,
        device_products=device_products,
        encoder_cache=encoder_cache,
        _device_events=device_events,
        _outputs=outputs,
        _cpu_tasks=cpu_tasks,
        weights=weights,
        mesh=mesh,
        transport=transport,
        tokenizer=tokenizer,
        model_name=model_name,
        weight_version=weight_version,
        allowed_work_variants=allowed_work_variants,
        mixed_buckets=frozenset(mixed_buckets),
        trace=trace,
        _device=device,
        _generation_device=generation_device,
    )


def close_execution(runtime: ExecutionResources) -> None:
    """Release all resources owned by an execution root in dependency-safe order."""

    runtime._collective_history.clear()
    runtime._transport_publications.clear()
    runtime._flow_prefix_slots.clear()
    runtime._qualified_mixed_buckets.clear()


def install_weights(runtime: ExecutionResources, weights: WeightSet) -> None:
    """Replace the live weight identity and invalidate captured graphs tied to its tensors."""

    if weights.version <= runtime.weights.version:
        raise ValueError("installed weight version must increase")
    if runtime.runner is not None:
        runtime.runner.invalidate_graphs(weights.version)
    runtime.weights = weights
    runtime.weight_version = weights.version


def _operation_identity(operation: Operation) -> OperationIdentity:
    """Return the request key and operation identifier for one operation."""

    return operation.request_key, int(operation.op_id)


def _reference_operation_identity(reference: ProductRef) -> OperationIdentity:
    """Return the producer request key and operation identifier for a product reference."""

    return reference.request_key, int(reference.producer_op_id)


def _unique_scopes(scopes: Sequence[LaneState]) -> tuple[LaneState, ...]:
    """Deduplicate lane scopes by object identity while preserving order."""

    unique: list[LaneState] = []
    seen: set[int] = set()
    for scope in scopes:
        identity = id(scope)
        if identity not in seen:
            seen.add(identity)
            unique.append(scope)
    return tuple(unique)


def _completion_error_code(code: WorkerErrorCode) -> ErrorCode:
    """Map internal failure classes to their completion-wire error codes."""

    if code == WorkerErrorCode.RESOURCE_ERROR:
        return ErrorCode.RESOURCE_EXHAUSTED
    if code == WorkerErrorCode.COMPUTE_ERROR:
        return ErrorCode.COMPUTE_ERROR
    if code in {WorkerErrorCode.INVARIANT_VIOLATION, WorkerErrorCode.FATAL_WORKER_FAILURE}:
        return ErrorCode.INTERNAL
    return ErrorCode.INVALID_OPERATION


def plan_run(runtime, batch: Run) -> Run:
    """Derive worker-local execution lanes from a flat physical run."""

    if batch.lanes or not batch.operations:
        return batch
    if any(
        placement.offset + placement.bytes > runtime.encoder_cache.byte_capacity
        for placement in batch.buffer_placements
    ):
        raise invalid_descriptor("run buffer placement exceeds the worker buffer pool")
    grouped: dict[Domain, list[tuple[int, Operation]]] = {}
    for index, operation in enumerate(batch.operations):
        grouped.setdefault(operation.domain, []).append((index, operation))
    lanes: list[RunLane] = []
    for lane_id, (domain, members) in enumerate(grouped.items(), start=1):
        global_to_local = {global_index: local_index for local_index, (global_index, _operation) in enumerate(members)}
        member_operations = tuple(operation for _index, operation in members)
        identities = {(operation.request_key, int(operation.op_id)) for operation in member_operations}
        rows = tuple(
            replace(row, operation_index=global_to_local[int(row.operation_index)])
            for row in batch.forward_rows
            if int(row.operation_index) in global_to_local
        )
        request_slots = {int(row.request_pool_index) for row in rows}
        attention = (
            AttentionRegime.CAUSAL
            if all(operation.kind in {RunKind.AR_EXTEND, RunKind.AR_DECODE, RunKind.AR_VERIFY} for operation in member_operations)
            else AttentionRegime.HYBRID
            if any(operation.kind is RunKind.DIFFUSION_STEP for operation in member_operations)
            else AttentionRegime.NONE
        )
        lanes.append(
            RunLane(
                lane_id=lane_id,
                launch_id=lane_id,
                collective_seq=batch.collective_seq,
                domain=domain,
                route=0,
                attention=attention,
                shape_class=0,
                operations=member_operations,
                block_tables=tuple(table for table in batch.block_tables if int(table.request_pool_idx) in request_slots),
                new_cache_pages=tuple(allocation for allocation in batch.new_cache_pages if int(allocation.request_pool_idx) in request_slots),
                forward_rows=rows,
                latent_placements=tuple(placement for placement in batch.latent_placements if (placement.request_key, int(placement.op_id)) in identities),
                decode_placements=tuple(placement for placement in batch.decode_placements if (placement.request_key, int(placement.op_id)) in identities),
                buffer_placements=tuple(
                    placement
                    for placement in batch.buffer_placements
                    if any(
                        product.buffer_id == placement.buffer
                        for operation in member_operations
                        for product in (
                            *operation.inputs,
                            *operation.outputs,
                            *((operation.predicate,) if operation.predicate is not None else ()),
                        )
                    )
                ),
            )
        )
    decode = next((lane for lane in lanes if lane.domain is Domain.DECODE), None)
    flow = next((lane for lane in lanes if lane.domain is Domain.FLOW), None)
    if (
        decode is not None
        and flow is not None
        and runtime.model.tensorized_mixed
        and {operation.kind for lane in (decode, flow) for operation in lane.operations}
        == {RunKind.AR_DECODE, RunKind.DIFFUSION_STEP}
        and _mixed_bucket(runtime, (decode, flow)) in runtime.mixed_buckets
    ):
        launch_id = min(decode.launch_id, flow.launch_id)
        lanes = [
            replace(lane, launch_id=launch_id) if lane is decode or lane is flow else lane
            for lane in lanes
        ]
    return replace(batch, lanes=tuple(lanes))


def prepare_batch(runtime, batch: Run) -> PreparedExecution | None:
    """Submit bounded transfer and predicate observations without waiting."""

    from . import transfer

    entries = tuple(
        payload
        for payload in batch.input_products
        if isinstance(payload.payload, TransferHandle)
    )
    transport = runtime.transport
    if entries and transport is None:
        raise unsupported_setup("cross-stage input requires a configured transport")
    transfers: list[PreparedTransferInput] = []
    for entry in entries:
        assert transport is not None
        kind, value = decode_transfer_handle(entry.payload)
        locators: tuple[Locator, ...]
        payload_kind: ProductKind | None = None
        height: int | None = None
        width: int | None = None
        latent_units: int | None = None
        step: int | None = None
        generation: int | None = None
        device_metadata: DeviceProductMetadata | None = None
        snapshot: CachePublication | None = None
        if kind == "encoder":
            if set(value) != {
                "generation",
                "height",
                "locator",
                "payload_kind",
                "width",
            }:
                raise invalid_descriptor("encoder transfer entry has an invalid shape")
            raw_locator = value["locator"]
            if not isinstance(raw_locator, dict):
                raise invalid_descriptor("encoder transfer entry locator is invalid")
            main = Locator.from_mapping(raw_locator)
            locators = (main,)
            raw_payload_kind = value["payload_kind"]
            height = transfer.metadata_uint(value, "height", 0)
            width = transfer.metadata_uint(value, "width", 0)
            generation = transfer.metadata_uint(value, "generation", 0)
            if (
                not isinstance(raw_payload_kind, str)
                or raw_payload_kind
                not in {ProductKind.VISION_FEATURE.value, ProductKind.LATENT_FEATURE.value}
                or min(height, width, generation) < 1
                or generation != int(entry.product.generation)
                or not transfer.locator_matches_product(main, entry.product)
            ):
                raise invalid_descriptor("encoder transfer metadata exceeds its product bounds")
            payload_kind = ProductKind(raw_payload_kind)
            if (
                entry.product.kind is not payload_kind
                or entry.product.storage_class is not StorageClass.LATENT_ARENA
            ):
                raise invalid_descriptor(
                    "encoder transfer entry disagrees with its product identity"
                )
        elif kind == "device_product":
            if set(value) != {
                "generation",
                "height",
                "locator",
                "value_range",
                "width",
            }:
                raise invalid_descriptor("device-product transfer entry has an invalid shape")
            raw_locator = value["locator"]
            if not isinstance(raw_locator, dict):
                raise invalid_descriptor("device-product transfer locator is invalid")
            main = Locator.from_mapping(raw_locator)
            locators = (main,)
            generation = transfer.metadata_uint(value, "generation", 0)
            height = transfer.metadata_uint(value, "height", 0)
            width = transfer.metadata_uint(value, "width", 0)
            raw_range = transfer.metadata_string(value, "value_range", "")
            if (height == 0) != (width == 0):
                raise invalid_descriptor("device-product image geometry is incomplete")
            if raw_range not in {"", *(value.value for value in ImageRange)}:
                raise invalid_descriptor("device-product value range is invalid")
            if height == 0 and raw_range:
                raise invalid_descriptor("non-image device product carries an image range")
            value_range = None if not raw_range else ImageRange(raw_range)
            device_metadata = (
                None
                if height == 0
                else DeviceProductMetadata(
                    height=height,
                    width=width,
                    value_range=value_range,
                )
            )
            if (
                generation != int(entry.product.generation)
                or not transfer.requires_device_product_binding(entry.product)
                or not transfer.locator_matches_product(main, entry.product)
            ):
                raise invalid_descriptor(
                    "device-product transfer metadata exceeds its product bounds"
                )
        elif kind == "latent":
            if set(value) != {
                "generation",
                "height",
                "latent_units",
                "locator",
                "step",
                "width",
            }:
                raise invalid_descriptor("latent transfer entry has an invalid shape")
            raw_locator = value["locator"]
            if not isinstance(raw_locator, dict):
                raise invalid_descriptor("latent transfer entry locator is invalid")
            main = Locator.from_mapping(raw_locator)
            locators = (main,)
            height = transfer.metadata_uint(value, "height", 0)
            width = transfer.metadata_uint(value, "width", 0)
            latent_units = transfer.metadata_uint(value, "latent_units", 0)
            step = transfer.metadata_uint(value, "step", 0)
            generation = transfer.metadata_uint(value, "generation", 0)
            pool = runtime.latent_pool
            expected_dtype = "" if pool is None else str(pool.dtype).removeprefix("torch.")
            expected_nbytes = (
                0
                if pool is None
                else latent_units * int(pool.latent_width) * int(pool.storage.element_size())
            )
            if (
                entry.product.kind is not ProductKind.LATENT
                or entry.product.storage_class is not StorageClass.LATENT_ARENA
                or pool is None
                or min(height, width, latent_units, generation) < 1
                or generation != int(entry.product.generation)
                or tuple(main.shape) != (latent_units, int(pool.latent_width))
                or main.dtype != expected_dtype
                or int(main.nbytes) != expected_nbytes
                or int(main.nbytes) > int(entry.product.max_bytes)
                or math.prod(main.shape) > int(entry.product.shape_bound.max_elements)
            ):
                raise invalid_descriptor("latent transfer metadata exceeds its product bounds")
            payload_kind = ProductKind.LATENT
        elif kind == "kv":
            if set(value) != {"generation", "snapshot"}:
                raise invalid_descriptor("KV transfer entry has an invalid shape")
            generation = transfer.metadata_uint(value, "generation", 0)
            snapshot = CachePublication.from_mapping(value["snapshot"])
            if (
                entry.product.kind is not ProductKind.KV
                or entry.product.storage_class is not StorageClass.PAGED_KV
                or generation != int(entry.product.generation)
            ):
                raise invalid_descriptor("KV transfer entry names a non-KV product")
            locators = tuple(snapshot.locators)
        else:
            raise invalid_descriptor("cross-stage transfer entry has an unknown kind")
        transfers.append(
            PreparedTransferInput(
                product=entry.product,
                kind=kind,
                locators=locators,
                tickets=tuple(transport.fetch_async(locator) for locator in locators),
                payload_kind=payload_kind,
                height=height,
                width=width,
                latent_units=latent_units,
                step=step,
                generation=generation,
                device_metadata=device_metadata,
                snapshot=snapshot,
            )
        )
    predicates = _prepare_predicates(
        runtime,
        batch,
        transfers=tuple(transfers),
    )
    if not transfers and predicates is None:
        return None
    return PreparedExecution(
        batch=batch,
        transfers=tuple(transfers),
        predicates=predicates,
    )


def _prepare_predicates(
    runtime,
    batch: Run,
    *,
    transfers: tuple[PreparedTransferInput, ...],
) -> PreparedPredicateBatch | None:
    """Capture completion-valued predicates from local products or prepared transfers."""

    # Predicate rows occupy one compact completion buffer regardless of whether
    # their source is already local or will arrive through a prepared transfer.
    operations = tuple(
        operation
        for operation in batch.operations
        if operation.predicate is not None and operation.predicate.kind is ProductKind.COMPLETION
    )
    if not operations:
        return None
    transferred = {transfer.product: transfer for transfer in transfers}
    buffer = runtime._outputs.acquire(
        len(operations),
        token_capacity=len(operations),
        devices=tuple(_operation_device(runtime, operation) for operation in operations),
    )
    captures: list[tuple[OperationIdentity, TokenCapture, int]] = []
    pending: list[tuple[OperationIdentity, PreparedTransferInput, int]] = []
    recorded: list[DeviceProductRead] = []
    try:
        # Local sources are consumed in device batches and captured directly;
        # transferred sources retain their target row for later completion.
        grouped: dict[torch.device, list[Operation]] = defaultdict(list)
        rows = {_operation_identity(operation): row for row, operation in enumerate(operations)}
        for operation in operations:
            transfer = transferred.get(cast(ProductRef, operation.predicate))
            if transfer is None:
                grouped[_operation_device(runtime, operation)].append(operation)
            else:
                pending.append(
                    (
                        _operation_identity(operation),
                        transfer,
                        rows[_operation_identity(operation)],
                    )
                )
        for device, device_operations in grouped.items():
            reads = runtime.device_products.consume_batch(
                tuple(
                    (
                        cast(ProductRef, operation.predicate),
                        int(operation.op_id),
                        device,
                    )
                    for operation in device_operations
                ),
                device=device,
            )
            recorded.extend(reads)
            for operation, read in zip(device_operations, reads, strict=True):
                identity = _operation_identity(operation)
                captures.append((identity, buffer.capture(read.tensor), rows[identity]))
            runtime.device_products.record_readers(reads, device=device)

        # No pending transfer can mutate the buffer once it is sealed.
        sealed = not pending
        if sealed:
            buffer.seal()
    except BaseException:
        # Every acquired read must receive a reader event even when preparation
        # fails before all device groups are captured.
        unrecorded = tuple(read for read in recorded if not read._recorded)
        if unrecorded:
            runtime.device_products.record_readers(unrecorded)
        buffer.abandon()
        raise
    return PreparedPredicateBatch(
        buffer=buffer,
        entries=captures,
        transferred=tuple(pending),
        sealed=sealed,
    )


def execute_prepared(runtime, prepared: PreparedExecution) -> RunResult:
    """Execute a fully staged batch through the bound single-use preparation callback."""

    if not prepared.ready():
        raise RuntimeError("prepared execution was observed before transfer readiness")
    return _execute(
        runtime,
        prepared.batch,
        prepared=prepared.transfers,
        predicate_values=prepared.predicate_values(),
        propagate_errors=False,
        graph_eligible=True,
    )


def complete_startup(runtime) -> None:
    """Retire pre-admission collective identities before serving traffic."""

    runtime.mixed_buckets = frozenset(runtime._qualified_mixed_buckets)
    if runtime.runner is not None:
        runtime.runner.complete_startup()
    if runtime.requests.request_ids():
        raise RuntimeError("startup completed with resident requests")
    runtime._collective_history.clear()


def execute_batch(
    runtime,
    batch: Run,
    *,
    prepared: tuple[PreparedTransferInput, ...] = (),
) -> RunResult:
    """Execute one canonical batch after server-side duplicate registration."""

    return _execute(
        runtime,
        batch,
        prepared=prepared,
        predicate_values={},
        propagate_errors=False,
        graph_eligible=True,
    )


def execute_startup(
    runtime,
    batch: Run,
    *,
    catalog_graphs: bool = True,
) -> RunResult:
    """Execute pre-admission work with direct error propagation."""

    return _execute(
        runtime,
        batch,
        prepared=(),
        predicate_values={},
        propagate_errors=True,
        graph_eligible=bool(catalog_graphs),
    )


def _execute(
    runtime,
    batch: Run,
    *,
    prepared: tuple[PreparedTransferInput, ...],
    predicate_values: Mapping[OperationIdentity, bool],
    propagate_errors: bool,
    graph_eligible: bool,
) -> RunResult:
    """Execute a prepared lane batch for startup or admitted traffic."""

    started = time.perf_counter_ns()
    operations = _trace_envelopes(batch.operations)
    validation_started = time.perf_counter_ns()
    try:
        _validate_batch(runtime, batch)
    except BaseException as error:
        runtime.trace.emit(
            ExecutionPhase.INPUT_VALIDATION,
            operations,
            duration_us=(time.perf_counter_ns() - validation_started) // 1000,
            error=error,
        )
        raise
    runtime.trace.emit(
        ExecutionPhase.INPUT_VALIDATION,
        operations,
        duration_us=(time.perf_counter_ns() - validation_started) // 1000,
    )
    required_predicates = {
        _operation_identity(operation)
        for operation in batch.operations
        if operation.predicate is not None and operation.predicate.kind is ProductKind.COMPLETION
    }
    if required_predicates != set(predicate_values):
        raise invalid_descriptor(
            "completion-predicated operations require exact prepared predicate values"
        )
    runtime.requests.apply_commands(batch.commands)
    _apply_release_controls(runtime, batch, before_execution=True)
    if not batch.operations:
        return RunResult(
            batch_id=batch.batch_id,
            run_id=batch.run_id,
            lanes=(),
        )
    reports: dict[int, LaneResult] = {}
    groups: dict[int, list[RunLane]] = {}
    for lane in batch.lanes:
        groups.setdefault(lane.launch_id, []).append(lane)

    for lanes in groups.values():
        scopes: list[LaneState] = []
        for lane in lanes:
            try:
                scopes.append(
                    _open_lane(
                        runtime,
                        batch,
                        lane,
                        prepared,
                        predicate_values,
                        graph_eligible,
                    )
                )
            except BaseException as error:
                classified = _classify_lane_failure(
                    runtime,
                    lane,
                    error,
                    phase="lane registration",
                )
                if propagate_errors or classified.fatal:
                    for scope in scopes:
                        _discard_lane(runtime, scope, classified)
                    raise classified
                reports[lane.lane_id] = _registration_error_lane(
                    runtime,
                    batch.run_id,
                    lane,
                    classified,
                    started,
                )

        if not scopes:
            continue
        try:
            outcomes, execution_errors = _execute_lane_group(
                runtime,
                tuple(scopes),
                qualify_mixed=propagate_errors,
            )
        except BaseException as error:
            classified = _classify_lane_failure(
                runtime,
                lanes[0],
                error,
                phase="lane execution",
            )
            for scope in scopes:
                _discard_lane(runtime, scope, classified)
            if propagate_errors or classified.fatal:
                raise classified
            for scope in scopes:
                reports[scope.lane.lane_id] = _error_lane(
                    runtime,
                    batch.run_id,
                    scope,
                    classified,
                    started,
                )
            continue

        if propagate_errors and execution_errors:
            first_lane = next(
                lane for lane in lanes if lane.lane_id in execution_errors
            )
            classified = _classify_lane_failure(
                runtime,
                first_lane,
                execution_errors[first_lane.lane_id],
                phase="lane execution",
            )
            for scope in scopes:
                _discard_lane(runtime, scope, classified)
            raise classified

        for scope in scopes:
            lane_error = execution_errors.get(scope.lane.lane_id)
            if lane_error is not None:
                classified = _classify_lane_failure(
                    runtime,
                    scope.lane,
                    lane_error,
                    phase="lane execution",
                )
                _discard_lane(runtime, scope, classified)
                if propagate_errors or classified.fatal:
                    raise classified
                reports[scope.lane.lane_id] = _error_lane(
                    runtime,
                    batch.run_id,
                    scope,
                    classified,
                    started,
                )
                continue
            lane_outcomes = outcomes[scope.lane.lane_id]
            try:
                reports[scope.lane.lane_id] = _commit_lane(
                    runtime,
                    batch.run_id,
                    scope,
                    lane_outcomes,
                    started,
                )
            except BaseException as error:
                if scope.publication_started:
                    classified = _published_lane_failure(
                        runtime,
                        scope.lane,
                        error,
                    )
                else:
                    classified = _classify_lane_failure(
                        runtime,
                        scope.lane,
                        error,
                        phase="lane commit",
                    )
                    _discard_lane(runtime, scope, classified)
                if propagate_errors or classified.fatal:
                    raise classified
                reports[scope.lane.lane_id] = _error_lane(
                    runtime,
                    batch.run_id,
                    scope,
                    classified,
                    started,
                )

    _apply_release_controls(runtime, batch, before_execution=False)
    report = RunResult(
        batch_id=batch.batch_id,
        run_id=batch.run_id,
        lanes=tuple(reports[lane.lane_id] for lane in batch.lanes),
    )
    runtime.trace.emit(
        ExecutionPhase.COMMIT,
        operations,
        duration_us=(time.perf_counter_ns() - started) // 1000,
    )
    return report


def _classify_lane_failure(
    runtime,
    lane: RunLane,
    error: BaseException,
    *,
    phase: str,
) -> WorkerError:
    """Classify a pre-publication lane failure with complete operation and route context."""

    operations = tuple(
        (
            int(operation.request_key.authority_id),
            int(operation.request_key.request_id),
            int(operation.request_key.epoch),
            int(operation.op_id),
        )
        for operation in lane.operations
    )
    sole = lane.operations[0] if len(lane.operations) == 1 else None
    classified = classify(
        error,
        context=phase,
        phase=phase,
        operations=operations,
        req_id=None if sole is None else int(sole.request_key.request_id),
        op_id=None if sole is None else int(sole.op_id),
        op_kind=None if sole is None else sole.kind.value,
        route=str(lane.route),
    )
    _log_lane_failure(runtime, lane, classified, cause=error)
    return classified


def _published_lane_failure(
    runtime,
    lane: RunLane,
    error: BaseException,
) -> WorkerError:
    """Classify a post-visibility publication failure as a fatal invariant violation."""

    operations = tuple(
        (
            int(operation.request_key.authority_id),
            int(operation.request_key.request_id),
            int(operation.request_key.epoch),
            int(operation.op_id),
        )
        for operation in lane.operations
    )
    classified = WorkerError(
        code=WorkerErrorCode.INVARIANT_VIOLATION,
        message=f"lane publication failed after visibility began: {error}",
        fatal=True,
        phase="lane publication",
        route=str(lane.route),
        operations=operations,
    )
    _log_lane_failure(runtime, lane, classified, cause=error)
    return classified


def _log_lane_failure(
    runtime,
    lane: RunLane,
    error: WorkerError,
    *,
    cause: BaseException | None = None,
) -> None:
    """Log a classified lane failure with traceback only for diagnostic error classes."""

    capture_trace = should_capture_trace(error.code)
    log = logger.error if capture_trace else logger.warning
    log(
        "lane failed: %s [code=%s lane_id=%s route=%s operations=%s]",
        error.message,
        error.code,
        lane.lane_id,
        lane.route,
        error.operations,
        exc_info=(type(cause), cause, cause.__traceback__)
        if capture_trace and cause is not None
        else None,
    )


def _open_lane(
    runtime,
    batch: Run,
    lane: RunLane,
    prepared: tuple[PreparedTransferInput, ...],
    predicate_values: Mapping[OperationIdentity, bool],
    graph_eligible: bool,
) -> LaneState:
    """Stage one lane's speculative state, resources, inputs, and completion storage."""

    operations = lane.operations

    # Predicated rows remain in aligned output/state tables but do not reserve
    # execution-only inputs, CPU tasks, or model resources.
    predicated = frozenset(
        identity
        for operation in operations
        if (identity := _operation_identity(operation)) in predicate_values
        and not predicate_values[identity]
    )
    active_operations = (
        operations
        if not predicated
        else tuple(
            operation
            for operation in operations
            if _operation_identity(operation) not in predicated
        )
    )
    # Restrict admissions and input payloads to identities declared by this lane.
    traced = _trace_envelopes(operations)
    started = time.perf_counter_ns()
    request_keys = {operation.request_key for operation in operations}
    admissions = tuple(
        admission for admission in batch.admissions if admission.request_key in request_keys
    )
    declared_inputs = {reference for operation in operations for reference in operation.inputs}
    declared_inputs.update(
        operation.predicate for operation in operations if operation.predicate is not None
    )
    input_products = tuple(
        payload for payload in batch.input_products if payload.product in declared_inputs
    )
    completion: OutputBuffer | None = None
    try:
        # Candidate drafts and completion slots form a speculative ownership unit:
        # either all later lane resources bind successfully or both are discarded.
        admission_slots = {
            admission.request_key: int(admission.request_pool_idx) for admission in admissions
        }
        request_pool_indices: list[int] = []
        for operation in operations:
            slot = admission_slots.get(operation.request_key)
            resident = runtime.requests.peek(operation.request_key.request_id)
            if (
                slot is None
                and resident is not None
                and resident.request_key == operation.request_key
            ):
                slot = int(resident.request_pool_idx)
            if slot is None:
                raise invalid_descriptor("operation request is not resident or admitted")
            request_pool_indices.append(slot)
        candidates, bases = runtime.requests.stage_lane(
            operations,
            admissions,
            tuple(request_pool_indices),
        )
        for operation, request in zip(operations, candidates, strict=True):
            request.install_runtime(_parent_runtime(runtime, operation, request))
        completion = runtime._outputs.acquire(
            len(operations),
            token_capacity=_lane_completion_words(runtime, operations),
            devices=_completion_devices(runtime, operations),
        )
    except BaseException as error:
        if completion is not None:
            completion.abandon()
        runtime.trace.emit(
            ExecutionPhase.CANDIDATE_STAGE,
            traced,
            duration_us=(time.perf_counter_ns() - started) // 1000,
            error=error,
        )
        raise
    assert completion is not None
    scope = LaneState(
        lane=lane,
        started_ns=started,
        graph_eligible=graph_eligible,
        request_candidates=candidates,
        request_bases=bases,
        request_rows={request.request_id: request for request in candidates},
        completion=completion,
        admissions={admission.request_key: admission for admission in admissions},
        prepared_transfers={
            transfer.product: transfer
            for transfer in prepared
            if transfer.product in declared_inputs
        },
        predicated_operations=predicated,
    )
    runtime.trace.emit(
        ExecutionPhase.CANDIDATE_STAGE,
        traced,
        duration_us=(time.perf_counter_ns() - started) // 1000,
    )
    try:
        # Bind physical state in dependency order before decoding transferred inputs.
        if runtime.runtime_states is not None:
            runtime.runtime_states.reset(
                tuple(
                    int(request.request_pool_idx)
                    for request, base in zip(candidates, bases, strict=True)
                    if base is None
                )
            )
        _reserve_cpu_tasks(runtime, active_operations, scope)
        active_lane = _active_lane(runtime, lane, active_operations)
        if active_lane is not None:
            if runtime.cache_pool is None or runtime.req_to_token_pool is None:
                if (
                    active_lane.block_tables
                    or active_lane.new_cache_pages
                    or active_lane.forward_rows
                ):
                    raise unsupported_setup(
                        "KV-free execution received cache tables or packed forward rows"
                    )
            else:
                _bind_cache_tables(runtime, active_lane, scope)
        # Preserve operation/request row alignment for forward packing and commit.
        scope.layout = LaneLayout(
            operations=operations,
            requests=candidates,
            seq_lens=tuple(
                int(_parent_runtime(runtime, operation, request).kv_visible_len)
                for operation, request in zip(operations, candidates, strict=True)
            ),
            weights=(
                _weights(
                    runtime,
                ),
            )
            * len(candidates),
            identities=tuple(_operation_identity(operation) for operation in operations),
        )
        if active_lane is not None:
            _bind_latent_rows(runtime, active_lane, scope)
        _reserve_outputs(runtime, operations, scope)

        # Only live operations consume inputs; predicated outputs are published
        # directly into their aligned completion rows.
        active_inputs = {
            reference for operation in active_operations for reference in operation.inputs
        }
        active_inputs.update(
            operation.predicate
            for operation in active_operations
            if operation.predicate is not None
        )
        _stage_input_products(
            runtime,
            tuple(payload for payload in input_products if payload.product in active_inputs),
            scope,
        )
        _consume_predicates(runtime, active_operations, scope)
        _publish_predicated_outputs(runtime, operations, scope)
        scope.registration_visible = True
        _record_component(scope, "open_lane", started)
        return scope
    except BaseException:
        _discard_lane(runtime, scope)
        raise


def _active_lane(
    runtime,
    lane: RunLane,
    operations: tuple[Operation, ...],
) -> RunLane | None:
    """Rebuild lane-indexed rows and placements after predicated operations are removed."""

    if not operations:
        return None
    if operations is lane.operations:
        return lane
    identities = {_operation_identity(operation) for operation in operations}
    old_to_new = {
        index: selected
        for selected, (index, operation) in enumerate(
            (
                item
                for item in enumerate(lane.operations)
                if _operation_identity(item[1]) in identities
            )
        )
    }
    return replace(
        lane,
        operations=operations,
        forward_rows=tuple(
            replace(row, operation_index=old_to_new[row.operation_index])
            for row in lane.forward_rows
            if row.operation_index in old_to_new
        ),
        latent_placements=tuple(
            placement
            for placement in lane.latent_placements
            if (placement.request_key, int(placement.op_id)) in identities
        ),
    )


def _lane_completion_words(runtime, operations: tuple[Operation, ...]) -> int:
    """Compute fixed completion-word capacity for all operations in a lane."""

    return max(
        1,
        SAMPLING_COMPLETION_FIELDS * len(operations)
        + sum((int(operation.bounds.max_completion_bytes) + 3) // 4 for operation in operations),
    )


def _execute_lane_group(
    runtime,
    scopes: tuple[LaneState, ...],
    *,
    qualify_mixed: bool,
) -> tuple[dict[int, tuple[Outcome, ...]], dict[int, BaseException]]:
    """Execute active operations across lanes and align outcomes with original lane order."""

    from . import token

    for scope in scopes:
        active = tuple(
            operation
            for operation in scope.lane.operations
            if _operation_identity(operation) not in scope.predicated_operations
        )
        for device in _completion_devices(runtime, active):
            scope.completion.begin_device(device)
    grouped: list[list[Outcome | None]] = [
        [None] * len(scope.lane.operations) for scope in scopes
    ]
    group_active = tuple(
        operation
        for scope in scopes
        for operation in scope.lane.operations
        if _operation_identity(operation) not in scope.predicated_operations
    )
    homogeneous_decode = bool(group_active) and all(
        operation.kind is RunKind.AR_DECODE for operation in group_active
    )
    states: list[OperationState] = []
    locations: dict[int, tuple[int, int]] = {}
    for scope_index, scope in enumerate(scopes):
        active = tuple(
            operation
            for operation in scope.lane.operations
            if _operation_identity(operation) not in scope.predicated_operations
        )
        if homogeneous_decode:
            decoded = token.decode_batch(runtime, active, scope)
            decoded_by_identity = dict(
                zip(
                    (_operation_identity(operation) for operation in active),
                    decoded,
                    strict=True,
                )
            )
        else:
            decoded_by_identity = {}
        for operation_index, operation in enumerate(scope.lane.operations):
            if _operation_identity(operation) in scope.predicated_operations:
                grouped[scope_index][operation_index] = _predicated_outcome(
                    runtime,
                    operation,
                    scope,
                )
                continue
            identity = _operation_identity(operation)
            if identity in decoded_by_identity:
                grouped[scope_index][operation_index] = decoded_by_identity[identity]
                continue
            state = OperationState(operation=operation, lane=scope)
            locations[id(state)] = (scope_index, operation_index)
            states.append(state)
    errors = _run_ready_set(runtime, states, qualify_mixed=qualify_mixed)
    for state in states:
        scope_index, operation_index = locations[id(state)]
        if state.outcome is not None:
            grouped[scope_index][operation_index] = state.outcome
    outcomes: dict[int, tuple[Outcome, ...]] = {}
    for scope, lane_outcomes in zip(scopes, grouped, strict=True):
        lane_id = scope.lane.lane_id
        if lane_id in errors:
            continue
        if any(outcome is None for outcome in lane_outcomes):
            raise RuntimeError("successful lane did not resolve every operation")
        outcomes[lane_id] = tuple(cast(Outcome, outcome) for outcome in lane_outcomes)
    return outcomes, errors


def _run_ready_set(
    runtime: ExecutionResources,
    states: list[OperationState],
    *,
    qualify_mixed: bool,
) -> dict[int, BaseException]:
    """Advance dependency-ready operations through forward, sampling, integration, and actions."""

    from . import encode, flow, token, transfer

    # Products define the in-lane dependency graph; failures suppress only the
    # affected lane while independent lanes continue through the ready set.
    producers = {output: state for state in states for output in state.operation.outputs}
    errors: dict[int, BaseException] = {}

    def live(state: OperationState) -> bool:
        """Select unresolved operations whose lane has not recorded a failure."""

        return state.outcome is None and state.lane.lane.lane_id not in errors

    while any(live(state) for state in states):
        # Pack all ready neural work first. Flow prefix preparation forms an
        # exclusive wave because it may change the rows available to peers.
        forward: list[tuple[OperationState, object]] = []
        ready = tuple(
            state for state in states if live(state) and dependencies_ready(state, producers)
        )
        flow_ready = tuple(
            state for state in ready if state.operation.kind is RunKind.DIFFUSION_STEP
        )
        flow_ready_ids = {id(state) for state in flow_ready}
        for state in flow_ready:
            try:
                rows = _pack_state_forward(runtime, state)
            except BaseException as error:
                errors[state.lane.lane.lane_id] = error
                continue
            forward.extend((state, row) for row in rows)
        preparing_flow_prefix = any(
            live(state) and state.phase == "prefix_pending" for state in flow_ready
        )
        if not preparing_flow_prefix:
            for state in ready:
                if id(state) in flow_ready_ids or not live(state):
                    continue
                try:
                    rows = _pack_state_forward(runtime, state)
                except BaseException as error:
                    errors[state.lane.lane.lane_id] = error
                    continue
                forward.extend((state, row) for row in rows)
        if forward:
            if runtime.runner is None:
                raise RuntimeError("KV-free execution packed a model forward row")
            outputs = _run_laneed_wave(
                runtime,
                tuple((cast(ForwardRow, row), state.lane) for state, row in forward),
                qualify_mixed=qualify_mixed,
            )
            aligned: dict[int, list[torch.Tensor]] = defaultdict(list)
            ordered: list[OperationState] = []
            for (state, _row), output in zip(forward, outputs, strict=True):
                if id(state) not in aligned:
                    ordered.append(state)
                aligned[id(state)].append(output)
            for state in ordered:
                try:
                    _consume_state_forward(runtime, state, tuple(aligned[id(state)]))
                except BaseException as error:
                    errors[state.lane.lane.lane_id] = error
            continue

        # Sampling runs after its logits dependencies land and publishes token
        # products that can unlock later operations in the same lane.
        samples: dict[int, list[tuple[OperationState, SampleWork]]] = defaultdict(list)
        for state in states:
            if not live(state) or not dependencies_ready(state, producers):
                continue
            sample = token.pack_sample(state)
            if sample is not None:
                samples[state.lane.lane.lane_id].append(
                    (state, cast(SampleWork, sample))
                )
        if samples:
            for lane_id, candidates in samples.items():
                state = candidates[0][0]
                try:
                    values = _sample_task_batch(
                        tuple(sample for _state, sample in candidates),
                        state.lane.completion,
                        device_products=runtime.device_products,
                        device_reads=tuple(state.lane.device_reads),
                        selection_broadcast=partial(_broadcast_tp_selection, runtime),
                    )
                    for (candidate, _sample), value in zip(candidates, values, strict=True):
                        token.consume_sample(runtime, candidate, value)
                except BaseException as error:
                    errors[lane_id] = error
            continue

        # Integration consumes velocity outputs without another model call.
        # Other host/device actions run only when no forward or sample is ready.
        progressed = False
        for state in states:
            if not live(state) or not dependencies_ready(state, producers):
                continue
            try:
                progressed = flow.integrate(runtime, state) or progressed
            except BaseException as error:
                errors[state.lane.lane.lane_id] = error
        if progressed:
            continue
        for state in states:
            if not live(state) or not dependencies_ready(state, producers):
                continue
            try:
                progressed = encode.run_action(runtime, state) or progressed
                progressed = transfer.run_action(runtime, state) or progressed
                progressed = runtime.model.run_operation(runtime, state) or progressed
            except BaseException as error:
                errors[state.lane.lane.lane_id] = error
        if progressed:
            continue

        # Reaching a live fixed point indicates a dependency or state-machine
        # invariant violation rather than ordinary asynchronous waiting.
        if not any(live(state) for state in states):
            break
        blocked = tuple(_operation_identity(state.operation) for state in states if live(state))
        raise RuntimeError(f"execution ready set made no progress: {blocked!r}")
    return errors


def _pack_state_forward(
    runtime: ExecutionResources,
    state: OperationState,
) -> tuple[object, ...]:
    """Dispatch an operation state to its token, flow, or encoder forward packer."""

    from . import encode, flow, token

    operation = state.operation
    if operation.kind.token_mode is not None:
        return token.pack_forward(runtime, state)
    if operation.kind is RunKind.DIFFUSION_STEP and runtime.latent_pool is not None:
        return flow.pack_forward(runtime, state)
    if operation.kind.encode_mode is not None or (
        operation.kind is RunKind.DIFFUSION_FINALIZE and runtime.latent_pool is not None
    ):
        return encode.pack_forward(runtime, state)
    return ()


def _consume_state_forward(
    runtime: ExecutionResources,
    state: OperationState,
    outputs: tuple[torch.Tensor, ...],
) -> None:
    """Dispatch aligned model outputs to the operation family's consumer."""

    from . import encode, flow, token

    operation = state.operation
    if operation.kind.token_mode is not None:
        token.consume_forward(runtime, state, outputs)
    elif operation.kind is RunKind.DIFFUSION_STEP and runtime.latent_pool is not None:
        flow.consume_forward(runtime, state, outputs)
    elif operation.kind.encode_mode is not None or (
        operation.kind is RunKind.DIFFUSION_FINALIZE and runtime.latent_pool is not None
    ):
        encode.consume_forward(runtime, state, outputs)
    else:
        raise RuntimeError("model output has no operation consumer")


def _commit_lane(
    runtime,
    run_id: int,
    scope: LaneState,
    outcomes: tuple[Outcome, ...],
    started: int,
) -> LaneResult:
    """Atomically publish validated lane resources, request versions, and output records."""

    commit_started = time.perf_counter_ns()
    lane = scope.lane
    operations = lane.operations

    # All device reads must finish and every staged resource must validate before
    # completion storage becomes immutable or any publication becomes visible.
    _finish_device_reads(runtime, scope)
    _publish_predicates(runtime, scope)
    runtime.device_products.validate_writes(tuple(scope.device_writes))
    runtime.encoder_cache.validate_writes(tuple(scope.encoder_writes))
    if runtime.latent_pool is None:
        if scope.latent_publications or scope.latent_releases:
            raise RuntimeError("latent publication has no physical pool")
    else:
        runtime.latent_pool.validate_commit(
            scope.latent_publications,
            scope.latent_releases,
        )
    scope.completion.seal()
    # Build the complete next-version description without mutating resident state.
    records: list[PendingOutput] = []
    selected_versions: dict[int, Checkpoint] = {}
    pending_completions: dict[int, CompletionState] = {}
    speculative_commits: dict[int, SpeculativeCommit] = {}
    report_products: list[ProductPayload] = []
    resolved_runtime: dict[int, RequestRuntime] = {}
    layout = scope.layout
    if layout is None or layout.operations != operations:
        raise RuntimeError("lane commit lost its aligned candidate layout")
    for row, (operation, request, outcome) in enumerate(
        zip(
            operations,
            layout.requests,
            outcomes,
            strict=True,
        )
    ):
        _validate_completion_products(runtime, operation, outcome.products)
        if int(runtime.deployment.tp_rank) == 0:
            report_products.extend(outcome.products)
        pending = PendingOutput(
            (
                request.pending_operations.get(int(operation.parent.op_id))
                if isinstance(operation.parent.point, DeviceSelected)
                else None
            ),
            scope.completion,
            row,
            partial(_finalize_predicated_runtime, runtime, operation),
            status=outcome.status,
            selected_point=cast(int, outcome.selected_point),
            completion_tasks=(
                *outcome.completion_tasks,
                *(
                    cast(LogprobPayload, product.payload)
                    for product in outcome.products
                    if isinstance(product.payload, LogprobPayload)
                ),
            ),
        )
        record = pending.bind_record(
            OutputRecord(
                request_key=operation.request_key,
                op_id=operation.op_id,
                kind=operation.kind,
                completion_slot_generation=scope.completion.generation,
                status=outcome.status,
                selected_point=cast(int, outcome.selected_point),
                logical_lengths=outcome.logical_lengths,
                token_span=outcome.token_span,
                committed_tokens=outcome.committed_tokens,
                sampling=outcome.sampling,
                finish_flags=outcome.finish_flags,
                product_generations=outcome.product_generations,
                error_code=None,
                next_cursor=outcome.next_cursor,
                done=outcome.done,
            )
        )
        records.append(record)
        if operation.advances_state:
            pending_completions[operation.request_key.request_id] = pending
            if outcome.status is OpStatus.PREDICATED:
                selected = request.resolve_version(operation.parent)
                if selected is None:
                    raise RuntimeError("predicated operation lost its selected parent")
                selected_versions[operation.request_key.request_id] = selected
            else:
                selected_versions[operation.request_key.request_id] = Checkpoint(
                    op_id=operation.op_id,
                    point=FixedCheckpoint(cast(int, outcome.selected_point)),
                )
        else:
            selected = request.resolve_version(operation.parent)
            if selected is None:
                raise RuntimeError("non-state operation lost its resolved parent")
            selected_versions[operation.request_key.request_id] = selected
        resolved_runtime[operation.request_key.request_id] = RequestRuntime(
            logical_position=request.logical_position,
            rng_counter=request.rng_counter,
            latent_product=request.latent_product,
            flow_step=request.flow_step,
            kv_visible_len=outcome.logical_lengths.kv_visible_len,
            kv_computed_len=outcome.logical_lengths.kv_computed_len,
        )
        selection = outcome.selection
        if selection is not None:
            speculative_commits[operation.request_key.request_id] = SpeculativeCommit(
                draft_tokens=selection.draft_tokens,
                terminal_prefix=selection.terminal_prefix,
                base_logical_position=selection.base_logical_position,
                base_rng_counter=selection.base_rng_counter,
                base_kv_visible=selection.base_kv_visible,
                initialized_kv=selection.initialized_kv,
            )
    _record_component(scope, "commit_lane", commit_started)

    # Prepare cross-resource commit records first so no publication is visible
    # until every participating owner has accepted its state transition.
    lane_report = LaneResult(
        lane_id=lane.lane_id,
        completions=tuple(records),
        products=tuple(report_products),
        registration=RegistrationAck(visible=True),
        worker_exec_us=(time.perf_counter_ns() - scope.started_ns) // 1000,
        forward_stats=_forward_stats(scope.observations, scope.component_us),
    )
    cache_publications = runtime.cache_publications
    if cache_publications is None:
        if scope.cache_publications or scope.cache_installations:
            raise RuntimeError("cache publication has no backing KV resources")
        cache_commit = ()
    else:
        cache_commit = cache_publications.prepare_commit(
            scope.cache_publications,
            scope.cache_installations,
            runtime.transport,
        )
    request_publication = runtime.requests.prepare_publication(
        run_id=run_id,
        operations=operations,
        candidates=scope.request_candidates,
        bases=scope.request_bases,
        selected_versions=selected_versions,
        runtimes=resolved_runtime,
        completions=pending_completions,
        speculative=speculative_commits,
    )
    for identity, locators in scope.stage_publications.items():
        existing = runtime._transport_publications.get(identity)
        if existing is not None and existing != locators:
            raise RuntimeError("committed transport publication identity was reused")
    # From this point the lane cannot be discarded: apply resource commits, then
    # reserve the request publication that gates successor readiness.
    scope.publication_started = True
    runtime.device_products.commit_writes(tuple(scope.device_writes))
    runtime.encoder_cache.commit_writes(tuple(scope.encoder_writes))
    if runtime.latent_pool is not None:
        runtime.latent_pool.apply_commit(
            scope.latent_publications,
            scope.latent_releases,
        )
    if cache_publications is not None:
        cache_publications.apply_commit(cache_commit)
    for publication_identity, locators in scope.stage_publications.items():
        runtime._transport_publications[publication_identity] = locators
    _commit_runtime_states(runtime, scope)
    request_publication.reserve()
    return replace(lane_report, publication=request_publication)


def _commit_runtime_states(runtime, scope: LaneState) -> None:
    """Publish committed token, predicate, position, cache-length, and penalty state to device rows."""

    states = runtime.runtime_states
    if states is None:
        if (
            scope.runtime_publications
            or scope.prompt_logits_publications
            or scope.runtime_cache_lengths
        ):
            raise RuntimeError("runtime state publication has no backing storage")
        return

    # Cache lengths may advance without token publication, so apply their
    # scalar updates before the row-level decode state transitions.
    for slot, length in scope.runtime_cache_lengths.items():
        _copy_runtime_scalar(states.valid_cache_lengths[slot : slot + 1], length)
    for publication in scope.runtime_publications:
        if isinstance(publication, DecodeRuntimePublication):
            # Batched decode uses the fused device-state kernel, then updates
            # request-owned penalty counts only for valid active selections.
            states.publish_decode(
                publication.slots,
                device_indices=publication.device_slots,
                tokens=publication.tokens,
                predicates=publication.predicates,
                selected_points=publication.selected_points,
            )
            for index, penalty_base in enumerate(publication.penalty_bases):
                if penalty_base is None:
                    continue
                weight = (
                    publication.valid[index : index + 1] & publication.active[index : index + 1]
                ).to(dtype=penalty_base.dtype)
                penalty_base.scatter_add_(
                    0,
                    publication.tokens[index : index + 1].to(dtype=torch.int64),
                    weight,
                )
            continue

        # Non-batched publications update the same fields explicitly while
        # stripping the continuation tag from future input tokens.
        slot = publication.slot
        future_token = states.future_input_tokens[slot, :1]
        future_token.copy_(publication.token.reshape(-1)[:1])
        future_token.bitwise_and_(TOKEN_VALUE_MASK)
        states.predicates[slot : slot + 1].copy_(
            publication.predicate.reshape(-1)[:1].to(dtype=torch.bool)
        )
        states.selected_points[slot : slot + 1].copy_(
            publication.selected_point.reshape(-1)[:1].to(dtype=torch.int32)
        )
        _copy_runtime_scalar(
            states.logical_lengths[slot : slot + 1],
            publication.logical_position,
        )
        _copy_runtime_scalar(
            states.sampling_positions[slot : slot + 1],
            publication.sampling_position,
        )
        penalty_base = publication.penalty_base
        if penalty_base is not None:
            weight = (publication.valid.reshape(-1)[:1] & publication.active.reshape(-1)[:1]).to(
                dtype=penalty_base.dtype
            )
            penalty_base.scatter_add_(
                0,
                future_token.to(dtype=torch.int64),
                weight,
            )

    # Prompt logits have request-row lifetime and become visible only after all
    # scalar transition fields for the lane are committed.
    for prompt_publication in scope.prompt_logits_publications:
        states.prompt_logits[prompt_publication.slot].copy_(
            prompt_publication.logits.to(dtype=states.prompt_logits.dtype)
        )


def _discard_lane(
    runtime,
    scope: LaneState,
    error: BaseException | None = None,
) -> None:
    """Release all provisional lane resources that have not crossed publication visibility."""

    _finish_device_reads(runtime, scope)
    for reservation in scope.cpu_tasks.values():
        reservation.abandon()
    for lease in scope.media_output_leases.values():
        lease.release()
    if scope.publication_started:
        raise RuntimeError("published lane state cannot be discarded")
    if scope.admissions:
        admissions = tuple(scope.admissions.values())
        runtime.requests.abort_model_admissions(admissions)
        if runtime._media_mux is not None:
            for admission in admissions:
                runtime._media_mux.drop(int(admission.request_key.request_id))
    scope.completion.abandon()
    runtime.device_products.abandon_writes(tuple(scope.device_writes))
    runtime.encoder_cache.abandon_writes(tuple(scope.encoder_writes))
    if runtime.latent_pool is not None and scope.latent_import_slots:
        runtime.latent_pool.release_slots(tuple(scope.latent_import_slots))
    _release_locators(runtime, scope.published)
    runtime.trace.emit(
        ExecutionPhase.CANDIDATE_DISCARD,
        _trace_envelopes(scope.lane.operations),
        error=error,
    )


def _reserve_cpu_tasks(
    runtime,
    operations: tuple[Operation, ...],
    scope: LaneState,
) -> None:
    """Reserve bounded CPU slots for active operations that schedule host-side work."""

    video_model = isinstance(runtime.model, VideoRunner)
    rank_zero = runtime.mesh.coord("sp") == 0 if video_model else True
    for operation in operations:
        if operation.kind is not RunKind.DIFFUSION_FINALIZE and not (
            video_model and operation.kind is RunKind.DIFFUSION_DECODE
        ):
            continue
        if not rank_zero:
            continue
        identity = _operation_identity(operation)
        if identity in scope.cpu_tasks:
            raise invalid_descriptor("materialization repeats its CPU task identity")
        reservation = runtime._cpu_tasks.reserve()
        try:
            if video_model and operation.kind is RunKind.DIFFUSION_DECODE:
                placement = next(
                    (
                        placement
                        for placement in scope.lane.decode_placements
                        if placement.request_key == operation.request_key
                        and int(placement.op_id) == int(operation.op_id)
                    ),
                    None,
                )
                if placement is None:
                    raise invalid_descriptor(
                        "video decode operation has no exact decode placement"
                    )
                request = runtime.request_row(scope, operation.request_key.request_id)
                slot = runtime.requests.model_state_slot(request.request_pool_idx)
                kind = runtime.model.decode_kind(slot, int(placement.cursor))
                if kind not in {DecodeKind.VIDEO, DecodeKind.AUDIO}:
                    raise invalid_descriptor(
                        "video decode placement does not capture media output"
                    )
                scope.media_output_leases[identity] = runtime.media_output_ring().reserve(
                    kind.value
                )
        except BaseException:
            reservation.abandon()
            raise
        scope.cpu_tasks[identity] = reservation


def _registration_error_lane(
    runtime,
    run_id: int,
    lane: RunLane,
    error: WorkerError,
    started: int,
) -> LaneResult:
    """Build aligned error outputs without committing candidate request state."""

    generation = 1
    report = _build_error_lane(
        runtime,
        lane,
        generation,
        False,
        error,
        started,
        WorkerForwardStats(),
    )
    return report


def _error_lane(
    runtime,
    run_id: int,
    scope: LaneState,
    error: WorkerError,
    started: int,
) -> LaneResult:
    """Build a failed lane report and release its unpublished resources."""

    report = _build_error_lane(
        runtime,
        scope.lane,
        scope.completion.generation,
        scope.registration_visible,
        error,
        scope.started_ns,
        _forward_stats(scope.observations, scope.component_us),
    )
    return report


def _build_error_lane(
    runtime,
    lane: RunLane,
    generation: int,
    registration_visible: bool,
    error: WorkerError,
    started: int,
    forward_stats: WorkerForwardStats,
) -> LaneResult:
    """Materialize one error completion per lane operation without mutating request state."""

    completion_code = _completion_error_code(error.code)
    records: list[ModelOutput] = []
    for operation in lane.operations:
        # Resolve only enough parent state to preserve the scheduler-visible
        # checkpoint and logical lengths in the failed completion.
        request = runtime.requests.peek(operation.request_key.request_id)
        selected_parent = (
            operation.parent
            if operation.parent.is_fixed()
            else None
            if request is None
            else request.resolve_version(operation.parent)
        )
        point = None if selected_parent is None else selected_parent.point
        selected_point = point.point_index if isinstance(point, FixedCheckpoint) else 0
        lengths = (
            LogicalLengths()
            if request is None
            else _logical_lengths(runtime, operation, request, None)
        )
        payload_type = (
            ArResult
            if operation.kind in {RunKind.AR_EXTEND, RunKind.AR_DECODE, RunKind.AR_VERIFY}
            else EncoderResult
            if operation.kind in {RunKind.ENCODER_VISION, RunKind.ENCODER_LATENT}
            else DiffusionResult
            if operation.kind in {
                RunKind.DIFFUSION_PREPARE,
                RunKind.DIFFUSION_STEP,
                RunKind.DIFFUSION_DECODE,
                RunKind.DIFFUSION_FINALIZE,
            }
            else TransferResult
        )

        # All result families share an empty token span. Diffusion additionally
        # carries its cursor fields so the wire payload remains schema-complete.
        payload_args = (
            lengths,
            TokenSpan(base=lengths.token_len, len=0),
            (),
            FinishFlags(),
            None,
        )
        payload = (
            DiffusionResult(*payload_args, next_cursor=0, done=False)
            if payload_type is DiffusionResult
            else payload_type(*payload_args)
        )
        placeholder = ModelOutput(
            request_key=operation.request_key,
            op_id=operation.op_id,
            completion_slot_generation=max(1, generation),
            status=OpStatus.ERROR,
            selected_point=selected_point,
            product_generations=(),
            error_code=completion_code,
            timing_counters=TimingCounters(),
            payload=payload,
        )
        records.append(placeholder)
    return LaneResult(
        lane_id=lane.lane_id,
        completions=tuple(records),
        registration=RegistrationAck(visible=registration_visible),
        worker_exec_us=(time.perf_counter_ns() - started) // 1000,
        forward_stats=forward_stats,
    )


def _finalize_predicated_runtime(
    runtime,
    operation: Operation,
) -> tuple[Checkpoint, RequestRuntime]:
    """Resolve request runtime state for an operation skipped by its predicate."""

    selected, runtime = runtime.requests.resolve_predicated(
        operation.request_key.request_id,
        operation.op_id,
        operation.parent,
    )
    return selected, runtime


def _validate_batch(runtime, batch: Run) -> None:
    """Validate run identity, lane resources, routing, and operation support before staging."""

    if len(batch.operations) > runtime.deployment.max_batch_operations:
        raise invalid_descriptor("execution batch exceeds the deployment operation limit")
    for operation in batch.operations:
        variant = operation.kind
        if variant not in runtime.allowed_work_variants:
            raise unsupported_operation(variant.value, operation.request_key.request_id)
    if any(
        index > runtime.deployment.max_request_pool_size
        for lane in batch.lanes
        for index in (
            *(table.request_pool_idx for table in lane.block_tables),
            *(row.request_pool_index for row in lane.forward_rows),
        )
    ):
        raise invalid_descriptor("execution batch exceeds request-slot capacity")
    groups: dict[int, list[RunLane]] = defaultdict(list)
    for lane in batch.lanes:
        groups[lane.launch_id].append(lane)
    for lanes in groups.values():
        if len(lanes) < 2:
            continue
        variants = {
            operation.kind for lane in lanes for operation in lane.operations
        }
        if not runtime.model.tensorized_mixed or variants != {
            RunKind.AR_DECODE,
            RunKind.DIFFUSION_STEP,
        }:
            raise invalid_descriptor(
                "tensorized mixed submission exceeds the supported mixed buckets"
            )
        bucket = _mixed_bucket(runtime, tuple(lanes))
        if bucket not in runtime.mixed_buckets:
            raise invalid_descriptor("tensorized mixed submission has no exact qualified bucket")
    runtime.model.validate_run(runtime, batch)
    validate_collective_sequence(runtime.mesh, runtime._collective_history, batch)


def validate_collective_sequence(
    mesh: DeviceMesh,
    history: OrderedDict[int, object],
    batch: Run,
) -> None:
    """Reject divergent or non-advancing collective identities across all worker roots."""

    if not batch.operations:
        return
    collective_seq = int(batch.collective_seq)
    collective_identity = (collective_seq, batch.operations)
    existing = history.get(collective_seq)
    if existing is not None:
        if existing != collective_identity:
            raise invalid_descriptor("collective sequence was reused with different work")
        return
    if mesh.tp_size > 1 and history and collective_seq <= next(reversed(history)):
        raise invalid_descriptor("collective sequence does not advance")
    history[collective_seq] = collective_identity
    while len(history) > 4096:
        history.popitem(last=False)


def _completion_devices(runtime, operations: tuple[Operation, ...]) -> tuple[str, ...]:
    """List distinct devices that may contribute asynchronous completion fields."""

    deployment = runtime.deployment
    generation_device = deployment.generation_device
    device = deployment.device
    selected: list[str] = []
    for operation in operations:
        target = (
            generation_device
            if generation_device is not None and operation.kind in _GENERATION_WORK_VARIANTS
            else device
        )
        if target not in selected:
            selected.append(target)
    return tuple(selected)


def _mixed_bucket(
    runtime,
    lanes: tuple[RunLane, ...],
) -> GraphBucket:
    """Resolve a shared captured-graph bucket for a compatible mixed lane group."""

    decode_rows = sum(
        operation.kind is RunKind.AR_DECODE
        for lane in lanes
        for operation in lane.operations
    )
    flow_operations = tuple(
        operation
        for lane in lanes
        for operation in lane.operations
        if operation.kind is RunKind.DIFFUSION_STEP
    )
    flow_placements = {
        (placement.request_key, int(placement.op_id)): placement
        for lane in lanes
        for placement in lane.latent_placements
    }
    branch_counts: dict[tuple[RequestKey, int], int] = defaultdict(int)
    generation = runtime.model.generation
    if flow_operations and generation is None:
        raise invalid_descriptor("tensorized mixed flow has no generation runtime")
    for lane in lanes:
        for index, operation in enumerate(lane.operations):
            placement = flow_placements.get((operation.request_key, int(operation.op_id)))
            query_len = (
                None
                if placement is None or generation is None
                else generation.physical_tokens(int(placement.height), int(placement.width))
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
            int(flow_placements[(operation.request_key, int(operation.op_id))].height),
            int(flow_placements[(operation.request_key, int(operation.op_id))].width),
            branch_counts[(operation.request_key, int(operation.op_id))],
        )
        for operation in flow_operations
        if (operation.request_key, int(operation.op_id)) in flow_placements
    }
    if len(geometries) != 1 or len(flow_placements) != len(flow_operations):
        raise invalid_descriptor("tensorized mixed flow rows disagree on physical geometry")
    height, width, cfg_branches = next(iter(geometries))
    return GraphBucket(
        decode_rows=decode_rows,
        flow_rows=len(flow_operations),
        height=height,
        width=width,
        cfg_branches=cfg_branches,
    )


def _reserve_outputs(
    runtime,
    operations: tuple[Operation, ...],
    scope: LaneState,
) -> None:
    """Bind each declared device value to its concrete bounded owner."""

    from . import transfer

    scalar_groups: dict[
        tuple[torch.device, ProductKind, DType, ShapeBound],
        list[tuple[ProductRef, torch.device | str]],
    ] = {}
    general_bindings: list[tuple[ProductRef, torch.device | str]] = []
    persistent_bindings: list[tuple[ProductRef, torch.device | str]] = []
    encoder_bindings: list[tuple[ProductRef, torch.device | str]] = []
    for operation in operations:
        device = _operation_device(runtime, operation)
        for output in operation.outputs:
            if (
                _operation_identity(operation) in scope.predicated_operations
                and output.kind is not ProductKind.COMPLETION
            ):
                continue
            if operation.kind is RunKind.TRANSFER_PRODUCT and transfer.transferable(output):
                continue
            if output.kind in {
                ProductKind.VISION_FEATURE,
                ProductKind.LATENT_FEATURE,
            }:
                encoder_bindings.append((output, device))
                continue
            if transfer.requires_device_product_binding(output):
                binding = (output, device)
                if output.uses_persistent_buffer():
                    persistent_bindings.append(binding)
                elif output.shape_bound.max_elements == 1:
                    scalar_groups.setdefault(
                        (device, output.kind, output.dtype, output.shape_bound),
                        [],
                    ).append(binding)
                else:
                    general_bindings.append(binding)
    groups = tuple(tuple(group) for group in scalar_groups.values())
    if general_bindings:
        groups = (*groups, tuple(general_bindings))
    if persistent_bindings:
        groups = (*groups, tuple(persistent_bindings))
    request_slots = {
        request.request_key: int(request.request_pool_idx)
        for request in scope.request_candidates
    }
    bound_groups = runtime.device_products.bind_output_groups(
        groups,
        request_slots=request_slots,
        buffer_placements={
            placement.buffer: placement for placement in scope.lane.buffer_placements
        },
    )
    scope.device_writes.extend(write for binding in bound_groups for write in binding.writes)
    scope.encoder_writes.extend(
        runtime.encoder_cache.bind_outputs(
            tuple(encoder_bindings),
            buffer_placements={
                placement.buffer: placement
                for placement in scope.lane.buffer_placements
            },
        )
    )
    operation_identities = {_operation_identity(operation) for operation in operations}
    token_operation_identities = {
        _operation_identity(operation)
        for operation in operations
        if operation.kind.token_mode is not None
    }
    for write in scope.device_writes:
        operation_identity = _reference_operation_identity(write.reference)
        if operation_identity not in operation_identities:
            raise RuntimeError("device output binding has no operation in the execution batch")
        if write.reference.kind is ProductKind.TOKEN:
            scope.token_writes[operation_identity] = write
            scope.operation_writes.setdefault(operation_identity, write)
        elif write.reference.kind is ProductKind.SELECTED_POINT:
            scope.selected_point_writes[operation_identity] = write
        elif (
            write.reference.kind is ProductKind.COMPLETION
            and operation_identity in token_operation_identities
            and int(write.reference.output_index) == 3
        ):
            scope.transition_writes[operation_identity] = write
        else:
            scope.operation_writes.setdefault(operation_identity, write)
        if (
            operation_identity in scope.predicated_operations
            and write.reference.kind is ProductKind.COMPLETION
        ):
            scope.propagated_predicate_writes.setdefault(operation_identity, ())
            scope.propagated_predicate_writes[operation_identity] = (
                *scope.propagated_predicate_writes[operation_identity],
                write,
            )


def _operation_device(runtime, operation: Operation) -> torch.device:
    """Resolve the execution device for an operation's model phase."""

    return (
        runtime._generation_device
        if operation.kind
        in {
            RunKind.DIFFUSION_PREPARE,
            RunKind.DIFFUSION_STEP,
            RunKind.DIFFUSION_FINALIZE,
        }
        else runtime._device
    )


def _validate_completion_products(
    runtime,
    operation: Operation,
    products: tuple[ProductPayload, ...],
) -> None:
    """Validate completion payloads against every product declared by the operation."""

    declared = {output: output for output in operation.outputs}
    for product in products:
        reference = declared.get(product.product)
        if reference is None:
            raise invalid_descriptor("completion carries a product not declared by its operation")
        payload_bound = (
            product.payload.max_encoded_bytes()
            if isinstance(
                product.payload,
                (
                    ImagePayload,
                    LogprobPayload,
                    TransferPayload,
                ),
            )
            else len(product.payload)
        )
        transferred = isinstance(product.payload, TransferPayload)
        if transferred and reference.storage_class in {
            StorageClass.HOST_STAGING,
            StorageClass.PINNED_OUTPUT,
        }:
            raise invalid_descriptor("host-visible output cannot carry a transfer entry")
        if not transferred and payload_bound > int(reference.max_bytes):
            raise invalid_descriptor("completion product exceeds its registered product byte bound")
        if reference.storage_class in (
            StorageClass.HOST_STAGING,
            StorageClass.PINNED_OUTPUT,
        ) and payload_bound > int(operation.bounds.max_completion_bytes):
            raise invalid_descriptor("completion product exceeds its registered byte bound")


def _consume_predicates(
    runtime,
    operations: tuple[Operation, ...],
    scope: LaneState,
) -> None:
    """Resolve operation predicates from local device products and register their readers."""

    grouped: dict[
        torch.device,
        list[
            tuple[
                Operation,
                tuple[ProductRef, int, str | None, torch.device | str | None],
            ]
        ],
    ] = {}
    for operation in operations:
        predicate = operation.predicate
        if predicate is None:
            continue
        device = _operation_device(runtime, operation)
        grouped.setdefault(device, []).append(
            (
                operation,
                (
                    predicate,
                    int(operation.op_id),
                    device,
                ),
            )
        )
    for device, entries in grouped.items():
        resident_entries = tuple(
            entry
            for entry in entries
            if cast(ProductRef, entry[0].predicate) not in scope.transferred_device_products
        )
        reads = runtime.device_products.consume_batch(
            tuple(request for _operation, request in resident_entries),
            device=device,
        )
        scope.device_reads.extend(reads)
        for (operation, _request), read in zip(resident_entries, reads, strict=True):
            predicate = cast(ProductRef, operation.predicate)
            tagged = predicate.kind is ProductKind.TOKEN and predicate.dtype is DType.U32
            scope.predicate_values[_operation_identity(operation)] = (read.tensor, tagged)
        for operation, request in entries:
            predicate = cast(ProductRef, operation.predicate)
            if predicate not in scope.transferred_device_products:
                continue
            _reference, consumer_op_id, target = request
            read = _consume_device_product(
                runtime,
                predicate,
                scope,
                consumer_op_id=consumer_op_id,
                device=target,
            )
            scope.device_reads.append(read)
            tagged = predicate.kind is ProductKind.TOKEN and predicate.dtype is DType.U32
            scope.predicate_values[_operation_identity(operation)] = (read.tensor, tagged)


def _publish_predicates(
    runtime,
    scope: LaneState,
) -> None:
    """Publish predicate outputs after their producing operations have resolved."""

    producers = {_operation_identity(operation) for operation in scope.lane.operations}
    transitions = {id(write) for write in scope.transition_writes.values()}
    propagated = {
        id(write) for writes in scope.propagated_predicate_writes.values() for write in writes
    }
    writes = tuple(
        write
        for write in scope.device_writes
        if write.reference.kind is ProductKind.COMPLETION
        and _reference_operation_identity(write.reference) in producers
        and id(write) not in transitions
        and id(write) not in propagated
    )
    if not writes:
        return
    batch = runtime.device_products.producer_scalar_batch(writes)
    if batch is not None:
        batch.tensor.fill_(1)
        runtime.device_products.publish_scalar_batch(batch)
        return
    views = runtime.device_products.producer_write_views(writes)
    first = views[0]
    runtime.device_products.publish_writes(
        writes,
        torch.ones(
            (len(writes),),
            dtype=first.dtype,
            device=first.device,
        ),
    )


def _publish_predicated_outputs(
    runtime,
    operations: tuple[Operation, ...],
    scope: LaneState,
) -> None:
    """Publish inactive sentinel values for products of predicated operations."""

    declared = {_operation_identity(operation) for operation in operations}
    for identity, writes in scope.propagated_predicate_writes.items():
        if identity not in declared:
            raise RuntimeError("predicated output has no operation in its lane")
        for write in writes:
            runtime.device_products.publish_scalar_write(write, False)


def _finish_device_reads(
    runtime,
    scope: LaneState,
) -> None:
    """Record or cancel every device-product read acquired by the lane."""

    reads = tuple(read for read in scope.device_reads if not read._recorded)
    if reads:
        after_writes: list[DeviceProductWrite] = []
        for read in reads:
            write = scope.operation_writes.get(
                (read.reference.request_key, int(read.consumer_op_id))
            )
            if write is None:
                after_writes.clear()
                break
            after_writes.append(write)
        runtime.device_products.record_readers(
            reads,
            after_writes=tuple(after_writes),
        )
    scope.device_reads.clear()
    encoder_reads = tuple(read for read in scope.encoder_reads if not read._recorded)
    if encoder_reads:
        runtime.encoder_cache.record_readers(encoder_reads)
    scope.encoder_reads.clear()


def _apply_release_controls(runtime, batch: Run, *, before_execution: bool) -> None:
    """Apply lifecycle releases in the phase required by their ownership contract."""

    consumed = {
        (reference.request_key, int(reference.producer_op_id))
        for operation in batch.operations
        for reference in (
            *operation.inputs,
            *(() if operation.predicate is None else (operation.predicate,)),
        )
    }
    releases = tuple(
        (operation.request_key, operation.parent.op_id)
        for operation in batch.operations
        if operation.parent.op_id > 0
        and (
            ((operation.request_key, int(operation.parent.op_id)) not in consumed)
            == before_execution
        )
    )
    runtime.device_products.release_operations(releases)
    runtime.encoder_cache.release_operations(releases)
    if runtime.cache_publications is not None:
        runtime.cache_publications.release_operations(releases)
    if before_execution:
        generations = tuple(
            int(command.buffer.generation)
            for command in batch.commands
            if isinstance(command, Free)
        )
        runtime.device_products.release_generations(generations)
        runtime.encoder_cache.release_generations(generations)
    if not before_execution:
        consumed_predicates = tuple(
            int(predicate.generation)
            for operation in batch.operations
            if (predicate := operation.predicate) is not None
            and predicate.producer_op_id != operation.parent.op_id
        )
        runtime.device_products.release_generations(consumed_predicates)
    if runtime.transport is not None:
        if before_execution:
            for command in batch.commands:
                if isinstance(command, Free):
                    identity = (
                        command.buffer.owner,
                        int(command.buffer.producer_op_id),
                    )
                    _release_locators(
                        runtime,
                        runtime._transport_publications.pop(identity, ()),
                    )


def drop_request(runtime, request_id: int) -> None:
    """Release stage publications owned by one dropped request."""

    request = runtime.requests.peek(int(request_id))
    if request is not None:
        if runtime.runtime_states is not None:
            runtime.runtime_states.release((int(request.request_pool_idx),))
        if runtime.req_to_token_pool is not None:
            runtime.req_to_token_pool.release((int(request.request_pool_idx),))
    if runtime.cache_publications is not None:
        runtime.cache_publications.drop(request_id)
    if request is not None and runtime.req_to_token_pool is not None:
        runtime.req_to_token_pool.release(
            tuple(runtime._flow_prefix_slots.pop(request.request_key, ()))
        )
    if runtime.transport is None:
        return
    selected = tuple(
        identity
        for identity in runtime._transport_publications
        if int(identity[0].request_id) == int(request_id)
    )
    for identity in selected:
        _release_locators(runtime, runtime._transport_publications.pop(identity))


def _bind_latent_rows(
    runtime,
    lane: RunLane,
    scope: LaneState,
) -> None:
    """Validate trajectory placements and bind rank-local latent staging views."""

    if not lane.latent_placements:
        return
    pool = runtime.latent_pool
    if pool is None:
        # Dedicated-state models use their request-pool slot as a capacity token;
        # scheduler placements still have to match resident solver progress.
        operations = {
            _operation_identity(operation): (
                operation,
                _request_row(runtime, scope, operation.request_key.request_id),
            )
            for operation in lane.operations
        }
        for placement in lane.latent_placements:
            identity = placement.request_key, int(placement.op_id)
            selected = operations.get(identity)
            if selected is None:
                raise invalid_descriptor(
                    "latent placement names an operation outside its lane"
                )
            operation, request = selected
            slot = int(request.request_pool_idx)
            if placement.page_table != (slot,):
                raise invalid_descriptor(
                    "pool-free latent placement must name its request-pool capacity token"
                )
            if operation.kind is RunKind.DIFFUSION_PREPARE:
                valid = int(placement.start_step) == 0 and int(placement.step_count) == 0
            elif operation.kind is RunKind.DIFFUSION_STEP:
                valid = (
                    int(placement.start_step) == int(request.flow_step)
                    and int(placement.step_count) == 1
                )
            else:
                valid = (
                    int(placement.start_step) == int(request.flow_step)
                    and int(placement.step_count) == 0
                )
            if not valid:
                raise invalid_descriptor(
                    "pool-free latent placement disagrees with resident generation state"
                )
        return
    # Pooled models bind each operation to validated image geometry and page ownership.
    operations = {
        _operation_identity(operation): (
            operation,
            int(_request_row(runtime, scope, operation.request_key.request_id).request_pool_idx),
        )
        for operation in lane.operations
    }
    rows: list[tuple[OperationIdentity, LatentPlacement, int]] = []
    for placement in lane.latent_placements:
        identity = (placement.request_key, int(placement.op_id))
        selected = operations.get(identity)
        if selected is None:
            raise invalid_descriptor("latent placement names an operation outside its lane")
        operation, slot = selected
        request = _request_row(runtime, scope, operation.request_key.request_id)
        image = request.image
        if image is None:
            raise invalid_descriptor("latent placement has no admitted image geometry")
        flow = _generation(
            runtime,
        )
        expected_units = int(flow.image_tokens(int(placement.height), int(placement.width)))
        if (
            int(placement.height) != int(image.height)
            or int(placement.width) != int(image.width)
            or int(placement.latent_units) != expected_units
        ):
            raise invalid_descriptor("latent placement disagrees with admitted model geometry")
        transferred = next(
            (
                scope.prepared_transfers[reference]
                for reference in operation.inputs
                if reference in scope.prepared_transfers
                and scope.prepared_transfers[reference].kind == "latent"
            ),
            None,
        )
        committed_step = (
            int(request.flow_step) if transferred is None else int(cast(int, transferred.step))
        )
        if operation.kind is RunKind.DIFFUSION_PREPARE:
            if int(placement.start_step) != 0 or int(placement.step_count) != 0:
                raise invalid_descriptor("media preparation placement carries denoise steps")
        elif operation.kind is RunKind.DIFFUSION_STEP:
            if (
                int(placement.start_step) != committed_step
                or int(placement.step_count) < 1
                or int(placement.start_step) + int(placement.step_count) > int(image.steps)
                or (
                    int(operation.bounds.max_tokens) > 0
                    and int(placement.step_count) > int(operation.bounds.max_tokens)
                )
            ):
                raise invalid_descriptor("media denoise placement exceeds its committed schedule")
        elif int(placement.start_step) != committed_step or int(placement.step_count) != 0:
            raise invalid_descriptor("latent reader placement disagrees with committed step state")
        rows.append((identity, placement, slot))
    # Stage every page table together so overlapping physical ownership is
    # rejected before any operation receives a writable tensor view.
    staged = pool.stage(
        tuple(placement.page_table for _identity, placement, _slot in rows),
        tuple(int(placement.latent_units) for _identity, placement, _slot in rows),
    )
    scope.latent_rows = {
        identity: LatentExecution(
            placement=placement,
            request_pool_idx=slot,
            staging=value,
        )
        for (identity, placement, slot), value in zip(rows, staged, strict=True)
    }


def _latent_row(runtime, operation: Operation, scope: LaneState) -> LatentExecution:
    """Resolve a latent placement into the request pool's staged row."""

    row = scope.latent_rows.get(_operation_identity(operation))
    if row is None:
        raise invalid_descriptor("trajectory operation has no staged latent placement")
    return row


def _bind_cache_tables(
    runtime,
    lane: RunLane,
    scope: LaneState,
) -> None:
    """Install scheduler tables and retain row-aligned forward coordinates."""

    started = time.perf_counter_ns()
    tables = []
    for table in lane.block_tables:
        pages = runtime.cache_pool.validate_pages(table.page_ids, group=table.group_id)
        if int(table.allocated_tokens) > len(pages) * runtime.cache_pool.block_size:
            raise invalid_descriptor("block-table allocation exceeds physical capacity")
        tables.append(
            (
                int(table.request_pool_idx),
                int(table.group_id),
                pages,
                int(table.allocated_tokens),
            )
        )
    runtime.req_to_token_pool.install(tuple(tables))
    for allocation in lane.new_cache_pages:
        pages = runtime.cache_pool.validate_pages(
            allocation.page_ids,
            group=allocation.group_id,
        )
        installed = runtime.req_to_token_pool.pages(
            allocation.request_pool_idx, allocation.group_id
        )
        if not set(pages).issubset(installed):
            raise invalid_descriptor("new cache pages are outside the installed block table")
        runtime.cache_pool.zero_pages(allocation.group_id, pages)

    rows_by_operation: dict[int, list] = defaultdict(list)
    for row in lane.forward_rows:
        rows_by_operation[int(row.operation_index)].append(row)

    for operation_index, operation in enumerate(lane.operations):
        request = _request_row(runtime, scope, operation.request_key.request_id)
        main_slot = int(request.request_pool_idx)
        parent_runtime = _parent_runtime(runtime, operation, request)
        operation_rows = rows_by_operation.get(operation_index, [])
        scope.forward_rows[_operation_identity(operation)] = tuple(operation_rows)
        main_descriptor = next(
            (row for row in operation_rows if int(row.request_pool_index) == main_slot),
            None,
        )
        if main_descriptor is not None and int(main_descriptor.seq_len) != int(
            parent_runtime.kv_visible_len
        ):
            raise invalid_descriptor("forward row sequence length disagrees with its parent")
        for descriptor in operation_rows:
            slot = int(descriptor.request_pool_index)
            runtime.req_to_token_pool.pages(slot, 0)
            if slot != main_slot and int(
                descriptor.seq_len
            ) > runtime.req_to_token_pool.allocated_length(slot):
                raise invalid_descriptor("forward row exceeds alternative-prefix capacity")
            if slot != main_slot:
                runtime._flow_prefix_slots.setdefault(operation.request_key, set()).add(slot)
    _record_component(scope, "bc_tables", started)


def _request_row(runtime, scope: LaneState, request_id: int) -> RequestDraft:
    """Return the staged request draft for a request identifier in this lane."""

    try:
        return scope.request_rows[int(request_id)]
    except KeyError:
        raise invalid_descriptor(f"lane has no request row for request {request_id}") from None


def _consume_device_product(
    runtime,
    reference: ProductRef,
    scope: LaneState,
    *,
    consumer_op_id: int,
    device: torch.device | str | None = None,
) -> DeviceProductRead:
    """Acquire a generation-checked device product and attach its read lease to the lane."""

    candidate = scope.transferred_device_products.get(reference)
    if candidate is not None:
        return runtime.device_products.consume_candidate(
            candidate,
            consumer_op_id=consumer_op_id,
            device=device,
        )
    return runtime.device_products.consume(
        reference,
        consumer_op_id=consumer_op_id,
        device=device,
    )


def _consume_encoder_feature(
    runtime,
    reference: ProductRef,
    scope: LaneState,
    *,
    consumer_op_id: int,
    device: torch.device | str | None = None,
) -> EncoderRead:
    """Acquire an immutable encoder feature and attach its read lease to the lane."""

    candidate = scope.transferred_encoder_features.get(reference)
    if candidate is not None:
        return runtime.encoder_cache.consume_candidate(
            candidate,
            consumer_op_id=consumer_op_id,
            device=device,
        )
    return runtime.encoder_cache.consume(
        reference,
        consumer_op_id=consumer_op_id,
        device=device,
    )


def _parent_runtime(
    runtime,
    operation: Operation,
    request: RequestDraft,
) -> RequestRuntime:
    """Resolve the request runtime checkpoint consumed by an operation."""

    parent = operation.parent
    point = parent.point
    runtime = (
        request.execution_runtime_for_operation(parent.op_id, 1)
        if isinstance(point, DeviceSelected)
        else None
    )
    if runtime is None:
        selected = request.resolve_version(parent)
        runtime = None if selected is None else request.runtime_for(selected)
    if runtime is None:
        raise invalid_descriptor("operation parent has no resolved runtime state")
    return runtime


def parent_runtime(runtime, operation: Operation, request: RequestDraft) -> RequestRuntime:
    """Resolve an operation’s semantic parent checkpoint against its request draft."""

    return _parent_runtime(runtime, operation, request)


def _cache_coordinates(
    runtime,
    operation: Operation,
    scope: LaneState,
    *,
    group_id: int = 0,
) -> tuple[int, int, int, int]:
    """Resolve request slot and verified, allocated, and logical cache lengths."""

    request = _request_row(runtime, scope, operation.request_key.request_id)
    slot = int(request.request_pool_idx)
    rows = scope.forward_rows.get(_operation_identity(operation), ())
    descriptor = next(
        (row for row in rows if int(row.request_pool_index) == slot),
        None,
    )
    parent = _parent_runtime(runtime, operation, request)
    visible = int(parent.kv_visible_len) if descriptor is None else int(descriptor.seq_len)
    runtime.req_to_token_pool.pages(slot, group_id)
    capacity = runtime.req_to_token_pool.allocated_length(slot)
    if visible > capacity:
        raise invalid_descriptor("operation visibility exceeds scheduler block table")
    return slot, int(group_id), visible, capacity


def _logical_lengths(
    runtime,
    operation: Operation,
    request: RequestDraft,
    cache: tuple[int, int, int, int] | None,
    *,
    latent_len: int | None = None,
    computed_len: int | None = None,
) -> LogicalLengths:
    """Derive logical input, cache, computed, and latent lengths for one forward row."""

    parent = _parent_runtime(runtime, operation, request)
    if cache is None:
        visible = parent.kv_visible_len
        computed = parent.kv_computed_len
    else:
        _slot, _group, visible, _capacity = cache
        computed = visible if computed_len is None else int(computed_len)
    return LogicalLengths(
        token_len=request.logical_position,
        kv_visible_len=visible,
        kv_computed_len=computed,
        latent_len=request.flow_step if latent_len is None else int(latent_len),
    )


def _stage_input_products(
    runtime,
    input_products: Sequence[ProductPayload],
    scope: LaneState,
) -> None:
    """Decode ephemeral host inputs and publish transferred physical values."""

    for entry in input_products:
        product = entry.product
        if isinstance(entry.payload, TransferHandle):
            # Transfer metadata determines which runtime owns the imported value;
            # each branch validates identity and geometry before publication.
            transfer = scope.prepared_transfers.get(product)
            if transfer is None or not transfer.ready():
                raise invalid_descriptor("cross-stage input has no query-ready prepared transfer")
            if transfer.kind == "kv":
                snapshot = transfer.snapshot
                if snapshot is None:
                    raise RuntimeError("prepared KV transfer has no validated snapshot")
                existing = runtime.cache_publications.resident(product)
                if existing is not None and existing != snapshot:
                    raise invalid_descriptor(
                        "staged KV publication conflicts with its product identity"
                    )
                staged = scope.cache_publication_inputs.get(product)
                if staged is not None and staged != snapshot:
                    raise invalid_descriptor(
                        "batch repeats a KV product with conflicting publication data"
                    )
                scope.cache_publication_inputs[product] = snapshot
                continue
            if transfer.kind == "latent":
                tensors = transfer.tensors()
                if len(tensors) != 1:
                    raise invalid_descriptor("latent transfer produced an invalid tensor set")
                consumers = tuple(
                    operation
                    for operation in scope.lane.operations
                    if product in operation.inputs
                )
                if len(consumers) != 1:
                    raise invalid_descriptor("latent transfer must have one lane consumer")
                row = _latent_row(runtime, consumers[0], scope)
                latent_units = transfer.latent_units
                height = transfer.height
                width = transfer.width
                step = transfer.step
                generation = transfer.generation
                if (
                    latent_units is None
                    or height is None
                    or width is None
                    or step is None
                    or generation is None
                ):
                    raise RuntimeError("prepared latent transfer lost validated metadata")
                if (
                    int(latent_units) != int(row.placement.latent_units)
                    or int(height) != int(row.placement.height)
                    or int(width) != int(row.placement.width)
                    or int(step) != int(row.placement.start_step)
                    or int(generation) != int(product.generation)
                ):
                    raise invalid_descriptor(
                        "latent transfer disagrees with its scheduler placement"
                    )
                request = _request_row(runtime, scope, product.request_key.request_id)
                if request.latent_product is not None or int(request.flow_step) != 0:
                    raise invalid_descriptor(
                        "latent transfer destination already owns a trajectory"
                    )
                _latent_pool(
                    runtime,
                ).restore(
                    LatentSnapshot(
                        generation=int(generation),
                        step=int(step),
                        latent_units=int(latent_units),
                        height=int(height),
                        width=int(width),
                        value=tensors[0],
                    ),
                    request_pool_idx=row.request_pool_idx,
                    page_table=row.placement.page_table,
                )
                scope.latent_import_slots.append(row.request_pool_idx)
                request.latent_product = product
                request.flow_step = int(step)
                continue
            tensors = transfer.tensors()
            if len(tensors) != 1:
                raise invalid_descriptor("product transfer produced an invalid tensor set")
            consumers = tuple(
                operation
                for operation in scope.lane.operations
                if product in operation.inputs or operation.predicate == product
            )
            if not consumers:
                raise invalid_descriptor("transferred product has no lane consumer")
            devices = {_operation_device(runtime, operation) for operation in consumers}
            if len(devices) != 1:
                raise invalid_descriptor("transferred product spans multiple consumer devices")
            device = next(iter(devices))
            if transfer.kind == "device_product":
                request_slots = {
                    request.request_key: int(request.request_pool_idx)
                    for request in scope.request_candidates
                }
                binding = runtime.device_products.bind_outputs(
                    ((product, device),),
                    request_slots=request_slots,
                    buffer_placements={
                        placement.buffer: placement
                        for placement in scope.lane.buffer_placements
                    },
                )[0]
                scope.device_writes.append(binding)
                scope.transferred_device_products[product] = binding
                runtime.device_products.publish_write(
                    binding,
                    tensors[0],
                    metadata=transfer.device_metadata,
                )
                continue
            if transfer.kind != "encoder":
                raise RuntimeError("prepared transfer lost its concrete resource kind")
            payload_kind = transfer.payload_kind
            height = transfer.height
            width = transfer.width
            if payload_kind is None or height is None or width is None:
                raise RuntimeError("prepared encoder transfer has no validated geometry")
            if (
                payload_kind
                not in {
                    ProductKind.VISION_FEATURE,
                    ProductKind.LATENT_FEATURE,
                }
                or product.kind is not payload_kind
            ):
                raise invalid_descriptor("encoder transfer payload geometry is invalid")
            encoder_binding = runtime.encoder_cache.bind_outputs(
                ((product, device),),
                buffer_placements={
                    placement.buffer: placement
                    for placement in scope.lane.buffer_placements
                },
            )[0]
            scope.encoder_writes.append(encoder_binding)
            scope.transferred_encoder_features[product] = encoder_binding
            runtime.encoder_cache.publish(
                encoder_binding,
                tensors[0],
                EncoderMetadata(height=height, width=width),
            )
            continue
        # Inline payloads remain host-owned until their consuming operation stages them.
        if product.kind is ProductKind.SAMPLING_STATE:
            scope.sampling_states[_reference_operation_identity(product)] = (
                decode_sampling_state_bytes(entry.payload)
            )
            continue
        if product.kind is ProductKind.TOKEN:
            scope.input_tokens[product] = decode_token_product_bytes(entry.payload)
            continue
        if product.kind is not ProductKind.ARTIFACT:
            raise invalid_descriptor("host-staging payload has no concrete product owner")
        if product.storage_class is not StorageClass.HOST_STAGING or not entry.payload:
            raise invalid_descriptor("source image payload has invalid storage metadata")
        scope.input_images[product] = entry.payload.decode("utf-8")


def _predicated_outcome(
    runtime,
    operation: Operation,
    scope: LaneState,
) -> Outcome:
    """Construct an inactive outcome while preserving declared product generations."""

    request = _request_row(runtime, scope, operation.request_key.request_id)
    lengths = _logical_lengths(runtime, operation, request, None)
    selected = request.resolve_version(operation.parent)
    if selected is None or not isinstance(selected.point, FixedCheckpoint):
        raise invalid_descriptor("predicated operation parent has no selected fixed checkpoint")
    selected_point = int(selected.point.point_index)
    return Outcome(
        status=OpStatus.PREDICATED,
        selected_point=selected_point,
        logical_lengths=lengths,
        token_span=TokenSpan(base=int(lengths.token_len), len=0),
        finish_flags=FinishFlags(),
        product_generations=(),
    )


def _run_laneed_wave(
    runtime,
    tasks: tuple[tuple[ForwardRow, LaneState], ...],
    *,
    qualify_mixed: bool,
) -> tuple[torch.Tensor, ...]:
    """Group compatible forward rows, execute each group, and restore task order."""

    # Launch identity, physical lane, and shape key jointly define rows that may
    # share one model invocation without changing scheduler ordering.
    grouped: dict[
        tuple[object, ...],
        list[tuple[int, ForwardRow, LaneState]],
    ] = defaultdict(list)
    for index, (task, scope) in enumerate(tasks):
        grouped[
            (
                scope.lane.launch_id,
                _lane_identity(
                    runtime.runner, _phase_device(runtime, task.phase), scope.lane.domain
                ),
                *_group_key(runtime, task),
            )
        ].append((index, task, scope))

    result: list[torch.Tensor | None] = [None] * len(tasks)
    output_events: list[tuple[torch.device, torch.cuda.Event]] = []
    for group in grouped.values():
        kinds = frozenset(task.kind for _index, task, _scope in group)
        if (
            len(kinds) > 1
            and not _model(
                runtime,
            ).tensorized_mixed
        ):
            raise invalid_descriptor("tensorized mixed submission is outside the model limits")
        indexes = tuple(index for index, _task, _scope in group)
        group_tasks = tuple(task for _index, task, _scope in group)
        group_scopes = tuple(scope for _index, _task, scope in group)
        target = _phase_device(runtime, group_tasks[0].phase)
        for scope in _unique_scopes(group_scopes):
            scope.completion.register_device(target)
        if qualify_mixed and len(kinds) > 1:
            # Startup qualification compares tensorized mixed output with
            # independently executed homogeneous groups for the same rows.
            output, observation, mixed_us = _run_startup_forward(
                runtime,
                group_tasks,
                group_scopes[0],
                target,
            )
            mixed_output = tuple(value.clone() for value in output)
            homogeneous: dict[
                str,
                list[tuple[int, ForwardRow, LaneState]],
            ] = defaultdict(list)
            for local_index, (_index, task, scope) in enumerate(group):
                homogeneous[task.kind].append((local_index, task, scope))
            references: list[torch.Tensor | None] = [None] * len(group)
            homogeneous_us: list[int] = []
            for members in homogeneous.values():
                reference, _reference_observation, reference_us = _run_startup_forward(
                    runtime,
                    tuple(task for _index, task, _scope in members),
                    members[0][2],
                    target,
                    force_eager=observation.path is RunPath.EAGER,
                )
                homogeneous_us.append(reference_us)
                for (local_index, _task, _scope), value in zip(
                    members,
                    reference,
                    strict=True,
                ):
                    references[local_index] = value.clone()
            if any(value is None for value in references):
                raise RuntimeError("mixed qualification lost a homogeneous output row")
            _assert_mixed_equivalence(
                runtime,
                mixed_output,
                tuple(cast(torch.Tensor, value) for value in references),
                group_tasks,
            )
            service_paths = {RunPath.EAGER, RunPath.GRAPH_REPLAY}
            if observation.path in service_paths:
                serial_us = sum(homogeneous_us)
                bucket = _mixed_bucket(
                    runtime, tuple(scope.lane for scope in _unique_scopes(group_scopes))
                )
                speedup = serial_us / max(1, mixed_us)
                if target.type != "cuda" or speedup >= _MIN_MIXED_SERVICE_SPEEDUP:
                    runtime._qualified_mixed_buckets.add(bucket)
                logger.info(
                    "evaluated mixed execution bucket=%r mixed_us=%d homogeneous_us=%r "
                    "serial_over_mixed=%.3f service_eligible=%s",
                    bucket,
                    mixed_us,
                    tuple(homogeneous_us),
                    speedup,
                    bucket in runtime._qualified_mixed_buckets,
                )
            output = mixed_output
            output_event = None
        else:
            forward_result = _run_forward_group(runtime, group_tasks, group_scopes[0])
            output = forward_result.values
            observation = forward_result.observation
            output_event = forward_result.output_event
        group_scopes[0].observations.append(observation)
        if output_event is not None:
            output_events.append((target, output_event))
        # Scatter each grouped result back to the caller's task ordering.
        for index, value in zip(indexes, output, strict=True):
            result[index] = value
    for device, event in output_events:
        torch.cuda.current_stream(device).wait_event(event)
    return tuple(cast(torch.Tensor, value) for value in result)


def _run_startup_forward(
    runtime,
    tasks: tuple[ForwardRow, ...],
    scope: LaneState,
    target: torch.device,
    *,
    force_eager: bool = False,
) -> tuple[tuple[torch.Tensor, ...], RunObservation, int]:
    """Execute startup forward rows eagerly or through graph qualification without publication."""

    with profile_range("uniserve.startup.mixed_oracle_forward"):
        if target.type != "cuda":
            started = time.perf_counter_ns()
            result = _run_forward_group(runtime, tasks, scope, force_eager=force_eager)
            elapsed_us = max(1, (time.perf_counter_ns() - started) // 1000)
        else:
            stream = torch.cuda.current_stream(target)
            start = torch.cuda.Event(blocking=False, enable_timing=True)
            end = torch.cuda.Event(blocking=False, enable_timing=True)
            start.record(stream)
            result = _run_forward_group(runtime, tasks, scope, force_eager=force_eager)
            if result.output_event is not None:
                stream.wait_event(result.output_event)
            end.record(stream)
            end.synchronize()
            elapsed_us = max(1, round(float(start.elapsed_time(end)) * 1000.0))
        return result.values, result.observation, elapsed_us


def _assert_mixed_equivalence(
    runtime,
    mixed: tuple[torch.Tensor, ...],
    homogeneous: tuple[torch.Tensor, ...],
    tasks: tuple[ForwardRow, ...],
) -> None:
    """Compare mixed-lane outputs with homogeneous execution across corresponding row slices."""

    from . import flow as flow_ops

    if len(mixed) != len(homogeneous) or len(mixed) != len(tasks):
        raise RuntimeError("mixed and homogeneous forwards returned different row counts")
    tolerances = {
        torch.bfloat16: (1.6e-2, 1.0e-5),
        torch.float16: (1.0e-3, 1.0e-5),
        torch.float32: (1.3e-6, 1.0e-5),
        torch.float64: (1.0e-7, 1.0e-7),
    }
    flow_rows: dict[OperationIdentity, list[int]] = defaultdict(list)
    for row, (actual, expected, task) in enumerate(zip(mixed, homogeneous, tasks, strict=True)):
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise RuntimeError(f"mixed qualification row {row} changed output structure")
        if task.kind == "token":
            actual_tokens = actual.argmax(dim=-1)
            expected_tokens = expected.argmax(dim=-1)
            if not torch.equal(actual_tokens, expected_tokens):
                actual_row = actual.reshape(-1, actual.shape[-1])[-1].float()
                expected_row = expected.reshape(-1, expected.shape[-1])[-1].float()
                actual_token = int(actual_tokens.reshape(-1)[-1].item())
                expected_token = int(expected_tokens.reshape(-1)[-1].item())
                compared = tuple(sorted({actual_token, expected_token}))
                score_pairs = tuple(
                    (
                        token,
                        float(actual_row[token].item()),
                        float(expected_row[token].item()),
                    )
                    for token in compared
                )
                raise RuntimeError(
                    f"mixed qualification row {row} changed the committed greedy token: "
                    f"actual={actual_token} expected={expected_token} "
                    f"candidate_scores={score_pairs!r} "
                    f"max_abs_logit_error="
                    f"{float((actual_row - expected_row).abs().max().item()):.6g}"
                )
            continue
        if task.kind != "flow":
            raise RuntimeError("mixed qualification contains an unsupported row kind")
        flow_rows[_operation_identity(task.operation)].append(row)

    flow = _generation(
        runtime,
    )
    for identity, rows in flow_rows.items():
        first = tasks[rows[0]]
        image = first.request.image
        timestep = first.timestep
        latent = first.latent
        if image is None or timestep is None or latent is None:
            raise RuntimeError("mixed flow qualification lost its committed-state inputs")
        host_t, host_t_next = flow.schedule_pair(
            int(image.steps),
            float(image.timestep_shift),
            int(first.request.flow_step),
        )
        guide = build_flow_cfg_plan(
            cfg_text_scale=float(image.cfg_text_scale),
            cfg_img_scale=float(image.cfg_img_scale),
            recipe=flow.cfg_recipe,
            renorm=image.cfg_renorm_type,
            renorm_min=float(image.cfg_renorm_min),
            use_cfg=float(image.cfg_interval[0]) <= host_t <= float(image.cfg_interval[1]),
        )
        if len(guide.branches) != len(rows) or any(
            _operation_identity(tasks[row].operation) != identity for row in rows
        ):
            raise RuntimeError("mixed flow qualification changed its CFG branch geometry")

        def committed(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
            """Combine CFG branches and integrate the candidate latent for comparison."""

            predictions = {
                branch: flow_ops.prediction(values[row])
                for branch, row in zip(guide.branches, rows, strict=True)
            }
            velocity = guide.combine(predictions)
            if flow.prediction in {"x", "x_prediction", "x_pred"}:
                velocity = x_pred_to_velocity(velocity, latent, timestep)
            elif flow.prediction != "velocity":
                raise invalid_descriptor(f"unsupported flow prediction {flow.prediction!r}")
            next_timestep = timestep.new_tensor([host_t_next])
            return euler_step(latent, velocity, timestep, next_timestep)

        actual = committed(mixed)
        expected = committed(homogeneous)
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise RuntimeError("mixed flow qualification changed committed latent structure")
        tolerance = tolerances.get(actual.dtype)
        if tolerance is None:
            if not torch.equal(actual, expected):
                raise RuntimeError("mixed flow qualification changed an exact committed latent")
            continue
        rtol, atol = tolerance
        torch.testing.assert_close(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
            msg=lambda message: (
                f"mixed flow qualification {identity!r} changed the committed latent: {message}"
            ),
        )


def _run_observed_forward_group(
    runtime,
    tasks: tuple[ForwardRow, ...],
    scope: LaneState,
) -> ForwardResult:
    """Run a forward group while recording timing and operation trace metadata."""

    result = _run_forward_group(runtime, tasks, scope)
    scope.observations.append(result.observation)
    return result


def _broadcast_tp_selection(runtime, value: torch.Tensor) -> torch.Tensor:
    """Broadcast sampled selection state from tensor-parallel rank zero."""

    if runtime.mesh.tp_size <= 1:
        return value
    transport = runtime.mesh.transport("tp")
    return transport.broadcast(value, src=0)


def _group_key(runtime, task: ForwardRow) -> tuple[object, ...]:
    """Build the route, mode, geometry, and weight identity used to batch forward rows."""

    phase = (
        "textual"
        if task.phase in {ModelPhase.TEXT, ModelPhase.DENOISE}
        and _model(
            runtime,
        ).tensorized_mixed
        else task.phase.value
    )
    return (
        phase,
        str(_phase_device(runtime, task.phase)),
        task.weights.version,
        (
            ()
            if _model(
                runtime,
            ).tensorized_mixed
            and not runtime.runner.uses_lanes
            else _task_shape(runtime, task)
        ),
    )


def _lane_identity(runner: ModelRunner, device: torch.device, domain: Domain) -> int:
    """Return the physical runner, device, and domain identity used for lane grouping."""

    for lane in runner.execution_lanes:
        if lane.device == device and domain in lane.domains:
            return id(lane)
    raise invalid_descriptor(f"execution has no {domain.value!r} lane for {device}")


def _run_forward_group(
    runtime,
    tasks: tuple[ForwardRow, ...],
    scope: LaneState,
    *,
    force_eager: bool = False,
) -> ForwardResult:
    """Execute one compatible forward-row group and register model output products."""

    target = _phase_device(runtime, tasks[0].phase)
    scope.completion.register_device(target)
    attention = (
        _attention_columns(runtime, tasks)
        if all(task.phase in {ModelPhase.TEXT, ModelPhase.DENOISE} for task in tasks)
        else _dense_attention_columns(len(tasks), tuple(task.query_tokens for task in tasks))
    )
    mesh = RouteMeshView(runtime.mesh, _phase_topology(runtime, tasks[0].phase))
    result = runtime.runner.run(
        tasks,
        device=target,
        attention=attention,
        mesh=mesh,
        graph_shape=_group_graph_shape(runtime, tasks),
        graph_eligible=(
            scope.graph_eligible
            and not force_eager
            and all(task.phase in {ModelPhase.TEXT, ModelPhase.DENOISE} for task in tasks)
        ),
        domain=scope.lane.domain,
    )
    if int(result.request_pool_indices.numel()) != len(tasks):
        raise RuntimeError("model runner returned without aligned request slots")
    for index, task in enumerate(tasks):
        task.request_pool_index = result.request_pool_indices[index : index + 1]
    return result


def _weights(runtime) -> WeightSet:
    """Return the runtime's installed live-weight registry."""

    return runtime.weights


def _model(runtime) -> ExecutionModel:
    """Return the execution model currently bound to the runtime."""

    return runtime.model


def _generation(runtime) -> GenerationPipeline:
    """Require and return the model's diffusion-generation pipeline."""

    value = _model(
        runtime,
    ).generation
    if not isinstance(value, GenerationPipeline):
        raise invalid_descriptor("operation requires model generation behavior")
    return value


def _latent_pool(runtime) -> LatentPool:
    """Require and return runtime-owned latent trajectory storage."""

    if runtime.latent_pool is None:
        raise unsupported_setup("operation requires a physical latent pool")
    return runtime.latent_pool


def _image_processor(runtime) -> ImageProcessor:
    """Require and return the model's image preprocessing contract."""

    value = _model(
        runtime,
    ).image_processor
    if not isinstance(value, ImageProcessor):
        raise invalid_descriptor("operation requires model image processing")
    return value


def _phase_device(runtime, phase: ModelPhase) -> torch.device:
    """Resolve the model device responsible for an execution phase."""

    deployment = runtime.deployment
    if phase in {ModelPhase.ENCODE_LATENT, ModelPhase.DECODE_LATENT}:
        return torch.device(deployment.generation_device or deployment.device)
    return torch.device(deployment.device)


def _phase_topology(runtime, phase: ModelPhase) -> tuple[str, ...]:
    """Resolve the distributed mesh axes used by an execution phase."""

    if phase in {ModelPhase.TEXT, ModelPhase.DENOISE}:
        return _model(
            runtime,
        ).text_topology
    return ("tp",)


def _task_shape(runtime, task: ForwardRow) -> tuple[int, ...]:
    """Build the graph-relevant shape signature for one forward row."""

    if task.encode_pixels is not None:
        return tuple(int(value) for value in task.encode_pixels.shape)
    if task.latent is not None:
        return task.image_height, task.image_width
    return ()


def _group_graph_shape(runtime, tasks: tuple[ForwardRow, ...]) -> tuple[object, ...]:
    """Require one shared graph-shape signature across grouped forward rows."""

    return (
        len(tasks),
        sum(task.query_tokens for task in tasks),
        tuple(task.query_tokens for task in tasks),
        tuple((task.image_height, task.image_width) for task in tasks if task.latent is not None),
    )


def _release_locators(runtime, locators: Iterable[Locator]) -> None:
    """Release transfer locators through the runtime transport owner."""

    if runtime.transport is None:
        return
    for locator in locators:
        runtime.transport.release(locator)


def _trace_envelopes(
    operations: Sequence[Operation],
) -> tuple[OperationTrace, ...]:
    """Encode operation identities and kinds for execution-trace records."""

    return tuple(
        OperationTrace(
            authority_id=int(operation.request_key.authority_id),
            request_id=int(operation.request_key.request_id),
            epoch=int(operation.request_key.epoch),
            op_id=int(operation.op_id),
            version=int(point.point_index) if isinstance(point, FixedCheckpoint) else 0,
        )
        for operation in operations
        for point in (operation.parent.point,)
    )


def _fixed_parent(operation: Operation) -> FixedCheckpoint:
    """Return the fixed parent point a depth-one operation commits over."""

    point = operation.parent.point
    if not isinstance(point, FixedCheckpoint):
        raise invalid_descriptor("operation names a device parent; depth one commits fixed")
    return point


def _output_generations(operation: Operation) -> tuple[int, ...]:
    """Return output product generations in declaration order."""

    return tuple(int(reference.generation) for reference in operation.outputs)


def _record_component(scope: LaneState, name: str, started_ns: int) -> None:
    """Accumulate elapsed microseconds for one lane execution component."""

    elapsed_us = max(0, (time.perf_counter_ns() - int(started_ns)) // 1000)
    scope.component_us[name] = scope.component_us.get(name, 0) + elapsed_us


def _forward_stats(
    observations: Sequence[RunObservation],
    component_us: Mapping[str, int] | None = None,
) -> WorkerForwardStats:
    """Aggregate forward observations into stable per-component and total timing statistics."""

    route_counts: dict[str, int] = {}
    route_rows: dict[str, int] = {}
    route_us: dict[str, int] = {}
    path_counts: dict[str, int] = {}
    captures = 0
    replays = 0
    fallbacks = 0
    graph_unpadded_tokens = 0
    graph_padded_tokens = 0
    for observation in observations:
        route_counts[observation.route] = route_counts.get(observation.route, 0) + 1
        route_rows[observation.route] = route_rows.get(observation.route, 0) + int(
            observation.row_count
        )
        route_us[observation.route] = route_us.get(observation.route, 0) + int(
            observation.duration_us
        )
        path_counts[observation.path.value] = path_counts.get(observation.path.value, 0) + 1
        captures += observation.path is RunPath.GRAPH_CAPTURE
        replays += observation.path is RunPath.GRAPH_REPLAY
        fallbacks += observation.path is RunPath.GRAPH_FALLBACK
        graph_unpadded_tokens += int(observation.graph_unpadded_tokens)
        graph_padded_tokens += int(observation.graph_padded_tokens)
    components: dict[str, int] = {}
    if observations:
        components["forward"] = sum(route_us.values())
    for name, value in (component_us or {}).items():
        components[str(name)] = components.get(str(name), 0) + max(0, int(value))
    return WorkerForwardStats(
        mode_counts=route_counts,
        mode_tokens=route_rows,
        mode_us=route_us,
        component_us=components,
        cuda_graph_captures=int(captures),
        cuda_graph_replays=int(replays),
        cuda_graph_misses=int(fallbacks),
        cuda_graph_fallbacks=int(fallbacks),
        cuda_graph_unpadded_tokens=graph_unpadded_tokens,
        cuda_graph_padded_tokens=graph_padded_tokens,
        cuda_graph_runtime_mode_counts=path_counts,
    )


__all__ = [
    "ExecutionResources",
    "PreparedExecution",
    "close_execution",
    "complete_startup",
    "create_execution_resources",
    "drop_request",
    "execute_batch",
    "execute_prepared",
    "execute_startup",
    "install_weights",
    "parent_runtime",
    "prepare_batch",
]
