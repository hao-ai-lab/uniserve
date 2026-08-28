"""Candidate preparation and resource-specific publication for execution batches."""

from __future__ import annotations

import logging
import math
import time
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch

from uniserve_worker.capabilities import GraphBucket
from uniserve_worker.execution.batch import (
    Batch,
    BatchPartition,
    CompletionReport,
    DeferredCompletion,
    DevicePoint,
    Domain,
    DType,
    ErrorCode,
    FinishFlags,
    FixedPoint,
    ForwardMode,
    LatentPlacement,
    LogicalLengths,
    ModelOutput,
    Operation,
    OpStatus,
    PartitionCompletion,
    ProductKind,
    ProductPayload,
    ProductRef,
    RegistrationAck,
    Release,
    RequestKey,
    ShapeBound,
    StorageClass,
    TimingCounters,
    TokenSpan,
    VersionRef,
    WorkerForwardStats,
    decode_sampling_state_bytes,
    decode_token_product_bytes,
)
from uniserve_worker.execution.forward_batch import (
    AttentionSelection,
    ModelPhase,
    RouteMeshView,
)
from uniserve_worker.execution.trace import (
    ExecutionPhase,
    ExecutionTrace,
    OperationTrace,
)
from uniserve_worker.foundation.errors import (
    WorkerError,
    WorkerErrorCode,
    capability_mismatch,
    classify,
    invalid_descriptor,
    should_capture_trace,
    unsupported_operation,
)
from uniserve_worker.loader.weight_set import WeightSet
from uniserve_worker.models.generation import (
    GenerationPipeline,
)
from uniserve_worker.models.inputs import ImageProcessor
from uniserve_worker.models.minimax_h3 import MiniMaxH3Model
from uniserve_worker.models.minimax_h3.execution import H3MuxCoordinator, H3OutputRing
from uniserve_worker.models.runtime import (
    ExecutionModel,
    WorkerDeployment,
)
from uniserve_worker.nn.diffusion.cfg import build_flow_cfg_plan
from uniserve_worker.nn.diffusion.integrator import euler_step
from uniserve_worker.nn.diffusion.schedule import (
    x_pred_to_velocity,
)
from uniserve_worker.nn.mesh import DeviceMesh
from uniserve_worker.runtime.cache_pool import CachePool
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
from uniserve_worker.runtime.runtime_states import RuntimeStates
from uniserve_worker.server.completion import (
    DeferredResult,
    DeferredImagePayload,
    DeferredLogprobPayload,
    DeferredTransferPayload,
    PinnedOutputBuffer,
    PinnedTokenCapture,
)
from uniserve_worker.server.cpu_tasks import BoundedCpuTaskPool
from uniserve_worker.server.profiler import profile_range
from uniserve_worker.server.request_state import (
    RequestRow,
    RequestRuntime,
    RequestTable,
)
from uniserve_worker.transfer.connector import CachePublication, CachePublications
from uniserve_worker.transfer.tickets import (
    TRANSFER_DESCRIPTOR_PREFIX,
    Locator,
    Transport,
    decode_transfer_descriptor,
)

from .attention import columns as _attention_columns
from .attention import dense_columns as _dense_attention_columns
from .cuda_graph import GraphExecutionError
from .model_runner import ForwardResult, ModelRunner, RunObservation, RunPath
from .resources import ExecutionResources
from .rows import (
    DecodeRuntimePublication,
    ForwardRow,
    LatentExecution,
    OperationIdentity,
    OperationState,
    Outcome,
    PartitionLayout,
    PartitionState,
    PreparedExecution,
    PreparedPredicateBatch,
    PreparedTransferInput,
    SampleWork,
    SpeculativeSelection,
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
        ForwardMode.GEN_TRANSITION,
        ForwardMode.GEN_FLOW,
        ForwardMode.GEN_DECODE,
        ForwardMode.MATERIALIZE,
    }
)
MIXED_SERVICE_SERIAL_NUMERATOR = 5
MIXED_SERVICE_SERIAL_DENOMINATOR = 4


def create_execution_resources(
    *,
    runner: ModelRunner | None,
    model: ExecutionModel | MiniMaxH3Model,
    deployment: WorkerDeployment,
    attention: AttentionSelection | None,
    requests: RequestTable,
    runtime_states: RuntimeStates | None,
    cache_pool: CachePool | None,
    req_to_token_pool: ReqToTokenPool | None,
    latent_pool: LatentPool | None,
    device_products: DeviceProducts,
    encoder_cache: EncoderCache,
    device_events: DeviceEventPool,
    cpu_tasks: BoundedCpuTaskPool,
    weights: WeightSet,
    mesh: DeviceMesh,
    transport: Transport | None,
    tokenizer: Any | None,
    model_name: str,
    weight_version: int,
    allowed_work_variants: frozenset[ForwardMode],
    mixed_buckets: tuple[GraphBucket, ...],
    trace: ExecutionTrace,
    h3_mux: H3MuxCoordinator | None = None,
    h3_output_ring: H3OutputRing | None = None,
    media_spool: Path | None = None,
) -> ExecutionResources:
    if not allowed_work_variants:
        raise ValueError("execution step must accept at least one work variant")
    if not model_name:
        raise capability_mismatch("execution model name is empty")
    if weights.version != weight_version:
        raise capability_mismatch("base-weight version does not match its weight set")
    unsupported = allowed_work_variants - model.supported_work
    if unsupported:
        raise capability_mismatch(
            "execution work set exceeds the model implementation: "
            f"{sorted(value.value for value in unsupported)!r}"
        )
    kv_resources = (runner, runtime_states, cache_pool, req_to_token_pool)
    if any(resource is None for resource in kv_resources) != all(
        resource is None for resource in kv_resources
    ):
        raise capability_mismatch("packed-forward resources must be allocated as one set")
    if model.resource_geometry.kv != (cache_pool is not None):
        raise capability_mismatch("execution resources disagree with model KV ownership")
    if cache_pool is not None and attention is None:
        raise capability_mismatch("packed-forward execution requires attention selection")
    h3_model = isinstance(model, MiniMaxH3Model)
    if h3_model != (media_spool is not None):
        raise capability_mismatch("H3 execution resources require one configured media spool")
    if h3_model and mesh.coord("sp") == 0 and (h3_mux is None or h3_output_ring is None):
        raise capability_mismatch("rank-zero H3 execution requires mux and output-ring resources")
    if not h3_model and (h3_mux is not None or h3_output_ring is not None):
        raise capability_mismatch("packed-forward execution cannot own H3 output resources")
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
        _h3_mux=h3_mux,
        _h3_output_ring=h3_output_ring,
        _media_spool=media_spool,
        device_products=device_products,
        encoder_cache=encoder_cache,
        _device_events=device_events,
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
    runtime._collective_history.clear()
    runtime._transport_publications.clear()
    runtime._flow_prefix_slots.clear()
    runtime._qualified_mixed_buckets.clear()


def install_weights(runtime: ExecutionResources, weights: WeightSet) -> None:
    if weights.version <= runtime.weights.version:
        raise ValueError("installed weight version must increase")
    if runtime.runner is not None:
        runtime.runner.invalidate_graphs(weights.version)
    runtime.weights = weights
    runtime.weight_version = weights.version


def _operation_identity(operation: Operation) -> OperationIdentity:
    return operation.request_key, int(operation.op_id)


def _reference_operation_identity(reference: ProductRef) -> OperationIdentity:
    return reference.request_key, int(reference.producer_op_id)


def _unique_scopes(scopes: Sequence[PartitionState]) -> tuple[PartitionState, ...]:
    unique: list[PartitionState] = []
    seen: set[int] = set()
    for scope in scopes:
        identity = id(scope)
        if identity not in seen:
            seen.add(identity)
            unique.append(scope)
    return tuple(unique)


def _protocol_error_code(code: WorkerErrorCode) -> ErrorCode:
    if code == WorkerErrorCode.RESOURCE_ERROR:
        return ErrorCode.RESOURCE_EXHAUSTED
    if code == WorkerErrorCode.COMPUTE_ERROR:
        return ErrorCode.COMPUTE_ERROR
    if code in {WorkerErrorCode.INVARIANT_VIOLATION, WorkerErrorCode.FATAL_WORKER_FAILURE}:
        return ErrorCode.INTERNAL
    return ErrorCode.INVALID_OPERATION


def prepare_batch(runtime, batch: Batch) -> PreparedExecution | None:
    """Submit bounded transfer and predicate observations without waiting."""

    from . import transfer

    entries = tuple(
        payload
        for payload in batch.input_products
        if payload.payload.startswith(TRANSFER_DESCRIPTOR_PREFIX)
    )
    transport = runtime.transport
    if entries and transport is None:
        raise capability_mismatch("cross-stage input requires a configured transport")
    transfers: list[PreparedTransferInput] = []
    for entry in entries:
        assert transport is not None
        kind, value = decode_transfer_descriptor(entry.payload)
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
                or any(
                    main.meta.get(name) != member
                    for name, member in {
                        "generation": generation,
                        "height": height,
                        "payload_kind": raw_payload_kind,
                        "width": width,
                    }.items()
                )
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
                or any(
                    main.meta.get(name) != member
                    for name, member in {
                        "generation": generation,
                        "height": height,
                        "value_range": raw_range,
                        "width": width,
                    }.items()
                )
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
            metadata = {
                "generation": generation,
                "height": height,
                "latent_units": latent_units,
                "step": step,
                "width": width,
            }
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
                or any(main.meta.get(name) != member for name, member in metadata.items())
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
            locators = tuple(Locator.from_wire_json(raw) for raw in snapshot.locators)
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
    batch: Batch,
    *,
    transfers: tuple[PreparedTransferInput, ...],
) -> PreparedPredicateBatch | None:
    operations = tuple(
        operation
        for operation in batch.operations
        if operation.predicate is not None and operation.predicate.kind is ProductKind.COMPLETION
    )
    if not operations:
        return None
    transferred = {transfer.product: transfer for transfer in transfers}
    buffer = PinnedOutputBuffer(
        len(operations),
        token_capacity=len(operations),
        devices=tuple(_operation_device(runtime, operation) for operation in operations),
        event_pool=runtime._device_events,
    )
    captures: list[tuple[OperationIdentity, PinnedTokenCapture, int]] = []
    pending: list[tuple[OperationIdentity, PreparedTransferInput, int]] = []
    recorded: list[DeviceProductRead] = []
    try:
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
        sealed = not pending
        if sealed:
            buffer.seal()
    except BaseException:
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


def execute_prepared(runtime, prepared: PreparedExecution) -> CompletionReport:
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

    missing_mixed = runtime.mixed_buckets - runtime._qualified_mixed_buckets
    if missing_mixed:
        raise GraphExecutionError(
            "mixed execution buckets lack a matched serving-path interference proof: "
            f"{sorted(missing_mixed, key=repr)!r}"
        )
    if runtime.runner is not None:
        runtime.runner.complete_startup()
    if runtime.requests.request_ids():
        raise RuntimeError("startup completed with resident requests")
    runtime._collective_history.clear()


def execute_batch(
    runtime,
    batch: Batch,
    *,
    prepared: tuple[PreparedTransferInput, ...] = (),
) -> CompletionReport:
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
    batch: Batch,
    *,
    catalog_graphs: bool = True,
) -> CompletionReport:
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
    batch: Batch,
    *,
    prepared: tuple[PreparedTransferInput, ...],
    predicate_values: Mapping[OperationIdentity, bool],
    propagate_errors: bool,
    graph_eligible: bool,
) -> CompletionReport:
    """Shared execution for startup and admitted traffic."""

    started = time.perf_counter_ns()
    operations = _trace_envelopes(batch.operations)
    validation_started = time.perf_counter_ns()
    try:
        _validate_batch(runtime, batch)
    except BaseException as error:
        runtime.trace.emit(
            ExecutionPhase.PROTOCOL_VALIDATION,
            operations,
            duration_us=(time.perf_counter_ns() - validation_started) // 1000,
            error=error,
        )
        raise
    runtime.trace.emit(
        ExecutionPhase.PROTOCOL_VALIDATION,
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
    runtime.requests.apply_controls(batch.controls)
    if not batch.operations:
        _apply_release_controls(runtime, batch)
        return CompletionReport(
            step_id=batch.step_id,
            partitions=(),
        )
    reports: dict[int, PartitionCompletion] = {}
    groups: dict[int, list[BatchPartition]] = {}
    for partition in batch.partitions:
        groups.setdefault(partition.submission_group, []).append(partition)

    for partitions in groups.values():
        scopes: list[PartitionState] = []
        for partition in partitions:
            try:
                scopes.append(
                    _open_partition(
                        runtime,
                        batch,
                        partition,
                        prepared,
                        predicate_values,
                        graph_eligible,
                    )
                )
            except BaseException as error:
                classified = _classify_partition_failure(
                    runtime,
                    partition,
                    error,
                    phase="partition registration",
                )
                if propagate_errors or classified.fatal:
                    for scope in scopes:
                        _discard_partition(runtime, scope, classified)
                    raise classified
                reports[partition.partition_id] = _registration_error_partition(
                    runtime,
                    batch.step_id,
                    partition,
                    classified,
                    started,
                )

        if not scopes:
            continue
        try:
            outcomes, execution_errors = _execute_partition_group(
                runtime,
                tuple(scopes),
                qualify_mixed=propagate_errors,
            )
        except BaseException as error:
            classified = _classify_partition_failure(
                runtime,
                partitions[0],
                error,
                phase="partition execution",
            )
            for scope in scopes:
                _discard_partition(runtime, scope, classified)
            if propagate_errors or classified.fatal:
                raise classified
            for scope in scopes:
                reports[scope.partition.partition_id] = _error_partition(
                    runtime,
                    batch.step_id,
                    scope,
                    classified,
                    started,
                )
            continue

        if propagate_errors and execution_errors:
            first_partition = next(
                partition for partition in partitions if partition.partition_id in execution_errors
            )
            classified = _classify_partition_failure(
                runtime,
                first_partition,
                execution_errors[first_partition.partition_id],
                phase="partition execution",
            )
            for scope in scopes:
                _discard_partition(runtime, scope, classified)
            raise classified

        for scope in scopes:
            partition_error = execution_errors.get(scope.partition.partition_id)
            if partition_error is not None:
                classified = _classify_partition_failure(
                    runtime,
                    scope.partition,
                    partition_error,
                    phase="partition execution",
                )
                _discard_partition(runtime, scope, classified)
                if propagate_errors or classified.fatal:
                    raise classified
                reports[scope.partition.partition_id] = _error_partition(
                    runtime,
                    batch.step_id,
                    scope,
                    classified,
                    started,
                )
                continue
            partition_outcomes = outcomes[scope.partition.partition_id]
            try:
                reports[scope.partition.partition_id] = _commit_partition(
                    runtime,
                    batch.step_id,
                    scope,
                    partition_outcomes,
                    started,
                )
            except BaseException as error:
                if scope.publication_started:
                    classified = _published_partition_failure(
                        runtime,
                        scope.partition,
                        error,
                    )
                else:
                    classified = _classify_partition_failure(
                        runtime,
                        scope.partition,
                        error,
                        phase="partition commit",
                    )
                    _discard_partition(runtime, scope, classified)
                if propagate_errors or classified.fatal:
                    raise classified
                reports[scope.partition.partition_id] = _error_partition(
                    runtime,
                    batch.step_id,
                    scope,
                    classified,
                    started,
                )

    _apply_release_controls(runtime, batch)
    report = CompletionReport(
        step_id=batch.step_id,
        partitions=tuple(reports[partition.partition_id] for partition in batch.partitions),
    )
    runtime.trace.emit(
        ExecutionPhase.COMMIT,
        operations,
        duration_us=(time.perf_counter_ns() - started) // 1000,
    )
    return report


def _classify_partition_failure(
    runtime,
    partition: BatchPartition,
    error: BaseException,
    *,
    phase: str,
) -> WorkerError:
    operations = tuple(
        (
            int(operation.request_key.authority_id),
            int(operation.request_key.session_id),
            int(operation.request_key.epoch),
            int(operation.op_id),
        )
        for operation in partition.operations
    )
    sole = partition.operations[0] if len(partition.operations) == 1 else None
    classified = classify(
        error,
        context=phase,
        phase=phase,
        operations=operations,
        req_id=None if sole is None else int(sole.request_key.session_id),
        op_id=None if sole is None else int(sole.op_id),
        op_kind=None if sole is None else sole.work.value,
        route=str(partition.route),
    )
    _log_partition_failure(runtime, partition, classified, cause=error)
    return classified


def _published_partition_failure(
    runtime,
    partition: BatchPartition,
    error: BaseException,
) -> WorkerError:
    operations = tuple(
        (
            int(operation.request_key.authority_id),
            int(operation.request_key.session_id),
            int(operation.request_key.epoch),
            int(operation.op_id),
        )
        for operation in partition.operations
    )
    classified = WorkerError(
        code=WorkerErrorCode.INVARIANT_VIOLATION,
        message=f"partition publication failed after visibility began: {error}",
        fatal=True,
        phase="partition publication",
        route=str(partition.route),
        operations=operations,
    )
    _log_partition_failure(runtime, partition, classified, cause=error)
    return classified


def _log_partition_failure(
    runtime,
    partition: BatchPartition,
    error: WorkerError,
    *,
    cause: BaseException | None = None,
) -> None:
    capture_trace = should_capture_trace(error.code)
    log = logger.error if capture_trace else logger.warning
    log(
        "partition failed: %s [code=%s partition_id=%s route=%s operations=%s]",
        error.message,
        error.code,
        partition.partition_id,
        partition.route,
        error.operations,
        exc_info=(type(cause), cause, cause.__traceback__)
        if capture_trace and cause is not None
        else None,
    )


def _open_partition(
    runtime,
    batch: Batch,
    partition: BatchPartition,
    prepared: tuple[PreparedTransferInput, ...],
    predicate_values: Mapping[OperationIdentity, bool],
    graph_eligible: bool,
) -> PartitionState:
    operations = partition.operations
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
    try:
        admission_slots = {
            admission.request_key: int(admission.request_pool_idx) for admission in admissions
        }
        request_pool_indices: list[int] = []
        for operation in operations:
            slot = admission_slots.get(operation.request_key)
            resident = runtime.requests.peek(operation.request_key.session_id)
            if (
                slot is None
                and resident is not None
                and resident.request_key == operation.request_key
            ):
                slot = int(resident.request_pool_idx)
            if slot is None:
                raise invalid_descriptor("operation request is not resident or admitted")
            request_pool_indices.append(slot)
        candidates, bases = runtime.requests.stage_partition(
            operations,
            admissions,
            tuple(request_pool_indices),
        )
        for operation, request in zip(operations, candidates, strict=True):
            request.install_runtime(_parent_runtime(runtime, operation, request))
        completion = PinnedOutputBuffer(
            len(operations),
            token_capacity=_partition_completion_words(runtime, operations),
            devices=_completion_devices(runtime, operations),
            event_pool=runtime._device_events,
        )
    except BaseException as error:
        runtime.trace.emit(
            ExecutionPhase.CANDIDATE_STAGE,
            traced,
            duration_us=(time.perf_counter_ns() - started) // 1000,
            error=error,
        )
        raise
    scope = PartitionState(
        partition=partition,
        started_ns=started,
        graph_eligible=graph_eligible,
        request_candidates=candidates,
        request_bases=bases,
        request_rows={request.session_id: request for request in candidates},
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
        if runtime.runtime_states is not None:
            runtime.runtime_states.reset(
                tuple(
                    int(request.request_pool_idx)
                    for request, base in zip(candidates, bases, strict=True)
                    if base is None
                )
            )
        _reserve_cpu_tasks(runtime, active_operations, scope)
        active_partition = _active_partition(runtime, partition, active_operations)
        if active_partition is not None:
            if runtime.cache_pool is None or runtime.req_to_token_pool is None:
                if (
                    active_partition.block_tables
                    or active_partition.new_cache_pages
                    or active_partition.forward_rows
                ):
                    raise capability_mismatch(
                        "KV-free execution received cache tables or packed forward rows"
                    )
            else:
                _bind_cache_tables(runtime, active_partition, scope)
        scope.layout = PartitionLayout(
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
        if active_partition is not None:
            _bind_latent_rows(runtime, active_partition, scope)
        _reserve_outputs(runtime, operations, scope)
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
        _record_component(scope, "open_partition", started)
        return scope
    except BaseException:
        _discard_partition(runtime, scope)
        raise


def _active_partition(
    runtime,
    partition: BatchPartition,
    operations: tuple[Operation, ...],
) -> BatchPartition | None:
    if not operations:
        return None
    if operations is partition.operations:
        return partition
    identities = {_operation_identity(operation) for operation in operations}
    old_to_new = {
        index: selected
        for selected, (index, operation) in enumerate(
            (
                item
                for item in enumerate(partition.operations)
                if _operation_identity(item[1]) in identities
            )
        )
    }
    return replace(
        partition,
        operations=operations,
        forward_rows=tuple(
            replace(row, operation_index=old_to_new[row.operation_index])
            for row in partition.forward_rows
            if row.operation_index in old_to_new
        ),
        latent_placements=tuple(
            placement
            for placement in partition.latent_placements
            if (placement.request_key, int(placement.op_id)) in identities
        ),
    )


def _partition_completion_words(runtime, operations: tuple[Operation, ...]) -> int:
    return max(
        1,
        SAMPLING_COMPLETION_FIELDS * len(operations)
        + sum((int(operation.bounds.max_completion_bytes) + 3) // 4 for operation in operations),
    )


def _execute_partition_group(
    runtime,
    scopes: tuple[PartitionState, ...],
    *,
    qualify_mixed: bool,
) -> tuple[dict[int, tuple[Outcome, ...]], dict[int, BaseException]]:
    from . import token

    for scope in scopes:
        active = tuple(
            operation
            for operation in scope.partition.operations
            if _operation_identity(operation) not in scope.predicated_operations
        )
        for device in _completion_devices(runtime, active):
            scope.completion.begin_device(device)
    grouped: list[list[Outcome | None]] = [
        [None] * len(scope.partition.operations) for scope in scopes
    ]
    group_active = tuple(
        operation
        for scope in scopes
        for operation in scope.partition.operations
        if _operation_identity(operation) not in scope.predicated_operations
    )
    homogeneous_decode = bool(group_active) and all(
        operation.work is ForwardMode.TOKEN_DECODE for operation in group_active
    )
    states: list[OperationState] = []
    locations: dict[int, tuple[int, int]] = {}
    for scope_index, scope in enumerate(scopes):
        active = tuple(
            operation
            for operation in scope.partition.operations
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
        for operation_index, operation in enumerate(scope.partition.operations):
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
            state = OperationState(operation=operation, partition=scope)
            locations[id(state)] = (scope_index, operation_index)
            states.append(state)
    errors = _run_ready_set(runtime, states, qualify_mixed=qualify_mixed)
    for state in states:
        scope_index, operation_index = locations[id(state)]
        if state.outcome is not None:
            grouped[scope_index][operation_index] = state.outcome
    outcomes: dict[int, tuple[Outcome, ...]] = {}
    for scope, partition_outcomes in zip(scopes, grouped, strict=True):
        partition_id = scope.partition.partition_id
        if partition_id in errors:
            continue
        if any(outcome is None for outcome in partition_outcomes):
            raise RuntimeError("successful partition did not resolve every operation")
        outcomes[partition_id] = tuple(cast(Outcome, outcome) for outcome in partition_outcomes)
    return outcomes, errors


def _run_ready_set(
    runtime: ExecutionResources,
    states: list[OperationState],
    *,
    qualify_mixed: bool,
) -> dict[int, BaseException]:
    from . import encode, flow, h3, token, transfer

    producers = {output: state for state in states for output in state.operation.outputs}
    errors: dict[int, BaseException] = {}

    def live(state: OperationState) -> bool:
        return state.outcome is None and state.partition.partition.partition_id not in errors

    while any(live(state) for state in states):
        forward: list[tuple[OperationState, object]] = []
        ready = tuple(
            state for state in states if live(state) and dependencies_ready(state, producers)
        )
        flow_ready = tuple(state for state in ready if state.operation.work is ForwardMode.GEN_FLOW)
        flow_ready_ids = {id(state) for state in flow_ready}
        for state in flow_ready:
            try:
                rows = _pack_state_forward(runtime, state)
            except BaseException as error:
                errors[state.partition.partition.partition_id] = error
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
                    errors[state.partition.partition.partition_id] = error
                    continue
                forward.extend((state, row) for row in rows)
        if forward:
            if runtime.runner is None:
                raise RuntimeError("KV-free execution packed a model forward row")
            outputs = _run_partitioned_wave(
                runtime,
                tuple((cast(ForwardRow, row), state.partition) for state, row in forward),
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
                    errors[state.partition.partition.partition_id] = error
            continue

        samples: dict[int, list[tuple[OperationState, SampleWork]]] = defaultdict(list)
        for state in states:
            if not live(state) or not dependencies_ready(state, producers):
                continue
            sample = token.pack_sample(state)
            if sample is not None:
                samples[state.partition.partition.partition_id].append(
                    (state, cast(SampleWork, sample))
                )
        if samples:
            for partition_id, candidates in samples.items():
                state = candidates[0][0]
                try:
                    values = _sample_task_batch(
                        tuple(sample for _state, sample in candidates),
                        state.partition.completion,
                        device_products=runtime.device_products,
                        device_reads=tuple(state.partition.device_reads),
                        selection_broadcast=partial(_broadcast_tp_selection, runtime),
                    )
                    for (candidate, _sample), value in zip(candidates, values, strict=True):
                        token.consume_sample(runtime, candidate, value)
                except BaseException as error:
                    errors[partition_id] = error
            continue

        progressed = False
        for state in states:
            if not live(state) or not dependencies_ready(state, producers):
                continue
            try:
                progressed = flow.integrate(runtime, state) or progressed
            except BaseException as error:
                errors[state.partition.partition.partition_id] = error
        if progressed:
            continue
        for state in states:
            if not live(state) or not dependencies_ready(state, producers):
                continue
            try:
                progressed = encode.run_action(runtime, state) or progressed
                progressed = transfer.run_action(runtime, state) or progressed
                progressed = h3.run_action(runtime, state) or progressed
            except BaseException as error:
                errors[state.partition.partition.partition_id] = error
        if progressed:
            continue
        if not any(live(state) for state in states):
            break
        blocked = tuple(_operation_identity(state.operation) for state in states if live(state))
        raise RuntimeError(f"execution ready set made no progress: {blocked!r}")
    return errors


def _pack_state_forward(
    runtime: ExecutionResources,
    state: OperationState,
) -> tuple[object, ...]:
    from . import encode, flow, token

    operation = state.operation
    if operation.work.token_mode is not None:
        return token.pack_forward(runtime, state)
    if operation.work is ForwardMode.GEN_FLOW and runtime.latent_pool is not None:
        return flow.pack_forward(runtime, state)
    if operation.work.encode_mode is not None or (
        operation.work is ForwardMode.MATERIALIZE and runtime.latent_pool is not None
    ):
        return encode.pack_forward(runtime, state)
    return ()


def _consume_state_forward(
    runtime: ExecutionResources,
    state: OperationState,
    outputs: tuple[torch.Tensor, ...],
) -> None:
    from . import encode, flow, token

    operation = state.operation
    if operation.work.token_mode is not None:
        token.consume_forward(runtime, state, outputs)
    elif operation.work is ForwardMode.GEN_FLOW and runtime.latent_pool is not None:
        flow.consume_forward(runtime, state, outputs)
    elif operation.work.encode_mode is not None or (
        operation.work is ForwardMode.MATERIALIZE and runtime.latent_pool is not None
    ):
        encode.consume_forward(runtime, state, outputs)
    else:
        raise RuntimeError("model output has no operation consumer")


def _commit_partition(
    runtime,
    step_id: int,
    scope: PartitionState,
    outcomes: tuple[Outcome, ...],
    started: int,
) -> PartitionCompletion:
    commit_started = time.perf_counter_ns()
    partition = scope.partition
    operations = partition.operations
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
    records: list[ModelOutput] = []
    selected_versions: dict[int, VersionRef] = {}
    pending_completions: dict[int, DeferredCompletion] = {}
    report_products: list[ProductPayload] = []
    resolved_runtime: dict[int, RequestRuntime] = {}
    layout = scope.layout
    if layout is None or layout.operations != operations:
        raise RuntimeError("partition commit lost its aligned candidate layout")
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
        pending = DeferredResult(
            (
                request.pending_operations.get(int(operation.parent.producer_op_id))
                if isinstance(operation.parent.point, DevicePoint)
                else None
            ),
            scope.completion,
            row,
            partial(_finalize_predicated_runtime, runtime, operation),
            status=outcome.status,
            selected_point=cast(int, outcome.selected_point),
            resolved_callback=(
                partial(_finalize_speculative_runtime, runtime, operation, outcome.selection)
                if outcome.selection is not None
                else None
            ),
            completion_tasks=(
                *outcome.completion_tasks,
                *(
                    cast(DeferredLogprobPayload, product.payload)
                    for product in outcome.products
                    if isinstance(product.payload, DeferredLogprobPayload)
                ),
            ),
        )
        record = pending.bind_record(
            ModelOutput(
                request_key=operation.request_key,
                op_id=operation.op_id,
                completion_slot_generation=scope.completion.generation,
                status=outcome.status,
                selected_point=cast(int, outcome.selected_point),
                logical_lengths=outcome.logical_lengths,
                token_span=outcome.token_span,
                committed_tokens=cast(tuple[int, ...], outcome.committed_tokens),
                finish_flags=outcome.finish_flags,
                product_generations=outcome.product_generations,
                error_code=None,
                timing_counters=TimingCounters(),
                deferred=pending,
            )
        )
        records.append(record)
        if operation.advances_state:
            pending_completions[operation.request_key.session_id] = pending
            selected_versions[operation.request_key.session_id] = VersionRef(
                request_key=operation.request_key,
                producer_op_id=operation.op_id,
                point=FixedPoint(cast(int, outcome.selected_point)),
            )
        else:
            selected = request.resolve_version(operation.parent)
            if selected is None:
                raise RuntimeError("non-state operation lost its resolved parent")
            selected_versions[operation.request_key.session_id] = selected
        resolved_runtime[operation.request_key.session_id] = RequestRuntime(
            logical_position=request.logical_position,
            rng_counter=request.rng_counter,
            latent_product=request.latent_product,
            flow_step=request.flow_step,
            kv_visible_len=outcome.logical_lengths.kv_visible_len,
            kv_computed_len=outcome.logical_lengths.kv_computed_len,
        )
    _record_component(scope, "commit_partition", commit_started)
    partition_report = PartitionCompletion(
        partition_id=partition.partition_id,
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
        step_id=step_id,
        operations=operations,
        candidates=scope.request_candidates,
        bases=scope.request_bases,
        selected_versions=selected_versions,
        runtimes=resolved_runtime,
        completions=pending_completions,
    )
    for identity, locators in scope.stage_publications.items():
        existing = runtime._transport_publications.get(identity)
        if existing is not None and existing != locators:
            raise RuntimeError("committed transport publication identity was reused")
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
    runtime.requests.publish(request_publication)
    return partition_report


def _commit_runtime_states(runtime, scope: PartitionState) -> None:
    states = runtime.runtime_states
    if states is None:
        if (
            scope.runtime_publications
            or scope.prompt_logits_publications
            or scope.runtime_cache_lengths
        ):
            raise RuntimeError("runtime state publication has no backing storage")
        return
    for slot, length in scope.runtime_cache_lengths.items():
        _copy_runtime_scalar(states.valid_cache_lengths[slot : slot + 1], length)
    for publication in scope.runtime_publications:
        if isinstance(publication, DecodeRuntimePublication):
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
    for prompt_publication in scope.prompt_logits_publications:
        states.prompt_logits[prompt_publication.slot].copy_(
            prompt_publication.logits.to(dtype=states.prompt_logits.dtype)
        )


def _discard_partition(
    runtime,
    scope: PartitionState,
    error: BaseException | None = None,
) -> None:
    _finish_device_reads(runtime, scope)
    for reservation in scope.cpu_tasks.values():
        reservation.abandon()
    for lease in scope.h3_output_leases.values():
        lease.release()
    if scope.publication_started:
        raise RuntimeError("published partition state cannot be discarded")
    scope.completion.abandon()
    runtime.device_products.abandon_writes(tuple(scope.device_writes))
    runtime.encoder_cache.abandon_writes(tuple(scope.encoder_writes))
    if runtime.latent_pool is not None and scope.latent_import_slots:
        runtime.latent_pool.release_slots(tuple(scope.latent_import_slots))
    _release_locators(runtime, scope.published)
    runtime.trace.emit(
        ExecutionPhase.CANDIDATE_DISCARD,
        _trace_envelopes(scope.partition.operations),
        error=error,
    )


def _reserve_cpu_tasks(
    runtime,
    operations: tuple[Operation, ...],
    scope: PartitionState,
) -> None:
    from uniserve_worker.models.minimax_h3 import MiniMaxH3Model

    h3_model = isinstance(runtime.model, MiniMaxH3Model)
    rank_zero = runtime.mesh.coord("sp") == 0 if h3_model else True
    for operation in operations:
        if operation.work is not ForwardMode.MATERIALIZE and not (
            h3_model and operation.work is ForwardMode.GEN_DECODE
        ):
            continue
        if not rank_zero:
            continue
        identity = _operation_identity(operation)
        if identity in scope.cpu_tasks:
            raise invalid_descriptor("materialization repeats its CPU task identity")
        reservation = runtime._cpu_tasks.reserve()
        try:
            if h3_model and operation.work is ForwardMode.GEN_DECODE:
                placement = next(
                    (
                        placement
                        for placement in scope.partition.decode_placements
                        if placement.request_key == operation.request_key
                        and int(placement.op_id) == int(operation.op_id)
                    ),
                    None,
                )
                if placement is None:
                    raise invalid_descriptor("H3 decode operation has no exact decode placement")
                scope.h3_output_leases[identity] = runtime.h3_output_ring().reserve(
                    placement.kind.value
                )
        except BaseException:
            reservation.abandon()
            raise
        scope.cpu_tasks[identity] = reservation


def _registration_error_partition(
    runtime,
    step_id: int,
    partition: BatchPartition,
    error: WorkerError,
    started: int,
) -> PartitionCompletion:
    generation = 1
    report = _build_error_partition(
        runtime,
        partition,
        generation,
        False,
        error,
        started,
        WorkerForwardStats(),
    )
    return report


def _error_partition(
    runtime,
    step_id: int,
    scope: PartitionState,
    error: WorkerError,
    started: int,
) -> PartitionCompletion:
    report = _build_error_partition(
        runtime,
        scope.partition,
        scope.completion.generation,
        scope.registration_visible,
        error,
        scope.started_ns,
        _forward_stats(scope.observations, scope.component_us),
    )
    return report


def _build_error_partition(
    runtime,
    partition: BatchPartition,
    generation: int,
    registration_visible: bool,
    error: WorkerError,
    started: int,
    forward_stats: WorkerForwardStats,
) -> PartitionCompletion:
    protocol_code = _protocol_error_code(error.code)
    records: list[ModelOutput] = []
    for operation in partition.operations:
        session = runtime.requests.peek(operation.request_key.session_id)
        selected_parent = (
            operation.parent
            if operation.parent.is_fixed()
            else None
            if session is None
            else session.resolve_version(operation.parent)
        )
        point = None if selected_parent is None else selected_parent.point
        selected_point = point.point_index if isinstance(point, FixedPoint) else 0
        lengths = (
            LogicalLengths()
            if session is None
            else _logical_lengths(runtime, operation, session, None)
        )
        placeholder = ModelOutput(
            request_key=operation.request_key,
            op_id=operation.op_id,
            completion_slot_generation=max(1, generation),
            status=OpStatus.ERROR,
            selected_point=selected_point,
            logical_lengths=lengths,
            token_span=TokenSpan(base=lengths.token_len, len=0),
            committed_tokens=(),
            finish_flags=FinishFlags(),
            product_generations=(),
            error_code=protocol_code,
            timing_counters=TimingCounters(),
        )
        records.append(placeholder)
    return PartitionCompletion(
        partition_id=partition.partition_id,
        completions=tuple(records),
        registration=RegistrationAck(visible=registration_visible),
        worker_exec_us=(time.perf_counter_ns() - started) // 1000,
        forward_stats=forward_stats,
    )


def _finalize_predicated_runtime(
    runtime,
    operation: Operation,
) -> tuple[VersionRef, RequestRuntime]:
    selected, runtime = runtime.requests.finalize_predicated(
        operation.request_key.session_id,
        operation.op_id,
        operation.parent,
    )
    return selected, runtime


def _finalize_speculative_runtime(
    runtime,
    operation: Operation,
    selection: SpeculativeSelection,
    record: ModelOutput,
) -> None:
    tokens = tuple(int(value) for value in record.committed_tokens)
    selected_point = len(tokens)
    accepted = int(selection.accepted)
    expected_point = int(selection.selected_point)
    if (
        selected_point != expected_point
        or accepted > len(selection.draft_tokens)
        or int(record.selected_point) != expected_point
    ):
        raise RuntimeError("speculative completion selection is inconsistent")
    selected_kv = selection.base_kv_visible + selected_point
    if (
        record.logical_lengths.kv_computed_len != selection.initialized_kv
        or selected_kv > record.logical_lengths.kv_computed_len
    ):
        raise RuntimeError("speculative KV selection is outside initialized state")
    prefixes: list[tuple[VersionRef, RequestRuntime]] = []
    for point_index in range(1, selected_point + 1):
        request = runtime.requests.get(operation.request_key.session_id)
        prefix_runtime = RequestRuntime(
            logical_position=selection.base_logical_position + point_index,
            rng_counter=selection.base_rng_counter + point_index,
            latent_product=request.latent_product,
            flow_step=request.flow_step,
            kv_visible_len=selection.base_kv_visible + point_index,
            kv_computed_len=record.logical_lengths.kv_computed_len,
        )
        prefixes.append(
            (
                VersionRef(
                    request_key=operation.request_key,
                    producer_op_id=operation.op_id,
                    point=FixedPoint(point_index),
                ),
                prefix_runtime,
            )
        )
    runtime.requests.finalize_prefixes(
        operation.request_key.session_id,
        operation.op_id,
        prefixes,
    )


def _validate_batch(runtime, batch: Batch) -> None:
    if len(batch.operations) > runtime.deployment.max_batch_operations:
        raise invalid_descriptor("execution batch exceeds the deployment operation limit")
    for operation in batch.operations:
        variant = operation.work
        if variant not in runtime.allowed_work_variants:
            raise unsupported_operation(variant.value, operation.request_key.session_id)
    if any(
        index > runtime.deployment.max_request_pool_size
        for partition in batch.partitions
        for index in (
            *(table.request_pool_idx for table in partition.block_tables),
            *(row.request_pool_index for row in partition.forward_rows),
        )
    ):
        raise invalid_descriptor("execution batch exceeds request-slot capacity")
    groups: dict[int, list[BatchPartition]] = defaultdict(list)
    for partition in batch.partitions:
        groups[partition.submission_group].append(partition)
    for partitions in groups.values():
        if len(partitions) < 2:
            continue
        variants = {
            operation.work for partition in partitions for operation in partition.operations
        }
        if not runtime.model.tensorized_mixed or variants != {
            ForwardMode.TOKEN_DECODE,
            ForwardMode.GEN_FLOW,
        }:
            raise invalid_descriptor(
                "tensorized mixed submission exceeds worker mixed-execution capabilities"
            )
        capability = _mixed_capability(runtime, tuple(partitions))
        if capability not in runtime.mixed_buckets:
            raise invalid_descriptor(
                "tensorized mixed submission has no exact qualified capability bucket"
            )
    if isinstance(runtime.model, MiniMaxH3Model):
        from .h3 import validate_batch

        validate_batch(runtime, batch)
    validate_collective_sequence(runtime.mesh, runtime._collective_history, batch)


def validate_collective_sequence(
    mesh: DeviceMesh,
    history: OrderedDict[int, object],
    batch: Batch,
) -> None:
    """Reject divergent or non-advancing collective identities across all worker roots."""

    groups: dict[int, list[BatchPartition]] = defaultdict(list)
    for partition in batch.partitions:
        groups[partition.submission_group].append(partition)
    group_identities: list[tuple[int, object]] = []
    for submission_group, partitions in groups.items():
        collective_seq = partitions[0].collective_seq
        identity = (
            int(submission_group),
            int(collective_seq),
            tuple(
                (partition.partition_id, partition.route, partition.domain, partition.operations)
                for partition in sorted(partitions, key=lambda value: value.partition_id)
            ),
        )
        group_identities.append((int(collective_seq), identity))
    for collective_seq, collective_identity in sorted(group_identities, key=lambda item: item[0]):
        existing = history.get(collective_seq)
        if existing is not None:
            if existing != collective_identity:
                raise invalid_descriptor("collective sequence was reused with different work")
            continue
        # Collective positions order cross-rank collectives, so multi-rank
        # execution must consume them monotonically. A single-rank worker
        # runs no collectives and may launch session-disjoint submissions
        # in admission-priority order, so only sequence reuse is checked.
        if mesh.tp_size > 1 and history and collective_seq <= next(reversed(history)):
            raise invalid_descriptor("collective sequence does not advance")
        history[collective_seq] = collective_identity
        while len(history) > 4096:
            history.popitem(last=False)


def _completion_devices(runtime, operations: tuple[Operation, ...]) -> tuple[str, ...]:
    deployment = runtime.deployment
    generation_device = deployment.generation_device
    device = deployment.device
    selected: list[str] = []
    for operation in operations:
        target = (
            generation_device
            if generation_device is not None and operation.work in _GENERATION_WORK_VARIANTS
            else device
        )
        if target not in selected:
            selected.append(target)
    return tuple(selected)


def _mixed_capability(
    runtime,
    partitions: tuple[BatchPartition, ...],
) -> GraphBucket:
    decode_rows = sum(
        operation.work is ForwardMode.TOKEN_DECODE
        for partition in partitions
        for operation in partition.operations
    )
    flow_operations = tuple(
        operation
        for partition in partitions
        for operation in partition.operations
        if operation.work is ForwardMode.GEN_FLOW
    )
    flow_placements = {
        (placement.request_key, int(placement.op_id)): placement
        for partition in partitions
        for placement in partition.latent_placements
    }
    branch_counts: dict[tuple[RequestKey, int], int] = defaultdict(int)
    generation = runtime.model.generation
    if flow_operations and generation is None:
        raise invalid_descriptor("tensorized mixed flow has no generation runtime")
    for partition in partitions:
        for index, operation in enumerate(partition.operations):
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
                    for row in partition.forward_rows
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
    scope: PartitionState,
) -> None:
    """Bind each declared device value to its concrete bounded owner."""

    from . import transfer

    scalar_groups: dict[
        tuple[torch.device, ProductKind, DType, ShapeBound],
        list[tuple[ProductRef, torch.device | str]],
    ] = {}
    general_bindings: list[tuple[ProductRef, torch.device | str]] = []
    encoder_bindings: list[tuple[ProductRef, torch.device | str]] = []
    for operation in operations:
        device = _operation_device(runtime, operation)
        for output in operation.outputs:
            if (
                _operation_identity(operation) in scope.predicated_operations
                and output.kind is not ProductKind.COMPLETION
            ):
                continue
            if operation.work is ForwardMode.TRANSFER_PRODUCT and transfer.transferable(output):
                continue
            if output.kind in {
                ProductKind.VISION_FEATURE,
                ProductKind.LATENT_FEATURE,
            }:
                encoder_bindings.append((output, device))
                continue
            if transfer.requires_device_product_binding(output):
                binding = (output, device)
                if output.shape_bound.max_elements == 1:
                    scalar_groups.setdefault(
                        (device, output.kind, output.dtype, output.shape_bound),
                        [],
                    ).append(binding)
                else:
                    general_bindings.append(binding)
    groups = tuple(tuple(group) for group in scalar_groups.values())
    if general_bindings:
        groups = (*groups, tuple(general_bindings))
    bound_groups = runtime.device_products.bind_output_groups(groups)
    scope.device_writes.extend(write for binding in bound_groups for write in binding.writes)
    scope.encoder_writes.extend(runtime.encoder_cache.bind_outputs(tuple(encoder_bindings)))
    operation_identities = {_operation_identity(operation) for operation in operations}
    token_operation_identities = {
        _operation_identity(operation)
        for operation in operations
        if operation.work.token_mode is not None
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
        elif write.reference.kind is ProductKind.ACCEPTED_SPAN:
            scope.accepted_span_writes[operation_identity] = write
        elif write.reference.kind is ProductKind.CONTINUATION:
            scope.state_continuation_writes[operation_identity] = write
        elif write.reference.kind is ProductKind.FINISH:
            scope.finish_writes[operation_identity] = write
        elif (
            write.reference.kind is ProductKind.COMPLETION
            and operation_identity in token_operation_identities
            and int(write.reference.output_index) in {4, 6}
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
    return (
        runtime._generation_device
        if operation.work
        in {
            ForwardMode.GEN_TRANSITION,
            ForwardMode.GEN_FLOW,
            ForwardMode.MATERIALIZE,
        }
        else runtime._device
    )


def _validate_completion_products(
    runtime,
    operation: Operation,
    products: tuple[ProductPayload, ...],
) -> None:
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
                    DeferredImagePayload,
                    DeferredLogprobPayload,
                    DeferredTransferPayload,
                ),
            )
            else len(product.payload)
        )
        transferred = isinstance(product.payload, DeferredTransferPayload)
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
    scope: PartitionState,
) -> None:
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
        point = operation.parent.point
        if isinstance(point, DevicePoint):
            selected = point.selected_point
            if selected is not None and selected.kind is not ProductKind.SELECTED_POINT:
                raise invalid_descriptor("device parent does not name a selected-point product")
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
    scope: PartitionState,
) -> None:
    producers = {_operation_identity(operation) for operation in scope.partition.operations}
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
    scope: PartitionState,
) -> None:
    declared = {_operation_identity(operation) for operation in operations}
    for identity, writes in scope.propagated_predicate_writes.items():
        if identity not in declared:
            raise RuntimeError("predicated output has no operation in its partition")
        for write in writes:
            runtime.device_products.publish_scalar_write(write, False)


def _finish_device_reads(
    runtime,
    scope: PartitionState,
) -> None:
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


def _apply_release_controls(runtime, batch: Batch) -> None:
    releases = tuple(
        (control.request_key, control.op_id)
        for control in batch.controls
        if isinstance(control, Release)
    )
    runtime.device_products.release_operations(releases)
    runtime.encoder_cache.release_operations(releases)
    if runtime.cache_publications is not None:
        runtime.cache_publications.release_operations(releases)
    consumed_predicates = tuple(
        int(predicate.generation)
        for operation in batch.operations
        if (predicate := operation.predicate) is not None
        and predicate.producer_op_id != operation.parent.producer_op_id
    )
    runtime.device_products.release_generations(consumed_predicates)
    if runtime.transport is not None:
        for identity in releases:
            _release_locators(runtime, runtime._transport_publications.pop(identity, ()))


def drop_session(runtime, session_id: int) -> None:
    """Release stage publications owned by one dropped request."""

    session = runtime.requests.peek(int(session_id))
    if session is not None:
        if runtime.runtime_states is not None:
            runtime.runtime_states.release((int(session.request_pool_idx),))
        if runtime.req_to_token_pool is not None:
            runtime.req_to_token_pool.release((int(session.request_pool_idx),))
    if runtime.cache_publications is not None:
        runtime.cache_publications.drop(session_id)
    if session is not None and runtime.req_to_token_pool is not None:
        runtime.req_to_token_pool.release(
            tuple(runtime._flow_prefix_slots.pop(session.request_key, ()))
        )
    if runtime.transport is None:
        return
    selected = tuple(
        identity
        for identity in runtime._transport_publications
        if int(identity[0].session_id) == int(session_id)
    )
    for identity in selected:
        _release_locators(runtime, runtime._transport_publications.pop(identity))


def _bind_latent_rows(
    runtime,
    partition: BatchPartition,
    scope: PartitionState,
) -> None:
    if not partition.latent_placements:
        return
    pool = runtime.latent_pool
    if pool is None:
        operations = {
            _operation_identity(operation): (
                operation,
                _request_row(runtime, scope, operation.request_key.session_id),
            )
            for operation in partition.operations
        }
        for placement in partition.latent_placements:
            identity = placement.request_key, int(placement.op_id)
            selected = operations.get(identity)
            if selected is None:
                raise invalid_descriptor(
                    "latent placement names an operation outside its partition"
                )
            operation, request = selected
            slot = int(request.request_pool_idx)
            if placement.page_table != (slot,):
                raise invalid_descriptor(
                    "pool-free latent placement must name its request-pool capacity token"
                )
            if operation.work is ForwardMode.GEN_TRANSITION:
                valid = int(placement.start_step) == 0 and int(placement.step_count) == 0
            elif operation.work is ForwardMode.GEN_FLOW:
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
    operations = {
        _operation_identity(operation): (
            operation,
            int(_request_row(runtime, scope, operation.request_key.session_id).request_pool_idx),
        )
        for operation in partition.operations
    }
    rows: list[tuple[OperationIdentity, LatentPlacement, int]] = []
    for placement in partition.latent_placements:
        identity = (placement.request_key, int(placement.op_id))
        selected = operations.get(identity)
        if selected is None:
            raise invalid_descriptor("latent placement names an operation outside its partition")
        operation, slot = selected
        session = _request_row(runtime, scope, operation.request_key.session_id)
        image = session.image
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
            int(session.flow_step) if transferred is None else int(cast(int, transferred.step))
        )
        if operation.work is ForwardMode.GEN_TRANSITION:
            if int(placement.start_step) != 0 or int(placement.step_count) != 0:
                raise invalid_descriptor("generation transition placement carries denoise steps")
        elif operation.work is ForwardMode.GEN_FLOW:
            if (
                int(placement.start_step) != committed_step
                or int(placement.step_count) < 1
                or int(placement.start_step) + int(placement.step_count) > int(image.steps)
                or (
                    int(operation.bounds.max_tokens) > 0
                    and int(placement.step_count) > int(operation.bounds.max_tokens)
                )
            ):
                raise invalid_descriptor("generation flow placement exceeds its committed schedule")
        elif int(placement.start_step) != committed_step or int(placement.step_count) != 0:
            raise invalid_descriptor("latent reader placement disagrees with committed step state")
        rows.append((identity, placement, slot))
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


def _latent_row(runtime, operation: Operation, scope: PartitionState) -> LatentExecution:
    row = scope.latent_rows.get(_operation_identity(operation))
    if row is None:
        raise invalid_descriptor("trajectory operation has no staged latent placement")
    return row


def _bind_cache_tables(
    runtime,
    partition: BatchPartition,
    scope: PartitionState,
) -> None:
    """Install scheduler tables and retain row-aligned forward coordinates."""

    started = time.perf_counter_ns()
    tables = []
    for table in partition.block_tables:
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
    for allocation in partition.new_cache_pages:
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
    for row in partition.forward_rows:
        rows_by_operation[int(row.operation_index)].append(row)

    for operation_index, operation in enumerate(partition.operations):
        session = _request_row(runtime, scope, operation.request_key.session_id)
        main_slot = int(session.request_pool_idx)
        parent_runtime = _parent_runtime(runtime, operation, session)
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


def _request_row(runtime, scope: PartitionState, session_id: int) -> RequestRow:
    try:
        return scope.request_rows[int(session_id)]
    except KeyError:
        raise invalid_descriptor(f"partition has no request row for session {session_id}") from None


def _consume_device_product(
    runtime,
    reference: ProductRef,
    scope: PartitionState,
    *,
    consumer_op_id: int,
    device: torch.device | str | None = None,
) -> DeviceProductRead:
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
    scope: PartitionState,
    *,
    consumer_op_id: int,
    device: torch.device | str | None = None,
) -> EncoderRead:
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
    session: RequestRow,
) -> RequestRuntime:
    parent = operation.parent
    point = parent.point
    runtime = (
        session.execution_runtime_for_operation(parent.producer_op_id, point.point_index)
        if isinstance(point, DevicePoint) and point.selected_point is None
        else None
    )
    if runtime is None:
        selected = session.resolve_version(parent)
        runtime = None if selected is None else session.runtime_for(selected)
    if runtime is None:
        raise invalid_descriptor("operation parent has no resolved runtime state")
    return runtime


def parent_runtime(runtime, operation: Operation, request: RequestRow) -> RequestRuntime:
    return _parent_runtime(runtime, operation, request)


def _cache_coordinates(
    runtime,
    operation: Operation,
    scope: PartitionState,
    *,
    group_id: int = 0,
) -> tuple[int, int, int, int]:
    request = _request_row(runtime, scope, operation.request_key.session_id)
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
    session: RequestRow,
    cache: tuple[int, int, int, int] | None,
    *,
    latent_len: int | None = None,
    computed_len: int | None = None,
) -> LogicalLengths:
    parent = _parent_runtime(runtime, operation, session)
    if cache is None:
        visible = parent.kv_visible_len
        computed = parent.kv_computed_len
    else:
        _slot, _group, visible, _capacity = cache
        computed = visible if computed_len is None else int(computed_len)
    return LogicalLengths(
        token_len=session.logical_position,
        kv_visible_len=visible,
        kv_computed_len=computed,
        latent_len=session.flow_step if latent_len is None else int(latent_len),
    )


def _stage_input_products(
    runtime,
    input_products: Sequence[ProductPayload],
    scope: PartitionState,
) -> None:
    """Decode ephemeral host inputs and publish transferred physical values."""

    for entry in input_products:
        product = entry.product
        if entry.payload.startswith(TRANSFER_DESCRIPTOR_PREFIX):
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
                    for operation in scope.partition.operations
                    if product in operation.inputs
                )
                if len(consumers) != 1:
                    raise invalid_descriptor("latent transfer must have one partition consumer")
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
                session = _request_row(runtime, scope, product.request_key.session_id)
                if session.latent_product is not None or int(session.flow_step) != 0:
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
                session.latent_product = product
                session.flow_step = int(step)
                continue
            tensors = transfer.tensors()
            if len(tensors) != 1:
                raise invalid_descriptor("product transfer produced an invalid tensor set")
            consumers = tuple(
                operation
                for operation in scope.partition.operations
                if product in operation.inputs or operation.predicate == product
            )
            if not consumers:
                raise invalid_descriptor("transferred product has no partition consumer")
            devices = {_operation_device(runtime, operation) for operation in consumers}
            if len(devices) != 1:
                raise invalid_descriptor("transferred product spans multiple consumer devices")
            device = next(iter(devices))
            if transfer.kind == "device_product":
                binding = runtime.device_products.bind_outputs(((product, device),))[0]
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
            encoder_binding = runtime.encoder_cache.bind_outputs(((product, device),))[0]
            scope.encoder_writes.append(encoder_binding)
            scope.transferred_encoder_features[product] = encoder_binding
            runtime.encoder_cache.publish(
                encoder_binding,
                tensors[0],
                EncoderMetadata(height=height, width=width),
            )
            continue
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
            raise invalid_descriptor("source image payload has an invalid storage contract")
        scope.input_images[product] = entry.payload.decode("utf-8")


def _predicated_outcome(
    runtime,
    operation: Operation,
    scope: PartitionState,
) -> Outcome:
    request = _request_row(runtime, scope, operation.request_key.session_id)
    lengths = _logical_lengths(runtime, operation, request, None)
    point = operation.parent.point
    selected_point = int(point.point_index)
    return Outcome(
        status=OpStatus.PREDICATED,
        selected_point=selected_point,
        logical_lengths=lengths,
        token_span=TokenSpan(base=int(lengths.token_len), len=0),
        finish_flags=FinishFlags(),
        product_generations=(),
    )


def _run_partitioned_wave(
    runtime,
    tasks: tuple[tuple[ForwardRow, PartitionState], ...],
    *,
    qualify_mixed: bool,
) -> tuple[torch.Tensor, ...]:
    grouped: dict[
        tuple[object, ...],
        list[tuple[int, ForwardRow, PartitionState]],
    ] = defaultdict(list)
    for index, (task, scope) in enumerate(tasks):
        grouped[
            (
                scope.partition.submission_group,
                _partition_identity(
                    runtime.runner, _phase_device(runtime, task.phase), scope.partition.domain
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
            raise invalid_descriptor("tensorized mixed submission is outside the model capability")
        indexes = tuple(index for index, _task, _scope in group)
        group_tasks = tuple(task for _index, task, _scope in group)
        group_scopes = tuple(scope for _index, _task, scope in group)
        target = _phase_device(runtime, group_tasks[0].phase)
        for scope in _unique_scopes(group_scopes):
            scope.completion.register_device(target)
        if qualify_mixed and len(kinds) > 1:
            output, observation, mixed_us = _run_startup_forward(
                runtime,
                group_tasks,
                group_scopes[0],
                target,
            )
            mixed_output = tuple(value.clone() for value in output)
            homogeneous: dict[
                str,
                list[tuple[int, ForwardRow, PartitionState]],
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
                if (
                    serial_us < 1
                    or mixed_us * MIXED_SERVICE_SERIAL_DENOMINATOR
                    > serial_us * MIXED_SERVICE_SERIAL_NUMERATOR
                ):
                    raise GraphExecutionError(
                        "mixed service exceeds the 5/4 serial homogeneous envelope: "
                        f"mixed_us={mixed_us} homogeneous_us={tuple(homogeneous_us)!r}"
                    )
                capability = _mixed_capability(
                    runtime, tuple(scope.partition for scope in _unique_scopes(group_scopes))
                )
                runtime._qualified_mixed_buckets.add(capability)
                logger.info(
                    "qualified mixed execution bucket=%r mixed_us=%d homogeneous_us=%r "
                    "serial_over_mixed=%.3f",
                    capability,
                    mixed_us,
                    tuple(homogeneous_us),
                    serial_us / mixed_us,
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
        for index, value in zip(indexes, output, strict=True):
            result[index] = value
    for device, event in output_events:
        torch.cuda.current_stream(device).wait_event(event)
    return tuple(cast(torch.Tensor, value) for value in result)


def _run_startup_forward(
    runtime,
    tasks: tuple[ForwardRow, ...],
    scope: PartitionState,
    target: torch.device,
    *,
    force_eager: bool = False,
) -> tuple[tuple[torch.Tensor, ...], RunObservation, int]:
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
    scope: PartitionState,
) -> ForwardResult:
    result = _run_forward_group(runtime, tasks, scope)
    scope.observations.append(result.observation)
    return result


def _broadcast_tp_selection(runtime, value: torch.Tensor) -> torch.Tensor:
    if runtime.mesh.tp_size <= 1:
        return value
    transport = runtime.mesh.transport("tp")
    return transport.broadcast(value, src=0)


def _group_key(runtime, task: ForwardRow) -> tuple[object, ...]:
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


def _partition_identity(runner: ModelRunner, device: torch.device, domain: Domain) -> int:
    for partition in runner.partitions:
        if partition.device == device and domain in partition.domains:
            return id(partition)
    raise invalid_descriptor(f"execution has no {domain.value!r} partition for {device}")


def _run_forward_group(
    runtime,
    tasks: tuple[ForwardRow, ...],
    scope: PartitionState,
    *,
    force_eager: bool = False,
) -> ForwardResult:
    target = _phase_device(runtime, tasks[0].phase)
    scope.completion.register_device(target)
    attention = (
        _attention_columns(runtime, tasks)
        if all(task.phase in {ModelPhase.TEXT, ModelPhase.DENOISE} for task in tasks)
        else _dense_attention_columns(len(tasks), tuple(task.query_tokens for task in tasks))
    )
    mesh = RouteMeshView(runtime.mesh, _phase_topology(runtime, tasks[0].phase))
    result = runtime.runner.forward(
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
        domain=scope.partition.domain,
    )
    if int(result.request_pool_indices.numel()) != len(tasks):
        raise RuntimeError("model runner returned without aligned request slots")
    for index, task in enumerate(tasks):
        task.request_pool_index = result.request_pool_indices[index : index + 1]
    return result


def _weights(runtime) -> WeightSet:
    return runtime.weights


def _model(runtime) -> ExecutionModel:
    return runtime.model


def _generation(runtime) -> GenerationPipeline:
    value = _model(
        runtime,
    ).generation
    if not isinstance(value, GenerationPipeline):
        raise invalid_descriptor("operation requires model generation behavior")
    return value


def _latent_pool(runtime) -> LatentPool:
    if runtime.latent_pool is None:
        raise capability_mismatch("operation requires a physical latent pool")
    return runtime.latent_pool


def _image_processor(runtime) -> ImageProcessor:
    value = _model(
        runtime,
    ).image_processor
    if not isinstance(value, ImageProcessor):
        raise invalid_descriptor("operation requires model image processing")
    return value


def _phase_device(runtime, phase: ModelPhase) -> torch.device:
    deployment = runtime.deployment
    if phase in {ModelPhase.ENCODE_LATENT, ModelPhase.DECODE_LATENT}:
        return torch.device(deployment.generation_device or deployment.device)
    return torch.device(deployment.device)


def _phase_topology(runtime, phase: ModelPhase) -> tuple[str, ...]:
    if phase in {ModelPhase.TEXT, ModelPhase.DENOISE}:
        return _model(
            runtime,
        ).text_topology
    return ("tp",)


def _task_shape(runtime, task: ForwardRow) -> tuple[int, ...]:
    if task.encode_pixels is not None:
        return tuple(int(value) for value in task.encode_pixels.shape)
    if task.latent is not None:
        return task.image_height, task.image_width
    return ()


def _group_graph_shape(runtime, tasks: tuple[ForwardRow, ...]) -> tuple[object, ...]:
    return (
        len(tasks),
        sum(task.query_tokens for task in tasks),
        tuple(task.query_tokens for task in tasks),
        tuple((task.image_height, task.image_width) for task in tasks if task.latent is not None),
    )


def _release_locators(runtime, locators: Iterable[Locator]) -> None:
    if runtime.transport is None:
        return
    for locator in locators:
        runtime.transport.release(locator)


def _trace_envelopes(
    operations: Sequence[Operation],
) -> tuple[OperationTrace, ...]:
    return tuple(
        OperationTrace(
            authority_id=int(operation.request_key.authority_id),
            session_id=int(operation.request_key.session_id),
            epoch=int(operation.request_key.epoch),
            op_id=int(operation.op_id),
            version=int(point.point_index) if isinstance(point, FixedPoint) else 0,
        )
        for operation in operations
        for point in (operation.parent.point,)
    )


def _fixed_parent(operation: Operation) -> FixedPoint:
    """Return the fixed parent point a depth-one operation commits over."""

    point = operation.parent.point
    if not isinstance(point, FixedPoint):
        raise invalid_descriptor("operation names a device parent; depth one commits fixed")
    return point


def _output_generations(operation: Operation) -> tuple[int, ...]:
    return tuple(int(reference.generation) for reference in operation.outputs)


def _record_component(scope: PartitionState, name: str, started_ns: int) -> None:
    elapsed_us = max(0, (time.perf_counter_ns() - int(started_ns)) // 1000)
    scope.component_us[name] = scope.component_us.get(name, 0) + elapsed_us


def _forward_stats(
    observations: Sequence[RunObservation],
    component_us: Mapping[str, int] | None = None,
) -> WorkerForwardStats:
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
    "drop_session",
    "execute_batch",
    "execute_prepared",
    "execute_startup",
    "install_weights",
    "parent_runtime",
    "prepare_batch",
]
