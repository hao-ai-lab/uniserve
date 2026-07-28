"""Transactional lowering and postprocessing for the canonical execution batch."""

from __future__ import annotations

import hashlib
import math
import time
from collections import defaultdict
from collections.abc import Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from importlib import import_module
from typing import Any, TypeAlias, cast

import torch

from uniserve_worker.batch import (
    Batch,
    CachedProduct,
    EncodeDelta,
    EncodeKind,
    EncodeOperation,
    ExecutionResult,
    FlowDelta,
    FlowOperation,
    FrameRecord,
    ImageArtifact,
    ImageParams,
    InlineImage,
    LatentProduct,
    MaterializeDelta,
    MaterializeKind,
    MaterializeOperation,
    OperationEnvelope,
    OperationResult,
    PublishedProduct,
    ResultDelta,
    SamplingParams,
    SequenceDelta,
    SequenceEffect,
    SequenceMode,
    SequenceOperation,
    StagedProduct,
    TokenLogprob,
    TokenPolicy,
    TokenSource,
    TransferDelta,
    TransferKind,
    TransferOperation,
    WorkerForwardStats,
)
from uniserve_worker.batch import (
    TokenInput as WireTokenInput,
)
from uniserve_worker.forward import (
    AttentionSelection,
    AttnPlan,
    DecodeOutput,
    DecodeRow,
    EmptyKvView,
    EmptyLatentView,
    EmptyMeshView,
    EmptyOutputView,
    EncodeOutput,
    EncodeRow,
    FlowOutput,
    FlowPatches,
    FlowRow,
    ForwardContext,
    ForwardRow,
    ForwardRowOutput,
    GraphBinding,
    KvView,
    NoAttention,
    NoFlowConditioning,
    PackedAttentionPlan,
    PagedDecodePlan,
    PagedVarlenPlan,
    PatchInput,
    RouteId,
    TokenEmbeddings,
    TokenHidden,
    TokenIds,
    TokenLogits,
    TokenOutput,
    TokenRow,
    TokenSegments,
    TokenSelection,
    packed_tensor_views,
)
from uniserve_worker.forward import (
    EncodeKind as ForwardEncodeKind,
)
from uniserve_worker.foundation.errors import (
    capability_mismatch,
    invalid_descriptor,
    unsupported_operation,
)
from uniserve_worker.foundation.sizing import bucketed_length
from uniserve_worker.foundation.triton_compat import triton_device_supported
from uniserve_worker.loader.weight_set import WeightSet
from uniserve_worker.nn.diffusion.cfg import Branch, CfgRecipe, build_flow_cfg_plan
from uniserve_worker.nn.diffusion.integrator import euler_step
from uniserve_worker.nn.diffusion.schedule import (
    FlowMatchSchedule,
    ScheduleDirection,
    ScheduleShiftDomain,
    x_pred_to_velocity,
)
from uniserve_worker.nn.vision import get_flattened_position_ids_extrapolate
from uniserve_worker.nn.vision.patching import patchify_batch, unpatchify_batch
from uniserve_worker.runtime.adapter_store import AdapterStore
from uniserve_worker.runtime.execution_trace import (
    ExecutionPhase,
    ExecutionTrace,
    OperationTrace,
)
from uniserve_worker.runtime.host_staging import (
    TensorStager,
    TensorStagingSlot,
    copy_cpu_to_device,
    cpu_int_staging_buffer,
    fill_cpu_ints,
    is_pinned,
)
from uniserve_worker.runtime.image_utils import tensor_to_png_b64
from uniserve_worker.runtime.kv_store import KvEntry, KvStore, KvTxn
from uniserve_worker.runtime.latent_store import LatentRecord, LatentStore, LatentTxn, LatentTxnView
from uniserve_worker.runtime.mesh_store import MeshStore
from uniserve_worker.runtime.product_store import (
    EncodedImageProduct,
    FrameCollectionProduct,
    ImageRange,
    ImageTensorProduct,
    LatentFeatureProduct,
    LogitsProduct,
    ProductRecord,
    ProductStore,
    ProductTxn,
    ProductView,
    VisionFeatureProduct,
    encoder_handle_from_content_hash,
)
from uniserve_worker.runtime.replay import ReplayStore
from uniserve_worker.runtime.request_session import (
    RequestSession,
    SampledTokenRelay,
    SessionStore,
    StepTxn,
)
from uniserve_worker.runtime.rng import (
    flow_noise_seed,
    normal_noise,
    sampling_draw_seed,
    uniform_samples,
)
from uniserve_worker.runtime.transfer import Locator, Transport, fetch_locator
from uniserve_worker.spec import (
    DeploymentOverlay,
    FeatureLayout,
    FlowBranchSource,
    FlowConditioningKind,
    FlowSpec,
    ImagePatchSpec,
    LatentLayout,
    MaterializationKind,
    ModelLoadScope,
    ModelSpec,
    NoiseScaleMode,
    OperationStageCondition,
    OperationStagePurpose,
    OperationStageSpec,
    OperationType,
    PositionLayout,
    RoutePlacement,
    RouteRowKind,
    RouteShapeGrouping,
    RouteSpec,
    resolved_digest,
)

from ._forward_plan import (
    ForwardPlan,
    GraphKey,
    OutputKind,
    OutputSlot,
    TransactionId,
)
from ._inputs import PreparedImage, prepare_image, prepare_tensor_image
from .model_runner import ModelRunner, RunObservation, RunPath


@dataclass(slots=True)
class _ForwardTask:
    envelope: OperationEnvelope
    stage: OperationStageSpec
    route: RouteSpec
    row: ForwardRow
    entry: KvEntry | None = None
    scratch: bool = False
    write_kv: bool = False
    causal: bool = True
    attention_indexes: torch.Tensor | None = None
    text_local_indices: tuple[int, ...] = ()

    @property
    def query_tokens(self) -> int:
        if isinstance(self.row, TokenRow):
            return _token_input_length(self.row)
        if isinstance(self.row, FlowRow):
            return int(self.row.image_tokens)
        return 0

    @property
    def row_kind(self) -> RouteRowKind:
        if isinstance(self.row, TokenRow):
            return RouteRowKind.TOKEN
        if isinstance(self.row, FlowRow):
            return RouteRowKind.FLOW
        if isinstance(self.row, EncodeRow):
            return RouteRowKind.ENCODE
        return RouteRowKind.DECODE


@dataclass(frozen=True, slots=True)
class _SamplingRow:
    parameters: SamplingParams
    recent_counts: tuple[tuple[int, int], ...]
    allowed: tuple[int, ...] | None
    suppress: tuple[int, ...]
    draw_seed: int
    n_logprobs: int


@dataclass(frozen=True, slots=True)
class _SampleTask:
    envelope: OperationEnvelope
    logits: torch.Tensor
    rows: tuple[_SamplingRow, ...]
    noise: torch.Tensor | None
    penalty_token_ids: torch.Tensor | None
    penalty_counts: torch.Tensor | None
    parameter_values: torch.Tensor | None
    defer_host_token: bool = False
    draft_token_ids: tuple[int, ...] = ()
    acceptance_uniforms: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class _SampleResult:
    token_id: int | _DeferredToken
    device_token: torch.Tensor | None
    logprob: float | None
    top_logprobs: tuple[tuple[int, float, int], ...] | None
    num_accepted_tokens: int = 0


class _DeferredTokenBatch:
    """One device token vector with one asynchronous pinned-host mirror."""

    __slots__ = ("device_tokens", "host_tokens", "event", "count", "_values")

    def __init__(
        self,
        device_tokens: torch.Tensor,
        host_tokens: torch.Tensor,
        event: torch.cuda.Event | None,
    ) -> None:
        self.device_tokens = device_tokens
        self.host_tokens = host_tokens
        self.event = event
        self.count = int(device_tokens.numel())
        self._values: tuple[int, ...] | None = None

    def ready(self) -> bool:
        return self._values is not None or self.event is None or bool(self.event.query())

    def finalize(self) -> tuple[int, ...]:
        if self._values is None:
            if self.event is not None:
                self.event.synchronize()
            self._values = tuple(int(value) for value in self.host_tokens[: self.count].tolist())
        return self._values


class _DeferredToken:
    """A protocol integer finalized only when the worker serializes its result."""

    __slots__ = ("batch", "index")

    def __init__(self, batch: _DeferredTokenBatch, index: int) -> None:
        self.batch = batch
        self.index = int(index)

    def ready(self) -> bool:
        return self.batch.ready()

    def finalize(self) -> int:
        return self.batch.finalize()[self.index]

    def __int__(self) -> int:
        return self.finalize()

    def __index__(self) -> int:
        return self.finalize()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _DeferredToken):
            return self.finalize() == other.finalize()
        if isinstance(other, int):
            return self.finalize() == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.finalize())


class _PinnedTokenRing:
    """Bounded pinned mirrors matched to the worker execution pipeline depth."""

    def __init__(self, depth: int, capacity: int) -> None:
        self.depth = max(1, int(depth))
        self.capacity = max(1, int(capacity))
        self._slots: dict[
            str, list[tuple[torch.Tensor, torch.cuda.Event, _DeferredTokenBatch | None]]
        ] = {}
        self._next: dict[str, int] = {}

    def capture(self, tokens: torch.Tensor) -> _DeferredTokenBatch:
        flat = tokens.reshape(-1)
        if flat.device.type != "cuda":
            host = flat.to(device="cpu")
            return _DeferredTokenBatch(flat, host, None)
        key = str(flat.device)
        slots = self._slots.get(key)
        if slots is None:
            slots = [
                (
                    torch.empty(self.capacity, dtype=torch.long, device="cpu", pin_memory=True),
                    torch.cuda.Event(blocking=False),
                    None,
                )
                for _ in range(self.depth)
            ]
            self._slots[key] = slots
            self._next[key] = 0
        index = self._next[key]
        host, event, owner = slots[index]
        if owner is not None:
            owner.finalize()
        count = int(flat.numel())
        if count > int(host.numel()):
            raise RuntimeError("sampled-token mirror capacity is smaller than the sampling batch")
        host[:count].copy_(flat, non_blocking=True)
        event.record(torch.cuda.current_stream(flat.device))
        captured = _DeferredTokenBatch(flat, host, event)
        slots[index] = (host, event, captured)
        self._next[key] = (index + 1) % self.depth
        return captured


_ExecutorTask: TypeAlias = _ForwardTask | _SampleTask
_TaskResult: TypeAlias = tuple[Any, ...]
_Driver: TypeAlias = Generator[tuple[_ExecutorTask, ...], _TaskResult, ResultDelta]


@dataclass(slots=True)
class _ExecutionScope:
    transaction: StepTxn
    kv: KvTxn
    latents: LatentTxn
    latent_view: LatentTxnView
    products: ProductTxn
    product_view: ProductView
    published: list[Locator] = field(default_factory=list)
    observations: list[RunObservation] = field(default_factory=list)
    component_us: dict[str, int] = field(default_factory=dict)
    next_row_id: int = 0

    def row_id(self) -> int:
        value = self.next_row_id
        self.next_row_id += 1
        return value


@dataclass(frozen=True, slots=True)
class _StateOutcome:
    kv_tokens: int
    sequence: SequenceEffect | None = None


class ModelExecutor:
    """Own one typed operation step from validation through atomic publication."""

    def __init__(
        self,
        *,
        spec: ModelSpec | None,
        deployment: DeploymentOverlay | None,
        runner: ModelRunner | None,
        attention: AttentionSelection | None,
        sessions: SessionStore,
        kv: KvStore,
        latents: LatentStore,
        products: ProductStore,
        replay: ReplayStore,
        adapters: AdapterStore | None,
        mesh: MeshStore | None,
        transport: Transport | None,
        tokenizer: Any | None,
        model_spec_digest: str | None,
        weight_digest: str | None,
        allowed_operation_types: frozenset[OperationType],
        trace: ExecutionTrace,
        pipeline_depth: int = 1,
        defer_sampling: bool = False,
    ) -> None:
        if not allowed_operation_types:
            raise ValueError("executor must accept at least one operation type")
        if (spec is None) != (deployment is None):
            raise ValueError("model spec and deployment overlay must be present together")
        if runner is not None and (spec is None or adapters is None):
            raise ValueError("model execution requires declarations and an adapter store")
        if (runner is None) != (attention is None):
            raise ValueError("model runner and attention selection must be provisioned together")
        if spec is not None:
            declared_digest = resolved_digest(spec, cast(DeploymentOverlay, deployment))
            if model_spec_digest is None or declared_digest != model_spec_digest:
                raise capability_mismatch(
                    "executor model-spec identity does not match its declarations"
                )
            if weight_digest is None or adapters is None or adapters.base.digest != weight_digest:
                raise capability_mismatch(
                    "executor base-weight identity does not match its adapter store"
                )
            unsupported = allowed_operation_types - spec.operation_types()
            system_only = {OperationType.SEQUENCE_SAMPLE, OperationType.MATERIALIZE_FRAME}
            if unsupported - system_only:
                raise capability_mismatch(
                    "executor operation set exceeds the model declaration: "
                    f"{sorted(value.value for value in unsupported - system_only)!r}"
                )
        self.spec = spec
        self.deployment = deployment
        self.runner = runner
        self.attention = attention
        self.sessions = sessions
        self.kv = kv
        self.latents = latents
        self.products = products
        self.replay = replay
        self.adapters = adapters
        self.mesh = mesh
        self.transport = transport
        self.tokenizer = tokenizer
        self.model_spec_digest = model_spec_digest
        self.weight_digest = weight_digest
        self.allowed_operation_types = allowed_operation_types
        self.trace = trace
        self.defer_sampling = bool(defer_sampling)
        max_operations = 1024 if deployment is None else int(deployment.max_batch_operations)
        self._token_mirrors = _PinnedTokenRing(pipeline_depth, max_operations)
        self._tensor_stager = TensorStager(ring_depth=pipeline_depth)
        self._routes = {} if spec is None else {route.name: route for route in spec.routes}

    def execute(self, batch: Batch) -> ExecutionResult:
        """Execute one canonical batch with replay-before-mutation semantics."""

        started = time.perf_counter_ns()
        operations = _trace_envelopes(batch.operations)
        validation_started = time.perf_counter_ns()
        try:
            batch.validate()
            self._validate_batch_identity(batch)
        except BaseException as error:
            self.trace.emit(
                ExecutionPhase.PROTOCOL_VALIDATION,
                operations,
                duration_us=(time.perf_counter_ns() - validation_started) // 1000,
                error=error,
            )
            raise
        self.trace.emit(
            ExecutionPhase.PROTOCOL_VALIDATION,
            operations,
            duration_us=(time.perf_counter_ns() - validation_started) // 1000,
        )
        try:
            replayed = self.replay.lookup(batch.operations)
        except BaseException as error:
            self.trace.emit(ExecutionPhase.REPLAY, operations, error=error)
            raise
        if replayed is not None:
            result = replace(
                replayed,
                step_id=batch.step_id,
                worker_exec_us=(time.perf_counter_ns() - started) // 1000,
                forward_stats=WorkerForwardStats(),
            )
            result.validate_for(batch)
            self.trace.emit(
                ExecutionPhase.REPLAY,
                operations,
                duration_us=(time.perf_counter_ns() - started) // 1000,
            )
            return result

        transaction_started = time.perf_counter_ns()
        try:
            transaction = self.sessions.begin_step(
                batch.step_id,
                batch.operations,
                (self.kv, self.latents, self.products),
            )
        except BaseException as error:
            self.trace.emit(
                ExecutionPhase.TRANSACTION_OPEN,
                operations,
                duration_us=(time.perf_counter_ns() - transaction_started) // 1000,
                error=error,
            )
            raise
        self.trace.emit(
            ExecutionPhase.TRANSACTION_OPEN,
            operations,
            duration_us=(time.perf_counter_ns() - transaction_started) // 1000,
        )
        scope = _ExecutionScope(
            transaction=transaction,
            kv=cast(KvTxn, transaction.store_transaction(self.kv)),
            latents=cast(LatentTxn, transaction.store_transaction(self.latents)),
            latent_view=cast(LatentTxn, transaction.store_transaction(self.latents)).view(),
            products=cast(ProductTxn, transaction.store_transaction(self.products)),
            product_view=cast(ProductTxn, transaction.store_transaction(self.products)).view(),
        )
        try:
            self.sessions.prepare(batch)
            for admission in batch.admissions:
                self.kv.admit(admission)
            self._apply_leases_and_imports(batch, scope)
            deltas = self._execute_operations(batch.operations, scope)
            self.trace.emit(
                ExecutionPhase.POSTPROCESS,
                operations,
                duration_us=(time.perf_counter_ns() - started) // 1000,
            )
            result_operations = tuple(
                OperationResult.for_operation(operation, delta)
                for operation, delta in zip(batch.operations, deltas, strict=True)
            )
            result = ExecutionResult(
                step_id=batch.step_id,
                operations=result_operations,
                worker_exec_us=(time.perf_counter_ns() - started) // 1000,
                forward_stats=_forward_stats(scope.observations, scope.component_us),
            )
            result.validate_for(batch)
            self.replay.commit_atomic(
                batch.operations,
                result,
                lambda publish: transaction.commit(publish),
            )
            self.trace.emit(
                ExecutionPhase.COMMIT,
                _trace_envelopes(batch.operations),
                duration_us=(time.perf_counter_ns() - started) // 1000,
            )
            return result
        except BaseException as error:
            transaction.rollback()
            self._release_locators(scope.published)
            self.trace.emit(
                ExecutionPhase.ROLLBACK,
                _trace_envelopes(batch.operations),
                duration_us=(time.perf_counter_ns() - started) // 1000,
                error=error,
            )
            raise

    def _validate_batch_identity(self, batch: Batch) -> None:
        if (
            self.deployment is not None
            and len(batch.operations) > self.deployment.max_batch_operations
        ):
            raise invalid_descriptor("execution batch exceeds the deployment operation limit")
        for operation in batch.operations:
            if operation.operation_type not in self.allowed_operation_types:
                raise unsupported_operation(operation.operation_type.value, operation.session_id)
            if self.spec is not None:
                if operation.model_spec_digest != self.model_spec_digest:
                    raise invalid_descriptor(
                        "operation model-spec digest does not match this worker"
                    )
                if operation.weight_digest != self.weight_digest:
                    raise invalid_descriptor("operation weight digest does not match this worker")

    def _apply_leases_and_imports(self, batch: Batch, scope: _ExecutionScope) -> None:
        for envelope in batch.operations:
            operation = envelope.operation
            lease = getattr(operation, "lease", None)
            if lease is not None:
                self.kv.apply_lease(envelope.session_id, lease)
            if isinstance(operation, FlowOperation) and operation.conditioning is not None:
                if self.transport is None:
                    raise capability_mismatch("flow conditioning import requires a transport")
                deployment = cast(DeploymentOverlay, self.deployment)
                scope.kv.import_snapshot(
                    envelope.session_id,
                    operation.conditioning.for_tensor_rank(
                        deployment.tp_rank,
                        deployment.tp_size,
                    ),
                    self.transport,
                )

    def _driver(self, envelope: OperationEnvelope, scope: _ExecutionScope) -> _Driver:
        operation = envelope.operation
        if isinstance(operation, SequenceOperation):
            return self._sequence_driver(envelope, operation, scope)
        if isinstance(operation, FlowOperation):
            return self._flow_driver(envelope, operation, scope)
        if isinstance(operation, EncodeOperation):
            return self._encode_driver(envelope, operation, scope)
        if isinstance(operation, MaterializeOperation):
            return self._materialize_driver(envelope, operation, scope)
        return self._transfer_driver(envelope, operation, scope)

    def _execute_operations(
        self,
        envelopes: tuple[OperationEnvelope, ...],
        scope: _ExecutionScope,
    ) -> tuple[ResultDelta, ...]:
        if all(
            isinstance(envelope.operation, SequenceOperation)
            and envelope.operation.mode is SequenceMode.DECODE
            for envelope in envelopes
        ):
            return self._decode_batch(envelopes, scope)
        drivers = tuple(self._driver(envelope, scope) for envelope in envelopes)
        return self._drive(drivers, scope)

    def _decode_batch(
        self,
        envelopes: tuple[OperationEnvelope, ...],
        scope: _ExecutionScope,
    ) -> tuple[ResultDelta, ...]:
        build_started = time.perf_counter_ns()
        operations: list[SequenceOperation] = []
        sessions: list[RequestSession] = []
        tasks: list[_ForwardTask] = []
        for envelope in envelopes:
            operation = cast(SequenceOperation, envelope.operation)
            if not isinstance(operation.input, WireTokenInput):
                raise invalid_descriptor("sequence decode requires inline token input")
            session = self.sessions.get(envelope.session_id)
            if session.sampling is None:
                raise invalid_descriptor("sequence operation has no admitted sampling state")
            current = self._resolve_decode_token(operation.input, session)
            start, end = operation.position
            if end - start != 1:
                raise invalid_descriptor("sequence decode must cover exactly one logical position")
            operations.append(operation)
            sessions.append(session)
            tasks.append(
                self._token_task(
                    envelope,
                    (current,),
                    (start,),
                    TokenSelection.LAST_LOGITS,
                    scope,
                )
            )

        _record_component(scope, "text_build_batch", build_started)
        forward_started = time.perf_counter_ns()
        outputs = self._run_wave(tuple(tasks), scope)
        _record_component(scope, "text_model_forward", forward_started)
        logits: list[torch.Tensor] = []
        for task, output in zip(tasks, outputs, strict=True):
            logits.append(_token_logits(output)[-1])
            self._commit_task_kv(task, 1, scope)

        if self.defer_sampling:
            deferred: list[ResultDelta] = []
            for envelope, operation, row_logits in zip(
                envelopes,
                operations,
                logits,
                strict=True,
            ):
                published = self._publish_logits(
                    envelope,
                    row_logits,
                    SequenceMode.DECODE,
                    scope,
                )
                deferred.append(
                    SequenceDelta(
                        SequenceEffect(
                            kv_tokens=self.kv.get(envelope.session_id).length,
                            published_logits=published,
                            published_kv=self._publish_kv_if_requested(
                                envelope,
                                operation,
                                (),
                                scope,
                            ),
                        )
                    )
                )
            return tuple(deferred)

        sample_started = time.perf_counter_ns()
        sample_tasks = tuple(
            self._sample_task(
                envelope,
                row_logits,
                session,
                operation.policy,
                positions=(operation.position[0] + 1,),
            )
            for envelope, operation, session, row_logits in zip(
                envelopes,
                operations,
                sessions,
                logits,
                strict=True,
            )
        )
        samples = _sample_task_batch(sample_tasks, self._token_mirrors)
        _record_component(scope, "text_sample", sample_started)
        finalize_started = time.perf_counter_ns()
        deltas: list[ResultDelta] = []
        for envelope, operation, session, sampled in zip(
            envelopes,
            operations,
            sessions,
            samples,
            strict=True,
        ):
            session.rng_counter += 1
            session.last_sampled_token = _sample_relay_token(sampled)
            token_ids = cast(tuple[int, ...], (sampled.token_id,))
            deltas.append(
                SequenceDelta(
                    SequenceEffect(
                        sampled_token_ids=token_ids,
                        sampled_logprob=sampled.logprob,
                        top_logprobs=_token_logprobs(sampled.top_logprobs),
                        kv_tokens=self.kv.get(envelope.session_id).length,
                        published_kv=self._publish_kv_if_requested(
                            envelope,
                            operation,
                            token_ids,
                            scope,
                        ),
                    )
                )
            )
        _record_component(scope, "text_finalize", finalize_started)
        return tuple(deltas)

    def _drive(
        self, drivers: tuple[_Driver, ...], scope: _ExecutionScope
    ) -> tuple[ResultDelta, ...]:
        active: dict[int, tuple[_Driver, tuple[_ExecutorTask, ...]]] = {}
        completed: dict[int, ResultDelta] = {}
        for index, driver in enumerate(drivers):
            try:
                active[index] = (driver, next(driver))
            except StopIteration as done:
                completed[index] = done.value
        while active:
            flat: list[tuple[int, int, _ExecutorTask]] = []
            for driver_index, (_driver, tasks) in active.items():
                for task_index, task in enumerate(tasks):
                    flat.append((driver_index, task_index, task))
            if not flat:
                raise RuntimeError("execution driver yielded an empty task wave")
            outputs = self._run_task_wave(
                tuple(task for _driver, _task, task in flat),
                scope,
            )
            by_driver: dict[int, list[Any | None]] = {
                index: [None] * len(tasks) for index, (_driver, tasks) in active.items()
            }
            for (driver_index, task_index, _task), output in zip(flat, outputs, strict=True):
                by_driver[driver_index][task_index] = output
            next_active: dict[int, tuple[_Driver, tuple[_ExecutorTask, ...]]] = {}
            for driver_index, (driver, _tasks) in active.items():
                aligned = tuple(by_driver[driver_index])
                try:
                    next_active[driver_index] = (driver, driver.send(aligned))
                except StopIteration as done:
                    completed[driver_index] = done.value
            active = next_active
        return tuple(completed[index] for index in range(len(drivers)))

    def _run_task_wave(
        self,
        tasks: tuple[_ExecutorTask, ...],
        scope: _ExecutionScope,
    ) -> _TaskResult:
        result: list[Any | None] = [None] * len(tasks)
        forward = tuple(
            (index, task) for index, task in enumerate(tasks) if isinstance(task, _ForwardTask)
        )
        if forward:
            indexes, forward_tasks = zip(*forward, strict=True)
            for index, forward_output in zip(
                indexes,
                self._run_wave(tuple(forward_tasks), scope),
                strict=True,
            ):
                result[index] = forward_output
        sampling = tuple(
            (index, task) for index, task in enumerate(tasks) if isinstance(task, _SampleTask)
        )
        if sampling:
            indexes, sample_tasks = zip(*sampling, strict=True)
            for index, sample_output in zip(
                indexes,
                _sample_task_batch(tuple(sample_tasks), self._token_mirrors),
                strict=True,
            ):
                result[index] = sample_output
        if any(value is None for value in result):
            raise RuntimeError("executor task wave contains an unknown task type")
        return tuple(result)

    def _run_wave(
        self,
        tasks: tuple[_ForwardTask, ...],
        scope: _ExecutionScope,
    ) -> tuple[ForwardRowOutput, ...]:
        if self.runner is None:
            raise capability_mismatch("system-only executor received a neural operation")
        grouped: dict[tuple[object, ...], list[tuple[int, _ForwardTask]]] = defaultdict(list)
        for index, task in enumerate(tasks):
            grouped[self._group_key(task)].append((index, task))
        groups: list[list[tuple[int, _ForwardTask]]] = []
        for candidates in grouped.values():
            kinds = frozenset(task.row_kind for _index, task in candidates)
            route = candidates[0][1].route
            legal_mixed = any(
                kinds <= frozenset(combination) for combination in route.mixed_combinations
            )
            if len(kinds) > 1 and not legal_mixed:
                by_kind: dict[RouteRowKind, list[tuple[int, _ForwardTask]]] = defaultdict(list)
                for item in candidates:
                    by_kind[item[1].row_kind].append(item)
                groups.extend(by_kind.values())
            else:
                groups.append(candidates)

        result: list[ForwardRowOutput | None] = [None] * len(tasks)
        for group in groups:
            indexes, group_tasks = zip(*group, strict=True)
            plan = self._forward_plan(tuple(group_tasks), scope)
            output = self.runner.run(plan)
            observation = self.runner.last_observation
            if observation is None:
                raise RuntimeError("model runner returned without an execution observation")
            scope.observations.append(observation)
            for index, row_output in zip(indexes, output.rows, strict=True):
                result[index] = row_output
        return tuple(cast(ForwardRowOutput, value) for value in result)

    def _group_key(self, task: _ForwardTask) -> tuple[object, ...]:
        session = self.sessions.get(task.envelope.session_id)
        weights = self._weights(session)
        return (
            task.route.name,
            self._route_device(task.route),
            task.route.dtype,
            task.route.topology_axes,
            weights.digest,
            weights.version,
            self._hard_shape_key(task.route, task.row),
        )

    def _forward_plan(
        self,
        tasks: tuple[_ForwardTask, ...],
        scope: _ExecutionScope,
    ) -> ForwardPlan:
        route = tasks[0].route
        device = self._route_device(route)
        target = torch.device(device)
        staging_slot = self._tensor_stager.acquire(target)
        weights = self._weights(self.sessions.get(tasks[0].envelope.session_id))
        if any(
            self._weights(self.sessions.get(task.envelope.session_id)) is not weights
            for task in tasks
        ):
            if any(
                self._weights(self.sessions.get(task.envelope.session_id)).digest != weights.digest
                or self._weights(self.sessions.get(task.envelope.session_id)).version
                != weights.version
                for task in tasks
            ):
                raise RuntimeError("forward group contains different immutable weight sets")

        kv_tasks = tuple(task for task in tasks if task.write_kv)
        if kv_tasks and len(kv_tasks) != len(tasks):
            raise invalid_descriptor("one physical route cannot mix KV and non-KV rows")
        kv_view: KvView | EmptyKvView
        attention: AttnPlan
        try:
            if kv_tasks:
                kv_view, attention = self._attention_plan(
                    tasks,
                    scope,
                    target,
                    staging_slot,
                )
            else:
                kv_view = EmptyKvView()
                attention = NoAttention(backends=self._attention_selection())
        except BaseException:
            self._tensor_stager.mark_submitted(staging_slot, target)
            raise
        mesh = EmptyMeshView() if self.mesh is None else self.mesh.view(route.topology_axes)
        context = ForwardContext(
            kv=kv_view,
            latent=scope.latent_view
            if any(isinstance(task.row, FlowRow) for task in tasks)
            else EmptyLatentView(),
            attention=attention,
            mesh=mesh,
            output=EmptyOutputView(),
        )
        graph_shape = self._hard_shape_key(route, tasks[0].row)
        graph_key = GraphKey(
            model_revision=cast(ModelSpec, self.spec).revision or cast(str, self.weight_digest),
            spec_digest=cast(str, self.model_spec_digest),
            route=RouteId(route.name),
            shape=graph_shape,
            dtype=route.dtype,
            backend=self._attention_selection().identity,
            topology=self._topology_key(route),
        )
        slots = tuple(
            OutputSlot(
                task.row.row_id,
                task.row.output_slot,
                _output_kind(task.row),
                (
                    cast(FlowSpec, cast(ModelSpec, self.spec).flow).prediction_dtype
                    if isinstance(task.row, FlowRow)
                    else route.dtype
                ),
            )
            for task in tasks
        )
        plan = ForwardPlan(
            route=RouteId(route.name),
            rows=tuple(task.row for task in tasks),
            context=context,
            outputs=slots,
            transaction=TransactionId(
                tuple(
                    (
                        task.envelope.session_id,
                        task.envelope.epoch,
                        task.envelope.op_id,
                    )
                    for task in tasks
                ),
                tuple(task.envelope.base_version for task in tasks),
            ),
            graph_key=graph_key,
            graph_eligible=route.graph_eligible,
            device=device,
            weights=weights,
            staging_slot=staging_slot,
        )
        row_counts: dict[str, int] = {}
        for task in tasks:
            name = task.row_kind.value
            row_counts[name] = row_counts.get(name, 0) + 1
        self.trace.emit(
            ExecutionPhase.PLAN_CREATION,
            _trace_envelopes(tuple(task.envelope for task in tasks)),
            route=route.name,
            row_kind_counts=row_counts,
        )
        return plan

    def _attention_plan(
        self,
        tasks: tuple[_ForwardTask, ...],
        scope: _ExecutionScope,
        device: torch.device,
        staging_slot: TensorStagingSlot,
    ) -> tuple[KvView, PagedDecodePlan | PagedVarlenPlan | PackedAttentionPlan]:
        route = tasks[0].route
        pure_token_decode = all(
            isinstance(task.row, TokenRow) and task.query_tokens == 1 for task in tasks
        )
        if RouteRowKind.FLOW in route.row_kinds and not pure_token_decode:
            return self._packed_attention_plan(tasks, scope, device, staging_slot)
        sessions = tuple(task.envelope.session_id for task in tasks)
        query_lens = tuple(task.query_tokens for task in tasks)
        view = scope.kv.view(sessions, query_lens=query_lens)
        block_table = view.block_table(device, slot=staging_slot)
        cache_seqlens = view.cache_seqlens(device, slot=staging_slot)
        kv_lens = tuple(
            base + query for base, query in zip(view.base_lens, query_lens, strict=True)
        )
        context_capacity = int(block_table.shape[1]) * int(view.block_size)
        causal_values = {bool(task.causal) for task in tasks}
        if len(causal_values) != 1:
            raise invalid_descriptor("paged attention rows must share causal semantics")
        causal = causal_values.pop()
        binding = GraphBinding(_binding_identity(tasks))
        if all(query == 1 for query in query_lens):
            page_ids = _stage_ints(
                tuple(
                    task.entry.block_ids[task.entry.length // view.block_size]
                    for task in tasks
                    if task.entry is not None
                ),
                dtype=torch.int32,
                device=device,
                slot=staging_slot,
                name="decode_page_ids",
            )
            page_offsets = _stage_ints(
                tuple(cast(KvEntry, task.entry).length % view.block_size for task in tasks),
                dtype=torch.int32,
                device=device,
                slot=staging_slot,
                name="decode_page_offsets",
            )
            decode_attention = PagedDecodePlan(
                backends=self._attention_selection(),
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                kv_seqlens=_stage_ints(
                    kv_lens,
                    dtype=torch.int32,
                    device=device,
                    slot=staging_slot,
                    name="kv_lengths",
                ),
                query_lens=_stage_ints(
                    (1,) * len(tasks),
                    dtype=torch.int32,
                    device=device,
                    slot=staging_slot,
                    name="query_lengths",
                ),
                cache_seqlens_cpu=tuple(view.base_lens),
                kv_seqlens_cpu=kv_lens,
                query_lens_cpu=query_lens,
                decode_page_ids=page_ids,
                decode_page_offsets=page_offsets,
                max_context_len=context_capacity,
                causal=causal,
                binding=binding,
            )
            return view, decode_attention
        cu_q = _cumulative(
            query_lens,
            device,
            slot=staging_slot,
            name="query_offsets",
        )
        cu_k = _cumulative(
            kv_lens,
            device,
            slot=staging_slot,
            name="kv_offsets",
        )
        varlen_attention = PagedVarlenPlan(
            backends=self._attention_selection(),
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            query_lens=_stage_ints(
                query_lens,
                dtype=torch.int32,
                device=device,
                slot=staging_slot,
                name="query_lengths",
            ),
            kv_seqlens=_stage_ints(
                kv_lens,
                dtype=torch.int32,
                device=device,
                slot=staging_slot,
                name="kv_lengths",
            ),
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            output_indices=_stage_ints(
                tuple(sum(query_lens[: index + 1]) - 1 for index in range(len(query_lens))),
                dtype=torch.int64,
                device=device,
                slot=staging_slot,
                name="output_indices",
            ),
            cache_seqlens_cpu=tuple(view.base_lens),
            query_lens_cpu=query_lens,
            kv_seqlens_cpu=kv_lens,
            max_seqlen_q=max(query_lens),
            max_seqlen_k=context_capacity,
            max_context_len=context_capacity,
            causal=causal,
            binding=binding,
        )
        return view, varlen_attention

    def _packed_attention_plan(
        self,
        tasks: tuple[_ForwardTask, ...],
        scope: _ExecutionScope,
        device: torch.device,
        staging_slot: TensorStagingSlot,
    ) -> tuple[KvView, PackedAttentionPlan]:
        rows = tuple(
            (cast(KvEntry, task.entry), task.query_tokens, task.write_kv) for task in tasks
        )
        view = scope.kv.packed_view(rows)
        query_lens = tuple(task.query_tokens for task in tasks)
        base_lens = view.base_lens
        key_lens = tuple(base + query for base, query in zip(base_lens, query_lens, strict=True))
        # The kernel checks ``visible_end`` against ``max_seqlen_q``, so the query
        # bound and the tensor it sizes are bucketed together: one executable then
        # serves a range of chunk widths instead of one per exact width. Positions
        # past a row's own query length stay zero, the padding this plan already
        # uses for rows shorter than the widest one.
        max_query = bucketed_length(max(query_lens))
        visible = torch.zeros((len(tasks), max_query), dtype=torch.int32, device=device)
        index_parts: list[torch.Tensor] = []
        route_parts: list[torch.Tensor] = []
        text_indices: list[int] = []
        offset = 0
        for row, (task, base, query) in enumerate(zip(tasks, base_lens, query_lens, strict=True)):
            if task.causal:
                visible[row, :query] = torch.arange(
                    base + 1,
                    base + query + 1,
                    dtype=torch.int32,
                    device=device,
                )
            else:
                visible[row, :query] = base + query
            indexes = task.attention_indexes
            if indexes is None:
                indexes = _three_axis_positions(task.row, query)
            if tuple(indexes.shape) != (3, query):
                raise invalid_descriptor("packed attention indexes must have shape [3, query]")
            index_parts.append(indexes.to(device=device, dtype=torch.long))
            is_flow = isinstance(task.row, FlowRow)
            route_parts.append(torch.full((query,), is_flow, dtype=torch.bool, device=device))
            if isinstance(task.row, TokenRow):
                text_indices.extend(range(offset, offset + query))
            else:
                text_indices.extend(offset + value for value in task.text_local_indices)
            offset += query
        text = torch.tensor(text_indices, dtype=torch.long, device=device)
        write_page_ids, write_page_offsets, write_token_indices = view.write_plan(device)
        page_table = view.block_table(device)
        context_capacity = int(page_table.shape[1]) * int(view.block_size)
        attention = PackedAttentionPlan(
            backends=self._attention_selection(),
            indexes=torch.cat(index_parts, dim=1),
            route_indicators=torch.cat(route_parts, dim=0),
            text_indices=text,
            has_text=bool(text_indices),
            has_flow=any(
                isinstance(task.row, FlowRow) and len(task.text_local_indices) < task.query_tokens
                for task in tasks
            ),
            visible_end=visible,
            cu_seqlens_q=_cumulative(
                query_lens,
                device,
                slot=staging_slot,
                name="packed_query_offsets",
            ),
            page_table=page_table,
            seqused_k=torch.tensor(key_lens, dtype=torch.int32, device=device),
            write_page_ids=write_page_ids,
            write_page_offsets=write_page_offsets,
            write_token_indices=write_token_indices,
            max_seqlen_q=max_query,
            max_seqlen_k=context_capacity,
            use_prefix_bounds=True,
            fully_visible=all(not task.causal for task in tasks),
            binding=GraphBinding(_binding_identity(tasks)),
        )
        return view, attention

    def _weights(self, session: RequestSession) -> WeightSet:
        if self.adapters is None:
            raise RuntimeError("model route has no immutable weight authority")
        return self.adapters.view(session.adapter_id)

    def _route(self, stage: OperationStageSpec) -> RouteSpec:
        try:
            return self._routes[stage.route]
        except KeyError:
            raise invalid_descriptor(
                f"operation stage references unknown route {stage.route!r}"
            ) from None

    def _stages(self, operation_type: OperationType) -> tuple[OperationStageSpec, ...]:
        if self.spec is None:
            return ()
        return self.spec.operation(operation_type).stages

    def _primary_stage(self, operation_type: OperationType) -> OperationStageSpec:
        stages = tuple(
            stage
            for stage in self._stages(operation_type)
            if stage.purpose is OperationStagePurpose.PRIMARY
        )
        if len(stages) != 1:
            raise invalid_descriptor(
                f"operation {operation_type.value!r} requires exactly one primary neural stage"
            )
        return stages[0]

    def _state_stages(
        self,
        operation_type: OperationType,
        *,
        retain_image: bool,
    ) -> tuple[OperationStageSpec, ...]:
        return tuple(
            stage
            for stage in self._stages(operation_type)
            if stage.purpose is OperationStagePurpose.STATE
            and (stage.condition is OperationStageCondition.ALWAYS or retain_image)
        )

    def _route_device(self, route: RouteSpec) -> str:
        deployment = cast(DeploymentOverlay, self.deployment)
        if route.placement is RoutePlacement.GENERATION:
            return deployment.generation_device or deployment.device
        return deployment.device

    def _attention_selection(self) -> AttentionSelection:
        if self.attention is None:
            raise RuntimeError("model route has no attention selection")
        return self.attention

    def _topology_key(self, route: RouteSpec) -> str:
        deployment = cast(DeploymentOverlay, self.deployment)
        return f"{','.join(route.topology_axes)}:{deployment.tp_rank}/{deployment.tp_size}"

    def _hard_shape_key(self, route: RouteSpec, row: ForwardRow) -> tuple[int, ...]:
        if route.shape.grouping is RouteShapeGrouping.FLEXIBLE:
            return ()
        if route.shape.grouping is RouteShapeGrouping.IMAGE_GEOMETRY:
            if isinstance(row, (FlowRow, DecodeRow)):
                return row.image_height, row.image_width
            if isinstance(row, EncodeRow):
                pixels = row.inputs.pixels
                return tuple(int(value) for value in pixels.shape[-2:])
            return (_token_input_length(row),)
        return _row_tensor_shape(row)

    def _release_locators(self, locators: Iterable[Locator]) -> None:
        if self.transport is None:
            return
        for locator in locators:
            self.transport.release(locator)

    def _sequence_driver(
        self,
        envelope: OperationEnvelope,
        operation: SequenceOperation,
        scope: _ExecutionScope,
    ) -> _Driver:
        session = self.sessions.get(envelope.session_id)
        if session.sampling is None:
            raise invalid_descriptor("sequence operation has no admitted sampling state")
        if operation.mode is SequenceMode.SAMPLE:
            source = cast(PublishedProduct, operation.input)
            logits = self._fetch_logits(source, scope)
            task = self._sample_task(
                envelope,
                logits.reshape(-1, logits.shape[-1])[-1],
                session,
                operation.policy,
                positions=(operation.position[1],),
            )
            outputs = yield (task,)
            sample = _sample_result(outputs[0])
            session.last_sampled_token = _sample_relay_token(sample)
            session.rng_counter += 1
            return SequenceDelta(
                SequenceEffect(
                    sampled_token_ids=cast(tuple[int, ...], (sample.token_id,)),
                    sampled_logprob=sample.logprob,
                    top_logprobs=_token_logprobs(sample.top_logprobs),
                    kv_tokens=self.kv.get(envelope.session_id).length,
                )
            )

        inputs = cast(WireTokenInput, operation.input)
        if operation.mode is SequenceMode.EXTEND:
            return (yield from self._extend(envelope, operation, inputs, session, scope))
        if operation.mode is SequenceMode.DECODE:
            return (yield from self._decode(envelope, operation, inputs, session, scope))
        return (yield from self._verify(envelope, operation, inputs, session, scope))

    def _extend(
        self,
        envelope: OperationEnvelope,
        operation: SequenceOperation,
        inputs: WireTokenInput,
        session: RequestSession,
        scope: _ExecutionScope,
    ) -> _Driver:
        tokens = tuple(int(value) for value in inputs.token_ids)
        start, end = operation.position
        if end - start != len(tokens):
            raise invalid_descriptor("sequence extend positions do not align with its tokens")
        sampling = _require_sampling(session)
        wants_prompt = bool(inputs.return_all_logits or sampling.return_prompt_logprobs)
        selection = TokenSelection.ALL_LOGITS if wants_prompt else TokenSelection.LAST_LOGITS
        task = self._token_task(
            envelope,
            tokens,
            tuple(range(start, end)),
            selection,
            scope,
        )
        outputs = yield (task,)
        logits = _token_logits(outputs[0])
        self._commit_task_kv(task, len(tokens), scope)
        prompt = self._prompt_logprobs(
            session,
            tokens,
            logits,
            scope,
            enabled=wants_prompt,
        )
        sample_task = self._sample_task(
            envelope,
            logits[-1],
            session,
            operation.policy,
            positions=(end,),
        )
        sample = _sample_result((yield (sample_task,))[0])
        session.last_sampled_token = _sample_relay_token(sample)
        session.rng_counter += 1
        effect = SequenceEffect(
            sampled_token_ids=cast(tuple[int, ...], (sample.token_id,)),
            sampled_logprob=sample.logprob,
            top_logprobs=_token_logprobs(sample.top_logprobs),
            prompt_logprobs=prompt,
            kv_tokens=self.kv.get(envelope.session_id).length,
            published_kv=self._publish_kv_if_requested(
                envelope,
                operation,
                cast(tuple[int, ...], (sample.token_id,)),
                scope,
            ),
        )
        return SequenceDelta(effect)

    def _decode(
        self,
        envelope: OperationEnvelope,
        operation: SequenceOperation,
        inputs: WireTokenInput,
        session: RequestSession,
        scope: _ExecutionScope,
    ) -> _Driver:
        current = self._resolve_decode_token(inputs, session)
        start, end = operation.position
        if end - start != 1:
            raise invalid_descriptor("sequence decode must cover exactly one logical position")
        task = self._token_task(
            envelope,
            (current,),
            (start,),
            TokenSelection.LAST_LOGITS,
            scope,
        )
        outputs = yield (task,)
        logits = _token_logits(outputs[0])[-1]
        self._commit_task_kv(task, 1, scope)
        if self.defer_sampling:
            published = self._publish_logits(
                envelope,
                logits,
                SequenceMode.DECODE,
                scope,
            )
            return SequenceDelta(
                SequenceEffect(
                    kv_tokens=self.kv.get(envelope.session_id).length,
                    published_logits=published,
                    published_kv=self._publish_kv_if_requested(
                        envelope,
                        operation,
                        (),
                        scope,
                    ),
                )
            )
        sample_task = self._sample_task(
            envelope,
            logits,
            session,
            operation.policy,
            positions=(start + 1,),
        )
        sampled = _sample_result((yield (sample_task,))[0])
        session.rng_counter += 1
        session.last_sampled_token = _sample_relay_token(sampled)
        token_ids = cast(tuple[int, ...], (sampled.token_id,))
        return SequenceDelta(
            SequenceEffect(
                sampled_token_ids=token_ids,
                sampled_logprob=sampled.logprob,
                top_logprobs=_token_logprobs(sampled.top_logprobs),
                kv_tokens=self.kv.get(envelope.session_id).length,
                published_kv=self._publish_kv_if_requested(
                    envelope,
                    operation,
                    token_ids,
                    scope,
                ),
            )
        )

    def _verify(
        self,
        envelope: OperationEnvelope,
        operation: SequenceOperation,
        inputs: WireTokenInput,
        session: RequestSession,
        scope: _ExecutionScope,
    ) -> _Driver:
        current = self._resolve_decode_token(inputs, session)
        draft = tuple(int(value) for value in inputs.draft_token_ids)
        start, end = operation.position
        if end - start != 1:
            raise invalid_descriptor("sequence verify must start from one current token position")
        tokens = (current, *draft)
        task = self._token_task(
            envelope,
            tokens,
            tuple(range(start, start + len(tokens))),
            TokenSelection.ALL_LOGITS,
            scope,
        )
        outputs = yield (task,)
        logits = _token_logits(outputs[0])
        sampling = _require_sampling(session)
        seed = int(sampling.seed or 0)
        coins = uniform_samples(
            (len(draft),),
            seed=sampling_draw_seed(seed, start),
            device=logits.device,
        )
        sample_task = self._sample_task(
            envelope,
            logits,
            session,
            operation.policy,
            positions=tuple(range(start + 1, start + len(draft) + 2)),
            draft_token_ids=draft,
            acceptance_uniforms=coins,
        )
        sampled = _sample_result((yield (sample_task,))[0])
        committed = 1 + int(sampled.num_accepted_tokens)
        self._commit_task_kv(task, committed, scope)
        sampled_ids = (
            *draft[: sampled.num_accepted_tokens],
            int(sampled.token_id),
        )
        session.last_sampled_token = _sample_relay_token(sampled)
        session.rng_counter += len(draft) + 1
        return SequenceDelta(
            SequenceEffect(
                sampled_token_ids=tuple(sampled_ids),
                sampled_logprob=sampled.logprob,
                top_logprobs=_token_logprobs(sampled.top_logprobs),
                accepted_draft_tokens=int(sampled.num_accepted_tokens),
                kv_tokens=self.kv.get(envelope.session_id).length,
                published_kv=self._publish_kv_if_requested(
                    envelope,
                    operation,
                    tuple(sampled_ids),
                    scope,
                ),
            )
        )

    def _token_task(
        self,
        envelope: OperationEnvelope,
        token_ids: tuple[int | torch.Tensor, ...],
        positions: tuple[int, ...],
        selection: TokenSelection,
        scope: _ExecutionScope,
    ) -> _ForwardTask:
        stage = self._primary_stage(envelope.operation_type)
        if stage.row is not RouteRowKind.TOKEN:
            raise invalid_descriptor("sequence operation primary stage is not a token row")
        if len(token_ids) != len(positions) or not token_ids:
            raise invalid_descriptor("token task ids and positions must align")
        if len(token_ids) == 1 and isinstance(token_ids[0], torch.Tensor):
            token_values = token_ids[0].reshape(1).to(dtype=torch.long)
        else:
            token_values = torch.tensor(
                tuple(int(value) for value in token_ids),
                dtype=torch.long,
            )
        row_id = scope.row_id()
        row = TokenRow(
            row_id=row_id,
            inputs=TokenIds(token_values),
            positions=torch.tensor(positions, dtype=torch.long),
            output_slot=row_id,
            selection=selection,
        )
        return _ForwardTask(
            envelope=envelope,
            stage=stage,
            route=self._route(stage),
            row=row,
            entry=self.kv.get(envelope.session_id),
            write_kv=True,
            causal=True,
        )

    def _commit_task_kv(
        self,
        task: _ForwardTask,
        tokens: int,
        scope: _ExecutionScope,
    ) -> None:
        count = int(tokens)
        if count < 0 or count > task.query_tokens:
            raise RuntimeError("KV commit count is outside the task query span")
        if count == 0:
            return
        if task.scratch:
            scope.kv.advance_entry(cast(KvEntry, task.entry), count)
        else:
            scope.kv.advance(task.envelope.session_id, count)

    def _resolve_decode_token(
        self,
        inputs: WireTokenInput,
        session: RequestSession,
    ) -> int | torch.Tensor:
        if inputs.source is TokenSource.WIRE:
            return int(inputs.token_ids[0])
        if session.last_sampled_token is None:
            raise invalid_descriptor("last-sampled token source has no committed token")
        if isinstance(session.last_sampled_token, SampledTokenRelay):
            return session.last_sampled_token.tensor
        return int(session.last_sampled_token)

    def _sample_task(
        self,
        envelope: OperationEnvelope,
        logits: torch.Tensor,
        session: RequestSession,
        policy: TokenPolicy,
        *,
        positions: tuple[int, ...],
        draft_token_ids: tuple[int, ...] = (),
        acceptance_uniforms: torch.Tensor | None = None,
    ) -> _SampleTask:
        sampling = _require_sampling(session)
        rows = logits.reshape(1, -1) if logits.ndim == 1 else logits
        if rows.ndim != 2 or int(rows.shape[0]) != len(positions):
            raise invalid_descriptor("sampling task positions do not align with its logits")
        allowed = policy.allowed_tokens or sampling.allowed_token_ids
        recent_counts = _recent_token_counts(policy.recent_tokens)
        descriptors = tuple(
            _SamplingRow(
                parameters=sampling,
                recent_counts=recent_counts,
                allowed=None if allowed is None else tuple(int(value) for value in allowed),
                suppress=tuple(int(value) for value in policy.suppress_tokens),
                draw_seed=sampling_draw_seed(
                    int(sampling.seed or 0),
                    int(position),
                ),
                n_logprobs=int(sampling.n_logprobs),
            )
            for position in positions
        )
        plain_greedy = not draft_token_ids and all(_plain_greedy_row(row) for row in descriptors)
        defer_host_token = (
            plain_greedy and not policy.publish_kv and not policy.publish_kv_on_tokens
        )
        if plain_greedy:
            noise = None
            penalty_token_ids = None
            penalty_counts = None
            parameter_values = None
        else:
            noise = _semantic_sampling_noise(
                descriptors,
                vocab=int(rows.shape[1]),
                device=rows.device,
            )
            penalty_token_ids, penalty_counts, parameter_values = _sampling_task_tensors(
                descriptors,
                vocab=int(rows.shape[1]),
                device=rows.device,
            )
        return _SampleTask(
            envelope=envelope,
            logits=rows,
            rows=descriptors,
            noise=noise,
            penalty_token_ids=penalty_token_ids,
            penalty_counts=penalty_counts,
            parameter_values=parameter_values,
            defer_host_token=defer_host_token,
            draft_token_ids=tuple(int(value) for value in draft_token_ids),
            acceptance_uniforms=acceptance_uniforms,
        )

    def _prompt_logprobs(
        self,
        session: RequestSession,
        tokens: tuple[int, ...],
        logits: torch.Tensor,
        scope: _ExecutionScope,
        *,
        enabled: bool,
    ) -> tuple[tuple[TokenLogprob, ...], ...]:
        if logits.ndim != 2 or int(logits.shape[0]) not in {1, len(tokens)}:
            raise invalid_descriptor("extend logits do not align with the prompt chunk")
        result: tuple[tuple[TokenLogprob, ...], ...] = ()
        prior = None
        if session.prompt_logits_handle is not None:
            record = scope.product_view.get(session.prompt_logits_handle)
            if record is not None and isinstance(record.payload, LogitsProduct):
                prior = record.payload.logits.reshape(1, -1)
        if enabled:
            if int(logits.shape[0]) != len(tokens):
                raise invalid_descriptor("prompt logprobs require all prompt logits")
            if prior is None:
                score_logits = logits[:-1]
                targets = tokens[1:]
            else:
                score_logits = torch.cat((prior.to(logits.device), logits[:-1]), dim=0)
                targets = tokens
            sampling = _require_sampling(session)
            scored = _score_prompt_token_logprobs(
                score_logits,
                targets,
                n_logprobs=int(sampling.n_prompt_logprobs),
                logprob_token_ids=sampling.logprob_token_ids,
            )
            result = tuple(tuple(TokenLogprob(*entry) for entry in position) for position in scored)
        handle = _stable_handle(session.session_id, session.epoch, 0, "prompt-logits")
        scope.product_view.put(
            ProductRecord(
                handle=handle,
                session_id=session.session_id,
                payload=LogitsProduct(logits[-1:].detach(), SequenceMode.EXTEND),
            )
        )
        session.prompt_logits_handle = handle
        return result

    def _publish_logits(
        self,
        envelope: OperationEnvelope,
        logits: torch.Tensor,
        source_mode: SequenceMode,
        scope: _ExecutionScope,
    ) -> PublishedProduct:
        if self.transport is None:
            raise capability_mismatch("deferred sampling requires a configured transport")
        handle = _stable_handle(envelope.session_id, envelope.epoch, envelope.op_id, "logits")
        value = logits.detach().contiguous()
        locator = self.transport.publish(value)
        scope.published.append(locator)
        encoded = locator.to_wire_json()
        scope.product_view.put(
            ProductRecord(
                handle=handle,
                session_id=envelope.session_id,
                payload=LogitsProduct(value, source_mode),
                locator=encoded,
            )
        )
        self.sessions.get(envelope.session_id).product_handles.add(handle)
        return PublishedProduct(handle=handle, locator=encoded)

    def _fetch_logits(
        self,
        source: PublishedProduct,
        scope: _ExecutionScope,
    ) -> torch.Tensor:
        if source.handle > 0:
            record = scope.product_view.get(source.handle)
            if record is not None:
                if not isinstance(record.payload, LogitsProduct):
                    raise invalid_descriptor("published sequence product is not logits")
                return record.payload.logits
        if not source.locator or self.transport is None:
            raise invalid_descriptor("published logits are not resident or transport-addressable")
        value = fetch_locator(self.transport, Locator.from_wire_json(source.locator))
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise invalid_descriptor("published logits transport returned an invalid tensor")
        return value

    def _publish_kv_if_requested(
        self,
        envelope: OperationEnvelope,
        operation: SequenceOperation,
        sampled: tuple[int, ...],
        scope: _ExecutionScope,
    ) -> Any:
        policy = operation.policy
        if not policy.publish_kv:
            triggers = policy.publish_kv_on_tokens
            if not triggers or not (set(sampled) & set(triggers)):
                return None
        published = self.kv.publish(
            envelope.session_id,
            source_version=envelope.base_version + 1,
            position=operation.position[1],
            transport=self.transport,
        )
        for encoded in published.locators:
            scope.published.append(Locator.from_wire_json(encoded))
        return published

    def _flow_driver(
        self,
        envelope: OperationEnvelope,
        operation: FlowOperation,
        scope: _ExecutionScope,
    ) -> _Driver:
        spec = cast(ModelSpec, self.spec)
        flow = spec.flow
        if flow is None:
            raise invalid_descriptor("flow operation requires a declared FlowSpec")
        session = self.sessions.get(envelope.session_id)
        image = session.image
        if image is None:
            raise invalid_descriptor("flow operation has no admitted image parameters")
        if operation.start_step + operation.step_count > image.steps:
            raise invalid_descriptor("flow operation exceeds the declared schedule")
        record = scope.latents.read(operation.latent_handle)
        if record is None:
            if operation.start_step != 0:
                raise invalid_descriptor("flow continuation references a missing latent")
            value = self._initial_latent(envelope, image.height, image.width, session)
            record = LatentRecord(
                handle=operation.latent_handle,
                session_id=envelope.session_id,
                value=value,
                step=0,
                height=image.height,
                width=image.width,
            )
            scope.latents.write(record)
            session.latent_handle = operation.latent_handle
        if record.session_id != envelope.session_id:
            raise invalid_descriptor("flow latent belongs to another session")
        if record.step != operation.start_step or session.flow_step != operation.start_step:
            raise invalid_descriptor("flow operation start step does not match committed state")

        schedule = FlowMatchSchedule(
            num_steps=int(image.steps),
            shift=float(
                image.timestep_shift if image.timestep_shift > 0 else flow.timestep_shift or 1.0
            ),
            direction=ScheduleDirection(flow.schedule_direction),
            shift_domain=ScheduleShiftDomain(flow.schedule_shift_domain),
        )
        current = record.value
        for step in range(operation.start_step, operation.start_step + operation.step_count):
            t, t_next = schedule.pair(
                step,
                device=current.device,
                dtype=torch.float32,
            )
            use_cfg = (
                float(operation.guidance.interval[0])
                <= float(t.item())
                <= float(operation.guidance.interval[1])
            )
            guide = build_flow_cfg_plan(
                cfg_text_scale=float(operation.guidance.text_scale),
                cfg_img_scale=float(operation.guidance.image_scale),
                recipe=CfgRecipe.coerce(flow.cfg_recipe),
                renorm=operation.guidance.renorm_type,
                renorm_min=float(operation.guidance.renorm_min),
                use_cfg=use_cfg,
            )
            if len(guide.branches) > int(operation.guidance.branch_count):
                raise invalid_descriptor(
                    "flow guidance branch bound is smaller than the declared CFG plan"
                )
            if len(guide.branches) > int(flow.max_cfg_branches):
                raise invalid_descriptor("flow CFG plan exceeds the model branch bound")

            entries: dict[Branch, KvEntry] = {}
            prefix_tasks: list[_ForwardTask] = []
            for branch in guide.branches:
                source = self._branch_source(branch)
                prefix, copy_conditioning = self._flow_prefix(
                    source,
                    operation,
                    session,
                )
                query = self._flow_physical_tokens(record.height, record.width)
                entry, created = scope.kv.scratch_entry(
                    envelope.session_id,
                    operation.latent_handle,
                    branch.value,
                    capacity_tokens=(
                        self.kv.get(envelope.session_id).length
                        if copy_conditioning
                        else len(prefix)
                    )
                    + query,
                    copy_conditioning=copy_conditioning,
                )
                entries[branch] = entry
                if created and prefix:
                    prefix_tasks.append(
                        self._flow_prefix_task(
                            envelope,
                            prefix,
                            entry,
                            branch,
                            scope,
                        )
                    )
            if prefix_tasks:
                prefix_outputs = yield tuple(prefix_tasks)
                for task, output in zip(prefix_tasks, prefix_outputs, strict=True):
                    _token_logits_or_hidden(output)
                    self._commit_task_kv(task, task.query_tokens, scope)

            tasks = tuple(
                self._flow_task(
                    envelope,
                    operation,
                    branch,
                    entries[branch],
                    current,
                    t,
                    record.height,
                    record.width,
                    scope,
                )
                for branch in guide.branches
            )
            outputs = yield tasks
            predictions = {
                branch: _flow_prediction(output)
                for branch, output in zip(guide.branches, outputs, strict=True)
            }
            velocity = guide.combine(predictions)
            if flow.prediction in {"x", "x_prediction", "x_pred"}:
                neural_latent = self._flow_neural_latent(current, record.height, record.width)
                velocity = x_pred_to_velocity(velocity, neural_latent, t)
            elif flow.prediction != "velocity":
                raise invalid_descriptor(f"unsupported flow prediction {flow.prediction!r}")
            neural_current = self._flow_neural_latent(current, record.height, record.width)
            updated = euler_step(neural_current, velocity, t, t_next)
            current = self._flow_store_latent(updated, record.height, record.width)
            record = replace(record, value=current, step=step + 1)
            scope.latents.write(record)
            session.flow_step = step + 1
        return FlowDelta(
            steps_completed=record.step,
            done=record.step == image.steps,
        )

    def _initial_latent(
        self,
        envelope: OperationEnvelope,
        height: int,
        width: int,
        session: RequestSession,
    ) -> torch.Tensor:
        flow = cast(ModelSpec, self.spec).flow
        assert flow is not None
        route = self._route(self._primary_stage(OperationType.FLOW))
        device = torch.device(self._route_device(route))
        dtype = _torch_dtype(route.dtype)
        seed = flow_noise_seed(int(_require_image(session).seed or 0), envelope.op_id)
        scale = self._noise_scale(height, width)
        shape: tuple[int, ...]
        if flow.latent_layout is LatentLayout.PATCH_TOKENS:
            token_height = height // int(flow.latent_downsample)
            token_width = width // int(flow.latent_downsample)
            feature_width = int(flow.latent_patch_size) ** 2 * int(flow.latent_channels)
            shape = (token_height * token_width, feature_width)
        else:
            shape = (1, int(flow.latent_channels), int(height), int(width))
        return normal_noise(shape, seed=seed, device=device, dtype=dtype) * float(scale)

    def _noise_scale(self, height: int, width: int) -> float:
        flow = cast(ModelSpec, self.spec).flow
        assert flow is not None
        declared = flow.noise_scale
        image_tokens = (height // int(flow.latent_downsample)) * (
            width // int(flow.latent_downsample)
        )
        value = float(declared.value)
        if declared.mode in {
            NoiseScaleMode.RESOLUTION,
            NoiseScaleMode.DYNAMIC,
            NoiseScaleMode.DYNAMIC_SQRT,
        }:
            value *= math.sqrt(float(image_tokens) / float(declared.base_image_tokens))
        if declared.mode is NoiseScaleMode.DYNAMIC_SQRT:
            value = math.sqrt(value)
        return min(value, float(declared.maximum))

    def _branch_source(self, branch: Branch) -> FlowBranchSource:
        flow = cast(ModelSpec, self.spec).flow
        assert flow is not None
        if branch is Branch.COND:
            return FlowBranchSource.CONDITIONING
        if branch is Branch.TEXT_UNCOND:
            return flow.text_unconditional
        return flow.image_unconditional

    def _flow_prefix(
        self,
        source: FlowBranchSource,
        operation: FlowOperation,
        session: RequestSession,
    ) -> tuple[tuple[int, ...], bool]:
        if source is FlowBranchSource.CONDITIONING and not operation.image_prompt.strip():
            return (), True
        if source is FlowBranchSource.NEGATIVE_OR_START and session.negative_token_ids:
            return session.negative_token_ids, False
        prompt = cast(ModelSpec, self.spec).inputs.flow_prompt
        if prompt is None:
            if source is FlowBranchSource.CONDITIONING:
                raise invalid_descriptor(
                    "flow image-prompt override requires declared prompt framing"
                )
            return (), False
        if self.tokenizer is None:
            raise capability_mismatch("declared flow prompt framing requires a tokenizer")
        if source is FlowBranchSource.CONDITIONING:
            text = operation.image_prompt.strip()
            append = prompt.conditioned_append
        elif source is FlowBranchSource.NEGATIVE_OR_START:
            text = _require_image(session).negative_prompt.strip()
            append = prompt.unconditional_append
        else:
            text = ""
            append = prompt.unconditional_append
        framed = (
            prompt.system_prefix
            + prompt.system_message
            + prompt.system_suffix
            + prompt.user_prefix
            + text
            + prompt.user_suffix
            + prompt.assistant_suffix
            + append
        )
        encoded = self.tokenizer.encode(
            framed,
            add_special_tokens=prompt.add_special_tokens,
        )
        return tuple(int(value) for value in encoded), False

    def _flow_prefix_task(
        self,
        envelope: OperationEnvelope,
        tokens: tuple[int, ...],
        entry: KvEntry,
        branch: Branch,
        scope: _ExecutionScope,
    ) -> _ForwardTask:
        primary = self._primary_stage(OperationType.FLOW)
        route = self._route(primary)
        if RouteRowKind.TOKEN not in route.row_kinds:
            raise invalid_descriptor("flow prefix route does not accept token rows")
        stage = OperationStageSpec(route.name, RouteRowKind.TOKEN)
        row_id = scope.row_id()
        positions = torch.arange(entry.length, entry.length + len(tokens), dtype=torch.long)
        row = TokenRow(
            row_id=row_id,
            inputs=TokenIds(torch.tensor(tokens, dtype=torch.long)),
            positions=positions,
            output_slot=row_id,
            selection=TokenSelection.HIDDEN,
        )
        return _ForwardTask(
            envelope=envelope,
            stage=stage,
            route=route,
            row=row,
            entry=entry,
            scratch=True,
            write_kv=True,
            causal=True,
            attention_indexes=torch.stack(
                (positions, torch.zeros_like(positions), torch.zeros_like(positions))
            ),
        )

    def _flow_task(
        self,
        envelope: OperationEnvelope,
        operation: FlowOperation,
        branch: Branch,
        entry: KvEntry,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        height: int,
        width: int,
        scope: _ExecutionScope,
    ) -> _ForwardTask:
        flow = cast(ModelSpec, self.spec).flow
        assert flow is not None
        stage = self._primary_stage(OperationType.FLOW)
        if stage.row is not RouteRowKind.FLOW:
            raise invalid_descriptor("flow operation primary stage is not a flow row")
        row_id = scope.row_id()
        neural_latent = self._flow_neural_latent(latent, height, width)
        image_tokens = self._flow_query_tokens(latent, height, width)
        text_local: tuple[int, ...]
        if flow.latent_layout is LatentLayout.PATCH_TOKENS:
            latent_positions = get_flattened_position_ids_extrapolate(
                height,
                width,
                int(flow.latent_downsample),
                int(math.isqrt(flow.max_latent_tokens)),
            )
            conditioning: Any = NoFlowConditioning()
            query_tokens = image_tokens + int(flow.commit_marker_tokens)
            temporal = self._flow_temporal_position(branch, operation, entry)
            attention_indexes = torch.stack(
                (
                    torch.full((query_tokens,), temporal, dtype=torch.long),
                    torch.zeros(query_tokens, dtype=torch.long),
                    torch.zeros(query_tokens, dtype=torch.long),
                )
            )
            text_local = (0, query_tokens - 1)
        else:
            latent_positions = self._flow_spatial_positions(
                height,
                width,
                int(flow.latent_patch_size),
                self._flow_temporal_position(branch, operation, entry),
            )
            conditioning = self._flow_conditioning(latent, height, width)
            query_tokens = image_tokens
            attention_indexes = latent_positions
            text_local = ()
        row = FlowRow(
            row_id=row_id,
            conditioning=conditioning,
            positions=latent_positions,
            timestep=timestep.reshape(1),
            latent=neural_latent,
            image_tokens=query_tokens,
            image_height=height,
            image_width=width,
            output_slot=row_id,
        )
        return _ForwardTask(
            envelope=envelope,
            stage=stage,
            route=self._route(stage),
            row=row,
            entry=entry,
            scratch=True,
            write_kv=True,
            causal=False,
            attention_indexes=attention_indexes,
            text_local_indices=text_local,
        )

    @staticmethod
    def _flow_temporal_position(
        branch: Branch,
        operation: FlowOperation,
        entry: KvEntry,
    ) -> int:
        if branch is Branch.COND:
            return int(operation.conditioning_position)
        return int(entry.length)

    def _flow_query_tokens(self, latent: torch.Tensor, height: int, width: int) -> int:
        del latent
        flow = cast(ModelSpec, self.spec).flow
        assert flow is not None
        return (height // int(flow.latent_downsample)) * (width // int(flow.latent_downsample))

    def _flow_physical_tokens(self, height: int, width: int) -> int:
        flow = cast(ModelSpec, self.spec).flow
        assert flow is not None
        image_tokens = (height // int(flow.latent_downsample)) * (
            width // int(flow.latent_downsample)
        )
        if flow.latent_layout is LatentLayout.PATCH_TOKENS:
            return image_tokens + int(flow.commit_marker_tokens)
        return image_tokens

    def _flow_neural_latent(
        self,
        latent: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        flow = cast(ModelSpec, self.spec).flow
        assert flow is not None
        if flow.latent_layout is LatentLayout.PATCH_TOKENS:
            return latent
        return patchify_batch(latent, int(flow.latent_patch_size))

    def _flow_store_latent(
        self,
        latent: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        flow = cast(ModelSpec, self.spec).flow
        assert flow is not None
        if flow.latent_layout is LatentLayout.PATCH_TOKENS:
            return latent
        return unpatchify_batch(
            latent,
            int(flow.latent_patch_size),
            height=height,
            width=width,
            channels=int(flow.latent_channels),
        )

    def _flow_conditioning(
        self,
        latent: torch.Tensor,
        height: int,
        width: int,
    ) -> FlowPatches:
        flow = cast(ModelSpec, self.spec).flow
        assert flow is not None
        if flow.conditioning is not FlowConditioningKind.IMAGE_PATCHES:
            raise invalid_descriptor(
                "image latent layout requires declared image-patch conditioning"
            )
        images = cast(ModelSpec, self.spec).inputs.images
        transform = None if images is None else images.vit
        if not isinstance(transform, ImagePatchSpec):
            raise invalid_descriptor("flow image patches require a declared patch transform")
        patch = int(transform.patch_size)
        pixels = patchify_batch(latent, patch, channel_first=True).reshape(
            -1,
            patch * patch * int(latent.shape[1]),
        )
        grid = torch.tensor(
            [[height // patch, width // patch]],
            dtype=torch.long,
            device=latent.device,
        )
        return FlowPatches(
            pixels=pixels,
            grid=grid,
            noise_scale=latent.new_tensor([self._noise_scale(height, width)]),
        )

    @staticmethod
    def _flow_spatial_positions(
        height: int,
        width: int,
        patch: int,
        temporal: int,
    ) -> torch.Tensor:
        grid_height = height // patch
        grid_width = width // patch
        y = torch.arange(grid_height, dtype=torch.long).repeat_interleave(grid_width)
        x = torch.arange(grid_width, dtype=torch.long).repeat(grid_height)
        return torch.stack((torch.full_like(x, int(temporal)), y, x))

    def _encode_driver(
        self,
        envelope: OperationEnvelope,
        operation: EncodeOperation,
        scope: _ExecutionScope,
    ) -> _Driver:
        spec = cast(ModelSpec, self.spec)
        image_spec = spec.inputs.images
        if image_spec is None:
            raise invalid_descriptor("encode operation requires declared image transforms")
        handle = encoder_handle_from_content_hash(operation.input.content_hash)
        expected = (
            VisionFeatureProduct if operation.kind is EncodeKind.VISION else LatentFeatureProduct
        )
        record = scope.product_view.get(handle)
        cached = isinstance(operation.input, CachedProduct)
        if cached:
            if record is None or not isinstance(record.payload, expected):
                raise invalid_descriptor(
                    "cached encoder product is not resident with the requested kind"
                )
            if record.content_hash != operation.input.content_hash:
                raise invalid_descriptor("cached encoder product identity does not match its input")
            payload = record.payload
        else:
            source = self._encode_source(operation.input, scope)
            stage = self._primary_stage(envelope.operation_type)
            if stage.row is not RouteRowKind.ENCODE:
                raise invalid_descriptor("encode primary stage is not an encode row")
            prepared = prepare_image(
                image_spec,
                operation.kind,
                source,
                device=torch.device(self._route_device(self._route(stage))),
            )
            task = self._encode_task(envelope, operation.kind, prepared, stage, scope)
            outputs = yield (task,)
            features = _encode_features(outputs[0]).detach()
            if operation.kind is EncodeKind.VISION:
                grid = (
                    prepared.inputs.grid.detach()
                    if isinstance(prepared.inputs, PatchInput)
                    else None
                )
                payload = VisionFeatureProduct(
                    features=features,
                    grid=grid,
                    height=prepared.height,
                    width=prepared.width,
                    source_base64=source,
                )
            else:
                payload = LatentFeatureProduct(
                    latent=features,
                    height=prepared.height,
                    width=prepared.width,
                    source_base64=source,
                )
            record = ProductRecord(
                handle=handle,
                session_id=envelope.session_id,
                payload=payload,
                content_hash=operation.input.content_hash,
            )
            scope.product_view.put(record)
        session = self.sessions.get(envelope.session_id)
        session.product_handles.add(handle)
        if isinstance(payload, VisionFeatureProduct):
            outcome = yield from self._state_driver(
                envelope,
                envelope.operation_type,
                scope,
                height=payload.height,
                width=payload.width,
                conditioning_position=operation.conditioning_position,
                features=payload.features,
                grid=payload.grid,
                retain_image=True,
            )
            image_size = (payload.height, payload.width)
        else:
            outcome = yield from self._state_driver(
                envelope,
                envelope.operation_type,
                scope,
                height=payload.height,
                width=payload.width,
                conditioning_position=operation.conditioning_position,
                latent=payload.latent,
                retain_image=True,
            )
            image_size = (payload.height, payload.width)
        return EncodeDelta(
            product_handle=handle,
            kv_tokens=outcome.kv_tokens,
            image_size=image_size,
        )

    def _materialize_driver(
        self,
        envelope: OperationEnvelope,
        operation: MaterializeOperation,
        scope: _ExecutionScope,
    ) -> _Driver:
        if operation.kind is MaterializeKind.FRAME:
            return self._materialize_frames(envelope, operation, scope)
        spec = cast(ModelSpec, self.spec)
        flow = spec.flow
        if flow is None:
            raise invalid_descriptor("image materialization requires a declared FlowSpec")
        if not isinstance(operation.input, LatentProduct):
            raise invalid_descriptor("image materialization requires a committed latent")
        latent_record = scope.latents.read(operation.input.handle)
        if latent_record is None or latent_record.session_id != envelope.session_id:
            raise invalid_descriptor("materialization latent is not resident for this session")
        session = self.sessions.get(envelope.session_id)
        image_params = session.image
        if image_params is None:
            raise invalid_descriptor("image materialization has no admitted image parameters")
        if latent_record.step != image_params.steps:
            raise invalid_descriptor("image materialization requires a completed latent trajectory")

        if flow.materialization is MaterializationKind.DECODE_ROUTE:
            stage = self._primary_stage(envelope.operation_type)
            if stage.row is not RouteRowKind.DECODE:
                raise invalid_descriptor("decode materialization requires a decode primary stage")
            row_id = scope.row_id()
            task = _ForwardTask(
                envelope=envelope,
                stage=stage,
                route=self._route(stage),
                row=DecodeRow(
                    row_id=row_id,
                    latent=latent_record.value,
                    image_height=latent_record.height,
                    image_width=latent_record.width,
                    output_slot=row_id,
                ),
            )
            outputs = yield (task,)
            image_tensor = _decoded_tensor(outputs[0]).detach()
            image_range = ImageRange.UNIT
        elif flow.materialization is MaterializationKind.RGB_LATENT:
            if any(
                stage.purpose is OperationStagePurpose.PRIMARY
                for stage in self._stages(envelope.operation_type)
            ):
                raise invalid_descriptor(
                    "RGB-latent materialization must not declare a decode route"
                )
            image_tensor = latent_record.value.detach()
            image_range = ImageRange.SIGNED_UNIT
        else:
            raise invalid_descriptor("model declares an unknown image materialization kind")

        png = tensor_to_png_b64(
            image_tensor,
            value_range=(-1.0, 1.0) if image_range is ImageRange.SIGNED_UNIT else (0.0, 1.0),
        )
        image_handle = _stable_handle(envelope.session_id, envelope.epoch, envelope.op_id, "image")
        locator_text = ""
        if (
            ModelLoadScope(cast(DeploymentOverlay, self.deployment).model_scope)
            is ModelLoadScope.GENERATION
        ):
            source = (
                latent_record.value
                if flow.latent_layout is LatentLayout.PATCH_TOKENS
                else image_tensor
            )
            locator_text = self._publish_tensor(
                source,
                scope,
                payload_kind=flow.latent_layout.value,
                height=latent_record.height,
                width=latent_record.width,
                value_range=image_range.value,
            )
            outcome = _StateOutcome(0)
        else:
            outcome = yield from self._state_driver(
                envelope,
                envelope.operation_type,
                scope,
                height=latent_record.height,
                width=latent_record.width,
                conditioning_position=operation.conditioning_position,
                image=image_tensor,
                image_range=image_range,
                latent=latent_record.value,
                policy=operation.policy,
                close_image=True,
                retain_image=image_params.retain_images,
            )

        scope.product_view.put(
            ProductRecord(
                handle=image_handle,
                session_id=envelope.session_id,
                payload=ImageTensorProduct(
                    image=image_tensor,
                    height=latent_record.height,
                    width=latent_record.width,
                    value_range=image_range,
                ),
                locator=locator_text,
            )
        )
        session.product_handles.add(image_handle)
        self._append_frame(envelope, png, scope)
        scope.latents.delete(operation.input.handle)
        scope.kv.release_generation(envelope.session_id, operation.input.handle)
        session.latent_handle = None
        session.flow_step = 0
        return MaterializeDelta(
            product=ImageArtifact(
                png_base64=png,
                height=latent_record.height,
                width=latent_record.width,
                handle=image_handle,
                locator=locator_text,
            ),
            kv_tokens=(outcome.kv_tokens if image_params.retain_images else None),
            sequence=outcome.sequence,
        )

    def _transfer_driver(
        self,
        envelope: OperationEnvelope,
        operation: TransferOperation,
        scope: _ExecutionScope,
    ) -> _Driver:
        if self.transport is None:
            raise capability_mismatch("product transfer requires a configured transport")
        value, metadata = self._fetch_product_tensor(operation.source, scope)
        if operation.kind is TransferKind.PRODUCT:
            handle = operation.source.handle or _stable_handle(
                envelope.session_id, envelope.epoch, envelope.op_id, "transfer"
            )
            locator = self._publish_tensor(
                value,
                scope,
                payload_kind=_metadata_string(metadata, "payload_kind", "tensor"),
                height=_metadata_uint(metadata, "height", 0),
                width=_metadata_uint(metadata, "width", 0),
                value_range=_metadata_string(metadata, "value_range", ""),
            )
            return TransferDelta(product=PublishedProduct(handle=handle, locator=locator))

        spec = cast(ModelSpec, self.spec)
        flow = spec.flow
        if flow is None:
            raise invalid_descriptor("KV transfer requires a declared FlowSpec")
        session = self.sessions.get(envelope.session_id)
        image = session.image
        height = _metadata_uint(metadata, "height", 0) or (0 if image is None else image.height)
        width = _metadata_uint(metadata, "width", 0) or (0 if image is None else image.width)
        if min(height, width) < 1:
            raise invalid_descriptor("transferred image product has no declared geometry")
        outcome = yield from self._state_driver(
            envelope,
            envelope.operation_type,
            scope,
            height=height,
            width=width,
            conditioning_position=operation.conditioning_position,
            image=value if flow.latent_layout is LatentLayout.IMAGE_NCHW else None,
            image_range=ImageRange(
                _metadata_string(metadata, "value_range", ImageRange.SIGNED_UNIT.value)
            ),
            latent=value if flow.latent_layout is LatentLayout.PATCH_TOKENS else None,
            policy=operation.policy,
            close_image=True,
            retain_image=True,
        )
        return TransferDelta(kv_tokens=outcome.kv_tokens, sequence=outcome.sequence)

    def _encode_source(
        self,
        source: InlineImage | StagedProduct | CachedProduct,
        scope: _ExecutionScope,
    ) -> str:
        if isinstance(source, InlineImage):
            return source.base64
        if isinstance(source, CachedProduct):
            raise invalid_descriptor("cached encoder input has no source image")
        record = scope.product_view.get(source.handle)
        if record is None:
            raise invalid_descriptor("staged encoder input handle is not resident")
        payload = record.payload
        if isinstance(payload, (VisionFeatureProduct, LatentFeatureProduct)):
            return payload.source_base64
        if isinstance(payload, EncodedImageProduct):
            return payload.base64
        raise invalid_descriptor("staged encoder input is not an encoded-image product")

    def _encode_task(
        self,
        envelope: OperationEnvelope,
        kind: EncodeKind,
        prepared: PreparedImage,
        stage: OperationStageSpec,
        scope: _ExecutionScope,
    ) -> _ForwardTask:
        row_id = scope.row_id()
        row = EncodeRow(
            row_id=row_id,
            kind=(
                ForwardEncodeKind.VISION if kind is EncodeKind.VISION else ForwardEncodeKind.LATENT
            ),
            inputs=prepared.inputs,
            output_slot=row_id,
        )
        return _ForwardTask(envelope=envelope, stage=stage, route=self._route(stage), row=row)

    def _state_driver(
        self,
        envelope: OperationEnvelope,
        operation_type: OperationType,
        scope: _ExecutionScope,
        *,
        height: int,
        width: int,
        conditioning_position: int,
        features: torch.Tensor | None = None,
        grid: torch.Tensor | None = None,
        image: torch.Tensor | None = None,
        image_range: ImageRange = ImageRange.SIGNED_UNIT,
        latent: torch.Tensor | None = None,
        policy: TokenPolicy | None = None,
        close_image: bool = False,
        retain_image: bool,
    ) -> Generator[tuple[_ExecutorTask, ...], _TaskResult, _StateOutcome]:
        total = 0
        sequence: SequenceEffect | None = None
        for stage in self._state_stages(operation_type, retain_image=retain_image):
            if stage.row is RouteRowKind.ENCODE:
                if image is None:
                    raise invalid_descriptor("image state encode stage has no image tensor")
                images = cast(ModelSpec, self.spec).inputs.images
                if images is None:
                    raise invalid_descriptor("image state encode stage has no declared transform")
                prepared = prepare_tensor_image(
                    images,
                    EncodeKind.VISION,
                    image,
                    device=torch.device(self._route_device(self._route(stage))),
                    signed_unit=image_range is ImageRange.SIGNED_UNIT,
                )
                task = self._encode_task(envelope, EncodeKind.VISION, prepared, stage, scope)
                outputs = yield (task,)
                features = _encode_features(outputs[0]).detach()
                grid = prepared.inputs.grid if isinstance(prepared.inputs, PatchInput) else None
                continue
            if stage.row is RouteRowKind.TOKEN:
                if features is None:
                    raise invalid_descriptor("token state stage has no vision features")
                task = self._vision_state_task(
                    envelope,
                    stage,
                    features,
                    grid,
                    conditioning_position,
                    scope,
                    close_image=close_image,
                    logits=policy is not None,
                )
                outputs = yield (task,)
                value = _token_logits_or_hidden(outputs[0])
                self._commit_task_kv(task, task.query_tokens, scope)
                total += task.query_tokens
                if policy is not None:
                    if not isinstance(outputs[0], TokenOutput) or not isinstance(
                        outputs[0].value, TokenLogits
                    ):
                        raise invalid_descriptor("image state token stage did not return logits")
                    session = self.sessions.get(envelope.session_id)
                    flow_spec = cast(ModelSpec, self.spec).flow
                    sample_task = self._sample_task(
                        envelope,
                        value[-1],
                        session,
                        policy,
                        positions=(
                            conditioning_position
                            + max(
                                1,
                                1 if flow_spec is None else flow_spec.rope_advance,
                            ),
                        ),
                    )
                    sampled = _sample_result((yield (sample_task,))[0])
                    session.last_sampled_token = _sample_relay_token(sampled)
                    session.rng_counter += 1
                    sequence = SequenceEffect(
                        sampled_token_ids=(int(sampled.token_id),),
                        sampled_logprob=(
                            None if sampled.logprob is None else float(sampled.logprob)
                        ),
                        top_logprobs=_token_logprobs(sampled.top_logprobs),
                    )
                continue
            if stage.row is RouteRowKind.FLOW:
                if latent is None:
                    raise invalid_descriptor("flow state stage has no latent tensor")
                task = self._latent_state_task(
                    envelope,
                    stage,
                    latent,
                    height,
                    width,
                    conditioning_position,
                    scope,
                )
                outputs = yield (task,)
                _flow_prediction(outputs[0])
                self._commit_task_kv(task, task.query_tokens, scope)
                total += task.query_tokens
                continue
            raise invalid_descriptor("state publication cannot use a decode row")
        return _StateOutcome(total, sequence)

    def _vision_state_task(
        self,
        envelope: OperationEnvelope,
        stage: OperationStageSpec,
        features: torch.Tensor,
        grid: torch.Tensor | None,
        conditioning_position: int,
        scope: _ExecutionScope,
        *,
        close_image: bool,
        logits: bool,
    ) -> _ForwardTask:
        images = cast(ModelSpec, self.spec).inputs.images
        injection = None if images is None else images.feature_injection
        if injection is None:
            raise invalid_descriptor("vision state stage requires declared feature injection")
        embeddings = (
            features.squeeze(0) if features.ndim == 3 and int(features.shape[0]) == 1 else features
        )
        if embeddings.ndim != 2 or int(embeddings.shape[0]) < 1:
            raise invalid_descriptor("vision features must have shape [tokens, hidden]")
        segments: list[TokenIds | TokenEmbeddings] = []
        leading = injection.layout is FeatureLayout.FRAMED
        trailing = leading or close_image
        if leading:
            segments.append(
                TokenIds(
                    torch.tensor((self._feature_token_id(injection, start=True),), dtype=torch.long)
                )
            )
        segments.append(TokenEmbeddings(embeddings))
        if trailing:
            segments.append(
                TokenIds(
                    torch.tensor(
                        (self._feature_token_id(injection, start=False),), dtype=torch.long
                    )
                )
            )
        inputs: TokenIds | TokenEmbeddings | TokenSegments
        inputs = segments[0] if len(segments) == 1 else TokenSegments(tuple(segments))
        positions = self._vision_positions(
            injection.positions,
            grid,
            int(embeddings.shape[0]),
            conditioning_position,
            leading=leading,
            trailing=trailing,
            close_image=close_image,
        )
        row_id = scope.row_id()
        row = TokenRow(
            row_id=row_id,
            inputs=inputs,
            positions=positions,
            output_slot=row_id,
            selection=TokenSelection.LAST_LOGITS if logits else TokenSelection.HIDDEN,
        )
        return _ForwardTask(
            envelope=envelope,
            stage=stage,
            route=self._route(stage),
            row=row,
            entry=self.kv.get(envelope.session_id),
            write_kv=True,
            causal=False,
            attention_indexes=_positions_as_three_axis(positions, _token_input_length(row)),
        )

    def _feature_token_id(self, injection: Any, *, start: bool) -> int:
        value = injection.start_token_id if start else injection.end_token_id
        text = injection.start_token if start else injection.end_token
        if value is not None:
            return int(value)
        if text is None or self.tokenizer is None:
            raise capability_mismatch("feature marker requires a worker tokenizer or token id")
        token_id = self.tokenizer.convert_tokens_to_ids(text)
        if token_id is None or int(token_id) < 0:
            raise invalid_descriptor("declared feature marker is absent from the tokenizer")
        return int(token_id)

    def _vision_positions(
        self,
        layout: PositionLayout,
        grid: torch.Tensor | None,
        feature_tokens: int,
        conditioning_position: int,
        *,
        leading: bool,
        trailing: bool,
        close_image: bool,
    ) -> torch.Tensor:
        query = int(leading) + feature_tokens + int(trailing)
        if layout is PositionLayout.TEMPORAL:
            return torch.full((query,), int(conditioning_position), dtype=torch.long)
        if grid is None or grid.numel() != 2:
            raise invalid_descriptor("temporal-spatial feature injection requires one image grid")
        raw_height, raw_width = (int(value) for value in grid.reshape(-1).tolist())
        factor_squared, remainder = divmod(raw_height * raw_width, feature_tokens)
        factor = math.isqrt(factor_squared)
        if remainder or factor < 1 or factor * factor != factor_squared:
            raise invalid_descriptor("vision feature count does not align with its patch grid")
        grid_height, grid_width = raw_height // factor, raw_width // factor
        if grid_height * grid_width != feature_tokens:
            raise invalid_descriptor("vision output grid is not integral")
        temporal = torch.full(
            (query,),
            int(conditioning_position + (1 if close_image else 0)),
            dtype=torch.long,
        )
        y = torch.arange(grid_height, dtype=torch.long).repeat_interleave(grid_width)
        x = torch.arange(grid_width, dtype=torch.long).repeat(grid_height)
        spatial_y = torch.zeros(query, dtype=torch.long)
        spatial_x = torch.zeros(query, dtype=torch.long)
        begin = int(leading)
        spatial_y[begin : begin + feature_tokens] = y
        spatial_x[begin : begin + feature_tokens] = x
        if trailing and close_image:
            temporal[-1] = conditioning_position + 2
        return torch.stack((temporal, spatial_y, spatial_x))

    def _latent_state_task(
        self,
        envelope: OperationEnvelope,
        stage: OperationStageSpec,
        latent: torch.Tensor,
        height: int,
        width: int,
        conditioning_position: int,
        scope: _ExecutionScope,
    ) -> _ForwardTask:
        flow = cast(ModelSpec, self.spec).flow
        if flow is None or flow.latent_layout is not LatentLayout.PATCH_TOKENS:
            raise invalid_descriptor("flow state publication requires patch-token latents")
        image_tokens = (height // int(flow.latent_downsample)) * (
            width // int(flow.latent_downsample)
        )
        if int(latent.reshape(-1, latent.shape[-1]).shape[0]) != image_tokens:
            raise invalid_descriptor("state latent does not match the declared image geometry")
        query = image_tokens + int(flow.commit_marker_tokens)
        row_id = scope.row_id()
        row = FlowRow(
            row_id=row_id,
            conditioning=NoFlowConditioning(),
            positions=get_flattened_position_ids_extrapolate(
                height,
                width,
                int(flow.latent_downsample),
                int(math.isqrt(flow.max_latent_tokens)),
            ),
            timestep=latent.new_zeros(1),
            latent=latent,
            image_tokens=query,
            image_height=height,
            image_width=width,
            output_slot=row_id,
        )
        temporal = torch.full((query,), conditioning_position + 1, dtype=torch.long)
        temporal[0] = conditioning_position
        temporal[-1] = conditioning_position + int(flow.rope_advance)
        indexes = torch.stack((temporal, torch.zeros_like(temporal), torch.zeros_like(temporal)))
        return _ForwardTask(
            envelope=envelope,
            stage=stage,
            route=self._route(stage),
            row=row,
            entry=self.kv.get(envelope.session_id),
            write_kv=True,
            causal=False,
            attention_indexes=indexes,
            text_local_indices=(0, query - 1),
        )

    def _materialize_frames(
        self,
        envelope: OperationEnvelope,
        operation: MaterializeOperation,
        scope: _ExecutionScope,
    ) -> MaterializeDelta:
        if not isinstance(operation.input, PublishedProduct):
            raise invalid_descriptor("frame materialization requires a published image product")
        source = scope.product_view.get(operation.input.handle) if operation.input.handle else None
        if source is not None and isinstance(source.payload, FrameCollectionProduct):
            return MaterializeDelta(product=FrameRecord(len(source.payload.frames)))
        image, metadata = self._fetch_product_tensor(operation.input, scope)
        if _metadata_string(metadata, "payload_kind", "") != "image_nchw":
            raise invalid_descriptor("frame materialization source is not an image tensor")
        value_range = ImageRange(
            _metadata_string(metadata, "value_range", ImageRange.SIGNED_UNIT.value)
        )
        png = tensor_to_png_b64(
            image,
            value_range=(-1.0, 1.0) if value_range is ImageRange.SIGNED_UNIT else (0.0, 1.0),
        )
        self._append_frame(envelope, png, scope)
        frame_handle = _stable_handle(envelope.session_id, envelope.epoch, 0, "frames")
        frames = scope.product_view.require(frame_handle).payload
        if not isinstance(frames, FrameCollectionProduct):
            raise RuntimeError("frame collection publication produced the wrong product variant")
        return MaterializeDelta(product=FrameRecord(len(frames.frames)))

    def _append_frame(
        self, envelope: OperationEnvelope, png_base64: str, scope: _ExecutionScope
    ) -> None:
        handle = _stable_handle(envelope.session_id, envelope.epoch, 0, "frames")
        existing = scope.product_view.get(handle)
        frames = (
            existing.payload.frames
            if existing is not None and isinstance(existing.payload, FrameCollectionProduct)
            else ()
        )
        scope.product_view.put(
            ProductRecord(
                handle=handle,
                session_id=envelope.session_id,
                payload=FrameCollectionProduct((*frames, EncodedImageProduct(png_base64))),
            )
        )
        self.sessions.get(envelope.session_id).product_handles.add(handle)

    def _publish_tensor(
        self,
        value: torch.Tensor,
        scope: _ExecutionScope,
        *,
        payload_kind: str,
        height: int,
        width: int,
        value_range: str,
    ) -> str:
        if self.transport is None:
            raise capability_mismatch("tensor publication requires a configured transport")
        locator = self.transport.publish(value.detach().contiguous())
        metadata: dict[str, object] = {"payload_kind": payload_kind}
        if height > 0 and width > 0:
            metadata.update({"height": int(height), "width": int(width)})
        if value_range:
            metadata["value_range"] = value_range
        locator = replace(locator, meta={**locator.meta, **metadata})
        scope.published.append(locator)
        return locator.to_wire_json()

    def _fetch_product_tensor(
        self,
        source: PublishedProduct,
        scope: _ExecutionScope,
    ) -> tuple[torch.Tensor, Mapping[str, object]]:
        if source.handle > 0:
            record = scope.product_view.get(source.handle)
            if record is not None:
                payload = record.payload
                if isinstance(payload, VisionFeatureProduct):
                    return payload.features, {
                        "payload_kind": "vision_features",
                        "height": payload.height,
                        "width": payload.width,
                    }
                if isinstance(payload, LatentFeatureProduct):
                    return payload.latent, {
                        "payload_kind": "latent_features",
                        "height": payload.height,
                        "width": payload.width,
                    }
                if isinstance(payload, LogitsProduct):
                    return payload.logits, {"payload_kind": "logits"}
                if isinstance(payload, ImageTensorProduct):
                    return payload.image, {
                        "payload_kind": "image_nchw",
                        "height": payload.height,
                        "width": payload.width,
                        "value_range": payload.value_range.value,
                    }
                raise invalid_descriptor("resident product is not tensor-transferable")
        if not source.locator or self.transport is None:
            raise invalid_descriptor("product is not resident or transport-addressable")
        locator = Locator.from_wire_json(source.locator)
        value = fetch_locator(self.transport, locator)
        if not isinstance(value, torch.Tensor):
            raise invalid_descriptor("product transport returned a non-tensor value")
        return value, locator.meta


def _token_input_length(row: TokenRow) -> int:
    inputs = row.inputs
    if isinstance(inputs, (TokenIds, TokenEmbeddings)):
        return int(inputs.values.shape[0])
    return sum(int(value.values.shape[0]) for value in inputs.values)


def _metadata_uint(metadata: Mapping[str, object], name: str, default: int) -> int:
    value = metadata.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise invalid_descriptor(f"product metadata field {name!r} must be a non-negative integer")
    return value


def _metadata_string(metadata: Mapping[str, object], name: str, default: str) -> str:
    value = metadata.get(name, default)
    if not isinstance(value, str):
        raise invalid_descriptor(f"product metadata field {name!r} must be a string")
    return value


def _three_axis_positions(row: ForwardRow, query: int) -> torch.Tensor:
    positions = row.positions if isinstance(row, (TokenRow, FlowRow)) else None
    if positions is None:
        return torch.zeros((3, query), dtype=torch.long)
    if positions.ndim == 1 and int(positions.numel()) == query:
        return torch.stack((positions, torch.zeros_like(positions), torch.zeros_like(positions)))
    if positions.ndim == 2 and tuple(positions.shape) == (3, query):
        return positions
    raise invalid_descriptor("row positions cannot be lowered to three-axis attention indexes")


def _stage_ints(
    values: Sequence[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
    slot: TensorStagingSlot,
    name: str,
) -> torch.Tensor:
    cpu = cpu_int_staging_buffer(
        len(values),
        dtype=dtype,
        pin=device.type == "cuda",
        slot=slot,
        name=name,
    )
    fill_cpu_ints(cpu, tuple(int(value) for value in values))
    return copy_cpu_to_device(
        cpu,
        device=device,
        non_blocking=device.type == "cuda" and is_pinned(cpu),
        slot=slot,
        name=name,
    )


def _cumulative(
    lengths: Sequence[int],
    device: torch.device,
    *,
    slot: TensorStagingSlot,
    name: str,
) -> torch.Tensor:
    values = [0]
    for length in lengths:
        values.append(values[-1] + int(length))
    return _stage_ints(
        values,
        dtype=torch.int32,
        device=device,
        slot=slot,
        name=name,
    )


def _output_kind(row: ForwardRow) -> OutputKind:
    if isinstance(row, TokenRow):
        return OutputKind.TOKEN
    if isinstance(row, FlowRow):
        return OutputKind.FLOW
    if isinstance(row, EncodeRow):
        return OutputKind.ENCODE
    return OutputKind.DECODE


def _binding_identity(tasks: Sequence[_ForwardTask]) -> int:
    digest = hashlib.sha256(b"uniserve-forward-binding\0")
    for task in tasks:
        digest.update(task.route.name.encode("utf-8"))
        digest.update(task.row_kind.value.encode("ascii"))
        digest.update(task.query_tokens.to_bytes(8, "little"))
    return int.from_bytes(digest.digest()[:8], "little")


def _trace_envelopes(
    envelopes: Sequence[OperationEnvelope],
) -> tuple[OperationTrace, ...]:
    return tuple(
        OperationTrace(
            session_id=int(envelope.session_id),
            epoch=int(envelope.epoch),
            op_id=int(envelope.op_id),
            version=int(envelope.base_version),
        )
        for envelope in envelopes
    )


def _record_component(scope: _ExecutionScope, name: str, started_ns: int) -> None:
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
    components = {"forward": sum(route_us.values())}
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


def _row_tensor_shape(row: ForwardRow) -> tuple[int, ...]:
    if isinstance(row, TokenRow):
        return (_token_input_length(row),)
    if isinstance(row, FlowRow):
        return tuple(int(value) for value in row.latent.shape)
    if isinstance(row, EncodeRow):
        return tuple(int(value) for value in row.inputs.pixels.shape)
    return tuple(int(value) for value in row.latent.shape)


def _row_element_count(row: ForwardRow) -> int:
    shape = _row_tensor_shape(row)
    return math.prod(shape) if shape else 0


def _token_logits(output: ForwardRowOutput) -> torch.Tensor:
    if not isinstance(output, TokenOutput) or not isinstance(output.value, TokenLogits):
        raise invalid_descriptor("token route did not return logits")
    return output.value.value


def _require_sampling(session: RequestSession) -> SamplingParams:
    if session.sampling is None:
        raise invalid_descriptor("sequence execution requires admitted sampling parameters")
    return session.sampling


def _require_image(session: RequestSession) -> ImageParams:
    if session.image is None:
        raise invalid_descriptor("flow execution requires admitted image parameters")
    return session.image


def _token_logits_or_hidden(output: ForwardRowOutput) -> torch.Tensor:
    if not isinstance(output, TokenOutput) or not isinstance(
        output.value, (TokenLogits, TokenHidden)
    ):
        raise invalid_descriptor("token route did not return a token tensor")
    return output.value.value


def _flow_prediction(output: ForwardRowOutput) -> torch.Tensor:
    if not isinstance(output, FlowOutput):
        raise invalid_descriptor("flow route did not return a flow prediction")
    return output.prediction


def _token_logprobs(
    values: Iterable[Sequence[float | int]] | None,
) -> tuple[TokenLogprob, ...]:
    if values is None:
        return ()
    result: list[TokenLogprob] = []
    for value in values:
        if len(value) != 3:
            raise invalid_descriptor("sample logprob entries must contain token, logprob, and rank")
        result.append(TokenLogprob(int(value[0]), float(value[1]), int(value[2])))
    return tuple(result)


def _sample_result(value: object) -> _SampleResult:
    if not isinstance(value, _SampleResult):
        raise RuntimeError("sampling task returned an invalid result")
    return value


def _sample_relay_token(sample: _SampleResult) -> int | SampledTokenRelay:
    if sample.device_token is not None:
        return SampledTokenRelay(sample.device_token)
    return int(sample.token_id)


def _recent_token_counts(values: Sequence[int]) -> tuple[tuple[int, int], ...]:
    counts: dict[int, int] = {}
    for value in values:
        token_id = int(value)
        counts[token_id] = counts.get(token_id, 0) + 1
    return tuple(counts.items())


def _sampling_task_tensors(
    rows: Sequence[_SamplingRow],
    *,
    vocab: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    first = rows[0]
    parameters = first.parameters
    uses_penalties = (
        parameters.repetition_penalty != 1.0
        or parameters.frequency_penalty != 0.0
        or parameters.presence_penalty != 0.0
    )
    recent = (
        tuple((token_id, count) for token_id, count in first.recent_counts if 0 <= token_id < vocab)
        if uses_penalties
        else ()
    )
    width = min(vocab, bucketed_length(len(recent)))
    used = {token_id for token_id, _count in recent}
    padding: list[int] = []
    candidate = vocab - 1
    while len(recent) + len(padding) < width:
        if candidate not in used:
            padding.append(candidate)
        candidate -= 1
    token_ids = torch.tensor(
        (*[token_id for token_id, _count in recent], *padding),
        dtype=torch.long,
        device=device,
    ).reshape(1, width)
    counts = torch.tensor(
        (*[count for _token_id, count in recent], *([0] * len(padding))),
        dtype=torch.float32,
        device=device,
    ).reshape(1, width)
    row_count = len(rows)
    parameter_values = torch.tensor(
        [
            (
                float(row.parameters.temperature),
                float(row.parameters.top_p),
                float(row.parameters.min_p),
                float(row.parameters.repetition_penalty),
                float(row.parameters.frequency_penalty),
                float(row.parameters.presence_penalty),
            )
            for row in rows
        ],
        dtype=torch.float32,
        device=device,
    )
    return (
        token_ids.expand(row_count, width),
        counts.expand(row_count, width),
        parameter_values,
    )


def _plain_greedy_row(row: _SamplingRow) -> bool:
    parameters = row.parameters
    return (
        float(parameters.temperature) <= 0.0
        and not parameters.return_logprobs
        and int(row.n_logprobs) == 0
        and not parameters.logprob_token_ids
        and row.allowed is None
        and not row.suppress
        and not parameters.logit_bias
        and parameters.repetition_penalty == 1.0
        and parameters.frequency_penalty == 0.0
        and parameters.presence_penalty == 0.0
    )


@torch.inference_mode()
def _sample_task_batch(
    tasks: Sequence[_SampleTask],
    token_mirrors: _PinnedTokenRing | None = None,
) -> tuple[_SampleResult, ...]:
    """Shape and draw every compatible sampling row in each device batch."""

    mirrors = token_mirrors or _PinnedTokenRing(1, max(1, len(tasks)))
    grouped: dict[tuple[torch.device, int, int, int], list[tuple[int, _SampleTask]]] = defaultdict(
        list
    )
    for index, task in enumerate(tasks):
        if (
            task.logits.ndim != 2
            or not task.logits.is_floating_point()
            or int(task.logits.shape[0]) < 1
            or int(task.logits.shape[1]) < 1
            or int(task.logits.shape[0]) != len(task.rows)
        ):
            raise invalid_descriptor("sampling task logits must be shaped [rows, vocab]")
        plain_greedy = not task.draft_token_ids and all(_plain_greedy_row(row) for row in task.rows)
        if plain_greedy:
            if (
                any(
                    value is not None
                    for value in (
                        task.noise,
                        task.penalty_token_ids,
                        task.penalty_counts,
                        task.parameter_values,
                    )
                )
                or task.draft_token_ids
                or len(task.rows) != 1
            ):
                raise invalid_descriptor("greedy sampling task has shaped metadata")
        else:
            noise = cast(torch.Tensor, task.noise)
            penalty_token_ids = cast(torch.Tensor, task.penalty_token_ids)
            penalty_counts = cast(torch.Tensor, task.penalty_counts)
            parameter_values = cast(torch.Tensor, task.parameter_values)
            if (
                noise.device != task.logits.device
                or tuple(noise.shape) != tuple(task.logits.shape)
                or not noise.is_floating_point()
            ):
                raise invalid_descriptor("sampling task noise must align with its logits")
            if (
                penalty_token_ids.device != task.logits.device
                or penalty_counts.device != task.logits.device
                or penalty_token_ids.ndim != 2
                or penalty_counts.shape != penalty_token_ids.shape
                or int(penalty_token_ids.shape[0]) != len(task.rows)
            ):
                raise invalid_descriptor("sampling task penalty tensors do not align")
            if parameter_values.device != task.logits.device or parameter_values.shape != (
                len(task.rows),
                6,
            ):
                raise invalid_descriptor("sampling task parameter vectors do not align")
        if task.draft_token_ids:
            if len(task.rows) != len(task.draft_token_ids) + 1:
                raise invalid_descriptor("speculative sampling rows do not cover the draft chain")
            if task.acceptance_uniforms is None or tuple(task.acceptance_uniforms.shape) != (
                len(task.draft_token_ids),
            ):
                raise invalid_descriptor(
                    "speculative acceptance RNG shape does not match the draft"
                )
        elif task.acceptance_uniforms is not None or len(task.rows) != 1:
            raise invalid_descriptor("ordinary sampling tasks must contain exactly one row")
        vocab = int(task.logits.shape[1])
        if any(value < 0 or value >= vocab for value in task.draft_token_ids):
            raise invalid_descriptor("speculative draft token is outside the model vocabulary")
        sampling_path = -1 if plain_greedy else _fused_top_k(task, vocab)
        penalty_width = (
            int(cast(torch.Tensor, task.penalty_token_ids).shape[1]) if sampling_path > 0 else 0
        )
        grouped[(task.logits.device, vocab, sampling_path, penalty_width)].append((index, task))

    result: list[_SampleResult | None] = [None] * len(tasks)
    for (_device, _vocab, sampling_path, _penalty_width), compatible in grouped.items():
        indexes, group = zip(*compatible, strict=True)
        if sampling_path == -1:
            sampled_group = _sample_plain_greedy_group(tuple(group), mirrors)
        elif sampling_path > 0:
            sampled_group = _sample_fused_top_k_group(tuple(group), sampling_path)
        else:
            sampled_group = _sample_task_group(tuple(group))
        for index, sampled in zip(indexes, sampled_group, strict=True):
            result[index] = sampled
    return tuple(cast(_SampleResult, value) for value in result)


def _sample_plain_greedy_group(
    tasks: tuple[_SampleTask, ...],
    token_mirrors: _PinnedTokenRing,
) -> tuple[_SampleResult, ...]:
    logits = packed_tensor_views(tuple(task.logits for task in tasks))
    if logits is None:
        logits = torch.cat(tuple(task.logits for task in tasks), dim=0)
    else:
        logits = logits.reshape(len(tasks), -1)
    device_tokens = torch.argmax(logits, dim=-1)
    mirror = token_mirrors.capture(device_tokens)
    host_tokens = mirror.finalize() if any(not task.defer_host_token for task in tasks) else None
    return tuple(
        _SampleResult(
            token_id=(
                _DeferredToken(mirror, index)
                if task.defer_host_token
                else cast(tuple[int, ...], host_tokens)[index]
            ),
            device_token=device_tokens[index : index + 1],
            logprob=None,
            top_logprobs=None,
        )
        for index, task in enumerate(tasks)
    )


def _fused_top_k(task: _SampleTask, vocab: int) -> int:
    if task.logits.device.type != "cuda":
        return 0
    if task.draft_token_ids:
        return 0
    row = task.rows[0]
    parameters = row.parameters
    top_k = int(parameters.top_k)
    wants_logprobs = (
        parameters.return_logprobs or int(row.n_logprobs) > 0 or bool(parameters.logprob_token_ids)
    )
    if (
        wants_logprobs
        or row.allowed is not None
        or row.suppress
        or parameters.logit_bias
        or top_k <= 0
        or top_k > 128
        or top_k >= vocab
    ):
        return 0
    return top_k


def _sample_fused_top_k_group(
    tasks: tuple[_SampleTask, ...],
    top_k: int,
) -> tuple[_SampleResult, ...]:
    rows = tuple(task.rows[0] for task in tasks)
    logits = torch.cat(tuple(task.logits for task in tasks), dim=0)
    noise = torch.cat(tuple(cast(torch.Tensor, task.noise) for task in tasks), dim=0)
    penalty_token_ids = torch.cat(
        tuple(cast(torch.Tensor, task.penalty_token_ids) for task in tasks), dim=0
    )
    penalty_counts = torch.cat(
        tuple(cast(torch.Tensor, task.penalty_counts) for task in tasks), dim=0
    )
    parameters = torch.cat(
        tuple(cast(torch.Tensor, task.parameter_values) for task in tasks), dim=0
    )
    tokens, valid = _run_fused_top_k_sampling(
        logits,
        noise,
        penalty_token_ids,
        penalty_counts,
        parameters,
        top_k,
    )
    metadata = torch.cat((valid.to(torch.long), tokens)).cpu().tolist()
    row_count = len(rows)
    if not all(bool(value) for value in metadata[:row_count]):
        raise invalid_descriptor("sampling policy masked every vocabulary entry")
    return tuple(_SampleResult(int(token), None, None, None) for token in metadata[row_count:])


def _run_fused_top_k_sampling(
    logits: torch.Tensor,
    noise: torch.Tensor,
    penalty_token_ids: torch.Tensor,
    penalty_counts: torch.Tensor,
    parameters: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not triton_device_supported(logits.device):
        raise capability_mismatch("fused sampling requires a supported Triton toolchain")
    provider = import_module("uniserve_kernel.sampling")
    result = provider.sample_top_k(
        logits,
        noise,
        penalty_token_ids,
        penalty_counts,
        parameters,
        int(top_k),
    )
    if (
        not isinstance(result, tuple)
        or len(result) != 2
        or not all(isinstance(value, torch.Tensor) for value in result)
    ):
        raise RuntimeError("sampling provider returned an invalid result")
    return result


def _sample_task_group(tasks: tuple[_SampleTask, ...]) -> tuple[_SampleResult, ...]:
    device = tasks[0].logits.device
    vocab = int(tasks[0].logits.shape[1])
    offsets: list[int] = []
    offset = 0
    for task in tasks:
        offsets.append(offset)
        offset += len(task.rows)
    rows = tuple(row for task in tasks for row in task.rows)
    logits = torch.cat(tuple(task.logits.float() for task in tasks), dim=0)
    noise = torch.cat(tuple(cast(torch.Tensor, task.noise) for task in tasks), dim=0)
    work, valid = _shape_sampling_logits_batch(logits, rows)

    acceptance_rows: list[int] = []
    acceptance_tokens: list[int] = []
    acceptance_parts: list[torch.Tensor] = []
    acceptance_spans: dict[int, tuple[int, int]] = {}
    acceptance_offset = 0
    for task_index, (task, row_offset) in enumerate(zip(tasks, offsets, strict=True)):
        count = len(task.draft_token_ids)
        if not count:
            continue
        acceptance_rows.extend(range(row_offset, row_offset + count))
        acceptance_tokens.extend(task.draft_token_ids)
        acceptance_parts.append(cast(torch.Tensor, task.acceptance_uniforms))
        acceptance_spans[task_index] = (acceptance_offset, count)
        acceptance_offset += count

    accepted_flags: torch.Tensor | None = None
    if acceptance_rows:
        acceptance_indexes = torch.tensor(
            acceptance_rows,
            dtype=torch.long,
            device=device,
        )
        target_indexes = torch.tensor(
            acceptance_tokens,
            dtype=torch.long,
            device=device,
        )
        target_probabilities = torch.softmax(
            work.index_select(0, acceptance_indexes),
            dim=-1,
        ).gather(1, target_indexes.unsqueeze(1))[:, 0]
        coins = torch.cat(
            tuple(
                value.to(device=device, dtype=target_probabilities.dtype)
                for value in acceptance_parts
            ),
            dim=0,
        )
        accepted_flags = (coins <= target_probabilities) | (target_probabilities >= 1.0)

    accepted_counts: list[torch.Tensor] = []
    output_rows: list[torch.Tensor] = []
    for task_index, row_offset in enumerate(offsets):
        span = acceptance_spans.get(task_index)
        if span is None:
            accepted = torch.zeros((), dtype=torch.long, device=device)
        else:
            start, count = span
            flags = cast(torch.Tensor, accepted_flags)[start : start + count]
            accepted = torch.cumprod(flags.to(torch.long), dim=0).sum()
        accepted_counts.append(accepted)
        output_rows.append(accepted + row_offset)

    sample_work = work
    if acceptance_rows:
        sample_work = work.clone()
        flat_exclusions = torch.tensor(
            [
                row * vocab + token
                for row, token in zip(
                    acceptance_rows,
                    acceptance_tokens,
                    strict=True,
                )
            ],
            dtype=torch.long,
            device=device,
        )
        sample_work.reshape(-1).index_fill_(0, flat_exclusions, float("-inf"))
        has_residual = torch.isfinite(sample_work).any(dim=-1)
        sample_work = torch.where(has_residual.unsqueeze(1), sample_work, work)

    temperatures = torch.tensor(
        [float(row.parameters.temperature) for row in rows],
        dtype=work.dtype,
        device=device,
    )
    gumbel = -torch.log(-torch.log(noise))
    semantic_noise = torch.where(
        temperatures.unsqueeze(1) > 0.0,
        gumbel,
        torch.zeros((), dtype=work.dtype, device=device),
    )
    row_tokens = torch.argmax(sample_work + semantic_noise, dim=-1)
    output_indexes = torch.stack(output_rows)
    task_tokens = row_tokens.index_select(0, output_indexes)
    counts = torch.stack(accepted_counts)

    metadata = torch.cat((valid.to(torch.long), task_tokens, counts)).cpu().tolist()
    row_count = len(rows)
    task_count = len(tasks)
    if not all(bool(value) for value in metadata[:row_count]):
        raise invalid_descriptor("sampling policy masked every vocabulary entry")
    token_values = tuple(int(value) for value in metadata[row_count : row_count + task_count])
    accepted_values = tuple(int(value) for value in metadata[row_count + task_count :])
    details = _sample_logprob_details(
        sample_work,
        output_indexes,
        task_tokens,
        tuple(task.rows[0] for task in tasks),
    )
    return tuple(
        _SampleResult(
            token_id=token,
            device_token=None,
            logprob=None if index not in details else details[index][0],
            top_logprobs=None if index not in details else details[index][1],
            num_accepted_tokens=accepted,
        )
        for index, (token, accepted) in enumerate(zip(token_values, accepted_values, strict=True))
    )


def _semantic_sampling_noise(
    rows: Sequence[_SamplingRow],
    *,
    vocab: int,
    device: torch.device,
) -> torch.Tensor:
    stochastic = tuple(
        index for index, row in enumerate(rows) if float(row.parameters.temperature) > 0.0
    )
    if len(stochastic) == len(rows):
        return torch.stack(
            tuple(
                uniform_samples(
                    (vocab,),
                    seed=row.draw_seed,
                    device=device,
                )
                for row in rows
            ),
            dim=0,
        )
    noise = torch.zeros((len(rows), vocab), dtype=torch.float32, device=device)
    if stochastic:
        indexes = torch.tensor(stochastic, dtype=torch.long, device=device)
        draws = torch.stack(
            tuple(
                uniform_samples(
                    (vocab,),
                    seed=rows[index].draw_seed,
                    device=device,
                )
                for index in stochastic
            ),
            dim=0,
        )
        noise.index_copy_(0, indexes, draws)
    return noise


def _shape_sampling_logits_batch(
    logits: torch.Tensor,
    rows: Sequence[_SamplingRow],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the canonical shaping and truncation order to a logits matrix."""

    work = logits.to(dtype=torch.float32, copy=True)
    row_count, vocab = (int(value) for value in work.shape)
    if row_count != len(rows):
        raise invalid_descriptor("sampling parameters do not align with logits rows")

    allowed_rows: list[int] = []
    allowed_flat: list[int] = []
    for row_index, row in enumerate(rows):
        if row.allowed is None:
            continue
        allowed = tuple(
            dict.fromkeys(int(value) for value in row.allowed if 0 <= int(value) < vocab)
        )
        if not allowed:
            raise invalid_descriptor("sampling allowed-token set has no vocabulary entries")
        allowed_rows.append(row_index)
        allowed_flat.extend(row_index * vocab + value for value in allowed)
    if allowed_rows:
        mask = torch.ones_like(work, dtype=torch.bool)
        mask.index_fill_(
            0,
            torch.tensor(allowed_rows, dtype=torch.long, device=work.device),
            False,
        )
        mask.reshape(-1)[torch.tensor(allowed_flat, dtype=torch.long, device=work.device)] = True
        work.masked_fill_(~mask, float("-inf"))

    suppressed_flat = tuple(
        row_index * vocab + value
        for row_index, row in enumerate(rows)
        for value in dict.fromkeys(int(token) for token in row.suppress if 0 <= int(token) < vocab)
    )
    if suppressed_flat:
        work.reshape(-1).index_fill_(
            0,
            torch.tensor(suppressed_flat, dtype=torch.long, device=work.device),
            float("-inf"),
        )

    bias_indexes: list[int] = []
    bias_values: list[float] = []
    for row_index, row in enumerate(rows):
        for token_id, bias in row.parameters.logit_bias:
            index = int(token_id)
            if 0 <= index < vocab:
                bias_indexes.append(row_index * vocab + index)
                bias_values.append(float(bias))
    if bias_indexes:
        work.reshape(-1).index_put_(
            (
                torch.tensor(
                    bias_indexes,
                    dtype=torch.long,
                    device=work.device,
                ),
            ),
            torch.tensor(bias_values, dtype=work.dtype, device=work.device),
            accumulate=True,
        )

    penalty_indexes: list[int] = []
    repetitions: list[float] = []
    frequencies: list[float] = []
    presences: list[float] = []
    occurrences: list[int] = []
    for row_index, row in enumerate(rows):
        parameters = row.parameters
        if (
            parameters.repetition_penalty == 1.0
            and parameters.frequency_penalty == 0.0
            and parameters.presence_penalty == 0.0
        ):
            continue
        for token_id, count in row.recent_counts:
            if 0 <= token_id < vocab:
                penalty_indexes.append(row_index * vocab + token_id)
                repetitions.append(float(parameters.repetition_penalty))
                frequencies.append(float(parameters.frequency_penalty))
                presences.append(float(parameters.presence_penalty))
                occurrences.append(count)
    if penalty_indexes:
        indexes = torch.tensor(
            penalty_indexes,
            dtype=torch.long,
            device=work.device,
        )
        penalty_values = torch.tensor(
            tuple(zip(repetitions, frequencies, presences, occurrences, strict=True)),
            dtype=work.dtype,
            device=work.device,
        )
        values = work.reshape(-1).index_select(0, indexes)
        repetition = penalty_values[:, 0]
        repeated = torch.where(
            values > 0.0,
            values / repetition,
            values * repetition,
        )
        adjusted = repeated - penalty_values[:, 1] * penalty_values[:, 3] - penalty_values[:, 2]
        work.reshape(-1).index_copy_(
            0,
            indexes,
            torch.where(torch.isneginf(values), values, adjusted),
        )

    parameter_values = torch.tensor(
        [
            (
                float(row.parameters.temperature),
                float(row.parameters.min_p),
                float(row.parameters.top_p),
            )
            for row in rows
        ],
        dtype=work.dtype,
        device=work.device,
    )
    temperatures = parameter_values[:, 0]
    divisors = torch.where(
        temperatures > 0.0,
        temperatures,
        torch.ones((), dtype=work.dtype, device=work.device),
    )
    work.div_(divisors.unsqueeze(1))

    min_p_rows = tuple(index for index, row in enumerate(rows) if float(row.parameters.min_p) > 0.0)
    if min_p_rows:
        indexes = torch.tensor(min_p_rows, dtype=torch.long, device=work.device)
        subset = work.index_select(0, indexes)
        min_p = parameter_values.index_select(0, indexes)[:, 1]
        min_threshold = subset.max(dim=-1).values + torch.log(min_p)
        subset.masked_fill_(subset < min_threshold.unsqueeze(1), float("-inf"))
        work.index_copy_(0, indexes, subset)

    top_k_groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        top_k = int(row.parameters.top_k)
        if 0 < top_k < vocab:
            top_k_groups[top_k].append(index)
    for top_k, row_indexes in top_k_groups.items():
        indexes = torch.tensor(row_indexes, dtype=torch.long, device=work.device)
        subset = work.index_select(0, indexes)
        values, token_indexes = torch.topk(
            subset,
            top_k,
            dim=-1,
            sorted=False,
        )
        ordered, order = torch.sort(values, dim=-1, descending=True)
        token_indexes = token_indexes.gather(1, order)
        top_p = parameter_values.index_select(0, indexes)[:, 2]
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        over = cumulative > top_p.unsqueeze(1)
        drop = torch.cat(
            (
                torch.zeros(
                    (len(row_indexes), 1),
                    dtype=torch.bool,
                    device=work.device,
                ),
                over[:, :-1],
            ),
            dim=1,
        )
        ordered.masked_fill_(drop, float("-inf"))
        truncated = torch.full_like(subset, float("-inf"))
        truncated.scatter_(1, token_indexes, ordered)
        work.index_copy_(0, indexes, truncated)

    top_p_rows = tuple(
        index
        for index, row in enumerate(rows)
        if (0.0 < float(row.parameters.top_p) < 1.0 and not 0 < int(row.parameters.top_k) < vocab)
    )
    if top_p_rows:
        indexes = torch.tensor(top_p_rows, dtype=torch.long, device=work.device)
        subset = work.index_select(0, indexes)
        ordered, token_indexes = torch.sort(subset, dim=-1, descending=True)
        top_p = parameter_values.index_select(0, indexes)[:, 2]
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        over = cumulative > top_p.unsqueeze(1)
        drop = torch.cat(
            (
                torch.zeros(
                    (len(top_p_rows), 1),
                    dtype=torch.bool,
                    device=work.device,
                ),
                over[:, :-1],
            ),
            dim=1,
        )
        ordered.masked_fill_(drop, float("-inf"))
        truncated = torch.full_like(subset, float("-inf"))
        truncated.scatter_(1, token_indexes, ordered)
        work.index_copy_(0, indexes, truncated)

    valid = ~torch.isnan(work).any(dim=-1) & torch.isfinite(work).any(dim=-1)
    return work, valid


def _sample_logprob_details(
    work: torch.Tensor,
    output_rows: torch.Tensor,
    output_tokens: torch.Tensor,
    rows: Sequence[_SamplingRow],
) -> dict[int, tuple[float, tuple[tuple[int, float, int], ...]]]:
    vocab = int(work.shape[1])
    requested_rows = tuple(
        index
        for index, row in enumerate(rows)
        if row.parameters.return_logprobs
        or int(row.n_logprobs) > 0
        or bool(row.parameters.logprob_token_ids)
    )
    if not requested_rows:
        return {}
    request_indexes = torch.tensor(
        requested_rows,
        dtype=torch.long,
        device=work.device,
    )
    score_rows = output_rows.index_select(0, request_indexes)
    selected_tokens = output_tokens.index_select(0, request_indexes)
    scores = torch.log_softmax(work.index_select(0, score_rows), dim=-1)
    selected_values = scores.gather(1, selected_tokens.unsqueeze(1))[:, 0]
    selected_ranks = (scores > selected_values.unsqueeze(1)).sum(dim=-1, dtype=torch.long) + 1

    counts = tuple(min(max(0, int(rows[index].n_logprobs)), vocab) for index in requested_rows)
    max_count = max(counts, default=0)
    if max_count:
        top_values, top_indexes = torch.topk(scores, max_count, dim=-1, sorted=True)
        positions = torch.arange(
            1,
            max_count + 1,
            dtype=torch.long,
            device=work.device,
        ).unsqueeze(0)
        starts = torch.cat(
            (
                torch.ones(
                    (len(requested_rows), 1),
                    dtype=torch.bool,
                    device=work.device,
                ),
                top_values[:, 1:] < top_values[:, :-1],
            ),
            dim=1,
        )
        top_ranks = torch.where(starts, positions, 0).cummax(dim=1).values
    else:
        top_values = torch.empty(
            (len(requested_rows), 0),
            dtype=scores.dtype,
            device=work.device,
        )
        top_indexes = torch.empty(
            (len(requested_rows), 0),
            dtype=torch.long,
            device=work.device,
        )
        top_ranks = torch.empty_like(top_indexes)

    requested_ids = tuple(
        tuple(
            dict.fromkeys(
                int(value)
                for value in rows[index].parameters.logprob_token_ids
                if 0 <= int(value) < vocab
            )
        )
        for index in requested_rows
    )
    max_requested = max((len(value) for value in requested_ids), default=0)
    if max_requested:
        candidate_indexes = torch.zeros(
            (len(requested_rows), max_requested),
            dtype=torch.long,
            device=work.device,
        )
        for row_index, values in enumerate(requested_ids):
            if values:
                candidate_indexes[row_index, : len(values)] = torch.tensor(
                    values,
                    dtype=torch.long,
                    device=work.device,
                )
        candidate_values = scores.gather(1, candidate_indexes)
        candidate_ranks = torch.stack(
            tuple(
                (scores > candidate_values[:, index].unsqueeze(1)).sum(dim=-1, dtype=torch.long) + 1
                for index in range(max_requested)
            ),
            dim=1,
        )
    else:
        candidate_values = torch.empty(
            (len(requested_rows), 0),
            dtype=scores.dtype,
            device=work.device,
        )
        candidate_ranks = torch.empty(
            (len(requested_rows), 0),
            dtype=torch.long,
            device=work.device,
        )

    selected_value_list = selected_values.cpu().tolist()
    selected_rank_list = selected_ranks.cpu().tolist()
    selected_token_list = selected_tokens.cpu().tolist()
    top_value_list = top_values.cpu().tolist()
    top_index_list = top_indexes.cpu().tolist()
    top_rank_list = top_ranks.cpu().tolist()
    candidate_value_list = candidate_values.cpu().tolist()
    candidate_rank_list = candidate_ranks.cpu().tolist()
    result: dict[int, tuple[float, tuple[tuple[int, float, int], ...]]] = {}
    for local_index, result_index in enumerate(requested_rows):
        selected = int(selected_token_list[local_index])
        selected_value = float(selected_value_list[local_index])
        entries: list[tuple[int, float, int]] = [
            (
                selected,
                selected_value,
                int(selected_rank_list[local_index]),
            )
        ]
        seen = {selected}
        for top_index in range(counts[local_index]):
            candidate = int(top_index_list[local_index][top_index])
            if candidate in seen:
                continue
            entries.append(
                (
                    candidate,
                    float(top_value_list[local_index][top_index]),
                    int(top_rank_list[local_index][top_index]),
                )
            )
            seen.add(candidate)
        for requested_index, candidate in enumerate(requested_ids[local_index]):
            if candidate in seen:
                continue
            entries.append(
                (
                    candidate,
                    float(candidate_value_list[local_index][requested_index]),
                    int(candidate_rank_list[local_index][requested_index]),
                )
            )
            seen.add(candidate)
        result[result_index] = (selected_value, tuple(entries))
    return result


def _score_prompt_token_logprobs(
    logits: torch.Tensor,
    target_token_ids: Sequence[int],
    *,
    n_logprobs: int,
    logprob_token_ids: Sequence[int] = (),
) -> list[list[tuple[int, float, int]]]:
    """Score prompt targets against their explicit left-context logits."""

    if logits.ndim != 2 or not logits.is_floating_point():
        raise invalid_descriptor("prompt-scoring logits must be shaped [positions, vocab]")
    positions, vocab = (int(value) for value in logits.shape)
    targets = tuple(int(value) for value in target_token_ids)
    if len(targets) != positions or any(value < 0 or value >= vocab for value in targets):
        raise invalid_descriptor("prompt-scoring targets do not match the logits vocabulary")
    requested = tuple(
        dict.fromkeys(int(value) for value in logprob_token_ids if 0 <= int(value) < vocab)
    )
    count = min(max(0, int(n_logprobs)), vocab)
    scores = torch.log_softmax(logits.float(), dim=-1)
    result: list[list[tuple[int, float, int]]] = []
    for row, target in enumerate(targets):
        row_scores = scores[row]
        candidates = [target]
        if count:
            candidates.extend(
                int(value) for value in torch.topk(row_scores, count).indices.tolist()
            )
        candidates.extend(requested)
        entries: list[tuple[int, float, int]] = []
        for candidate in dict.fromkeys(candidates):
            value = row_scores[candidate]
            entries.append(
                (candidate, float(value.item()), int((row_scores > value).sum().item()) + 1)
            )
        result.append(entries)
    return result


def _encode_features(output: ForwardRowOutput) -> torch.Tensor:
    if not isinstance(output, EncodeOutput):
        raise invalid_descriptor("encode route did not return encoder features")
    return output.features


def _decoded_tensor(output: ForwardRowOutput) -> torch.Tensor:
    if not isinstance(output, DecodeOutput):
        raise invalid_descriptor("decode route did not return an image tensor")
    return output.tensor


def _positions_as_three_axis(positions: torch.Tensor, query: int) -> torch.Tensor:
    if positions.ndim == 1 and int(positions.numel()) == query:
        return torch.stack((positions, torch.zeros_like(positions), torch.zeros_like(positions)))
    if positions.ndim == 2 and tuple(positions.shape) == (3, query):
        return positions
    raise invalid_descriptor("state positions do not align with their physical token row")


def _stable_handle(session_id: int, epoch: int, op_id: int, role: str) -> int:
    digest = hashlib.sha256(b"uniserve-product-handle\0")
    for value in (session_id, epoch, op_id):
        digest.update(int(value).to_bytes(8, "little", signed=False))
    digest.update(role.encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "little") or 1


def _torch_dtype(name: str) -> torch.dtype:
    value = getattr(torch, str(name).removeprefix("torch."), None)
    if not isinstance(value, torch.dtype):
        raise invalid_descriptor(f"unsupported route dtype {name!r}")
    return value


__all__ = ["ModelExecutor"]
