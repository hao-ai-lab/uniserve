"""Transactional lowering and postprocessing for the canonical execution batch."""

from __future__ import annotations

import hashlib
import math
import time
from collections import defaultdict
from collections.abc import Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
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
)
from uniserve_worker.forward import (
    EncodeKind as ForwardEncodeKind,
)
from uniserve_worker.foundation.errors import (
    capability_mismatch,
    invalid_descriptor,
    unsupported_operation,
)
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
from uniserve_worker.runtime.request_session import RequestSession, SessionStore, StepTxn
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
    promote: bool = False
    temporary_generation: int | None = None

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


_TaskResult: TypeAlias = tuple[ForwardRowOutput, ...]
_Driver: TypeAlias = Generator[tuple[_ForwardTask, ...], _TaskResult, ResultDelta]


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
            drivers = tuple(self._driver(operation, scope) for operation in batch.operations)
            deltas = self._drive(drivers, scope)
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
                forward_stats=_forward_stats(scope.observations),
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

    def _drive(
        self, drivers: tuple[_Driver, ...], scope: _ExecutionScope
    ) -> tuple[ResultDelta, ...]:
        active: dict[int, tuple[_Driver, tuple[_ForwardTask, ...]]] = {}
        completed: dict[int, ResultDelta] = {}
        for index, driver in enumerate(drivers):
            try:
                active[index] = (driver, next(driver))
            except StopIteration as done:
                completed[index] = done.value
        while active:
            flat: list[tuple[int, int, _ForwardTask]] = []
            for driver_index, (_driver, tasks) in active.items():
                for task_index, task in enumerate(tasks):
                    flat.append((driver_index, task_index, task))
            if not flat:
                raise RuntimeError("execution driver yielded an empty forward wave")
            outputs = self._run_wave(tuple(task for _driver, _task, task in flat), scope)
            by_driver: dict[int, list[ForwardRowOutput | None]] = {
                index: [None] * len(tasks) for index, (_driver, tasks) in active.items()
            }
            for (driver_index, task_index, _task), output in zip(flat, outputs, strict=True):
                by_driver[driver_index][task_index] = output
            next_active: dict[int, tuple[_Driver, tuple[_ForwardTask, ...]]] = {}
            for driver_index, (driver, _tasks) in active.items():
                aligned = tuple(cast(ForwardRowOutput, value) for value in by_driver[driver_index])
                try:
                    next_active[driver_index] = (driver, driver.send(aligned))
                except StopIteration as done:
                    completed[driver_index] = done.value
            active = next_active
        return tuple(completed[index] for index in range(len(drivers)))

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
        if kv_tasks:
            kv_view, attention = self._attention_plan(tasks, scope, torch.device(device))
        else:
            kv_view = EmptyKvView()
            attention = NoAttention(backends=self._attention_selection())
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
        graph_shape = self._graph_shape(tasks)
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
    ) -> tuple[KvView, PagedDecodePlan | PagedVarlenPlan | PackedAttentionPlan]:
        route = tasks[0].route
        if RouteRowKind.FLOW in route.row_kinds:
            return self._packed_attention_plan(tasks, scope, device)
        sessions = tuple(task.envelope.session_id for task in tasks)
        query_lens = tuple(task.query_tokens for task in tasks)
        view = scope.kv.view(sessions, query_lens=query_lens)
        block_table = view.block_table(device)
        cache_seqlens = view.cache_seqlens(device)
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
            page_ids = torch.tensor(
                [
                    task.entry.block_ids[task.entry.length // view.block_size]
                    for task in tasks
                    if task.entry is not None
                ],
                dtype=torch.int32,
                device=device,
            )
            page_offsets = torch.tensor(
                [cast(KvEntry, task.entry).length % view.block_size for task in tasks],
                dtype=torch.int32,
                device=device,
            )
            decode_attention = PagedDecodePlan(
                backends=self._attention_selection(),
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                kv_seqlens=torch.tensor(kv_lens, dtype=torch.int32, device=device),
                query_lens=torch.ones(len(tasks), dtype=torch.int32, device=device),
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
        cu_q = _cumulative(query_lens, device)
        cu_k = _cumulative(kv_lens, device)
        varlen_attention = PagedVarlenPlan(
            backends=self._attention_selection(),
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            query_lens=torch.tensor(query_lens, dtype=torch.int32, device=device),
            kv_seqlens=torch.tensor(kv_lens, dtype=torch.int32, device=device),
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
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
    ) -> tuple[KvView, PackedAttentionPlan]:
        uses_scratch = any(task.scratch for task in tasks)
        if uses_scratch:
            for task in tasks:
                if task.scratch:
                    continue
                source = self.kv.get(task.envelope.session_id)
                generation = -int(task.envelope.op_id)
                entry, _created = scope.kv.scratch_entry(
                    task.envelope.session_id,
                    generation,
                    f"mixed-{task.row.row_id}",
                    capacity_tokens=source.length + task.query_tokens,
                    copy_conditioning=True,
                )
                task.entry = entry
                task.scratch = True
                task.promote = True
                task.temporary_generation = generation
                scope.kv.release_generation(task.envelope.session_id, generation)
        rows = tuple(
            (cast(KvEntry, task.entry), task.query_tokens, task.write_kv) for task in tasks
        )
        view = scope.kv.packed_view(rows, scratch=uses_scratch)
        query_lens = tuple(task.query_tokens for task in tasks)
        base_lens = view.base_lens
        key_lens = tuple(base + query for base, query in zip(base_lens, query_lens, strict=True))
        max_query = max(query_lens)
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
            cu_seqlens_q=_cumulative(query_lens, device),
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

    def _graph_shape(self, tasks: tuple[_ForwardTask, ...]) -> tuple[int, ...]:
        counts = [0, 0, 0, 0]
        tokens: list[int] = []
        order = (RouteRowKind.TOKEN, RouteRowKind.FLOW, RouteRowKind.ENCODE, RouteRowKind.DECODE)
        for task in tasks:
            counts[order.index(task.row_kind)] += 1
            tokens.append(task.query_tokens or _row_element_count(task.row))
        return (*counts, *tokens)

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
            sample = self._sample(
                logits.reshape(-1, logits.shape[-1])[-1],
                session,
                operation,
                position=operation.position[1],
            )
            session.last_sampled_token = sample.token_id
            session.rng_counter += 1
            return SequenceDelta(
                SequenceEffect(
                    sampled_token_ids=(sample.token_id,),
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
        wants_prompt = bool(
            inputs.return_all_logits or sampling.return_prompt_logprobs
        )
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
        sample = self._sample(logits[-1], session, operation, position=end)
        session.last_sampled_token = sample.token_id
        session.rng_counter += 1
        effect = SequenceEffect(
            sampled_token_ids=(sample.token_id,),
            sampled_logprob=sample.logprob,
            top_logprobs=_token_logprobs(sample.top_logprobs),
            prompt_logprobs=prompt,
            kv_tokens=self.kv.get(envelope.session_id).length,
            published_kv=self._publish_kv_if_requested(
                envelope,
                operation,
                (sample.token_id,),
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
        samples: list[Any] = []
        stop = frozenset(int(value) for value in inputs.stop_token_ids)
        stopped_at: int | None = None
        for index in range(int(inputs.burst_tokens)):
            task = self._token_task(
                envelope,
                (current,),
                (start + index,),
                TokenSelection.LAST_LOGITS,
                scope,
            )
            outputs = yield (task,)
            logits = _token_logits(outputs[0])[-1]
            self._commit_task_kv(task, 1, scope)
            if self.defer_sampling and inputs.burst_tokens == 1 and not stop:
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
            sampled = self._sample(
                logits,
                session,
                operation,
                position=start + index + 1,
                generated=tuple(value.token_id for value in samples),
            )
            samples.append(sampled)
            session.rng_counter += 1
            current = sampled.token_id
            if stopped_at is None and sampled.token_id in stop:
                stopped_at = len(samples) - 1
                if not inputs.stop_terminal:
                    tail = self._token_task(
                        envelope,
                        (sampled.token_id,),
                        (start + index + 1,),
                        TokenSelection.LAST_LOGITS,
                        scope,
                    )
                    tail_outputs = yield (tail,)
                    tail_logits = _token_logits(tail_outputs[0])[-1]
                    self._commit_task_kv(tail, 1, scope)
                    self._sample(
                        tail_logits,
                        session,
                        operation,
                        position=start + index + 2,
                        generated=tuple(value.token_id for value in samples),
                    )
                    session.rng_counter += 1
                    break
        reported = samples if stopped_at is None else samples[: stopped_at + 1]
        if not reported:
            raise RuntimeError("decode produced no token result")
        session.last_sampled_token = int(reported[-1].token_id)
        first = reported[0]
        token_ids = tuple(int(value.token_id) for value in reported)
        return SequenceDelta(
            SequenceEffect(
                sampled_token_ids=token_ids,
                sampled_logprob=first.logprob,
                top_logprobs=_token_logprobs(first.top_logprobs),
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
        final_coin = uniform_samples(
            (1,),
            seed=sampling_draw_seed(seed, start + len(draft) + 1),
            device=logits.device,
        )
        sampled = _speculative_sample_target_only(
            logits,
            draft,
            sampling,
            recent=operation.policy.recent_tokens,
            allowed=self._allowed_tokens(session, operation),
            suppress=operation.policy.suppress_tokens,
            uniform_samples=coins,
            uniform_sample_for_final=final_coin,
        )
        committed = 1 + int(sampled.num_accepted_tokens)
        self._commit_task_kv(task, committed, scope)
        sampled_ids = (
            *draft[: sampled.num_accepted_tokens],
            int(sampled.sampled_token_id),
        )
        detail_row = int(sampled.num_accepted_tokens)
        details = _score_prompt_token_logprobs(
            logits[detail_row : detail_row + 1],
            (sampled.sampled_token_id,),
            n_logprobs=int(sampling.n_logprobs),
            logprob_token_ids=sampling.logprob_token_ids,
        )[0]
        session.last_sampled_token = int(sampled.sampled_token_id)
        session.rng_counter += len(draft) + 1
        actual = details[0]
        return SequenceDelta(
            SequenceEffect(
                sampled_token_ids=tuple(sampled_ids),
                sampled_logprob=float(actual[1]),
                top_logprobs=tuple(TokenLogprob(*entry) for entry in details),
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
        token_ids: tuple[int, ...],
        positions: tuple[int, ...],
        selection: TokenSelection,
        scope: _ExecutionScope,
    ) -> _ForwardTask:
        stage = self._primary_stage(envelope.operation_type)
        if stage.row is not RouteRowKind.TOKEN:
            raise invalid_descriptor("sequence operation primary stage is not a token row")
        if len(token_ids) != len(positions) or not token_ids:
            raise invalid_descriptor("token task ids and positions must align")
        row_id = scope.row_id()
        row = TokenRow(
            row_id=row_id,
            inputs=TokenIds(torch.tensor(token_ids, dtype=torch.long)),
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
        if task.promote:
            scope.kv.promote_scratch(task.envelope.session_id, cast(KvEntry, task.entry), count)
        elif task.scratch:
            scope.kv.advance_entry(cast(KvEntry, task.entry), count, scratch=True)
        else:
            scope.kv.advance(task.envelope.session_id, count)

    def _resolve_decode_token(
        self,
        inputs: WireTokenInput,
        session: RequestSession,
    ) -> int:
        if inputs.source is TokenSource.WIRE:
            return int(inputs.token_ids[0])
        if session.last_sampled_token is None:
            raise invalid_descriptor("last-sampled token source has no committed token")
        return int(session.last_sampled_token)

    def _sample(
        self,
        logits: torch.Tensor,
        session: RequestSession,
        operation: SequenceOperation,
        *,
        position: int,
        generated: tuple[int, ...] = (),
    ) -> Any:
        return self._sample_policy(
            logits,
            session,
            operation.policy,
            position=position,
            generated=generated,
        )

    def _sample_policy(
        self,
        logits: torch.Tensor,
        session: RequestSession,
        policy: TokenPolicy,
        *,
        position: int,
        generated: tuple[int, ...] = (),
    ) -> Any:
        sampling = _require_sampling(session)
        generator = torch.Generator(device=logits.device)
        generator.manual_seed(sampling_draw_seed(int(sampling.seed or 0), int(position)))
        return _sample_one_from_logits(
            logits,
            sampling,
            recent=[*policy.recent_tokens, *generated],
            allowed=(policy.allowed_tokens or sampling.allowed_token_ids),
            suppress=policy.suppress_tokens or None,
            n_logprobs=int(sampling.n_logprobs),
            generator=generator,
        )

    @staticmethod
    def _allowed_tokens(
        session: RequestSession,
        operation: SequenceOperation,
    ) -> tuple[int, ...] | None:
        if operation.policy.allowed_tokens:
            return operation.policy.allowed_tokens
        return _require_sampling(session).allowed_token_ids

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
        if not policy.publish_kv and not (set(sampled) & set(policy.publish_kv_on_tokens)):
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
    ) -> Generator[tuple[_ForwardTask, ...], _TaskResult, _StateOutcome]:
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
                    sampled = self._sample_policy(
                        value[-1],
                        session,
                        policy,
                        position=conditioning_position
                        + max(
                            1,
                            1 if flow_spec is None else flow_spec.rope_advance,
                        ),
                    )
                    session.last_sampled_token = int(sampled.token_id)
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


def _cumulative(lengths: Sequence[int], device: torch.device) -> torch.Tensor:
    result = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    if lengths:
        result[1:] = torch.cumsum(
            torch.tensor(tuple(int(value) for value in lengths), dtype=torch.int32, device=device),
            dim=0,
        )
    return result


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


def _forward_stats(observations: Sequence[RunObservation]) -> WorkerForwardStats:
    route_counts: dict[str, int] = {}
    route_rows: dict[str, int] = {}
    route_us: dict[str, int] = {}
    path_counts: dict[str, int] = {}
    captures = 0
    replays = 0
    fallbacks = 0
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
    return WorkerForwardStats(
        mode_counts=route_counts,
        mode_tokens=route_rows,
        mode_us=route_us,
        component_us={"forward": sum(route_us.values())},
        cuda_graph_captures=int(captures),
        cuda_graph_replays=int(replays),
        cuda_graph_misses=int(fallbacks),
        cuda_graph_fallbacks=int(fallbacks),
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


@dataclass(frozen=True, slots=True)
class _TokenSample:
    token_id: int
    logprob: float | None
    top_logprobs: list[list[float | int]] | None


def _sample_one_from_logits(
    logits: torch.Tensor,
    parameters: SamplingParams,
    *,
    recent: Sequence[int],
    allowed: Sequence[int] | None,
    suppress: Sequence[int] | None,
    n_logprobs: int,
    generator: torch.Generator,
) -> _TokenSample:
    """Apply the canonical single-row sampling policy with an explicit RNG."""

    if logits.ndim != 1 or not logits.is_floating_point() or int(logits.numel()) < 1:
        raise invalid_descriptor("sampling logits must be a non-empty floating vector")
    work = logits.float().clone()
    vocab = int(work.numel())
    temperature = _shape_sampling_logits(
        work,
        parameters,
        recent=recent,
        allowed=allowed,
        suppress=suppress,
    )
    if temperature <= 0.0:
        token_id = int(torch.argmax(work).item())
    else:
        token_id = int(
            torch.multinomial(torch.softmax(work, dim=-1), 1, generator=generator).item()
        )
    requested_ids = tuple(
        dict.fromkeys(
            int(value) for value in parameters.logprob_token_ids if 0 <= int(value) < vocab
        )
    )
    count = min(max(0, int(n_logprobs)), vocab)
    wants_logprobs = (
        parameters.return_logprobs or count > 0 or bool(requested_ids)
    )
    if not wants_logprobs:
        return _TokenSample(token_id, None, None)
    logprobs = torch.log_softmax(work, dim=-1)
    selected = float(logprobs[token_id].item())
    entries: list[list[float | int]] = [
        [token_id, selected, int((logprobs > logprobs[token_id]).sum().item()) + 1]
    ]
    seen = {token_id}
    if count:
        values, indexes = torch.topk(logprobs, count)
        for index, value in zip(indexes.tolist(), values.tolist(), strict=True):
            candidate = int(index)
            if candidate not in seen:
                entries.append(
                    [
                        candidate,
                        float(value),
                        int((logprobs > logprobs[candidate]).sum().item()) + 1,
                    ]
                )
                seen.add(candidate)
    for candidate in requested_ids:
        if candidate not in seen:
            entries.append(
                [
                    candidate,
                    float(logprobs[candidate].item()),
                    int((logprobs > logprobs[candidate]).sum().item()) + 1,
                ]
            )
            seen.add(candidate)
    return _TokenSample(token_id, selected, entries)


def _shape_sampling_logits(
    work: torch.Tensor,
    parameters: SamplingParams,
    *,
    recent: Sequence[int],
    allowed: Sequence[int] | None,
    suppress: Sequence[int] | None,
) -> float:
    """Apply the one canonical logits-shaping order in place."""

    vocab = int(work.numel())
    if allowed is not None:
        allowed_indexes = tuple(
            dict.fromkeys(int(value) for value in allowed if 0 <= int(value) < vocab)
        )
        if not allowed_indexes:
            raise invalid_descriptor("sampling allowed-token set has no vocabulary entries")
        selected = torch.tensor(allowed_indexes, dtype=torch.long, device=work.device)
        masked = torch.full_like(work, float("-inf"))
        masked[selected] = work[selected]
        work.copy_(masked)
    if suppress:
        suppressed_indexes = tuple(
            dict.fromkeys(int(value) for value in suppress if 0 <= int(value) < vocab)
        )
        if suppressed_indexes:
            work.index_fill_(
                0,
                torch.tensor(suppressed_indexes, dtype=torch.long, device=work.device),
                float("-inf"),
            )
    for token_id, bias in parameters.logit_bias:
        index = int(token_id)
        if 0 <= index < vocab and not bool(torch.isneginf(work[index])):
            work[index] += float(bias)

    repetition = parameters.repetition_penalty
    frequency = parameters.frequency_penalty
    presence = parameters.presence_penalty
    counts: dict[int, int] = {}
    for value in recent:
        token_id = int(value)
        if 0 <= token_id < vocab:
            counts[token_id] = counts.get(token_id, 0) + 1
    if counts and (repetition != 1.0 or frequency != 0.0 or presence != 0.0):
        recent_indexes = torch.tensor(tuple(counts), dtype=torch.long, device=work.device)
        recent_values = work[recent_indexes]
        if repetition != 1.0:
            recent_values = torch.where(
                recent_values > 0,
                recent_values / repetition,
                recent_values * repetition,
            )
        occurrences = torch.tensor(
            tuple(counts[index] for index in counts), dtype=work.dtype, device=work.device
        )
        adjusted = recent_values - frequency * occurrences - presence
        work[recent_indexes] = torch.where(
            torch.isneginf(recent_values), recent_values, adjusted
        )

    temperature = parameters.temperature
    if temperature > 0.0:
        work.div_(temperature)
    min_p = parameters.min_p
    if min_p > 0.0:
        probabilities = torch.softmax(work, dim=-1)
        work.masked_fill_(probabilities < min_p * probabilities.max(), float("-inf"))
    top_k = parameters.top_k
    if 0 < top_k < vocab:
        top_values, top_indexes = torch.topk(work, top_k, sorted=False)
        masked = torch.full_like(work, float("-inf"))
        masked.scatter_(0, top_indexes, top_values)
        work.copy_(masked)
    top_p = parameters.top_p
    if 0.0 < top_p < 1.0:
        ordered, ordered_indexes = torch.sort(work, descending=True)
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        drop = cumulative > top_p
        drop[1:] = drop[:-1].clone()
        drop[0] = False
        work[ordered_indexes[drop]] = float("-inf")
    if bool(torch.isnan(work).any()) or not bool(torch.isfinite(work).any()):
        raise invalid_descriptor("sampling policy masked every vocabulary entry")
    return temperature


@dataclass(frozen=True, slots=True)
class _SpeculativeSample:
    sampled_token_id: int
    num_accepted_tokens: int


def _speculative_sample_target_only(
    logits: torch.Tensor,
    draft_token_ids: Sequence[int],
    parameters: SamplingParams,
    *,
    recent: Sequence[int],
    allowed: Sequence[int] | None,
    suppress: Sequence[int] | None,
    uniform_samples: torch.Tensor,
    uniform_sample_for_final: torch.Tensor,
) -> _SpeculativeSample:
    """Apply target-only acceptance to one explicit linear draft chain."""

    draft = tuple(int(value) for value in draft_token_ids)
    rows = len(draft) + 1
    if logits.ndim != 2 or int(logits.shape[0]) < rows or int(logits.shape[1]) < 1:
        raise invalid_descriptor("speculative verification logits do not cover the draft chain")
    vocab = int(logits.shape[1])
    if any(value < 0 or value >= vocab for value in draft):
        raise invalid_descriptor("speculative draft token is outside the model vocabulary")
    if tuple(uniform_samples.shape) != (len(draft),):
        raise invalid_descriptor("speculative acceptance RNG shape does not match the draft")
    if tuple(uniform_sample_for_final.shape) != (1,):
        raise invalid_descriptor("speculative final RNG must contain exactly one coordinate")

    work = logits[:rows].float().clone()
    for row in work:
        _shape_sampling_logits(
            row,
            parameters,
            recent=recent,
            allowed=allowed,
            suppress=suppress,
        )
    probabilities = torch.softmax(work, dim=-1)
    accepted = 0
    for index, token_id in enumerate(draft):
        target_probability = probabilities[index, token_id]
        coin = uniform_samples[index].to(
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        if bool((coin <= target_probability).item()) or bool((target_probability >= 1.0).item()):
            accepted += 1
            continue
        break
    sample_probabilities = probabilities[accepted].clone()
    if accepted < len(draft):
        sample_probabilities[draft[accepted]] = 0.0
    total = sample_probabilities.sum()
    if not bool((total > 0).item()):
        sample_probabilities = probabilities[accepted]
        total = sample_probabilities.sum()
    coin = (
        uniform_sample_for_final[0]
        .to(
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        .clamp(0.0, 1.0)
    )
    token = torch.searchsorted(
        torch.cumsum(sample_probabilities, dim=-1),
        coin * total,
        right=False,
    ).clamp(max=vocab - 1)
    return _SpeculativeSample(int(token.item()), accepted)


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
