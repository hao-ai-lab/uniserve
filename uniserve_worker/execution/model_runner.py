"""Candidate preparation and resource-specific publication for execution batches."""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from importlib import import_module
from typing import Any, TypeAlias, cast

import torch

from uniserve_worker.batch import (
    Batch,
    BatchPartition,
    CompletionRecord,
    CompletionReport,
    DeviceDim,
    DevicePoint,
    DrawLayout,
    DType,
    EncodeMode,
    ExecutionCapability,
    FinishFlags,
    FixedPoint,
    GenMode,
    ImageParams,
    LatentPlacement,
    LogicalLengths,
    Operation,
    OpStatus,
    PartitionCompletion,
    ProductKind,
    ProductPayload,
    ProductRef,
    RegistrationAck,
    Release,
    RequestKey,
    SamplingParams,
    SamplingState,
    ShapeBound,
    StaticDim,
    StorageClass,
    TimingCounters,
    TokenMode,
    TokenSpan,
    TransferMode,
    VersionRef,
    WorkerForwardStats,
    WorkVariant,
    decode_sampling_state_bytes,
    decode_token_product_bytes,
)
from uniserve_worker.batch import (
    ErrorCode as ProtocolErrorCode,
)
from uniserve_worker.capabilities import MixedExecutionCapability
from uniserve_worker.execution.forward_batch import (
    AttentionSelection,
    AttnPlan,
    EmptyKvView,
    ExpertRoute,
    FlowPatches,
    KvView,
    ModelPhase,
    NoAttention,
    PackedAttentionPlan,
    PagedDecodePlan,
    PagedVarlenPlan,
    RouteMeshView,
    RouteSpan,
    TokenSelection,
    packed_tensor_views,
)
from uniserve_worker.execution.rng import (
    DRAW_LAYOUT_TARGET,
    flow_noise_seed,
    normal_noise,
    sampling_key,
    sampling_uniform,
)
from uniserve_worker.execution.trace import (
    ExecutionPhase,
    ExecutionTrace,
    OperationTrace,
)
from uniserve_worker.foundation.device import canonical_device
from uniserve_worker.foundation.errors import (
    ErrorCode as WorkerErrorCode,
)
from uniserve_worker.foundation.errors import (
    WorkerError,
    capability_mismatch,
    classify,
    invalid_descriptor,
    should_capture_trace,
    unsupported_operation,
)
from uniserve_worker.foundation.profiling import profile_range
from uniserve_worker.foundation.sizing import bucketed_length
from uniserve_worker.foundation.triton_compat import triton_device_supported
from uniserve_worker.loader.weight_set import WeightSet
from uniserve_worker.models.generation import (
    BranchSource,
    GenerationPipeline,
    LatentLayout,
    Materialization,
)
from uniserve_worker.models.inputs import FeatureLayout, ImageProcessor, PatchTransform
from uniserve_worker.models.runtime import (
    ExecutionModel,
    PositionLayout,
    WorkerDeployment,
)
from uniserve_worker.nn.diffusion.cfg import Branch, build_flow_cfg_plan
from uniserve_worker.nn.diffusion.integrator import euler_step
from uniserve_worker.nn.diffusion.schedule import (
    x_pred_to_velocity,
)
from uniserve_worker.nn.mesh import BroadcastTransport, DeviceMesh
from uniserve_worker.nn.vision import get_flattened_position_ids_extrapolate
from uniserve_worker.runtime.cache_pool import CacheBatchView, CachePool, CacheRow
from uniserve_worker.runtime.device_products import (
    DeviceProductMetadata,
    DeviceProductRead,
    DeviceProducts,
    DeviceProductScalarBatch,
    DeviceProductWrite,
    ImageRange,
    device_product_storage,
)
from uniserve_worker.runtime.encoder_cache import (
    EncoderCache,
    EncoderMetadata,
    EncoderRead,
    EncoderWrite,
)
from uniserve_worker.runtime.latent_pool import (
    LatentPool,
    LatentPublication,
    LatentRelease,
    LatentSnapshot,
    LatentStaging,
)
from uniserve_worker.runtime.runtime_states import RuntimeStates
from uniserve_worker.server.completion import (
    CompletionArena,
    CompletionCapture,
    CompletionLease,
    _CompletionDerivedInteger,
    _CompletionImagePayload,
    _CompletionInteger,
    _CompletionLogprobBatch,
    _CompletionLogprobPayload,
    _CompletionLogprobValue,
    _CompletionSampleSpan,
    _CompletionSampleToken,
    _CompletionSpeculativePoint,
    _CompletionSpeculativeTokens,
    _CompletionToken,
    _CompletionTopLogprobs,
    _CompletionTransferPayload,
    _PendingDigest,
    _PendingErrorDigest,
)
from uniserve_worker.server.cpu_tasks import BoundedCpuTaskPool, CpuTaskReservation
from uniserve_worker.server.image_codec import (
    quantize_image_hwc,
)
from uniserve_worker.server.request_state import (
    RequestRow,
    RequestRuntime,
    RequestTable,
)
from uniserve_worker.transfer.connector import CachePublication, CachePublications
from uniserve_worker.transfer.tickets import (
    TRANSFER_DESCRIPTOR_PREFIX,
    Locator,
    TransferTicket,
    Transport,
    decode_transfer_descriptor,
)

from ._inputs import (
    PreparedImage,
    patch_grid_shape,
    prepare_image,
    prepare_tensor_image,
)
from .cuda_graph import GraphExecutionError
from .model_invocation import RunObservation, RunPath, _ModelInvocation

logger = logging.getLogger(__name__)

SAMPLING_COMPLETION_FIELDS = 4
TOKEN_CONTINUATION_BIT = 1 << 31
TOKEN_VALUE_MASK = TOKEN_CONTINUATION_BIT - 1
MIXED_SERVICE_SERIAL_NUMERATOR = 5
MIXED_SERVICE_SERIAL_DENOMINATOR = 4


@dataclass(slots=True)
class _ForwardTask:
    operation: Operation
    request: RequestRow
    weights: WeightSet
    phase: ModelPhase
    token_ids: torch.Tensor | None = None
    token_embeddings: torch.Tensor | None = None
    token_embedding_mask: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    selection: TokenSelection | None = None
    flow_conditioning: FlowPatches | None = None
    timestep: torch.Tensor | None = None
    latent: torch.Tensor | None = None
    image_tokens: int = 0
    image_height: int = 0
    image_width: int = 0
    encode_pixels: torch.Tensor | None = None
    encode_grid: torch.Tensor | None = None
    encode_grid_shape: tuple[int, int] | None = None
    entry: CacheRow | None = None
    scratch: bool = False
    write_kv: bool = False
    causal: bool = True
    attention_indexes: torch.Tensor | None = None
    text_local_indices: tuple[int, ...] = ()
    request_pool_index: torch.Tensor | None = None

    @property
    def query_tokens(self) -> int:
        if self.token_ids is not None:
            return int(self.token_ids.numel())
        if self.latent is not None and self.image_tokens > 0:
            return int(self.image_tokens)
        return 0

    @property
    def kind(self) -> str:
        if self.token_ids is not None:
            return "token"
        if self.latent is not None and self.image_tokens > 0:
            return "flow"
        if self.encode_pixels is not None:
            return "encode"
        return "decode"


@dataclass(frozen=True, slots=True)
class _SamplingRow:
    parameters: SamplingParams
    # Dense per-vocabulary count of generated tokens preceding this point, read
    # from the request's device-resident committed penalty base plus any draft
    # prefix. ``None`` when the row uses no penalties.
    penalty_counts: torch.Tensor | None
    allowed: tuple[int, ...] | None
    suppress: tuple[int, ...]
    draw: float
    n_logprobs: int
    finish_token_ids: tuple[int, ...] = ()
    transition_token_ids: tuple[int, ...] = ()
    force_finish: bool = False


@dataclass(frozen=True, slots=True)
class _SampleTask:
    operation: Operation
    logits: torch.Tensor
    rows: tuple[_SamplingRow, ...]
    draws: torch.Tensor | None
    penalty_token_ids: torch.Tensor | None
    penalty_counts: torch.Tensor | None
    parameter_values: torch.Tensor | None
    draft_token_ids: tuple[int, ...] = ()
    terminal_draft_prefix: int | None = None
    token_product: DeviceProductWrite | None = None
    finish_product: DeviceProductWrite | None = None
    transition_product: DeviceProductWrite | None = None
    predicate: torch.Tensor | None = None
    tagged_predicate: bool = False
    request_pool_index: torch.Tensor | None = None
    # The request's device-resident committed penalty base to fold this
    # operation's selected token into after sampling. ``None`` when the request
    # uses no penalties.
    penalty_base: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class _SampleResult:
    token_id: int | _CompletionToken
    device_token: torch.Tensor | None
    logprob: float | _CompletionLogprobValue | None
    top_logprobs: tuple[tuple[int, float, int], ...] | _CompletionTopLogprobs | None
    num_accepted_tokens: int | _CompletionInteger = 0
    device_accepted_tokens: torch.Tensor | None = None
    device_selected_point: torch.Tensor | None = None
    device_valid: torch.Tensor | None = None
    device_active: torch.Tensor | None = None
    prompt_logprobs: tuple[
        tuple[tuple[int, float, int], ...] | _CompletionTopLogprobs,
        ...,
    ] = ()
    device_finish: torch.Tensor | None = None
    device_continuation: torch.Tensor | None = None
    device_product_published: bool = False


@dataclass(frozen=True, slots=True)
class _RuntimePublication:
    slot: int
    token: torch.Tensor
    predicate: torch.Tensor
    selected_point: torch.Tensor
    logical_position: int | torch.Tensor
    sampling_position: int | torch.Tensor
    penalty_base: torch.Tensor | None
    valid: torch.Tensor
    active: torch.Tensor


@dataclass(frozen=True, slots=True)
class _DecodeRuntimePublication:
    slots: tuple[int, ...]
    device_slots: torch.Tensor
    tokens: torch.Tensor
    predicates: torch.Tensor
    selected_points: torch.Tensor | None
    penalty_bases: tuple[torch.Tensor | None, ...]
    valid: torch.Tensor
    active: torch.Tensor


@dataclass(frozen=True, slots=True)
class _PromptLogitsPublication:
    slot: int
    logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class _PreparedTransferInput:
    product: ProductRef
    kind: str
    producer_plan_digest: str
    locators: tuple[Locator, ...]
    tickets: tuple[TransferTicket, ...]
    payload_kind: ProductKind | None
    height: int | None
    width: int | None
    latent_units: int | None
    step: int | None
    generation: int | None
    device_metadata: DeviceProductMetadata | None
    snapshot: CachePublication | None

    def ready(self) -> bool:
        return all(ticket.ready() for ticket in self.tickets)

    def tensors(self) -> tuple[torch.Tensor, ...]:
        if not self.ready():
            raise RuntimeError("prepared transfer input was observed before readiness")
        tensors = tuple(ticket.result() for ticket in self.tickets)
        for locator, tensor in zip(self.locators, tensors, strict=True):
            if (
                tuple(int(value) for value in tensor.shape) != locator.shape
                or str(tensor.dtype).removeprefix("torch.") != locator.dtype
                or int(tensor.numel()) * int(tensor.element_size()) != int(locator.nbytes)
            ):
                raise invalid_descriptor("transport result disagrees with its exact locator")
        return tensors


@dataclass(slots=True)
class _PreparedPredicateBatch:
    lease: CompletionLease
    entries: list[tuple[_OperationIdentity, CompletionCapture, int]]
    transferred: tuple[tuple[_OperationIdentity, _PreparedTransferInput, int], ...]
    sealed: bool
    _values: dict[_OperationIdentity, bool] | None = None

    def ready(self) -> bool:
        if self._values is not None:
            return True
        if not self.sealed:
            if not all(transfer.ready() for _, transfer, _ in self.transferred):
                return False
            try:
                for identity, transfer, row in self.transferred:
                    tensors = transfer.tensors()
                    if len(tensors) != 1:
                        raise invalid_descriptor(
                            "transferred operation predicate has an invalid tensor set"
                        )
                    self.entries.append((identity, self.lease.capture(tensors[0]), row))
                self.lease.seal()
            except BaseException:
                self.lease.abandon()
                raise
            self.sealed = True
        return self.lease.ready()

    def resolve(self) -> dict[_OperationIdentity, bool]:
        if self._values is not None:
            return self._values
        if not self.lease.ready():
            raise RuntimeError("prepared predicates were observed before readiness")
        values: dict[_OperationIdentity, bool] = {}
        generation = self.lease.generation
        try:
            for identity, capture, row in sorted(self.entries, key=lambda entry: entry[2]):
                captured = capture.values()
                if len(captured) != 1 or captured[0] not in {0, 1}:
                    raise invalid_descriptor("operation predicate is not a canonical boolean")
                values[identity] = bool(captured[0])
                self.lease.observe(row, generation)
        except BaseException:
            self.lease.abandon()
            raise
        self._values = values
        return values

    def abandon(self) -> None:
        if self._values is None:
            self.lease.abandon()

    def __del__(self) -> None:
        try:
            self.abandon()
        except Exception:
            pass


@dataclass(frozen=True, slots=True)
class PreparedExecution:
    batch: Batch
    transfers: tuple[_PreparedTransferInput, ...]
    predicates: _PreparedPredicateBatch | None = None

    def ready(self) -> bool:
        return all(transfer.ready() for transfer in self.transfers) and (
            self.predicates is None or self.predicates.ready()
        )

    def predicate_values(self) -> dict[_OperationIdentity, bool]:
        return {} if self.predicates is None else self.predicates.resolve()


_ModelTask: TypeAlias = _ForwardTask | _SampleTask
_TaskResult: TypeAlias = tuple[Any, ...]
_Driver: TypeAlias = Generator[tuple[_ModelTask, ...], _TaskResult, "_Outcome"]
_OperationIdentity: TypeAlias = tuple[RequestKey, int]


def _operation_identity(operation: Operation) -> _OperationIdentity:
    return operation.request_key, int(operation.op_id)


def _reference_operation_identity(reference: ProductRef) -> _OperationIdentity:
    return reference.request_key, int(reference.producer_op_id)


def _unique_scopes(scopes: Sequence[_ExecutionScope]) -> tuple[_ExecutionScope, ...]:
    unique: list[_ExecutionScope] = []
    seen: set[int] = set()
    for scope in scopes:
        identity = id(scope)
        if identity not in seen:
            seen.add(identity)
            unique.append(scope)
    return tuple(unique)


def _protocol_error_code(code: str) -> ProtocolErrorCode:
    if code == WorkerErrorCode.RESOURCE_ERROR:
        return ProtocolErrorCode.RESOURCE_EXHAUSTED
    if code == WorkerErrorCode.COMPUTE_ERROR:
        return ProtocolErrorCode.COMPUTE_ERROR
    if code in {WorkerErrorCode.INVARIANT_VIOLATION, WorkerErrorCode.FATAL_WORKER_FAILURE}:
        return ProtocolErrorCode.INTERNAL
    return ProtocolErrorCode.INVALID_OPERATION


@dataclass(frozen=True, slots=True)
class _PartitionLayout:
    operations: tuple[Operation, ...]
    requests: tuple[RequestRow, ...]
    cache_rows: tuple[CacheRow | None, ...]
    weights: tuple[WeightSet, ...]
    identities: tuple[_OperationIdentity, ...]

    def __post_init__(self) -> None:
        width = len(self.operations)
        if not all(
            len(values) == width
            for values in (
                self.requests,
                self.cache_rows,
                self.weights,
                self.identities,
            )
        ):
            raise RuntimeError("partition layout columns are not aligned")


@dataclass(frozen=True, slots=True)
class _LatentExecution:
    placement: LatentPlacement
    request_pool_idx: int
    staging: LatentStaging


@dataclass(slots=True)
class _ExecutionScope:
    partition: BatchPartition
    started_ns: int
    graph_eligible: bool
    request_candidates: tuple[RequestRow, ...]
    request_bases: tuple[RequestRow | None, ...]
    request_rows: dict[int, RequestRow]
    completion: CompletionLease
    input_tokens: dict[ProductRef, tuple[int, ...]] = field(default_factory=dict)
    input_images: dict[ProductRef, str] = field(default_factory=dict)
    cache_rows: dict[tuple[RequestKey, int, int], CacheRow] = field(default_factory=dict)
    branch_rows: dict[tuple[RequestKey, int, int, int], CacheRow] = field(default_factory=dict)
    branch_publications: dict[tuple[RequestKey, int, int], CacheRow] = field(default_factory=dict)
    layout: _PartitionLayout | None = None
    prepared_transfers: dict[ProductRef, _PreparedTransferInput] = field(default_factory=dict)
    transferred_device_products: dict[ProductRef, DeviceProductWrite] = field(default_factory=dict)
    transferred_encoder_features: dict[ProductRef, EncoderWrite] = field(default_factory=dict)
    cache_publication_inputs: dict[ProductRef, CachePublication] = field(default_factory=dict)
    cache_publications: list[tuple[ProductRef, CachePublication]] = field(default_factory=list)
    cache_installations: list[tuple[ProductRef, ProductRef, CachePublication]] = field(
        default_factory=list
    )
    stage_publications: dict[_OperationIdentity, tuple[Locator, ...]] = field(default_factory=dict)
    published: list[Locator] = field(default_factory=list)
    observations: list[RunObservation] = field(default_factory=list)
    component_us: dict[str, int] = field(default_factory=dict)
    device_reads: list[DeviceProductRead] = field(default_factory=list)
    device_writes: list[DeviceProductWrite] = field(default_factory=list)
    encoder_reads: list[EncoderRead] = field(default_factory=list)
    encoder_writes: list[EncoderWrite] = field(default_factory=list)
    operation_writes: dict[_OperationIdentity, DeviceProductWrite] = field(default_factory=dict)
    token_writes: dict[_OperationIdentity, DeviceProductWrite] = field(default_factory=dict)
    selected_point_writes: dict[_OperationIdentity, DeviceProductWrite] = field(
        default_factory=dict
    )
    accepted_span_writes: dict[_OperationIdentity, DeviceProductWrite] = field(default_factory=dict)
    state_continuation_writes: dict[_OperationIdentity, DeviceProductWrite] = field(
        default_factory=dict
    )
    finish_writes: dict[_OperationIdentity, DeviceProductWrite] = field(default_factory=dict)
    transition_writes: dict[_OperationIdentity, DeviceProductWrite] = field(default_factory=dict)
    propagated_predicate_writes: dict[_OperationIdentity, tuple[DeviceProductWrite, ...]] = field(
        default_factory=dict
    )
    predicate_values: dict[_OperationIdentity, tuple[torch.Tensor, bool]] = field(
        default_factory=dict
    )
    predicated_operations: frozenset[_OperationIdentity] = frozenset()
    sampling_states: dict[_OperationIdentity, SamplingState] = field(default_factory=dict)
    runtime_publications: list[_RuntimePublication | _DecodeRuntimePublication] = field(
        default_factory=list
    )
    prompt_logits_publications: list[_PromptLogitsPublication] = field(default_factory=list)
    runtime_cache_lengths: dict[int, int | torch.Tensor] = field(default_factory=dict)
    registration_visible: bool = False
    cpu_tasks: dict[_OperationIdentity, CpuTaskReservation] = field(default_factory=dict)
    latent_rows: dict[_OperationIdentity, _LatentExecution] = field(default_factory=dict)
    latent_publications: list[LatentPublication] = field(default_factory=list)
    latent_releases: list[LatentRelease] = field(default_factory=list)
    latent_import_slots: list[int] = field(default_factory=list)
    publication_started: bool = False


@dataclass(frozen=True, slots=True)
class _SpeculativeSelection:
    accepted: _CompletionInteger
    selected_point: _CompletionSpeculativePoint
    draft_tokens: tuple[int, ...]
    terminal_prefix: int | None
    base_logical_position: int
    base_rng_counter: int
    base_kv_visible: int
    initialized_kv: int


@dataclass(frozen=True, slots=True)
class _Outcome:
    """The selected result of one operation, projected onto its completion record.

    ``committed_tokens`` are the semantic tokens the sampler selected and copied
    to completion storage for the semantic digest. ``products`` are the
    operation's non-semantic host-facing payloads (a materialized image artifact,
    requested logprobs) carried by the completion report under their product
    references.
    """

    status: OpStatus
    selected_point: int | _CompletionSpeculativePoint
    logical_lengths: LogicalLengths
    token_span: TokenSpan
    finish_flags: FinishFlags
    product_generations: tuple[int, ...]
    committed_tokens: tuple[int | _CompletionToken, ...] = ()
    products: tuple[ProductPayload, ...] = ()
    selection: _SpeculativeSelection | None = None
    completion_tasks: tuple[_CompletionImagePayload, ...] = ()


@dataclass(frozen=True, slots=True)
class _StateOutcome:
    """Semantic tokens and products committed while publishing image state."""

    committed_tokens: tuple[int | _CompletionToken, ...] = ()
    products: tuple[ProductPayload, ...] = ()

    @property
    def sampled_tokens(self) -> int:
        return len(self.committed_tokens)


class ModelRunner:
    """Own one typed operation step from validation through atomic publication."""

    def __init__(
        self,
        *,
        model: ExecutionModel,
        deployment: WorkerDeployment,
        model_invocation: _ModelInvocation,
        attention: AttentionSelection,
        requests: RequestTable,
        runtime_states: RuntimeStates,
        cache_pool: CachePool,
        latent_pool: LatentPool | None,
        device_products: DeviceProducts,
        encoder_cache: EncoderCache,
        completion_arena: CompletionArena,
        cpu_tasks: BoundedCpuTaskPool,
        weights: WeightSet,
        mesh: DeviceMesh,
        transport: Transport | None,
        tokenizer: Any | None,
        architecture_digest: str,
        weight_digest: str,
        allowed_work_variants: frozenset[WorkVariant],
        mixed_buckets: tuple[MixedExecutionCapability, ...],
        trace: ExecutionTrace,
    ) -> None:
        if not allowed_work_variants:
            raise ValueError("model runner must accept at least one work variant")
        if len(architecture_digest) != 64:
            raise capability_mismatch("model runner identity is invalid")
        if weights.digest != weight_digest:
            raise capability_mismatch(
                "model runner base-weight identity does not match its weight set"
            )
        unsupported = allowed_work_variants - model.supported_work
        if unsupported:
            raise capability_mismatch(
                "model runner work set exceeds the model implementation: "
                f"{sorted(value.value for value in unsupported)!r}"
            )
        self.model = model
        self.deployment = deployment
        self._model_invocation = model_invocation
        self.attention = attention
        self.requests = requests
        self.runtime_states = runtime_states
        self.cache_pool = cache_pool
        self.cache_publications = CachePublications(cache_pool)
        self.latent_pool = latent_pool
        self.device_products = device_products
        self.encoder_cache = encoder_cache
        self._completions = completion_arena
        self._cpu_tasks = cpu_tasks
        self.weights = weights
        self.mesh = mesh
        self.transport = transport
        self.tokenizer = tokenizer
        self.architecture_digest = architecture_digest
        self.weight_digest = weight_digest
        self.allowed_work_variants = allowed_work_variants
        self.mixed_buckets = frozenset(mixed_buckets)
        self._qualified_mixed_buckets: set[MixedExecutionCapability] = set()
        self.trace = trace
        self._device = canonical_device(deployment.device)
        self._generation_device = (
            self._device
            if deployment.generation_device is None
            else canonical_device(deployment.generation_device)
        )
        self._collective_history: OrderedDict[int, str] = OrderedDict()
        self._transport_publications: dict[_OperationIdentity, tuple[Locator, ...]] = {}
        self._branch_cache_rows: dict[tuple[RequestKey, int, int], CacheRow] = {}

    def close(self) -> None:
        self._collective_history.clear()
        self._transport_publications.clear()
        self._qualified_mixed_buckets.clear()
        self._model_invocation.close()

    def synchronize(self) -> None:
        self._model_invocation.synchronize()

    def prepare(self, batch: Batch) -> PreparedExecution | None:
        """Submit bounded transfer and predicate observations without waiting."""

        entries = tuple(
            payload
            for payload in batch.input_products
            if payload.payload.startswith(TRANSFER_DESCRIPTOR_PREFIX)
        )
        transport = self.transport
        if entries and transport is None:
            raise capability_mismatch("cross-stage input requires a configured transport")
        transfers: list[_PreparedTransferInput] = []
        for entry in entries:
            assert transport is not None
            kind, value, producer_plan_digest = decode_transfer_descriptor(entry.payload)
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
                main = Locator.from_wire(raw_locator)
                locators = (main,)
                raw_payload_kind = value["payload_kind"]
                height = _metadata_uint(value, "height", 0)
                width = _metadata_uint(value, "width", 0)
                generation = _metadata_uint(value, "generation", 0)
                if (
                    not isinstance(raw_payload_kind, str)
                    or raw_payload_kind
                    not in {ProductKind.VISION_FEATURE.value, ProductKind.LATENT_FEATURE.value}
                    or min(height, width, generation) < 1
                    or generation != int(entry.product.generation)
                    or not _locator_matches_product(main, entry.product)
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
                main = Locator.from_wire(raw_locator)
                locators = (main,)
                generation = _metadata_uint(value, "generation", 0)
                height = _metadata_uint(value, "height", 0)
                width = _metadata_uint(value, "width", 0)
                raw_range = _metadata_string(value, "value_range", "")
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
                    or not _requires_device_product_binding(entry.product)
                    or not _locator_matches_product(main, entry.product)
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
                main = Locator.from_wire(raw_locator)
                locators = (main,)
                height = _metadata_uint(value, "height", 0)
                width = _metadata_uint(value, "width", 0)
                latent_units = _metadata_uint(value, "latent_units", 0)
                step = _metadata_uint(value, "step", 0)
                generation = _metadata_uint(value, "generation", 0)
                pool = self.latent_pool
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
                generation = _metadata_uint(value, "generation", 0)
                snapshot = CachePublication.from_wire(value["snapshot"])
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
                _PreparedTransferInput(
                    product=entry.product,
                    kind=kind,
                    producer_plan_digest=producer_plan_digest,
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
        predicates = self._prepare_predicates(
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
        self,
        batch: Batch,
        *,
        transfers: tuple[_PreparedTransferInput, ...],
    ) -> _PreparedPredicateBatch | None:
        operations = tuple(
            operation
            for operation in batch.operations
            if operation.predicate is not None
            and operation.predicate.kind is ProductKind.COMPLETION
        )
        if not operations:
            return None
        transferred = {transfer.product: transfer for transfer in transfers}
        lease = self._completions.reserve(
            len(operations),
            token_capacity=len(operations),
            devices=tuple(self._operation_device(operation) for operation in operations),
        )
        captures: list[tuple[_OperationIdentity, CompletionCapture, int]] = []
        pending: list[tuple[_OperationIdentity, _PreparedTransferInput, int]] = []
        recorded: list[DeviceProductRead] = []
        try:
            grouped: dict[torch.device, list[Operation]] = defaultdict(list)
            rows = {_operation_identity(operation): row for row, operation in enumerate(operations)}
            for operation in operations:
                transfer = transferred.get(cast(ProductRef, operation.predicate))
                if transfer is None:
                    grouped[self._operation_device(operation)].append(operation)
                else:
                    pending.append(
                        (
                            _operation_identity(operation),
                            transfer,
                            rows[_operation_identity(operation)],
                        )
                    )
            for device, device_operations in grouped.items():
                reads = self.device_products.consume_batch(
                    tuple(
                        (
                            cast(ProductRef, operation.predicate),
                            int(operation.op_id),
                            None,
                            device,
                        )
                        for operation in device_operations
                    ),
                    device=device,
                )
                recorded.extend(reads)
                for operation, read in zip(device_operations, reads, strict=True):
                    identity = _operation_identity(operation)
                    captures.append((identity, lease.capture(read.tensor), rows[identity]))
                self.device_products.record_readers(reads, device=device)
            sealed = not pending
            if sealed:
                lease.seal()
        except BaseException:
            unrecorded = tuple(read for read in recorded if not read._recorded)
            if unrecorded:
                self.device_products.record_readers(unrecorded)
            lease.abandon()
            raise
        return _PreparedPredicateBatch(
            lease=lease,
            entries=captures,
            transferred=tuple(pending),
            sealed=sealed,
        )

    def execute_prepared(self, prepared: PreparedExecution) -> CompletionReport:
        if not prepared.ready():
            raise RuntimeError("prepared execution was observed before transfer readiness")
        return self._execute(
            prepared.batch,
            prepared=prepared.transfers,
            predicate_values=prepared.predicate_values(),
            propagate_errors=False,
            graph_eligible=True,
        )

    def complete_startup(self) -> None:
        """Retire pre-admission collective identities before serving traffic."""

        missing_mixed = self.mixed_buckets - self._qualified_mixed_buckets
        if missing_mixed:
            raise GraphExecutionError(
                "mixed execution buckets lack a matched serving-path interference proof: "
                f"{sorted(missing_mixed, key=repr)!r}"
            )
        if self._model_invocation is not None:
            self._model_invocation.complete_startup()
        if self.requests.request_ids():
            raise RuntimeError("startup completed with resident requests")
        self._collective_history.clear()

    def execute(
        self,
        batch: Batch,
        *,
        prepared: tuple[_PreparedTransferInput, ...] = (),
    ) -> CompletionReport:
        """Execute one canonical batch after server-side duplicate registration."""

        return self._execute(
            batch,
            prepared=prepared,
            predicate_values={},
            propagate_errors=False,
            graph_eligible=True,
        )

    def execute_startup(
        self,
        batch: Batch,
        *,
        catalog_graphs: bool = True,
    ) -> CompletionReport:
        """Execute pre-admission work with direct error propagation."""

        return self._execute(
            batch,
            prepared=(),
            predicate_values={},
            propagate_errors=True,
            graph_eligible=bool(catalog_graphs),
        )

    def _execute(
        self,
        batch: Batch,
        *,
        prepared: tuple[_PreparedTransferInput, ...],
        predicate_values: Mapping[_OperationIdentity, bool],
        propagate_errors: bool,
        graph_eligible: bool,
    ) -> CompletionReport:
        """Shared execution for startup and admitted traffic."""

        started = time.perf_counter_ns()
        operations = _trace_envelopes(batch.operations)
        validation_started = time.perf_counter_ns()
        try:
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
        required_predicates = {
            _operation_identity(operation)
            for operation in batch.operations
            if operation.predicate is not None
            and operation.predicate.kind is ProductKind.COMPLETION
        }
        if required_predicates != set(predicate_values):
            raise invalid_descriptor(
                "completion-predicated operations require exact prepared predicate values"
            )
        self.requests.apply_controls(batch.controls)
        if not batch.operations:
            self._apply_release_controls(batch)
            return CompletionReport(
                step_id=batch.step_id,
                partitions=(),
            )
        reports: dict[int, PartitionCompletion] = {}
        groups: dict[int, list[BatchPartition]] = {}
        for partition in batch.partitions:
            groups.setdefault(partition.submission_group, []).append(partition)

        for partitions in groups.values():
            scopes: list[_ExecutionScope] = []
            for partition in partitions:
                try:
                    scopes.append(
                        self._open_partition(
                            batch,
                            partition,
                            prepared,
                            predicate_values,
                            graph_eligible,
                        )
                    )
                except BaseException as error:
                    classified = self._classify_partition_failure(
                        partition,
                        error,
                        phase="partition registration",
                    )
                    if propagate_errors or classified.fatal:
                        for scope in scopes:
                            self._discard_partition(scope, classified)
                        raise classified
                    reports[partition.partition_id] = self._registration_error_partition(
                        batch.step_id,
                        partition,
                        classified,
                        started,
                    )

            if not scopes:
                continue
            try:
                outcomes, execution_errors = self._execute_partition_group(
                    tuple(scopes),
                    qualify_mixed=propagate_errors,
                )
            except BaseException as error:
                classified = self._classify_partition_failure(
                    partitions[0],
                    error,
                    phase="partition execution",
                )
                for scope in scopes:
                    self._discard_partition(scope, classified)
                if propagate_errors or classified.fatal:
                    raise classified
                for scope in scopes:
                    reports[scope.partition.partition_id] = self._error_partition(
                        batch.step_id,
                        scope,
                        classified,
                        started,
                    )
                continue

            if propagate_errors and execution_errors:
                first_partition = next(
                    partition
                    for partition in partitions
                    if partition.partition_id in execution_errors
                )
                classified = self._classify_partition_failure(
                    first_partition,
                    execution_errors[first_partition.partition_id],
                    phase="partition execution",
                )
                for scope in scopes:
                    self._discard_partition(scope, classified)
                raise classified

            for scope in scopes:
                partition_error = execution_errors.get(scope.partition.partition_id)
                if partition_error is not None:
                    classified = self._classify_partition_failure(
                        scope.partition,
                        partition_error,
                        phase="partition execution",
                    )
                    self._discard_partition(scope, classified)
                    if propagate_errors or classified.fatal:
                        raise classified
                    reports[scope.partition.partition_id] = self._error_partition(
                        batch.step_id,
                        scope,
                        classified,
                        started,
                    )
                    continue
                partition_outcomes = outcomes[scope.partition.partition_id]
                try:
                    reports[scope.partition.partition_id] = self._commit_partition(
                        batch.step_id,
                        scope,
                        partition_outcomes,
                        started,
                    )
                except BaseException as error:
                    if scope.publication_started:
                        classified = self._published_partition_failure(
                            scope.partition,
                            error,
                        )
                    else:
                        classified = self._classify_partition_failure(
                            scope.partition,
                            error,
                            phase="partition commit",
                        )
                        self._discard_partition(scope, classified)
                    if propagate_errors or classified.fatal:
                        raise classified
                    reports[scope.partition.partition_id] = self._error_partition(
                        batch.step_id,
                        scope,
                        classified,
                        started,
                    )

        self._apply_release_controls(batch)
        report = CompletionReport(
            step_id=batch.step_id,
            partitions=tuple(reports[partition.partition_id] for partition in batch.partitions),
        )
        self.trace.emit(
            ExecutionPhase.COMMIT,
            operations,
            duration_us=(time.perf_counter_ns() - started) // 1000,
        )
        return report

    def _classify_partition_failure(
        self,
        partition: BatchPartition,
        error: BaseException,
        *,
        phase: str,
    ) -> WorkerError:
        operations = tuple(
            (
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
            op_kind=None if sole is None else sole.work.variant.value,
            route=str(partition.route),
        )
        self._log_partition_failure(partition, classified, cause=error)
        return classified

    def _published_partition_failure(
        self,
        partition: BatchPartition,
        error: BaseException,
    ) -> WorkerError:
        operations = tuple(
            (
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
        self._log_partition_failure(partition, classified, cause=error)
        return classified

    @staticmethod
    def _log_partition_failure(
        partition: BatchPartition,
        error: WorkerError,
        *,
        cause: BaseException | None = None,
    ) -> None:
        capture_trace = should_capture_trace(str(error.code))
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
        self,
        batch: Batch,
        partition: BatchPartition,
        prepared: tuple[_PreparedTransferInput, ...],
        predicate_values: Mapping[_OperationIdentity, bool],
        graph_eligible: bool,
    ) -> _ExecutionScope:
        operations = partition.operations
        predicated = frozenset(
            identity
            for operation in operations
            if (identity := _operation_identity(operation)) in predicate_values
            and not predicate_values[identity]
        )
        active_operations = tuple(
            operation
            for operation in operations
            if _operation_identity(operation) not in predicated
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
            candidates, bases = self.requests.stage_partition(
                operations,
                admissions,
                partition.request_pool_indices,
            )
            for operation, request in zip(operations, candidates, strict=True):
                request.install_runtime(self._parent_runtime(operation, request))
            completion = self._completions.reserve(
                len(operations),
                token_capacity=self._partition_completion_words(operations),
                devices=self._completion_devices(operations),
            )
        except BaseException as error:
            self.trace.emit(
                ExecutionPhase.CANDIDATE_STAGE,
                traced,
                duration_us=(time.perf_counter_ns() - started) // 1000,
                error=error,
            )
            raise
        scope = _ExecutionScope(
            partition=partition,
            started_ns=started,
            graph_eligible=graph_eligible,
            request_candidates=candidates,
            request_bases=bases,
            request_rows={request.session_id: request for request in candidates},
            completion=completion,
            prepared_transfers={
                transfer.product: transfer
                for transfer in prepared
                if transfer.product in declared_inputs
            },
            predicated_operations=predicated,
        )
        self.trace.emit(
            ExecutionPhase.CANDIDATE_STAGE,
            traced,
            duration_us=(time.perf_counter_ns() - started) // 1000,
        )
        try:
            if self.runtime_states is not None:
                self.runtime_states.reset(
                    tuple(
                        int(request.request_pool_idx)
                        for request, base in zip(candidates, bases, strict=True)
                        if base is None
                    )
                )
            self._reserve_cpu_tasks(active_operations, scope)
            active_partition = self._active_partition(partition, active_operations)
            if active_partition is not None:
                self._bind_cache_rows(active_partition, scope)
            scope.layout = _PartitionLayout(
                operations=operations,
                requests=candidates,
                cache_rows=tuple(
                    scope.cache_rows.get((operation.request_key, operation.op_id, 0))
                    for operation in operations
                ),
                weights=tuple(self._weights() for _ in candidates),
                identities=tuple(_operation_identity(operation) for operation in operations),
            )
            if active_partition is not None:
                self._bind_latent_rows(active_partition, scope)
            self._reserve_outputs(operations, scope)
            active_inputs = {
                reference for operation in active_operations for reference in operation.inputs
            }
            active_inputs.update(
                operation.predicate
                for operation in active_operations
                if operation.predicate is not None
            )
            self._stage_input_products(
                tuple(payload for payload in input_products if payload.product in active_inputs),
                scope,
            )
            self._consume_predicates(active_operations, scope)
            self._publish_predicated_outputs(operations, scope)
            scope.registration_visible = True
            return scope
        except BaseException:
            self._discard_partition(scope)
            raise

    @staticmethod
    def _active_partition(
        partition: BatchPartition,
        operations: tuple[Operation, ...],
    ) -> BatchPartition | None:
        if not operations:
            return None
        identities = {_operation_identity(operation) for operation in operations}
        indices = tuple(
            request_pool_idx
            for operation, request_pool_idx in zip(
                partition.operations,
                partition.request_pool_indices,
                strict=True,
            )
            if _operation_identity(operation) in identities
        )
        return replace(
            partition,
            operations=operations,
            request_pool_indices=indices,
            kv_placements=tuple(
                placement
                for placement in partition.kv_placements
                if (placement.request_key, int(placement.op_id)) in identities
            ),
            kv_branch_placements=tuple(
                placement
                for placement in partition.kv_branch_placements
                if (placement.request_key, int(placement.op_id)) in identities
            ),
            latent_placements=tuple(
                placement
                for placement in partition.latent_placements
                if (placement.request_key, int(placement.op_id)) in identities
            ),
        )

    @staticmethod
    def _partition_completion_words(operations: tuple[Operation, ...]) -> int:
        return max(
            1,
            SAMPLING_COMPLETION_FIELDS * len(operations)
            + sum(
                (int(operation.bounds.max_completion_bytes) + 3) // 4 for operation in operations
            ),
        )

    def _execute_partition_group(
        self,
        scopes: tuple[_ExecutionScope, ...],
        *,
        qualify_mixed: bool,
    ) -> tuple[dict[int, tuple[_Outcome, ...]], dict[int, BaseException]]:
        if len(scopes) == 1:
            scope = scopes[0]
            return (
                {
                    scope.partition.partition_id: self._execute_operations(
                        scope.partition.operations,
                        scope,
                    )
                },
                {},
            )
        for scope in scopes:
            active = tuple(
                operation
                for operation in scope.partition.operations
                if _operation_identity(operation) not in scope.predicated_operations
            )
            for device in self._completion_devices(active):
                scope.completion.begin_device(device)
        drivers: list[tuple[int, int, _Driver, _ExecutionScope]] = []
        grouped: list[list[_Outcome | None]] = [
            [None] * len(scope.partition.operations) for scope in scopes
        ]
        for scope_index, scope in enumerate(scopes):
            for operation_index, operation in enumerate(scope.partition.operations):
                if _operation_identity(operation) in scope.predicated_operations:
                    grouped[scope_index][operation_index] = self._predicated_outcome(
                        operation,
                        scope,
                    )
                    continue
                drivers.append(
                    (scope_index, operation_index, self._driver(operation, scope), scope)
                )
        flat_outcomes, errors = (
            self._drive_partitioned(tuple(drivers), qualify_mixed=qualify_mixed)
            if drivers
            else ((), {})
        )
        for (scope_index, operation_index, _driver, _scope), outcome in zip(
            drivers,
            flat_outcomes,
            strict=True,
        ):
            if outcome is not None:
                grouped[scope_index][operation_index] = outcome
        outcomes: dict[int, tuple[_Outcome, ...]] = {}
        for scope, partition_outcomes in zip(scopes, grouped, strict=True):
            partition_id = scope.partition.partition_id
            if partition_id in errors:
                continue
            if any(outcome is None for outcome in partition_outcomes):
                raise RuntimeError("successful partition did not resolve every operation")
            outcomes[partition_id] = tuple(
                cast(_Outcome, outcome) for outcome in partition_outcomes
            )
        return outcomes, errors

    def _commit_partition(
        self,
        step_id: int,
        scope: _ExecutionScope,
        outcomes: tuple[_Outcome, ...],
        started: int,
    ) -> PartitionCompletion:
        partition = scope.partition
        operations = partition.operations
        self._finish_device_reads(scope)
        self._publish_predicates(scope)
        self.device_products.validate_writes(tuple(scope.device_writes))
        self.encoder_cache.validate_writes(tuple(scope.encoder_writes))
        if self.latent_pool is None:
            if scope.latent_publications or scope.latent_releases:
                raise RuntimeError("latent publication has no physical pool")
        else:
            self.latent_pool.validate_commit(
                scope.latent_publications,
                scope.latent_releases,
            )
        scope.completion.seal()
        records: list[CompletionRecord] = []
        selected_versions: dict[int, VersionRef] = {}
        report_products: list[ProductPayload] = []
        pending_by_session: dict[int, _PendingDigest] = {}
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
            self._validate_completion_products(operation, outcome.products)
            if int(self.deployment.tp_rank) == 0:
                report_products.extend(outcome.products)
            parent_semantic = _parent_semantic(operation, request)
            placeholder = CompletionRecord(
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
                semantic_digest="0" * 64,
                error_code=None,
                timing_counters=TimingCounters(),
            )
            pending = _PendingDigest(
                placeholder,
                parent_semantic,
                operation.plan_digest,
                scope.completion,
                row,
                partial(self._finalize_predicated_runtime, operation),
                (
                    partial(self._finalize_speculative_runtime, operation, outcome.selection)
                    if outcome.selection is not None
                    else None
                ),
                completion_tasks=(
                    *outcome.completion_tasks,
                    *(
                        cast(_CompletionLogprobPayload, product.payload)
                        for product in outcome.products
                        if isinstance(product.payload, _CompletionLogprobPayload)
                    ),
                ),
            )
            records.append(replace(placeholder, semantic_digest=cast(str, pending)))
            if operation.advances_state:
                selected_versions[operation.request_key.session_id] = VersionRef(
                    request_key=operation.request_key,
                    producer_op_id=operation.op_id,
                    point=FixedPoint(cast(int, outcome.selected_point), cast(str, pending)),
                )
                pending_by_session[operation.request_key.session_id] = pending
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
                kv_reserved_len=outcome.logical_lengths.kv_reserved_len,
                kv_initialized_len=outcome.logical_lengths.kv_initialized_len,
                kv_visible_len=outcome.logical_lengths.kv_visible_len,
                kv_committed_len=outcome.logical_lengths.kv_committed_len,
                kv_published_len=outcome.logical_lengths.kv_published_len,
            )
        partition_report = PartitionCompletion(
            partition_id=partition.partition_id,
            completions=tuple(records),
            products=tuple(report_products),
            registration=RegistrationAck(visible=True),
            worker_exec_us=(time.perf_counter_ns() - scope.started_ns) // 1000,
            forward_stats=_forward_stats(scope.observations, scope.component_us),
        )
        cache_commit = self.cache_publications.prepare_commit(
            scope.cache_publications,
            scope.cache_installations,
            self.transport,
        )
        request_publication = self.requests.prepare_publication(
            step_id=step_id,
            operations=operations,
            candidates=scope.request_candidates,
            bases=scope.request_bases,
            selected_versions=selected_versions,
            runtimes=resolved_runtime,
        )
        for identity, locators in scope.stage_publications.items():
            existing = self._transport_publications.get(identity)
            if existing is not None and existing != locators:
                raise RuntimeError("committed transport publication identity was reused")
        scope.publication_started = True
        self.device_products.commit_writes(tuple(scope.device_writes))
        self.encoder_cache.commit_writes(tuple(scope.encoder_writes))
        if self.latent_pool is not None:
            self.latent_pool.apply_commit(
                scope.latent_publications,
                scope.latent_releases,
            )
        self.cache_publications.apply_commit(cache_commit)
        for branch_identity, branch_row in scope.branch_publications.items():
            self._branch_cache_rows[branch_identity] = branch_row
        for publication_identity, locators in scope.stage_publications.items():
            self._transport_publications[publication_identity] = locators
        self._commit_runtime_states(scope)
        self.requests.publish(request_publication)
        for session_id, pending in pending_by_session.items():
            if pending.ready():
                self.requests.get(session_id).resolved_digest = pending.resolve()
        return partition_report

    def _commit_runtime_states(self, scope: _ExecutionScope) -> None:
        states = self.runtime_states
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
            if isinstance(publication, _DecodeRuntimePublication):
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
                weight = (
                    publication.valid.reshape(-1)[:1] & publication.active.reshape(-1)[:1]
                ).to(dtype=penalty_base.dtype)
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
        self,
        scope: _ExecutionScope,
        error: BaseException | None = None,
    ) -> None:
        self._finish_device_reads(scope)
        for reservation in scope.cpu_tasks.values():
            reservation.abandon()
        if scope.publication_started:
            raise RuntimeError("published partition state cannot be discarded")
        scope.completion.abandon()
        self.device_products.abandon_writes(tuple(scope.device_writes))
        self.encoder_cache.abandon_writes(tuple(scope.encoder_writes))
        if self.latent_pool is not None and scope.latent_import_slots:
            self.latent_pool.release_slots(tuple(scope.latent_import_slots))
        self._release_locators(scope.published)
        self.trace.emit(
            ExecutionPhase.CANDIDATE_DISCARD,
            _trace_envelopes(scope.partition.operations),
            error=error,
        )

    def _reserve_cpu_tasks(
        self,
        operations: tuple[Operation, ...],
        scope: _ExecutionScope,
    ) -> None:
        for operation in operations:
            if operation.work.variant is not WorkVariant.MATERIALIZE:
                continue
            identity = _operation_identity(operation)
            if identity in scope.cpu_tasks:
                raise invalid_descriptor("materialization repeats its CPU task identity")
            scope.cpu_tasks[identity] = self._cpu_tasks.reserve()

    def _registration_error_partition(
        self,
        step_id: int,
        partition: BatchPartition,
        error: WorkerError,
        started: int,
    ) -> PartitionCompletion:
        generation = 1
        report = self._build_error_partition(
            partition,
            generation,
            False,
            error,
            started,
            WorkerForwardStats(),
        )
        return report

    def _error_partition(
        self,
        step_id: int,
        scope: _ExecutionScope,
        error: WorkerError,
        started: int,
    ) -> PartitionCompletion:
        report = self._build_error_partition(
            scope.partition,
            scope.completion.generation,
            scope.registration_visible,
            error,
            scope.started_ns,
            _forward_stats(scope.observations, scope.component_us),
        )
        return report

    def _build_error_partition(
        self,
        partition: BatchPartition,
        generation: int,
        registration_visible: bool,
        error: WorkerError,
        started: int,
        forward_stats: WorkerForwardStats,
    ) -> PartitionCompletion:
        protocol_code = _protocol_error_code(error.code)
        records: list[CompletionRecord] = []
        for operation in partition.operations:
            session = self.requests.peek(operation.request_key.session_id)
            selected_parent = (
                operation.parent
                if operation.parent.is_fixed()
                else None
                if session is None
                else session.resolve_version(operation.parent)
            )
            point = None if selected_parent is None else selected_parent.point
            selected_point = point.point_index if isinstance(point, FixedPoint) else 0
            parent_semantic: object = (
                point.semantic_digest if isinstance(point, FixedPoint) else "0" * 64
            )
            lengths = (
                LogicalLengths()
                if session is None
                else self._logical_lengths(operation, session, None)
            )
            placeholder = CompletionRecord(
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
                semantic_digest="0" * 64,
                error_code=protocol_code,
                timing_counters=TimingCounters(),
            )
            semantic_digest: object
            if isinstance(parent_semantic, (_PendingDigest, _PendingErrorDigest)):
                semantic_digest = _PendingErrorDigest(
                    parent_semantic,
                    placeholder,
                    operation.plan_digest,
                )
            else:
                semantic_digest = placeholder.compute_semantic_digest(
                    cast(str, parent_semantic),
                    operation.plan_digest,
                )
            records.append(
                replace(
                    placeholder,
                    semantic_digest=cast(str, semantic_digest),
                )
            )
        return PartitionCompletion(
            partition_id=partition.partition_id,
            completions=tuple(records),
            registration=RegistrationAck(visible=registration_visible),
            worker_exec_us=(time.perf_counter_ns() - started) // 1000,
            forward_stats=forward_stats,
        )

    def _finalize_predicated_runtime(
        self,
        operation: Operation,
    ) -> tuple[VersionRef, RequestRuntime]:
        selected, runtime = self.requests.finalize_predicated(
            operation.request_key.session_id,
            operation.op_id,
            operation.parent,
        )
        return selected, runtime

    def _finalize_speculative_runtime(
        self,
        operation: Operation,
        selection: _SpeculativeSelection,
        record: CompletionRecord,
        selected_digest: str,
        parent_semantic: str,
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
            record.logical_lengths.kv_initialized_len != selection.initialized_kv
            or selected_kv > record.logical_lengths.kv_initialized_len
        ):
            raise RuntimeError("speculative KV selection is outside initialized state")
        prefixes: list[tuple[VersionRef, RequestRuntime]] = []
        for point_index in range(1, selected_point + 1):
            prefix_record = replace(
                record,
                selected_point=point_index,
                logical_lengths=replace(
                    record.logical_lengths,
                    token_len=selection.base_logical_position + point_index,
                    kv_visible_len=selection.base_kv_visible + point_index,
                ),
                token_span=replace(record.token_span, len=point_index),
                committed_tokens=tokens[:point_index],
            )
            digest = prefix_record.compute_semantic_digest(
                parent_semantic=parent_semantic,
                plan_digest=operation.plan_digest,
            )
            request = self.requests.get(operation.request_key.session_id)
            runtime = RequestRuntime(
                logical_position=selection.base_logical_position + point_index,
                rng_counter=selection.base_rng_counter + point_index,
                latent_product=request.latent_product,
                flow_step=request.flow_step,
                kv_reserved_len=record.logical_lengths.kv_reserved_len,
                kv_initialized_len=record.logical_lengths.kv_initialized_len,
                kv_visible_len=selection.base_kv_visible + point_index,
                kv_committed_len=record.logical_lengths.kv_committed_len,
                kv_published_len=record.logical_lengths.kv_published_len,
            )
            prefixes.append(
                (
                    VersionRef(
                        request_key=operation.request_key,
                        producer_op_id=operation.op_id,
                        point=FixedPoint(point_index, digest),
                    ),
                    runtime,
                )
            )
        selected, _runtime = prefixes[-1]
        point = cast(FixedPoint, selected.point)
        if point.semantic_digest != selected_digest:
            raise RuntimeError("selected speculative prefix digest is inconsistent")
        self.requests.finalize_prefixes(
            operation.request_key.session_id,
            operation.op_id,
            prefixes,
        )

    def _validate_batch_identity(self, batch: Batch) -> None:
        if len(batch.operations) > self.deployment.max_batch_operations:
            raise invalid_descriptor("execution batch exceeds the deployment operation limit")
        for operation in batch.operations:
            variant = operation.work.variant
            if variant not in self.allowed_work_variants:
                raise unsupported_operation(variant.value, operation.request_key.session_id)
        if any(
            index > self.deployment.max_request_pool_size
            for partition in batch.partitions
            for index in partition.request_pool_indices
        ):
            raise invalid_descriptor("execution batch exceeds request-slot capacity")
        groups: dict[int, list[BatchPartition]] = defaultdict(list)
        for partition in batch.partitions:
            groups[partition.submission_group].append(partition)
        for partitions in groups.values():
            first = partitions[0]
            if first.execution is not ExecutionCapability.TENSORIZED_MIXED:
                continue
            variants = {
                operation.work.variant
                for partition in partitions
                for operation in partition.operations
            }
            if not self.model.tensorized_mixed or variants != {
                WorkVariant.TOKEN_DECODE,
                WorkVariant.GEN_FLOW,
            }:
                raise invalid_descriptor(
                    "tensorized mixed submission exceeds worker mixed-execution capabilities"
                )
            capability = self._mixed_capability(tuple(partitions))
            if capability not in self.mixed_buckets:
                raise invalid_descriptor(
                    "tensorized mixed submission has no exact qualified capability bucket"
                )
        group_identities: list[tuple[int, str]] = []
        for submission_group, partitions in groups.items():
            collective_seq = partitions[0].collective_seq
            digest = hashlib.sha256()
            digest.update(int(submission_group).to_bytes(4, "little"))
            digest.update(int(collective_seq).to_bytes(8, "little"))
            for partition in sorted(partitions, key=lambda value: value.partition_id):
                digest.update(int(partition.partition_id).to_bytes(4, "little"))
                digest.update(int(partition.route).to_bytes(4, "little"))
                digest.update(partition.domain.value.encode("ascii"))
                digest.update(partition.execution.value.encode("ascii"))
                for operation in partition.operations:
                    digest.update(operation.plan_digest.encode("ascii"))
            group_identities.append((int(collective_seq), digest.hexdigest()))
        for collective_seq, collective_digest in sorted(group_identities):
            existing = self._collective_history.get(collective_seq)
            if existing is not None:
                if existing != collective_digest:
                    raise invalid_descriptor("collective sequence was reused with different work")
                continue
            if self._collective_history and collective_seq <= next(
                reversed(self._collective_history)
            ):
                raise invalid_descriptor("collective sequence does not advance")
            self._collective_history[collective_seq] = collective_digest
            while len(self._collective_history) > 4096:
                self._collective_history.popitem(last=False)

    def _completion_devices(self, operations: tuple[Operation, ...]) -> tuple[str, ...]:
        deployment = self.deployment
        selected: list[str] = []
        for operation in operations:
            device = (
                deployment.generation_device
                if operation.work.variant
                in {
                    WorkVariant.GEN_TRANSITION,
                    WorkVariant.GEN_FLOW,
                    WorkVariant.MATERIALIZE,
                }
                and deployment.generation_device is not None
                else deployment.device
            )
            if device not in selected:
                selected.append(device)
        return tuple(selected)

    @staticmethod
    def _mixed_capability(
        partitions: tuple[BatchPartition, ...],
    ) -> MixedExecutionCapability:
        decode_rows = sum(
            operation.work.variant is WorkVariant.TOKEN_DECODE
            for partition in partitions
            for operation in partition.operations
        )
        flow_operations = tuple(
            operation
            for partition in partitions
            for operation in partition.operations
            if operation.work.variant is WorkVariant.GEN_FLOW
        )
        flow_placements = {
            (placement.request_key, int(placement.op_id)): placement
            for partition in partitions
            for placement in partition.latent_placements
        }
        branch_counts: dict[tuple[RequestKey, int], int] = defaultdict(int)
        for partition in partitions:
            for placement in partition.kv_branch_placements:
                branch_counts[(placement.request_key, int(placement.op_id))] += 1
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
        return MixedExecutionCapability(
            decode_rows=decode_rows,
            flow_rows=len(flow_operations),
            height=height,
            width=width,
            cfg_branches=cfg_branches,
        )

    def _reserve_outputs(
        self,
        operations: tuple[Operation, ...],
        scope: _ExecutionScope,
    ) -> None:
        """Bind each declared device value to its concrete bounded owner."""

        scalar_groups: dict[
            tuple[torch.device, ProductKind, DType, ShapeBound],
            list[tuple[ProductRef, str, torch.device | str]],
        ] = {}
        general_bindings: list[tuple[ProductRef, str, torch.device | str]] = []
        encoder_bindings: list[tuple[ProductRef, str, torch.device | str]] = []
        for operation in operations:
            device = self._operation_device(operation)
            for output in operation.outputs:
                if (
                    _operation_identity(operation) in scope.predicated_operations
                    and output.kind is not ProductKind.COMPLETION
                ):
                    continue
                if (
                    operation.work.variant is WorkVariant.TRANSFER_PRODUCT
                    and _is_transferable_product(output)
                ):
                    continue
                if output.kind in {
                    ProductKind.VISION_FEATURE,
                    ProductKind.LATENT_FEATURE,
                }:
                    encoder_bindings.append((output, operation.plan_digest, device))
                    continue
                if _requires_device_product_binding(output):
                    binding = (output, operation.plan_digest, device)
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
        bound_groups = self.device_products.bind_output_groups(groups)
        scope.device_writes.extend(write for binding in bound_groups for write in binding.writes)
        scope.encoder_writes.extend(self.encoder_cache.bind_outputs(tuple(encoder_bindings)))
        operation_identities = {_operation_identity(operation) for operation in operations}
        token_operation_identities = {
            _operation_identity(operation)
            for operation in operations
            if operation.work.kind == "token"
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

    def _operation_device(self, operation: Operation) -> torch.device:
        return (
            self._generation_device
            if operation.work.variant
            in {
                WorkVariant.GEN_TRANSITION,
                WorkVariant.GEN_FLOW,
                WorkVariant.MATERIALIZE,
            }
            else self._device
        )

    def _validate_completion_products(
        self,
        operation: Operation,
        products: tuple[ProductPayload, ...],
    ) -> None:
        declared = {output: output for output in operation.outputs}
        for product in products:
            reference = declared.get(product.product)
            if reference is None:
                raise invalid_descriptor(
                    "completion carries a product not declared by its operation"
                )
            payload_bound = (
                product.payload.max_encoded_bytes()
                if isinstance(
                    product.payload,
                    (
                        _CompletionImagePayload,
                        _CompletionLogprobPayload,
                        _CompletionTransferPayload,
                    ),
                )
                else len(product.payload)
            )
            transferred = isinstance(product.payload, _CompletionTransferPayload)
            if transferred and reference.storage_class in {
                StorageClass.HOST_STAGING,
                StorageClass.COMPLETION_ARENA,
            }:
                raise invalid_descriptor("host-visible output cannot carry a transfer entry")
            if not transferred and payload_bound > int(reference.max_bytes):
                raise invalid_descriptor(
                    "completion product exceeds its registered product byte bound"
                )
            if reference.storage_class in (
                StorageClass.HOST_STAGING,
                StorageClass.COMPLETION_ARENA,
            ) and payload_bound > int(operation.bounds.max_completion_bytes):
                raise invalid_descriptor("completion product exceeds its registered byte bound")

    def _consume_predicates(
        self,
        operations: tuple[Operation, ...],
        scope: _ExecutionScope,
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
            device = self._operation_device(operation)
            grouped.setdefault(device, []).append(
                (
                    operation,
                    (
                        predicate,
                        int(operation.op_id),
                        None,
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
            reads = self.device_products.consume_batch(
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
                _reference, consumer_op_id, producer_digest, target = request
                read = self._consume_device_product(
                    predicate,
                    scope,
                    consumer_op_id=consumer_op_id,
                    producer_plan_digest=producer_digest,
                    device=target,
                )
                scope.device_reads.append(read)
                tagged = predicate.kind is ProductKind.TOKEN and predicate.dtype is DType.U32
                scope.predicate_values[_operation_identity(operation)] = (read.tensor, tagged)

    def _publish_predicates(
        self,
        scope: _ExecutionScope,
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
        batch = self.device_products.producer_scalar_batch(writes)
        if batch is not None:
            batch.tensor.fill_(1)
            self.device_products.publish_scalar_batch(batch)
            return
        views = self.device_products.producer_write_views(writes)
        first = views[0]
        self.device_products.publish_writes(
            writes,
            torch.ones(
                (len(writes),),
                dtype=first.dtype,
                device=first.device,
            ),
        )

    def _publish_predicated_outputs(
        self,
        operations: tuple[Operation, ...],
        scope: _ExecutionScope,
    ) -> None:
        declared = {_operation_identity(operation) for operation in operations}
        for identity, writes in scope.propagated_predicate_writes.items():
            if identity not in declared:
                raise RuntimeError("predicated output has no operation in its partition")
            for write in writes:
                self.device_products.publish_scalar_write(write, False)

    def _finish_device_reads(
        self,
        scope: _ExecutionScope,
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
            self.device_products.record_readers(
                reads,
                after_writes=tuple(after_writes),
            )
        scope.device_reads.clear()
        encoder_reads = tuple(read for read in scope.encoder_reads if not read._recorded)
        if encoder_reads:
            self.encoder_cache.record_readers(encoder_reads)
        scope.encoder_reads.clear()

    def _apply_release_controls(self, batch: Batch) -> None:
        releases = tuple(
            (control.request_key, control.op_id)
            for control in batch.controls
            if isinstance(control, Release)
        )
        self.device_products.release_operations(releases)
        self.encoder_cache.release_operations(releases)
        self.cache_publications.release_operations(releases)
        if self.transport is not None:
            for identity in releases:
                self._release_locators(self._transport_publications.pop(identity, ()))

    def drop_session(self, session_id: int) -> None:
        """Release stage publications owned by one dropped request."""

        session = self.requests.peek(int(session_id))
        if session is not None and self.runtime_states is not None:
            self.runtime_states.release((int(session.request_pool_idx),))
        self.cache_publications.drop(session_id)
        branch_rows = tuple(
            identity
            for identity in self._branch_cache_rows
            if int(identity[0].session_id) == int(session_id)
        )
        for branch_identity in branch_rows:
            del self._branch_cache_rows[branch_identity]
        if self.transport is None:
            return
        selected = tuple(
            identity
            for identity in self._transport_publications
            if int(identity[0].session_id) == int(session_id)
        )
        for identity in selected:
            self._release_locators(self._transport_publications.pop(identity))

    def _bind_latent_rows(
        self,
        partition: BatchPartition,
        scope: _ExecutionScope,
    ) -> None:
        if not partition.latent_placements:
            return
        pool = self.latent_pool
        if pool is None:
            raise capability_mismatch("scheduler latent placement has no worker physical pool")
        operations = {
            _operation_identity(operation): (operation, int(request_pool_idx))
            for operation, request_pool_idx in zip(
                partition.operations,
                partition.request_pool_indices,
                strict=True,
            )
        }
        rows: list[tuple[_OperationIdentity, LatentPlacement, int]] = []
        for placement in partition.latent_placements:
            identity = (placement.request_key, int(placement.op_id))
            selected = operations.get(identity)
            if selected is None:
                raise invalid_descriptor(
                    "latent placement names an operation outside its partition"
                )
            operation, slot = selected
            session = self._request_row(scope, operation.request_key.session_id)
            image = session.image
            if image is None:
                raise invalid_descriptor("latent placement has no admitted image geometry")
            flow = self._generation()
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
            if operation.work.variant is WorkVariant.GEN_TRANSITION:
                if int(placement.start_step) != 0 or int(placement.step_count) != 0:
                    raise invalid_descriptor(
                        "generation transition placement carries denoise steps"
                    )
            elif operation.work.variant is WorkVariant.GEN_FLOW:
                if (
                    int(placement.start_step) != committed_step
                    or int(placement.step_count) < 1
                    or int(placement.start_step) + int(placement.step_count) > int(image.steps)
                    or (
                        int(operation.bounds.max_tokens) > 0
                        and int(placement.step_count) > int(operation.bounds.max_tokens)
                    )
                ):
                    raise invalid_descriptor(
                        "generation flow placement exceeds its committed schedule"
                    )
            elif int(placement.start_step) != committed_step or int(placement.step_count) != 0:
                raise invalid_descriptor(
                    "latent reader placement disagrees with committed step state"
                )
            rows.append((identity, placement, slot))
        staged = pool.stage(
            tuple(placement.page_table for _identity, placement, _slot in rows),
            tuple(int(placement.latent_units) for _identity, placement, _slot in rows),
        )
        scope.latent_rows = {
            identity: _LatentExecution(
                placement=placement,
                request_pool_idx=slot,
                staging=value,
            )
            for (identity, placement, slot), value in zip(rows, staged, strict=True)
        }

    @staticmethod
    def _latent_row(operation: Operation, scope: _ExecutionScope) -> _LatentExecution:
        row = scope.latent_rows.get(_operation_identity(operation))
        if row is None:
            raise invalid_descriptor("trajectory operation has no staged latent placement")
        return row

    def _bind_cache_rows(
        self,
        partition: BatchPartition,
        scope: _ExecutionScope,
    ) -> None:
        """Validate scheduler placement and zero exactly its declared fresh pages."""

        operations = {
            (operation.request_key, operation.op_id): operation
            for operation in partition.operations
        }
        cache_rows: list[tuple[tuple[RequestKey, int, int], CacheRow, tuple[int, ...]]] = []
        for placement in partition.kv_placements:
            operation = operations.get((placement.request_key, placement.op_id))
            if operation is None:
                raise invalid_descriptor("KV placement names an operation outside its partition")
            session = self._request_row(scope, placement.request_key.session_id)
            parent_runtime = self._parent_runtime(operation, session)
            committed_runtime = session.runtime_for(session.committed_version())
            if committed_runtime is None:
                raise invalid_descriptor("KV placement session has no committed runtime")
            pages = self.cache_pool.validate_pages(
                placement.block_table,
                scratch=False,
                group=placement.group_id,
            )
            pages_to_zero = self.cache_pool.validate_pages(
                placement.pages_to_zero,
                scratch=False,
                group=placement.group_id,
            )
            if placement.resulting_length > len(pages) * self.cache_pool.block_size:
                raise invalid_descriptor("KV placement resulting extent exceeds its block table")
            if placement.prefix_length != placement.visible_length:
                raise invalid_descriptor(
                    "KV placement write cursor differs from its visible extent"
                )
            if placement.visible_length != parent_runtime.kv_visible_len:
                raise invalid_descriptor("KV placement visible extent disagrees with its parent")
            if committed_runtime.kv_visible_len > placement.visible_length:
                raise invalid_descriptor("KV placement precedes the committed KV extent")
            cache_rows.append(
                (
                    (placement.request_key, placement.op_id, placement.group_id),
                    CacheRow(
                        block_table=pages,
                        length=placement.visible_length,
                        capacity=len(pages) * self.cache_pool.block_size,
                        group_id=placement.group_id,
                        initialized_length=max(
                            placement.visible_length,
                            parent_runtime.kv_initialized_len,
                        ),
                        committed_length=committed_runtime.kv_visible_len,
                        published_length=min(
                            committed_runtime.kv_visible_len,
                            max(
                                parent_runtime.kv_published_len,
                                self.cache_publications.published_extent(
                                    placement.request_key.session_id
                                ),
                            ),
                        ),
                    ),
                    pages_to_zero,
                )
            )
        branch_rows: list[
            tuple[
                tuple[RequestKey, int, int],
                tuple[RequestKey, int, int, int],
                CacheRow,
                tuple[int, ...],
            ]
        ] = []
        for branch_placement in partition.kv_branch_placements:
            pages = self.cache_pool.validate_pages(
                branch_placement.block_table,
                scratch=True,
                group=branch_placement.group_id,
            )
            pages_to_zero = self.cache_pool.validate_pages(
                branch_placement.pages_to_zero,
                scratch=True,
                group=branch_placement.group_id,
            )
            persistent_identity = (
                branch_placement.request_key,
                branch_placement.branch_index,
                branch_placement.group_id,
            )
            if pages_to_zero:
                if set(pages_to_zero) != set(pages):
                    raise invalid_descriptor(
                        "fresh generation KV placement must initialize its complete block table"
                    )
                row = CacheRow(
                    block_table=pages,
                    length=0,
                    capacity=len(pages) * self.cache_pool.block_size,
                    group_id=branch_placement.group_id,
                )
            else:
                continued_row = self._branch_cache_rows.get(persistent_identity)
                if continued_row is None:
                    raise invalid_descriptor(
                        "generation KV placement continues an unknown physical branch"
                    )
                row = replace(continued_row)
                if row.block_table != pages or row.group_id != branch_placement.group_id:
                    raise invalid_descriptor(
                        "generation KV continuation changes its physical branch placement"
                    )
            branch_rows.append(
                (
                    persistent_identity,
                    (
                        branch_placement.request_key,
                        branch_placement.op_id,
                        branch_placement.branch_index,
                        branch_placement.group_id,
                    ),
                    row,
                    pages_to_zero,
                )
            )
        for cache_identity, row, pages_to_zero in cache_rows:
            if pages_to_zero:
                self.cache_pool.zero_pages(row.group_id, pages_to_zero)
            scope.cache_rows[cache_identity] = row
        for persistent_identity, branch_identity, row, pages_to_zero in branch_rows:
            if pages_to_zero:
                self.cache_pool.zero_pages(row.group_id, pages_to_zero)
            scope.branch_publications[persistent_identity] = row
            scope.branch_rows[branch_identity] = row

    @staticmethod
    def _request_row(scope: _ExecutionScope, session_id: int) -> RequestRow:
        try:
            return scope.request_rows[int(session_id)]
        except KeyError:
            raise invalid_descriptor(
                f"partition has no request row for session {session_id}"
            ) from None

    def _consume_device_product(
        self,
        reference: ProductRef,
        scope: _ExecutionScope,
        *,
        consumer_op_id: int,
        producer_plan_digest: str | None = None,
        device: torch.device | str | None = None,
    ) -> DeviceProductRead:
        candidate = scope.transferred_device_products.get(reference)
        if candidate is not None:
            return self.device_products.consume_candidate(
                candidate,
                consumer_op_id=consumer_op_id,
                producer_plan_digest=producer_plan_digest,
                device=device,
            )
        return self.device_products.consume(
            reference,
            consumer_op_id=consumer_op_id,
            producer_plan_digest=producer_plan_digest,
            device=device,
        )

    def _consume_encoder_feature(
        self,
        reference: ProductRef,
        scope: _ExecutionScope,
        *,
        consumer_op_id: int,
        producer_plan_digest: str | None = None,
        device: torch.device | str | None = None,
    ) -> EncoderRead:
        candidate = scope.transferred_encoder_features.get(reference)
        if candidate is not None:
            return self.encoder_cache.consume_candidate(
                candidate,
                consumer_op_id=consumer_op_id,
                producer_plan_digest=producer_plan_digest,
                device=device,
            )
        return self.encoder_cache.consume(
            reference,
            consumer_op_id=consumer_op_id,
            producer_plan_digest=producer_plan_digest,
            device=device,
        )

    @staticmethod
    def _parent_runtime(
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

    def parent_runtime(self, operation: Operation, request: RequestRow) -> RequestRuntime:
        return self._parent_runtime(operation, request)

    def _cache_row(
        self,
        operation: Operation,
        scope: _ExecutionScope,
        *,
        group_id: int = 0,
    ) -> CacheRow:
        row = scope.cache_rows.get((operation.request_key, operation.op_id, int(group_id)))
        if row is None:
            raise invalid_descriptor("operation has no scheduler KV placement")
        return row

    def _logical_lengths(
        self,
        operation: Operation,
        session: RequestRow,
        row: CacheRow | None,
        *,
        latent_len: int | None = None,
    ) -> LogicalLengths:
        if row is None:
            runtime = self._parent_runtime(operation, session)
            reserved = runtime.kv_reserved_len
            initialized = runtime.kv_initialized_len
            visible = runtime.kv_visible_len
            committed = runtime.kv_committed_len
            published = runtime.kv_published_len
        else:
            extents = row.extents()
            reserved = extents.reserved
            initialized = extents.initialized
            visible = extents.visible
            committed = extents.committed
            published = extents.published
        return LogicalLengths(
            token_len=session.logical_position,
            kv_visible_len=visible,
            latent_len=session.flow_step if latent_len is None else int(latent_len),
            kv_reserved_len=reserved,
            kv_initialized_len=initialized,
            kv_committed_len=committed,
            kv_published_len=published,
        )

    def _stage_input_products(
        self,
        input_products: Sequence[ProductPayload],
        scope: _ExecutionScope,
    ) -> None:
        """Decode ephemeral host inputs and publish transferred physical values."""

        for entry in input_products:
            product = entry.product
            if entry.payload.startswith(TRANSFER_DESCRIPTOR_PREFIX):
                transfer = scope.prepared_transfers.get(product)
                if transfer is None or not transfer.ready():
                    raise invalid_descriptor(
                        "cross-stage input has no query-ready prepared transfer"
                    )
                if transfer.kind == "kv":
                    snapshot = transfer.snapshot
                    if snapshot is None:
                        raise RuntimeError("prepared KV transfer has no validated snapshot")
                    existing = self.cache_publications.resident(product)
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
                    row = self._latent_row(consumers[0], scope)
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
                    session = self._request_row(scope, product.request_key.session_id)
                    if session.latent_product is not None or int(session.flow_step) != 0:
                        raise invalid_descriptor(
                            "latent transfer destination already owns a trajectory"
                        )
                    self._latent_pool().restore(
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
                devices = {self._operation_device(operation) for operation in consumers}
                if len(devices) != 1:
                    raise invalid_descriptor("transferred product spans multiple consumer devices")
                device = next(iter(devices))
                if transfer.kind == "device_product":
                    binding = self.device_products.bind_outputs(
                        ((product, transfer.producer_plan_digest, device),)
                    )[0]
                    scope.device_writes.append(binding)
                    scope.transferred_device_products[product] = binding
                    self.device_products.publish_write(
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
                encoder_binding = self.encoder_cache.bind_outputs(
                    ((product, transfer.producer_plan_digest, device),)
                )[0]
                scope.encoder_writes.append(encoder_binding)
                scope.transferred_encoder_features[product] = encoder_binding
                self.encoder_cache.publish(
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

    def _driver(self, operation: Operation, scope: _ExecutionScope) -> _Driver:
        kind = operation.work.kind
        if kind == "token":
            return self._sequence_driver(operation, scope)
        if kind == "gen":
            if operation.work.mode == GenMode.TRANSITION.value:
                return self._transition_driver(operation, scope)
            if operation.work.mode == GenMode.FLOW.value:
                return self._flow_driver(operation, scope)
            raise invalid_descriptor("generation operation names an unknown mode")
        if kind == "encode":
            return self._encode_driver(operation, scope)
        if kind == "materialize":
            return self._materialize_driver(operation, scope)
        return self._transfer_driver(operation, scope)

    def _execute_operations(
        self,
        operations: tuple[Operation, ...],
        scope: _ExecutionScope,
    ) -> tuple[_Outcome, ...]:
        predicated = scope.predicated_operations
        active = (
            operations
            if not predicated
            else tuple(
                operation
                for operation in operations
                if _operation_identity(operation) not in predicated
            )
        )
        if not active:
            return tuple(self._predicated_outcome(operation, scope) for operation in operations)
        decode = all(
            operation.work.kind == "token" and operation.work.mode == TokenMode.DECODE.value
            for operation in active
        )
        devices = (self._device,) if decode else self._completion_devices(active)
        for device in devices:
            scope.completion.begin_device(device)
        if decode:
            active_outcomes = self._decode_batch(active, scope)
        else:
            drivers = tuple(self._driver(operation, scope) for operation in active)
            active_outcomes = self._drive(drivers, scope)
        if active is operations:
            return active_outcomes
        resolved = dict(
            zip(
                (_operation_identity(operation) for operation in active),
                active_outcomes,
                strict=True,
            )
        )
        return tuple(
            self._predicated_outcome(operation, scope)
            if _operation_identity(operation) in scope.predicated_operations
            else resolved[_operation_identity(operation)]
            for operation in operations
        )

    def _predicated_outcome(
        self,
        operation: Operation,
        scope: _ExecutionScope,
    ) -> _Outcome:
        request = self._request_row(scope, operation.request_key.session_id)
        lengths = self._logical_lengths(operation, request, None)
        point = operation.parent.point
        selected_point = int(point.point_index)
        return _Outcome(
            status=OpStatus.PREDICATED,
            selected_point=selected_point,
            logical_lengths=lengths,
            token_span=TokenSpan(base=int(lengths.token_len), len=0),
            finish_flags=FinishFlags(),
            product_generations=(),
        )

    def _decode_batch(
        self,
        operations: tuple[Operation, ...],
        scope: _ExecutionScope,
    ) -> tuple[_Outcome, ...]:
        build_started = time.perf_counter_ns()
        starts: list[int] = []
        tasks: list[_ForwardTask] = []
        current_tokens = self._resolve_decode_tokens(operations, scope)
        layout = scope.layout
        if layout is None:
            raise RuntimeError("partition lost its aligned request-row view")
        if layout.operations == operations:
            requests = layout.requests
            cache_rows = layout.cache_rows
            weights = layout.weights
        else:
            aligned = {
                identity: (request, cache_row, weight)
                for identity, request, cache_row, weight in zip(
                    layout.identities,
                    layout.requests,
                    layout.cache_rows,
                    layout.weights,
                    strict=True,
                )
            }
            selected = tuple(aligned[_operation_identity(operation)] for operation in operations)
            requests = tuple(value[0] for value in selected)
            cache_rows = tuple(value[1] for value in selected)
            weights = tuple(value[2] for value in selected)
        starts.extend(int(request.logical_position) for request in requests)
        position_values = torch.tensor(starts, dtype=torch.long)
        for row_index, (operation, session, entry, weight, current) in enumerate(
            zip(
                operations,
                requests,
                cache_rows,
                weights,
                current_tokens,
                strict=True,
            )
        ):
            if session.sampling is None:
                raise invalid_descriptor("sequence operation has no admitted sampling state")
            tasks.append(
                self._token_task(
                    operation,
                    session,
                    (current,),
                    position_values[row_index : row_index + 1],
                    TokenSelection.LAST_LOGITS,
                    scope,
                    entry=entry,
                    weights=weight,
                )
            )

        _record_component(scope, "text_build_batch", build_started)
        forward_started = time.perf_counter_ns()
        outputs = self._run_wave(tuple(tasks), scope)
        _record_component(scope, "text_model_forward", forward_started)
        logits: list[torch.Tensor] = []
        for task, output in zip(tasks, outputs, strict=True):
            logits.append(_token_logits(output)[-1])
            self._commit_task_kv(task, 1, scope, publish_runtime=False)

        sample_started = time.perf_counter_ns()
        sample_tasks = tuple(
            self._sample_task(
                operation,
                row_logits,
                session,
                scope,
                positions=(start + 1,),
                request_pool_index=_request_pool_index(task),
            )
            for operation, session, task, start, row_logits in zip(
                operations,
                requests,
                tasks,
                starts,
                logits,
                strict=True,
            )
        )
        samples = _sample_task_batch(
            sample_tasks,
            scope.completion,
            device_products=self.device_products,
            device_reads=tuple(scope.device_reads),
            selection_broadcast=self._broadcast_tp_selection,
        )
        _record_component(scope, "text_sample", sample_started)
        finalize_started = time.perf_counter_ns()
        self._publish_token_products(operations, samples, scope)
        outcomes: list[_Outcome] = []
        for operation, session, entry, start, sampled in zip(
            operations,
            requests,
            cache_rows,
            starts,
            samples,
            strict=True,
        ):
            session.rng_counter += 1
            session.logical_position = start + 1
            # Keep the sampled token deferred: materializing it here (``int()``)
            # blocks on the copy event and stalls the decode pipeline. It is
            # finalized when the response is serialized, after the next forward
            # has launched.
            outcomes.append(
                self._token_outcome(
                    operation,
                    scope,
                    session=session,
                    kv_entry=entry,
                    base=start,
                    tokens=1,
                    committed_tokens=(sampled.token_id,),
                    sample=sampled,
                )
            )
        self._publish_runtime_samples(
            operations,
            requests,
            samples,
            scope=scope,
            sample_tasks=sample_tasks,
            logical_positions=tuple(start + 1 for start in starts),
            sampling_positions=tuple(request.rng_counter for request in requests),
            decode_increment=True,
        )
        _record_component(scope, "text_finalize", finalize_started)
        return tuple(outcomes)

    def _drive(self, drivers: tuple[_Driver, ...], scope: _ExecutionScope) -> tuple[_Outcome, ...]:
        active: dict[int, tuple[_Driver, tuple[_ModelTask, ...]]] = {}
        completed: dict[int, _Outcome] = {}
        for index, driver in enumerate(drivers):
            try:
                active[index] = (driver, next(driver))
            except StopIteration as done:
                completed[index] = done.value
        while active:
            flat: list[tuple[int, int, _ModelTask]] = []
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
            next_active: dict[int, tuple[_Driver, tuple[_ModelTask, ...]]] = {}
            for driver_index, (driver, _tasks) in active.items():
                aligned = tuple(by_driver[driver_index])
                try:
                    next_active[driver_index] = (driver, driver.send(aligned))
                except StopIteration as done:
                    completed[driver_index] = done.value
            active = next_active
        return tuple(completed[index] for index in range(len(drivers)))

    def _drive_partitioned(
        self,
        drivers: tuple[tuple[int, int, _Driver, _ExecutionScope], ...],
        *,
        qualify_mixed: bool,
    ) -> tuple[tuple[_Outcome | None, ...], dict[int, BaseException]]:
        active: dict[int, tuple[_Driver, tuple[_ModelTask, ...], _ExecutionScope]] = {}
        completed: dict[int, _Outcome] = {}
        errors: dict[int, BaseException] = {}
        for index, (_scope_index, _operation_index, driver, scope) in enumerate(drivers):
            partition_id = scope.partition.partition_id
            if partition_id in errors:
                continue
            try:
                active[index] = (driver, next(driver), scope)
            except StopIteration as done:
                completed[index] = done.value
            except BaseException as error:
                errors[partition_id] = error
                active = {
                    active_index: value
                    for active_index, value in active.items()
                    if value[2].partition.partition_id != partition_id
                }
        while active:
            flat: list[tuple[int, int, _ModelTask, _ExecutionScope]] = []
            for driver_index, (_driver, tasks, scope) in active.items():
                for task_index, task in enumerate(tasks):
                    flat.append((driver_index, task_index, task, scope))
            if not flat:
                raise RuntimeError("execution driver yielded an empty task wave")
            outputs, wave_errors = self._run_partitioned_task_wave(
                tuple((task, scope) for _driver, _task, task, scope in flat),
                qualify_mixed=qualify_mixed,
            )
            errors.update(wave_errors)
            by_driver: dict[int, list[Any | None]] = {
                index: [None] * len(tasks) for index, (_driver, tasks, _scope) in active.items()
            }
            for (driver_index, task_index, _task, _scope), output in zip(
                flat,
                outputs,
                strict=True,
            ):
                by_driver[driver_index][task_index] = output
            next_active: dict[
                int,
                tuple[_Driver, tuple[_ModelTask, ...], _ExecutionScope],
            ] = {}
            for driver_index, (driver, _tasks, scope) in active.items():
                partition_id = scope.partition.partition_id
                if partition_id in errors:
                    continue
                try:
                    next_active[driver_index] = (
                        driver,
                        driver.send(tuple(by_driver[driver_index])),
                        scope,
                    )
                except StopIteration as done:
                    completed[driver_index] = done.value
                except BaseException as error:
                    errors[partition_id] = error
                    next_active = {
                        active_index: value
                        for active_index, value in next_active.items()
                        if value[2].partition.partition_id != partition_id
                    }
            active = next_active
        return (
            tuple(completed.get(index) for index in range(len(drivers))),
            errors,
        )

    def _run_partitioned_task_wave(
        self,
        tasks: tuple[tuple[_ModelTask, _ExecutionScope], ...],
        *,
        qualify_mixed: bool,
    ) -> tuple[tuple[Any | None, ...], dict[int, BaseException]]:
        result: list[Any | None] = [None] * len(tasks)
        errors: dict[int, BaseException] = {}
        forward = tuple(
            (index, task, scope)
            for index, (task, scope) in enumerate(tasks)
            if isinstance(task, _ForwardTask)
        )
        if forward:
            indexes = tuple(index for index, _task, _scope in forward)
            outputs = self._run_partitioned_wave(
                tuple((task, scope) for _index, task, scope in forward),
                qualify_mixed=qualify_mixed,
            )
            for index, output in zip(indexes, outputs, strict=True):
                result[index] = output
        sampling: dict[int, list[tuple[int, _SampleTask, _ExecutionScope]]] = defaultdict(list)
        for index, (task, scope) in enumerate(tasks):
            if isinstance(task, _SampleTask):
                sampling[scope.partition.partition_id].append((index, task, scope))
        for candidates in sampling.values():
            scope = candidates[0][2]
            try:
                sample_outputs = _sample_task_batch(
                    tuple(task for _index, task, _scope in candidates),
                    scope.completion,
                    device_products=self.device_products,
                    device_reads=tuple(scope.device_reads),
                    selection_broadcast=self._broadcast_tp_selection,
                )
            except BaseException as error:
                errors[scope.partition.partition_id] = error
                continue
            for (index, _task, _scope), sample_output in zip(
                candidates, sample_outputs, strict=True
            ):
                result[index] = sample_output
        if any(
            value is None and scope.partition.partition_id not in errors
            for value, (_task, scope) in zip(result, tasks, strict=True)
        ):
            raise RuntimeError("model-runner task wave contains an unknown task type")
        return tuple(result), errors

    def _run_partitioned_wave(
        self,
        tasks: tuple[tuple[_ForwardTask, _ExecutionScope], ...],
        *,
        qualify_mixed: bool,
    ) -> tuple[torch.Tensor, ...]:
        grouped: dict[
            tuple[object, ...],
            list[tuple[int, _ForwardTask, _ExecutionScope]],
        ] = defaultdict(list)
        for index, (task, scope) in enumerate(tasks):
            grouped[
                (
                    scope.partition.submission_group,
                    self._model_invocation.partition_identity(
                        self._phase_device(task.phase), scope.partition.domain
                    ),
                    *self._group_key(task),
                )
            ].append((index, task, scope))

        result: list[torch.Tensor | None] = [None] * len(tasks)
        output_events: list[tuple[torch.device, torch.cuda.Event]] = []
        for group in grouped.values():
            kinds = frozenset(task.kind for _index, task, _scope in group)
            if len(kinds) > 1 and not self._model().tensorized_mixed:
                raise invalid_descriptor(
                    "tensorized mixed submission is outside the model capability"
                )
            indexes = tuple(index for index, _task, _scope in group)
            group_tasks = tuple(task for _index, task, _scope in group)
            group_scopes = tuple(scope for _index, _task, scope in group)
            target = self._phase_device(group_tasks[0].phase)
            for scope in _unique_scopes(group_scopes):
                scope.completion.register_device(target)
            if qualify_mixed and len(kinds) > 1:
                output, observation, mixed_us = self._run_startup_forward(
                    group_tasks,
                    group_scopes[0],
                    target,
                )
                mixed_output = tuple(value.clone() for value in output)
                homogeneous: dict[
                    str,
                    list[tuple[int, _ForwardTask, _ExecutionScope]],
                ] = defaultdict(list)
                for local_index, (_index, task, scope) in enumerate(group):
                    homogeneous[task.kind].append((local_index, task, scope))
                references: list[torch.Tensor | None] = [None] * len(group)
                reference_observations: list[RunObservation] = []
                homogeneous_us: list[int] = []
                for members in homogeneous.values():
                    reference, reference_observation, reference_us = self._run_startup_forward(
                        tuple(task for _index, task, _scope in members),
                        members[0][2],
                        target,
                        force_eager=observation.path is RunPath.EAGER,
                    )
                    reference_observations.append(reference_observation)
                    homogeneous_us.append(reference_us)
                    for (local_index, _task, _scope), value in zip(
                        members,
                        reference,
                        strict=True,
                    ):
                        references[local_index] = value
                if any(value is None for value in references):
                    raise RuntimeError("mixed qualification lost a homogeneous output row")
                self._assert_mixed_equivalence(
                    mixed_output,
                    tuple(cast(torch.Tensor, value) for value in references),
                    group_tasks,
                )
                service_paths = {
                    RunPath.EAGER,
                    RunPath.GRAPH_REPLAY,
                }
                if observation.path in service_paths:
                    if any(
                        reference.path is not observation.path
                        for reference in reference_observations
                    ):
                        raise GraphExecutionError(
                            "mixed and homogeneous service paths do not match"
                        )
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
                    capability = self._mixed_capability(
                        tuple(scope.partition for scope in _unique_scopes(group_scopes))
                    )
                    self._qualified_mixed_buckets.add(capability)
                    logger.info(
                        "qualified mixed execution bucket=%r mixed_us=%d homogeneous_us=%r "
                        "serial_over_mixed=%.3f",
                        capability,
                        mixed_us,
                        tuple(homogeneous_us),
                        serial_us / mixed_us,
                    )
                output = mixed_output
            else:
                output = self._run_forward_group(group_tasks, group_scopes[0])
                last_observation = self._model_invocation.last_observation
                if last_observation is None:
                    raise RuntimeError("model runner returned without an execution observation")
                observation = last_observation
            group_scopes[0].observations.append(observation)
            if (
                not (qualify_mixed and len(kinds) > 1)
                and self._model_invocation.last_output_event is not None
            ):
                output_events.append((target, self._model_invocation.last_output_event))
            for index, value in zip(indexes, output, strict=True):
                result[index] = value
        for device, event in output_events:
            torch.cuda.current_stream(device).wait_event(event)
        return tuple(cast(torch.Tensor, value) for value in result)

    def _run_startup_forward(
        self,
        tasks: tuple[_ForwardTask, ...],
        scope: _ExecutionScope,
        target: torch.device,
        *,
        force_eager: bool = False,
    ) -> tuple[tuple[torch.Tensor, ...], RunObservation, int]:
        with profile_range("uniserve.startup.mixed_oracle_forward"):
            if target.type != "cuda":
                started = time.perf_counter_ns()
                output = self._run_forward_group(tasks, scope, force_eager=force_eager)
                elapsed_us = max(1, (time.perf_counter_ns() - started) // 1000)
            else:
                stream = torch.cuda.current_stream(target)
                start = torch.cuda.Event(blocking=False, enable_timing=True)
                end = torch.cuda.Event(blocking=False, enable_timing=True)
                start.record(stream)
                output = self._run_forward_group(tasks, scope, force_eager=force_eager)
                output_event = self._model_invocation.last_output_event
                if output_event is not None:
                    stream.wait_event(output_event)
                end.record(stream)
                end.synchronize()
                elapsed_us = max(1, round(float(start.elapsed_time(end)) * 1000.0))
            observation = self._model_invocation.last_observation
            if observation is None:
                raise RuntimeError("model runner returned without an execution observation")
            return output, observation, elapsed_us

    def _assert_mixed_equivalence(
        self,
        mixed: tuple[torch.Tensor, ...],
        homogeneous: tuple[torch.Tensor, ...],
        tasks: tuple[_ForwardTask, ...],
    ) -> None:
        if len(mixed) != len(homogeneous) or len(mixed) != len(tasks):
            raise RuntimeError("mixed and homogeneous forwards returned different row counts")
        tolerances = {
            torch.bfloat16: (1.6e-2, 1.0e-5),
            torch.float16: (1.0e-3, 1.0e-5),
            torch.float32: (1.3e-6, 1.0e-5),
            torch.float64: (1.0e-7, 1.0e-7),
        }
        flow_rows: dict[_OperationIdentity, list[int]] = defaultdict(list)
        for row, (actual, expected, task) in enumerate(zip(mixed, homogeneous, tasks, strict=True)):
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise RuntimeError(f"mixed qualification row {row} changed output structure")
            if task.kind == "token":
                if not torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1)):
                    raise RuntimeError(
                        f"mixed qualification row {row} changed the committed greedy token"
                    )
                continue
            if task.kind != "flow":
                raise RuntimeError("mixed qualification contains an unsupported row kind")
            flow_rows[_operation_identity(task.operation)].append(row)

        flow = self._generation()
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
                    branch: _flow_prediction(values[row])
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

    def _run_task_wave(
        self,
        tasks: tuple[_ModelTask, ...],
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
                _sample_task_batch(
                    tuple(sample_tasks),
                    scope.completion,
                    device_products=self.device_products,
                    device_reads=tuple(scope.device_reads),
                    selection_broadcast=self._broadcast_tp_selection,
                ),
                strict=True,
            ):
                result[index] = sample_output
        if any(value is None for value in result):
            raise RuntimeError("model-runner task wave contains an unknown task type")
        return tuple(result)

    def _run_wave(
        self,
        tasks: tuple[_ForwardTask, ...],
        scope: _ExecutionScope,
    ) -> tuple[torch.Tensor, ...]:
        grouped: dict[tuple[object, ...], list[tuple[int, _ForwardTask]]] = defaultdict(list)
        for index, task in enumerate(tasks):
            grouped[self._group_key(task)].append((index, task))
        groups: list[list[tuple[int, _ForwardTask]]] = []
        for candidates in grouped.values():
            kinds = frozenset(task.kind for _index, task in candidates)
            if len(kinds) > 1 and not self._model().tensorized_mixed:
                by_kind: dict[str, list[tuple[int, _ForwardTask]]] = defaultdict(list)
                for item in candidates:
                    by_kind[item[1].kind].append(item)
                groups.extend(by_kind.values())
            else:
                groups.append(candidates)

        result: list[torch.Tensor | None] = [None] * len(tasks)
        output_events: list[tuple[torch.device, torch.cuda.Event]] = []
        for group in groups:
            indexes, group_tasks = zip(*group, strict=True)
            output = self._run_forward_group(tuple(group_tasks), scope)
            observation = self._model_invocation.last_observation
            if observation is None:
                raise RuntimeError("model runner returned without an execution observation")
            scope.observations.append(observation)
            if self._model_invocation.last_output_event is not None:
                output_events.append(
                    (
                        self._phase_device(group_tasks[0].phase),
                        self._model_invocation.last_output_event,
                    )
                )
            for index, value in zip(indexes, output, strict=True):
                result[index] = value
        for device, event in output_events:
            torch.cuda.current_stream(device).wait_event(event)
        return tuple(cast(torch.Tensor, value) for value in result)

    def _broadcast_tp_selection(self, value: torch.Tensor) -> torch.Tensor:
        if self.mesh.tp_size <= 1:
            return value
        transport = self.mesh.transport("tp")
        if not isinstance(transport, BroadcastTransport):
            raise RuntimeError("designated-rank sampling requires TP broadcast transport")
        return transport.broadcast(value, src=0)

    def _group_key(self, task: _ForwardTask) -> tuple[object, ...]:
        phase = (
            "textual"
            if task.phase in {ModelPhase.TEXT, ModelPhase.DENOISE}
            and self._model().tensorized_mixed
            else task.phase.value
        )
        return (
            phase,
            str(self._phase_device(task.phase)),
            task.weights.digest,
            task.weights.version,
            (
                ()
                if self._model().tensorized_mixed
                and not self._model_invocation.uses_lanes
                else self._task_shape(task)
            ),
            bool(task.write_kv),
        )

    def _run_forward_group(
        self,
        tasks: tuple[_ForwardTask, ...],
        scope: _ExecutionScope,
        *,
        force_eager: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        target = self._phase_device(tasks[0].phase)
        scope.completion.register_device(target)
        kv_tasks = tuple(task for task in tasks if task.write_kv)
        if kv_tasks and len(kv_tasks) != len(tasks):
            raise invalid_descriptor("one physical call cannot mix KV and non-KV rows")
        kv_view: KvView | EmptyKvView
        attention: AttnPlan
        if kv_tasks:
            kv_view, attention = self._attention_plan(tasks, scope)
        else:
            kv_view = EmptyKvView()
            attention = NoAttention(backends=self._attention_selection())
        mesh = RouteMeshView(self.mesh, self._phase_topology(tasks[0].phase))
        outputs = self._model_invocation.run(
            tasks,
            device=target,
            kv=kv_view,
            attention=attention,
            mesh=mesh,
            graph_shape=self._group_graph_shape(tasks),
            graph_eligible=(
                scope.graph_eligible
                and not force_eager
                and all(task.phase in {ModelPhase.TEXT, ModelPhase.DENOISE} for task in tasks)
            ),
            domain=scope.partition.domain,
        )
        request_pool_indices = self._model_invocation.last_request_pool_indices
        if request_pool_indices is None or int(request_pool_indices.numel()) != len(tasks):
            raise RuntimeError("model runner returned without aligned request slots")
        for index, task in enumerate(tasks):
            task.request_pool_index = request_pool_indices[index : index + 1]
        return outputs

    def _attention_plan(
        self,
        tasks: tuple[_ForwardTask, ...],
        scope: _ExecutionScope,
    ) -> tuple[KvView, PagedDecodePlan | PagedVarlenPlan | PackedAttentionPlan]:
        pure_token_decode = all(
            task.operation.work.variant is WorkVariant.TOKEN_DECODE
            and task.token_ids is not None
            and task.query_tokens == 1
            for task in tasks
        )
        if self._model().tensorized_mixed and not pure_token_decode:
            return self._packed_attention_plan(tasks, scope)
        query_lens = tuple(task.query_tokens for task in tasks)
        if any(task.entry is None for task in tasks):
            raise RuntimeError("paged attention task has no aligned KV entry")
        view = CacheBatchView.from_validated_rows(
            self.cache_pool,
            tuple(cast(CacheRow, task.entry) for task in tasks),
            query_lengths=query_lens,
        )
        host = torch.device("cpu")
        block_table = view.block_table(host)
        cache_seqlens = view.cache_seqlens(host)
        kv_lens = tuple(
            base + query for base, query in zip(view.base_lens, query_lens, strict=True)
        )
        context_capacity = int(block_table.shape[1]) * int(view.block_size)
        causal_values = {bool(task.causal) for task in tasks}
        if len(causal_values) != 1:
            raise invalid_descriptor("paged attention rows must share causal semantics")
        causal = causal_values.pop()
        binding = _binding_identity(tasks)
        if all(query == 1 for query in query_lens):
            page_ids = _stage_ints(
                tuple(
                    task.entry.block_table[task.entry.length // view.block_size]
                    for task in tasks
                    if task.entry is not None
                ),
                dtype=torch.int32,
            )
            page_offsets = _stage_ints(
                tuple(cast(CacheRow, task.entry).length % view.block_size for task in tasks),
                dtype=torch.int32,
            )
            decode_attention = PagedDecodePlan(
                backends=self._attention_selection(),
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                kv_seqlens=_stage_ints(
                    kv_lens,
                    dtype=torch.int32,
                ),
                query_lens=_stage_ints(
                    (1,) * len(tasks),
                    dtype=torch.int32,
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
        cu_q = _cumulative(query_lens)
        cu_k = _cumulative(kv_lens)
        varlen_attention = PagedVarlenPlan(
            backends=self._attention_selection(),
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            query_lens=_stage_ints(
                query_lens,
                dtype=torch.int32,
            ),
            kv_seqlens=_stage_ints(
                kv_lens,
                dtype=torch.int32,
            ),
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            output_indices=_stage_ints(
                tuple(sum(query_lens[: index + 1]) - 1 for index in range(len(query_lens))),
                dtype=torch.int64,
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
    ) -> tuple[KvView, PackedAttentionPlan]:
        if any(task.entry is None for task in tasks):
            raise RuntimeError("packed attention task has no aligned KV row")
        view = CacheBatchView(
            self.cache_pool,
            tuple(cast(CacheRow, task.entry) for task in tasks),
            tuple(task.query_tokens for task in tasks),
            tuple(task.write_kv for task in tasks),
        )
        query_lens = tuple(task.query_tokens for task in tasks)
        base_lens = view.base_lens
        key_lens = tuple(base + query for base, query in zip(base_lens, query_lens, strict=True))
        # The kernel checks ``visible_end`` against ``max_seqlen_q``, so the query
        # bound and the tensor it sizes are bucketed together: one executable then
        # serves a range of chunk widths instead of one per exact width. Positions
        # past a row's own query length stay zero, the padding this plan already
        # uses for rows shorter than the widest one.
        max_query = bucketed_length(max(query_lens))
        visible = torch.zeros((len(tasks), max_query), dtype=torch.int32)
        index_parts: list[torch.Tensor] = []
        route_spans: list[RouteSpan] = []
        offset = 0

        def append_span(route: ExpertRoute, count: int) -> None:
            nonlocal offset
            if count < 1:
                return
            if route_spans and route_spans[-1].route is route:
                previous = route_spans[-1]
                route_spans[-1] = RouteSpan(
                    route,
                    previous.token_start,
                    previous.token_count + count,
                )
            else:
                route_spans.append(RouteSpan(route, offset, count))
            offset += count

        for row, (task, base, query) in enumerate(zip(tasks, base_lens, query_lens, strict=True)):
            if task.causal:
                visible[row, :query] = torch.arange(
                    base + 1,
                    base + query + 1,
                    dtype=torch.int32,
                )
            else:
                visible[row, :query] = base + query
            indexes = task.attention_indexes
            if indexes is None:
                indexes = _three_axis_positions(task.positions, query)
            if tuple(indexes.shape) != (3, query):
                raise invalid_descriptor("packed attention indexes must have shape [3, query]")
            index_parts.append(indexes.to(device="cpu", dtype=torch.long))
            if task.token_ids is not None:
                append_span(ExpertRoute.TEXT, query)
            else:
                local_text = tuple(int(value) for value in task.text_local_indices)
                if local_text != tuple(sorted(set(local_text))) or any(
                    value < 0 or value >= query for value in local_text
                ):
                    raise invalid_descriptor("packed text-local indexes are invalid")
                cursor = 0
                local_index = 0
                while local_index < len(local_text):
                    run_start = local_text[local_index]
                    append_span(ExpertRoute.FLOW, run_start - cursor)
                    run_end = run_start + 1
                    local_index += 1
                    while local_index < len(local_text) and local_text[local_index] == run_end:
                        run_end += 1
                        local_index += 1
                    append_span(ExpertRoute.TEXT, run_end - run_start)
                    cursor = run_end
                append_span(ExpertRoute.FLOW, query - cursor)
        host = torch.device("cpu")
        write_page_ids, write_page_offsets, write_token_indices = view.write_plan(host)
        page_table = view.block_table(host)
        context_capacity = int(page_table.shape[1]) * int(view.block_size)
        attention = PackedAttentionPlan(
            backends=self._attention_selection(),
            indexes=torch.cat(index_parts, dim=1),
            route_spans=tuple(route_spans),
            visible_end=visible,
            cu_seqlens_q=_cumulative(query_lens),
            page_table=page_table,
            seqused_k=torch.tensor(key_lens, dtype=torch.int32),
            write_page_ids=write_page_ids,
            write_page_offsets=write_page_offsets,
            write_token_indices=write_token_indices,
            max_seqlen_q=max_query,
            max_seqlen_k=context_capacity,
            use_prefix_bounds=True,
            fully_visible=all(not task.causal for task in tasks),
            binding=_binding_identity(tasks),
            query_lens_cpu=query_lens,
            key_lens_cpu=key_lens,
        )
        return view, attention

    def _weights(self) -> WeightSet:
        return self.weights

    def _model(self) -> ExecutionModel:
        return self.model

    def _generation(self) -> GenerationPipeline:
        value = self._model().generation
        if not isinstance(value, GenerationPipeline):
            raise invalid_descriptor("operation requires model generation behavior")
        return value

    def _latent_pool(self) -> LatentPool:
        if self.latent_pool is None:
            raise capability_mismatch("operation requires a physical latent pool")
        return self.latent_pool

    def _image_processor(self) -> ImageProcessor:
        value = self._model().image_processor
        if not isinstance(value, ImageProcessor):
            raise invalid_descriptor("operation requires model image processing")
        return value

    def _phase_device(self, phase: ModelPhase) -> torch.device:
        deployment = self.deployment
        if phase in {ModelPhase.ENCODE_LATENT, ModelPhase.DECODE_LATENT}:
            return torch.device(deployment.generation_device or deployment.device)
        return torch.device(deployment.device)

    def _phase_topology(self, phase: ModelPhase) -> tuple[str, ...]:
        if phase in {ModelPhase.TEXT, ModelPhase.DENOISE}:
            return self._model().text_topology
        return ("tp",)

    @staticmethod
    def _task_shape(task: _ForwardTask) -> tuple[int, ...]:
        if task.encode_pixels is not None:
            return tuple(int(value) for value in task.encode_pixels.shape)
        if task.latent is not None:
            return task.image_height, task.image_width
        return ()

    @staticmethod
    def _group_graph_shape(tasks: tuple[_ForwardTask, ...]) -> tuple[object, ...]:
        return (
            len(tasks),
            sum(task.query_tokens for task in tasks),
            tuple(task.query_tokens for task in tasks),
            tuple(
                (task.image_height, task.image_width) for task in tasks if task.latent is not None
            ),
        )

    def _attention_selection(self) -> AttentionSelection:
        if self.attention is None:
            raise RuntimeError("model route has no attention selection")
        return self.attention

    def _release_locators(self, locators: Iterable[Locator]) -> None:
        if self.transport is None:
            return
        for locator in locators:
            self.transport.release(locator)

    def _sequence_driver(
        self,
        operation: Operation,
        scope: _ExecutionScope,
    ) -> _Driver:
        session = self._request_row(scope, operation.request_key.session_id)
        if session.sampling is None:
            raise invalid_descriptor("sequence operation has no admitted sampling state")
        mode = operation.work.mode
        if mode == TokenMode.EXTEND.value:
            if any(
                reference.kind in {ProductKind.VISION_FEATURE, ProductKind.LATENT_FEATURE}
                for reference in operation.inputs
            ):
                return (yield from self._visual_extend(operation, session, scope))
            return (yield from self._extend(operation, session, scope))
        if mode == TokenMode.DECODE.value:
            return (yield from self._decode(operation, session, scope))
        return (yield from self._verify(operation, session, scope))

    def _extend(
        self,
        operation: Operation,
        session: RequestRow,
        scope: _ExecutionScope,
    ) -> _Driver:
        tokens: tuple[int | torch.Tensor, ...]
        if isinstance(operation.parent.point, DevicePoint) and not any(
            reference.kind is ProductKind.TOKEN for reference in operation.inputs
        ):
            tokens = (self._resolve_decode_token(operation, session, scope),)
        else:
            tokens = self._operation_token_ids(operation, scope)
        start = session.logical_position
        sampling = _require_sampling(session)
        scores_prompt = bool(sampling.return_prompt_logprobs or int(sampling.n_prompt_logprobs) > 0)
        task = self._token_task(
            operation,
            session,
            tokens,
            tuple(range(start, start + len(tokens))),
            TokenSelection.ALL_LOGITS if scores_prompt else TokenSelection.LAST_LOGITS,
            scope,
        )
        outputs = yield (task,)
        logits = _token_logits(outputs[0])
        self._commit_task_kv(task, len(tokens), scope)
        if not any(output.kind is ProductKind.TOKEN for output in operation.outputs):
            return self._token_outcome(
                operation,
                scope,
                base=start,
                tokens=0,
                committed_tokens=(),
            )
        sample_task = self._sample_task(
            operation,
            logits[-1],
            session,
            scope,
            positions=(start + len(tokens),),
            request_pool_index=_request_pool_index(task),
        )
        sample = _sample_result((yield (sample_task,))[0])
        if scores_prompt:
            sample = replace(
                sample,
                prompt_logprobs=self._prompt_logprob_details(
                    session,
                    start,
                    cast(torch.Tensor, task.token_ids),
                    logits,
                    scope,
                ),
            )
        self._publish_token_product(operation, sample, scope)
        session.rng_counter += 1
        session.logical_position = start + len(tokens)
        self._publish_runtime_samples(
            (operation,),
            (session,),
            (sample,),
            scope=scope,
            sample_tasks=(sample_task,),
            logical_positions=(session.logical_position,),
            sampling_positions=(session.rng_counter,),
        )
        # Keep the sampled token deferred (see _decode_batch): a plain-greedy
        # extend still copies its token asynchronously, and forcing it here
        # would reintroduce the per-step host stall.
        return self._token_outcome(
            operation,
            scope,
            base=start,
            tokens=len(tokens),
            committed_tokens=(sample.token_id,),
            sample=sample,
        )

    def _prompt_logprob_details(
        self,
        session: RequestRow,
        start: int,
        tokens: torch.Tensor,
        logits: torch.Tensor,
        scope: _ExecutionScope,
    ) -> tuple[
        tuple[tuple[int, float, int], ...] | _CompletionTopLogprobs,
        ...,
    ]:
        tokens = tokens.reshape(-1).to(device=logits.device, dtype=torch.long)
        if logits.ndim != 2 or int(logits.shape[0]) != int(tokens.numel()):
            raise invalid_descriptor("prompt scoring logits do not align with input tokens")
        states = self.runtime_states
        if states is None:
            raise capability_mismatch("prompt scoring has no request-indexed runtime state")
        slot = int(session.request_pool_idx)
        if start == 0:
            score_logits = logits[:-1]
            targets = tokens[1:]
        else:
            if not session.prompt_logits_ready:
                raise invalid_descriptor("continued prompt scoring has no preceding logits")
            pending = next(
                (
                    publication.logits
                    for publication in reversed(scope.prompt_logits_publications)
                    if publication.slot == slot
                ),
                states.prompt_logits[slot],
            )
            previous = pending.reshape(1, -1).to(
                device=logits.device,
                dtype=logits.dtype,
            )
            score_logits = torch.cat((previous, logits[:-1]), dim=0)
            targets = tokens
        scope.prompt_logits_publications.append(
            _PromptLogitsPublication(slot=slot, logits=logits[-1].detach())
        )
        session.prompt_logits_ready = True
        if int(targets.numel()) == 0:
            return ()
        sampling = _require_sampling(session)
        prompt_parameters = replace(
            sampling,
            return_logprobs=True,
            n_logprobs=int(sampling.n_prompt_logprobs),
        )
        rows = tuple(
            _SamplingRow(
                parameters=prompt_parameters,
                penalty_counts=None,
                allowed=None,
                suppress=(),
                draw=0.0,
                n_logprobs=int(sampling.n_prompt_logprobs),
            )
            for _ in range(int(targets.numel()))
        )
        indexes = torch.arange(
            int(targets.numel()),
            dtype=torch.long,
            device=score_logits.device,
        )
        details = _sample_logprob_details(
            score_logits.float(),
            indexes,
            targets,
            rows,
            scope.completion,
        )
        return tuple(details[index][1] for index in range(int(targets.numel())))

    def _visual_extend(
        self,
        operation: Operation,
        session: RequestRow,
        scope: _ExecutionScope,
    ) -> _Driver:
        references = tuple(
            reference
            for reference in operation.inputs
            if reference.kind in {ProductKind.VISION_FEATURE, ProductKind.LATENT_FEATURE}
        )
        if len(references) != 1:
            raise invalid_descriptor("visual extend requires exactly one feature product")
        reference = references[0]
        read = self._consume_encoder_feature(
            reference,
            scope,
            consumer_op_id=operation.op_id,
            device=self._operation_device(operation),
        )
        scope.encoder_reads.append(read)
        position = session.logical_position
        closes_feedback = any(output.kind is ProductKind.COMPLETION for output in operation.outputs)
        samples_continuation = any(output.kind is ProductKind.TOKEN for output in operation.outputs)
        if reference.kind is ProductKind.VISION_FEATURE:
            outcome = yield from self._state_driver(
                operation,
                WorkVariant.ENCODE_VISION,
                scope,
                height=read.metadata.height,
                width=read.metadata.width,
                conditioning_position=position,
                features=read.tensor,
                sample_token=samples_continuation,
                close_image=closes_feedback,
                retain_image=True,
            )
        else:
            outcome = yield from self._state_driver(
                operation,
                WorkVariant.ENCODE_LATENT,
                scope,
                height=read.metadata.height,
                width=read.metadata.width,
                conditioning_position=position,
                latent=read.tensor,
                sample_token=samples_continuation,
                close_image=closes_feedback,
                retain_image=True,
            )
        if closes_feedback:
            flow = self.model.generation if self.model is not None else None
            session.logical_position = position + max(
                1,
                1 if flow is None else int(flow.rope_advance),
            )
        elif reference.kind is ProductKind.VISION_FEATURE:
            session.logical_position = position + 1
        return self._state_outcome(operation, outcome, scope, base=position)

    def _decode(
        self,
        operation: Operation,
        session: RequestRow,
        scope: _ExecutionScope,
    ) -> _Driver:
        current = self._resolve_decode_token(operation, session, scope)
        start = session.logical_position
        task = self._token_task(
            operation,
            session,
            (current,),
            (start,),
            TokenSelection.LAST_LOGITS,
            scope,
        )
        outputs = yield (task,)
        logits = _token_logits(outputs[0])[-1]
        self._commit_task_kv(task, 1, scope)
        sample_task = self._sample_task(
            operation,
            logits,
            session,
            scope,
            positions=(start + 1,),
            request_pool_index=_request_pool_index(task),
        )
        sampled = _sample_result((yield (sample_task,))[0])
        session.rng_counter += 1
        session.logical_position = start + 1
        self._publish_token_product(operation, sampled, scope)
        self._publish_runtime_samples(
            (operation,),
            (session,),
            (sampled,),
            scope=scope,
            sample_tasks=(sample_task,),
            logical_positions=(session.logical_position,),
            sampling_positions=(session.rng_counter,),
        )
        # Keep the sampled token deferred (see _decode_batch): finalized when the
        # response is serialized, off the decode critical path.
        return self._token_outcome(
            operation,
            scope,
            base=start,
            tokens=1,
            committed_tokens=(sampled.token_id,),
            sample=sampled,
        )

    def _verify(
        self,
        operation: Operation,
        session: RequestRow,
        scope: _ExecutionScope,
    ) -> _Driver:
        if isinstance(operation.parent.point, DevicePoint):
            current = self._resolve_decode_token(operation, session, scope)
            draft = self._operation_token_ids(operation, scope)
        else:
            input_tokens = self._operation_token_ids(operation, scope)
            if len(input_tokens) < 2:
                raise invalid_descriptor(
                    "fixed-parent verification requires current and draft tokens"
                )
            current = int(input_tokens[0])
            draft = input_tokens[1:]
        start = session.logical_position
        tokens: tuple[int | torch.Tensor, ...] = (current, *draft)
        task = self._token_task(
            operation,
            session,
            tokens,
            tuple(range(start, start + len(tokens))),
            TokenSelection.ALL_LOGITS,
            scope,
        )
        outputs = yield (task,)
        logits = _token_logits(outputs[0])
        sample_task = self._sample_task(
            operation,
            logits,
            session,
            scope,
            positions=tuple(range(start + 1, start + len(draft) + 2)),
            draft_token_ids=draft,
            request_pool_index=_request_pool_index(task),
        )
        sampled = _sample_result((yield (sample_task,))[0])
        if task.entry is None:
            raise RuntimeError("verification task has no scheduler KV row")
        initialized = task.entry.initialize(task.query_tokens)
        self._publish_token_product(operation, sampled, scope)
        accepted = cast(_CompletionInteger, sampled.num_accepted_tokens)
        selected_point = _CompletionSpeculativePoint(
            accepted,
            sample_task.terminal_draft_prefix,
        )
        committed_tokens = _CompletionSpeculativeTokens(
            draft,
            accepted,
            sampled.token_id,
            sample_task.terminal_draft_prefix,
        )
        device_selected = sampled.device_selected_point
        if device_selected is None:
            accepted_device = sampled.device_accepted_tokens
            if accepted_device is None:
                raise RuntimeError("speculative sampling lost its selected point")
            device_selected = accepted_device.to(dtype=torch.int32) + 1
        scope.runtime_cache_lengths[int(session.request_pool_idx)] = device_selected + int(
            initialized - task.query_tokens
        )
        self._publish_runtime_samples(
            (operation,),
            (session,),
            (sampled,),
            scope=scope,
            sample_tasks=(sample_task,),
            logical_positions=(device_selected + int(start),),
            sampling_positions=(device_selected + int(session.rng_counter),),
        )
        return self._token_outcome(
            operation,
            scope,
            base=start,
            tokens=selected_point,
            committed_tokens=cast(tuple[int | _CompletionToken, ...], committed_tokens),
            sample=sampled,
            selection=_SpeculativeSelection(
                accepted=accepted,
                selected_point=selected_point,
                draft_tokens=draft,
                terminal_prefix=sample_task.terminal_draft_prefix,
                base_logical_position=start,
                base_rng_counter=session.rng_counter,
                base_kv_visible=initialized - task.query_tokens,
                initialized_kv=initialized,
            ),
        )

    def _token_outcome(
        self,
        operation: Operation,
        scope: _ExecutionScope,
        *,
        session: RequestRow | None = None,
        kv_entry: CacheRow | None = None,
        base: int,
        tokens: int | _CompletionDerivedInteger | _CompletionSpeculativePoint,
        committed_tokens: tuple[int | _CompletionToken, ...],
        sample: _SampleResult | None = None,
        selection: _SpeculativeSelection | None = None,
    ) -> _Outcome:
        if session is None:
            session = self._request_row(scope, operation.request_key.session_id)
        row = self._cache_row(operation, scope) if kv_entry is None else kv_entry
        extents = row.extents()
        visible = (
            extents.visible
            if selection is None
            else _CompletionDerivedInteger(
                cast(_CompletionInteger, selection.selected_point),
                selection.base_kv_visible,
            )
        )
        token_len = (
            session.logical_position
            if selection is None
            else _CompletionDerivedInteger(
                cast(_CompletionInteger, selection.selected_point),
                selection.base_logical_position,
            )
        )
        if sample is not None and selection is not None:
            self._publish_selection_products(
                operation,
                sample,
                scope,
                logical_position=session.logical_position,
                kv_visible=extents.visible,
                selection=selection,
            )
        return _Outcome(
            status=OpStatus.OK,
            selected_point=(1 if selection is None else selection.selected_point),
            logical_lengths=LogicalLengths(
                token_len=cast(int, token_len),
                kv_visible_len=cast(int, visible),
                kv_reserved_len=extents.reserved,
                kv_initialized_len=extents.initialized,
                kv_committed_len=extents.committed,
                kv_published_len=extents.published,
            ),
            token_span=TokenSpan(base=base, len=cast(int, tokens)),
            finish_flags=FinishFlags(),
            product_generations=_output_generations(operation),
            committed_tokens=committed_tokens,
            products=_sample_product_payloads(operation, sample),
            selection=selection,
        )

    def _token_task(
        self,
        operation: Operation,
        session: RequestRow,
        token_ids: tuple[int | torch.Tensor, ...],
        positions: tuple[int, ...] | torch.Tensor,
        selection: TokenSelection,
        scope: _ExecutionScope,
        *,
        entry: CacheRow | None = None,
        weights: WeightSet | None = None,
    ) -> _ForwardTask:
        if len(token_ids) != len(positions) or not token_ids:
            raise invalid_descriptor("token task ids and positions must align")
        if len(token_ids) == 1 and isinstance(token_ids[0], torch.Tensor):
            token_values = token_ids[0].reshape(1).to(dtype=torch.long)
        else:
            token_values = torch.tensor(
                tuple(int(value) for value in token_ids),
                dtype=torch.long,
            )
        return _ForwardTask(
            operation=operation,
            request=session,
            weights=self._weights() if weights is None else weights,
            phase=ModelPhase.TEXT,
            token_ids=token_values,
            positions=(
                positions.reshape(-1).to(dtype=torch.long)
                if isinstance(positions, torch.Tensor)
                else torch.tensor(positions, dtype=torch.long)
            ),
            selection=selection,
            entry=self._cache_row(operation, scope) if entry is None else entry,
            write_kv=True,
            causal=True,
        )

    def _commit_task_kv(
        self,
        task: _ForwardTask,
        tokens: int,
        scope: _ExecutionScope,
        *,
        publish_runtime: bool = True,
    ) -> None:
        count = int(tokens)
        if count < 0 or count > task.query_tokens:
            raise RuntimeError("KV commit count is outside the task query span")
        if count == 0:
            return
        if task.entry is None:
            raise RuntimeError("KV task has no scheduler cache row")
        task.entry.advance(count)
        if publish_runtime and self.runtime_states is not None:
            slot = int(task.request.request_pool_idx)
            scope.runtime_cache_lengths[slot] = int(task.entry.length)

    def _operation_token_ids(
        self,
        operation: Operation,
        scope: _ExecutionScope,
    ) -> tuple[int, ...]:
        """Read the token id values a token operation names as an input product.

        Host-known prompt and draft tokens arrive through a declared host-staging
        product. Device-rooted decode tokens are read from request-indexed runtime
        state.
        """

        for reference in operation.inputs:
            if reference.kind is not ProductKind.TOKEN:
                continue
            values = scope.input_tokens.get(reference)
            if values is not None:
                return values
        raise invalid_descriptor("token operation has no input token product")

    def _resolve_decode_token(
        self,
        operation: Operation,
        session: RequestRow,
        scope: _ExecutionScope,
    ) -> int | torch.Tensor:
        point = operation.parent.point
        if isinstance(point, DevicePoint):
            predicate = scope.predicate_values.get(_operation_identity(operation))
            if predicate is None:
                raise invalid_descriptor("device token continuation is not registered")
            states = self.runtime_states
            if states is None:
                raise capability_mismatch("device continuation has no request runtime state")
            slot = int(session.request_pool_idx)
            pending = self._pending_runtime_token(slot, scope)
            if pending is not None:
                return pending.reshape(-1)[:1].bitwise_and(TOKEN_VALUE_MASK)
            return states.future_input_tokens[slot, :1]
        tokens = self._operation_token_ids(operation, scope)
        if not tokens:
            raise invalid_descriptor("last-sampled token source has no committed token")
        return int(tokens[0])

    def _resolve_decode_tokens(
        self,
        operations: tuple[Operation, ...],
        scope: _ExecutionScope,
    ) -> tuple[int | torch.Tensor, ...]:
        resolved: list[int | torch.Tensor | None] = [None] * len(operations)
        for index, operation in enumerate(operations):
            point = operation.parent.point
            if isinstance(point, DevicePoint):
                predicate = scope.predicate_values.get(_operation_identity(operation))
                if predicate is None:
                    raise invalid_descriptor("device token continuation is not registered")
                states = self.runtime_states
                if states is None:
                    raise capability_mismatch("device continuation has no request runtime state")
                session = self._request_row(scope, operation.request_key.session_id)
                slot = int(session.request_pool_idx)
                pending = self._pending_runtime_token(slot, scope)
                resolved[index] = (
                    states.future_input_tokens[slot, :1]
                    if pending is None
                    else pending.reshape(-1)[:1].bitwise_and(TOKEN_VALUE_MASK)
                )
                continue
            tokens = self._operation_token_ids(operation, scope)
            if not tokens:
                raise invalid_descriptor("last-sampled token source has no committed token")
            resolved[index] = int(tokens[0])
        if any(value is None for value in resolved):
            raise RuntimeError("decode token resolution left an operation without input")
        return tuple(cast(int | torch.Tensor, value) for value in resolved)

    @staticmethod
    def _pending_runtime_token(
        slot: int,
        scope: _ExecutionScope,
    ) -> torch.Tensor | None:
        for publication in reversed(scope.runtime_publications):
            if isinstance(publication, _RuntimePublication):
                if publication.slot == slot:
                    return publication.token
                continue
            try:
                index = publication.slots.index(slot)
            except ValueError:
                continue
            return publication.tokens[index : index + 1]
        return None

    def _publish_token_product(
        self,
        operation: Operation,
        sample: _SampleResult,
        scope: _ExecutionScope,
    ) -> None:
        if sample.device_product_published:
            return
        write = scope.token_writes.get(_operation_identity(operation))
        if write is None:
            return
        device_token = sample.device_token
        if device_token is None:
            device_token = torch.tensor(
                (int(sample.token_id),),
                dtype=torch.long,
                device=self._operation_device(operation),
            )
        continuation = sample.device_continuation
        if continuation is None:
            continuation = torch.ones_like(device_token, dtype=torch.bool)
        self.device_products.publish_write(
            write,
            _tagged_token_values(device_token, continuation),
        )

    def _publish_runtime_samples(
        self,
        operations: Sequence[Operation],
        sessions: Sequence[RequestRow],
        samples: Sequence[_SampleResult],
        *,
        scope: _ExecutionScope,
        sample_tasks: Sequence[_SampleTask],
        logical_positions: Sequence[int | torch.Tensor],
        sampling_positions: Sequence[int | torch.Tensor],
        decode_increment: bool = False,
    ) -> None:
        states = self.runtime_states
        if states is None:
            return
        columns = (
            operations,
            sessions,
            samples,
            sample_tasks,
            logical_positions,
            sampling_positions,
        )
        if len({len(values) for values in columns}) != 1:
            raise RuntimeError("runtime sampling publication columns are not aligned")
        if decode_increment:
            if any(
                operation.request_key != session.request_key
                for operation, session in zip(operations, sessions, strict=True)
            ):
                raise RuntimeError("runtime sampling publication crossed request rows")
            selected_values = tuple(sample.device_selected_point for sample in samples)
            accepted_values = tuple(sample.device_accepted_tokens for sample in samples)
            selected_points = (
                None
                if all(value is None for value in (*selected_values, *accepted_values))
                else _runtime_selected_points(selected_values, accepted_values)
            )
            scope.runtime_publications.append(
                _DecodeRuntimePublication(
                    slots=tuple(int(session.request_pool_idx) for session in sessions),
                    device_slots=_sample_request_pool_indices(sample_tasks),
                    tokens=_sample_result_vector(samples, "device_token"),
                    predicates=_sample_result_vector(samples, "device_continuation"),
                    selected_points=selected_points,
                    penalty_bases=tuple(task.penalty_base for task in sample_tasks),
                    valid=_sample_result_vector(samples, "device_valid"),
                    active=_sample_result_vector(samples, "device_active"),
                )
            )
            return
        for operation, session, sample, sample_task, logical, sampling in zip(
            operations,
            sessions,
            samples,
            sample_tasks,
            logical_positions,
            sampling_positions,
            strict=True,
        ):
            if operation.request_key != session.request_key:
                raise RuntimeError("runtime sampling publication crossed request rows")
            token = sample.device_token
            predicate = sample.device_continuation
            valid = sample.device_valid
            active = sample.device_active
            if token is None or predicate is None or valid is None or active is None:
                raise RuntimeError("runtime sampling publication lost device state")
            selected = sample.device_selected_point
            if selected is None:
                accepted = sample.device_accepted_tokens
                selected = (
                    torch.ones_like(token, dtype=torch.int32)
                    if accepted is None
                    else accepted.to(dtype=torch.int32) + 1
                )
            slot = int(session.request_pool_idx)
            scope.runtime_publications.append(
                _RuntimePublication(
                    slot=slot,
                    token=token,
                    predicate=predicate,
                    selected_point=selected,
                    logical_position=logical,
                    sampling_position=sampling,
                    penalty_base=sample_task.penalty_base,
                    valid=valid,
                    active=active,
                )
            )

    def _publish_token_products(
        self,
        operations: tuple[Operation, ...],
        samples: tuple[_SampleResult, ...],
        scope: _ExecutionScope,
    ) -> None:
        if all(sample.device_product_published for sample in samples):
            return
        writes: list[DeviceProductWrite] = []
        device_tokens: list[torch.Tensor] = []
        for operation, sample in zip(operations, samples, strict=True):
            write = scope.token_writes.get(_operation_identity(operation))
            if write is None or sample.device_token is None:
                for candidate_operation, candidate_sample in zip(
                    operations,
                    samples,
                    strict=True,
                ):
                    self._publish_token_product(
                        candidate_operation,
                        candidate_sample,
                        scope,
                    )
                return
            writes.append(write)
            device_tokens.append(sample.device_token)
        packed = packed_tensor_views(tuple(device_tokens))
        if packed is None:
            for operation, sample in zip(operations, samples, strict=True):
                self._publish_token_product(operation, sample, scope)
            return
        continuation_flags = tuple(
            sample.device_continuation
            for sample in samples
            if sample.device_continuation is not None
        )
        if len(continuation_flags) != len(samples):
            raise RuntimeError("sampled token publication lost its continuation state")
        packed_flags = packed_tensor_views(continuation_flags)
        if packed_flags is None:
            packed_flags = torch.cat(continuation_flags, dim=0)
        self.device_products.publish_writes(
            tuple(writes),
            _tagged_token_values(packed, packed_flags),
        )

    def _publish_selection_products(
        self,
        operation: Operation,
        sample: _SampleResult,
        scope: _ExecutionScope,
        *,
        logical_position: int,
        kv_visible: int,
        selection: _SpeculativeSelection | None,
    ) -> None:
        device_token = sample.device_token
        if device_token is None:
            raise RuntimeError("device selection products require a device token")
        accepted = sample.device_accepted_tokens
        if accepted is None:
            accepted = torch.zeros_like(device_token, dtype=torch.long)
        selected_point = sample.device_selected_point
        if selected_point is None:
            selected_point = accepted.to(dtype=torch.long) + 1
        semantic_token = device_token.reshape(-1).to(dtype=torch.long).bitwise_and(TOKEN_VALUE_MASK)
        operation_identity = _operation_identity(operation)
        selected_write = scope.selected_point_writes.get(operation_identity)
        span_write = scope.accepted_span_writes.get(operation_identity)
        continuation_write = scope.state_continuation_writes.get(operation_identity)
        if selected_write is None:
            raise invalid_descriptor("token operation is missing its selected-point product")
        self.device_products.publish_write(selected_write, selected_point)
        if (
            span_write is None
            and continuation_write is None
            and int(operation.bounds.max_points) == 1
        ):
            return
        if span_write is None or continuation_write is None:
            raise invalid_descriptor("multi-point token operation is missing its branch products")
        if selection is None:
            candidates = semantic_token
            logical = torch.full_like(selected_point, int(logical_position))
            visible = torch.full_like(selected_point, int(kv_visible))
        else:
            draft = torch.tensor(
                selection.draft_tokens,
                dtype=torch.long,
                device=device_token.device,
            )
            candidates = torch.cat((draft, semantic_token))
            logical = selected_point + int(selection.base_logical_position)
            visible = selected_point + int(selection.base_kv_visible)
        max_points = int(operation.bounds.max_points)
        if int(candidates.numel()) != max_points:
            raise RuntimeError("selection candidates do not match the operation point bound")
        indexes = torch.arange(max_points, dtype=torch.long, device=device_token.device)
        visible_tokens = torch.where(
            indexes < selected_point.reshape(()),
            candidates,
            torch.zeros_like(candidates),
        )
        self.device_products.publish_write(
            span_write,
            torch.cat((selected_point.reshape(-1), visible_tokens)),
        )
        self.device_products.publish_write(
            continuation_write,
            torch.stack(
                (
                    semantic_token[0],
                    selected_point.reshape(-1)[0],
                    visible.reshape(-1)[0],
                    logical.reshape(-1)[0],
                )
            ),
        )

    def _sample_task(
        self,
        operation: Operation,
        logits: torch.Tensor,
        session: RequestRow,
        scope: _ExecutionScope,
        *,
        positions: tuple[int, ...],
        request_pool_index: torch.Tensor,
        draft_token_ids: tuple[int, ...] = (),
    ) -> _SampleTask:
        sampling = _require_sampling(session)
        state = scope.sampling_states.get(_operation_identity(operation), SamplingState())
        allowed_token_ids = (
            state.allowed_token_ids
            if state.allowed_token_ids is not None
            else sampling.allowed_token_ids
        )
        if not state.finish_token_ids:
            finish_token_ids = session.finish_token_ids
        elif not session.finish_token_ids:
            finish_token_ids = state.finish_token_ids
        else:
            finish_token_ids = tuple(
                sorted(
                    {
                        *session.finish_token_ids,
                        *state.finish_token_ids,
                    }
                )
            )
        rng = operation.rng
        if float(sampling.temperature) > 0.0:
            if rng is None or rng.draw_layout is not DrawLayout.TARGET_SAMPLING:
                raise invalid_descriptor(
                    "stochastic sampling requires target-sampling RNG coordinates"
                )
            if int(rng.seed) != int(sampling.seed or 0):
                raise invalid_descriptor("operation RNG seed disagrees with admitted sampling")
            expected_positions = tuple(
                range(int(rng.semantic_index_base), int(rng.semantic_index_base) + len(positions))
            )
            if positions != expected_positions:
                raise invalid_descriptor(
                    "sampling positions disagree with registered semantic RNG coordinates"
                )
        rng_seed = 0 if rng is None else int(rng.seed)
        stochastic = float(sampling.temperature) > 0.0
        draw_key = (
            sampling_key(
                rng_seed,
                int(operation.request_key.authority_id),
                int(operation.request_key.session_id),
                int(operation.request_key.epoch),
                DRAW_LAYOUT_TARGET,
            )
            if stochastic
            else 0
        )
        rows = logits.reshape(1, -1) if logits.ndim == 1 else logits
        if rows.ndim != 2 or int(rows.shape[0]) != len(positions):
            raise invalid_descriptor("sampling task positions do not align with its logits")
        vocab = int(rows.shape[1])
        uses_penalties = (
            sampling.repetition_penalty != 1.0
            or sampling.frequency_penalty != 0.0
            or sampling.presence_penalty != 0.0
        )
        penalty_base = (
            self._session_penalty_base(session, vocab, rows.device) if uses_penalties else None
        )
        penalty_view = (
            None
            if penalty_base is None
            else self._candidate_penalty_counts(session, penalty_base, scope)
        )
        forced_token_ids = sampling.forced_token_ids
        descriptors: list[_SamplingRow] = []
        for index, position in enumerate(positions):
            if penalty_view is None:
                row_counts = None
            elif index == 0 or not draft_token_ids:
                row_counts = penalty_view
            else:
                row_counts = penalty_view.clone()
                for token_id in draft_token_ids[:index]:
                    row_counts[int(token_id)] += 1
            # Processor step 2 forced-token constraint: point `index` of the
            # operation's span narrows selection to `forced_token_ids[index]`,
            # overriding any allowed-token whitelist for that point.
            row_allowed = (
                (int(forced_token_ids[index]),)
                if index < len(forced_token_ids)
                else allowed_token_ids
            )
            descriptors.append(
                _SamplingRow(
                    parameters=sampling,
                    penalty_counts=row_counts,
                    allowed=row_allowed,
                    suppress=state.suppressed_token_ids,
                    finish_token_ids=finish_token_ids,
                    transition_token_ids=state.transition_token_ids,
                    force_finish=state.force_finish,
                    draw=(sampling_uniform(draw_key, int(position)) if stochastic else 0.0),
                    n_logprobs=int(sampling.n_logprobs),
                )
            )
        descriptor_rows = tuple(descriptors)
        operation_identity = _operation_identity(operation)
        device_greedy = not draft_token_ids and all(
            _device_greedy_row(row) for row in descriptor_rows
        )
        if device_greedy:
            draws = None
            penalty_token_ids = None
            penalty_counts = None
            parameter_values = None
        else:
            draws = _semantic_sampling_draws(
                descriptor_rows,
                device=rows.device,
            )
            penalty_token_ids, penalty_counts, parameter_values = _sampling_task_tensors(
                descriptor_rows,
                vocab=vocab,
                device=rows.device,
            )
        token_product = scope.token_writes.get(operation_identity)
        finish_product = scope.finish_writes.get(operation_identity)
        transition_product = scope.transition_writes.get(operation_identity)
        predicate_value = scope.predicate_values.get(operation_identity)
        finish_set = set(descriptor_rows[0].finish_token_ids)
        terminal_draft_prefix = next(
            (index + 1 for index, token_id in enumerate(draft_token_ids) if token_id in finish_set),
            None,
        )
        return _SampleTask(
            operation=operation,
            logits=rows,
            rows=descriptor_rows,
            draws=draws,
            penalty_token_ids=penalty_token_ids,
            penalty_counts=penalty_counts,
            parameter_values=parameter_values,
            draft_token_ids=tuple(int(value) for value in draft_token_ids),
            terminal_draft_prefix=terminal_draft_prefix,
            token_product=token_product,
            finish_product=finish_product,
            transition_product=transition_product,
            predicate=None if predicate_value is None else predicate_value[0],
            tagged_predicate=False if predicate_value is None else predicate_value[1],
            request_pool_index=request_pool_index,
            penalty_base=penalty_base,
        )

    def _session_penalty_base(
        self,
        session: RequestRow,
        vocab: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the fixed request-indexed committed penalty-count row."""

        states = self.runtime_states
        if states is None:
            raise RuntimeError("token sampling has no request runtime-state owner")
        if states.vocab_size != int(vocab) or states.device != device:
            raise capability_mismatch("sampling geometry disagrees with request runtime state")
        return states.penalty_counts[int(session.request_pool_idx)]

    def _candidate_penalty_counts(
        self,
        session: RequestRow,
        committed: torch.Tensor,
        scope: _ExecutionScope,
    ) -> torch.Tensor:
        slot = int(session.request_pool_idx)
        counts = committed.clone()
        found = False
        for publication in scope.runtime_publications:
            if isinstance(publication, _RuntimePublication):
                if publication.slot != slot:
                    continue
                token = publication.token.reshape(-1)[:1]
                valid = publication.valid.reshape(-1)[:1]
                active = publication.active.reshape(-1)[:1]
            else:
                try:
                    index = publication.slots.index(slot)
                except ValueError:
                    continue
                token = publication.tokens[index : index + 1]
                valid = publication.valid[index : index + 1]
                active = publication.active[index : index + 1]
            found = True
            token = token.bitwise_and(TOKEN_VALUE_MASK)
            weight = (valid & active).to(dtype=counts.dtype)
            counts.scatter_add_(0, token.to(dtype=torch.int64), weight)
        return counts if found else committed

    def _transition_driver(
        self,
        operation: Operation,
        scope: _ExecutionScope,
    ) -> _Driver:
        yield from ()
        self._generation()
        session_id = operation.request_key.session_id
        conditioning = tuple(
            reference for reference in operation.inputs if reference.kind is ProductKind.KV
        )
        latent_outputs = tuple(
            reference for reference in operation.outputs if reference.kind is ProductKind.LATENT
        )
        if len(conditioning) != 1 or len(latent_outputs) != 1:
            raise invalid_descriptor(
                "generation transition requires one exact conditioning input and latent output"
            )
        cache_row = self._cache_row(operation, scope)
        self.cache_publications.validate_conditioning(
            session_id,
            conditioning[0],
            cache_row,
            scope.cache_publication_inputs.get(conditioning[0]),
        )
        session = self._request_row(scope, session_id)
        image = session.image
        if image is None:
            raise invalid_descriptor("generation transition has no admitted image parameters")
        if session.latent_product is not None or session.flow_step != 0:
            raise invalid_descriptor("generation transition repeats an active latent trajectory")
        rng = operation.rng
        if rng is None or rng.draw_layout is not DrawLayout.FLOW_NOISE:
            raise invalid_descriptor(
                "generation transition requires semantic flow-noise RNG coordinates"
            )
        if int(rng.seed) != int(image.seed or 0):
            raise invalid_descriptor(
                "generation transition seed disagrees with admitted image seed"
            )
        if int(rng.semantic_index_base) < 1:
            raise invalid_descriptor("flow-noise semantic image index must be positive")
        output = latent_outputs[0]
        if int(output.generation) < 1:
            raise invalid_descriptor("generation transition latent has no logical generation")
        row = self._latent_row(operation, scope)
        pool = self._latent_pool()
        row.staging.value.zero_()
        initial = row.staging.value[: int(row.placement.latent_units)]
        self._initial_latent(
            operation,
            int(row.placement.height),
            int(row.placement.width),
            initial,
        )
        pool.initialize(
            row.request_pool_idx,
            row.staging,
            latent_units=int(row.placement.latent_units),
        )
        scope.latent_publications.append(
            LatentPublication(
                request_pool_idx=row.request_pool_idx,
                page_table=row.placement.page_table,
                expected_generation=0,
                expected_step=0,
                generation=int(output.generation),
                step=0,
                latent_units=int(row.placement.latent_units),
                height=int(row.placement.height),
                width=int(row.placement.width),
            )
        )
        session.latent_product = output
        products = self._publish_latent_transfer(
            operation,
            output,
            initial,
            row,
            step=0,
            scope=scope,
        )
        return _Outcome(
            status=OpStatus.OK,
            selected_point=1,
            logical_lengths=self._logical_lengths(
                operation,
                session,
                cache_row,
                latent_len=0,
            ),
            token_span=TokenSpan(base=session.logical_position, len=0),
            finish_flags=FinishFlags(),
            product_generations=_output_generations(operation),
            products=products,
        )

    def _flow_driver(
        self,
        operation: Operation,
        scope: _ExecutionScope,
    ) -> _Driver:
        flow = self._generation()
        session_id = operation.request_key.session_id
        conditioning = tuple(
            reference for reference in operation.inputs if reference.kind is ProductKind.KV
        )
        latent_inputs = tuple(
            reference for reference in operation.inputs if reference.kind is ProductKind.LATENT
        )
        latent_outputs = tuple(
            reference for reference in operation.outputs if reference.kind is ProductKind.LATENT
        )
        if len(conditioning) != 1 or len(latent_inputs) != 1 or len(latent_outputs) != 1:
            raise invalid_descriptor(
                "flow operation requires exact conditioning and one latent input/output generation"
            )
        cache_row = self._cache_row(operation, scope)
        self.cache_publications.validate_conditioning(
            session_id,
            conditioning[0],
            cache_row,
            scope.cache_publication_inputs.get(conditioning[0]),
        )
        session = self._request_row(scope, session_id)
        image = session.image
        if image is None:
            raise invalid_descriptor("flow operation has no admitted image parameters")
        if operation.rng is not None:
            raise invalid_descriptor("flow continuation must inherit transition RNG state")
        latent_input = latent_inputs[0]
        latent_output = latent_outputs[0]
        if (
            int(latent_input.generation) < 1
            or int(latent_output.generation) < 1
            or latent_input == latent_output
        ):
            raise invalid_descriptor("flow latent generations are invalid")
        if session.latent_product != latent_input:
            raise invalid_descriptor("flow operation does not name the current latent generation")
        row = self._latent_row(operation, scope)
        pool = self._latent_pool()
        start_step = int(row.placement.start_step)
        step_count = int(row.placement.step_count)
        conditioning_position = session.logical_position
        image_prompt = image.image_prompts[0] if image.image_prompts else ""
        current = pool.gather_current(
            row.request_pool_idx,
            row.staging,
            step=start_step,
            generation=int(latent_input.generation),
            latent_units=int(row.placement.latent_units),
            height=int(row.placement.height),
            width=int(row.placement.width),
        )
        entries: dict[Branch, CacheRow] = {}
        for step in range(start_step, start_step + step_count):
            host_t, host_t_next = flow.schedule_pair(
                int(image.steps),
                float(image.timestep_shift),
                step,
            )
            t, t_next = pool.stage_timestep(row.request_pool_idx, host_t, host_t_next)
            use_cfg = float(image.cfg_interval[0]) <= host_t <= float(image.cfg_interval[1])
            guide = build_flow_cfg_plan(
                cfg_text_scale=float(image.cfg_text_scale),
                cfg_img_scale=float(image.cfg_img_scale),
                recipe=flow.cfg_recipe,
                renorm=image.cfg_renorm_type,
                renorm_min=float(image.cfg_renorm_min),
                use_cfg=use_cfg,
            )
            if len(guide.branches) > int(flow.max_cfg_branches):
                raise invalid_descriptor("flow CFG plan exceeds the model branch bound")

            prefix_tasks: list[_ForwardTask] = []
            for branch_index, branch in enumerate(guide.branches, start=1):
                if branch in entries:
                    continue
                source = self._branch_source(branch)
                prefix, copy_conditioning = self._flow_prefix(
                    source,
                    image_prompt,
                    session,
                )
                entry = scope.branch_rows.get(
                    (operation.request_key, operation.op_id, branch_index, 0)
                )
                if entry is None:
                    raise invalid_descriptor("flow branch has no scheduler scratch placement")
                query = self._flow_physical_tokens(
                    int(row.placement.height),
                    int(row.placement.width),
                )
                prefix_length = cache_row.length if copy_conditioning else len(prefix)
                if prefix_length + query > entry.capacity:
                    raise invalid_descriptor("flow branch exceeds scheduler scratch placement")
                if entry.length not in {0, prefix_length}:
                    raise invalid_descriptor(
                        "flow branch prefix disagrees with its initialized physical state"
                    )
                initialize_prefix = entry.length == 0 and prefix_length > 0
                if initialize_prefix and copy_conditioning:
                    prefix_pages = (
                        cache_row.length + self.cache_pool.block_size - 1
                    ) // self.cache_pool.block_size
                    self.cache_pool.copy_pages(
                        entry.group_id,
                        cache_row.block_table[:prefix_pages],
                        entry.block_table[:prefix_pages],
                    )
                    entry.length = cache_row.length
                    entry.initialized_length = cache_row.length
                entries[branch] = entry
                if initialize_prefix and prefix:
                    prefix_tasks.append(
                        self._flow_prefix_task(
                            operation,
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
                    operation,
                    conditioning_position,
                    branch,
                    entries[branch],
                    current,
                    t,
                    int(row.placement.height),
                    int(row.placement.width),
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
                velocity = x_pred_to_velocity(velocity, current, t)
            elif flow.prediction != "velocity":
                raise invalid_descriptor(f"unsupported flow prediction {flow.prediction!r}")
            current.copy_(euler_step(current, velocity, t, t_next))
            session.flow_step = step + 1
        pool.write_inactive(
            row.request_pool_idx,
            row.staging,
            expected_step=start_step,
            expected_generation=int(latent_input.generation),
            latent_units=int(row.placement.latent_units),
            height=int(row.placement.height),
            width=int(row.placement.width),
        )
        final_step = start_step + step_count
        scope.latent_publications.append(
            LatentPublication(
                request_pool_idx=row.request_pool_idx,
                page_table=row.placement.page_table,
                expected_generation=int(latent_input.generation),
                expected_step=start_step,
                generation=int(latent_output.generation),
                step=final_step,
                latent_units=int(row.placement.latent_units),
                height=int(row.placement.height),
                width=int(row.placement.width),
            )
        )
        session.latent_product = latent_output
        products = self._publish_latent_transfer(
            operation,
            latent_output,
            current,
            row,
            step=final_step,
            scope=scope,
        )
        return _Outcome(
            status=OpStatus.OK,
            selected_point=1,
            logical_lengths=self._logical_lengths(
                operation,
                session,
                cache_row,
                latent_len=final_step,
            ),
            token_span=TokenSpan(base=session.logical_position, len=0),
            finish_flags=FinishFlags(),
            product_generations=_output_generations(operation),
            products=products,
        )

    def _publish_latent_transfer(
        self,
        operation: Operation,
        product: ProductRef,
        value: torch.Tensor,
        row: _LatentExecution,
        *,
        step: int,
        scope: _ExecutionScope,
    ) -> tuple[ProductPayload, ...]:
        """Publish a committed-candidate trajectory for an exact staged consumer."""

        transport = self.transport
        if (
            transport is None
            or transport.name == "local"
            or (self.deployment is not None and int(self.deployment.tp_rank) != 0)
        ):
            return ()
        locator = transport.publish_async(value.detach().contiguous())
        metadata = {
            "generation": int(product.generation),
            "height": int(row.placement.height),
            "latent_units": int(row.placement.latent_units),
            "step": int(step),
            "width": int(row.placement.width),
        }
        locator = replace(locator, meta={**locator.meta, **metadata})
        scope.published.append(locator)
        scope.stage_publications[_operation_identity(operation)] = (locator,)
        descriptor = _CompletionTransferPayload(
            "latent",
            {"locator": locator.to_wire(), **metadata},
            (locator,),
            operation.plan_digest,
            transport,
        )
        return (ProductPayload(product=product, payload=cast(bytes, descriptor)),)

    def _initial_latent(
        self,
        operation: Operation,
        height: int,
        width: int,
        target: torch.Tensor,
    ) -> None:
        flow = self._generation()
        rng = operation.rng
        assert rng is not None and rng.draw_layout is DrawLayout.FLOW_NOISE
        seed = flow_noise_seed(int(rng.seed), int(rng.semantic_index_base))
        raw = target.reshape(flow.latent_shape(height, width))
        normal_noise(
            tuple(int(value) for value in raw.shape),
            seed=seed,
            device=target.device,
            dtype=target.dtype,
            out=raw,
        )
        raw.mul_(flow.noise_scale(height, width))
        neural = flow.neural_latent(raw)
        if neural.data_ptr() != target.data_ptr() or tuple(neural.shape) != tuple(target.shape):
            target.copy_(neural.reshape_as(target))

    def _branch_source(self, branch: Branch) -> BranchSource:
        return self._generation().branch_source(branch)

    def _flow_prefix(
        self,
        source: BranchSource,
        image_prompt: str,
        session: RequestRow,
    ) -> tuple[tuple[int, ...], bool]:
        return self._generation().prefix(
            source,
            image_prompt=image_prompt,
            negative_prompt=_require_image(session).negative_prompt,
            negative_token_ids=session.negative_token_ids,
            tokenizer=self.tokenizer,
        )

    def _flow_prefix_task(
        self,
        operation: Operation,
        tokens: tuple[int, ...],
        entry: CacheRow,
        branch: Branch,
        scope: _ExecutionScope,
    ) -> _ForwardTask:
        session = self._request_row(scope, operation.request_key.session_id)
        positions = torch.arange(entry.length, entry.length + len(tokens), dtype=torch.long)
        return _ForwardTask(
            operation=operation,
            request=session,
            weights=self._weights(),
            phase=ModelPhase.TEXT,
            token_ids=torch.tensor(tokens, dtype=torch.long),
            positions=positions,
            selection=TokenSelection.HIDDEN,
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
        operation: Operation,
        conditioning_position: int,
        branch: Branch,
        entry: CacheRow,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        height: int,
        width: int,
        scope: _ExecutionScope,
    ) -> _ForwardTask:
        session = self._request_row(scope, operation.request_key.session_id)
        flow = self._generation()
        image_tokens = self._flow_query_tokens(latent, height, width)
        text_local: tuple[int, ...]
        if flow.latent_layout is LatentLayout.PATCH_TOKENS:
            latent_positions = get_flattened_position_ids_extrapolate(
                height,
                width,
                int(flow.latent_downsample),
                int(math.isqrt(flow.max_latent_tokens)),
            )
            conditioning: FlowPatches | None = None
            query_tokens = image_tokens + int(flow.commit_marker_tokens)
            temporal = self._flow_temporal_position(branch, conditioning_position, entry)
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
                self._flow_temporal_position(branch, conditioning_position, entry),
            )
            conditioning = self._flow_conditioning(latent, height, width)
            query_tokens = image_tokens
            attention_indexes = latent_positions
            text_local = ()
        return _ForwardTask(
            operation=operation,
            request=session,
            weights=self._weights(),
            phase=ModelPhase.DENOISE,
            flow_conditioning=conditioning,
            positions=latent_positions,
            timestep=timestep.reshape(1),
            latent=latent,
            image_tokens=query_tokens,
            image_height=height,
            image_width=width,
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
        conditioning_position: int,
        entry: CacheRow,
    ) -> int:
        if branch is Branch.COND:
            return int(conditioning_position)
        return int(entry.length)

    def _flow_query_tokens(self, latent: torch.Tensor, height: int, width: int) -> int:
        del latent
        return self._generation().image_tokens(height, width)

    def _flow_physical_tokens(self, height: int, width: int) -> int:
        return self._generation().physical_tokens(height, width)

    def _flow_conditioning(
        self,
        latent: torch.Tensor,
        height: int,
        width: int,
    ) -> FlowPatches | None:
        transform = self._image_processor().vit
        generation = self._generation()
        return generation.conditioning(
            generation.materialization_latent(latent, height, width),
            height,
            width,
            patch_size=(
                int(transform.patch_size) if isinstance(transform, PatchTransform) else None
            ),
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

    def _state_outcome(
        self,
        operation: Operation,
        outcome: _StateOutcome,
        scope: _ExecutionScope,
        *,
        base: int | None = None,
        products: tuple[ProductPayload, ...] = (),
    ) -> _Outcome:
        session = self._request_row(scope, operation.request_key.session_id)
        row = scope.cache_rows.get((operation.request_key, operation.op_id, 0))
        span_base = session.logical_position if base is None else int(base)
        return _Outcome(
            status=OpStatus.OK,
            selected_point=1 if operation.advances_state else 0,
            logical_lengths=self._logical_lengths(operation, session, row),
            token_span=TokenSpan(base=span_base, len=outcome.sampled_tokens),
            finish_flags=FinishFlags(),
            product_generations=_output_generations(operation),
            committed_tokens=outcome.committed_tokens,
            products=(*products, *outcome.products),
        )

    def _non_state_outcome(
        self,
        operation: Operation,
        scope: _ExecutionScope,
        *,
        products: tuple[ProductPayload, ...] = (),
        completion_tasks: tuple[_CompletionImagePayload, ...] = (),
    ) -> _Outcome:
        session = self._request_row(scope, operation.request_key.session_id)
        row = scope.cache_rows.get((operation.request_key, operation.op_id, 0))
        base = session.logical_position
        return _Outcome(
            status=OpStatus.OK,
            selected_point=1 if operation.advances_state else 0,
            logical_lengths=self._logical_lengths(operation, session, row),
            token_span=TokenSpan(base=base, len=0),
            finish_flags=FinishFlags(),
            product_generations=_output_generations(operation),
            products=products,
            completion_tasks=completion_tasks,
        )

    def _encode_driver(
        self,
        operation: Operation,
        scope: _ExecutionScope,
    ) -> _Driver:
        image_spec = self._image_processor()
        mode = EncodeMode(cast(str, operation.work.mode))
        feature_outputs = tuple(
            output
            for output in operation.outputs
            if output.storage_class is StorageClass.LATENT_ARENA
            and output.kind in {ProductKind.VISION_FEATURE, ProductKind.LATENT_FEATURE}
        )
        if len(feature_outputs) != 1:
            raise invalid_descriptor("encode operation requires one resident feature output")
        feature_output = feature_outputs[0]
        if int(feature_output.generation) < 1:
            raise invalid_descriptor("encode feature output requires a positive generation")
        source = self._encode_source(operation, scope)
        target_device = self._generation_device if mode is EncodeMode.LATENT else self._device
        if isinstance(source, tuple):
            source_tensor, source_metadata = source
            prepared = prepare_tensor_image(
                image_spec,
                mode,
                source_tensor,
                device=target_device,
                signed_unit=source_metadata.value_range is ImageRange.SIGNED_UNIT,
            )
        else:
            prepared = prepare_image(
                image_spec,
                mode,
                source,
                device=target_device,
            )
        task = self._encode_task(operation, mode, prepared, scope)
        outputs = yield (task,)
        features = _encode_features(outputs[0]).detach()
        write = _bound_encoder_write(scope, feature_output)
        resident = self.encoder_cache.publish(
            write,
            features,
            EncoderMetadata(height=prepared.height, width=prepared.width),
        )
        products: tuple[ProductPayload, ...] = ()
        if (
            self.transport is not None
            and self.transport.name != "local"
            and int(self.deployment.tp_rank) == 0
        ):
            locator = self.transport.publish_async(resident)
            locator = replace(
                locator,
                meta={
                    **locator.meta,
                    "generation": int(feature_output.generation),
                    "height": int(prepared.height),
                    "payload_kind": feature_output.kind.value,
                    "width": int(prepared.width),
                },
            )
            scope.published.append(locator)
            scope.stage_publications[_operation_identity(operation)] = (locator,)
            descriptor = _CompletionTransferPayload(
                "encoder",
                {
                    "generation": int(feature_output.generation),
                    "locator": locator.to_wire(),
                    "payload_kind": feature_output.kind.value,
                    "height": prepared.height,
                    "width": prepared.width,
                },
                (locator,),
                operation.plan_digest,
                self.transport,
            )
            products = (
                ProductPayload(
                    product=feature_output,
                    payload=cast(bytes, descriptor),
                ),
            )
        return self._non_state_outcome(operation, scope, products=products)

    def _materialize_driver(
        self,
        operation: Operation,
        scope: _ExecutionScope,
    ) -> _Driver:
        session_id = operation.request_key.session_id
        session = self._request_row(scope, session_id)
        latent_inputs = tuple(
            reference for reference in operation.inputs if reference.kind is ProductKind.LATENT
        )
        if not latent_inputs:
            return self._materialize_frames(operation, scope)
        if len(latent_inputs) != 1:
            raise invalid_descriptor("materialization requires one exact latent generation")
        latent_input = latent_inputs[0]
        if int(latent_input.generation) < 1 or session.latent_product != latent_input:
            raise invalid_descriptor("materialization does not name the current latent generation")
        flow = self._generation()
        image_params = session.image
        if image_params is None:
            raise invalid_descriptor("image materialization has no admitted image parameters")
        row = self._latent_row(operation, scope)
        if int(row.placement.start_step) != int(image_params.steps):
            raise invalid_descriptor("image materialization requires a completed latent trajectory")
        current = self._latent_pool().gather_current(
            row.request_pool_idx,
            row.staging,
            step=int(row.placement.start_step),
            generation=int(latent_input.generation),
            latent_units=int(row.placement.latent_units),
            height=int(row.placement.height),
            width=int(row.placement.width),
        )
        materialization_latent = flow.materialization_latent(
            current,
            int(row.placement.height),
            int(row.placement.width),
        )

        if flow.materialization is Materialization.DECODE_ROUTE:
            task = _ForwardTask(
                operation=operation,
                request=session,
                weights=self._weights(),
                phase=ModelPhase.DECODE_LATENT,
                latent=materialization_latent,
                image_height=int(row.placement.height),
                image_width=int(row.placement.width),
            )
            outputs = yield (task,)
            image_tensor = _decoded_tensor(outputs[0]).detach()
            image_range = ImageRange.UNIT
        elif flow.materialization is Materialization.RGB_LATENT:
            image_tensor = materialization_latent.detach()
            image_range = ImageRange.SIGNED_UNIT
        else:
            raise invalid_descriptor("model declares an unknown image materialization kind")

        # Materialize owns the independently fenced resident image used by a
        # later feedback encode and a query-ready CPU encoding continuation.
        artifact = _artifact_product_ref(operation)
        resident_outputs = tuple(
            output
            for output in operation.outputs
            if output.storage_class is StorageClass.LATENT_ARENA
            and output.kind is ProductKind.ARTIFACT
        )
        if len(resident_outputs) > 1:
            raise invalid_descriptor("materialize operation repeats its resident image product")
        if resident_outputs:
            resident_output = resident_outputs[0]
            image_handle = int(resident_output.generation)
            if image_handle < 1:
                raise invalid_descriptor(
                    "materialized resident image requires a positive generation"
                )
            write = _bound_device_write(scope, resident_output)
            self.device_products.publish_write(
                write,
                image_tensor,
                metadata=DeviceProductMetadata(
                    height=int(row.placement.height),
                    width=int(row.placement.width),
                    value_range=image_range,
                ),
            )
            scope.operation_writes.setdefault(_operation_identity(operation), write)
        image_task = self._defer_image_encoding(
            operation,
            image_tensor,
            image_range,
            scope,
            max_bytes=int(artifact.max_bytes),
        )
        session.latent_product = None
        session.flow_step = 0
        scope.latent_releases.append(
            LatentRelease(
                request_pool_idx=row.request_pool_idx,
                page_table=row.placement.page_table,
                generation=int(latent_input.generation),
                step=int(row.placement.start_step),
                latent_units=int(row.placement.latent_units),
                height=int(row.placement.height),
                width=int(row.placement.width),
            )
        )
        products = (
            ProductPayload(
                product=artifact,
                payload=cast(bytes, image_task),
            ),
        )
        return self._non_state_outcome(
            operation,
            scope,
            products=products,
            completion_tasks=(image_task,),
        )

    def _transfer_driver(
        self,
        operation: Operation,
        scope: _ExecutionScope,
    ) -> _Driver:
        if self.transport is None:
            raise capability_mismatch("product transfer requires a configured transport")
        yield from ()
        session_id = operation.request_key.session_id
        mode = operation.work.mode
        if mode == TransferMode.KV_PUBLISH.value:
            point = _fixed_parent(operation)
            outputs = tuple(output for output in operation.outputs if output.kind is ProductKind.KV)
            if len(outputs) != 1:
                raise invalid_descriptor("KV publication requires one KV output product")
            if any(reference.kind is ProductKind.KV for reference in operation.inputs):
                raise invalid_descriptor("KV publication is rooted only by its fixed parent")
            row = self._cache_row(operation, scope)
            expected_base = self.cache_publications.destination_base(session_id, "gen")
            snapshot = self.cache_publications.publish(
                row,
                source_version=operation.parent,
                source_digest=point.semantic_digest,
                destination="gen",
                expected_base=expected_base,
                product=outputs[0],
                transport=self.transport,
            )
            scope.cache_publications.append((outputs[0], snapshot))
            row.published_length = snapshot.published_extent
            for encoded in snapshot.locators:
                scope.published.append(Locator.from_wire_json(encoded))
            payload = _CompletionTransferPayload(
                "kv",
                {
                    "generation": int(outputs[0].generation),
                    "snapshot": snapshot.to_wire(),
                },
                tuple(Locator.from_wire_json(encoded) for encoded in snapshot.locators),
                operation.plan_digest,
                self.transport,
            )
            return self._non_state_outcome(
                operation,
                scope,
                products=(ProductPayload(product=outputs[0], payload=cast(bytes, payload)),),
            )
        if mode == TransferMode.KV_INSTALL.value:
            inputs = tuple(
                reference for reference in operation.inputs if reference.kind is ProductKind.KV
            )
            outputs = tuple(output for output in operation.outputs if output.kind is ProductKind.KV)
            if len(inputs) != 1 or len(outputs) != 1:
                raise invalid_descriptor("KV installation requires one input and one output")
            row = self._cache_row(operation, scope)
            installed = self.cache_publications.install(
                row,
                session_id=session_id,
                source=inputs[0],
                installed_product=outputs[0],
                transport=self.transport,
                transferred_tensors=(
                    None
                    if (prepared := scope.prepared_transfers.get(inputs[0])) is None
                    else prepared.tensors()
                ),
                publication=scope.cache_publication_inputs.get(inputs[0]),
            )
            scope.cache_installations.append((inputs[0], outputs[0], installed))
            return self._non_state_outcome(
                operation,
                scope,
                products=(ProductPayload(product=outputs[0], payload=b""),),
            )
        inputs = tuple(
            reference for reference in operation.inputs if _is_transferable_product(reference)
        )
        outputs = tuple(
            reference for reference in operation.outputs if _is_transferable_product(reference)
        )
        if len(inputs) != 1 or len(outputs) != 1:
            raise invalid_descriptor("product transfer requires one physical input and one output")
        value, metadata = self._fetch_product_tensor(operation, scope)
        product_payload = self._publish_product_transfer(
            operation,
            outputs[0],
            value,
            metadata,
            scope,
        )
        return self._non_state_outcome(operation, scope, products=(product_payload,))

    def _encode_source(
        self,
        operation: Operation,
        scope: _ExecutionScope,
    ) -> str | tuple[torch.Tensor, DeviceProductMetadata]:
        for reference in operation.inputs:
            inline = scope.input_images.get(reference)
            if inline is not None:
                return inline
            if reference.kind is not ProductKind.ARTIFACT:
                continue
            read = self._consume_device_product(
                reference,
                scope,
                consumer_op_id=operation.op_id,
                device=self._operation_device(operation),
            )
            metadata = read.metadata
            if (
                metadata is None
                or metadata.height < 1
                or metadata.width < 1
                or metadata.value_range is None
            ):
                raise invalid_descriptor("resident image product has incomplete geometry")
            scope.device_reads.append(read)
            return read.tensor, metadata
        raise invalid_descriptor("encode operation has no source image product")

    def _encode_task(
        self,
        operation: Operation,
        mode: EncodeMode,
        prepared: PreparedImage,
        scope: _ExecutionScope,
    ) -> _ForwardTask:
        session = self._request_row(scope, operation.request_key.session_id)
        return _ForwardTask(
            operation=operation,
            request=session,
            weights=self._weights(),
            phase=(
                ModelPhase.ENCODE_VISION if mode is EncodeMode.VISION else ModelPhase.ENCODE_LATENT
            ),
            encode_pixels=prepared.pixels,
            encode_grid=prepared.grid,
            encode_grid_shape=prepared.grid_shape,
        )

    def _state_driver(
        self,
        operation: Operation,
        variant: WorkVariant,
        scope: _ExecutionScope,
        *,
        height: int,
        width: int,
        conditioning_position: int,
        features: torch.Tensor | None = None,
        image: torch.Tensor | None = None,
        image_range: ImageRange = ImageRange.SIGNED_UNIT,
        latent: torch.Tensor | None = None,
        sample_token: bool = False,
        close_image: bool = False,
        retain_image: bool,
    ) -> Generator[tuple[_ModelTask, ...], _TaskResult, _StateOutcome]:
        committed_tokens: tuple[int | _CompletionToken, ...] = ()
        products: tuple[ProductPayload, ...] = ()
        state_query_tokens = 0
        del image, image_range, retain_image
        phases = (
            (ModelPhase.TEXT,)
            if variant is WorkVariant.ENCODE_VISION
            else (ModelPhase.DENOISE,)
            if variant is WorkVariant.ENCODE_LATENT
            else ()
        )
        if not phases:
            raise invalid_descriptor("state publication has no concrete model phase")
        for phase in phases:
            if phase is ModelPhase.TEXT:
                if features is None:
                    raise invalid_descriptor("token state stage has no vision features")
                task = self._vision_state_task(
                    operation,
                    features,
                    height,
                    width,
                    conditioning_position,
                    scope,
                    close_image=close_image,
                    logits=sample_token,
                )
                state_query_tokens += task.query_tokens
                if state_query_tokens > int(operation.bounds.max_tokens):
                    raise invalid_descriptor(
                        "image state query span exceeds the operation token bound"
                    )
                outputs = yield (task,)
                value = _token_logits_or_hidden(outputs[0])
                self._commit_task_kv(task, task.query_tokens, scope)
                if sample_token:
                    session = self._request_row(scope, operation.request_key.session_id)
                    flow_spec = self.model.generation if self.model is not None else None
                    sample_task = self._sample_task(
                        operation,
                        value[-1],
                        session,
                        scope,
                        positions=(
                            conditioning_position
                            + max(
                                1,
                                1 if flow_spec is None else flow_spec.rope_advance,
                            ),
                        ),
                        request_pool_index=_request_pool_index(task),
                    )
                    sampled = _sample_result((yield (sample_task,))[0])
                    self._publish_token_product(operation, sampled, scope)
                    session.rng_counter += 1
                    logical_position = conditioning_position + (
                        max(1, 1 if flow_spec is None else int(flow_spec.rope_advance))
                        if close_image
                        else 1
                    )
                    self._publish_runtime_samples(
                        (operation,),
                        (session,),
                        (sampled,),
                        scope=scope,
                        sample_tasks=(sample_task,),
                        logical_positions=(logical_position,),
                        sampling_positions=(session.rng_counter,),
                    )
                    committed_tokens = (sampled.token_id,)
                    products = _sample_product_payloads(operation, sampled)
                continue
            if phase is ModelPhase.DENOISE:
                if latent is None:
                    raise invalid_descriptor("flow state stage has no latent tensor")
                task = self._latent_state_task(
                    operation,
                    latent,
                    height,
                    width,
                    conditioning_position,
                    scope,
                )
                state_query_tokens += task.query_tokens
                if state_query_tokens > int(operation.bounds.max_tokens):
                    raise invalid_descriptor(
                        "image state query span exceeds the operation token bound"
                    )
                outputs = yield (task,)
                _flow_prediction(outputs[0])
                self._commit_task_kv(task, task.query_tokens, scope)
                continue
        return _StateOutcome(committed_tokens, products)

    def _vision_state_task(
        self,
        operation: Operation,
        features: torch.Tensor,
        height: int,
        width: int,
        conditioning_position: int,
        scope: _ExecutionScope,
        *,
        close_image: bool,
        logits: bool,
    ) -> _ForwardTask:
        session = self._request_row(scope, operation.request_key.session_id)
        injection = self._image_processor().feature_injection
        if injection is None:
            raise invalid_descriptor("vision state stage requires declared feature injection")
        embeddings = (
            features.squeeze(0) if features.ndim == 3 and int(features.shape[0]) == 1 else features
        )
        if embeddings.ndim != 2 or int(embeddings.shape[0]) < 1:
            raise invalid_descriptor("vision features must have shape [tokens, hidden]")
        leading = injection.layout is FeatureLayout.FRAMED
        trailing = leading or close_image
        query = int(leading) + int(embeddings.shape[0]) + int(trailing)
        token_ids = torch.ones(query, dtype=torch.long)
        token_embeddings = embeddings.new_zeros((query, int(embeddings.shape[1])))
        embedding_mask = torch.zeros(query, dtype=torch.bool, device=embeddings.device)
        begin = int(leading)
        token_embeddings[begin : begin + int(embeddings.shape[0])] = embeddings
        embedding_mask[begin : begin + int(embeddings.shape[0])] = True
        if leading:
            token_ids[0] = self._feature_token_id(injection, start=True)
        if trailing:
            token_ids[-1] = self._feature_token_id(injection, start=False)
        positions = self._vision_positions(
            injection.positions,
            int(embeddings.shape[0]),
            conditioning_position,
            height=height,
            width=width,
            leading=leading,
            trailing=trailing,
            close_image=close_image,
        )
        return _ForwardTask(
            operation=operation,
            request=session,
            weights=self._weights(),
            phase=ModelPhase.TEXT,
            token_ids=token_ids,
            token_embeddings=token_embeddings,
            token_embedding_mask=embedding_mask,
            positions=positions,
            selection=TokenSelection.LAST_LOGITS if logits else TokenSelection.HIDDEN,
            entry=self._cache_row(operation, scope),
            write_kv=True,
            causal=False,
            attention_indexes=_positions_as_three_axis(positions, query),
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
        feature_tokens: int,
        conditioning_position: int,
        *,
        height: int,
        width: int,
        leading: bool,
        trailing: bool,
        close_image: bool,
    ) -> torch.Tensor:
        query = int(leading) + feature_tokens + int(trailing)
        if layout is PositionLayout.TEMPORAL:
            return torch.full((query,), int(conditioning_position), dtype=torch.long)
        transform = self._image_processor().vit
        if not isinstance(transform, PatchTransform):
            raise invalid_descriptor(
                "temporal-spatial feature injection requires a patch image transform"
            )
        raw_height, raw_width = patch_grid_shape(transform, height, width)
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
        operation: Operation,
        latent: torch.Tensor,
        height: int,
        width: int,
        conditioning_position: int,
        scope: _ExecutionScope,
    ) -> _ForwardTask:
        session = self._request_row(scope, operation.request_key.session_id)
        flow = self._generation()
        if flow.latent_layout is not LatentLayout.PATCH_TOKENS:
            raise invalid_descriptor("flow state publication requires patch-token latents")
        image_tokens = (height // int(flow.latent_downsample)) * (
            width // int(flow.latent_downsample)
        )
        if int(latent.reshape(-1, latent.shape[-1]).shape[0]) != image_tokens:
            raise invalid_descriptor("state latent does not match the declared image geometry")
        query = image_tokens + int(flow.commit_marker_tokens)
        positions = get_flattened_position_ids_extrapolate(
            height,
            width,
            int(flow.latent_downsample),
            int(math.isqrt(flow.max_latent_tokens)),
        )
        temporal = torch.full((query,), conditioning_position + 1, dtype=torch.long)
        temporal[0] = conditioning_position
        temporal[-1] = conditioning_position + int(flow.rope_advance)
        indexes = torch.stack((temporal, torch.zeros_like(temporal), torch.zeros_like(temporal)))
        return _ForwardTask(
            operation=operation,
            request=session,
            weights=self._weights(),
            phase=ModelPhase.DENOISE,
            positions=positions,
            timestep=latent.new_zeros(1),
            latent=latent,
            image_tokens=query,
            image_height=height,
            image_width=width,
            entry=self._cache_row(operation, scope),
            write_kv=True,
            causal=False,
            attention_indexes=indexes,
            text_local_indices=(0, query - 1),
        )

    def _materialize_frames(
        self,
        operation: Operation,
        scope: _ExecutionScope,
    ) -> _Outcome:
        image, metadata = self._fetch_product_tensor(operation, scope)
        if _metadata_string(metadata, "payload_kind", "") != "image_nchw":
            raise invalid_descriptor("frame materialization source is not an image tensor")
        value_range = ImageRange(
            _metadata_string(metadata, "value_range", ImageRange.SIGNED_UNIT.value)
        )
        image_task = self._defer_image_encoding(
            operation,
            image,
            value_range,
            scope,
            max_bytes=int(operation.bounds.max_completion_bytes),
        )
        return self._non_state_outcome(operation, scope, completion_tasks=(image_task,))

    def _defer_image_encoding(
        self,
        operation: Operation,
        image: torch.Tensor,
        value_range: ImageRange,
        scope: _ExecutionScope,
        *,
        max_bytes: int,
    ) -> _CompletionImagePayload:
        if max_bytes < 1:
            raise invalid_descriptor("image materialization requires a positive completion bound")
        quantized = quantize_image_hwc(
            image,
            value_range=((-1.0, 1.0) if value_range is ImageRange.SIGNED_UNIT else (0.0, 1.0)),
        )
        if int(quantized.numel()) > max_bytes:
            raise invalid_descriptor("image staging exceeds its registered completion byte bound")
        capture = scope.completion.capture_bytes(quantized)
        identity = _operation_identity(operation)
        reservation = scope.cpu_tasks.get(identity)
        if reservation is None:
            raise RuntimeError("materialization has no registered CPU task slot")
        return _CompletionImagePayload(
            capture,
            reservation,
            max_bytes,
        )

    def _publish_product_transfer(
        self,
        operation: Operation,
        product: ProductRef,
        value: torch.Tensor,
        source_metadata: Mapping[str, object],
        scope: _ExecutionScope,
    ) -> ProductPayload:
        transport = self.transport
        if transport is None:
            raise capability_mismatch("product publication requires a configured transport")
        source_kind = _metadata_string(source_metadata, "payload_kind", "")
        if source_kind != product.kind.value and not (
            product.kind is ProductKind.ARTIFACT and source_kind == "image_nchw"
        ):
            raise invalid_descriptor("product transfer changes the physical product kind")
        generation = int(product.generation)
        height = _metadata_uint(source_metadata, "height", 0)
        width = _metadata_uint(source_metadata, "width", 0)
        locator_metadata: dict[str, object]
        descriptor_value: dict[str, object]
        if product.kind is ProductKind.LATENT:
            latent_units = _metadata_uint(source_metadata, "latent_units", 0)
            step = _metadata_uint(source_metadata, "step", 0)
            if min(height, width, latent_units, generation) < 1:
                raise invalid_descriptor("latent transfer has incomplete physical metadata")
            locator_metadata = {
                "generation": generation,
                "height": height,
                "latent_units": latent_units,
                "step": step,
                "width": width,
            }
            descriptor_kind = "latent"
            descriptor_value = dict(locator_metadata)
        elif product.kind in {ProductKind.VISION_FEATURE, ProductKind.LATENT_FEATURE}:
            if min(height, width, generation) < 1:
                raise invalid_descriptor("encoder transfer has incomplete geometry")
            locator_metadata = {
                "generation": generation,
                "height": height,
                "payload_kind": product.kind.value,
                "width": width,
            }
            descriptor_kind = "encoder"
            descriptor_value = dict(locator_metadata)
        elif _requires_device_product_binding(product):
            value_range = _metadata_string(source_metadata, "value_range", "")
            if (height == 0) != (width == 0):
                raise invalid_descriptor("device-product transfer has incomplete geometry")
            if value_range not in {"", *(member.value for member in ImageRange)}:
                raise invalid_descriptor("device-product transfer has an invalid value range")
            if height == 0 and value_range:
                raise invalid_descriptor("non-image device product carries an image range")
            locator_metadata = {
                "generation": generation,
                "height": height,
                "value_range": value_range,
                "width": width,
            }
            descriptor_kind = "device_product"
            descriptor_value = dict(locator_metadata)
        else:
            raise invalid_descriptor("product transfer output has no concrete physical owner")
        locator = transport.publish_async(value.detach().contiguous())
        locator = replace(locator, meta={**locator.meta, **locator_metadata})
        if not _locator_matches_product(locator, product):
            transport.release(locator)
            raise invalid_descriptor("product transfer value disagrees with its output bound")
        scope.published.append(locator)
        scope.stage_publications[_operation_identity(operation)] = (locator,)
        descriptor_value["locator"] = locator.to_wire()
        descriptor = _CompletionTransferPayload(
            descriptor_kind,
            descriptor_value,
            (locator,),
            operation.plan_digest,
            transport,
        )
        return ProductPayload(product=product, payload=cast(bytes, descriptor))

    def _fetch_product_tensor(
        self,
        operation: Operation,
        scope: _ExecutionScope,
    ) -> tuple[torch.Tensor, Mapping[str, object]]:
        for reference in operation.inputs:
            if reference.kind is ProductKind.LATENT:
                session = self._request_row(scope, operation.request_key.session_id)
                if session.latent_product != reference:
                    raise invalid_descriptor(
                        "latent transfer does not name the committed trajectory"
                    )
                row = self._latent_row(operation, scope)
                value = self._latent_pool().gather_current(
                    row.request_pool_idx,
                    row.staging,
                    step=int(row.placement.start_step),
                    generation=int(reference.generation),
                    latent_units=int(row.placement.latent_units),
                    height=int(row.placement.height),
                    width=int(row.placement.width),
                )
                return value, {
                    "payload_kind": ProductKind.LATENT.value,
                    "height": int(row.placement.height),
                    "latent_units": int(row.placement.latent_units),
                    "width": int(row.placement.width),
                    "step": int(row.placement.start_step),
                    "generation": int(reference.generation),
                }
            if reference.storage_class is StorageClass.DEVICE_TENSOR:
                device_read = self._consume_device_product(
                    reference,
                    scope,
                    consumer_op_id=operation.op_id,
                    device=self._operation_device(operation),
                )
                scope.device_reads.append(device_read)
                metadata = device_read.metadata
                values: dict[str, object] = {"payload_kind": reference.kind.value}
                if metadata is not None and metadata.height > 0:
                    values.update(
                        {
                            "payload_kind": "image_nchw",
                            "height": metadata.height,
                            "width": metadata.width,
                            "value_range": (
                                "" if metadata.value_range is None else metadata.value_range.value
                            ),
                        }
                    )
                return device_read.tensor, values
            if reference.kind in {
                ProductKind.VISION_FEATURE,
                ProductKind.LATENT_FEATURE,
            }:
                encoder_read = self._consume_encoder_feature(
                    reference,
                    scope,
                    consumer_op_id=operation.op_id,
                    device=self._operation_device(operation),
                )
                scope.encoder_reads.append(encoder_read)
                return encoder_read.tensor, {
                    "payload_kind": reference.kind.value,
                    "height": encoder_read.metadata.height,
                    "width": encoder_read.metadata.width,
                }
        raise invalid_descriptor("transfer product is not resident or transport-addressable")


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


def _locator_matches_product(locator: Locator, product: ProductRef) -> bool:
    shape = tuple(int(value) for value in locator.shape)
    elements = math.prod(shape)
    bounds = product.shape_bound.dims
    if any(isinstance(bound, DeviceDim) for bound in bounds):
        shape_matches = 0 < elements <= product.shape_bound.max_elements
    elif bounds:
        shape_matches = shape == tuple(cast(StaticDim, bound).extent for bound in bounds)
    else:
        shape_matches = elements == 1
    dtype, element_bytes = device_product_storage(product.dtype)
    return (
        all(value > 0 for value in shape)
        and shape_matches
        and locator.dtype == dtype
        and int(locator.nbytes) == elements * element_bytes
    )


def _three_axis_positions(positions: torch.Tensor | None, query: int) -> torch.Tensor:
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
) -> torch.Tensor:
    return torch.tensor(tuple(int(value) for value in values), dtype=dtype)


def _cumulative(
    lengths: Sequence[int],
) -> torch.Tensor:
    values = [0]
    for length in lengths:
        values.append(values[-1] + int(length))
    return _stage_ints(
        values,
        dtype=torch.int32,
    )


def _binding_identity(tasks: Sequence[_ForwardTask]) -> int:
    digest = hashlib.sha256(b"uniserve-forward-binding\0")
    for task in tasks:
        digest.update(task.phase.value.encode("ascii"))
        digest.update(task.kind.encode("ascii"))
        digest.update(task.query_tokens.to_bytes(8, "little"))
    return int.from_bytes(digest.digest()[:8], "little")


def _trace_envelopes(
    operations: Sequence[Operation],
) -> tuple[OperationTrace, ...]:
    return tuple(
        OperationTrace(
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


def _parent_semantic(operation: Operation, session: RequestRow) -> str:
    """The parent semantic digest a completion's own semantic digest chains from.

    A fixed parent names it directly; a device parent chains from the session's
    resolved semantic digest.
    """

    selected = session.resolve_version(operation.parent)
    if selected is None or not isinstance(selected.point, FixedPoint):
        raise invalid_descriptor("operation parent has no resolved semantic state")
    return selected.point.semantic_digest


def _output_generations(operation: Operation) -> tuple[int, ...]:
    return tuple(int(reference.generation) for reference in operation.outputs)


def _artifact_product_ref(operation: Operation) -> ProductRef:
    """The declared host-visible artifact output of image materialization."""

    for output in operation.outputs:
        if (
            output.kind is ProductKind.ARTIFACT
            and output.storage_class is StorageClass.COMPLETION_ARENA
        ):
            return output
    raise invalid_descriptor("materialize operation has no host-visible artifact output")


def _logprob_product_ref(operation: Operation) -> ProductRef | None:
    matches = tuple(output for output in operation.outputs if output.kind is ProductKind.LOGPROB)
    if len(matches) > 1:
        raise invalid_descriptor("operation declares multiple logprob products")
    return matches[0] if matches else None


def _sample_product_payloads(
    operation: Operation,
    sample: _SampleResult | None,
) -> tuple[ProductPayload, ...]:
    if sample is None or (
        (sample.logprob is None or sample.top_logprobs is None) and not sample.prompt_logprobs
    ):
        return ()
    reference = _logprob_product_ref(operation)
    if reference is None:
        raise invalid_descriptor("sampler produced undeclared logprob output")
    payload = _CompletionLogprobPayload(
        sample.logprob,
        sample.top_logprobs,
        sample.prompt_logprobs,
    )
    return (
        ProductPayload(
            product=reference,
            payload=cast(bytes, payload),
        ),
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


def _token_logits(output: torch.Tensor) -> torch.Tensor:
    if not isinstance(output, torch.Tensor) or output.ndim < 2:
        raise invalid_descriptor("token route did not return logits")
    return output


def _require_sampling(session: RequestRow) -> SamplingParams:
    if session.sampling is None:
        raise invalid_descriptor("sequence execution requires admitted sampling parameters")
    return session.sampling


def _require_image(session: RequestRow) -> ImageParams:
    if session.image is None:
        raise invalid_descriptor("flow execution requires admitted image parameters")
    return session.image


def _token_logits_or_hidden(output: torch.Tensor) -> torch.Tensor:
    if not isinstance(output, torch.Tensor) or output.ndim < 2:
        raise invalid_descriptor("token route did not return a token tensor")
    return output


def _flow_prediction(output: torch.Tensor) -> torch.Tensor:
    if not isinstance(output, torch.Tensor) or not output.is_floating_point():
        raise invalid_descriptor("flow route did not return a flow prediction")
    return output


def _sample_result(value: object) -> _SampleResult:
    if not isinstance(value, _SampleResult):
        raise RuntimeError("sampling task returned an invalid result")
    return value


def _sampling_task_tensors(
    rows: Sequence[_SamplingRow],
    *,
    vocab: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Penalties are applied densely from the device-resident count base in the
    # general sampling path; the fused top-k path never receives a penalty-
    # bearing operation. The sparse penalty tensors are therefore always empty
    # and exist only to satisfy the shared batch layout for the fused path.
    width = min(vocab, bucketed_length(0))
    padding: list[int] = []
    candidate = vocab - 1
    while len(padding) < width:
        padding.append(candidate)
        candidate -= 1
    token_ids = torch.tensor(
        tuple(padding),
        dtype=torch.long,
        device=device,
    ).reshape(1, width)
    counts = torch.zeros((1, width), dtype=torch.float32, device=device)
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


def _device_greedy_row(row: _SamplingRow) -> bool:
    parameters = row.parameters
    return (
        float(parameters.temperature) <= 0.0
        and not parameters.return_logprobs
        and int(row.n_logprobs) == 0
        and not parameters.logprob_token_ids
        and row.allowed is None
        and not parameters.logit_bias
        and parameters.repetition_penalty == 1.0
        and parameters.frequency_penalty == 0.0
        and parameters.presence_penalty == 0.0
    )


@torch.inference_mode()
def _sample_task_batch(
    tasks: Sequence[_SampleTask],
    completion: CompletionLease | None = None,
    *,
    device_products: DeviceProducts | None = None,
    device_reads: tuple[DeviceProductRead, ...] = (),
    selection_broadcast: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[_SampleResult, ...]:
    """Shape and draw every compatible sampling row in each device batch."""

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
        device_greedy = not task.draft_token_ids and all(
            _device_greedy_row(row) for row in task.rows
        )
        if device_greedy:
            if (
                any(
                    value is not None
                    for value in (
                        task.draws,
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
            draws = cast(torch.Tensor, task.draws)
            penalty_token_ids = cast(torch.Tensor, task.penalty_token_ids)
            penalty_counts = cast(torch.Tensor, task.penalty_counts)
            parameter_values = cast(torch.Tensor, task.parameter_values)
            if (
                draws.device != task.logits.device
                or tuple(draws.shape) != (len(task.rows),)
                or not draws.is_floating_point()
            ):
                raise invalid_descriptor("sampling task draws must align with its rows")
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
        elif len(task.rows) != 1:
            raise invalid_descriptor("ordinary sampling tasks must contain exactly one row")
        vocab = int(task.logits.shape[1])
        if vocab > TOKEN_VALUE_MASK:
            raise capability_mismatch("vocabulary exceeds the device token decision range")
        if any(value < 0 or value >= vocab for value in task.draft_token_ids):
            raise invalid_descriptor("speculative draft token is outside the model vocabulary")
        sampling_path = (
            (-2 if any(row.suppress for row in task.rows) else -1)
            if device_greedy
            else _fused_top_k(task, vocab)
        )
        penalty_width = (
            int(cast(torch.Tensor, task.penalty_token_ids).shape[1]) if sampling_path > 0 else 0
        )
        grouped[(task.logits.device, vocab, sampling_path, penalty_width)].append((index, task))

    result: list[_SampleResult | None] = [None] * len(tasks)
    for (_device, _vocab, sampling_path, _penalty_width), compatible in grouped.items():
        indexes, group = zip(*compatible, strict=True)
        if sampling_path < 0:
            sampled_group = _sample_device_greedy_group(
                tuple(group),
                completion,
                apply_suppression=sampling_path == -2,
                device_products=device_products,
                device_reads=device_reads,
                selection_broadcast=selection_broadcast,
            )
        elif sampling_path > 0:
            sampled_group = _sample_fused_top_k_group(
                tuple(group),
                sampling_path,
                completion,
                device_products=device_products,
                device_reads=device_reads,
                selection_broadcast=selection_broadcast,
            )
        else:
            sampled_group = _sample_task_group(
                tuple(group),
                completion,
                device_products=device_products,
                device_reads=device_reads,
                selection_broadcast=selection_broadcast,
            )
        for index, sampled in zip(indexes, sampled_group, strict=True):
            result[index] = sampled
    return tuple(cast(_SampleResult, value) for value in result)


def _sample_device_greedy_group(
    tasks: tuple[_SampleTask, ...],
    completion: CompletionLease | None,
    *,
    apply_suppression: bool,
    device_products: DeviceProducts | None,
    device_reads: tuple[DeviceProductRead, ...],
    selection_broadcast: Callable[[torch.Tensor], torch.Tensor] | None,
) -> tuple[_SampleResult, ...]:
    logits = packed_tensor_views(tuple(task.logits for task in tasks))
    if logits is None:
        logits = torch.cat(tuple(task.logits for task in tasks), dim=0)
    else:
        logits = logits.reshape(len(tasks), -1)
    selection_logits = logits
    if apply_suppression:
        selection_logits = logits.to(dtype=torch.float32, copy=True)
        vocab = int(selection_logits.shape[1])
        for row_index, task in enumerate(tasks):
            for token_id in dict.fromkeys(int(value) for value in task.rows[0].suppress):
                if 0 <= token_id < vocab:
                    selection_logits[row_index, token_id].fill_(float("-inf"))
    products = tuple(task.token_product for task in tasks)
    bound_products = device_products is not None and all(
        product is not None for product in products
    )
    product_table = cast(DeviceProducts, device_products) if bound_products else None
    product_writes = (
        tuple(cast(DeviceProductWrite, product) for product in products)
        if product_table is not None
        else ()
    )
    product_batch = (
        product_table.producer_scalar_batch(product_writes) if product_table is not None else None
    )
    packed_output = product_batch.tensor if product_batch is not None else None
    finish_indexes = tuple(
        index for index, task in enumerate(tasks) if task.finish_product is not None
    )
    finish_writes = tuple(
        cast(DeviceProductWrite, tasks[index].finish_product) for index in finish_indexes
    )
    finish_batch: DeviceProductScalarBatch | None = None
    if finish_writes and device_products is not None:
        finish_batch = device_products.producer_scalar_batch(finish_writes)
    transition_writes = tuple(
        cast(DeviceProductWrite, task.transition_product)
        for task in tasks
        if task.transition_product is not None
    )
    transition_batch: DeviceProductScalarBatch | None = None
    if transition_writes and device_products is not None:
        transition_batch = device_products.producer_scalar_batch(transition_writes)
    grouped_publication = (
        product_table is not None
        and product_batch is not None
        and (not finish_writes or finish_batch is not None)
        and (not transition_writes or transition_batch is not None)
    )
    if packed_output is None or int(packed_output.numel()) != len(tasks):
        device_tokens = torch.argmax(selection_logits, dim=-1)
        max_values = None
    elif grouped_publication:
        device_tokens = packed_output
        max_values = torch.empty(
            len(tasks),
            dtype=selection_logits.dtype,
            device=selection_logits.device,
        )
        torch.max(selection_logits, dim=-1, out=(max_values, device_tokens))
    else:
        device_tokens = packed_output
        max_values = None
        torch.argmax(selection_logits, dim=-1, out=device_tokens)
    valid = (
        torch.isfinite(max_values)
        if max_values is not None
        else (
            ~torch.isnan(selection_logits).any(dim=-1)
            & ~torch.isposinf(selection_logits).any(dim=-1)
            & torch.isfinite(selection_logits).any(dim=-1)
        )
    )
    if selection_broadcast is not None:
        selection_broadcast(device_tokens)
    active = _sample_predicates(tasks, device_tokens.device)
    device_finish: torch.Tensor | None
    empty_finish = grouped_publication and all(
        not task.rows[0].force_finish and not task.rows[0].finish_token_ids for task in tasks
    )
    if grouped_publication:
        if empty_finish:
            device_finish = torch.zeros_like(valid, dtype=torch.bool)
            continuation_values = active & valid
        else:
            device_finish = _device_finish_values(tasks, device_tokens, valid & active)
            continuation_values = active & valid & ~device_finish
    else:
        device_finish, continuation_values, _producer_event = _resolve_sampled_finish_values(
            tasks,
            device_tokens,
            valid,
            active,
            torch.zeros_like(active, dtype=torch.bool),
            device_products,
            device_reads,
        )
    resolved_transition_writes, transition_values = _sampled_transition_values(
        tasks,
        device_tokens,
        valid,
        active,
        destination=(
            transition_batch.tensor
            if grouped_publication
            and transition_batch is not None
            and transition_batch.tensor.dtype is torch.bool
            else None
        ),
    )
    if len(resolved_transition_writes) != len(transition_writes) or any(
        resolved is not expected
        for resolved, expected in zip(
            resolved_transition_writes,
            transition_writes,
            strict=True,
        )
    ):
        raise RuntimeError("sampling transition publication lost its output alignment")
    if transition_writes and device_products is None:
        raise RuntimeError("sampling transition outputs have no device-product owner")
    if grouped_publication and transition_batch is not None:
        if transition_values is None:
            raise RuntimeError("sampling transition publication lost its device values")
        target = transition_batch.tensor.reshape(-1)
        aliases_target = (
            transition_values.device == target.device
            and transition_values.dtype == target.dtype
            and transition_values.untyped_storage().data_ptr()
            == target.untyped_storage().data_ptr()
            and int(transition_values.storage_offset()) == int(target.storage_offset())
        )
        if not aliases_target:
            target.copy_(transition_values)
    if transition_writes and not grouped_publication:
        if transition_values is None:
            raise RuntimeError("sampling transition publication lost its device values")
        _publish_device_writes(
            transition_writes,
            transition_values,
            cast(DeviceProducts, device_products),
            device_reads,
        )
    span = _capture_sample_span(
        valid,
        active,
        device_tokens,
        cast(torch.Tensor, device_finish) if empty_finish else torch.zeros_like(device_tokens),
        completion,
    )
    tagged_tokens = _tagged_token_values(
        device_tokens,
        continuation_values,
        in_place=packed_output is not None and int(packed_output.numel()) == len(tasks),
    )
    published = product_table is not None
    if product_table is not None:
        if grouped_publication:
            side_batches: tuple[DeviceProductScalarBatch, ...] = (
                (transition_batch,) if transition_batch is not None else ()
            )
            if finish_batch is not None:
                if device_finish is None:
                    raise RuntimeError("grouped sampling has no device finish values")
                finish_batch.tensor.copy_(
                    _select_device_values(device_finish, finish_indexes),
                )
                side_batches = (*side_batches, finish_batch)
            product_table.publish_scalar_group(
                (*side_batches, cast(DeviceProductScalarBatch, product_batch)),
                after_reads=device_reads,
            )
            device_finish = None
        elif product_batch is None:
            product_table.publish_writes(
                product_writes,
                tagged_tokens,
            )
        else:
            product_table.publish_scalar_batch(
                product_batch,
                after_reads=device_reads,
            )
    return tuple(
        _SampleResult(
            token_id=_CompletionSampleToken(span, index),
            device_token=device_tokens[index : index + 1],
            logprob=None,
            top_logprobs=None,
            device_valid=valid[index : index + 1],
            device_active=active[index : index + 1],
            device_finish=(None if device_finish is None else device_finish[index : index + 1]),
            device_continuation=continuation_values[index : index + 1],
            device_product_published=published,
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
    uses_penalties = (
        parameters.repetition_penalty != 1.0
        or parameters.frequency_penalty != 0.0
        or parameters.presence_penalty != 0.0
    )
    if (
        wants_logprobs
        or uses_penalties
        or row.allowed is not None
        or row.suppress
        or parameters.logit_bias
        or float(parameters.typical_p) < 1.0
        or top_k <= 0
        or top_k > 128
        or top_k >= vocab
    ):
        return 0
    return top_k


def _sample_fused_top_k_group(
    tasks: tuple[_SampleTask, ...],
    top_k: int,
    completion: CompletionLease | None,
    *,
    device_products: DeviceProducts | None,
    device_reads: tuple[DeviceProductRead, ...],
    selection_broadcast: Callable[[torch.Tensor], torch.Tensor] | None,
) -> tuple[_SampleResult, ...]:
    rows = tuple(task.rows[0] for task in tasks)
    logits = torch.cat(tuple(task.logits for task in tasks), dim=0)
    draws = torch.cat(tuple(cast(torch.Tensor, task.draws) for task in tasks), dim=0)
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
        draws,
        penalty_token_ids,
        penalty_counts,
        parameters,
        top_k,
    )
    if selection_broadcast is not None:
        selection_broadcast(tokens)
    active = _sample_predicates(tasks, tokens.device)
    device_finish, continuation_values, _producer_event = _resolve_sampled_finish_values(
        tasks,
        tokens,
        valid,
        active,
        torch.zeros_like(active, dtype=torch.bool),
        device_products,
        device_reads,
    )
    _publish_sampled_transition_values(
        tasks,
        tokens,
        valid,
        active,
        device_products,
        device_reads,
    )
    published = _publish_sampled_device_values(
        tasks,
        "token_product",
        _tagged_token_values(tokens, continuation_values),
        device_products,
        device_reads,
    )
    span = _capture_sample_span(valid, active, tokens, torch.zeros_like(tokens), completion)
    return tuple(
        _SampleResult(
            _CompletionSampleToken(span, index),
            tokens[index : index + 1],
            None,
            None,
            device_valid=valid[index : index + 1],
            device_active=active[index : index + 1],
            device_finish=(None if device_finish is None else device_finish[index : index + 1]),
            device_continuation=continuation_values[index : index + 1],
            device_product_published=published,
        )
        for index in range(len(rows))
    )


def _run_fused_top_k_sampling(
    logits: torch.Tensor,
    draws: torch.Tensor,
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
        draws,
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


def _sample_task_group(
    tasks: tuple[_SampleTask, ...],
    completion: CompletionLease | None,
    *,
    device_products: DeviceProducts | None,
    device_reads: tuple[DeviceProductRead, ...],
    selection_broadcast: Callable[[torch.Tensor], torch.Tensor] | None,
) -> tuple[_SampleResult, ...]:
    device = tasks[0].logits.device
    vocab = int(tasks[0].logits.shape[1])
    offsets: list[int] = []
    offset = 0
    for task in tasks:
        offsets.append(offset)
        offset += len(task.rows)
    rows = tuple(row for task in tasks for row in task.rows)
    logits = torch.cat(tuple(task.logits.float() for task in tasks), dim=0)
    draws = torch.cat(tuple(cast(torch.Tensor, task.draws) for task in tasks), dim=0)
    work, valid = _shape_sampling_logits_batch(logits, rows)

    temperatures = torch.tensor(
        [float(row.parameters.temperature) for row in rows],
        dtype=work.dtype,
        device=device,
    )
    probabilities = torch.softmax(work, dim=-1)
    cumulative = probabilities.cumsum(dim=-1)
    sampled_tokens = (
        (cumulative < draws.to(dtype=cumulative.dtype).unsqueeze(1))
        .sum(dim=-1)
        .clamp_max(vocab - 1)
    )
    row_tokens = torch.where(
        temperatures > 0.0,
        sampled_tokens,
        torch.argmax(work, dim=-1),
    )
    accepted_counts: list[torch.Tensor] = []
    selected_points: list[torch.Tensor] = []
    terminal_finishes: list[torch.Tensor] = []
    output_rows: list[torch.Tensor] = []
    terminal_tokens: list[torch.Tensor] = []
    for task, row_offset in zip(tasks, offsets, strict=True):
        if task.draft_token_ids:
            draft = torch.tensor(task.draft_token_ids, dtype=row_tokens.dtype, device=device)
            matches = row_tokens[row_offset : row_offset + len(task.draft_token_ids)] == draft
            raw_accepted = torch.cumprod(matches.to(torch.long), dim=0).sum()
        else:
            draft = torch.empty((0,), dtype=row_tokens.dtype, device=device)
            raw_accepted = torch.zeros((), dtype=torch.long, device=device)
        if task.terminal_draft_prefix is None:
            accepted = raw_accepted
            terminal = torch.zeros((), dtype=torch.bool, device=device)
            terminal_token = torch.zeros((), dtype=row_tokens.dtype, device=device)
        else:
            terminal_prefix = int(task.terminal_draft_prefix)
            accepted = raw_accepted.clamp_max(terminal_prefix)
            terminal = raw_accepted >= terminal_prefix
            terminal_token = draft[terminal_prefix - 1]
        accepted_counts.append(accepted)
        selected_points.append(accepted + (~terminal).to(dtype=torch.long))
        terminal_finishes.append(terminal)
        output_rows.append(accepted + row_offset)
        terminal_tokens.append(terminal_token)
    output_indexes = torch.stack(output_rows)
    task_tokens = row_tokens.index_select(0, output_indexes)
    counts = torch.stack(accepted_counts)
    points = torch.stack(selected_points)
    terminal_finish = torch.stack(terminal_finishes)
    task_tokens = torch.where(terminal_finish, torch.stack(terminal_tokens), task_tokens)
    if selection_broadcast is not None:
        selection = torch.stack(
            (
                task_tokens.to(dtype=torch.int64),
                counts.to(dtype=torch.int64),
                points.to(dtype=torch.int64),
                terminal_finish.to(dtype=torch.int64),
            )
        )
        selection_broadcast(selection)
        task_tokens = selection[0].to(dtype=task_tokens.dtype)
        counts = selection[1].to(dtype=counts.dtype)
        points = selection[2].to(dtype=points.dtype)
        terminal_finish = selection[3].to(dtype=torch.bool)

    task_valid = torch.stack(
        tuple(
            valid[offsets[index] : offsets[index] + len(task.rows)][
                torch.arange(len(task.rows), device=device) < points[index]
            ].all()
            for index, task in enumerate(tasks)
        )
    )
    active = _sample_predicates(tasks, task_tokens.device)
    device_finish, continuation_values, _producer_event = _resolve_sampled_finish_values(
        tasks,
        task_tokens,
        task_valid,
        active,
        terminal_finish,
        device_products,
        device_reads,
    )
    _publish_sampled_transition_values(
        tasks,
        task_tokens,
        task_valid,
        active,
        device_products,
        device_reads,
    )
    published = _publish_sampled_device_values(
        tasks,
        "token_product",
        _tagged_token_values(task_tokens, continuation_values),
        device_products,
        device_reads,
    )
    span = _capture_sample_span(task_valid, active, task_tokens, counts, completion)
    details = _sample_logprob_details(
        work,
        output_indexes - terminal_finish.to(dtype=torch.long),
        task_tokens,
        tuple(task.rows[0] for task in tasks),
        completion,
    )
    return tuple(
        _SampleResult(
            token_id=(_CompletionSampleToken(span, index)),
            device_token=task_tokens[index : index + 1],
            logprob=None if index not in details else details[index][0],
            top_logprobs=None if index not in details else details[index][1],
            num_accepted_tokens=(_CompletionInteger(span, index)),
            device_accepted_tokens=counts[index : index + 1],
            device_selected_point=points[index : index + 1],
            device_valid=task_valid[index : index + 1],
            device_active=active[index : index + 1],
            device_finish=(None if device_finish is None else device_finish[index : index + 1]),
            device_continuation=continuation_values[index : index + 1],
            device_product_published=published,
        )
        for index in range(len(tasks))
    )


def _capture_sample_span(
    valid: torch.Tensor,
    active: torch.Tensor,
    tokens: torch.Tensor,
    accepted: torch.Tensor,
    completion: CompletionLease | None,
) -> _CompletionSampleSpan:
    count = int(tokens.numel())
    if (
        int(valid.numel()) != count
        or int(active.numel()) != count
        or int(accepted.numel()) != count
    ):
        raise RuntimeError("sampling completion vectors do not align")
    metadata = torch.cat(
        (
            valid.reshape(-1),
            active.reshape(-1),
            tokens.reshape(-1),
            accepted.reshape(-1),
        )
    )
    if int(metadata.numel()) != SAMPLING_COMPLETION_FIELDS * count:
        raise RuntimeError("sampling completion field count diverged from its capacity contract")
    owns_completion = completion is None
    if metadata.device.type != "cuda":
        values = tuple(int(value) for value in metadata.tolist())
        if owns_completion and not all(
            bool(values[index]) or not bool(values[count + index]) for index in range(count)
        ):
            raise invalid_descriptor("sampling policy masked every vocabulary entry")
        return _CompletionSampleSpan(None, count, values)
    if completion is None:
        raise RuntimeError("CUDA sampling requires a server completion lease")
    span = _CompletionSampleSpan(completion.capture(metadata), count)
    return span


def _device_finish_values(
    tasks: tuple[_SampleTask, ...],
    device_tokens: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    count = len(tasks)
    tokens = device_tokens.reshape(-1)
    validity = valid.reshape(-1)
    if int(tokens.numel()) != count or int(validity.numel()) != count:
        raise RuntimeError("sampling finish vectors do not align")
    if count == 0:
        return validity.to(dtype=torch.bool)

    rows = tuple(task.rows[0] for task in tasks)
    first = rows[0]
    if all(
        row.force_finish == first.force_finish and row.finish_token_ids == first.finish_token_ids
        for row in rows[1:]
    ):
        if first.force_finish:
            return validity.to(dtype=torch.bool)
        if not first.finish_token_ids:
            return torch.zeros_like(validity, dtype=torch.bool)
        matched = tokens == first.finish_token_ids[0]
        for token_id in first.finish_token_ids[1:]:
            matched |= tokens == token_id
        return matched & validity

    values: list[torch.Tensor] = []
    for index, row in enumerate(rows):
        selected = tokens[index]
        if row.force_finish:
            finish = torch.ones((), dtype=torch.bool, device=tokens.device)
        elif row.finish_token_ids:
            finish_ids = torch.tensor(
                row.finish_token_ids,
                dtype=tokens.dtype,
                device=tokens.device,
            )
            finish = (finish_ids == selected).any()
        else:
            finish = torch.zeros((), dtype=torch.bool, device=tokens.device)
        values.append(finish & validity[index])
    return torch.stack(values)


def _sample_predicates(
    tasks: tuple[_SampleTask, ...],
    device: torch.device,
) -> torch.Tensor:
    if tasks and all(task.predicate is not None and task.tagged_predicate for task in tasks):
        predicates = tuple(cast(torch.Tensor, task.predicate).reshape(-1)[:1] for task in tasks)
        packed = packed_tensor_views(predicates)
        if packed is None:
            packed = torch.cat(predicates, dim=0)
        return packed.reshape(-1).ge(TOKEN_CONTINUATION_BIT)
    values = tuple(
        (
            torch.ones((1,), dtype=torch.bool, device=device)
            if task.predicate is None
            else (
                task.predicate.reshape(-1)[:1].ge(TOKEN_CONTINUATION_BIT)
                if task.tagged_predicate
                else task.predicate.reshape(-1)[:1].to(device=device, dtype=torch.bool)
            )
        )
        for task in tasks
    )
    return torch.cat(values, dim=0)


def _tagged_token_values(
    tokens: torch.Tensor,
    continuation: torch.Tensor,
    *,
    in_place: bool = False,
) -> torch.Tensor:
    if int(tokens.numel()) != int(continuation.numel()):
        raise RuntimeError("token continuation vector does not align with selected tokens")
    tags = torch.where(continuation.reshape(-1), TOKEN_CONTINUATION_BIT, 0)
    target = tokens.reshape(-1) if in_place else tokens.reshape(-1).clone()
    target.bitwise_or_(tags)
    return target


def _copy_runtime_scalar(target: torch.Tensor, value: int | torch.Tensor) -> None:
    if isinstance(value, torch.Tensor):
        target.copy_(value.reshape(-1)[:1].to(device=target.device, dtype=target.dtype))
    else:
        target.fill_(int(value))


def _request_pool_index(task: _ForwardTask) -> torch.Tensor:
    index = task.request_pool_index
    if index is None:
        raise RuntimeError("forward task has no staged request slot")
    return index


def _packed_required_views(
    values: Sequence[torch.Tensor | None],
    message: str,
) -> torch.Tensor:
    if not values or any(value is None for value in values):
        raise RuntimeError(message)
    tensors = tuple(cast(torch.Tensor, value).reshape(-1) for value in values)
    packed = packed_tensor_views(tensors)
    return torch.cat(tensors, dim=0) if packed is None else packed


def _sample_request_pool_indices(tasks: Sequence[_SampleTask]) -> torch.Tensor:
    return _packed_required_views(
        tuple(task.request_pool_index for task in tasks),
        "sample batch lost its staged request slots",
    )


def _sample_result_vector(
    samples: Sequence[_SampleResult],
    field_name: str,
) -> torch.Tensor:
    return _packed_required_views(
        tuple(cast(torch.Tensor | None, getattr(sample, field_name)) for sample in samples),
        f"sample batch lost device field {field_name}",
    )


def _runtime_selected_points(
    selected_values: Sequence[torch.Tensor | None],
    accepted_values: Sequence[torch.Tensor | None],
) -> torch.Tensor:
    if len(selected_values) != len(accepted_values):
        raise RuntimeError("runtime selected-point columns are not aligned")
    points: list[torch.Tensor] = []
    for selected, accepted in zip(selected_values, accepted_values, strict=True):
        if selected is not None:
            points.append(selected.reshape(-1).to(dtype=torch.int32))
        elif accepted is not None:
            points.append(accepted.reshape(-1).to(dtype=torch.int32) + 1)
        else:
            raise RuntimeError("runtime selected-point publication lost device state")
    packed = packed_tensor_views(points)
    return torch.cat(points, dim=0) if packed is None else packed


def _publish_sampled_device_values(
    tasks: tuple[_SampleTask, ...],
    product_field: str,
    device_values: torch.Tensor,
    device_products: DeviceProducts | None,
    device_reads: tuple[DeviceProductRead, ...],
    *,
    producer_event: torch.cuda.Event | None = None,
) -> bool:
    products = tuple(getattr(task, product_field) for task in tasks)
    if device_products is None or not all(product is not None for product in products):
        return False
    writes = tuple(cast(DeviceProductWrite, product) for product in products)
    _publish_device_writes(
        writes,
        device_values.reshape(-1),
        device_products,
        device_reads,
        producer_event=producer_event,
    )
    return True


def _publish_sampled_transition_values(
    tasks: tuple[_SampleTask, ...],
    device_tokens: torch.Tensor,
    valid: torch.Tensor,
    active: torch.Tensor,
    device_products: DeviceProducts | None,
    device_reads: tuple[DeviceProductRead, ...],
) -> None:
    writes, transitions = _sampled_transition_values(
        tasks,
        device_tokens,
        valid,
        active,
    )
    if not writes:
        return
    if device_products is None:
        raise RuntimeError("sampling transition outputs have no device-product owner")
    if transitions is None:
        raise RuntimeError("sampling transition publication lost its device values")
    _publish_device_writes(
        writes,
        transitions,
        device_products,
        device_reads,
    )


def _sampled_transition_values(
    tasks: tuple[_SampleTask, ...],
    device_tokens: torch.Tensor,
    valid: torch.Tensor,
    active: torch.Tensor,
    *,
    destination: torch.Tensor | None = None,
) -> tuple[tuple[DeviceProductWrite, ...], torch.Tensor | None]:
    selected = tuple(
        (index, task, task.transition_product)
        for index, task in enumerate(tasks)
        if task.transition_product is not None
    )
    if not selected:
        return (), None
    tokens = device_tokens.reshape(-1)
    eligibility = valid.reshape(-1).to(dtype=torch.bool) & active.reshape(-1).to(
        dtype=torch.bool
    )
    if int(tokens.numel()) != len(tasks) or int(eligibility.numel()) != len(tasks):
        raise RuntimeError("sampling transition vectors do not align")
    indexes = tuple(index for index, _task, _write in selected)
    writes = tuple(cast(DeviceProductWrite, write) for _index, _task, write in selected)
    selected_tokens = _select_device_values(tokens, indexes)
    selected_eligibility = _select_device_values(eligibility, indexes)
    target: torch.Tensor | None = None
    if destination is not None:
        target = destination.reshape(-1)
        if (
            int(target.numel()) != len(selected)
            or target.device != selected_tokens.device
            or target.dtype is not torch.bool
        ):
            raise RuntimeError("sampling transition destination does not align")
    transition_sets = tuple(
        task.rows[0].transition_token_ids for _index, task, _write in selected
    )
    first = transition_sets[0]
    if all(values == first for values in transition_sets[1:]):
        if not first:
            transitions = (
                target.zero_()
                if target is not None
                else torch.zeros_like(selected_eligibility, dtype=torch.bool)
            )
        else:
            if target is None:
                transitions = selected_tokens == int(first[0])
            else:
                torch.eq(selected_tokens, int(first[0]), out=target)
                transitions = target
            for token_id in first[1:]:
                transitions.logical_or_(selected_tokens == int(token_id))
    else:
        width = max(1, *(len(values) for values in transition_sets))
        transition_ids = torch.tensor(
            tuple((*values, *((-1,) * (width - len(values)))) for values in transition_sets),
            dtype=selected_tokens.dtype,
            device=selected_tokens.device,
        )
        transitions = selected_tokens.unsqueeze(1).eq(transition_ids).any(dim=1)
        if target is not None:
            target.copy_(transitions)
            transitions = target
    transitions.logical_and_(selected_eligibility)
    return writes, transitions


def _resolve_sampled_finish_values(
    tasks: tuple[_SampleTask, ...],
    device_tokens: torch.Tensor,
    valid: torch.Tensor,
    active: torch.Tensor,
    terminal_finish: torch.Tensor,
    device_products: DeviceProducts | None,
    device_reads: tuple[DeviceProductRead, ...],
) -> tuple[torch.Tensor | None, torch.Tensor, torch.cuda.Event | None]:
    finish_values = _device_finish_values(tasks, device_tokens, valid & active) | (
        terminal_finish.reshape(-1).to(dtype=torch.bool) & valid & active
    )
    continuation_values = active & valid & ~finish_values
    if device_products is None:
        return finish_values, continuation_values, None
    selected = tuple(
        (index, task, task.finish_product)
        for index, task in enumerate(tasks)
        if task.finish_product is not None
    )
    if not selected:
        return None, continuation_values, None
    indexes = tuple(index for index, _task, _write in selected)
    writes = tuple(write for _index, _task, write in selected)
    selected_finish_values = _select_device_values(finish_values, indexes)
    producer_event = _publish_device_writes(
        writes,
        selected_finish_values,
        device_products,
        device_reads,
    )
    return None, continuation_values, producer_event


def _select_device_values(values: torch.Tensor, indexes: tuple[int, ...]) -> torch.Tensor:
    flat = values.reshape(-1)
    if len(indexes) == int(flat.numel()) and all(
        index == expected for expected, index in enumerate(indexes)
    ):
        return flat
    views = tuple(flat[index : index + 1] for index in indexes)
    if len(views) == 1:
        return views[0]
    packed = packed_tensor_views(views)
    return torch.cat(views, dim=0) if packed is None else packed.reshape(-1)


def _publish_device_writes(
    writes: tuple[DeviceProductWrite, ...],
    device_values: torch.Tensor,
    device_products: DeviceProducts,
    device_reads: tuple[DeviceProductRead, ...],
    *,
    producer_event: torch.cuda.Event | None = None,
) -> torch.cuda.Event | None:
    values = device_values.reshape(-1)
    product_batch = device_products.producer_scalar_batch(writes)
    if product_batch is not None and int(product_batch.tensor.numel()) == len(writes):
        product_batch.tensor.reshape(-1).copy_(values)
        device_products.publish_scalar_batch(
            product_batch,
            after_reads=device_reads,
            producer_event=producer_event,
        )
    else:
        device_products.publish_writes(
            writes,
            values,
            producer_event=producer_event,
        )
    return writes[0].producer_event


def _semantic_sampling_draws(
    rows: Sequence[_SamplingRow],
    *,
    device: torch.device,
) -> torch.Tensor:
    # Each row's draw is the canonical Philox uniform for its semantic
    # coordinate, computed from host-known identity when the descriptor was
    # built. Greedy rows carry a zero draw. Uploading the host-resident vector
    # keeps the request path free of any device-to-host observation.
    return torch.tensor(
        [float(row.draw) for row in rows],
        dtype=torch.float32,
        device=device,
    )


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

    # Penalties over the device-resident committed count base. Each penalty row
    # carries a dense per-vocabulary count vector (committed generated tokens
    # plus any speculative prefix); repetition is multiplicative and sign-aware,
    # frequency scales with the count, and presence is a flat once-appeared
    # subtraction. Masked (-inf) entries are preserved.
    penalty_rows = [
        row_index for row_index, row in enumerate(rows) if row.penalty_counts is not None
    ]
    if penalty_rows:
        row_index_tensor = torch.tensor(penalty_rows, dtype=torch.long, device=work.device)
        counts = torch.stack(
            [cast(torch.Tensor, rows[row_index].penalty_counts) for row_index in penalty_rows]
        ).to(dtype=work.dtype)
        params = torch.tensor(
            [
                (
                    float(rows[row_index].parameters.repetition_penalty),
                    float(rows[row_index].parameters.frequency_penalty),
                    float(rows[row_index].parameters.presence_penalty),
                )
                for row_index in penalty_rows
            ],
            dtype=work.dtype,
            device=work.device,
        )
        repetition = params[:, 0].unsqueeze(1)
        frequency = params[:, 1].unsqueeze(1)
        presence = params[:, 2].unsqueeze(1)
        values = work.index_select(0, row_index_tensor)
        seen = counts > 0
        repeated = torch.where(values > 0.0, values / repetition, values * repetition)
        adjusted = repeated - frequency * counts - presence
        apply = seen & ~torch.isneginf(values)
        work.index_copy_(0, row_index_tensor, torch.where(apply, adjusted, values))

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

    min_p_rows = tuple(index for index, row in enumerate(rows) if float(row.parameters.min_p) > 0.0)
    if min_p_rows:
        indexes = torch.tensor(min_p_rows, dtype=torch.long, device=work.device)
        subset = work.index_select(0, indexes)
        min_p = parameter_values.index_select(0, indexes)[:, 1]
        min_threshold = subset.max(dim=-1).values + torch.log(min_p)
        subset.masked_fill_(subset < min_threshold.unsqueeze(1), float("-inf"))
        work.index_copy_(0, indexes, subset)

    typical_rows = tuple(
        index for index, row in enumerate(rows) if float(row.parameters.typical_p) < 1.0
    )
    if typical_rows:
        indexes = torch.tensor(typical_rows, dtype=torch.long, device=work.device)
        subset = work.index_select(0, indexes)
        typical_p = torch.tensor(
            [float(rows[index].parameters.typical_p) for index in typical_rows],
            dtype=work.dtype,
            device=work.device,
        )
        probs = torch.softmax(subset, dim=-1)
        log_probs = probs.log()
        entropy = -(probs * log_probs).nan_to_num(0.0).sum(dim=-1, keepdim=True)
        scores = ((-log_probs) - entropy).abs()
        order = torch.argsort(scores, dim=-1)
        cumulative = probs.gather(1, order).cumsum(dim=-1)
        over = cumulative >= typical_p.unsqueeze(1)
        drop = torch.cat(
            (
                torch.zeros((len(typical_rows), 1), dtype=torch.bool, device=work.device),
                over[:, :-1],
            ),
            dim=1,
        )
        subset.scatter_(1, order, subset.gather(1, order).masked_fill(drop, float("-inf")))
        work.index_copy_(0, indexes, subset)

    valid = (
        ~torch.isnan(work).any(dim=-1)
        & ~torch.isposinf(work).any(dim=-1)
        & torch.isfinite(work).any(dim=-1)
    )
    return work, valid


def _sample_logprob_details(
    work: torch.Tensor,
    output_rows: torch.Tensor,
    output_tokens: torch.Tensor,
    rows: Sequence[_SamplingRow],
    completion: CompletionLease | None,
) -> Mapping[
    int,
    tuple[
        float | _CompletionLogprobValue,
        tuple[tuple[int, float, int], ...] | _CompletionTopLogprobs,
    ],
]:
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

    def float_bits(values: torch.Tensor) -> torch.Tensor:
        return values.to(dtype=torch.float32).contiguous().view(torch.int32).to(torch.long)

    packed = torch.cat(
        (
            selected_tokens.reshape(-1).to(torch.long),
            float_bits(selected_values.reshape(-1)),
            selected_ranks.reshape(-1).to(torch.long),
            top_indexes.reshape(-1).to(torch.long),
            float_bits(top_values.reshape(-1)),
            top_ranks.reshape(-1).to(torch.long),
            float_bits(candidate_values.reshape(-1)),
            candidate_ranks.reshape(-1).to(torch.long),
        )
    )
    if packed.device.type != "cuda":
        values = tuple(int(value) for value in packed.tolist())
        batch = _CompletionLogprobBatch(
            None,
            requested_rows,
            counts,
            requested_ids,
            max_count,
            max_requested,
            values,
        )
    else:
        if completion is None:
            raise RuntimeError("CUDA logprob materialization requires a server completion lease")
        batch = _CompletionLogprobBatch(
            completion.capture(packed),
            requested_rows,
            counts,
            requested_ids,
            max_count,
            max_requested,
        )
    if packed.device.type != "cuda":
        return batch.finalize()
    return {
        result_index: (
            _CompletionLogprobValue(batch, result_index),
            _CompletionTopLogprobs(batch, result_index),
        )
        for result_index in requested_rows
    }


def _bound_device_write(
    scope: _ExecutionScope,
    reference: ProductRef,
) -> DeviceProductWrite:
    matches = tuple(write for write in scope.device_writes if write.reference == reference)
    if len(matches) != 1:
        raise invalid_descriptor(
            "resident product does not have exactly one atomic registration binding"
        )
    return matches[0]


def _bound_encoder_write(
    scope: _ExecutionScope,
    reference: ProductRef,
) -> EncoderWrite:
    matches = tuple(write for write in scope.encoder_writes if write.reference == reference)
    if len(matches) != 1:
        raise invalid_descriptor(
            "encoder feature does not have exactly one atomic registration binding"
        )
    return matches[0]


def _requires_device_product_binding(reference: ProductRef) -> bool:
    return reference.storage_class is StorageClass.DEVICE_TENSOR or (
        reference.storage_class is StorageClass.LATENT_ARENA
        and reference.kind is ProductKind.ARTIFACT
    )


def _is_transferable_product(reference: ProductRef) -> bool:
    return (
        reference.kind is ProductKind.LATENT
        or reference.kind in {ProductKind.VISION_FEATURE, ProductKind.LATENT_FEATURE}
        or _requires_device_product_binding(reference)
    )


def _encode_features(output: torch.Tensor) -> torch.Tensor:
    if not isinstance(output, torch.Tensor) or not output.is_floating_point():
        raise invalid_descriptor("encode route did not return encoder features")
    return output


def _decoded_tensor(output: torch.Tensor) -> torch.Tensor:
    if not isinstance(output, torch.Tensor) or not output.is_floating_point():
        raise invalid_descriptor("decode route did not return an image tensor")
    return output


def _positions_as_three_axis(positions: torch.Tensor, query: int) -> torch.Tensor:
    if positions.ndim == 1 and int(positions.numel()) == query:
        return torch.stack((positions, torch.zeros_like(positions), torch.zeros_like(positions)))
    if positions.ndim == 2 and tuple(positions.shape) == (3, query):
        return positions
    raise invalid_descriptor("state positions do not align with their physical token row")


__all__ = ["ModelRunner", "PreparedExecution"]
