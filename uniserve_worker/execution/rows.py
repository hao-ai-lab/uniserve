"""CPU values that move through one execution step."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

import torch

from uniserve_worker.batch import (
    Batch,
    BatchPartition,
    CompletionReport,
    FinishFlags,
    LatentPlacement,
    LogicalLengths,
    Operation,
    OpStatus,
    ProductKind,
    ProductPayload,
    ProductRef,
    RequestKey,
    SamplingParams,
    SamplingState,
    TokenSpan,
)
from uniserve_worker.batch import (
    ForwardRow as WireForwardRow,
)
from uniserve_worker.execution.forward_batch import FlowPatches, ModelPhase, TokenSelection
from uniserve_worker.foundation.errors import WorkerError, classify, invalid_descriptor
from uniserve_worker.loader.weight_set import WeightSet
from uniserve_worker.runtime.device_products import (
    DeviceProductMetadata,
    DeviceProductRead,
    DeviceProductWrite,
)
from uniserve_worker.runtime.encoder_cache import EncoderRead, EncoderWrite
from uniserve_worker.runtime.latent_pool import LatentPublication, LatentRelease, LatentStaging
from uniserve_worker.server.completion import (
    PinnedOutputBuffer,
    PinnedTokenCapture,
    _CompletionImagePayload,
    _CompletionInteger,
    _CompletionLogprobValue,
    _CompletionSpeculativePoint,
    _CompletionToken,
    _CompletionTopLogprobs,
)
from uniserve_worker.server.cpu_tasks import CpuTaskReservation
from uniserve_worker.server.request_state import RequestRow
from uniserve_worker.transfer.connector import CachePublication
from uniserve_worker.transfer.tickets import Locator, TransferTicket

if TYPE_CHECKING:
    from .model_runner import RunObservation

OperationIdentity: TypeAlias = tuple[RequestKey, int]


@dataclass(slots=True)
class ForwardRow:
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
    request_pool_idx: int = 0
    seq_len: int = 0
    group_id: int = 0
    write_kv: bool = False
    causal: bool = True
    attention_indexes: torch.Tensor | None = None
    text_local_indices: tuple[int, ...] = ()
    request_pool_index: torch.Tensor | None = None
    decode_predicate: torch.Tensor | None = None
    decode_predicate_tagged: bool = False
    decode_force_finish: bool = False
    request_indexed_decode: bool = False

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
class SampleRow:
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
class SampleWork:
    operation: Operation
    logits: torch.Tensor
    rows: tuple[SampleRow, ...]
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
class SampleBatchVectors:
    request_pool_indices: torch.Tensor
    tokens: torch.Tensor
    valid: torch.Tensor
    active: torch.Tensor
    continuation: torch.Tensor
    selected_points: torch.Tensor | None
    penalty_bases: tuple[torch.Tensor | None, ...]


@dataclass(frozen=True, slots=True)
class SampleResult:
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
    device_batch: SampleBatchVectors | None = None
    device_batch_index: int = -1


@dataclass(frozen=True, slots=True)
class RuntimePublication:
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
class DecodeRuntimePublication:
    slots: tuple[int, ...]
    device_slots: torch.Tensor
    tokens: torch.Tensor
    predicates: torch.Tensor
    selected_points: torch.Tensor | None
    penalty_bases: tuple[torch.Tensor | None, ...]
    valid: torch.Tensor
    active: torch.Tensor


@dataclass(frozen=True, slots=True)
class PromptLogitsPublication:
    slot: int
    logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class PreparedTransferInput:
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
class PreparedPredicateBatch:
    buffer: PinnedOutputBuffer
    entries: list[tuple[OperationIdentity, PinnedTokenCapture, int]]
    transferred: tuple[tuple[OperationIdentity, PreparedTransferInput, int], ...]
    sealed: bool
    _values: dict[OperationIdentity, bool] | None = None

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
                    self.entries.append((identity, self.buffer.capture(tensors[0]), row))
                self.buffer.seal()
            except BaseException:
                self.buffer.abandon()
                raise
            self.sealed = True
        return self.buffer.ready()

    def resolve(self) -> dict[OperationIdentity, bool]:
        if self._values is not None:
            return self._values
        if not self.buffer.ready():
            raise RuntimeError("prepared predicates were observed before readiness")
        values: dict[OperationIdentity, bool] = {}
        generation = self.buffer.generation
        try:
            for identity, capture, row in sorted(self.entries, key=lambda entry: entry[2]):
                captured = capture.values()
                if len(captured) != 1 or captured[0] not in {0, 1}:
                    raise invalid_descriptor("operation predicate is not a canonical boolean")
                values[identity] = bool(captured[0])
                self.buffer.observe(row, generation)
        except BaseException:
            self.buffer.abandon()
            raise
        self._values = values
        return values

    def abandon(self) -> None:
        if self._values is None:
            self.buffer.abandon()

    def __del__(self) -> None:
        try:
            self.abandon()
        except Exception:
            pass


@dataclass(slots=True)
class PreparedExecution:
    batch: Batch
    transfers: tuple[PreparedTransferInput, ...]
    predicates: PreparedPredicateBatch | None = None
    _execute: Callable[[PreparedExecution], CompletionReport] | None = field(
        default=None,
        repr=False,
    )
    _release: Callable[[], None] | None = field(default=None, repr=False)
    _finished: bool = field(default=False, init=False, repr=False)

    def ready(self) -> bool:
        return all(transfer.ready() for transfer in self.transfers) and (
            self.predicates is None or self.predicates.ready()
        )

    def predicate_values(self) -> dict[OperationIdentity, bool]:
        return {} if self.predicates is None else self.predicates.resolve()

    def bind(
        self,
        execute: Callable[[PreparedExecution], CompletionReport],
        release: Callable[[], None],
    ) -> PreparedExecution:
        if self._execute is not None or self._release is not None:
            raise RuntimeError("prepared execution is already bound")
        self._execute = execute
        self._release = release
        return self

    def resolve(self) -> CompletionReport:
        if not self.ready():
            raise RuntimeError("prepared execution was observed before transfer readiness")
        execute = self._execute
        if execute is None:
            raise RuntimeError("prepared execution has no worker binding")
        try:
            return execute(self)
        finally:
            self._finish()

    def record_failure(self, error: BaseException) -> WorkerError:
        self.abandon()
        return classify(error, context="execute")

    def abandon(self) -> None:
        if self.predicates is not None:
            self.predicates.abandon()
        self._finish()

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        release = self._release
        if release is not None:
            release()

    def __del__(self) -> None:
        try:
            self.abandon()
        except Exception:
            pass


@dataclass(frozen=True, slots=True)
class PartitionLayout:
    operations: tuple[Operation, ...]
    requests: tuple[RequestRow, ...]
    seq_lens: tuple[int, ...]
    weights: tuple[WeightSet, ...]
    identities: tuple[OperationIdentity, ...]

    def __post_init__(self) -> None:
        width = len(self.operations)
        if not all(
            len(values) == width
            for values in (
                self.requests,
                self.seq_lens,
                self.weights,
                self.identities,
            )
        ):
            raise RuntimeError("partition layout columns are not aligned")


@dataclass(frozen=True, slots=True)
class LatentExecution:
    placement: LatentPlacement
    request_pool_idx: int
    staging: LatentStaging


@dataclass(slots=True)
class PartitionState:
    partition: BatchPartition
    started_ns: int
    graph_eligible: bool
    request_candidates: tuple[RequestRow, ...]
    request_bases: tuple[RequestRow | None, ...]
    request_rows: dict[int, RequestRow]
    completion: PinnedOutputBuffer
    input_tokens: dict[ProductRef, tuple[int, ...]] = field(default_factory=dict)
    input_images: dict[ProductRef, str] = field(default_factory=dict)
    forward_rows: dict[OperationIdentity, tuple[WireForwardRow, ...]] = field(default_factory=dict)
    layout: PartitionLayout | None = None
    prepared_transfers: dict[ProductRef, PreparedTransferInput] = field(default_factory=dict)
    transferred_device_products: dict[ProductRef, DeviceProductWrite] = field(default_factory=dict)
    transferred_encoder_features: dict[ProductRef, EncoderWrite] = field(default_factory=dict)
    cache_publication_inputs: dict[ProductRef, CachePublication] = field(default_factory=dict)
    cache_publications: list[tuple[ProductRef, CachePublication]] = field(default_factory=list)
    cache_installations: list[tuple[ProductRef, ProductRef, CachePublication]] = field(
        default_factory=list
    )
    stage_publications: dict[OperationIdentity, tuple[Locator, ...]] = field(default_factory=dict)
    published: list[Locator] = field(default_factory=list)
    observations: list[RunObservation] = field(default_factory=list)
    component_us: dict[str, int] = field(default_factory=dict)
    device_reads: list[DeviceProductRead] = field(default_factory=list)
    device_writes: list[DeviceProductWrite] = field(default_factory=list)
    encoder_reads: list[EncoderRead] = field(default_factory=list)
    encoder_writes: list[EncoderWrite] = field(default_factory=list)
    operation_writes: dict[OperationIdentity, DeviceProductWrite] = field(default_factory=dict)
    token_writes: dict[OperationIdentity, DeviceProductWrite] = field(default_factory=dict)
    selected_point_writes: dict[OperationIdentity, DeviceProductWrite] = field(default_factory=dict)
    accepted_span_writes: dict[OperationIdentity, DeviceProductWrite] = field(default_factory=dict)
    state_continuation_writes: dict[OperationIdentity, DeviceProductWrite] = field(
        default_factory=dict
    )
    finish_writes: dict[OperationIdentity, DeviceProductWrite] = field(default_factory=dict)
    transition_writes: dict[OperationIdentity, DeviceProductWrite] = field(default_factory=dict)
    propagated_predicate_writes: dict[OperationIdentity, tuple[DeviceProductWrite, ...]] = field(
        default_factory=dict
    )
    predicate_values: dict[OperationIdentity, tuple[torch.Tensor, bool]] = field(
        default_factory=dict
    )
    predicated_operations: frozenset[OperationIdentity] = frozenset()
    sampling_states: dict[OperationIdentity, SamplingState] = field(default_factory=dict)
    runtime_publications: list[RuntimePublication | DecodeRuntimePublication] = field(
        default_factory=list
    )
    prompt_logits_publications: list[PromptLogitsPublication] = field(default_factory=list)
    runtime_cache_lengths: dict[int, int | torch.Tensor] = field(default_factory=dict)
    registration_visible: bool = False
    cpu_tasks: dict[OperationIdentity, CpuTaskReservation] = field(default_factory=dict)
    latent_rows: dict[OperationIdentity, LatentExecution] = field(default_factory=dict)
    latent_publications: list[LatentPublication] = field(default_factory=list)
    latent_releases: list[LatentRelease] = field(default_factory=list)
    latent_import_slots: list[int] = field(default_factory=list)
    publication_started: bool = False


@dataclass(frozen=True, slots=True)
class SpeculativeSelection:
    accepted: _CompletionInteger
    selected_point: _CompletionSpeculativePoint
    draft_tokens: tuple[int, ...]
    terminal_prefix: int | None
    base_logical_position: int
    base_rng_counter: int
    base_kv_visible: int
    initialized_kv: int


@dataclass(frozen=True, slots=True)
class Outcome:
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
    selection: SpeculativeSelection | None = None
    completion_tasks: tuple[_CompletionImagePayload, ...] = ()


@dataclass(frozen=True, slots=True)
class StateOutcome:
    """Semantic tokens and products committed while publishing image state."""

    committed_tokens: tuple[int | _CompletionToken, ...] = ()
    products: tuple[ProductPayload, ...] = ()

    @property
    def sampled_tokens(self) -> int:
        return len(self.committed_tokens)


@dataclass(slots=True)
class OperationState:
    operation: Operation
    partition: Any
    phase: str = "initial"
    data: dict[str, Any] = field(default_factory=dict)
    rows: tuple[Any, ...] = ()
    sample: Any | None = None
    outcome: Any | None = None


def dependencies_ready(
    state: OperationState,
    producers: dict[ProductRef, OperationState],
) -> bool:
    return all(
        producer.outcome is not None
        for reference in state.operation.inputs
        if (producer := producers.get(reference)) is not None
    )


__all__ = [
    "DecodeRuntimePublication",
    "ForwardRow",
    "LatentExecution",
    "OperationIdentity",
    "OperationState",
    "Outcome",
    "PartitionLayout",
    "PartitionState",
    "PreparedExecution",
    "PreparedPredicateBatch",
    "PreparedTransferInput",
    "PromptLogitsPublication",
    "RuntimePublication",
    "SampleBatchVectors",
    "SampleResult",
    "SampleRow",
    "SampleWork",
    "SpeculativeSelection",
    "StateOutcome",
    "dependencies_ready",
]
