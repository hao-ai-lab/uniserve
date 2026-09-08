"""CPU values that move through one execution step."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any, TypeAlias

import torch

from uniserve_worker.execution.batch import (
    BufferId,
    FinishFlags,
    KvTransferValue,
    LatentParams,
    Locator,
    LogicalLengths,
    Operation,
    OpStatus,
    ProductPayload,
    ProductRef,
    RequestKey,
    RowGeometry,
    Run,
    RunLane,
    RunResult,
    SamplingParams,
    SamplingState,
    TokenSpan,
    TransferValue,
)
from uniserve_worker.execution.forward_batch import FlowPatches, ModelPhase, TokenSelection
from uniserve_worker.execution.output import (
    CpuJob,
    ImagePayload,
    LogprobOutputRow,
    OutputBuffer,
    SamplingOutputRow,
    TokenCapture,
)
from uniserve_worker.foundation.errors import WorkerError, classify, invalid_descriptor
from uniserve_worker.loader.weight_set import WeightSet
from uniserve_worker.runtime.cache_transfer import CacheWrite
from uniserve_worker.runtime.cpu import CpuTaskReservation
from uniserve_worker.runtime.device_products import (
    DeviceProductImport,
    DeviceProductRead,
    DeviceProductWrite,
)
from uniserve_worker.runtime.encoder_cache import EncoderRead, EncoderWrite
from uniserve_worker.runtime.latent_pool import (
    LatentPublication,
    LatentRelease,
    LatentStaging,
    LatentWrite,
)
from uniserve_worker.runtime.request import RequestDraft
from uniserve_worker.transfer.tickets import TransferTicket

if TYPE_CHECKING:
    from .model_runner import RunObservation
    from .video import VideoOutputRingLease

OperationIdentity: TypeAlias = tuple[RequestKey, int]


@dataclass(slots=True)
class ForwardRow:
    """Carries one operation’s staged tokens, positions, media tensors, routing, and cache coordinates."""

    operation: Operation
    request: RequestDraft
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
        """Return the live token or image-patch count represented by this row."""

        if self.token_ids is not None:
            return int(self.token_ids.numel())
        if self.latent is not None and self.image_tokens > 0:
            return int(self.image_tokens)
        return 0

    @property
    def kind(self) -> str:
        """Classify the row as token, flow, encode, or latent-decode work."""

        if self.token_ids is not None:
            return "token"
        if self.latent is not None and self.image_tokens > 0:
            return "flow"
        if self.encode_pixels is not None:
            return "encode"
        return "decode"


@dataclass(frozen=True, slots=True)
class SampleRow:
    """Defines filters, penalties, RNG draw, and finish rules for one sampled token."""

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
    """Carries logits, RNG draws, penalties, predicates, and publication targets for one sampling task."""

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
    """Holds device vectors produced by a batched sampling transition."""

    request_pool_indices: torch.Tensor
    tokens: torch.Tensor
    valid: torch.Tensor
    active: torch.Tensor
    continuation: torch.Tensor
    selected_points: torch.Tensor | None
    penalty_bases: tuple[torch.Tensor | None, ...]


@dataclass(frozen=True, slots=True)
class SampleResult:
    """Owns a pending sampling completion and its device-resident token, acceptance, and validity outputs."""

    completion: SamplingOutputRow
    device_token: torch.Tensor | None
    logprobs: LogprobOutputRow | None
    device_accepted_tokens: torch.Tensor | None = None
    device_selected_point: torch.Tensor | None = None
    device_valid: torch.Tensor | None = None
    device_active: torch.Tensor | None = None
    prompt_logprobs: tuple[LogprobOutputRow, ...] = ()
    device_finish: torch.Tensor | None = None
    device_continuation: torch.Tensor | None = None
    device_product_published: bool = False
    device_batch: SampleBatchVectors | None = None
    device_batch_index: int = -1


@dataclass(frozen=True, slots=True)
class RuntimePublication:
    """Publishes a request runtime transition after its lane results become final."""

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
    """Publishes batched device-selected decode transitions for a group of requests."""

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
    """Publishes prompt-logit completion without advancing persistent request state."""

    slot: int
    logits: torch.Tensor


@dataclass(slots=True)
class PreparedTransferInput:
    """Retain one canonical product descriptor and its bounded physical reads."""

    product: ProductRef
    value: TransferValue
    tickets: tuple[TransferTicket, ...]
    buffers: tuple[torch.Tensor | tuple[torch.Tensor, ...], ...]
    destination: (
        DeviceProductImport | DeviceProductWrite | EncoderWrite | LatentWrite | CacheWrite | None
    ) = None
    _discard_destination: Callable[[], None] | None = field(default=None, repr=False)

    def adopt_destination(self) -> None:
        """Transfer a completed destination reservation to its resident product store."""

        self._discard_destination = None

    @property
    def destination_adopted(self) -> bool:
        """Indicate whether resident storage owns this input independently of preparation."""

        return self._discard_destination is None

    def discard_destination(self) -> None:
        """Cancel unconsumed reads and retire a destination that no lane adopted."""

        discard = self._discard_destination
        if discard is None:
            return
        self._discard_destination = None
        if not isinstance(self.destination, DeviceProductImport):
            for ticket in self.tickets:
                ticket.cancel()
        discard()

    def close(self) -> None:
        """Release preparation access while preserving shared materialization."""

        self.discard_destination()
        if isinstance(self.destination, DeviceProductImport):
            self.destination.close()
        else:
            for ticket in self.tickets:
                ticket.close()

    def ready(self) -> bool:
        """Indicate whether every remote product tensor is available to consume."""

        return all(ticket.ready() for ticket in self.tickets) and (
            not isinstance(self.destination, CacheWrite) or self.destination.completion.done()
        )

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Return fetched tensors only after every transfer ticket is ready."""

        if not self.ready():
            raise RuntimeError("prepared transfer input was observed before readiness")
        if isinstance(self.destination, DeviceProductImport):
            self.destination.wait()
        else:
            for ticket in self.tickets:
                ticket.result()
        if isinstance(self.value, KvTransferValue):
            raise RuntimeError("KV inputs are consumed through their physical cache reservation")
        representations = (self.value.tensor,)
        tensors: list[torch.Tensor] = []
        for representation, value in zip(representations, self.buffers, strict=True):
            spans = value if isinstance(value, tuple) else (value,)
            shape = (
                (sum(int(span.shape[0]) for span in spans), *spans[0].shape[1:])
                if isinstance(value, tuple)
                else tuple(value.shape)
            )
            if (
                shape != representation.shape
                or any(
                    str(span.dtype).removeprefix("torch.") != representation.dtype for span in spans
                )
                or sum(int(span.numel()) * int(span.element_size()) for span in spans)
                != int(representation.nbytes)
            ):
                raise invalid_descriptor("transport result disagrees with its logical tensor")
            tensors.extend(spans)
        return tuple(tensors)


@dataclass(slots=True)
class PreparedPredicateBatch:
    """Holds staged predicate tensors and their request-local lookup table."""

    buffer: OutputBuffer
    entries: list[tuple[OperationIdentity, TokenCapture, int]]
    transferred: tuple[tuple[OperationIdentity, PreparedTransferInput, int], ...]
    sealed: bool
    _values: dict[OperationIdentity, bool] | None = None

    def ready(self) -> bool:
        """Return whether all predicate reads and their producer events are query-ready."""

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
        """Read validated predicate scalars and index them by semantic product reference."""

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
        """Release the predicate capture when its values will not be observed."""

        if self._values is None:
            self.buffer.abandon()

    def __del__(self) -> None:
        """Abandon predicate completion storage that was never closed explicitly."""

        try:
            self.abandon()
        except Exception:
            pass


@dataclass(slots=True)
class PreparedExecution:
    """Owns staged transfers and predicates until a batch executes, fails, or is abandoned."""

    batch: Run
    transfers: tuple[PreparedTransferInput, ...]
    predicates: PreparedPredicateBatch | None = None
    storage_dependencies: tuple[Future[None], ...] = ()
    _prepare_inputs: (
        Callable[[], tuple[tuple[PreparedTransferInput, ...], PreparedPredicateBatch | None]] | None
    ) = field(default=None, repr=False)
    _execute: Callable[[PreparedExecution], RunResult] | None = field(
        default=None,
        repr=False,
    )
    _release: Callable[[], None] | None = field(default=None, repr=False)
    _finished: bool = field(default=False, init=False, repr=False)

    def ready(self) -> bool:
        """Query input readiness and the retirements required for destination reuse."""

        return (
            self._prepare_inputs is None
            and all(dependency.done() for dependency in self.storage_dependencies)
            and all(transfer.ready() for transfer in self.transfers)
            and (self.predicates is None or self.predicates.ready())
        )

    def advance(self) -> bool:
        """Submit deferred inputs on the execution thread once their storage retires."""

        if self._finished:
            return False
        prepare_inputs = self._prepare_inputs
        if prepare_inputs is not None:
            if not all(dependency.done() for dependency in self.storage_dependencies):
                return False
            try:
                for dependency in self.storage_dependencies:
                    dependency.result()
                self._prepare_inputs = None
                self.transfers, self.predicates = prepare_inputs()
            except BaseException:
                self._finish()
                raise
        return self.ready()

    def predicate_values(self) -> dict[OperationIdentity, bool]:
        """Resolve staged device predicates by request generation and operation id."""

        return {} if self.predicates is None else self.predicates.resolve()

    def on_dependencies_ready(self, callback: Callable[[], None]) -> None:
        """Wake the execution thread when the next preparation stage can advance.

        Deferred reads first wait for storage retirement. After submitting them,
        the owner registers again if their physical inputs are still pending.
        Callbacks only wake the owner; they never submit device work themselves.
        """

        tickets = tuple(ticket for transfer in self.transfers for ticket in transfer.tickets)
        dependencies = self.storage_dependencies + tuple(
            transfer.destination.completion
            for transfer in self.transfers
            if isinstance(transfer.destination, CacheWrite)
        )
        if not tickets and not dependencies:
            callback()
            return
        lock = Lock()
        fired = False

        def notify_if_ready() -> None:
            """Invoke the callback once after every transfer ticket reports readiness."""

            nonlocal fired
            if self._finished:
                return
            if not all(ticket.ready() for ticket in tickets) or not all(
                dependency.done() for dependency in dependencies
            ):
                return
            with lock:
                if fired:
                    return
                fired = True
            callback()

        for ticket in tickets:
            ticket.add_done_callback(notify_if_ready)
        for dependency in dependencies:
            dependency.add_done_callback(lambda _future: notify_if_ready())
        notify_if_ready()

    def bind(
        self,
        execute: Callable[[PreparedExecution], RunResult],
        release: Callable[[], None],
    ) -> PreparedExecution:
        """Install the single-use execution and release callbacks for this prepared batch."""

        if self._execute is not None or self._release is not None:
            raise RuntimeError("prepared execution is already bound")
        self._execute = execute
        self._release = release
        return self

    def resolve(self) -> RunResult:
        """Execute the bound batch once and guarantee release of its preparation resources."""

        if self._finished:
            raise RuntimeError("prepared execution has already finished")
        if not self.advance():
            raise RuntimeError("prepared execution was observed before dependency readiness")
        execute = self._execute
        if execute is None:
            raise RuntimeError("prepared execution has no worker binding")
        try:
            for dependency in self.storage_dependencies:
                dependency.result()
            return execute(self)
        finally:
            self._finish()

    def record_failure(self, error: BaseException) -> WorkerError:
        """Abandon prepared resources and classify the execution failure for the wire response."""

        self.abandon()
        return classify(error, context="execute")

    def abandon(self) -> None:
        """Release predicate captures, transfer reads, and bound output candidates."""

        if self._finished:
            return
        for transfer in self.transfers:
            transfer.discard_destination()
        if self.predicates is not None:
            self.predicates.abandon()
        self._finish()

    def _finish(self) -> None:
        """Close predicate and transfer preparation resources exactly once."""

        if self._finished:
            return
        self._finished = True
        release = self._release
        self._release = None
        self._execute = None
        self._prepare_inputs = None
        error: BaseException | None = None
        try:
            for transfer in self.transfers:
                try:
                    transfer.close()
                except BaseException as failure:
                    if error is None:
                        error = failure
        finally:
            if release is not None:
                release()
        if error is not None:
            raise error

    def __del__(self) -> None:
        """Release unfinished preparation resources during finalization."""

        try:
            self.abandon()
        except Exception:
            pass


@dataclass(frozen=True, slots=True)
class LaneLayout:
    """Aligned operation, request, sequence, weight, and identity columns for one lane."""

    operations: tuple[Operation, ...]
    requests: tuple[RequestDraft, ...]
    seq_lens: tuple[int, ...]
    weights: tuple[WeightSet, ...]
    identities: tuple[OperationIdentity, ...]

    def __post_init__(self) -> None:
        """Validate equal-length lane columns and unique operation identities."""

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
            raise RuntimeError("lane layout columns are not aligned")


@dataclass(frozen=True, slots=True)
class LatentExecution:
    """Binds a latent params and request slot to its staged trajectory view."""

    params: LatentParams
    request_pool_idx: int
    staging: LatentStaging


@dataclass(slots=True)
class LaneState:
    """Tracks staged rows, sample tasks, publications, outputs, and stats for one executing lane."""

    lane: RunLane
    started_ns: int
    graph_eligible: bool
    request_candidates: tuple[RequestDraft, ...]
    request_rows: dict[int, RequestDraft]
    completion: OutputBuffer
    input_tokens: dict[ProductRef, tuple[int, ...]] = field(default_factory=dict)
    input_images: dict[ProductRef, str] = field(default_factory=dict)
    forward_rows: dict[OperationIdentity, tuple[RowGeometry, ...]] = field(default_factory=dict)
    layout: LaneLayout | None = None
    prepared_transfers: dict[ProductRef, PreparedTransferInput] = field(default_factory=dict)
    cache_publication_inputs: dict[ProductRef, KvTransferValue] = field(default_factory=dict)
    cache_publications: list[tuple[ProductRef, KvTransferValue]] = field(default_factory=list)
    cache_installations: list[tuple[ProductRef, ProductRef, KvTransferValue]] = field(
        default_factory=list
    )
    stage_publications: dict[BufferId, tuple[Locator, ...]] = field(default_factory=dict)
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
    media_output_leases: dict[OperationIdentity, VideoOutputRingLease] = field(default_factory=dict)
    latent_rows: dict[OperationIdentity, LatentExecution] = field(default_factory=dict)
    latent_publications: list[LatentPublication] = field(default_factory=list)
    latent_releases: list[LatentRelease] = field(default_factory=list)
    latent_import_slots: list[int] = field(default_factory=list)
    publication_started: bool = False


@dataclass(frozen=True, slots=True)
class SpeculativeSelection:
    """Captures draft tokens, selected checkpoint metadata, and the completion that resolves acceptance."""

    completion: SamplingOutputRow
    draft_tokens: tuple[int, ...]
    terminal_prefix: int | None
    base_logical_position: int
    base_rng_counter: int
    base_kv_visible: int
    initialized_kv: int


@dataclass(frozen=True, slots=True)
class Outcome:
    """The selected result of one operation, projected onto its completion record.

    ``committed_tokens`` are the tokens selected by the sampler and copied to
    completion storage. ``products`` are the operation's host-facing payloads
    (a materialized image artifact,
    requested logprobs) carried by the completion report under their product
    references.
    """

    status: OpStatus
    selected_point: int
    logical_lengths: LogicalLengths
    token_span: TokenSpan
    finish_flags: FinishFlags
    product_generations: tuple[int, ...]
    committed_tokens: tuple[int, ...] = ()
    sampling: SamplingOutputRow | None = None
    products: tuple[ProductPayload, ...] = ()
    selection: SpeculativeSelection | None = None
    completion_tasks: tuple[CpuJob | ImagePayload, ...] = ()
    next_cursor: int = 0
    done: bool = False


@dataclass(frozen=True, slots=True)
class StateOutcome:
    """Semantic tokens and products committed while publishing image state."""

    committed_tokens: tuple[int, ...] = ()
    sampling: SamplingOutputRow | None = None
    products: tuple[ProductPayload, ...] = ()

    @property
    def sampled_tokens(self) -> int:
        """Count committed prefix tokens plus one deferred sampling position."""

        return len(self.committed_tokens) + int(self.sampling is not None)


@dataclass(slots=True)
class OperationState:
    """Tracks one operation from candidate preparation through result publication."""

    operation: Operation
    lane: LaneState
    phase: str = "initial"
    data: dict[str, Any] = field(default_factory=dict)
    rows: tuple[ForwardRow, ...] = ()
    sample: SampleWork | None = None
    outcome: Outcome | None = None


def dependencies_ready(
    state: OperationState,
    producers: dict[ProductRef, OperationState],
) -> bool:
    """Return whether every declared product input has a completed local producer."""

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
    "LaneLayout",
    "LaneState",
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
