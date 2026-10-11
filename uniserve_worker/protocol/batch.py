"""Scheduler-to-worker execution records and their validation.

A `Batch` is one numerical call on one component together with everything a
rank needs to run it: the per-request `Call` descriptors, the lifecycle
commands (`Start`, `Finish`, `Free`) applied with it, the KV unit tables and
latent, decode, and persistent-buffer allocations the scheduler chose, and
host-supplied input tensors (`TensorPublication`). The records mirror the
Rust `Batch` in `uniserve_worker_ipc`, whose `Batch::validate` is the
authoritative check; `Batch.validate` here re-implements part of it.

The PyO3 transport (`crates/worker-ipc-py`) decodes and validates a frame in
Rust, builds each member record through its constructor, and assembles the
`Batch` with `construction.batch_from_validated`, which skips
`Batch.validate`. It calls `BlockTable`, `CacheUnitAllocation`, `Start`,
`Finish`, and `Free` positionally, so their field order is part of that
contract, and the other records by keyword, so their field names are. Python
callers build a `Batch` through its constructor, which runs `Batch.validate`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

from uniserve import sampling
from uniserve.media import image
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol import call, identity, tensor, transfer
from uniserve_worker.protocol.validation import (
    _float,
    _map,
    _nonnegative,
    _uint,
)
from uniserve_worker.protocol.video import VideoAdmission


@dataclass(frozen=True, slots=True)
class Start:
    """Registers a new request and its immutable execution parameters."""

    request: NewRequest

    @property
    def request_key(self) -> identity.RequestKey:
        """Expose the request generation registered by this start command."""
        return self.request.request_key


@dataclass(frozen=True, slots=True)
class Finish:
    """Closes one request epoch, keeping the buffers it names as retained.

    The worker retires the request's state and storage and revokes its
    unretained exports once their readers finish (`Executor` handles this
    in ``_release_commands`` and ``_retire_commands``). Each buffer in
    ``retained_buffers`` stays readable until a `Free` command names it.
    `__post_init__` requires every retained buffer to be owned by
    ``request_key`` and listed once.
    """

    request_key: identity.RequestKey
    retained_buffers: tuple[identity.BufferId, ...] = ()

    def __post_init__(self) -> None:
        _validate_retained_buffers(self.request_key, self.retained_buffers)


def _validate_retained_buffers(
    request: identity.RequestKey, retained: tuple[identity.BufferId, ...]
) -> None:
    """Require each retained buffer to be owned by ``request``, listed once."""
    if any(buffer.owner != request for buffer in retained):
        raise invalid_descriptor("retained buffer belongs to another request")
    if len(set(retained)) != len(retained):
        raise invalid_descriptor("request retirement repeats a retained buffer")


@dataclass(frozen=True, slots=True)
class Free:
    """Releases one scheduler-owned persistent buffer."""

    buffer: identity.BufferId

    @property
    def request_key(self) -> identity.RequestKey:
        """Expose the request generation owning the freed persistent buffer."""
        return self.buffer.owner


BatchCommand: TypeAlias = Start | Finish | Free


def _command_variant_index(command: BatchCommand) -> int:
    """Return the variant discriminant used in a command's identity.

    `Batch.validate` keys repeated-command detection on it. The values match
    the Rust `BatchCommand::variant_index`.
    """
    if isinstance(command, Start):
        return 0
    if isinstance(command, Finish):
        return 1
    return 2


@dataclass(frozen=True, slots=True)
class CanvasSampling:
    """Block-diffusion sampling of a request that generates in canvases.

    A block of ``canvas_length`` tokens starts as random tokens and is
    denoised for at most ``max_steps`` steps: each samples every position at
    a temperature falling linearly from ``t_max`` to ``t_min``, accepts the
    lowest-entropy samples within ``entropy_bound`` nats and renoises the
    rest. A block stops early once its argmax canvas has held for
    ``stability_threshold`` steps and its mean entropy is below
    ``confidence_threshold`` nats. Draws follow the admitted sampling seed.
    """

    canvas_length: int
    max_steps: int
    entropy_bound: float
    t_min: float
    t_max: float
    confidence_threshold: float
    stability_threshold: int

    def __post_init__(self) -> None:
        """Require positive lengths and finite sampling values.

        The sampler (``uniserve.diffusion.canvas.CanvasSampling``) checks
        the values' domain when the worker runs the canvas.
        """
        if self.canvas_length < 1 or self.max_steps < 1:
            raise invalid_descriptor(
                "a canvas requires a positive length and step limit"
            )
        if not all(
            math.isfinite(value)
            for value in (
                self.entropy_bound,
                self.t_min,
                self.t_max,
                self.confidence_threshold,
            )
        ):
            raise invalid_descriptor("canvas sampling values are not finite")
        _nonnegative(self.stability_threshold, "canvas stability threshold")

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "canvas sampling"
    ) -> CanvasSampling:
        """Parse the sampling a worker's launch descriptor states."""
        data = _map(value, where)
        return cls(
            canvas_length=_uint(
                data.get("canvas_length"), f"{where}.canvas_length"
            ),
            max_steps=_uint(data.get("max_steps"), f"{where}.max_steps"),
            entropy_bound=_float(
                data.get("entropy_bound"), f"{where}.entropy_bound"
            ),
            t_min=_float(data.get("t_min"), f"{where}.t_min"),
            t_max=_float(data.get("t_max"), f"{where}.t_max"),
            confidence_threshold=_float(
                data.get("confidence_threshold"),
                f"{where}.confidence_threshold",
            ),
            stability_threshold=_uint(
                data.get("stability_threshold"), f"{where}.stability_threshold"
            ),
        )


@dataclass(frozen=True, slots=True)
class GenerationParams:
    """Sampling policy and token controls for autoregressive execution."""

    sampling: sampling.SamplingParams = field(
        default_factory=sampling.SamplingParams
    )
    # Tokenized negative prompt, used as negative-conditioning input.
    negative_token_ids: tuple[int, ...] = ()
    # Stop tokens; must be strictly ascending (canonical form, no duplicates).
    finish_token_ids: tuple[int, ...] = ()
    # Prompt tokens already computed at admission; the engine sets it from a
    # prefix-cache hit, and `RequestPool.start` begins the request with this
    # many tokens visible and computed.
    initial_position: int = 0
    # Block-diffusion sampling of a request generating in canvases.
    canvas: CanvasSampling | None = None

    def __post_init__(self) -> None:
        """Require a nonnegative initial position and canonical stop tokens.

        `SamplingParams` checks its own invariants when constructed.
        """
        _nonnegative(self.initial_position, "autoregressive initial position")
        if any(
            left >= right
            for left, right in zip(
                self.finish_token_ids,
                self.finish_token_ids[1:],
                strict=False,
            )
        ):
            raise invalid_descriptor(
                "autoregressive finish token ids are not canonical"
            )


@dataclass(frozen=True, slots=True)
class DiffusionParams:
    """Effective diffusion bounds and seed resolved by model preprocessing.

    Mirrors the Rust `DiffusionSamplingParams`.
    """

    # Output frames after model-specific alignment.
    num_frames: int
    # Video media units the decoder reconstructs, independent of rank count.
    video_units: int
    # Deterministic noise seed; nonnegative.
    seed: int
    # Output raster in pixels: the request's canvas.
    width: int
    height: int

    def __post_init__(self) -> None:
        """Require positive work bounds and a nonnegative deterministic seed."""
        for name in (
            "num_frames",
            "video_units",
            "width",
            "height",
        ):
            if getattr(self, name) < 1:
                raise invalid_descriptor(f"diffusion {name} must be positive")
        _nonnegative(self.seed, "diffusion seed")

    @property
    def canvas(self) -> image.Config:
        """The request's output raster."""
        return image.Config(self.height, self.width)


@dataclass(frozen=True, slots=True)
class NewRequest:
    """Binds one request key to its slot and execution-family parameters.

    The families are autoregressive (``generation``), image (``image``), and
    diffusion (``diffusion``). An image-generating request carries both
    ``generation`` and ``image``.
    """

    request_key: identity.RequestKey
    # Scheduler-assigned request-pool slot; slot 0 is reserved and rejected.
    request_pool_idx: int
    # At least one family parameter set must be present. ``generation`` is
    # keyed ``ar`` on the wire.
    generation: GenerationParams | None = None
    image: call.ImageParams | None = None
    diffusion: DiffusionParams | None = None
    # Prompt tokens; required (non-empty) for diffusion requests.
    prompt_token_ids: tuple[int, ...] = ()
    # Number of input images the request carries. Model preprocessing may
    # bound each input image by its share of a pixel budget the images
    # share (``PatchTransform.pixel_bound``).
    input_images: int = 0
    # A video request's task, presentation tags and conditions; present
    # exactly with ``diffusion``.
    video: VideoAdmission | None = None

    def __post_init__(self) -> None:
        """Require a positive slot and at least one family parameter set.

        Diffusion requests also need non-empty prompt tokens, each with its
        presentation tag. Each family's own parameters are validated by its
        record.
        """
        if self.diffusion is not None and not self.prompt_token_ids:
            raise invalid_descriptor(
                "diffusion prompt tokens must not be empty"
            )
        if (self.video is None) != (self.diffusion is None):
            raise invalid_descriptor(
                "a video admission carries both its sampling and its inputs"
            )
        if self.video is not None and len(self.video.text_tags) != len(
            self.prompt_token_ids
        ):
            raise invalid_descriptor(
                "every video prompt token requires one tag"
            )
        if self.request_pool_idx < 1:
            raise invalid_descriptor("request-pool index must be positive")
        if (
            self.generation is None
            and self.image is None
            and self.diffusion is None
        ):
            raise invalid_descriptor(
                "request start must declare one runtime-family parameter set"
            )


@dataclass(frozen=True, slots=True)
class BlockTable:
    """The complete unit table of one request slot and KV cache group.

    ``unit_ids`` lists the units of logical pages ``start_page..``,
    page-major: the group's ``units_per_page`` units of ``start_page`` come
    first. Earlier pages are retired window pages that no call reads.
    ``allocated_tokens`` is the absolute token extent the table covers; the
    worker checks it against the group's page shape.
    """

    # Request-pool slot owning the table; slot 0 is rejected.
    request_pool_idx: int
    # KV cache group addressed by the table.
    group_id: int
    # First logical page the units cover.
    start_page: int
    # Physical unit identifiers; unique, unit 0 rejected.
    unit_ids: tuple[int, ...]
    # Absolute token extent the table covers.
    allocated_tokens: int

    def __post_init__(self) -> None:
        """Validate the slot, group, start page and units."""
        if (
            self.request_pool_idx < 1
            or self.group_id < 0
            or self.start_page < 0
            or self.allocated_tokens < 0
            or any(unit < 1 for unit in self.unit_ids)
            or len(set(self.unit_ids)) != len(self.unit_ids)
        ):
            raise invalid_descriptor("block table is invalid")


@dataclass(frozen=True, slots=True)
class CacheUnitAllocation:
    """Physical KV units newly assigned to a request slot and KV group.

    The units are a subset of the batch's `BlockTable` for the same slot and
    group; the Rust `Batch::validate` checks this, `Batch.validate` does not.
    The worker resets every listed unit before a call uses it.
    """

    # Request-pool slot receiving the units; slot 0 is rejected.
    request_pool_idx: int
    group_id: int
    # Newly assigned physical units; non-empty, unique, unit 0 rejected.
    unit_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate the slot, group, and newly assigned units."""
        if (
            self.request_pool_idx < 1
            or self.group_id < 0
            or not self.unit_ids
            or any(unit < 1 for unit in self.unit_ids)
            or len(set(self.unit_ids)) != len(self.unit_ids)
        ):
            raise invalid_descriptor("cache-unit allocation is invalid")


def _validate_forward_inputs(
    call_count: int,
    call_indices: tuple[int, ...],
    request_pool_indices: tuple[int, ...],
    seq_lens: tuple[int, ...],
    query_lens: tuple[int, ...],
    write_kv: tuple[bool, ...],
) -> None:
    """Validate the columnar forward inputs before lanes index them.

    Every column must have one entry per row, each call index must address
    a call of the batch, no row may use slot 0, and each row needs a
    positive query length no longer than its sequence length.

    Raises:
        WorkerError: From `invalid_descriptor` on the first violated rule.
    """
    rows = len(call_indices)
    if any(
        len(column) != rows
        for column in (request_pool_indices, seq_lens, query_lens, write_kv)
    ):
        raise invalid_descriptor("forward input columns have different lengths")
    if any(index < 0 or index >= call_count for index in call_indices):
        raise invalid_descriptor("forward call index is outside its batch")
    if any(slot < 1 for slot in request_pool_indices):
        raise invalid_descriptor(
            "forward input carries the reserved request slot"
        )
    if any(
        total < query for total, query in zip(seq_lens, query_lens, strict=True)
    ) or any(length < 1 for length in query_lens):
        raise invalid_descriptor("forward input has invalid token lengths")


@dataclass(frozen=True, slots=True)
class LatentParams:
    """Solver-step range and optional paged storage of one trajectory.

    For a paged trajectory, `LatentPool` records the pages, units, and raster
    when the trajectory's slot is initialized and checks later calls against
    them.
    """

    request_key: identity.RequestKey
    call_id: identity.CallId
    # `LatentPool` pages in logical order; unique, page 0 rejected. Empty
    # exactly when ``latent_units`` is zero, which means the trajectory lives
    # in request-owned tensors instead of the pool.
    page_table: tuple[int, ...]
    # Latent rows stored in ``page_table``; a `LatentPool` page holds
    # ``page_units`` rows.
    latent_units: int
    # Output height and width in pixels.
    height: int
    width: int
    # The call runs solver steps [start_step, start_step + step_count).
    start_step: int
    step_count: int

    def __post_init__(self) -> None:
        """Validate the call identity, raster, and page-table/unit agreement.

        Raises:
            WorkerError: From `invalid_descriptor` when the call id's batch
                is not positive, the raster is empty, ``latent_units`` is
                negative, or the page table disagrees with the units,
                repeats a page, or carries page 0.
        """
        if self.call_id.batch_id < 1:
            raise invalid_descriptor("latent params call id must be positive")
        if min(self.height, self.width) < 1 or self.latent_units < 0:
            raise invalid_descriptor("latent dimensions must be positive")
        if (
            bool(self.page_table) != (self.latent_units > 0)
            or any(page < 1 for page in self.page_table)
            or len(set(self.page_table)) != len(self.page_table)
        ):
            raise invalid_descriptor(
                "latent params page table disagrees with its units, repeats "
                "a page, or carries page zero"
            )


class MediaTrack(StrEnum):
    """Independent media stream addressed by a bounded call."""

    VIDEO = "video"
    AUDIO = "audio"


@dataclass(frozen=True, slots=True)
class DecodeRange:
    """Selects a bounded range of media units for one media call.

    The Rust `Batch::validate` requires one for each video or audio decoding
    or encoding call in a batch.
    """

    request_key: identity.RequestKey
    call_id: identity.CallId
    # First media unit assigned to the call.
    cursor: int
    # Maximum media units the call may process; positive.
    max_units: int
    # Ranks of a distributed component that hold the call's units, in unit
    # order: each takes the next ``units_per_rank`` units and the last what
    # remains. Empty deals the units to the component's ranks in configured
    # order.
    ranks: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Validate the call identity, cursor, unit bound, and ranks."""
        if self.call_id.batch_id < 1 or self.max_units < 1:
            raise invalid_descriptor(
                "decode params identity and unit bound must be positive"
            )
        _nonnegative(self.cursor, "decode params cursor")
        if len(set(self.ranks)) != len(self.ranks):
            raise invalid_descriptor("decode params name each rank once")


@dataclass(frozen=True, slots=True)
class BufferAllocation:
    """Scheduler-chosen byte span for one persistent cross-call buffer."""

    buffer: identity.BufferId
    # Half-open byte span [offset, offset + bytes) in a device's
    # persistent-buffer arena; the end must fit in u64. `BufferPool` binds
    # the buffer at ``offset`` unless it places buffers compactly.
    offset: int
    bytes: int

    def __post_init__(self) -> None:
        """Require a non-empty span whose end fits in u64."""
        if self.offset < 0 or self.bytes < 1:
            raise invalid_descriptor("buffer params span is invalid")
        if self.offset + self.bytes > (1 << 64) - 1:
            raise invalid_descriptor("buffer params span overflows")


def _validate_buffer_allocations(
    calls: Sequence[call.Call],
    parameters: Sequence[BufferAllocation],
    where: str,
) -> None:
    """Validate persistent-buffer spans against each other and the calls.

    Buffer ids must be unique and spans must not overlap. Every buffer output
    of every call needs an allocation of at least its declared
    ``max_bytes``.

    Raises:
        WorkerError: From `invalid_descriptor` on the first violated rule.
    """
    by_id: dict[identity.BufferId, BufferAllocation] = {}
    spans: list[tuple[int, int]] = []
    for params in parameters:
        if params.buffer in by_id:
            raise invalid_descriptor(
                f"{where} repeats a buffer params identity"
            )
        by_id[params.buffer] = params
        spans.append((params.offset, params.offset + params.bytes))

    # Sorted half-open spans overlap only if one ends past the next start.
    spans.sort()
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise invalid_descriptor(f"{where} buffer parameters overlap")

    for scheduled in calls:
        for output in scheduled.buffer_outputs():
            output_allocation = by_id.get(output.buffer_id)
            if output_allocation is None:
                raise invalid_descriptor(
                    "persistent call output has no buffer params"
                )
            if output_allocation.bytes < output.max_bytes:
                raise invalid_descriptor(
                    "buffer params is smaller than its declared output"
                )


@dataclass(frozen=True, slots=True)
class Batch:
    """One numerical call on one component, with every request in it.

    ``__post_init__`` runs `validate`, so every constructed batch has passed
    it, except those `construction.batch_from_validated` assembles from
    frames the Rust decoder validated.
    """

    # Strictly increasing in each worker's submission order; `Executor.submit`
    # rejects a batch id that does not exceed every earlier one.
    batch_id: int
    # Positive sequence shared by collective participants. When the world
    # size exceeds one, batches with calls must launch in strictly increasing
    # order; `Executor` fails the rank fatally otherwise, because its peers
    # would wait in a collective it never joins.
    collective_seq: int = 1
    calls: tuple[call.Call, ...] = ()
    block_tables: tuple[BlockTable, ...] = ()
    new_cache_units: tuple[CacheUnitAllocation, ...] = ()

    # Columnar model-forward inputs: one row per forward, all columns the same
    # length. ``forward_call_indices`` indexes ``calls``; ``seq_lens`` counts
    # attended tokens (cached prefix plus query) and ``query_lens`` the tokens
    # the row evaluates; ``write_kv`` says whether the query tokens persist in
    # the KV cache.
    forward_call_indices: tuple[int, ...] = ()
    request_pool_indices: tuple[int, ...] = ()
    seq_lens: tuple[int, ...] = ()
    query_lens: tuple[int, ...] = ()
    write_kv: tuple[bool, ...] = ()

    latent_params: tuple[LatentParams, ...] = ()
    decode_ranges: tuple[DecodeRange, ...] = ()
    buffer_allocations: tuple[BufferAllocation, ...] = ()
    commands: tuple[BatchCommand, ...] = ()
    input_products: tuple[TensorPublication, ...] = ()
    kv_inputs: tuple[transfer.KvTransfer, ...] = ()

    def __post_init__(self) -> None:
        """Run `validate`."""
        self.validate()

    @property
    def admissions(self) -> tuple[NewRequest, ...]:
        """The `NewRequest` of every `Start` command, in command order."""
        return tuple(
            command.request
            for command in self.commands
            if isinstance(command, Start)
        )

    def validate(self) -> None:
        """Check the cross-record rules of a batch.

        The batch carries a call or a command, a positive collective
        sequence, and valid forward inputs. Its calls belong to
        ``batch_id``, have unique ids, cover each request at most once, and
        share one kind and component. Admissions are unique, repeated
        commands are identical, each input product is declared by a call and
        supplied once, each KV input has one ``KV_INSTALL`` consumer within
        its byte bound, and buffer allocations cover the buffer outputs.

        It covers part of the Rust `Batch::validate`. Among other rules, it
        does not validate each call, require unique block tables or
        cache-page allocations inside their tables, match latent params and
        decode ranges to the calls, or check latent page overlap.

        Raises:
            WorkerError: From `invalid_descriptor` on the first violated rule.
        """
        if not self.calls and not self.commands:
            raise invalid_descriptor(
                "a submission batch must carry at least one call or command"
            )
        if self.collective_seq < 1:
            raise invalid_descriptor(
                "batch collective sequence must be positive"
            )
        _validate_forward_inputs(
            len(self.calls),
            self.forward_call_indices,
            self.request_pool_indices,
            self.seq_lens,
            self.query_lens,
            self.write_kv,
        )
        # Every computation belongs to this logical batch, with unique
        # computation identities and at most one call per request.
        computation_ids = [call.call_id for call in self.calls]
        if any(
            identity.batch_id != self.batch_id for identity in computation_ids
        ):
            raise invalid_descriptor(
                "computation identity belongs to another logical batch"
            )
        if len(set(computation_ids)) != len(computation_ids):
            raise invalid_descriptor(
                "a submission batch repeats a computation identity"
            )
        request_keys = [call.request_key for call in self.calls]
        if len(set(request_keys)) != len(request_keys):
            raise invalid_descriptor(
                "a submission batch carries multiple calls for one request"
            )
        # A batch is one numerical call on one component: every call in it
        # performs the same computation through the same component, so the rank
        # executes it as a single homogeneous group and returns one result.
        calls = {(call.kind, call.component) for call in self.calls}
        if len(calls) > 1:
            raise invalid_descriptor(
                "a submission batch mixes call kinds or components"
            )
        admitted = [admission.request_key for admission in self.admissions]
        if len(set(admitted)) != len(admitted):
            raise invalid_descriptor(
                "a submission batch carries a duplicate admission"
            )

        # Lifecycle commands may repeat identically but not conflict. A
        # command's identity is its request, variant, and freed buffer.
        identities: dict[
            tuple[identity.RequestKey, int, identity.BufferId | None],
            BatchCommand,
        ] = {}
        for command in self.commands:
            buffer = command.buffer if isinstance(command, Free) else None
            command_key = (
                command.request_key,
                _command_variant_index(command),
                buffer,
            )
            existing = identities.get(command_key)
            if existing is not None and existing != command:
                raise invalid_descriptor(
                    "a submission batch reuses a command identity with "
                    "different content"
                )
            identities[command_key] = command

        # Every input product payload must feed a declared call input or
        # predicate, exactly once.
        declared_inputs = {
            product
            for call in self.calls
            for product in (*call.tensor_inputs(), call.predicate)
            if product is not None
        }
        supplied_inputs: set[tensor.TensorRef] = set()
        for payload in self.input_products:
            product = payload.product
            if product not in declared_inputs:
                raise invalid_descriptor(
                    "an input product payload is not declared by any call"
                )
            if product in supplied_inputs:
                raise invalid_descriptor(
                    "a submission batch repeats an input product payload"
                )
            supplied_inputs.add(product)

        # Each KV transfer's descriptor must fit the transfer-handle bound and
        # its source must be unique. It installs into exactly one
        # ``KV_INSTALL`` call whose transfer-byte bound covers its tensors.
        sources: set[identity.BufferId] = set()
        for publication in self.kv_inputs:
            publication.encoded_size_bound()
            if publication.source in sources:
                raise invalid_descriptor("batch repeats a KV input")
            sources.add(publication.source)
            consumers = tuple(
                call
                for call in self.calls
                if call.kv_input == publication.source
            )
            if (
                len(consumers) != 1
                or consumers[0].kind is not call.TransferMode.KV_INSTALL
            ):
                raise invalid_descriptor(
                    "KV transfer requires one installation consumer"
                )
            if (
                sum(tensor.nbytes for tensor in publication.tensors)
                > consumers[0].bounds.max_transfer_bytes
            ):
                raise invalid_descriptor(
                    "KV input exceeds its installation transfer-byte bound"
                )
        _validate_buffer_allocations(
            self.calls, self.buffer_allocations, "batch"
        )


@dataclass(frozen=True, slots=True)
class TensorPublication:
    """A tensor identity and the physical metadata needed by its consumer.

    Batches carry these as host-supplied ``input_products``, and a rank
    reports the products it published in its `BatchOutput`.
    """

    product: tensor.TensorRef
    value: transfer.TransferValue

    def encoded_size_bound(self) -> int:
        """Return the estimated encoded size of the tensor's locators.

        Raises:
            WorkerError: From `invalid_descriptor` when the estimate exceeds
                ``transfer.MAX_TRANSFER_HANDLE_BYTES`` or a locator names an
                unknown transport.
        """
        size = transfer._tensor_transfers_size((self.value.tensor,))
        if size > transfer.MAX_TRANSFER_HANDLE_BYTES:
            raise invalid_descriptor("transfer metadata exceeds its byte bound")
        return size

    def to_mapping(self) -> dict[str, object]:
        """Encode the tensor identity and its kind-tagged transfer value."""
        typed = self.value
        value: dict[str, object] = {
            "height": typed.height,
            "width": typed.width,
            "tensor": typed.tensor.to_mapping(),
        }
        if isinstance(typed, transfer.EncoderTransferValue):
            kind = "encoder"
            value["payload_kind"] = typed.payload_kind
        elif isinstance(typed, transfer.DeviceProductTransferValue):
            kind = "device_product"
            value["value_range"] = typed.value_range
        else:
            kind = "latent"
            value["latent_units"] = typed.latent_units
            value["step"] = typed.step
        return {
            "product": self.product.to_mapping(),
            "value": {"kind": kind, "value": value},
        }
