"""Scheduler-to-worker execution records.

A `Batch` is one numerical call on one component together with everything a
rank needs to run it: the per-request `Call` descriptors, the lifecycle
commands (`Start`, `Finish`, `Free`) applied with it, the KV unit tables and
latent, decode, and persistent-buffer allocations the scheduler chose, and
host-supplied input tensors (`TensorPublication`). The records mirror the
Rust `Batch` in `uniserve_worker_ipc`, whose decoder validates every frame
(`Batch::validate`); these records hold what it accepted and check nothing
again.

The PyO3 transport (`crates/worker-ipc-py`) builds each member record
through its constructor. It calls `BlockTable`, `CacheUnitAllocation`,
`Start`, `Finish`, and `Free` positionally, so their field order is part of
that contract, and the other records by keyword, so their field names are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

from uniserve import sampling
from uniserve.media import image
from uniserve_worker.protocol import call, identity, tensor, transfer
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
    Every retained buffer is owned by ``request_key`` and listed once.
    """

    request_key: identity.RequestKey
    retained_buffers: tuple[identity.BufferId, ...] = ()


@dataclass(frozen=True, slots=True)
class Free:
    """Releases one scheduler-owned persistent buffer."""

    buffer: identity.BufferId

    @property
    def request_key(self) -> identity.RequestKey:
        """Expose the request generation owning the freed persistent buffer."""
        return self.buffer.owner


BatchCommand: TypeAlias = Start | Finish | Free


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


@dataclass(frozen=True, slots=True)
class DiffusionSamplingParams:
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
    diffusion: DiffusionSamplingParams | None = None
    # Prompt tokens; required (non-empty) for diffusion requests.
    prompt_token_ids: tuple[int, ...] = ()
    # Number of input images the request carries. Model preprocessing may
    # bound each input image by its share of a pixel budget the images
    # share (``PatchTransform.pixel_bound``).
    input_images: int = 0
    # A video request's task, presentation tags and conditions; present
    # exactly with ``diffusion``.
    video: VideoAdmission | None = None


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


@dataclass(frozen=True, slots=True)
class LatentParams:
    """Paged storage and raster of one image trajectory.

    The scheduler's allocator assigns the pages. `LatentPool` records the
    pages, units, and raster when the trajectory's slot is initialized and
    checks later calls against them. A standalone denoiser's request slot
    fixes its own pages, so its calls carry no parameters.
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


@dataclass(frozen=True, slots=True)
class BufferAllocation:
    """Scheduler-chosen byte span for one persistent cross-call buffer."""

    buffer: identity.BufferId
    # Half-open byte span [offset, offset + bytes) in a device's
    # persistent-buffer arena; the end must fit in u64. `BufferPool` binds
    # the buffer at ``offset`` unless it places buffers compactly.
    offset: int
    bytes: int


@dataclass(frozen=True, slots=True)
class Batch:
    """One numerical call on one component, with every request in it.

    The Rust decoder validates a frame's batch (`Batch::validate`) before the
    transport builds this record from it.
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

    @property
    def admissions(self) -> tuple[NewRequest, ...]:
        """The `NewRequest` of every `Start` command, in command order."""
        return tuple(
            command.request
            for command in self.commands
            if isinstance(command, Start)
        )


@dataclass(frozen=True, slots=True)
class TensorPublication:
    """A tensor identity and the physical metadata needed by its consumer.

    Batches carry these as host-supplied ``input_products``, and a rank
    reports the products it published in its `BatchOutput`.
    """

    product: tensor.TensorRef
    value: transfer.TransferValue

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
