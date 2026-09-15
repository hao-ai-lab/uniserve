"""Scheduler-to-worker execution records and their validation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

from uniserve import sampling

from ..foundation.errors import invalid_descriptor
from . import identity, operation, tensor, transfer
from .validation import (
    _bool,
    _map,
    _nonnegative,
    _seq,
    _str,
    _tagged,
    _uint,
    _uints,
)


def native_run(
    batch_id: int,
    run_id: int,
    collective_seq: int,
    operations: tuple[operation.ScheduledRequest, ...],
    block_tables: tuple[BlockTable, ...],
    new_cache_pages: tuple[CachePageAllocation, ...],
    forward_inputs: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[bool, ...],
    ],
    latent_params: Sequence[object],
    decode_ranges: Sequence[object],
    buffer_allocations: Sequence[object],
    commands: tuple[BatchCommand, ...],
    input_products: Sequence[object],
    kv_inputs: Sequence[object],
) -> ScheduleBatch:
    """Assemble a validated run from transport-constructed members.

    Called by the Rust IPC transport, which has already decoded and validated
    every field. Construction therefore bypasses ``ScheduleBatch.__init__`` so
    typed leaves are not reparsed and ``__post_init__`` validation is not
    repeated.
    """
    run = object.__new__(ScheduleBatch)
    set_field = object.__setattr__
    set_field(run, "batch_id", batch_id)
    set_field(run, "run_id", run_id)
    set_field(run, "collective_seq", collective_seq)
    set_field(run, "operations", operations)
    set_field(run, "block_tables", block_tables)
    set_field(run, "new_cache_pages", new_cache_pages)
    set_field(run, "forward_operation_indices", forward_inputs[0])
    set_field(run, "request_pool_indices", forward_inputs[1])
    set_field(run, "seq_lens", forward_inputs[2])
    set_field(run, "query_lens", forward_inputs[3])
    set_field(run, "write_kv", forward_inputs[4])

    set_field(
        run,
        "latent_params",
        tuple(
            LatentParams.from_mapping(item, f"run.latent_params[{index}]")
            for index, item in enumerate(latent_params)
        ),
    )
    set_field(
        run,
        "decode_ranges",
        tuple(
            DecodeRange.from_mapping(item, f"run.decode_ranges[{index}]")
            for index, item in enumerate(decode_ranges)
        ),
    )
    set_field(
        run,
        "buffer_allocations",
        tuple(
            BufferAllocation.from_mapping(
                item, f"run.buffer_allocations[{index}]"
            )
            for index, item in enumerate(buffer_allocations)
        ),
    )
    set_field(run, "commands", commands)

    set_field(
        run,
        "input_products",
        tuple(
            TensorPublication.from_mapping(item, f"run.input_products[{index}]")
            for index, item in enumerate(input_products)
        ),
    )
    set_field(
        run,
        "kv_inputs",
        tuple(transfer.KvTransfer.from_mapping(value) for value in kv_inputs),
    )
    return run


@dataclass(frozen=True, slots=True)
class Start:
    """Registers a new request and its immutable execution parameters."""

    request: NewRequest

    @property
    def request_key(self) -> identity.RequestKey:
        """Expose the request generation registered by this start command."""
        return self.request.request_key

    @classmethod
    def from_mapping(cls, value: object, where: str = "start command") -> Start:
        """Parse a request admission from a start-command payload."""
        data = _map(value, where)
        return cls(
            request=NewRequest.from_mapping(
                data.get("request"), f"{where}.request"
            )
        )


@dataclass(frozen=True, slots=True)
class Finish:
    """Close a lineage while preserving retained persistent allocations.

    The retained allocations are explicit. Retained buffers are owned outside
    the request and remain readable until Free; request slots, KV state, and
    unretained products retire after readers finish.
    """

    request_key: identity.RequestKey
    retained_buffers: tuple[identity.BufferId, ...] = ()

    def __post_init__(self) -> None:
        _validate_retained_buffers(self.request_key, self.retained_buffers)


def _validate_retained_buffers(
    request: identity.RequestKey, retained: tuple[identity.BufferId, ...]
) -> None:
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
    """Return the stable wire-union variant index for a lifecycle command."""
    if isinstance(command, Start):
        return 0
    if isinstance(command, Finish):
        return 1
    return 2


def command_from_mapping(
    value: object,
    where: str = "batch command",
) -> BatchCommand:
    """Parse a tagged start, finish, or free lifecycle command."""
    kind, payload = _tagged(value, where)
    data = _map(payload, f"{where}.value")
    if kind == "start":
        return Start.from_mapping(data, f"{where}.value")

    # Free carries no request_key; finish falls back to the validating parser
    # when the fast decode does not recognize the value.
    request_key = identity._fast_request_key(data.get("request_key"))
    if kind != "free" and request_key is None:
        request_key = identity.RequestKey.from_mapping(
            data.get("request_key"), f"{where}.value.request_key"
        )

    if kind == "finish":
        assert request_key is not None
        command: BatchCommand = Finish(
            request_key=request_key,
            retained_buffers=tuple(
                identity.BufferId.from_mapping(
                    buffer, f"{where}.value.retained_buffers[{index}]"
                )
                for index, buffer in enumerate(
                    _seq(
                        data.get("retained_buffers"),
                        f"{where}.value.retained_buffers",
                    )
                )
            ),
        )
    elif kind == "free":
        command = Free(
            buffer=identity.BufferId.from_mapping(
                data.get("buffer"), f"{where}.value.buffer"
            )
        )
    else:
        raise invalid_descriptor(f"{where} has unknown variant {kind!r}")
    return command


def command_to_mapping(command: BatchCommand) -> dict[str, object]:
    """Encode a lifecycle command with its stable variant tag."""
    if isinstance(command, Start):
        return {
            "kind": "start",
            "value": {"request": command.request.to_mapping()},
        }
    if isinstance(command, Finish):
        return {
            "kind": "finish",
            "value": {
                "request_key": command.request_key.to_mapping(),
                "retained_buffers": [
                    buffer.to_mapping() for buffer in command.retained_buffers
                ],
            },
        }
    return {"kind": "free", "value": {"buffer": command.buffer.to_mapping()}}


@dataclass(frozen=True, slots=True)
class GenerationParams:
    """Sampling policy and token controls for autoregressive execution."""

    sampling: sampling.SamplingParams = field(
        default_factory=sampling.SamplingParams
    )
    # Tokens suppressed from generation.
    negative_token_ids: tuple[int, ...] = ()
    # Stop tokens; must be strictly ascending (canonical form, no duplicates).
    finish_token_ids: tuple[int, ...] = ()
    # Accepted prefix length at admission (tokens already computed, e.g.
    # cache reuse).
    initial_position: int = 0

    def __post_init__(self) -> None:
        """Validate prompt, generation limit, and draft tokens.

        Also validates the sampling policy.
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

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "autoregressive parameters"
    ) -> GenerationParams:
        """Parse sampling policy, token controls, and initial position.

        Parses parameters for autoregressive work.
        """
        data = _map(value, where)
        return cls(
            sampling=operation._sampling_params_from_mapping(
                data.get("sampling", {}), f"{where}.sampling"
            ),
            negative_token_ids=_uints(
                data.get("negative_token_ids", ()),
                f"{where}.negative_token_ids",
            ),
            finish_token_ids=_uints(
                data.get("finish_token_ids", ()), f"{where}.finish_token_ids"
            ),
            initial_position=_uint(
                data.get("initial_position", 0), f"{where}.initial_position"
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize autoregressive sampling and token controls.

        Produces the admission wire mapping.
        """
        return {
            "sampling": operation._sampling_params_to_mapping(self.sampling),
            "negative_token_ids": list(self.negative_token_ids),
            "finish_token_ids": list(self.finish_token_ids),
            "initial_position": self.initial_position,
        }


@dataclass(frozen=True, slots=True)
class DiffusionParams:
    """Effective diffusion bounds and seed resolved by model preprocessing."""

    num_frames: int
    num_decode_chunks: int
    num_inference_steps: int
    # Deterministic noise seed; nonnegative.
    seed: int

    def __post_init__(self) -> None:
        """Require positive work bounds and a nonnegative deterministic seed."""
        for name in ("num_frames", "num_decode_chunks", "num_inference_steps"):
            if getattr(self, name) < 1:
                raise invalid_descriptor(f"diffusion {name} must be positive")
        _nonnegative(self.seed, "diffusion seed")

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "diffusion sampling"
    ) -> DiffusionParams:
        """Parse the core's effective diffusion parameters."""
        data = _map(value, where)
        return cls(
            num_frames=_uint(data.get("num_frames"), f"{where}.num_frames"),
            num_decode_chunks=_uint(
                data.get("num_decode_chunks"), f"{where}.num_decode_chunks"
            ),
            num_inference_steps=_uint(
                data.get("num_inference_steps"), f"{where}.num_inference_steps"
            ),
            seed=_uint(data.get("seed"), f"{where}.seed"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize the shared diffusion parameter definition."""
        return {
            "num_frames": self.num_frames,
            "num_decode_chunks": self.num_decode_chunks,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class NewRequest:
    """Binds one request key to its execution-family parameters.

    The families are autoregressive, multimodal, and diffusion.
    """

    request_key: identity.RequestKey
    # 1-based request-pool slot; index 0 is reserved.
    request_pool_idx: int
    # Exactly one of the three runtime-family parameter sets must be present.
    generation: GenerationParams | None = None
    image: operation.ImageParams | None = None
    diffusion: DiffusionParams | None = None
    # Prompt tokens; required (non-empty) for diffusion requests.
    prompt_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Require a valid request slot and family parameters.

        Also requires diffusion prompt tokens.
        """
        if self.diffusion is not None and not self.prompt_token_ids:
            raise invalid_descriptor(
                "diffusion prompt tokens must not be empty"
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

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "admission"
    ) -> NewRequest:
        """Parse a request slot.

        Also parses its optional execution-family parameter sets.
        """
        data = _map(value, where)
        image_data = data.get("umm")
        admission = cls(
            request_key=identity.RequestKey.from_mapping(
                data.get("request_key"), f"{where}.request_key"
            ),
            request_pool_idx=_uint(
                data.get("request_pool_idx"), f"{where}.request_pool_idx"
            ),
            prompt_token_ids=_uints(
                data.get("prompt_token_ids", ()), f"{where}.prompt_token_ids"
            ),
            generation=(
                None
                if data.get("ar") is None
                else GenerationParams.from_mapping(data["ar"], f"{where}.ar")
            ),
            image=(
                None
                if image_data is None
                else operation.ImageParams.from_mapping(
                    _map(image_data, f"{where}.umm").get("image", {}),
                    f"{where}.umm.image",
                )
            ),
            diffusion=(
                None
                if data.get("diffusion") is None
                else DiffusionParams.from_mapping(
                    data["diffusion"], f"{where}.diffusion"
                )
            ),
        )
        return admission

    def to_mapping(self) -> dict[str, object]:
        """Serialize request identity, slot, and family parameters.

        Produces the admission wire mapping.
        """
        return {
            "request_key": self.request_key.to_mapping(),
            "request_pool_idx": self.request_pool_idx,
            "prompt_token_ids": list(self.prompt_token_ids),
            "ar": None
            if self.generation is None
            else self.generation.to_mapping(),
            "umm": None
            if self.image is None
            else {"image": self.image.to_mapping()},
            "diffusion": None
            if self.diffusion is None
            else self.diffusion.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class BlockTable:
    """Maps a request and KV group to its ordered physical pages.

    Also maps to the allocated token extent.
    """

    # 1-based request-pool slot.
    request_pool_idx: int
    group_id: int
    # Ordered 1-based physical page identifiers; unique within the table.
    page_ids: tuple[int, ...]
    # Tokens covered by this table; zero only when no pages are installed.
    allocated_tokens: int

    def __post_init__(self) -> None:
        """Validate request/group identity and ordered pages.

        Also validates the allocated token extent.
        """
        if (
            self.request_pool_idx < 1
            or self.group_id < 0
            or self.allocated_tokens < 0
            or any(page < 1 for page in self.page_ids)
            or len(set(self.page_ids)) != len(self.page_ids)
            or (not self.page_ids and self.allocated_tokens != 0)
        ):
            raise invalid_descriptor("block table is invalid")

    @classmethod
    def from_mapping(
        cls,
        value: object,
        where: str = "block table",
    ) -> BlockTable:
        """Parse an installed KV page table and its allocated token extent."""
        data = _map(value, where)

        def uint_field(name: str) -> int:
            """Decode a nonnegative integer field via the fast scalar path."""
            raw = data.get(name)
            return (
                raw
                if type(raw) is int and raw >= 0
                else _uint(raw, f"{where}.{name}")
            )

        page_ids = operation._fast_uints(data.get("page_ids", ()))
        if page_ids is None:
            page_ids = _uints(data.get("page_ids", ()), f"{where}.page_ids")
        fields = (
            uint_field("request_pool_idx"),
            uint_field("group_id"),
            page_ids,
            uint_field("allocated_tokens"),
        )
        return cls(*fields)

    def to_mapping(self) -> dict[str, object]:
        """Serialize an installed request-and-group KV page table."""
        return {
            "request_pool_idx": self.request_pool_idx,
            "group_id": self.group_id,
            "page_ids": list(self.page_ids),
            "allocated_tokens": self.allocated_tokens,
        }


@dataclass(frozen=True, slots=True)
class CachePageAllocation:
    """Declares newly assigned physical pages for a request KV group."""

    # 1-based request-pool slot.
    request_pool_idx: int
    group_id: int
    # Newly assigned 1-based physical pages; non-empty and unique.
    page_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate newly assigned page identifiers.

        Also validates request/group ownership.
        """
        if (
            self.request_pool_idx < 1
            or self.group_id < 0
            or not self.page_ids
            or any(page < 1 for page in self.page_ids)
            or len(set(self.page_ids)) != len(self.page_ids)
        ):
            raise invalid_descriptor("cache-page allocation is invalid")

    @classmethod
    def from_mapping(
        cls,
        value: object,
        where: str = "cache-page allocation",
    ) -> CachePageAllocation:
        """Parse newly assigned physical KV pages.

        Parses pages for one request and cache group.
        """
        data = _map(value, where)

        def uint_field(name: str) -> int:
            """Decode a nonnegative integer field via the fast scalar path."""
            raw = data.get(name)
            return (
                raw
                if type(raw) is int and raw >= 0
                else _uint(raw, f"{where}.{name}")
            )

        page_ids = operation._fast_uints(data.get("page_ids", ()))
        if page_ids is None:
            page_ids = _uints(data.get("page_ids", ()), f"{where}.page_ids")
        fields = (
            uint_field("request_pool_idx"),
            uint_field("group_id"),
            page_ids,
        )
        return cls(*fields)

    def to_mapping(self) -> dict[str, object]:
        """Serialize newly assigned KV pages for lane registration."""
        return {
            "request_pool_idx": self.request_pool_idx,
            "group_id": self.group_id,
            "page_ids": list(self.page_ids),
        }


def _validate_forward_inputs(
    operation_count: int,
    operation_indices: tuple[int, ...],
    request_pool_indices: tuple[int, ...],
    seq_lens: tuple[int, ...],
    query_lens: tuple[int, ...],
    write_kv: tuple[bool, ...],
) -> None:
    """Validate aligned model inputs.

    Validation happens before lane preparation indexes their columns.
    """
    rows = len(operation_indices)
    if any(
        len(column) != rows
        for column in (request_pool_indices, seq_lens, query_lens, write_kv)
    ):
        raise invalid_descriptor("forward input columns have different lengths")
    if any(
        index < 0 or index >= operation_count for index in operation_indices
    ):
        raise invalid_descriptor("forward operation index is outside its batch")
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
    """Solver-step range and optional paged storage.

    Applies to one request trajectory.
    """

    request_key: identity.RequestKey
    op_id: identity.ComputationId
    # 1-based physical pages backing paged latent storage; empty iff
    # latent_units is zero.
    page_table: tuple[int, ...]
    # Number of latent units (trajectory rows); zero iff page_table is empty.
    latent_units: int
    # Raster shape of each latent frame.
    height: int
    width: int
    # Inclusive solver-step window [start_step, start_step + step_count).
    start_step: int
    step_count: int

    def __post_init__(self) -> None:
        """Validate the operation identity and raster shape.

        Also validates page-table/unit consistency.
        """
        if self.op_id.batch_id < 1:
            raise invalid_descriptor(
                "latent params operation id must be positive"
            )
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

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "latent params"
    ) -> LatentParams:
        """Parse request-owned latent pages and raster shape.

        Also parses the solver-step range.
        """
        data = _map(value, where)
        return cls(
            request_key=identity.RequestKey.from_mapping(
                data.get("request_key"), f"{where}.request_key"
            ),
            op_id=identity.ComputationId.from_mapping(
                data.get("op_id"), f"{where}.op_id"
            ),
            page_table=_uints(
                data.get("page_table", ()), f"{where}.page_table"
            ),
            latent_units=_uint(
                data.get("latent_units"), f"{where}.latent_units"
            ),
            height=_uint(data.get("height"), f"{where}.height"),
            width=_uint(data.get("width"), f"{where}.width"),
            start_step=_uint(data.get("start_step"), f"{where}.start_step"),
            step_count=_uint(data.get("step_count"), f"{where}.step_count"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize a physical latent trajectory params for lane execution."""
        return {
            "request_key": self.request_key.to_mapping(),
            "op_id": self.op_id.to_mapping(),
            "page_table": list(self.page_table),
            "latent_units": self.latent_units,
            "height": self.height,
            "width": self.width,
            "start_step": self.start_step,
            "step_count": self.step_count,
        }


class MediaTrack(StrEnum):
    """Independent media stream addressed by a bounded operation."""

    VIDEO = "video"
    AUDIO = "audio"


@dataclass(frozen=True, slots=True)
class DecodeRange:
    """Selects a bounded temporal range for a concrete media computation."""

    request_key: identity.RequestKey
    op_id: identity.ComputationId
    # Position (in media units) at which reconstruction resumes.
    cursor: int
    # Maximum media units this operation may emit.
    max_units: int

    def __post_init__(self) -> None:
        """Validate the operation identity, cursor, and unit bound."""
        if self.op_id.batch_id < 1 or self.max_units < 1:
            raise invalid_descriptor(
                "decode params identity and unit bound must be positive"
            )
        _nonnegative(self.cursor, "decode params cursor")

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "decode params"
    ) -> DecodeRange:
        """Parse the reconstruction cursor and bounded unit count.

        Parses parameters for one operation.
        """
        data = _map(value, where)
        return cls(
            request_key=identity.RequestKey.from_mapping(
                data.get("request_key"), f"{where}.request_key"
            ),
            op_id=identity.ComputationId.from_mapping(
                data.get("op_id"), f"{where}.op_id"
            ),
            cursor=_uint(data.get("cursor"), f"{where}.cursor"),
            max_units=_uint(data.get("max_units"), f"{where}.max_units"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize a media reconstruction params for lane execution."""
        return {
            "request_key": self.request_key.to_mapping(),
            "op_id": self.op_id.to_mapping(),
            "cursor": self.cursor,
            "max_units": self.max_units,
        }


@dataclass(frozen=True, slots=True)
class BufferAllocation:
    """Assigns a product to a bounded slice of persistent storage.

    The storage is scheduler-managed.
    """

    buffer: identity.BufferId
    # Byte range [offset, offset + bytes) within the buffer; must not
    # overflow u64.
    offset: int
    bytes: int

    def __post_init__(self) -> None:
        """Validate the persistent-buffer identifier and offset.

        Also validates the bounded shape.
        """
        if self.offset < 0 or self.bytes < 1:
            raise invalid_descriptor("buffer params span is invalid")
        if self.offset + self.bytes > (1 << 64) - 1:
            raise invalid_descriptor("buffer params span overflows")

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "buffer params"
    ) -> BufferAllocation:
        """Parse a bounded byte slice of persistent storage.

        The storage is scheduler-managed.
        """
        data = _map(value, where)
        return cls(
            buffer=identity.BufferId.from_mapping(
                data.get("buffer"), f"{where}.buffer"
            ),
            offset=_uint(data.get("offset"), f"{where}.offset"),
            bytes=_uint(data.get("bytes"), f"{where}.bytes"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize persistent-buffer identity, offset, and byte extent."""
        return {
            "buffer": self.buffer.to_mapping(),
            "offset": self.offset,
            "bytes": self.bytes,
        }


def _validate_buffer_allocations(
    operations: Sequence[operation.ScheduledRequest],
    parameters: Sequence[BufferAllocation],
    where: str,
) -> None:
    """Validate persistent-buffer parameters.

    Validation checks producing operations and shape bounds.
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

    spans.sort()
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise invalid_descriptor(f"{where} buffer parameters overlap")

    for scheduled in operations:
        for output in scheduled.buffer_outputs():
            output_allocation = by_id.get(output.buffer_id)
            if output_allocation is None:
                raise invalid_descriptor(
                    "persistent operation output has no buffer params"
                )
            if output_allocation.bytes < output.max_bytes:
                raise invalid_descriptor(
                    "buffer params is smaller than its declared output"
                )


@dataclass(frozen=True, slots=True)
class ScheduleBatch:
    """Describes one scheduler-submitted collection of operations."""

    batch_id: int
    run_id: int
    # Monotonic sequence number ordering collective communication across
    # workers.
    collective_seq: int = 1
    operations: tuple[operation.ScheduledRequest, ...] = ()
    block_tables: tuple[BlockTable, ...] = ()
    new_cache_pages: tuple[CachePageAllocation, ...] = ()

    # Columnar model-forward inputs: one row per forward, all columns the same
    # length. seq_lens counts total tokens per row, query_lens the new tokens.
    forward_operation_indices: tuple[int, ...] = ()
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
        """Validate run identity and lifecycle-operation consistency."""
        self.validate()

    @property
    def admissions(self) -> tuple[NewRequest, ...]:
        """Extract new-request payloads from lifecycle commands.

        Preserves submission order.
        """
        return tuple(
            command.request
            for command in self.commands
            if isinstance(command, Start)
        )

    def validate(self) -> None:
        """Enforce run identity, command ordering, and operation counts.

        Also enforces token bounds.
        """
        if not self.operations and not self.commands:
            raise invalid_descriptor(
                "a submission batch must carry at least one operation or "
                "command"
            )
        if self.collective_seq < 1:
            raise invalid_descriptor("run collective sequence must be positive")
        _validate_forward_inputs(
            len(self.operations),
            self.forward_operation_indices,
            self.request_pool_indices,
            self.seq_lens,
            self.query_lens,
            self.write_kv,
        )
        # Every computation belongs to this logical batch, with unique
        # computation identities and at most one operation per request.
        computation_ids = [operation.op_id for operation in self.operations]
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
        request_keys = [operation.request_key for operation in self.operations]
        if len(set(request_keys)) != len(request_keys):
            raise invalid_descriptor(
                "a submission batch carries multiple operations for one request"
            )
        admitted = [admission.request_key for admission in self.admissions]
        if len(set(admitted)) != len(admitted):
            raise invalid_descriptor(
                "a submission batch carries a duplicate admission"
            )

        # Lifecycle commands may repeat identically but not conflict.
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

        # Every input product payload must feed a declared operation input or
        # predicate, exactly once.
        declared_inputs = {
            product
            for operation in self.operations
            for product in (*operation.tensor_inputs(), operation.predicate)
            if product is not None
        }
        supplied_inputs: set[tensor.TensorRef] = set()
        for payload in self.input_products:
            product = payload.product
            if product not in declared_inputs:
                raise invalid_descriptor(
                    "an input product payload is not declared by any operation"
                )
            if product in supplied_inputs:
                raise invalid_descriptor(
                    "a submission batch repeats an input product payload"
                )
            supplied_inputs.add(product)

        # Each KV transfer installs into exactly one operation and must fit
        # that operation's transfer-byte bound.
        sources: set[identity.BufferId] = set()
        for publication in self.kv_inputs:
            publication.encoded_size_bound()
            if publication.source in sources:
                raise invalid_descriptor("run repeats a KV input")
            sources.add(publication.source)
            consumers = tuple(
                operation
                for operation in self.operations
                if operation.kv_input == publication.source
            )
            if (
                len(consumers) != 1
                or consumers[0].kind is not operation.TransferMode.KV_INSTALL
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
            self.operations, self.buffer_allocations, "run"
        )

    @classmethod
    def from_mapping(cls, value: object) -> ScheduleBatch:
        """Parse a scheduler run and validate it.

        Validation covers all lifecycle commands and physical inputs.
        """
        data = _map(value, "execute run")
        batch_id = _uint(data.get("batch_id"), "execute run.batch_id")
        run_id = _uint(data.get("run_id"), "execute run.run_id")

        operations = tuple(
            operation.ScheduledRequest.from_mapping(
                item, f"execute run.operations[{index}]"
            )
            for index, item in enumerate(
                _seq(data.get("operations", ()), "execute run.operations")
            )
        )
        commands = tuple(
            command_from_mapping(
                item,
                f"execute run.commands[{index}]",
            )
            for index, item in enumerate(
                _seq(data.get("commands", ()), "execute run.commands")
            )
        )
        input_products = tuple(
            TensorPublication.from_mapping(
                item, f"execute run.input_products[{index}]"
            )
            for index, item in enumerate(
                _seq(
                    data.get("input_products", ()), "execute run.input_products"
                )
            )
        )

        return cls(
            batch_id=batch_id,
            run_id=run_id,
            collective_seq=_uint(
                data.get("collective_seq"), "execute run.collective_seq"
            ),
            operations=operations,
            block_tables=tuple(
                BlockTable.from_mapping(
                    item, f"execute run.block_tables[{index}]"
                )
                for index, item in enumerate(
                    _seq(
                        data.get("block_tables", ()), "execute run.block_tables"
                    )
                )
            ),
            new_cache_pages=tuple(
                CachePageAllocation.from_mapping(
                    item, f"execute run.new_cache_pages[{index}]"
                )
                for index, item in enumerate(
                    _seq(
                        data.get("new_cache_pages", ()),
                        "execute run.new_cache_pages",
                    )
                )
            ),
            forward_operation_indices=_uints(
                data.get("forward_operation_indices", ()),
                "forward inputs.forward_operation_indices",
            ),
            request_pool_indices=_uints(
                data.get("request_pool_indices", ()),
                "forward inputs.request_pool_indices",
            ),
            seq_lens=_uints(
                data.get("seq_lens", ()), "forward inputs.seq_lens"
            ),
            query_lens=_uints(
                data.get("query_lens", ()), "forward inputs.query_lens"
            ),
            write_kv=tuple(
                _bool(value, "forward inputs.write_kv")
                for value in _seq(
                    data.get("write_kv", ()), "forward inputs.write_kv"
                )
            ),
            latent_params=tuple(
                LatentParams.from_mapping(
                    item, f"execute run.latent_params[{index}]"
                )
                for index, item in enumerate(
                    _seq(
                        data.get("latent_params", ()),
                        "execute run.latent_params",
                    )
                )
            ),
            decode_ranges=tuple(
                DecodeRange.from_mapping(
                    item, f"execute run.decode_ranges[{index}]"
                )
                for index, item in enumerate(
                    _seq(
                        data.get("decode_ranges", ()),
                        "execute run.decode_ranges",
                    )
                )
            ),
            buffer_allocations=tuple(
                BufferAllocation.from_mapping(
                    item, f"execute run.buffer_allocations[{index}]"
                )
                for index, item in enumerate(
                    _seq(
                        data.get("buffer_allocations", ()),
                        "execute run.buffer_allocations",
                    )
                )
            ),
            commands=commands,
            input_products=input_products,
            kv_inputs=tuple(
                transfer.KvTransfer.from_mapping(value)
                for value in _seq(data.get("kv_inputs", ()), "run.kv_inputs")
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        """Encode run identity and lifecycle commands.

        Also encodes physical lane descriptors.
        """
        return {
            "batch_id": self.batch_id,
            "run_id": self.run_id,
            "collective_seq": self.collective_seq,
            "operations": [value.to_mapping() for value in self.operations],
            "block_tables": [value.to_mapping() for value in self.block_tables],
            "new_cache_pages": [
                value.to_mapping() for value in self.new_cache_pages
            ],
            "forward_operation_indices": list(self.forward_operation_indices),
            "request_pool_indices": list(self.request_pool_indices),
            "seq_lens": list(self.seq_lens),
            "query_lens": list(self.query_lens),
            "write_kv": list(self.write_kv),
            "latent_params": [
                value.to_mapping() for value in self.latent_params
            ],
            "decode_ranges": [
                value.to_mapping() for value in self.decode_ranges
            ],
            "buffer_allocations": [
                value.to_mapping() for value in self.buffer_allocations
            ],
            "commands": [command_to_mapping(value) for value in self.commands],
            "input_products": [
                value.to_mapping() for value in self.input_products
            ],
            "kv_inputs": [value.to_mapping() for value in self.kv_inputs],
        }


@dataclass(frozen=True, slots=True)
class RegistrationAck:
    """Reports whether a lane publication is visible to consumers."""

    visible: bool = False

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "registration"
    ) -> RegistrationAck:
        """Parse registration visibility.

        Reports whether registration effects became visible to successor
        runs.
        """
        data = _map(value, where)
        return cls(
            visible=_bool(data.get("visible", False), f"{where}.visible")
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize registration visibility for IPC."""
        return {"visible": self.visible}


@dataclass(frozen=True, slots=True)
class TensorPublication:
    """A tensor identity and the physical metadata needed by its consumer."""

    product: tensor.TensorRef
    value: transfer.TransferValue

    def encoded_size_bound(self) -> int:
        """Bound the transfer metadata bytes.

        Includes every tensor location.
        """
        size = transfer._tensor_transfers_size((self.value.tensor,))
        if size > transfer.MAX_TRANSFER_HANDLE_BYTES:
            raise invalid_descriptor("transfer metadata exceeds its byte bound")
        return size

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "tensor publication"
    ) -> TensorPublication:
        """Parse the single generation owner and its typed transfer metadata."""
        data = _map(value, where)
        product = tensor.TensorRef.from_mapping(
            data.get("product"), f"{where}.product"
        )
        transfer_data = _map(data.get("value"), f"{where}.value")
        kind = _str(transfer_data.get("kind"), f"{where}.value.kind")
        payload = _map(transfer_data.get("value"), f"{where}.value.value")

        # Height/width default to zero for device products and are required
        # for every other transfer kind.
        height = _uint(
            payload.get("height", 0)
            if kind == "device_product"
            else payload.get("height"),
            f"{where}.value.height",
        )
        width = _uint(
            payload.get("width", 0)
            if kind == "device_product"
            else payload.get("width"),
            f"{where}.value.width",
        )
        tensor_value = transfer.TensorTransfer.from_mapping(
            payload.get("tensor"), f"{where}.value.tensor"
        )
        if kind == "encoder":
            typed: transfer.TransferValue = transfer.EncoderTransferValue(
                height=height,
                width=width,
                payload_kind=_str(
                    payload.get("payload_kind"), f"{where}.value.payload_kind"
                ),
                tensor=tensor_value,
            )
        elif kind == "device_product":
            typed = transfer.DeviceProductTransferValue(
                height=height,
                width=width,
                value_range=_str(
                    payload.get("value_range", ""), f"{where}.value.value_range"
                ),
                tensor=tensor_value,
            )
        elif kind == "latent":
            typed = transfer.LatentTransferValue(
                height=height,
                width=width,
                latent_units=_uint(
                    payload.get("latent_units"), f"{where}.value.latent_units"
                ),
                step=_uint(payload.get("step", 0), f"{where}.value.step"),
                tensor=tensor_value,
            )
        else:
            raise invalid_descriptor(f"{where}.value.kind is invalid")
        return cls(product=product, value=typed)

    def to_mapping(self) -> dict[str, object]:
        """Encode tensor identity once.

        Appears alongside the concrete transfer variant.
        """
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
