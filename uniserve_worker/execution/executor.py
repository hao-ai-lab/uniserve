"""Transactional lowering and postprocessing for the canonical execution batch."""

from __future__ import annotations

import hashlib
import logging
import math
import struct
import time
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from dataclasses import fields as dataclass_fields
from functools import partial
from importlib import import_module
from typing import Any, TypeAlias, cast, overload

import torch

from uniserve_worker.batch import (
    Batch,
    BatchPartition,
    CompletionRecord,
    CompletionReport,
    DevicePoint,
    Domain,
    DrawLayout,
    DType,
    EncodeMode,
    ExecutionCapability,
    FinishFlags,
    FixedPoint,
    GenMode,
    ImageParams,
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
    DeviceRole,
    ExecutionModel,
    LoweredStage,
    PositionLayout,
    RowKind,
    WorkerDeployment,
)
from uniserve_worker.nn.diffusion.cfg import Branch, build_flow_cfg_plan
from uniserve_worker.nn.diffusion.integrator import euler_step
from uniserve_worker.nn.diffusion.schedule import (
    x_pred_to_velocity,
)
from uniserve_worker.nn.mesh import BroadcastTransport
from uniserve_worker.nn.vision import get_flattened_position_ids_extrapolate
from uniserve_worker.runtime.completion_store import (
    CompletionArena,
    CompletionByteCapture,
    CompletionCapture,
    CompletionLease,
)
from uniserve_worker.runtime.cpu_tasks import BoundedCpuTaskPool, CpuTaskReservation
from uniserve_worker.runtime.execution_trace import (
    ExecutionPhase,
    ExecutionTrace,
    OperationTrace,
)
from uniserve_worker.runtime.host_staging import (
    TensorStager,
    TensorStagingSlot,
    canonical_device,
    copy_cpu_to_device,
    cpu_int_staging_buffer,
    fill_cpu_ints,
    is_pinned,
)
from uniserve_worker.runtime.image_utils import (
    quantize_image_hwc,
    uint8_image_to_png_base64_bytes,
)
from uniserve_worker.runtime.kv_store import KvEntry, KvSnapshot, KvStore, KvTxn
from uniserve_worker.runtime.latent_store import LatentRecord, LatentStore, LatentTxn
from uniserve_worker.runtime.mesh_store import MeshStore
from uniserve_worker.runtime.product_store import (
    DeviceProductContinuationBatch,
    DeviceProductRead,
    DeviceProductScalarBatch,
    DeviceProductTable,
    DeviceProductWrite,
    EncodedImageProduct,
    ImageRange,
    ImageTensorProduct,
    LatentFeatureProduct,
    LogitsProduct,
    ProductRecord,
    ProductStore,
    ProductTxn,
    ProductView,
    VisionFeatureProduct,
)
from uniserve_worker.runtime.replay import ReplayStore
from uniserve_worker.runtime.request_session import (
    RequestSession,
    ResolvedRuntimeState,
    SessionStore,
    StepTxn,
)
from uniserve_worker.runtime.rng import (
    DRAW_LAYOUT_TARGET,
    flow_noise_seed,
    normal_noise,
    sampling_key,
    sampling_uniform,
)
from uniserve_worker.runtime.transfer import (
    TRANSFER_DESCRIPTOR_PREFIX,
    Locator,
    TransferTicket,
    Transport,
    decode_transfer_descriptor,
    encode_transfer_descriptor,
)

from ._forward_plan import (
    ForwardBinding,
    ForwardPlan,
    GraphKey,
)
from ._inputs import (
    PreparedImage,
    patch_grid_shape,
    prepare_image,
    prepare_tensor_image,
)
from .model_runner import ModelRunner, RunObservation, RunPath

logger = logging.getLogger(__name__)

SAMPLING_COMPLETION_FIELDS = 4
TOKEN_CONTINUATION_BIT = 1 << 31
TOKEN_VALUE_MASK = TOKEN_CONTINUATION_BIT - 1


@dataclass(slots=True)
class _ForwardTask:
    operation: Operation
    session: RequestSession
    weights: WeightSet
    stage: LoweredStage
    route: RouteId
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
    def row_kind(self) -> RowKind:
        if isinstance(self.row, TokenRow):
            return RowKind.TOKEN
        if isinstance(self.row, FlowRow):
            return RowKind.FLOW
        if isinstance(self.row, EncodeRow):
            return RowKind.ENCODE
        return RowKind.DECODE


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
    continuation_product: DeviceProductWrite | None = None
    predicate: torch.Tensor | None = None
    tagged_predicate: bool = False
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
    prompt_logprobs: tuple[
        tuple[tuple[int, float, int], ...] | _CompletionTopLogprobs,
        ...,
    ] = ()
    device_finish: torch.Tensor | None = None
    device_continuation: torch.Tensor | None = None
    device_product_published: bool = False


class _CompletionTokenSpan:
    """One token vector backed exclusively by host-observation storage."""

    __slots__ = ("capture", "count", "_values")

    def __init__(self, capture: CompletionCapture) -> None:
        self.capture = capture
        self.count = int(capture.count)
        self._values: tuple[int, ...] | None = None

    def ready(self) -> bool:
        return self._values is not None or self.capture.ready()

    def finalize(self) -> tuple[int, ...]:
        if self._values is None:
            self._values = self.capture.values()
        return self._values


class _CompletionToken:
    """A protocol integer finalized only when the worker serializes its result."""

    __slots__ = ("span", "index")

    def __init__(self, span: _CompletionTokenSpan, index: int) -> None:
        self.span = span
        self.index = int(index)

    def ready(self) -> bool:
        return self.span.ready()

    def finalize(self) -> int:
        return self.span.finalize()[self.index]

    def __int__(self) -> int:
        return self.finalize()

    def __index__(self) -> int:
        return self.finalize()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _CompletionToken):
            return self.finalize() == other.finalize()
        if isinstance(other, int):
            return self.finalize() == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.finalize())


class _InvalidSamplingDistribution(RuntimeError):
    pass


class _PredicatedOperation(RuntimeError):
    pass


class _CompletionSampleSpan:
    """Selected tokens, row validity, predicates, and accepted counts."""

    __slots__ = ("capture", "count", "_values")

    def __init__(
        self,
        capture: CompletionCapture | None,
        count: int,
        values: tuple[int, ...] | None = None,
    ) -> None:
        self.capture = capture
        self.count = int(count)
        self._values = values

    def ready(self) -> bool:
        return self._values is not None or (self.capture is not None and self.capture.ready())

    def finalize(self) -> tuple[int, ...]:
        if self._values is None:
            if self.capture is None:
                raise RuntimeError("sampling completion metadata has no capture")
            values = self.capture.values()
            if len(values) != self.count * 4:
                raise RuntimeError("sampling completion metadata has an invalid extent")
            self._values = values
        return self._values

    def token(self, index: int) -> int:
        values = self.finalize()
        if not bool(values[self.count + index]):
            raise _PredicatedOperation("operation predicate selected no state")
        if not bool(values[index]):
            raise _InvalidSamplingDistribution("sampling policy produced an invalid distribution")
        return values[self.count * 2 + index]

    def accepted(self, index: int) -> int:
        values = self.finalize()
        if not bool(values[self.count + index]):
            raise _PredicatedOperation("operation predicate selected no state")
        if not bool(values[index]):
            raise _InvalidSamplingDistribution("sampling policy produced an invalid distribution")
        return values[self.count * 3 + index]


class _CompletionSampleToken(_CompletionToken):
    __slots__ = ("sample_span",)

    def __init__(self, span: _CompletionSampleSpan, index: int) -> None:
        self.sample_span = span
        self.span = cast(_CompletionTokenSpan, span)
        self.index = int(index)

    def ready(self) -> bool:
        return self.sample_span.ready()

    def finalize(self) -> int:
        return self.sample_span.token(self.index)


class _CompletionInteger:
    __slots__ = ("span", "index")

    def __init__(self, span: _CompletionSampleSpan, index: int) -> None:
        self.span = span
        self.index = int(index)

    def ready(self) -> bool:
        return self.span.ready()

    def finalize(self) -> int:
        return self.span.accepted(self.index)

    def __int__(self) -> int:
        return self.finalize()

    def __index__(self) -> int:
        return self.finalize()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _CompletionInteger):
            return self.finalize() == other.finalize()
        if isinstance(other, int):
            return self.finalize() == other
        return NotImplemented


class _CompletionDerivedInteger:
    __slots__ = ("source", "offset")

    def __init__(self, source: _CompletionInteger, offset: int) -> None:
        self.source = source
        self.offset = int(offset)

    def ready(self) -> bool:
        return self.source.ready()

    def finalize(self) -> int:
        return int(self.source) + self.offset

    def __int__(self) -> int:
        return self.finalize()

    def __index__(self) -> int:
        return self.finalize()


class _CompletionSpeculativePoint:
    __slots__ = ("accepted", "terminal_prefix")

    def __init__(self, accepted: _CompletionInteger, terminal_prefix: int | None) -> None:
        self.accepted = accepted
        self.terminal_prefix = terminal_prefix

    def ready(self) -> bool:
        return self.accepted.ready()

    def finalize(self) -> int:
        accepted = int(self.accepted)
        if self.terminal_prefix is not None and accepted >= self.terminal_prefix:
            return accepted
        return accepted + 1

    def __int__(self) -> int:
        return self.finalize()

    def __index__(self) -> int:
        return self.finalize()


class _CompletionSpeculativeTokens(Sequence[int]):
    __slots__ = ("draft", "accepted", "continuation", "terminal_prefix", "_value")

    def __init__(
        self,
        draft: tuple[int, ...],
        accepted: _CompletionInteger,
        continuation: int | _CompletionToken,
        terminal_prefix: int | None,
    ) -> None:
        self.draft = tuple(int(value) for value in draft)
        self.accepted = accepted
        self.continuation = continuation
        self.terminal_prefix = terminal_prefix
        self._value: tuple[int, ...] | None = None

    def ready(self) -> bool:
        continuation = self.continuation
        return self.accepted.ready() and (
            not isinstance(continuation, _CompletionToken) or continuation.ready()
        )

    def finalize(self) -> tuple[int, ...]:
        if self._value is None:
            accepted = int(self.accepted)
            if accepted < 0 or accepted > len(self.draft):
                raise RuntimeError("speculative acceptance count is outside the draft span")
            if self.terminal_prefix is not None and accepted >= self.terminal_prefix:
                self._value = self.draft[:accepted]
            else:
                self._value = (*self.draft[:accepted], int(self.continuation))
        return self._value

    def __len__(self) -> int:
        return len(self.finalize())

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[int, ...]: ...

    def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
        return self.finalize()[index]


class _CompletionLogprobBatch:
    """Packed query-ready logprob tensors shared by a sampling group."""

    __slots__ = (
        "capture",
        "rows",
        "counts",
        "requested_ids",
        "max_count",
        "max_requested",
        "_details",
    )

    def __init__(
        self,
        capture: CompletionCapture | None,
        rows: tuple[int, ...],
        counts: tuple[int, ...],
        requested_ids: tuple[tuple[int, ...], ...],
        max_count: int,
        max_requested: int,
        values: tuple[int, ...] | None = None,
    ) -> None:
        self.capture = capture
        self.rows = rows
        self.counts = counts
        self.requested_ids = requested_ids
        self.max_count = int(max_count)
        self.max_requested = int(max_requested)
        self._details: dict[int, tuple[float, tuple[tuple[int, float, int], ...]]] | None = None
        if values is not None:
            self._details = self._decode(values)

    def ready(self) -> bool:
        return self._details is not None or (self.capture is not None and self.capture.ready())

    @staticmethod
    def _float(value: int) -> float:
        return struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0]

    def finalize(self) -> dict[int, tuple[float, tuple[tuple[int, float, int], ...]]]:
        if self._details is not None:
            return self._details
        if self.capture is None:
            raise RuntimeError("logprob completion metadata has no capture")
        self._details = self._decode(self.capture.values())
        return self._details

    def _decode(
        self,
        values: tuple[int, ...],
    ) -> dict[int, tuple[float, tuple[tuple[int, float, int], ...]]]:
        row_count = len(self.rows)
        cursor = 0

        def vector(width: int) -> tuple[tuple[int, ...], ...]:
            nonlocal cursor
            total = row_count * width
            part = values[cursor : cursor + total]
            if len(part) != total:
                raise RuntimeError("logprob completion metadata is truncated")
            cursor += total
            return tuple(tuple(part[row * width : (row + 1) * width]) for row in range(row_count))

        selected_tokens = vector(1)
        selected_values = vector(1)
        selected_ranks = vector(1)
        top_indexes = vector(self.max_count)
        top_values = vector(self.max_count)
        top_ranks = vector(self.max_count)
        candidate_values = vector(self.max_requested)
        candidate_ranks = vector(self.max_requested)
        if cursor != len(values):
            raise RuntimeError("logprob completion metadata has trailing values")
        details: dict[int, tuple[float, tuple[tuple[int, float, int], ...]]] = {}
        for local, result_index in enumerate(self.rows):
            selected = selected_tokens[local][0]
            selected_value = self._float(selected_values[local][0])
            entries: list[tuple[int, float, int]] = [
                (selected, selected_value, selected_ranks[local][0])
            ]
            seen = {selected}
            for index in range(self.counts[local]):
                candidate = top_indexes[local][index]
                if candidate not in seen:
                    entries.append(
                        (
                            candidate,
                            self._float(top_values[local][index]),
                            top_ranks[local][index],
                        )
                    )
                    seen.add(candidate)
            for index, candidate in enumerate(self.requested_ids[local]):
                if candidate not in seen:
                    entries.append(
                        (
                            candidate,
                            self._float(candidate_values[local][index]),
                            candidate_ranks[local][index],
                        )
                    )
                    seen.add(candidate)
            details[result_index] = (selected_value, tuple(entries))
        return details


class _CompletionLogprobValue:
    __slots__ = ("batch", "index")

    def __init__(self, batch: _CompletionLogprobBatch, index: int) -> None:
        self.batch = batch
        self.index = int(index)

    def ready(self) -> bool:
        return self.batch.ready()

    def finalize(self) -> float:
        return self.batch.finalize()[self.index][0]

    def __float__(self) -> float:
        return self.finalize()


class _CompletionTopLogprobs:
    __slots__ = ("batch", "index")

    def __init__(self, batch: _CompletionLogprobBatch, index: int) -> None:
        self.batch = batch
        self.index = int(index)

    def ready(self) -> bool:
        return self.batch.ready()

    def finalize(self) -> tuple[tuple[int, float, int], ...]:
        return self.batch.finalize()[self.index][1]

    def max_entries(self) -> int:
        local = self.batch.rows.index(self.index)
        return 1 + int(self.batch.counts[local]) + len(self.batch.requested_ids[local])


class _CompletionLogprobPayload:
    __slots__ = ("logprob", "top_logprobs", "prompt_logprobs", "_value")

    def __init__(
        self,
        logprob: float | _CompletionLogprobValue | None,
        top_logprobs: tuple[tuple[int, float, int], ...] | _CompletionTopLogprobs | None,
        prompt_logprobs: tuple[
            tuple[tuple[int, float, int], ...] | _CompletionTopLogprobs,
            ...,
        ] = (),
    ) -> None:
        self.logprob = logprob
        self.top_logprobs = top_logprobs
        self.prompt_logprobs = prompt_logprobs
        self._value: bytes | None = None

    def ready(self) -> bool:
        if self._value is not None:
            return True
        return (
            (not isinstance(self.logprob, _CompletionLogprobValue) or self.logprob.ready())
            and (
                not isinstance(self.top_logprobs, _CompletionTopLogprobs)
                or self.top_logprobs.ready()
            )
            and all(
                not isinstance(position, _CompletionTopLogprobs) or position.ready()
                for position in self.prompt_logprobs
            )
        )

    def max_encoded_bytes(self) -> int:
        def entry_bound(
            entries: tuple[tuple[int, float, int], ...] | _CompletionTopLogprobs | None,
        ) -> int:
            if isinstance(entries, _CompletionTopLogprobs):
                return entries.max_entries()
            return len(entries or ())

        return (
            (5 if self.logprob is not None else 1)
            + 4
            + 12 * entry_bound(self.top_logprobs)
            + 4
            + sum(4 + 12 * entry_bound(position) for position in self.prompt_logprobs)
        )

    def finalize(self) -> bytes:
        if self._value is not None:
            return self._value
        if not self.ready():
            raise RuntimeError("logprob payload was observed before query-ready")
        logprob = None if self.logprob is None else float(self.logprob)
        top = (
            self.top_logprobs.finalize()
            if isinstance(self.top_logprobs, _CompletionTopLogprobs)
            else self.top_logprobs or ()
        )
        out = bytearray(b"\x00" if logprob is None else b"\x01" + struct.pack("<f", logprob))
        out += struct.pack("<I", len(top))
        for token_id, value, rank in top:
            out += struct.pack("<IfI", int(token_id), float(value), int(rank))
        out += struct.pack("<I", len(self.prompt_logprobs))
        for position in self.prompt_logprobs:
            entries = (
                position.finalize() if isinstance(position, _CompletionTopLogprobs) else position
            )
            out += struct.pack("<I", len(entries))
            for token_id, value, rank in entries:
                out += struct.pack("<IfI", int(token_id), float(value), int(rank))
        self._value = bytes(out)
        return self._value

    def __bytes__(self) -> bytes:
        return self.finalize()


class _CompletionTransferPayload:
    __slots__ = (
        "kind",
        "descriptor_value",
        "locators",
        "producer_plan_digest",
        "transport",
        "_value",
    )

    def __init__(
        self,
        kind: str,
        descriptor_value: dict[str, object],
        locators: tuple[Locator, ...],
        producer_plan_digest: str,
        transport: Transport,
    ) -> None:
        self.kind = kind
        self.descriptor_value = descriptor_value
        self.locators = locators
        self.producer_plan_digest = producer_plan_digest
        self.transport = transport
        self._value: bytes | None = None

    def ready(self) -> bool:
        return self._value is not None or all(
            self.transport.ready(locator) for locator in self.locators
        )

    def max_encoded_bytes(self) -> int:
        return len(
            encode_transfer_descriptor(
                self.kind,
                self.descriptor_value,
                self.producer_plan_digest,
            )
        )

    def finalize(self) -> bytes:
        if self._value is None:
            if not self.ready():
                raise RuntimeError("transport descriptor was observed before producer readiness")
            self._value = encode_transfer_descriptor(
                self.kind,
                self.descriptor_value,
                self.producer_plan_digest,
            )
        return self._value

    def __bytes__(self) -> bytes:
        return self.finalize()


class _CompletionImagePayload:
    """Pinned D2H image capture followed by bounded asynchronous PNG encoding."""

    __slots__ = (
        "capture",
        "reservation",
        "max_bytes",
        "publish",
        "discard",
        "_future",
        "_value",
        "_submission_error",
        "_discarded",
        "_published",
    )

    def __init__(
        self,
        capture: CompletionByteCapture,
        reservation: CpuTaskReservation,
        max_bytes: int,
        publish: Callable[[bytes], None],
        discard: Callable[[], None],
    ) -> None:
        self.capture = capture
        self.reservation = reservation
        self.max_bytes = int(max_bytes)
        self.publish = publish
        self.discard = discard
        self._future: Any | None = None
        self._value: bytes | None = None
        self._submission_error: Exception | None = None
        self._discarded = False
        self._published = False

    def ready(self) -> bool:
        if self._value is not None or self._submission_error is not None:
            return True
        if self._future is None:
            if not self.capture.ready():
                return False
            try:
                self._future = self.reservation.submit(
                    uint8_image_to_png_base64_bytes,
                    self.capture.tensor(),
                )
            except Exception as error:
                self._submission_error = error
                return True
        return bool(self._future.done())

    def max_encoded_bytes(self) -> int:
        return self.max_bytes

    def finalize(self) -> bytes:
        if self._value is not None:
            return self._value
        if not self.ready():
            raise RuntimeError("image payload was observed before CPU encoding was ready")
        try:
            if self._submission_error is not None:
                raise self._submission_error
            if self._future is None:
                raise RuntimeError("image encoding task lost its CPU future")
            value = self._future.result(timeout=0)
            if not isinstance(value, bytes) or not value:
                raise RuntimeError("image encoding task produced an invalid payload")
            if len(value) > self.max_bytes:
                raise RuntimeError("encoded image exceeds its registered product byte bound")
            if not self._published:
                self.publish(value)
                self._published = True
            self._value = value
        except Exception:
            self._discard_once()
            raise
        return self._value

    def _discard_once(self) -> None:
        if not self._discarded:
            self._discarded = True
            self.discard()

    def __bytes__(self) -> bytes:
        return self.finalize()

    def __del__(self) -> None:
        self.reservation.abandon()


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
    snapshot: KvSnapshot | None

    def ready(self) -> bool:
        return all(ticket.ready() for ticket in self.tickets)

    def tensors(self) -> tuple[torch.Tensor, ...]:
        if not self.ready():
            raise RuntimeError("prepared transfer input was observed before readiness")
        return tuple(ticket.result() for ticket in self.tickets)


@dataclass(frozen=True, slots=True)
class PreparedExecution:
    batch: Batch
    transfers: tuple[_PreparedTransferInput, ...]

    def ready(self) -> bool:
        return all(transfer.ready() for transfer in self.transfers)


_COMPLETION_FIELD_NAMES: tuple[str, ...] = tuple(
    completion_field.name for completion_field in dataclass_fields(CompletionRecord)
)


def _record_with_tokens(
    record: CompletionRecord,
    committed_tokens: tuple[int, ...],
    semantic_digest: object | None = None,
    timing_counters: TimingCounters | None = None,
) -> CompletionRecord:
    """A ``CompletionRecord`` copy with concrete tokens (and optional digest).

    A slot-for-slot copy of ``dataclasses.replace`` for the per-record
    finalization hot path: ``CompletionRecord`` declares no ``__post_init__``,
    so bypassing ``__init__`` produces an identical frozen record without the
    per-call field introspection.
    """

    copy = object.__new__(CompletionRecord)
    set_field = object.__setattr__
    for name in _COMPLETION_FIELD_NAMES:
        set_field(copy, name, getattr(record, name))
    set_field(copy, "committed_tokens", committed_tokens)
    if semantic_digest is not None:
        set_field(copy, "semantic_digest", semantic_digest)
    if timing_counters is not None:
        set_field(copy, "timing_counters", timing_counters)
    return copy


class _PendingDigest:
    """A semantic digest finalized from a query-ready completion generation.

    The digest includes committed tokens copied asynchronously into the pinned
    completion arena. Resolution reads that host storage only after every copy
    event reports ready, validates the physical slot generation, and releases
    the observed row. A device-parent successor may retain its predecessor's
    pending digest, so resolution follows the request lineage while unrelated
    completions remain independently dispatchable.
    """

    __slots__ = (
        "_record",
        "_parent",
        "_plan_digest",
        "_lease",
        "_row",
        "_generation",
        "_completion_timing",
        "_value",
        "_observed",
        "_invalid_sampling",
        "_predicated",
        "_predicated_parent",
        "_selected_point",
        "_selected_runtime",
        "_resolved_callback",
        "_completion_tasks",
        "_completion_error",
    )

    def __init__(
        self,
        record: CompletionRecord,
        parent: object,
        plan_digest: str,
        lease: CompletionLease,
        row: int,
        predicated_parent: Callable[[], tuple[VersionRef, ResolvedRuntimeState]],
        resolved_callback: Callable[[CompletionRecord, str, str], None] | None = None,
        completion_tasks: tuple[_CompletionImagePayload, ...] = (),
    ) -> None:
        self._record = record
        self._parent = parent
        self._plan_digest = plan_digest
        self._lease: CompletionLease | None = lease
        self._row = int(row)
        self._generation = int(record.completion_slot_generation)
        self._completion_timing: tuple[int, int] | None = None
        self._value: str | None = None
        self._observed = False
        self._invalid_sampling = False
        self._predicated = False
        self._predicated_parent: Callable[[], tuple[VersionRef, ResolvedRuntimeState]] | None = (
            predicated_parent
        )
        self._selected_point = record.selected_point
        self._selected_runtime: ResolvedRuntimeState | None = None
        self._resolved_callback = resolved_callback
        self._completion_tasks = completion_tasks
        self._completion_error = False

    def ready(self) -> bool:
        if self._value is not None:
            return True
        if isinstance(self._parent, _PendingDigest) and not self._parent.ready():
            return False
        if self._lease is None or not self._lease.ready():
            return False
        for task in self._completion_tasks:
            if not task.ready():
                return False
        return True

    def resolve(self) -> str:
        if self._value is None:
            if not self.ready():
                raise RuntimeError("completion digest was resolved before query-ready")
            parent = (
                self._parent.resolve() if isinstance(self._parent, _PendingDigest) else self._parent
            )
            try:
                for task in self._completion_tasks:
                    task.finalize()
            except Exception:
                self._completion_error = True
                self._value = _completion_error_record(self._record).compute_semantic_digest(
                    parent_semantic=cast(str, parent),
                    plan_digest=self._plan_digest,
                )
            else:
                try:
                    # The digest packs each committed token via ``__index__``, which
                    # finalizes a deferred token exactly as ``int(value)`` would, so
                    # the record is hashed in place without a concrete-token copy.
                    value = self._record.compute_semantic_digest(
                        parent_semantic=cast(str, parent),
                        plan_digest=self._plan_digest,
                    )
                    if self._resolved_callback is not None:
                        self._resolved_callback(self._record, value, cast(str, parent))
                    self._value = value
                except _PredicatedOperation:
                    self._predicated = True
                    self._value = cast(str, parent)
                    predicated_parent = self._predicated_parent
                    if predicated_parent is None:
                        raise RuntimeError("predicated completion lost its parent resolver")
                    selected, runtime = predicated_parent()
                    point = selected.point
                    if not isinstance(point, FixedPoint):
                        raise RuntimeError("predicated operation selected a non-fixed parent")
                    self._selected_point = int(point.point_index)
                    self._selected_runtime = runtime
                except _InvalidSamplingDistribution:
                    self._invalid_sampling = True
                    self._value = _invalid_sampling_record(self._record).compute_semantic_digest(
                        parent_semantic=cast(str, parent),
                        plan_digest=self._plan_digest,
                    )
            lease = self._lease
            if lease is None:
                raise RuntimeError("completion digest lost its arena lease")
            self._completion_timing = lease.observe(self._row, self._generation)
            self._observed = True
            self._lease = None
        return self._value

    def __str__(self) -> str:
        return self.resolve()

    def completion_timing(self) -> tuple[int, int]:
        self.resolve()
        return self._completion_timing or (0, 0)

    @property
    def invalid_sampling(self) -> bool:
        self.resolve()
        return self._invalid_sampling

    @property
    def predicated(self) -> bool:
        self.resolve()
        return self._predicated

    @property
    def completion_error(self) -> bool:
        self.resolve()
        return self._completion_error

    @property
    def selected_point(self) -> int:
        self.resolve()
        return int(self._selected_point)

    @property
    def selected_runtime(self) -> ResolvedRuntimeState:
        self.resolve()
        if self._selected_runtime is None:
            raise RuntimeError("predicated operation lost its selected runtime state")
        return self._selected_runtime

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _PendingDigest):
            return self.resolve() == other.resolve()
        if isinstance(other, str):
            return self.resolve() == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.resolve())

    def __deepcopy__(self, memo: dict[int, object]) -> _PendingDigest:
        # A committed session snapshot shares ownership of the exact pinned
        # completion generation and its lineage digest.
        memo[id(self)] = self
        return self

    def __del__(self) -> None:
        lease = self._lease
        if lease is not None and not self._observed:
            lease.discard(self._row, self._generation)


def _record_ready(record: CompletionRecord) -> bool:
    """Whether a completion's deferred token copy and digest chain have landed."""

    digest = record.semantic_digest
    if isinstance(digest, _PendingDigest):
        return digest.ready()
    for value in cast(tuple[object, ...], record.committed_tokens):
        if isinstance(value, _CompletionToken) and not value.ready():
            return False
    return True


def _finalized_record(record: CompletionRecord) -> CompletionRecord:
    digest = record.semantic_digest
    if isinstance(digest, _PendingDigest):
        resolved = digest.resolve()
        copy_us, host_us = digest.completion_timing()
        timing = replace(record.timing_counters, copy_us=copy_us, host_us=host_us)
        if digest.completion_error:
            return replace(
                _completion_error_record(record),
                semantic_digest=resolved,
                timing_counters=timing,
            )
        if digest.invalid_sampling:
            return replace(
                _invalid_sampling_record(record),
                semantic_digest=resolved,
                timing_counters=timing,
            )
        if digest.predicated:
            return replace(
                _predicated_record(record, digest.selected_point, digest.selected_runtime),
                semantic_digest=resolved,
                timing_counters=timing,
            )
    else:
        resolved = digest
        timing = record.timing_counters
    tokens = tuple(int(value) for value in record.committed_tokens)
    lengths = record.logical_lengths
    span = record.token_span
    return replace(
        record,
        selected_point=int(record.selected_point),
        logical_lengths=LogicalLengths(
            token_len=int(lengths.token_len),
            kv_visible_len=int(lengths.kv_visible_len),
            latent_len=int(lengths.latent_len),
            kv_reserved_len=int(lengths.kv_reserved_len),
            kv_initialized_len=int(lengths.kv_initialized_len),
            kv_committed_len=int(lengths.kv_committed_len),
            kv_published_len=int(lengths.kv_published_len),
        ),
        token_span=TokenSpan(base=int(span.base), len=int(span.len)),
        committed_tokens=tokens,
        semantic_digest=resolved,
        timing_counters=timing,
    )


def _invalid_sampling_record(record: CompletionRecord) -> CompletionRecord:
    return replace(
        record,
        status=OpStatus.ERROR,
        selected_point=max(0, int(record.selected_point) - 1),
        token_span=replace(record.token_span, len=0),
        committed_tokens=(),
        finish_flags=FinishFlags(),
        product_generations=(),
        error_code=ProtocolErrorCode.INVALID_OPERATION,
    )


def _completion_error_record(record: CompletionRecord) -> CompletionRecord:
    return replace(
        record,
        status=OpStatus.ERROR,
        token_span=replace(record.token_span, len=0),
        committed_tokens=(),
        finish_flags=FinishFlags(),
        product_generations=(),
        error_code=ProtocolErrorCode.COMPUTE_ERROR,
    )


def _predicated_record(
    record: CompletionRecord,
    selected_point: int,
    runtime: ResolvedRuntimeState,
) -> CompletionRecord:
    return replace(
        record,
        status=OpStatus.PREDICATED,
        selected_point=int(selected_point),
        logical_lengths=LogicalLengths(
            token_len=runtime.logical_position,
            kv_visible_len=runtime.kv_visible_len,
            latent_len=0,
            kv_reserved_len=runtime.kv_reserved_len,
            kv_initialized_len=runtime.kv_initialized_len,
            kv_committed_len=runtime.kv_committed_len,
            kv_published_len=runtime.kv_published_len,
        ),
        token_span=replace(record.token_span, len=0),
        committed_tokens=(),
        finish_flags=FinishFlags(),
        product_generations=(),
        error_code=None,
    )


def completion_report_ready(report: CompletionReport) -> bool:
    """True once every completion's deferred token/digest/artifact can be read
    without a stall."""

    for record in report.completions:
        if not _record_ready(record):
            return False
    for product in report.products:
        if not _completion_payload_ready(product.payload):
            return False
    return True


def partition_completion_ready(partition: PartitionCompletion) -> bool:
    for record in partition.completions:
        if not _record_ready(record):
            return False
    for product in partition.products:
        if not _completion_payload_ready(product.payload):
            return False
    return True


def _completion_payload_ready(payload: object) -> bool:
    return (
        not isinstance(
            payload,
            (_CompletionImagePayload, _CompletionLogprobPayload, _CompletionTransferPayload),
        )
        or payload.ready()
    )


def finalize_completion_report(report: CompletionReport) -> CompletionReport:
    """Materialize every ready completion's committed tokens and semantic digest.

    Records whose deferred copy has not yet landed are left pending; the caller
    (execute-end opportunistic pass, or replay) only observes the ready ones. At
    response-serialization time the server has already gated on
    :func:`completion_report_ready`, so everything resolves here.
    """

    changed = False
    partitions: list[PartitionCompletion] = []
    for partition in report.partitions:
        completions = tuple(
            _finalized_record(record) if _record_ready(record) else record
            for record in partition.completions
        )
        nonpublishing_ops = {
            int(record.op_id) for record in completions if record.status is not OpStatus.OK
        }
        retained_products = tuple(
            product
            for product in partition.products
            if int(product.product.producer_op_id) not in nonpublishing_ops
        )
        products = tuple(
            replace(product, payload=product.payload.finalize())
            if isinstance(
                product.payload,
                (
                    _CompletionImagePayload,
                    _CompletionLogprobPayload,
                    _CompletionTransferPayload,
                ),
            )
            and product.payload.ready()
            else product
            for product in retained_products
        )
        for product in products:
            if (
                isinstance(product.payload, bytes)
                and not product.payload.startswith(TRANSFER_DESCRIPTOR_PREFIX)
                and len(product.payload) > int(product.product.max_bytes)
            ):
                raise invalid_descriptor(
                    "completion product exceeds its registered product byte bound"
                )
        if (
            not all(new is old for new, old in zip(completions, partition.completions, strict=True))
            or len(products) != len(partition.products)
            or not all(new is old for new, old in zip(products, partition.products))
        ):
            changed = True
            partition = replace(partition, completions=completions, products=products)
        partitions.append(partition)
    return replace(report, partitions=tuple(partitions)) if changed else report


_ExecutorTask: TypeAlias = _ForwardTask | _SampleTask
_TaskResult: TypeAlias = tuple[Any, ...]
_Driver: TypeAlias = Generator[tuple[_ExecutorTask, ...], _TaskResult, "_Outcome"]
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


@dataclass(slots=True)
class _RowIdentity:
    next_value: int = 0

    def acquire(self) -> int:
        value = self.next_value
        self.next_value += 1
        return value


@dataclass(frozen=True, slots=True)
class _PartitionLayout:
    operations: tuple[Operation, ...]
    sessions: tuple[RequestSession, ...]
    kv_entries: tuple[KvEntry, ...]
    weights: tuple[WeightSet, ...]
    identities: tuple[_OperationIdentity, ...]

    def __post_init__(self) -> None:
        width = len(self.operations)
        if not all(
            len(values) == width
            for values in (
                self.sessions,
                self.kv_entries,
                self.weights,
                self.identities,
            )
        ):
            raise RuntimeError("partition layout columns are not aligned")


@dataclass(slots=True)
class _ExecutionScope:
    partition: BatchPartition
    started_ns: int
    transaction: StepTxn
    completion: CompletionLease
    kv: KvTxn
    latents: LatentTxn
    products: ProductTxn
    product_view: ProductView
    layout: _PartitionLayout | None = None
    prepared_transfers: dict[ProductRef, _PreparedTransferInput] = field(default_factory=dict)
    stage_publications: dict[_OperationIdentity, tuple[Locator, ...]] = field(default_factory=dict)
    published: list[Locator] = field(default_factory=list)
    observations: list[RunObservation] = field(default_factory=list)
    component_us: dict[str, int] = field(default_factory=dict)
    device_reads: list[DeviceProductRead] = field(default_factory=list)
    device_writes: list[DeviceProductWrite] = field(default_factory=list)
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
    continuation_writes: dict[_OperationIdentity, DeviceProductWrite] = field(default_factory=dict)
    predicate_reads: dict[_OperationIdentity, DeviceProductRead] = field(default_factory=dict)
    selected_point_reads: dict[_OperationIdentity, DeviceProductRead] = field(default_factory=dict)
    predicate_values: dict[_OperationIdentity, tuple[torch.Tensor, bool]] = field(
        default_factory=dict
    )
    sampling_states: dict[_OperationIdentity, SamplingState] = field(default_factory=dict)
    device_continuation: DeviceProductContinuationBatch | None = None
    row_identity: _RowIdentity = field(default_factory=_RowIdentity)
    registration_visible: bool = False
    cpu_tasks: dict[_OperationIdentity, CpuTaskReservation] = field(default_factory=dict)

    def row_id(self) -> int:
        return self.row_identity.acquire()


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


class ModelExecutor:
    """Own one typed operation step from validation through atomic publication."""

    def __init__(
        self,
        *,
        model: ExecutionModel | None,
        deployment: WorkerDeployment | None,
        runner: ModelRunner | None,
        attention: AttentionSelection | None,
        sessions: SessionStore,
        kv: KvStore,
        latents: LatentStore,
        products: ProductStore,
        replay: ReplayStore,
        weights: WeightSet | None,
        mesh: MeshStore | None,
        transport: Transport | None,
        tokenizer: Any | None,
        architecture_digest: str | None,
        weight_digest: str | None,
        allowed_work_variants: frozenset[WorkVariant],
        trace: ExecutionTrace,
        pipeline_depth: int = 1,
        defer_sampling: bool = False,
        completion_payload_bytes: int,
        cpu_task_capacity: int,
        pinned_staging_capacity: int,
    ) -> None:
        if not allowed_work_variants:
            raise ValueError("executor must accept at least one work variant")
        if (model is None) != (deployment is None):
            raise ValueError("model and worker deployment must be present together")
        if runner is not None and (model is None or weights is None):
            raise ValueError("model execution requires a model and base weights")
        if (runner is None) != (attention is None):
            raise ValueError("model runner and attention selection must be provisioned together")
        if model is not None:
            if architecture_digest is None or len(architecture_digest) != 64:
                raise capability_mismatch("executor model identity is invalid")
            if weight_digest is None or weights is None or weights.digest != weight_digest:
                raise capability_mismatch(
                    "executor base-weight identity does not match its weight set"
                )
            unsupported = allowed_work_variants - model.supported_work
            system_only = frozenset({WorkVariant.MATERIALIZE})
            if unsupported - system_only:
                raise capability_mismatch(
                    "executor work set exceeds the model implementation: "
                    f"{sorted(value.value for value in unsupported - system_only)!r}"
                )
        self.model = model
        self.deployment = deployment
        self.runner = runner
        self.attention = attention
        self.sessions = sessions
        self.kv = kv
        self.latents = latents
        self.products = products
        self.replay = replay
        self.weights = weights
        self.mesh = mesh
        self.transport = transport
        self.tokenizer = tokenizer
        self.architecture_digest = architecture_digest
        self.weight_digest = weight_digest
        self.allowed_work_variants = allowed_work_variants
        self.trace = trace
        self.defer_sampling = bool(defer_sampling)
        self._device = (
            torch.device("cpu") if deployment is None else canonical_device(deployment.device)
        )
        self._generation_device = (
            self._device
            if deployment is None or deployment.generation_device is None
            else canonical_device(deployment.generation_device)
        )
        max_operations = 1024 if deployment is None else int(deployment.max_batch_operations)
        if int(completion_payload_bytes) < 1:
            raise ValueError("completion payload capacity must be positive")
        completion_words = (
            SAMPLING_COMPLETION_FIELDS * max_operations + (int(completion_payload_bytes) + 3) // 4
        )
        completion_devices: list[str] = []
        if deployment is not None:
            completion_devices.append(deployment.device)
            if deployment.generation_device is not None:
                completion_devices.append(deployment.generation_device)
        self._completions = CompletionArena(
            depth=pipeline_depth * max_operations,
            token_capacity=completion_words,
            total_token_capacity=pipeline_depth * completion_words,
            devices=tuple(completion_devices),
            event_pool=products.device_events,
        )
        self._tensor_stager = TensorStager(
            capacity=pipeline_depth * max_operations,
            byte_capacity=int(pinned_staging_capacity),
        )
        self._cpu_tasks = BoundedCpuTaskPool(
            capacity=int(cpu_task_capacity),
            workers=min(4, int(cpu_task_capacity)),
        )
        self._collective_history: OrderedDict[int, str] = OrderedDict()
        self._transport_publications: dict[_OperationIdentity, tuple[Locator, ...]] = {}

    def close(self) -> None:
        self._cpu_tasks.close()

    def prepare(self, batch: Batch) -> PreparedExecution | None:
        """Submit every declared cross-stage read without waiting for it."""

        entries = tuple(
            payload
            for payload in batch.input_products
            if payload.payload.startswith(TRANSFER_DESCRIPTOR_PREFIX)
        )
        if not entries:
            return None
        transport = self.transport
        if transport is None:
            raise capability_mismatch("cross-stage input requires a configured transport")
        transfers: list[_PreparedTransferInput] = []
        for entry in entries:
            kind, value, producer_plan_digest = decode_transfer_descriptor(entry.payload)
            locators: tuple[Locator, ...]
            payload_kind: ProductKind | None = None
            height: int | None = None
            width: int | None = None
            snapshot: KvSnapshot | None = None
            if kind == "tensor":
                if set(value) != {
                    "height",
                    "locator",
                    "payload_kind",
                    "width",
                }:
                    raise invalid_descriptor("tensor transfer entry has an invalid shape")
                raw_locator = value["locator"]
                if not isinstance(raw_locator, dict):
                    raise invalid_descriptor("tensor transfer entry locator is invalid")
                main = Locator.from_wire(raw_locator)
                locators = (main,)
                raw_payload_kind = value["payload_kind"]
                raw_height = value["height"]
                raw_width = value["width"]
                if (
                    not isinstance(raw_payload_kind, str)
                    or raw_payload_kind
                    not in {ProductKind.VISION_FEATURE.value, ProductKind.LATENT_FEATURE.value}
                    or not isinstance(raw_height, int)
                    or isinstance(raw_height, bool)
                    or not isinstance(raw_width, int)
                    or isinstance(raw_width, bool)
                    or raw_height < 1
                    or raw_width < 1
                    or main.nbytes > entry.product.max_bytes
                    or math.prod(main.shape) > entry.product.shape_bound.max_elements
                ):
                    raise invalid_descriptor("tensor transfer metadata exceeds its product bounds")
                payload_kind = ProductKind(raw_payload_kind)
                height = raw_height
                width = raw_width
            else:
                if set(value) != {"snapshot"}:
                    raise invalid_descriptor("KV transfer entry has an invalid shape")
                snapshot = KvSnapshot.from_wire(value["snapshot"])
                if entry.product.kind is not ProductKind.KV:
                    raise invalid_descriptor("KV transfer entry names a non-KV product")
                locators = tuple(Locator.from_wire_json(raw) for raw in snapshot.locators)
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
                    snapshot=snapshot,
                )
            )
        return PreparedExecution(batch=batch, transfers=tuple(transfers))

    def execute_prepared(self, prepared: PreparedExecution) -> CompletionReport:
        if not prepared.ready():
            raise RuntimeError("prepared execution was observed before transfer readiness")
        return self.execute(prepared.batch, prepared=prepared.transfers)

    def complete_startup(self) -> None:
        """Retire pre-admission collective identities before serving traffic."""

        if self.sessions.session_ids():
            raise RuntimeError("startup completed with resident request sessions")
        self._collective_history.clear()

    def execute(
        self,
        batch: Batch,
        *,
        prepared: tuple[_PreparedTransferInput, ...] = (),
    ) -> CompletionReport:
        """Execute one canonical batch with replay-before-mutation semantics."""

        return self._execute(batch, prepared=prepared, propagate_errors=False)

    def execute_startup(self, batch: Batch) -> CompletionReport:
        """Execute pre-admission work with transactional error propagation."""

        return self._execute(batch, prepared=(), propagate_errors=True)

    def _execute(
        self,
        batch: Batch,
        *,
        prepared: tuple[_PreparedTransferInput, ...],
        propagate_errors: bool,
    ) -> CompletionReport:
        """Shared transactional execution for startup and admitted traffic."""

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
        for update in self.sessions.apply_controls(batch.controls):
            if update.rewind:
                self.kv.rewind(update.session_id, update.visible_len)
            self.kv.commit(update.session_id, update.committed_len)
        if not batch.operations:
            self._apply_release_controls(batch)
            return CompletionReport(
                step_id=batch.step_id,
                partitions=(),
            )
        try:
            replayed = self.replay.lookup(batch.partitions)
        except BaseException as error:
            self.trace.emit(ExecutionPhase.REPLAY, operations, error=error)
            raise
        if replayed is not None:
            self._apply_release_controls(batch)
            report = finalize_completion_report(
                replace(
                    replayed,
                    step_id=batch.step_id,
                    partitions=tuple(
                        replace(
                            partition,
                            worker_exec_us=(time.perf_counter_ns() - started) // 1000,
                            forward_stats=WorkerForwardStats(),
                        )
                        for partition in replayed.partitions
                    ),
                )
            )
            self.trace.emit(
                ExecutionPhase.REPLAY,
                operations,
                duration_us=(time.perf_counter_ns() - started) // 1000,
            )
            return report

        row_identity = _RowIdentity()
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
                            row_identity,
                            prepared,
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
                            self._rollback_partition(scope, classified)
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
                outcomes, execution_errors = self._execute_partition_group(tuple(scopes))
            except BaseException as error:
                classified = self._classify_partition_failure(
                    partitions[0],
                    error,
                    phase="partition execution",
                )
                for scope in scopes:
                    self._rollback_partition(scope, classified)
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
                    self._rollback_partition(scope, classified)
                raise classified

            for scope in scopes:
                partition_error = execution_errors.get(scope.partition.partition_id)
                if partition_error is not None:
                    classified = self._classify_partition_failure(
                        scope.partition,
                        partition_error,
                        phase="partition execution",
                    )
                    self._rollback_partition(scope, classified)
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
                    classified = self._classify_partition_failure(
                        scope.partition,
                        error,
                        phase="partition commit",
                    )
                    self._rollback_partition(scope, classified)
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
        report = finalize_completion_report(report)
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
        row_identity: _RowIdentity,
        prepared: tuple[_PreparedTransferInput, ...],
    ) -> _ExecutionScope:
        operations = partition.operations
        traced = _trace_envelopes(operations)
        started = time.perf_counter_ns()
        completion = self._completions.reserve(
            len(operations),
            token_capacity=self._partition_completion_words(operations),
            devices=self._completion_devices(operations),
        )
        try:
            transaction = self.sessions.begin_step(
                batch.step_id,
                operations,
                (self.kv, self.latents, self.products),
            )
        except BaseException as error:
            completion.abandon()
            self.trace.emit(
                ExecutionPhase.TRANSACTION_OPEN,
                traced,
                duration_us=(time.perf_counter_ns() - started) // 1000,
                error=error,
            )
            raise
        scope = _ExecutionScope(
            partition=partition,
            started_ns=started,
            transaction=transaction,
            completion=completion,
            kv=cast(KvTxn, transaction.store_transaction(self.kv)),
            latents=cast(LatentTxn, transaction.store_transaction(self.latents)),
            products=cast(ProductTxn, transaction.store_transaction(self.products)),
            product_view=cast(ProductTxn, transaction.store_transaction(self.products)).view(),
            prepared_transfers={
                transfer.product: transfer
                for transfer in prepared
                if transfer.product
                in {reference for operation in operations for reference in operation.inputs}
            },
            row_identity=row_identity,
        )
        self.trace.emit(
            ExecutionPhase.TRANSACTION_OPEN,
            traced,
            duration_us=(time.perf_counter_ns() - started) // 1000,
        )
        request_keys = {operation.request_key for operation in operations}
        admissions = tuple(
            admission for admission in batch.admissions if admission.request_key in request_keys
        )
        declared_inputs = {reference for operation in operations for reference in operation.inputs}
        input_products = tuple(
            payload for payload in batch.input_products if payload.product in declared_inputs
        )
        try:
            for admission in admissions:
                self.sessions.admit(admission)
            self.sessions.validate_operations(
                operations,
                admissions,
                partition.request_pool_indices,
            )
            self._reserve_cpu_tasks(operations, scope)
            for admission in admissions:
                self.kv.admit(admission)
            self._reserve_kv_pages(partition, scope)
            aligned_sessions = transaction.aligned_sessions()
            scope.layout = _PartitionLayout(
                operations=operations,
                sessions=aligned_sessions,
                kv_entries=scope.kv.entries(
                    tuple(operation.request_key.session_id for operation in operations)
                ),
                weights=tuple(self._weights() for _ in aligned_sessions),
                identities=tuple(_operation_identity(operation) for operation in operations),
            )
            self._reserve_outputs(operations, scope)
            self._stage_input_products(input_products, scope)
            self._consume_predicates(operations, scope)
            scope.registration_visible = True
            return scope
        except BaseException:
            self._rollback_partition(scope)
            raise

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
        drivers: list[tuple[int, int, _Driver, _ExecutionScope]] = []
        for scope_index, scope in enumerate(scopes):
            for operation_index, operation in enumerate(scope.partition.operations):
                drivers.append(
                    (scope_index, operation_index, self._driver(operation, scope), scope)
                )
        flat_outcomes, errors = self._drive_partitioned(tuple(drivers))
        grouped: list[list[_Outcome | None]] = [
            [None] * len(scope.partition.operations) for scope in scopes
        ]
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
        if scope.device_continuation is not None:
            self.products.device_products.validate_continuation(scope.device_continuation)
        continuation_writes = (
            frozenset(id(write) for write in scope.device_continuation.writes)
            if scope.device_continuation is not None
            else frozenset()
        )
        self.products.device_products.validate_writes(
            tuple(write for write in scope.device_writes if id(write) not in continuation_writes)
        )
        scope.completion.seal()
        records: list[CompletionRecord] = []
        committed: dict[int, VersionRef] = {}
        report_products: list[ProductPayload] = []
        pending_by_session: dict[int, _PendingDigest] = {}
        resolved_runtime: dict[int, ResolvedRuntimeState] = {}
        layout = scope.layout
        if layout is None or layout.operations != operations:
            raise RuntimeError("partition commit lost its aligned transaction layout")
        for row, (operation, session, entry, outcome) in enumerate(
            zip(
                operations,
                layout.sessions,
                layout.kv_entries,
                outcomes,
                strict=True,
            )
        ):
            self._validate_completion_products(operation, outcome.products)
            if self.deployment is None or int(self.deployment.tp_rank) == 0:
                report_products.extend(outcome.products)
            parent_semantic = _parent_semantic(operation, session)
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
                completion_tasks=outcome.completion_tasks,
            )
            records.append(replace(placeholder, semantic_digest=cast(str, pending)))
            if operation.advances_state:
                committed[operation.request_key.session_id] = VersionRef(
                    request_key=operation.request_key,
                    producer_op_id=operation.op_id,
                    point=FixedPoint(cast(int, outcome.selected_point), cast(str, pending)),
                )
                pending_by_session[operation.request_key.session_id] = pending
                extents = entry.extents()
                resolved_runtime[operation.request_key.session_id] = ResolvedRuntimeState(
                    logical_position=session.logical_position,
                    rng_counter=session.rng_counter,
                    latent_product=session.latent_product,
                    flow_step=session.flow_step,
                    kv_reserved_len=extents.reserved,
                    kv_initialized_len=extents.initialized,
                    kv_visible_len=outcome.logical_lengths.kv_visible_len,
                    kv_committed_len=extents.committed,
                    kv_published_len=extents.published,
                )
            else:
                committed[operation.request_key.session_id] = operation.parent
        partition_report = PartitionCompletion(
            partition_id=partition.partition_id,
            completions=tuple(records),
            products=tuple(report_products),
            registration=RegistrationAck(visible=True),
            worker_exec_us=(time.perf_counter_ns() - scope.started_ns) // 1000,
            forward_stats=_forward_stats(scope.observations, scope.component_us),
        )
        report = CompletionReport(step_id=step_id, partitions=(partition_report,))
        self.replay.commit_atomic(
            operations,
            report,
            lambda publish: scope.transaction.commit(
                committed,
                resolved_runtime,
                publish=publish,
            ),
        )
        for identity, locators in scope.stage_publications.items():
            existing = self._transport_publications.get(identity)
            if existing is not None and existing != locators:
                raise RuntimeError("committed transport publication identity was reused")
            self._transport_publications[identity] = locators
        for session_id, pending in pending_by_session.items():
            if pending.ready():
                self.sessions.get(session_id).resolved_digest = pending.resolve()
        return partition_report

    def _rollback_partition(
        self,
        scope: _ExecutionScope,
        error: BaseException | None = None,
    ) -> None:
        self._finish_device_reads(scope)
        for reservation in scope.cpu_tasks.values():
            reservation.abandon()
        scope.transaction.rollback()
        scope.completion.abandon()
        self.products.device_products.abandon_writes(tuple(scope.device_writes))
        self._release_locators(scope.published)
        self.trace.emit(
            ExecutionPhase.ROLLBACK,
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
        self.replay.commit_atomic(
            partition.operations,
            CompletionReport(step_id=step_id, partitions=(report,)),
            lambda publish: publish(),
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
        self.replay.commit_atomic(
            scope.partition.operations,
            CompletionReport(step_id=step_id, partitions=(report,)),
            lambda publish: publish(),
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
            session = self.sessions.peek(operation.request_key.session_id)
            point = operation.parent.point
            selected_point = (
                point.point_index
                if isinstance(point, FixedPoint)
                else 0
                if session is None
                else session.version
            )
            parent_semantic = (
                point.semantic_digest
                if isinstance(point, FixedPoint)
                else "0" * 64
                if session is None
                else session.resolved_digest
            )
            lengths = LogicalLengths()
            if session is not None:
                entry = self.kv.get(operation.request_key.session_id)
                extents = entry.extents()
                lengths = LogicalLengths(
                    token_len=session.logical_position,
                    kv_visible_len=extents.visible,
                    latent_len=session.flow_step,
                    kv_reserved_len=extents.reserved,
                    kv_initialized_len=extents.initialized,
                    kv_committed_len=extents.committed,
                    kv_published_len=extents.published,
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
            records.append(
                replace(
                    placeholder,
                    semantic_digest=placeholder.compute_semantic_digest(
                        parent_semantic,
                        operation.plan_digest,
                    ),
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
    ) -> tuple[VersionRef, ResolvedRuntimeState]:
        selected, runtime, latest = self.sessions.finalize_predicated(
            operation.request_key.session_id,
            operation.op_id,
            operation.parent,
        )
        if latest:
            self.kv.select(operation.request_key.session_id, runtime.kv_visible_len)
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
        entry = self.kv.get(operation.request_key.session_id)
        extents = entry.extents()
        selected_kv = selection.base_kv_visible + selected_point
        if extents.initialized != selection.initialized_kv or selected_kv > extents.initialized:
            raise RuntimeError("speculative KV selection is outside initialized state")
        prefixes: list[tuple[VersionRef, ResolvedRuntimeState]] = []
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
            runtime = ResolvedRuntimeState(
                logical_position=selection.base_logical_position + point_index,
                rng_counter=selection.base_rng_counter + point_index,
                latent_product=self.sessions.get(operation.request_key.session_id).latent_product,
                flow_step=self.sessions.get(operation.request_key.session_id).flow_step,
                kv_reserved_len=extents.reserved,
                kv_initialized_len=extents.initialized,
                kv_visible_len=selection.base_kv_visible + point_index,
                kv_committed_len=extents.committed,
                kv_published_len=extents.published,
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
        self.kv.select(operation.request_key.session_id, selected_kv)
        self.sessions.finalize_prefixes(
            operation.request_key.session_id,
            operation.op_id,
            prefixes,
        )

    def _validate_batch_identity(self, batch: Batch) -> None:
        if (
            self.deployment is not None
            and len(batch.operations) > self.deployment.max_batch_operations
        ):
            raise invalid_descriptor("execution batch exceeds the deployment operation limit")
        for operation in batch.operations:
            variant = operation.work.variant
            if variant not in self.allowed_work_variants:
                raise unsupported_operation(variant.value, operation.request_key.session_id)
        groups: dict[int, list[BatchPartition]] = defaultdict(list)
        for partition in batch.partitions:
            groups[partition.submission_group].append(partition)
        for partitions in groups.values():
            first = partitions[0]
            if first.execution is not ExecutionCapability.TENSORIZED_MIXED:
                continue
            if self.model is None:
                raise invalid_descriptor("tensorized mixed submission has no model route")
            primary_stages = tuple(
                self._primary_stage(operation.work.variant)
                for partition in partitions
                for operation in partition.operations
            )
            model_routes = {stage.route for stage in primary_stages}
            if len(model_routes) != 1:
                raise invalid_descriptor(
                    "tensorized mixed submission spans distinct model runner routes"
                )
            row_kinds = frozenset(stage.row for stage in primary_stages)
            if not self.model.allows_mixed(next(iter(model_routes)), row_kinds):
                raise invalid_descriptor(
                    "tensorized mixed submission exceeds worker mixed-execution capabilities"
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
        if deployment is None:
            return ()
        selected: list[str] = []
        for operation in operations:
            device = (
                deployment.generation_device
                if operation.domain is Domain.GEN and deployment.generation_device is not None
                else deployment.device
            )
            if device not in selected:
                selected.append(device)
        return tuple(selected)

    def _reserve_outputs(
        self,
        operations: tuple[Operation, ...],
        scope: _ExecutionScope,
    ) -> None:
        """Bind each operation's declared output products to worker-local handles.

        The reservation lives inside the step's product transaction, so a rejected
        registration unwinds every bound handle and leaves no product behind.
        """

        self._reserve_device_continuation(operations, scope)
        continuation_outputs = (
            frozenset(write.reference for write in scope.device_continuation.writes)
            if scope.device_continuation is not None
            else frozenset()
        )
        scalar_groups: dict[
            tuple[torch.device, ProductKind, DType, ShapeBound],
            list[tuple[ProductRef, str, torch.device | str]],
        ] = {}
        general_bindings: list[tuple[ProductRef, str, torch.device | str]] = []
        for operation in operations:
            device = self._operation_device(operation)
            for output in operation.outputs:
                if _requires_device_product_binding(output) and output not in continuation_outputs:
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
        bound_groups = self.products.device_products.bind_output_groups(groups)
        scope.device_writes.extend(write for binding in bound_groups for write in binding.writes)
        operation_identities = {_operation_identity(operation) for operation in operations}
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
                and int(write.reference.output_index) == 4
            ):
                scope.continuation_writes[operation_identity] = write
            else:
                scope.operation_writes.setdefault(operation_identity, write)

    def _reserve_device_continuation(
        self,
        operations: tuple[Operation, ...],
        scope: _ExecutionScope,
    ) -> bool:
        if not operations:
            return False
        device = self._operation_device(operations[0])
        outputs: list[tuple[ProductRef, str]] = []
        parents: list[tuple[ProductRef, int, str]] = []
        for operation in operations:
            if (
                operation.work.kind != "token"
                or operation.work.mode != TokenMode.DECODE.value
                or self._operation_device(operation) != device
                or any(
                    input_product.kind is not ProductKind.SAMPLING_STATE
                    for input_product in operation.inputs
                )
                or not isinstance(operation.parent.point, DevicePoint)
            ):
                return False
            token_outputs = tuple(
                output
                for output in operation.outputs
                if output.kind is ProductKind.TOKEN and _requires_device_product_binding(output)
            )
            if (
                len(token_outputs) != 1
                or token_outputs[0].storage_class is not StorageClass.DEVICE_TENSOR
                or token_outputs[0].shape_bound.max_elements != 1
            ):
                return False
            point = operation.parent.point
            parent = operation.predicate
            if (
                parent is None
                or parent.kind is not ProductKind.TOKEN
                or parent.storage_class is not StorageClass.DEVICE_TENSOR
                or parent.shape_bound.max_elements != 1
            ):
                return False
            outputs.append((token_outputs[0], operation.plan_digest))
            parents.append(
                (
                    parent,
                    int(operation.op_id),
                    point.producer_plan_digest,
                )
            )
        continuation = self.products.device_products.bind_scalar_continuation(
            outputs=tuple(outputs),
            parents=tuple(parents),
            device=device,
        )
        scope.device_continuation = continuation
        scope.device_writes.extend(continuation.writes)
        for operation, parent_value in zip(operations, continuation.inputs, strict=True):
            if isinstance(operation.parent.point, DevicePoint) and operation.predicate is not None:
                scope.predicate_values[_operation_identity(operation)] = (parent_value, True)
        return True

    def _operation_device(self, operation: Operation) -> torch.device:
        return self._generation_device if operation.domain is Domain.GEN else self._device

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
        for operation in operations:
            operation_identity = _operation_identity(operation)
            point = operation.parent.point
            if isinstance(point, DevicePoint):
                selected = point.selected_point
                if selected is not None:
                    if selected.kind is not ProductKind.SELECTED_POINT:
                        raise invalid_descriptor(
                            "device parent does not name a selected-point product"
                        )
                    selected_read = self.products.device_products.consume(
                        selected,
                        consumer_op_id=operation.op_id,
                        producer_plan_digest=point.producer_plan_digest,
                        device=self._operation_device(operation),
                    )
                    scope.device_reads.append(selected_read)
                    scope.selected_point_reads[operation_identity] = selected_read
            predicate = operation.predicate
            if predicate is None:
                continue
            if operation_identity in scope.predicate_values:
                continue
            read = self.products.device_products.consume(
                predicate,
                consumer_op_id=operation.op_id,
                device=self._operation_device(operation),
            )
            scope.device_reads.append(read)
            scope.predicate_reads[operation_identity] = read
            tagged = predicate.kind is ProductKind.TOKEN and predicate.dtype is DType.U32
            scope.predicate_values[operation_identity] = (read.tensor, tagged)

    def _publish_predicates(
        self,
        scope: _ExecutionScope,
    ) -> None:
        writes = tuple(
            write
            for write in scope.device_writes
            if write.reference.kind is ProductKind.COMPLETION
            and _reference_operation_identity(write.reference) not in scope.continuation_writes
        )
        if not writes:
            return
        batch = self.products.device_products.producer_scalar_batch(writes)
        if batch is not None:
            batch.tensor.fill_(1)
            self.products.device_products.publish_scalar_batch(batch)
            return
        views = self.products.device_products.producer_write_views(writes)
        first = views[0]
        self.products.device_products.publish_writes(
            writes,
            torch.ones(
                (len(writes),),
                dtype=first.dtype,
                device=first.device,
            ),
        )

    def _finish_device_reads(
        self,
        scope: _ExecutionScope,
    ) -> None:
        if scope.device_continuation is not None:
            self.products.device_products.finish_continuation(scope.device_continuation)
        if not scope.device_reads:
            return
        reads = tuple(read for read in scope.device_reads if not read._recorded)
        if not reads:
            scope.device_reads.clear()
            return
        after_writes: list[DeviceProductWrite] = []
        for read in reads:
            write = scope.operation_writes.get(
                (read.reference.request_key, int(read.consumer_op_id))
            )
            if write is None:
                after_writes.clear()
                break
            after_writes.append(write)
        self.products.device_products.record_readers(
            reads,
            after_writes=tuple(after_writes),
        )
        scope.device_reads.clear()

    def _apply_release_controls(self, batch: Batch) -> None:
        releases = tuple(
            (control.request_key, control.op_id)
            for control in batch.controls
            if isinstance(control, Release)
        )
        self.products.device_products.release_operations(releases)
        self.latents.release_operations(releases)
        self.kv.release_operations(releases)
        if self.transport is not None:
            for identity in releases:
                self._release_locators(self._transport_publications.pop(identity, ()))

    def drop_session(self, session_id: int) -> None:
        """Release stage publications owned by one dropped request."""

        if self.transport is None:
            return
        selected = tuple(
            identity
            for identity in self._transport_publications
            if int(identity[0].session_id) == int(session_id)
        )
        for identity in selected:
            self._release_locators(self._transport_publications.pop(identity))

    def _reserve_kv_pages(
        self,
        partition: BatchPartition,
        scope: _ExecutionScope,
    ) -> None:
        """Validate complete scheduler mappings before binding physical pages."""

        operations = {
            (operation.request_key, operation.op_id): operation
            for operation in partition.operations
        }
        for placement in partition.kv_placements:
            operation = operations[(placement.request_key, placement.op_id)]
            scope.kv.apply_placement(
                placement.request_key,
                group_id=placement.group_id,
                block_table=placement.block_table,
                pages_to_zero=placement.pages_to_zero,
                expected_capacity_pages=operation.kv_capacity_pages,
            )

    def _stage_input_products(
        self,
        input_products: Sequence[ProductPayload],
        scope: _ExecutionScope,
    ) -> None:
        """Bind host-supplied input product values into the step product store.

        The submission batch carries each operation's input values as
        ``ProductPayload`` entries: prompt, forced, previous-sampled, and draft
        token ids for token work (a ``ProductKind.TOKEN`` product), and encoded
        image bytes for encode work. Each is decoded and put under its declared
        product identity BEFORE any operation runs, so the durable token and
        image sources the drivers read are resident at submission. Token ids ride
        the ``LogitsProduct.draft_token_ids`` channel the drivers already read.
        """

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
                    if self.transport is None:
                        raise capability_mismatch(
                            "cross-stage KV input requires a configured transport"
                        )
                    scope.kv.import_snapshot(
                        product.request_key.session_id,
                        snapshot,
                        self.transport,
                        transferred_tensors=transfer.tensors(),
                    )
                    scope.kv.stage_publication(product, snapshot)
                    continue
                tensors = transfer.tensors()
                if not tensors:
                    raise invalid_descriptor("tensor transfer produced no resident value")
                binding = self.products.device_products.bind_outputs(
                    (
                        (
                            product,
                            transfer.producer_plan_digest,
                            self._operation_device(scope.partition.operations[0]),
                        ),
                    )
                )[0]
                scope.device_writes.append(binding)
                resident = self.products.device_products.publish_write(binding, tensors[0])
                payload_kind = transfer.payload_kind
                height = transfer.height
                width = transfer.width
                if payload_kind is None or height is None or width is None:
                    raise RuntimeError("prepared tensor transfer has no validated geometry")
                if payload_kind is ProductKind.VISION_FEATURE:
                    transferred_payload: VisionFeatureProduct | LatentFeatureProduct = (
                        VisionFeatureProduct(
                            features=resident,
                            height=height,
                            width=width,
                            source_base64=None,
                        )
                    )
                elif payload_kind is ProductKind.LATENT_FEATURE and len(tensors) == 1:
                    transferred_payload = LatentFeatureProduct(
                        latent=resident,
                        height=height,
                        width=width,
                        source_base64=None,
                    )
                else:
                    raise invalid_descriptor("tensor transfer payload geometry is invalid")
                handle = self._input_product_handle(product)
                scope.product_view.put(
                    ProductRecord(
                        handle=handle,
                        session_id=product.request_key.session_id,
                        payload=transferred_payload,
                    )
                )
                self.sessions.get(product.request_key.session_id).product_handles.add(handle)
                continue
            if product.kind is ProductKind.SAMPLING_STATE:
                scope.sampling_states[_reference_operation_identity(product)] = (
                    decode_sampling_state_bytes(entry.payload)
                )
                continue
            handle = self._input_product_handle(product)
            inline_payload: LogitsProduct | EncodedImageProduct
            if product.kind is ProductKind.TOKEN:
                inline_payload = LogitsProduct(
                    logits=torch.empty(0),
                    source_mode=TokenMode.VERIFY,
                    draft_token_ids=decode_token_product_bytes(entry.payload),
                )
            elif entry.payload:
                inline_payload = EncodedImageProduct(entry.payload.decode("utf-8"))
            else:
                continue
            scope.product_view.put(
                ProductRecord(
                    handle=handle,
                    session_id=product.request_key.session_id,
                    payload=inline_payload,
                )
            )

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
        if all(
            operation.work.kind == "token" and operation.work.mode == TokenMode.DECODE.value
            for operation in operations
        ):
            return self._decode_batch(operations, scope)
        drivers = tuple(self._driver(operation, scope) for operation in operations)
        return self._drive(drivers, scope)

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
        if layout is None or layout.operations != operations:
            raise RuntimeError("partition transaction lost its aligned session view")
        sessions = layout.sessions
        for operation, session, entry, weights, current in zip(
            operations,
            sessions,
            layout.kv_entries,
            layout.weights,
            current_tokens,
            strict=True,
        ):
            if session.sampling is None:
                raise invalid_descriptor("sequence operation has no admitted sampling state")
            start = session.logical_position
            starts.append(start)
            tasks.append(
                self._token_task(
                    operation,
                    session,
                    (current,),
                    (start,),
                    TokenSelection.LAST_LOGITS,
                    scope,
                    entry=entry,
                    weights=weights,
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

        sample_started = time.perf_counter_ns()
        sample_tasks = tuple(
            self._sample_task(
                operation,
                row_logits,
                session,
                scope,
                positions=(start + 1,),
            )
            for operation, session, start, row_logits in zip(
                operations,
                sessions,
                starts,
                logits,
                strict=True,
            )
        )
        samples = _sample_task_batch(
            sample_tasks,
            scope.completion,
            device_products=self.products.device_products,
            device_reads=tuple(scope.device_reads),
            device_continuation=scope.device_continuation,
            selection_broadcast=self._broadcast_tp_selection,
        )
        _record_component(scope, "text_sample", sample_started)
        finalize_started = time.perf_counter_ns()
        self._publish_token_products(operations, samples, scope)
        outcomes: list[_Outcome] = []
        for operation, session, entry, start, sampled in zip(
            operations,
            sessions,
            layout.kv_entries,
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
        _record_component(scope, "text_finalize", finalize_started)
        return tuple(outcomes)

    def _drive(self, drivers: tuple[_Driver, ...], scope: _ExecutionScope) -> tuple[_Outcome, ...]:
        active: dict[int, tuple[_Driver, tuple[_ExecutorTask, ...]]] = {}
        completed: dict[int, _Outcome] = {}
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

    def _drive_partitioned(
        self,
        drivers: tuple[tuple[int, int, _Driver, _ExecutionScope], ...],
    ) -> tuple[tuple[_Outcome | None, ...], dict[int, BaseException]]:
        active: dict[int, tuple[_Driver, tuple[_ExecutorTask, ...], _ExecutionScope]] = {}
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
            flat: list[tuple[int, int, _ExecutorTask, _ExecutionScope]] = []
            for driver_index, (_driver, tasks, scope) in active.items():
                for task_index, task in enumerate(tasks):
                    flat.append((driver_index, task_index, task, scope))
            if not flat:
                raise RuntimeError("execution driver yielded an empty task wave")
            outputs, wave_errors = self._run_partitioned_task_wave(
                tuple((task, scope) for _driver, _task, task, scope in flat)
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
                tuple[_Driver, tuple[_ExecutorTask, ...], _ExecutionScope],
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
        tasks: tuple[tuple[_ExecutorTask, _ExecutionScope], ...],
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
                tuple((task, scope) for _index, task, scope in forward)
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
                    device_products=self.products.device_products,
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
            raise RuntimeError("executor task wave contains an unknown task type")
        return tuple(result), errors

    def _run_partitioned_wave(
        self,
        tasks: tuple[tuple[_ForwardTask, _ExecutionScope], ...],
    ) -> tuple[ForwardRowOutput, ...]:
        if self.runner is None:
            raise capability_mismatch("system-only executor received a neural operation")
        grouped: dict[
            tuple[object, ...],
            list[tuple[int, _ForwardTask, _ExecutionScope]],
        ] = defaultdict(list)
        for index, (task, scope) in enumerate(tasks):
            grouped[(scope.partition.submission_group, *self._group_key(task))].append(
                (index, task, scope)
            )
        groups: list[list[tuple[int, _ForwardTask, _ExecutionScope]]] = []
        for candidates in grouped.values():
            kinds = frozenset(task.row_kind for _index, task, _scope in candidates)
            route = candidates[0][1].route
            legal_mixed = self._model().allows_mixed(route, kinds)
            if len(kinds) > 1 and not legal_mixed:
                raise invalid_descriptor(
                    "tensorized mixed submission is outside the model runner capability"
                )
            groups.append(candidates)

        result: list[ForwardRowOutput | None] = [None] * len(tasks)
        for group in groups:
            indexes = tuple(index for index, _task, _scope in group)
            group_tasks = tuple(task for _index, task, _scope in group)
            group_scopes = tuple(scope for _index, _task, scope in group)
            target = torch.device(self._route_device(group_tasks[0].route))
            for scope in _unique_scopes(group_scopes):
                scope.completion.register_device(target)
            plan = self._forward_plan(group_tasks, group_scopes[0])
            output = self.runner.run(plan)
            observation = self.runner.last_observation
            if observation is None:
                raise RuntimeError("model runner returned without an execution observation")
            for scope in _unique_scopes(group_scopes):
                scope.observations.append(observation)
            for index, row_output in zip(indexes, output.rows, strict=True):
                result[index] = row_output
        return tuple(cast(ForwardRowOutput, value) for value in result)

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
                _sample_task_batch(
                    tuple(sample_tasks),
                    scope.completion,
                    device_products=self.products.device_products,
                    device_reads=tuple(scope.device_reads),
                    selection_broadcast=self._broadcast_tp_selection,
                ),
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
            model = self._model()
            legal_mixed = model.allows_mixed(route, kinds)
            if len(kinds) > 1 and not legal_mixed:
                by_kind: dict[RowKind, list[tuple[int, _ForwardTask]]] = defaultdict(list)
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

    def _broadcast_tp_selection(self, value: torch.Tensor) -> torch.Tensor:
        if self.mesh is None or self.mesh.mesh.tp_size <= 1:
            return value
        transport = self.mesh.mesh.transport("tp")
        if not isinstance(transport, BroadcastTransport):
            raise RuntimeError("designated-rank sampling requires TP broadcast transport")
        return transport.broadcast(value, src=0)

    def _group_key(self, task: _ForwardTask) -> tuple[object, ...]:
        return (
            task.route,
            self._route_device(task.route),
            self._model().route_dtype(task.route),
            self._model().route_topology(task.route),
            task.weights.digest,
            task.weights.version,
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
        scope.completion.register_device(target)
        weights = tasks[0].weights
        if any(task.weights is not weights for task in tasks):
            if any(
                task.weights.digest != weights.digest or task.weights.version != weights.version
                for task in tasks
            ):
                raise RuntimeError("forward group contains different immutable weight sets")

        kv_tasks = tuple(task for task in tasks if task.write_kv)
        if kv_tasks and len(kv_tasks) != len(tasks):
            raise invalid_descriptor("one physical route cannot mix KV and non-KV rows")
        staging_slot = self._tensor_stager.acquire(target)
        try:
            kv_view: KvView | EmptyKvView
            attention: AttnPlan
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
            mesh = (
                EmptyMeshView()
                if self.mesh is None
                else self.mesh.view(self._model().route_topology(route))
            )
            context = ForwardContext(
                kv=kv_view,
                latent=EmptyLatentView(),
                attention=attention,
                mesh=mesh,
                output=EmptyOutputView(),
            )
            graph_shape = self._hard_shape_key(route, tasks[0].row)
            graph_key = GraphKey(
                architecture_digest=cast(str, self.architecture_digest),
                weight_digest=cast(str, self.weight_digest),
                route=route,
                shape=graph_shape,
                dtype=self._model().route_dtype(route),
                backend=self._attention_selection().identity,
                topology=self._topology_key(route),
            )
            bindings = tuple(
                ForwardBinding(
                    row_id=task.row.row_id,
                    slot=task.row.output_slot,
                    output_dtype=(
                        self._generation().prediction_dtype
                        if isinstance(task.row, FlowRow)
                        else self._model().route_dtype(route)
                    ),
                    session_id=task.operation.request_key.session_id,
                    epoch=task.operation.request_key.epoch,
                    op_id=task.operation.op_id,
                    base_version=_parent_base_point(task.operation, task.session),
                )
                for task in tasks
            )
            plan = ForwardPlan(
                route=route,
                rows=tuple(task.row for task in tasks),
                context=context,
                bindings=bindings,
                graph_key=graph_key,
                graph_eligible=self._model().route_graph_eligible(route),
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
                _trace_envelopes(tuple(task.operation for task in tasks)),
                route=str(route),
                row_kind_counts=row_counts,
            )
            return plan
        except BaseException:
            self._tensor_stager.mark_submitted(staging_slot, target)
            raise

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
        if self._model().route_uses_packed_attention(route) and not pure_token_decode:
            return self._packed_attention_plan(tasks, scope, device, staging_slot)
        query_lens = tuple(task.query_tokens for task in tasks)
        if any(task.entry is None for task in tasks):
            raise RuntimeError("paged attention task has no aligned KV entry")
        view = scope.kv.view_entries(
            tuple(cast(KvEntry, task.entry) for task in tasks),
            query_lens=query_lens,
        )
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

    def _weights(self) -> WeightSet:
        if self.weights is None:
            raise RuntimeError("model route has no immutable weight authority")
        return self.weights

    def _model(self) -> ExecutionModel:
        if self.model is None:
            raise capability_mismatch("system-only executor received a neural operation")
        return self.model

    def _generation(self) -> GenerationPipeline:
        value = self._model().generation
        if not isinstance(value, GenerationPipeline):
            raise invalid_descriptor("operation requires model generation behavior")
        return value

    def _image_processor(self) -> ImageProcessor:
        value = self._model().image_processor
        if not isinstance(value, ImageProcessor):
            raise invalid_descriptor("operation requires model image processing")
        return value

    @staticmethod
    def _route(stage: LoweredStage) -> RouteId:
        return stage.route

    def _stages(
        self,
        variant: WorkVariant,
        *,
        retain_image: bool = False,
    ) -> tuple[LoweredStage, ...]:
        if self.model is None:
            return ()
        stages = self.model.lower(variant, retain_image=retain_image)
        if not stages and variant not in self.model.supported_work:
            raise invalid_descriptor(f"model does not implement operation {variant.value!r}")
        return stages

    def _operation_stages_for(self, operation: Operation) -> tuple[LoweredStage, ...]:
        """The model stages one registered operation lowers to.

        Materialization is polymorphic on its input: a latent input drives the
        model image-decode stages, while a transported or resident image frame
        is model-free and lowers to no neural stage.
        """

        if operation.work.variant is WorkVariant.MATERIALIZE and not any(
            reference.kind is ProductKind.LATENT for reference in operation.inputs
        ):
            return ()
        return self._stages(operation.work.variant)

    def _primary_stage(self, variant: WorkVariant) -> LoweredStage:
        return self._model().primary_stage(variant)

    def _state_stages(
        self,
        variant: WorkVariant,
        *,
        retain_image: bool,
    ) -> tuple[LoweredStage, ...]:
        return self._model().state_stages(variant, retain_image=retain_image)

    def _route_device(self, route: RouteId) -> str:
        deployment = cast(WorkerDeployment, self.deployment)
        if self._model().route_device_role(route) is DeviceRole.GENERATION:
            return deployment.generation_device or deployment.device
        return deployment.device

    def _attention_selection(self) -> AttentionSelection:
        if self.attention is None:
            raise RuntimeError("model route has no attention selection")
        return self.attention

    def _topology_key(self, route: RouteId) -> str:
        deployment = cast(WorkerDeployment, self.deployment)
        return f"{','.join(self._model().route_topology(route))}:{deployment.tp_rank}/{deployment.tp_size}"

    def _hard_shape_key(self, route: RouteId, row: ForwardRow) -> tuple[int, ...]:
        return self._model().route_shape_key(route, row)

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
        session = self.sessions.get(operation.request_key.session_id)
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
        session: RequestSession,
        scope: _ExecutionScope,
    ) -> _Driver:
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
        )
        sample = _sample_result((yield (sample_task,))[0])
        if scores_prompt:
            sample = replace(
                sample,
                prompt_logprobs=self._prompt_logprob_details(
                    session,
                    start,
                    tokens,
                    logits,
                    scope,
                ),
            )
        self._publish_token_product(operation, sample, scope)
        session.rng_counter += 1
        session.logical_position = start + len(tokens)
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
        session: RequestSession,
        start: int,
        tokens: tuple[int, ...],
        logits: torch.Tensor,
        scope: _ExecutionScope,
    ) -> tuple[
        tuple[tuple[int, float, int], ...] | _CompletionTopLogprobs,
        ...,
    ]:
        if logits.ndim != 2 or int(logits.shape[0]) != len(tokens):
            raise invalid_descriptor("prompt scoring logits do not align with input tokens")
        handle = _stable_handle(
            session.request_key.session_id,
            session.request_key.epoch,
            0,
            "prompt_logits",
        )
        if start == 0:
            score_logits = logits[:-1]
            targets = tokens[1:]
        else:
            record = scope.product_view.get(handle)
            if record is None or not isinstance(record.payload, LogitsProduct):
                raise invalid_descriptor("continued prompt scoring has no preceding logits")
            previous = record.payload.logits.reshape(1, -1).to(
                device=logits.device,
                dtype=logits.dtype,
            )
            score_logits = torch.cat((previous, logits[:-1]), dim=0)
            targets = tokens
        scope.product_view.put(
            ProductRecord(
                handle=handle,
                session_id=session.session_id,
                payload=LogitsProduct(
                    logits=logits[-1].detach(),
                    source_mode=TokenMode.EXTEND,
                ),
            )
        )
        session.prompt_logits_handle = handle
        session.product_handles.add(handle)
        if not targets:
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
            for _ in targets
        )
        target_tensor = torch.tensor(
            targets,
            dtype=torch.long,
            device=score_logits.device,
        )
        indexes = torch.arange(
            len(targets),
            dtype=torch.long,
            device=score_logits.device,
        )
        details = _sample_logprob_details(
            score_logits.float(),
            indexes,
            target_tensor,
            rows,
            scope.completion,
        )
        return tuple(details[index][1] for index in range(len(targets)))

    def _visual_extend(
        self,
        operation: Operation,
        session: RequestSession,
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
        record = scope.product_view.get(self._input_product_handle(reference))
        if record is None:
            raise invalid_descriptor("visual extend feature product is not resident")
        read = self.products.device_products.consume(
            reference,
            consumer_op_id=operation.op_id,
            device=self._operation_device(operation),
        )
        scope.device_reads.append(read)
        payload = record.payload
        position = session.logical_position
        closes_feedback = any(output.kind is ProductKind.COMPLETION for output in operation.outputs)
        samples_continuation = any(output.kind is ProductKind.TOKEN for output in operation.outputs)
        if reference.kind is ProductKind.VISION_FEATURE:
            if not isinstance(payload, VisionFeatureProduct):
                raise invalid_descriptor("vision feature product has the wrong resident payload")
            outcome = yield from self._state_driver(
                operation,
                WorkVariant.ENCODE_VISION,
                scope,
                height=payload.height,
                width=payload.width,
                conditioning_position=position,
                features=read.tensor,
                sample_token=samples_continuation,
                close_image=closes_feedback,
                retain_image=True,
            )
        else:
            if not isinstance(payload, LatentFeatureProduct):
                raise invalid_descriptor("latent feature product has the wrong resident payload")
            outcome = yield from self._state_driver(
                operation,
                WorkVariant.ENCODE_LATENT,
                scope,
                height=payload.height,
                width=payload.width,
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
        return self._state_outcome(operation, outcome, base=position)

    def _decode(
        self,
        operation: Operation,
        session: RequestSession,
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
        )
        sampled = _sample_result((yield (sample_task,))[0])
        session.rng_counter += 1
        session.logical_position = start + 1
        self._publish_token_product(operation, sampled, scope)
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
        session: RequestSession,
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
        )
        sampled = _sample_result((yield (sample_task,))[0])
        initialized = scope.kv.initialize(operation.request_key.session_id, task.query_tokens)
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
        session: RequestSession | None = None,
        kv_entry: KvEntry | None = None,
        base: int,
        tokens: int | _CompletionDerivedInteger | _CompletionSpeculativePoint,
        committed_tokens: tuple[int | _CompletionToken, ...],
        sample: _SampleResult | None = None,
        selection: _SpeculativeSelection | None = None,
    ) -> _Outcome:
        if session is None:
            session = self.sessions.get(operation.request_key.session_id)
        extents = (
            self.kv.get(operation.request_key.session_id) if kv_entry is None else kv_entry
        ).extents()
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
        session: RequestSession,
        token_ids: tuple[int | torch.Tensor, ...],
        positions: tuple[int, ...],
        selection: TokenSelection,
        scope: _ExecutionScope,
        *,
        entry: KvEntry | None = None,
        weights: WeightSet | None = None,
    ) -> _ForwardTask:
        stage = self._primary_stage(operation.work.variant)
        if stage.row is not RowKind.TOKEN:
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
            operation=operation,
            session=session,
            weights=self._weights() if weights is None else weights,
            stage=stage,
            route=self._route(stage),
            row=row,
            entry=self.kv.get(operation.request_key.session_id) if entry is None else entry,
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
            scope.kv.advance(task.operation.request_key.session_id, count)

    def _operation_token_ids(
        self,
        operation: Operation,
        scope: _ExecutionScope,
    ) -> tuple[int, ...]:
        """Read the token id values a token operation names as an input product.

        Host-known prompt and draft tokens arrive through a declared host-staging
        product. Device-rooted decode tokens are resolved separately through the
        generation-tagged device product table.
        """

        for reference in operation.inputs:
            if reference.kind is not ProductKind.TOKEN:
                continue
            record = scope.product_view.get(self._input_product_handle(reference))
            if record is not None and isinstance(record.payload, LogitsProduct):
                return tuple(int(value) for value in record.payload.draft_token_ids)
        raise invalid_descriptor("token operation has no input token product")

    def _input_product_handle(self, reference: ProductRef) -> int:
        """Bind one declared product identity to a stable worker-local handle."""

        if reference.storage_class is StorageClass.LATENT_ARENA:
            generation = int(reference.generation)
            if generation < 1:
                raise invalid_descriptor("resident product requires a positive generation")
            return generation
        return _stable_handle(
            reference.request_key.session_id,
            reference.request_key.epoch,
            reference.producer_op_id,
            f"product:{reference.output_index}",
        )

    def _resolve_decode_token(
        self,
        operation: Operation,
        session: RequestSession,
        scope: _ExecutionScope,
    ) -> int | torch.Tensor:
        point = operation.parent.point
        if isinstance(point, DevicePoint):
            predicate = scope.predicate_values.get(_operation_identity(operation))
            if predicate is None or not predicate[1]:
                raise invalid_descriptor("device token continuation is not registered")
            return _decode_tagged_token_views((predicate[0],))[0]
        tokens = self._operation_token_ids(operation, scope)
        if not tokens:
            raise invalid_descriptor("last-sampled token source has no committed token")
        return int(tokens[0])

    def _resolve_decode_tokens(
        self,
        operations: tuple[Operation, ...],
        scope: _ExecutionScope,
    ) -> tuple[int | torch.Tensor, ...]:
        if scope.device_continuation is not None:
            return _decode_tagged_token_views(scope.device_continuation.inputs)
        resolved: list[int | torch.Tensor | None] = [None] * len(operations)
        for index, operation in enumerate(operations):
            point = operation.parent.point
            if isinstance(point, DevicePoint):
                predicate = scope.predicate_values.get(_operation_identity(operation))
                if predicate is None or not predicate[1]:
                    raise invalid_descriptor("device token continuation is not registered")
                resolved[index] = _decode_tagged_token_views((predicate[0],))[0]
                continue
            tokens = self._operation_token_ids(operation, scope)
            if not tokens:
                raise invalid_descriptor("last-sampled token source has no committed token")
            resolved[index] = int(tokens[0])
        if any(value is None for value in resolved):
            raise RuntimeError("decode token resolution left an operation without input")
        return tuple(cast(int | torch.Tensor, value) for value in resolved)

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
        self.products.device_products.publish_write(
            write,
            _tagged_token_values(device_token, continuation),
        )

    def _publish_token_products(
        self,
        operations: tuple[Operation, ...],
        samples: tuple[_SampleResult, ...],
        scope: _ExecutionScope,
    ) -> None:
        continuation = scope.device_continuation
        if continuation is not None and continuation.published:
            return
        if all(sample.device_product_published for sample in samples):
            return
        if continuation is not None:
            continuation_tokens = tuple(
                sample.device_token for sample in samples if sample.device_token is not None
            )
            continuation_flags = tuple(
                sample.device_continuation
                for sample in samples
                if sample.device_continuation is not None
            )
            if len(continuation_tokens) != len(samples) or len(continuation_flags) != len(samples):
                raise RuntimeError("device continuation sampling lost a device token")
            packed = packed_tensor_views(continuation_tokens)
            if packed is None:
                packed = torch.cat(continuation_tokens, dim=0)
            packed_flags = packed_tensor_views(continuation_flags)
            if packed_flags is None:
                packed_flags = torch.cat(continuation_flags, dim=0)
            self.products.device_products.publish_continuation(
                continuation,
                _tagged_token_values(packed, packed_flags),
                after_reads=tuple(scope.device_reads),
            )
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
        self.products.device_products.publish_writes(
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
        self.products.device_products.publish_write(selected_write, selected_point)
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
        self.products.device_products.publish_write(
            span_write,
            torch.cat((selected_point.reshape(-1), visible_tokens)),
        )
        self.products.device_products.publish_write(
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
        session: RequestSession,
        scope: _ExecutionScope,
        *,
        positions: tuple[int, ...],
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
        # The device-resident committed penalty base. Every generated token folds
        # in as its operation executes, so a successor registered before its
        # predecessors are observed still reads their counts. No host token
        # history participates.
        penalty_base = (
            self._session_penalty_base(session, vocab, rows.device) if uses_penalties else None
        )
        forced_token_ids = sampling.forced_token_ids
        descriptors: list[_SamplingRow] = []
        for index, position in enumerate(positions):
            if penalty_base is None:
                row_counts = None
            elif index == 0 or not draft_token_ids:
                row_counts = penalty_base
            else:
                # A speculative point folds its accepted draft prefix on top of
                # the committed base for that point's penalties.
                row_counts = penalty_base.clone()
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
        continuation_product = scope.continuation_writes.get(operation_identity)
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
            continuation_product=continuation_product,
            predicate=None if predicate_value is None else predicate_value[0],
            tagged_predicate=False if predicate_value is None else predicate_value[1],
            penalty_base=penalty_base,
        )

    def _session_penalty_base(
        self,
        session: RequestSession,
        vocab: int,
        device: torch.device,
    ) -> torch.Tensor:
        """The request's device-resident committed penalty count base.

        Allocated lazily as a dense per-vocabulary int32 count tensor the first
        time a penalty-bearing operation samples for this session. Every
        subsequent operation reads and folds into the same tensor, so penalties
        stay device-continuous across the unresolved window.
        """

        base = session.penalty_counts
        if base is None or int(base.numel()) != vocab or base.device != device:
            base = torch.zeros(vocab, dtype=torch.int32, device=device)
            session.penalty_counts = base
        return base

    def _transition_driver(
        self,
        operation: Operation,
        scope: _ExecutionScope,
    ) -> _Driver:
        if False:  # pragma: no cover - keeps the driver protocol uniform
            yield ()
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
        self.kv.validate_conditioning(session_id, conditioning[0])
        session = self.sessions.get(session_id)
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
        initial = self._initial_latent(operation, image.height, image.width, session)
        write = _bound_device_write(scope, output)
        resident = self.products.device_products.publish_write(write, initial)
        scope.operation_writes[_operation_identity(operation)] = write
        scope.latents.write(
            LatentRecord(
                reference=output,
                producer_plan_digest=operation.plan_digest,
                value=resident,
                step=0,
                height=image.height,
                width=image.width,
            )
        )
        session.latent_product = output
        length = self.kv.get(session_id).length
        extents = self.kv.get(session_id).extents()
        return _Outcome(
            status=OpStatus.OK,
            selected_point=1,
            logical_lengths=LogicalLengths(
                token_len=session.logical_position,
                kv_visible_len=length,
                latent_len=0,
                kv_reserved_len=extents.reserved,
                kv_initialized_len=extents.initialized,
                kv_committed_len=extents.committed,
                kv_published_len=extents.published,
            ),
            token_span=TokenSpan(base=session.logical_position, len=0),
            finish_flags=FinishFlags(),
            product_generations=_output_generations(operation),
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
        self.kv.validate_conditioning(session_id, conditioning[0])
        session = self.sessions.get(session_id)
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
        latent_read = self.products.device_products.consume(
            latent_input,
            consumer_op_id=operation.op_id,
            device=self._operation_device(operation),
        )
        scope.device_reads.append(latent_read)
        start_step = session.flow_step
        remaining = int(image.steps) - start_step
        step_count = (
            remaining
            if operation.bounds.max_tokens <= 0
            else min(int(operation.bounds.max_tokens), remaining)
        )
        if step_count < 0 or start_step + step_count > image.steps:
            raise invalid_descriptor("flow operation exceeds the declared schedule")
        conditioning_position = session.logical_position
        image_prompt = image.image_prompts[0] if image.image_prompts else ""
        record = scope.latents.read(latent_input)
        if record is None:
            raise invalid_descriptor("flow continuation references a missing latent generation")
        if record.reference.request_key.session_id != session_id:
            raise invalid_descriptor("flow latent belongs to another session")
        if record.step != start_step or session.flow_step != start_step:
            raise invalid_descriptor("flow operation start step does not match committed state")

        # A flow quantum publishes a new immutable latent generation. Carry the
        # already-computed CFG branch prefixes forward under that exact successor
        # owner instead of rebuilding them for every denoise step.
        scope.kv.rebind_scratch_owner(latent_input, latent_output)

        schedule = flow.schedule(int(image.steps), float(image.timestep_shift))
        current = latent_read.tensor
        for step in range(start_step, start_step + step_count):
            t, t_next = schedule.pair(
                step,
                device=current.device,
                dtype=torch.float32,
            )
            host_t = schedule.scalar(step)
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

            entries: dict[Branch, KvEntry] = {}
            prefix_tasks: list[_ForwardTask] = []
            for branch in guide.branches:
                source = self._branch_source(branch)
                prefix, copy_conditioning = self._flow_prefix(
                    source,
                    image_prompt,
                    session,
                )
                query = self._flow_physical_tokens(record.height, record.width)
                entry, created = scope.kv.scratch_entry(
                    latent_output,
                    branch.value,
                    capacity_tokens=(
                        self.kv.get(session_id).length if copy_conditioning else len(prefix)
                    )
                    + query,
                    copy_conditioning=copy_conditioning,
                )
                entries[branch] = entry
                if created and prefix:
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
            session.flow_step = step + 1
        write = _bound_device_write(scope, latent_output)
        resident = self.products.device_products.publish_write(write, current)
        scope.operation_writes[_operation_identity(operation)] = write
        scope.latents.write(
            replace(
                record,
                reference=latent_output,
                producer_plan_digest=operation.plan_digest,
                value=resident,
            )
        )
        session.latent_product = latent_output
        if record.step == int(image.steps):
            scope.kv.release_scratch_owner(latent_output)
        length = self.kv.get(session_id).length
        extents = self.kv.get(session_id).extents()
        return _Outcome(
            status=OpStatus.OK,
            selected_point=1,
            logical_lengths=LogicalLengths(
                token_len=session.logical_position,
                kv_visible_len=length,
                # The cumulative denoise steps this lineage has advanced to after
                # the quantum (start_step + step_count), which the ordered-commit
                # validator matches against the request's expected denoise step.
                latent_len=record.step,
                kv_reserved_len=extents.reserved,
                kv_initialized_len=extents.initialized,
                kv_committed_len=extents.committed,
                kv_published_len=extents.published,
            ),
            token_span=TokenSpan(base=session.logical_position, len=0),
            finish_flags=FinishFlags(),
            product_generations=_output_generations(operation),
        )

    def _initial_latent(
        self,
        operation: Operation,
        height: int,
        width: int,
        session: RequestSession,
    ) -> torch.Tensor:
        flow = self._generation()
        route = self._route(self._primary_stage(operation.work.variant))
        device = torch.device(self._route_device(route))
        dtype = _torch_dtype(self._model().route_dtype(route))
        rng = operation.rng
        assert rng is not None and rng.draw_layout is DrawLayout.FLOW_NOISE
        seed = flow_noise_seed(int(rng.seed), int(rng.semantic_index_base))
        return normal_noise(
            flow.latent_shape(height, width),
            seed=seed,
            device=device,
            dtype=dtype,
        ) * flow.noise_scale(height, width)

    def _noise_scale(self, height: int, width: int) -> float:
        return self._generation().noise_scale(height, width)

    def _branch_source(self, branch: Branch) -> BranchSource:
        return self._generation().branch_source(branch)

    def _flow_prefix(
        self,
        source: BranchSource,
        image_prompt: str,
        session: RequestSession,
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
        entry: KvEntry,
        branch: Branch,
        scope: _ExecutionScope,
    ) -> _ForwardTask:
        session = self.sessions.get(operation.request_key.session_id)
        primary = self._primary_stage(operation.work.variant)
        route = self._route(primary)
        stage = LoweredStage(route, RowKind.TOKEN)
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
            operation=operation,
            session=session,
            weights=self._weights(),
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
        operation: Operation,
        conditioning_position: int,
        branch: Branch,
        entry: KvEntry,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        height: int,
        width: int,
        scope: _ExecutionScope,
    ) -> _ForwardTask:
        session = self.sessions.get(operation.request_key.session_id)
        flow = self._generation()
        stage = self._primary_stage(operation.work.variant)
        if stage.row is not RowKind.FLOW:
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
            operation=operation,
            session=session,
            weights=self._weights(),
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
        conditioning_position: int,
        entry: KvEntry,
    ) -> int:
        if branch is Branch.COND:
            return int(conditioning_position)
        return int(entry.length)

    def _flow_query_tokens(self, latent: torch.Tensor, height: int, width: int) -> int:
        del latent
        return self._generation().image_tokens(height, width)

    def _flow_physical_tokens(self, height: int, width: int) -> int:
        return self._generation().physical_tokens(height, width)

    def _flow_neural_latent(
        self,
        latent: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        return self._generation().neural_latent(latent)

    def _flow_store_latent(
        self,
        latent: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        return self._generation().stored_latent(latent, height, width)

    def _flow_conditioning(
        self,
        latent: torch.Tensor,
        height: int,
        width: int,
    ) -> FlowPatches:
        transform = self._image_processor().vit
        return cast(
            FlowPatches,
            self._generation().conditioning(
                latent,
                height,
                width,
                patch_size=(
                    int(transform.patch_size) if isinstance(transform, PatchTransform) else None
                ),
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
        *,
        base: int | None = None,
        products: tuple[ProductPayload, ...] = (),
    ) -> _Outcome:
        session = self.sessions.get(operation.request_key.session_id)
        extents = self.kv.get(operation.request_key.session_id).extents()
        span_base = session.logical_position if base is None else int(base)
        return _Outcome(
            status=OpStatus.OK,
            selected_point=1 if operation.advances_state else 0,
            logical_lengths=LogicalLengths(
                token_len=session.logical_position,
                kv_visible_len=extents.visible,
                kv_reserved_len=extents.reserved,
                kv_initialized_len=extents.initialized,
                kv_committed_len=extents.committed,
                kv_published_len=extents.published,
            ),
            token_span=TokenSpan(base=span_base, len=outcome.sampled_tokens),
            finish_flags=FinishFlags(),
            product_generations=_output_generations(operation),
            committed_tokens=outcome.committed_tokens,
            products=(*products, *outcome.products),
        )

    def _non_state_outcome(
        self,
        operation: Operation,
        *,
        products: tuple[ProductPayload, ...] = (),
        completion_tasks: tuple[_CompletionImagePayload, ...] = (),
    ) -> _Outcome:
        session = self.sessions.get(operation.request_key.session_id)
        extents = self.kv.get(operation.request_key.session_id).extents()
        base = session.logical_position
        return _Outcome(
            status=OpStatus.OK,
            selected_point=1 if operation.advances_state else 0,
            logical_lengths=LogicalLengths(
                token_len=session.logical_position,
                kv_visible_len=extents.visible,
                kv_reserved_len=extents.reserved,
                kv_initialized_len=extents.initialized,
                kv_committed_len=extents.committed,
                kv_published_len=extents.published,
            ),
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
        selected_type = operation.work.variant
        mode = EncodeMode(cast(str, operation.work.mode))
        session_id = operation.request_key.session_id
        feature_outputs = tuple(
            output
            for output in operation.outputs
            if output.storage_class is StorageClass.LATENT_ARENA
            and output.kind in {ProductKind.VISION_FEATURE, ProductKind.LATENT_FEATURE}
        )
        if len(feature_outputs) != 1:
            raise invalid_descriptor("encode operation requires one resident feature output")
        feature_output = feature_outputs[0]
        handle = int(feature_output.generation)
        if handle < 1:
            raise invalid_descriptor("encode feature output requires a positive generation")
        payload: VisionFeatureProduct | LatentFeatureProduct
        source = self._encode_source(operation, scope)
        stage = self._primary_stage(selected_type)
        if stage.row is not RowKind.ENCODE:
            raise invalid_descriptor("encode primary stage is not an encode row")
        target_device = torch.device(self._route_device(self._route(stage)))
        if isinstance(source, ImageTensorProduct):
            prepared = prepare_tensor_image(
                image_spec,
                mode,
                source.image,
                device=target_device,
                signed_unit=source.value_range is ImageRange.SIGNED_UNIT,
            )
            source_base64 = None
        else:
            prepared = prepare_image(
                image_spec,
                mode,
                source,
                device=target_device,
            )
            source_base64 = source
        task = self._encode_task(operation, mode, prepared, stage, scope)
        outputs = yield (task,)
        features = _encode_features(outputs[0]).detach()
        if mode is EncodeMode.VISION:
            payload = VisionFeatureProduct(
                features=features,
                height=prepared.height,
                width=prepared.width,
                source_base64=source_base64,
            )
        else:
            payload = LatentFeatureProduct(
                latent=features,
                height=prepared.height,
                width=prepared.width,
                source_base64=source_base64,
            )
        value = payload.features if isinstance(payload, VisionFeatureProduct) else payload.latent
        write = _bound_device_write(scope, feature_output)
        resident = self.products.device_products.publish_write(
            write,
            value,
        )
        scope.operation_writes.setdefault(_operation_identity(operation), write)
        payload = (
            replace(payload, features=resident)
            if isinstance(payload, VisionFeatureProduct)
            else replace(payload, latent=resident)
        )
        scope.product_view.put(
            ProductRecord(
                handle=handle,
                session_id=session_id,
                payload=payload,
            )
        )
        session = self.sessions.get(session_id)
        session.product_handles.add(handle)
        products: tuple[ProductPayload, ...] = ()
        if (
            self.transport is not None
            and self.transport.name != "local"
            and (self.deployment is None or int(self.deployment.tp_rank) == 0)
        ):
            locator = self.transport.publish_async(resident)
            scope.published.append(locator)
            scope.stage_publications[_operation_identity(operation)] = (locator,)
            descriptor = _CompletionTransferPayload(
                "tensor",
                {
                    "locator": locator.to_wire(),
                    "payload_kind": feature_output.kind.value,
                    "height": payload.height,
                    "width": payload.width,
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
        return self._non_state_outcome(operation, products=products)

    def _materialize_driver(
        self,
        operation: Operation,
        scope: _ExecutionScope,
    ) -> _Driver:
        session_id = operation.request_key.session_id
        session = self.sessions.get(session_id)
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
        latent_read = self.products.device_products.consume(
            latent_input,
            consumer_op_id=operation.op_id,
            device=self._operation_device(operation),
        )
        scope.device_reads.append(latent_read)
        flow = self._generation()
        latent_record = scope.latents.read(latent_input)
        if latent_record is None or latent_record.reference.request_key.session_id != session_id:
            raise invalid_descriptor("materialization latent is not resident for this session")
        image_params = session.image
        if image_params is None:
            raise invalid_descriptor("image materialization has no admitted image parameters")
        if latent_record.step != image_params.steps:
            raise invalid_descriptor("image materialization requires a completed latent trajectory")

        if flow.materialization is Materialization.DECODE_ROUTE:
            stage = self._primary_stage(operation.work.variant)
            if stage.row is not RowKind.DECODE:
                raise invalid_descriptor("decode materialization requires a decode primary stage")
            row_id = scope.row_id()
            task = _ForwardTask(
                operation=operation,
                session=session,
                weights=self._weights(),
                stage=stage,
                route=self._route(stage),
                row=DecodeRow(
                    row_id=row_id,
                    latent=latent_read.tensor,
                    image_height=latent_record.height,
                    image_width=latent_record.width,
                    output_slot=row_id,
                ),
            )
            outputs = yield (task,)
            image_tensor = _decoded_tensor(outputs[0]).detach()
            image_range = ImageRange.UNIT
        elif flow.materialization is Materialization.RGB_LATENT:
            if any(not stage.publishes_state for stage in self._operation_stages_for(operation)):
                raise invalid_descriptor(
                    "RGB-latent materialization must not declare a decode route"
                )
            image_tensor = latent_read.tensor.detach()
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
            resident_image = self.products.device_products.publish_write(
                write,
                image_tensor,
            )
            scope.operation_writes.setdefault(_operation_identity(operation), write)
            scope.product_view.put(
                ProductRecord(
                    handle=image_handle,
                    session_id=session_id,
                    payload=ImageTensorProduct(
                        image=resident_image,
                        height=latent_record.height,
                        width=latent_record.width,
                        value_range=image_range,
                    ),
                )
            )
            session.product_handles.add(image_handle)
        image_task = self._defer_image_encoding(
            operation,
            image_tensor,
            image_range,
            scope,
            max_bytes=int(artifact.max_bytes),
            discard_handles=tuple(int(output.generation) for output in resident_outputs),
        )
        session.latent_product = None
        session.flow_step = 0
        products = (
            ProductPayload(
                product=artifact,
                payload=cast(bytes, image_task),
            ),
        )
        return self._non_state_outcome(
            operation,
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
        # Every depth-one transfer resolves without a device wave; the guarded yield
        # keeps this a generator so the shared drive loop can complete it in place.
        if False:  # pragma: no cover - marks this driver a generator
            yield ()
        session_id = operation.request_key.session_id
        mode = operation.work.mode
        if mode == TransferMode.KV_PUBLISH.value:
            point = _fixed_parent(operation)
            outputs = tuple(output for output in operation.outputs if output.kind is ProductKind.KV)
            if len(outputs) != 1:
                raise invalid_descriptor("KV publication requires one KV output product")
            if any(reference.kind is ProductKind.KV for reference in operation.inputs):
                raise invalid_descriptor("KV publication is rooted only by its fixed parent")
            expected_base = scope.kv.destination_base(session_id, "gen")
            snapshot = scope.kv.publish_kv(
                session_id,
                source_version=operation.parent,
                source_digest=point.semantic_digest,
                destination="gen",
                expected_base=expected_base,
                product=outputs[0],
                transport=self.transport,
            )
            for encoded in snapshot.locators:
                scope.published.append(Locator.from_wire_json(encoded))
            payload = _CompletionTransferPayload(
                "kv",
                {"snapshot": snapshot.to_wire()},
                tuple(Locator.from_wire_json(encoded) for encoded in snapshot.locators),
                operation.plan_digest,
                self.transport,
            )
            return self._non_state_outcome(
                operation,
                products=(ProductPayload(product=outputs[0], payload=cast(bytes, payload)),),
            )
        if mode == TransferMode.KV_INSTALL.value:
            inputs = tuple(
                reference for reference in operation.inputs if reference.kind is ProductKind.KV
            )
            outputs = tuple(output for output in operation.outputs if output.kind is ProductKind.KV)
            if len(inputs) != 1 or len(outputs) != 1:
                raise invalid_descriptor("KV installation requires one input and one output")
            snapshot = scope.kv.install_publication(
                session_id,
                inputs[0],
                outputs[0],
                self.transport,
                transferred_tensors=(
                    None
                    if (prepared := scope.prepared_transfers.get(inputs[0])) is None
                    else prepared.tensors()
                ),
            )
            return self._non_state_outcome(
                operation,
                products=(ProductPayload(product=outputs[0], payload=b""),),
            )
        value, metadata = self._fetch_product_tensor(operation, scope)
        self._publish_tensor(
            value,
            scope,
            payload_kind=_metadata_string(metadata, "payload_kind", "tensor"),
            height=_metadata_uint(metadata, "height", 0),
            width=_metadata_uint(metadata, "width", 0),
            value_range=_metadata_string(metadata, "value_range", ""),
        )
        return self._non_state_outcome(operation)

    def _encode_source(
        self,
        operation: Operation,
        scope: _ExecutionScope,
    ) -> str | ImageTensorProduct:
        for reference in operation.inputs:
            record = scope.product_view.get(self._input_product_handle(reference))
            if record is None:
                continue
            payload = record.payload
            if isinstance(payload, (VisionFeatureProduct, LatentFeatureProduct)):
                if payload.source_base64 is not None:
                    return payload.source_base64
                continue
            if isinstance(payload, ImageTensorProduct):
                if reference.storage_class is StorageClass.LATENT_ARENA:
                    read = self.products.device_products.consume(
                        reference,
                        consumer_op_id=operation.op_id,
                        device=self._operation_device(operation),
                    )
                    scope.device_reads.append(read)
                    return replace(payload, image=read.tensor)
                return payload
            if isinstance(payload, EncodedImageProduct):
                return payload.base64
        raise invalid_descriptor("encode operation has no source image product")

    def _encode_task(
        self,
        operation: Operation,
        mode: EncodeMode,
        prepared: PreparedImage,
        stage: LoweredStage,
        scope: _ExecutionScope,
    ) -> _ForwardTask:
        session = self.sessions.get(operation.request_key.session_id)
        row_id = scope.row_id()
        row = EncodeRow(
            row_id=row_id,
            kind=(
                ForwardEncodeKind.VISION if mode is EncodeMode.VISION else ForwardEncodeKind.LATENT
            ),
            inputs=prepared.inputs,
            output_slot=row_id,
        )
        return _ForwardTask(
            operation=operation,
            session=session,
            weights=self._weights(),
            stage=stage,
            route=self._route(stage),
            row=row,
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
    ) -> Generator[tuple[_ExecutorTask, ...], _TaskResult, _StateOutcome]:
        committed_tokens: tuple[int | _CompletionToken, ...] = ()
        products: tuple[ProductPayload, ...] = ()
        state_query_tokens = 0
        for stage in self._state_stages(variant, retain_image=retain_image):
            if stage.row is RowKind.ENCODE:
                if image is None:
                    raise invalid_descriptor("image state encode stage has no image tensor")
                images = self._image_processor()
                prepared = prepare_tensor_image(
                    images,
                    EncodeMode.VISION,
                    image,
                    device=torch.device(self._route_device(self._route(stage))),
                    signed_unit=image_range is ImageRange.SIGNED_UNIT,
                )
                task = self._encode_task(operation, EncodeMode.VISION, prepared, stage, scope)
                outputs = yield (task,)
                features = _encode_features(outputs[0]).detach()
                continue
            if stage.row is RowKind.TOKEN:
                if features is None:
                    raise invalid_descriptor("token state stage has no vision features")
                task = self._vision_state_task(
                    operation,
                    stage,
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
                    if not isinstance(outputs[0], TokenOutput) or not isinstance(
                        outputs[0].value, TokenLogits
                    ):
                        raise invalid_descriptor("image state token stage did not return logits")
                    session = self.sessions.get(operation.request_key.session_id)
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
                    )
                    sampled = _sample_result((yield (sample_task,))[0])
                    self._publish_token_product(operation, sampled, scope)
                    session.rng_counter += 1
                    committed_tokens = (sampled.token_id,)
                    products = _sample_product_payloads(operation, sampled)
                continue
            if stage.row is RowKind.FLOW:
                if latent is None:
                    raise invalid_descriptor("flow state stage has no latent tensor")
                task = self._latent_state_task(
                    operation,
                    stage,
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
            raise invalid_descriptor("state publication cannot use a decode row")
        return _StateOutcome(committed_tokens, products)

    def _vision_state_task(
        self,
        operation: Operation,
        stage: LoweredStage,
        features: torch.Tensor,
        height: int,
        width: int,
        conditioning_position: int,
        scope: _ExecutionScope,
        *,
        close_image: bool,
        logits: bool,
    ) -> _ForwardTask:
        session = self.sessions.get(operation.request_key.session_id)
        injection = self._image_processor().feature_injection
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
            int(embeddings.shape[0]),
            conditioning_position,
            height=height,
            width=width,
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
            operation=operation,
            session=session,
            weights=self._weights(),
            stage=stage,
            route=self._route(stage),
            row=row,
            entry=self.kv.get(operation.request_key.session_id),
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
        stage: LoweredStage,
        latent: torch.Tensor,
        height: int,
        width: int,
        conditioning_position: int,
        scope: _ExecutionScope,
    ) -> _ForwardTask:
        session = self.sessions.get(operation.request_key.session_id)
        flow = self._generation()
        if flow.latent_layout is not LatentLayout.PATCH_TOKENS:
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
            operation=operation,
            session=session,
            weights=self._weights(),
            stage=stage,
            route=self._route(stage),
            row=row,
            entry=self.kv.get(operation.request_key.session_id),
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
            discard_handles=(),
        )
        return self._non_state_outcome(operation, completion_tasks=(image_task,))

    def _defer_image_encoding(
        self,
        operation: Operation,
        image: torch.Tensor,
        value_range: ImageRange,
        scope: _ExecutionScope,
        *,
        max_bytes: int,
        discard_handles: tuple[int, ...],
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
        session_id = operation.request_key.session_id
        epoch = int(operation.request_key.epoch)
        handle = _stable_handle(session_id, operation.request_key.epoch, 0, "frames")

        def publish(value: bytes) -> None:
            session = self.sessions.peek(session_id)
            if session is None or int(session.epoch) != epoch:
                return
            self.products.append_encoded_frame(session_id, handle, value.decode("ascii"))
            session.product_handles.add(handle)

        def discard() -> None:
            if not discard_handles:
                return
            self.products.release(discard_handles)
            self.sessions.discard_product_handles(set(discard_handles))

        return _CompletionImagePayload(
            capture,
            reservation,
            max_bytes,
            publish,
            discard,
        )

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
        locator = self.transport.publish_async(value.detach().contiguous())
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
        operation: Operation,
        scope: _ExecutionScope,
    ) -> tuple[torch.Tensor, Mapping[str, object]]:
        for reference in operation.inputs:
            if reference.storage_class is StorageClass.DEVICE_TENSOR:
                read = self.products.device_products.consume(
                    reference,
                    consumer_op_id=operation.op_id,
                    device=self._operation_device(operation),
                )
                scope.device_reads.append(read)
                return read.tensor, {"payload_kind": reference.kind.value}
            record = scope.product_view.get(self._input_product_handle(reference))
            if record is None:
                continue
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
            if record.locator and self.transport is not None:
                transfer = scope.prepared_transfers.get(reference)
                if transfer is None or not transfer.ready():
                    raise capability_mismatch(
                        "cross-stage product transfer has no query-ready prepared ticket"
                    )
                tensors = transfer.tensors()
                if not tensors or not isinstance(tensors[0], torch.Tensor):
                    raise invalid_descriptor("product transport returned a non-tensor value")
                return tensors[0], Locator.from_wire_json(record.locator).meta
        raise invalid_descriptor("transfer product is not resident or transport-addressable")


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


def _binding_identity(tasks: Sequence[_ForwardTask]) -> int:
    digest = hashlib.sha256(b"uniserve-forward-binding\0")
    for task in tasks:
        digest.update(str(task.route).encode("utf-8"))
        digest.update(task.row_kind.value.encode("ascii"))
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


def _parent_base_point(operation: Operation, session: RequestSession) -> int:
    """The point index an operation advances from.

    A fixed parent carries the host-observed point index directly. A device
    parent (a device-relay successor rooted on its predecessor's not-yet-observed
    selected point) carries no host point index; its base is the session's
    latest resolved point.
    """

    point = operation.parent.point
    if isinstance(point, FixedPoint):
        return point.point_index
    return session.version


def _parent_semantic(operation: Operation, session: RequestSession) -> str:
    """The parent semantic digest a completion's own semantic digest chains from.

    A fixed parent names it directly; a device parent chains from the session's
    resolved semantic digest.
    """

    point = operation.parent.point
    if isinstance(point, FixedPoint):
        return point.semantic_digest
    return session.resolved_digest


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
    device_products: DeviceProductTable | None = None,
    device_reads: tuple[DeviceProductRead, ...] = (),
    device_continuation: DeviceProductContinuationBatch | None = None,
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
                device_continuation=(
                    device_continuation if len(compatible) == len(tasks) else None
                ),
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
    device_products: DeviceProductTable | None,
    device_reads: tuple[DeviceProductRead, ...],
    device_continuation: DeviceProductContinuationBatch | None,
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
    product_table: DeviceProductTable | None
    if device_continuation is not None:
        if device_products is None:
            raise RuntimeError("device continuation has no product table")
        product_table = device_products
        product_writes = device_continuation.writes
        product_batch = device_continuation.scalar
    else:
        products = tuple(task.token_product for task in tasks)
        bound_products = device_products is not None and all(
            product is not None for product in products
        )
        product_table = cast(DeviceProductTable, device_products) if bound_products else None
        product_writes = (
            tuple(cast(DeviceProductWrite, product) for product in products)
            if product_table is not None
            else ()
        )
        product_batch = (
            product_table.producer_scalar_batch(product_writes)
            if product_table is not None
            else None
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
    grouped_publication = (
        product_table is not None
        and product_batch is not None
        and (not finish_writes or finish_batch is not None)
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
            side_batches: tuple[DeviceProductScalarBatch, ...] = ()
            if finish_batch is not None:
                if device_finish is None:
                    raise RuntimeError("grouped sampling has no device finish values")
                finish_batch.tensor.copy_(
                    _select_device_values(device_finish, finish_indexes),
                )
                side_batches = (finish_batch,)
            if device_continuation is None:
                product_table.publish_scalar_group(
                    (*side_batches, cast(DeviceProductScalarBatch, product_batch)),
                    after_reads=device_reads,
                )
            else:
                product_table.publish_continuation_group(
                    device_continuation,
                    side_batches,
                    after_reads=device_reads,
                )
            device_finish = None
        elif device_continuation is not None:
            product_table.publish_continuation(
                device_continuation,
                None if product_batch is not None else tagged_tokens,
                after_reads=device_reads,
            )
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
    device_products: DeviceProductTable | None,
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
    device_products: DeviceProductTable | None,
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
    # Fold each operation's selected token into its request's device-resident
    # committed penalty base, weighted by whether the point is active and valid
    # so a predicated no-op or an invalid distribution never contributes. The
    # base was already read for this operation's own penalties above, so its own
    # token never penalizes itself; the fold makes it visible to the next
    # operation in the window before this one is host-observed.
    for index, task in enumerate(tasks):
        base = task.penalty_base
        if base is None:
            continue
        weight = (active[index] & task_valid[index]).to(dtype=base.dtype).reshape(1)
        base.scatter_add_(0, task_tokens[index].reshape(1).to(dtype=torch.int64), weight)
    device_finish, continuation_values, _producer_event = _resolve_sampled_finish_values(
        tasks,
        task_tokens,
        task_valid,
        active,
        terminal_finish,
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
        arena = CompletionArena(
            depth=1,
            token_capacity=max(1, int(metadata.numel())),
            devices=((metadata.device,) if metadata.device.type == "cuda" else ()),
        )
        completion = arena.reserve(max(1, count))
    span = _CompletionSampleSpan(completion.capture(metadata), count)
    if owns_completion:
        completion.seal()
        if span.ready():
            for index in range(count):
                span.token(index)
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


def _decode_tagged_token_views(values: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    if not values:
        return ()
    packed = packed_tensor_views(values)
    if packed is None:
        packed = torch.cat(tuple(value.reshape(-1)[:1] for value in values), dim=0)
    decoded = packed.reshape(-1).bitwise_and(TOKEN_VALUE_MASK)
    return tuple(decoded[index : index + 1] for index in range(len(values)))


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


def _publish_sampled_device_values(
    tasks: tuple[_SampleTask, ...],
    product_field: str,
    device_values: torch.Tensor,
    device_products: DeviceProductTable | None,
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


def _resolve_sampled_finish_values(
    tasks: tuple[_SampleTask, ...],
    device_tokens: torch.Tensor,
    valid: torch.Tensor,
    active: torch.Tensor,
    terminal_finish: torch.Tensor,
    device_products: DeviceProductTable | None,
    device_reads: tuple[DeviceProductRead, ...],
) -> tuple[torch.Tensor | None, torch.Tensor, torch.cuda.Event | None]:
    finish_values = _device_finish_values(tasks, device_tokens, valid & active) | (
        terminal_finish.reshape(-1).to(dtype=torch.bool) & valid & active
    )
    continuation_values = active & valid & ~finish_values
    _publish_sampled_device_values(
        tasks,
        "continuation_product",
        continuation_values,
        device_products,
        device_reads,
    )
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
    device_products: DeviceProductTable,
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
    owns_completion = completion is None
    if completion is None:
        arena = CompletionArena(
            depth=1,
            token_capacity=max(1, int(packed.numel())),
            devices=((packed.device,) if packed.device.type == "cuda" else ()),
        )
        completion = arena.reserve(max(1, len(requested_rows)))
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
        batch = _CompletionLogprobBatch(
            completion.capture(packed),
            requested_rows,
            counts,
            requested_ids,
            max_count,
            max_requested,
        )
    if owns_completion:
        completion.seal()
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


def _requires_device_product_binding(reference: ProductRef) -> bool:
    return reference.storage_class is StorageClass.DEVICE_TENSOR or (
        reference.storage_class is StorageClass.LATENT_ARENA
        and reference.kind
        in {
            ProductKind.ARTIFACT,
            ProductKind.LATENT,
            ProductKind.LATENT_FEATURE,
            ProductKind.VISION_FEATURE,
        }
    )


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
