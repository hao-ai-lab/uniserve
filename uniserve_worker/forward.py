"""Immutable values crossing the runner-to-model boundary.

The model receives one :class:`ForwardBatch` and returns one
:class:`ForwardOutput`.  Both sides are closed, row-aligned value algebras;
request identities, scheduling payloads, executable callbacks, and owning
runtime objects are deliberately absent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeAlias, runtime_checkable

import torch


class RouteId(str):
    """Validated physical route identity."""

    def __new__(cls, value: str) -> RouteId:
        if not isinstance(value, str) or not value:
            raise ValueError("route id must be a non-empty string")
        return str.__new__(cls, value)


@runtime_checkable
class KvView(Protocol):
    """Batch-bounded access to system-owned paged KV storage."""

    @property
    def block_size(self) -> int: ...

    @property
    def supports_paged_attention_storage(self) -> bool: ...

    @property
    def base_lens(self) -> Sequence[int]: ...

    def layer_kv(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]: ...

    def append(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None: ...

    def append_varlen(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        row_lengths: Sequence[int],
        *,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        query_offsets: torch.Tensor,
    ) -> None: ...

    def append_packed(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        page_ids: torch.Tensor,
        page_offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class EmptyKvView:
    """Typed absence of KV access for a route."""

    @property
    def block_size(self) -> int:
        return 0

    @property
    def supports_paged_attention_storage(self) -> bool:
        return False

    @property
    def base_lens(self) -> tuple[int, ...]:
        return ()

    def layer_kv(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        del layer
        raise RuntimeError("this forward route has no KV view")

    def append(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        del layer, key, value
        raise RuntimeError("this forward route has no KV view")

    def append_varlen(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        row_lengths: Sequence[int],
        *,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        query_offsets: torch.Tensor,
    ) -> None:
        del layer, key, value, row_lengths, block_table, cache_seqlens, query_offsets
        raise RuntimeError("this forward route has no KV view")

    def append_packed(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        page_ids: torch.Tensor,
        page_offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> None:
        del layer, key, value, page_ids, page_offsets, token_indices
        raise RuntimeError("this forward route has no KV view")


@runtime_checkable
class LatentView(Protocol):
    """Transaction-bounded latent scratch access."""

    def read(self, handle: int) -> torch.Tensor: ...

    def write(self, handle: int, value: torch.Tensor) -> None: ...


@dataclass(frozen=True, slots=True)
class EmptyLatentView:
    """Typed absence of latent access for a route."""

    def read(self, handle: int) -> torch.Tensor:
        del handle
        raise RuntimeError("this forward route has no latent view")

    def write(self, handle: int, value: torch.Tensor) -> None:
        del handle, value
        raise RuntimeError("this forward route has no latent view")


@runtime_checkable
class MeshView(Protocol):
    """The collectives available to one immutable route topology."""

    def all_reduce(self, value: torch.Tensor, axis: str) -> torch.Tensor: ...

    def all_gather(self, value: torch.Tensor, axis: str, dimension: int) -> torch.Tensor: ...

    def dispatch(self, value: torch.Tensor, axis: str, coordinate: int) -> torch.Tensor: ...

    def combine(
        self,
        value: torch.Tensor,
        axis: str,
        coordinate: int,
        target: torch.device,
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class EmptyMeshView:
    """Single-rank mesh implementation."""

    def all_reduce(self, value: torch.Tensor, axis: str) -> torch.Tensor:
        del axis
        return value

    def all_gather(self, value: torch.Tensor, axis: str, dimension: int) -> torch.Tensor:
        del axis, dimension
        return value

    def dispatch(self, value: torch.Tensor, axis: str, coordinate: int) -> torch.Tensor:
        del axis, coordinate
        return value

    def combine(
        self,
        value: torch.Tensor,
        axis: str,
        coordinate: int,
        target: torch.device,
    ) -> torch.Tensor:
        del axis, coordinate
        return value.to(target) if value.device != target else value


@dataclass(frozen=True, slots=True)
class WrittenRange:
    """A validated span written through an :class:`OutputView`."""

    slot: int
    begin: int
    end: int

    def __post_init__(self) -> None:
        if self.slot < 0 or self.begin < 0 or self.end < self.begin:
            raise ValueError("output write range is invalid")


@runtime_checkable
class OutputView(Protocol):
    """Bounded transaction scratch for large model outputs."""

    def write(self, slot: int, value: torch.Tensor) -> WrittenRange: ...


@dataclass(frozen=True, slots=True)
class EmptyOutputView:
    """Typed absence of output scratch for a route."""

    def write(self, slot: int, value: torch.Tensor) -> WrittenRange:
        del slot, value
        raise RuntimeError("this forward route has no output view")


@dataclass(frozen=True, slots=True)
class GraphBinding:
    """Opaque per-capture identity with no executable behavior."""

    identity: int


@runtime_checkable
class AttentionBackend(Protocol):
    """One explicitly provisioned attention implementation."""

    name: str

    def capabilities(self) -> object: ...


@dataclass(frozen=True, slots=True)
class AttentionSelection:
    """Immutable ordered backend set resolved at worker startup."""

    identity: str
    providers: tuple[AttentionBackend, ...]

    def __post_init__(self) -> None:
        if not self.identity or not self.providers:
            raise ValueError("attention selection requires an identity and providers")
        names = tuple(str(provider.name) for provider in self.providers)
        if len(set(names)) != len(names):
            raise ValueError("attention selection contains duplicate providers")


@dataclass(frozen=True, slots=True)
class NoAttention:
    """Typed dense/no-paged-attention plan."""

    backends: AttentionSelection


@dataclass(frozen=True, slots=True)
class PagedDecodePlan:
    """One query token per row against paged KV."""

    backends: AttentionSelection
    block_table: torch.Tensor
    cache_seqlens: torch.Tensor
    kv_seqlens: torch.Tensor
    query_lens: torch.Tensor
    cache_seqlens_cpu: tuple[int, ...]
    kv_seqlens_cpu: tuple[int, ...]
    query_lens_cpu: tuple[int, ...]
    decode_page_ids: torch.Tensor
    decode_page_offsets: torch.Tensor
    max_context_len: int
    causal: bool
    binding: GraphBinding


@dataclass(frozen=True, slots=True)
class PagedVarlenPlan:
    """Ragged paged attention over explicit query and key spans."""

    backends: AttentionSelection
    block_table: torch.Tensor
    cache_seqlens: torch.Tensor
    query_lens: torch.Tensor
    kv_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    output_indices: torch.Tensor
    cache_seqlens_cpu: tuple[int, ...]
    query_lens_cpu: tuple[int, ...]
    kv_seqlens_cpu: tuple[int, ...]
    max_seqlen_q: int
    max_seqlen_k: int
    max_context_len: int
    causal: bool
    binding: GraphBinding


@dataclass(frozen=True, slots=True)
class PackedAttentionPlan:
    """Visible-segment attention for a mixed token/flow route."""

    backends: AttentionSelection
    indexes: torch.Tensor
    route_indicators: torch.Tensor
    text_indices: torch.Tensor
    has_text: bool
    has_flow: bool
    visible_end: torch.Tensor
    cu_seqlens_q: torch.Tensor
    page_table: torch.Tensor
    seqused_k: torch.Tensor
    write_page_ids: torch.Tensor
    write_page_offsets: torch.Tensor
    write_token_indices: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    use_prefix_bounds: bool
    fully_visible: bool
    binding: GraphBinding


AttnPlan: TypeAlias = NoAttention | PagedDecodePlan | PagedVarlenPlan | PackedAttentionPlan


@dataclass(frozen=True, slots=True)
class ForwardContext:
    """The complete bounded capability set for one model call."""

    kv: KvView | EmptyKvView
    latent: LatentView | EmptyLatentView
    attention: AttnPlan
    mesh: MeshView | EmptyMeshView
    output: OutputView | EmptyOutputView


class TokenSelection(StrEnum):
    LAST_LOGITS = "last_logits"
    ALL_LOGITS = "all_logits"
    HIDDEN = "hidden"


@dataclass(frozen=True, slots=True)
class TokenIds:
    values: torch.Tensor

    def __post_init__(self) -> None:
        if self.values.dtype not in (torch.int32, torch.int64):
            raise ValueError("token ids must use an integer dtype")


@dataclass(frozen=True, slots=True)
class TokenEmbeddings:
    values: torch.Tensor

    def __post_init__(self) -> None:
        if not self.values.is_floating_point():
            raise ValueError("token embeddings must use a floating dtype")


TokenSegment: TypeAlias = TokenIds | TokenEmbeddings


@dataclass(frozen=True, slots=True)
class TokenSegments:
    """Ordered token-id and embedding spans forming one logical token row."""

    values: tuple[TokenSegment, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("segmented token input must contain at least one span")


TokenInput: TypeAlias = TokenIds | TokenEmbeddings | TokenSegments


@dataclass(frozen=True, slots=True)
class TokenRow:
    """Autoregressive or multimodal token computation."""

    row_id: int
    inputs: TokenInput
    positions: torch.Tensor
    output_slot: int
    selection: TokenSelection

    def __post_init__(self) -> None:
        _validate_row(self.row_id, self.output_slot)
        if self.positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("token positions must use an integer dtype")


def packed_token_ids(rows: Sequence[TokenRow]) -> torch.Tensor | None:
    """Return the shared contiguous token-id storage behind aligned row views."""

    if not rows or any(not isinstance(row.inputs, TokenIds) for row in rows):
        return None
    return packed_tensor_views(
        tuple(row.inputs.values for row in rows if isinstance(row.inputs, TokenIds))
    )


def packed_token_positions(rows: Sequence[TokenRow]) -> torch.Tensor | None:
    """Return the shared contiguous position storage behind aligned row views."""

    return packed_tensor_views(tuple(row.positions for row in rows))


def packed_tensor_views(values: Sequence[torch.Tensor]) -> torch.Tensor | None:
    """Recover one tensor from ordered contiguous views without copying."""

    if not values:
        return None
    flat = tuple(value.reshape(-1) for value in values)
    first = flat[0]
    if (
        not first.is_contiguous()
        or any(not value.is_contiguous() for value in flat)
        or any(value.dtype != first.dtype or value.device != first.device for value in flat)
    ):
        return None
    storage = first.untyped_storage().data_ptr()
    offset = int(first.storage_offset())
    expected = offset
    for value in flat:
        if value.untyped_storage().data_ptr() != storage or int(value.storage_offset()) != expected:
            return None
        expected += int(value.numel())
    return first.as_strided((expected - offset,), (1,), storage_offset=offset)


@dataclass(frozen=True, slots=True)
class NoFlowConditioning:
    """Typed absence of external conditioning features."""


@dataclass(frozen=True, slots=True)
class FlowPatches:
    """Explicit current-latent patches for a flow branch's neural tower."""

    pixels: torch.Tensor
    grid: torch.Tensor
    noise_scale: torch.Tensor

    def __post_init__(self) -> None:
        if self.pixels.ndim not in (2, 4):
            raise ValueError("flow patches must be flattened rows or NCHW patches")
        if self.grid.ndim != 2 or int(self.grid.shape[1]) != 2:
            raise ValueError("flow patch grid must have shape [images, 2]")
        if self.noise_scale.numel() != 1:
            raise ValueError("flow noise scale must be scalar")


FlowConditioning: TypeAlias = NoFlowConditioning | FlowPatches


@dataclass(frozen=True, slots=True)
class FlowRow:
    """One raw diffusion or rectified-flow branch evaluation."""

    row_id: int
    conditioning: FlowConditioning
    positions: torch.Tensor
    timestep: torch.Tensor
    latent: torch.Tensor
    image_tokens: int
    image_height: int
    image_width: int
    output_slot: int

    def __post_init__(self) -> None:
        _validate_row(self.row_id, self.output_slot)
        if self.image_tokens < 1 or self.image_height < 1 or self.image_width < 1:
            raise ValueError("flow image geometry must be positive")


class EncodeKind(StrEnum):
    VISION = "vision"
    LATENT = "latent"


@dataclass(frozen=True, slots=True)
class PatchInput:
    pixels: torch.Tensor
    grid: torch.Tensor
    # Host-known patch grid (grid_height, grid_width) for this one image, taken
    # from the registration-time image transform. Lets the encoder resolve the
    # per-image conv geometry without reading the device ``grid`` tensor back.
    grid_shape: tuple[int, int]


@dataclass(frozen=True, slots=True)
class TowerInput:
    pixels: torch.Tensor


EncodeInput: TypeAlias = PatchInput | TowerInput


@dataclass(frozen=True, slots=True)
class EncodeRow:
    """Encoder or tower computation over an explicit tensor input."""

    row_id: int
    kind: EncodeKind
    inputs: EncodeInput
    output_slot: int

    def __post_init__(self) -> None:
        _validate_row(self.row_id, self.output_slot)


@dataclass(frozen=True, slots=True)
class DecodeRow:
    """Neural decoding of an explicit latent into a product tensor."""

    row_id: int
    latent: torch.Tensor
    image_height: int
    image_width: int
    output_slot: int

    def __post_init__(self) -> None:
        _validate_row(self.row_id, self.output_slot)
        if self.image_height < 1 or self.image_width < 1:
            raise ValueError("decode image geometry must be positive")


ForwardRow: TypeAlias = TokenRow | FlowRow | EncodeRow | DecodeRow


@dataclass(frozen=True, slots=True)
class ForwardBatch:
    """The sole immutable argument accepted by a concrete model."""

    route: RouteId
    rows: tuple[ForwardRow, ...]
    context: ForwardContext

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("forward batch must contain at least one row")
        row_ids = tuple(row.row_id for row in self.rows)
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("forward row ids must be unique")
        slots = tuple(row.output_slot for row in self.rows)
        if len(set(slots)) != len(slots):
            raise ValueError("forward output slots must be unique")


@dataclass(frozen=True, slots=True)
class TokenLogits:
    value: torch.Tensor


@dataclass(frozen=True, slots=True)
class TokenHidden:
    value: torch.Tensor


TokenValue: TypeAlias = TokenLogits | TokenHidden


@dataclass(frozen=True, slots=True)
class TokenOutput:
    row_id: int
    output_slot: int
    value: TokenValue


@dataclass(frozen=True, slots=True)
class FlowOutput:
    row_id: int
    output_slot: int
    prediction: torch.Tensor


@dataclass(frozen=True, slots=True)
class EncodeOutput:
    row_id: int
    output_slot: int
    features: torch.Tensor


@dataclass(frozen=True, slots=True)
class DecodeOutput:
    row_id: int
    output_slot: int
    tensor: torch.Tensor


ForwardRowOutput: TypeAlias = TokenOutput | FlowOutput | EncodeOutput | DecodeOutput


@dataclass(frozen=True, slots=True)
class ForwardOutput:
    """Ordered raw neural outputs aligned with :class:`ForwardBatch.rows`."""

    rows: tuple[ForwardRowOutput, ...]

    def validate_for(self, batch: ForwardBatch) -> None:
        if len(self.rows) != len(batch.rows):
            raise ValueError("model output count does not match forward rows")
        for input_row, output_row in zip(batch.rows, self.rows, strict=True):
            if output_row.row_id != input_row.row_id:
                raise ValueError("model output row id does not match its input row")
            if output_row.output_slot != input_row.output_slot:
                raise ValueError("model output slot does not match its input row")
            if not _matching_output(input_row, output_row):
                raise ValueError("model output variant does not match its input row")


def _validate_row(row_id: int, output_slot: int) -> None:
    if row_id < 0 or output_slot < 0:
        raise ValueError("row id and output slot must be non-negative")


def _matching_output(input_row: ForwardRow, output_row: ForwardRowOutput) -> bool:
    return (
        isinstance(input_row, TokenRow)
        and isinstance(output_row, TokenOutput)
        or isinstance(input_row, FlowRow)
        and isinstance(output_row, FlowOutput)
        or isinstance(input_row, EncodeRow)
        and isinstance(output_row, EncodeOutput)
        or isinstance(input_row, DecodeRow)
        and isinstance(output_row, DecodeOutput)
    )


__all__ = [
    "AttnPlan",
    "DecodeOutput",
    "DecodeRow",
    "EmptyKvView",
    "EmptyLatentView",
    "EmptyMeshView",
    "EmptyOutputView",
    "EncodeInput",
    "EncodeKind",
    "EncodeOutput",
    "EncodeRow",
    "FlowOutput",
    "FlowConditioning",
    "FlowPatches",
    "FlowRow",
    "ForwardBatch",
    "ForwardContext",
    "ForwardOutput",
    "ForwardRow",
    "ForwardRowOutput",
    "GraphBinding",
    "KvView",
    "LatentView",
    "MeshView",
    "NoAttention",
    "NoFlowConditioning",
    "OutputView",
    "PackedAttentionPlan",
    "PagedDecodePlan",
    "PagedVarlenPlan",
    "PatchInput",
    "RouteId",
    "TokenEmbeddings",
    "TokenHidden",
    "TokenIds",
    "TokenInput",
    "TokenLogits",
    "TokenOutput",
    "TokenRow",
    "TokenSegment",
    "TokenSegments",
    "TokenSelection",
    "TowerInput",
    "WrittenRange",
    "packed_tensor_views",
    "packed_token_ids",
    "packed_token_positions",
]
