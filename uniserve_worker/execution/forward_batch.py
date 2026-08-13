"""Ephemeral tensor views crossing the runner-to-model boundary.

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
    binding: int


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
    binding: int


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
    binding: int
    query_lens_cpu: tuple[int, ...]
    key_lens_cpu: tuple[int, ...]


AttnPlan: TypeAlias = NoAttention | PagedDecodePlan | PagedVarlenPlan | PackedAttentionPlan


class TokenSelection(StrEnum):
    LAST_LOGITS = "last_logits"
    ALL_LOGITS = "all_logits"
    HIDDEN = "hidden"


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


class EncodeKind(StrEnum):
    VISION = "vision"
    LATENT = "latent"


class ModelPhase(StrEnum):
    TEXT = "text"
    DENOISE = "denoise"
    ENCODE_VISION = "encode_vision"
    ENCODE_LATENT = "encode_latent"
    DECODE_LATENT = "decode_latent"


@dataclass(frozen=True, slots=True)
class ForwardBatch:
    """One borrowed columnar view over execution-partition input buffers."""

    phase: ModelPhase
    row_count: int
    request_pool_indices: torch.Tensor
    attention: AttnPlan
    token_row_indices: tuple[int, ...] = ()
    flow_row_indices: tuple[int, ...] = ()
    input_ids: torch.Tensor | None = None
    input_embeddings: torch.Tensor | None = None
    embedding_mask: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    query_lens: tuple[int, ...] = ()
    token_selections: tuple[TokenSelection, ...] = ()
    flow_positions: tuple[torch.Tensor, ...] = ()
    flow_timesteps: tuple[torch.Tensor, ...] = ()
    flow_latents: tuple[torch.Tensor, ...] = ()
    flow_conditioning: tuple[FlowPatches | None, ...] = ()
    flow_image_tokens: tuple[int, ...] = ()
    flow_heights: tuple[int, ...] = ()
    flow_widths: tuple[int, ...] = ()
    encode_pixels: tuple[torch.Tensor, ...] = ()
    encode_grids: tuple[torch.Tensor | None, ...] = ()
    encode_grid_shapes: tuple[tuple[int, int] | None, ...] = ()
    decode_latents: tuple[torch.Tensor, ...] = ()
    decode_heights: tuple[int, ...] = ()
    decode_widths: tuple[int, ...] = ()
    kv: KvView | EmptyKvView = EmptyKvView()
    mesh: MeshView | EmptyMeshView = EmptyMeshView()
    output: OutputView | EmptyOutputView = EmptyOutputView()

    def __post_init__(self) -> None:
        if self.row_count < 1:
            raise ValueError("forward batch must contain at least one row")
        if int(self.request_pool_indices.numel()) != self.row_count:
            raise ValueError("forward request indices do not align with rows")
        row_indices = (*self.token_row_indices, *self.flow_row_indices)
        if row_indices and (
            len(set(row_indices)) != len(row_indices)
            or min(row_indices) < 0
            or max(row_indices) >= self.row_count
        ):
            raise ValueError("forward row indexes are invalid")
        if len(self.token_row_indices) != len(self.query_lens) or len(
            self.token_row_indices
        ) != len(self.token_selections):
            raise ValueError("forward token columns are not aligned")
        flow_count = len(self.flow_row_indices)
        if any(
            len(values) != flow_count
            for values in (
                self.flow_positions,
                self.flow_timesteps,
                self.flow_latents,
                self.flow_conditioning,
                self.flow_image_tokens,
                self.flow_heights,
                self.flow_widths,
            )
        ):
            raise ValueError("forward flow columns are not aligned")
        encode_count = len(self.encode_pixels)
        if any(
            len(values) != encode_count for values in (self.encode_grids, self.encode_grid_shapes)
        ):
            raise ValueError("forward encoder columns are not aligned")
        decode_count = len(self.decode_latents)
        if any(len(values) != decode_count for values in (self.decode_heights, self.decode_widths)):
            raise ValueError("forward decoder columns are not aligned")


@dataclass(frozen=True, slots=True)
class ForwardOutput:
    """Ordered raw tensors aligned with a :class:`ForwardBatch`."""

    values: tuple[torch.Tensor, ...]

    def validate_for(self, batch: ForwardBatch) -> None:
        if len(self.values) != batch.row_count:
            raise ValueError("model output count does not match forward rows")
        if any(not isinstance(value, torch.Tensor) for value in self.values):
            raise TypeError("model output values must be tensors")


__all__ = [
    "AttnPlan",
    "EmptyKvView",
    "EmptyLatentView",
    "EmptyMeshView",
    "EmptyOutputView",
    "EncodeKind",
    "FlowPatches",
    "ForwardBatch",
    "ForwardOutput",
    "KvView",
    "LatentView",
    "MeshView",
    "NoAttention",
    "OutputView",
    "PackedAttentionPlan",
    "PagedDecodePlan",
    "PagedVarlenPlan",
    "ModelPhase",
    "TokenSelection",
    "WrittenRange",
    "packed_tensor_views",
]
