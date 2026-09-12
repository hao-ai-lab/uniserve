"""Ephemeral tensor views crossing the runner-to-model boundary.

The model receives one :class:`ForwardBatch` and returns one
:class:`ForwardOutput`.  Both sides are closed, row-aligned value algebras;
request identities, scheduling payloads, executable callbacks, and owning
runtime objects are deliberately absent.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING

import torch

from uniserve_worker.protocol.batch import ForwardMode, ForwardStats, PipelineStage

if TYPE_CHECKING:
    from ..backends.attention.base import AttentionBackend
    from .sampling import SamplerOutput


@dataclass(frozen=True, slots=True)
class AttentionSelection:
    """Immutable ordered backend set resolved at worker startup."""

    identity: str
    providers: tuple[AttentionBackend, ...]

    def __post_init__(self) -> None:
        """Require a non-empty ordered set of distinct attention backends."""

        if not self.identity or not self.providers:
            raise ValueError("attention selection requires an identity and providers")
        names = tuple(str(provider.name) for provider in self.providers)
        if len(set(names)) != len(names):
            raise ValueError("attention selection contains duplicate providers")


class AttentionMode(StrEnum):
    """Selects dense, paged decode, paged prefill, or packed attention execution."""

    DENSE = "dense"
    PAGED_DECODE = "paged_decode"
    PAGED_VARLEN = "paged_varlen"
    PACKED = "packed"


class ExpertRoute(StrEnum):
    """Selects text or diffusion-flow experts for routed transformer tokens."""

    TEXT = "text"
    FLOW = "flow"


@dataclass(frozen=True, slots=True)
class RouteSpan:
    """One contiguous expert span in packed attention token order."""

    route: ExpertRoute
    token_start: int
    token_count: int

    def __post_init__(self) -> None:
        """Validate a non-empty half-open span for one expert route."""

        if self.token_start < 0 or self.token_count < 1:
            raise ValueError("packed expert span geometry is invalid")

    @property
    def token_end(self) -> int:
        """Give the exclusive packed-token boundary of this expert route."""

        return self.token_start + self.token_count


class TokenSelection(StrEnum):
    """Selects final-token logits, all-token logits, or hidden states from a model forward."""

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
        """Validate flow patch tensor rank, grid geometry, and token count."""

        if self.pixels.ndim not in (2, 4):
            raise ValueError("flow patches must be flattened rows or NCHW patches")
        if self.grid.ndim != 2 or int(self.grid.shape[1]) != 2:
            raise ValueError("flow patch grid must have shape [images, 2]")
        if self.noise_scale.numel() != 1:
            raise ValueError("flow noise scale must be scalar")


@dataclass(frozen=True, slots=True)
class AttentionMetadata:
    """Borrowed numerical geometry consumed independently by attention backends.

    Prefix lengths exclude the current query. Paged modes materialize total
    lengths; packed attention reads the prefix and query segments separately.
    Capture retains stable tensor addresses and a backend binding identity.
    """

    attention_mode: AttentionMode
    prefix_lens: torch.Tensor
    query_lens: torch.Tensor
    out_cache_loc: torch.Tensor
    has_cache_writes: bool = True
    block_table: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    output_indices: torch.Tensor | None = None
    attention_indexes: torch.Tensor | None = None
    visible_end: torch.Tensor | None = None
    route_spans: tuple[RouteSpan, ...] = ()
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    causal: bool = True
    causal_rows_cpu: tuple[bool, ...] = ()
    prefix_lens_cpu: tuple[int, ...] = ()
    query_lens_cpu: tuple[int, ...] = ()
    seq_lens_cpu: tuple[int, ...] = ()
    group_id: int = 0
    fully_visible: bool = False
    binding: int = 0
    cuda_graph_capture: bool = False


@dataclass(frozen=True, slots=True)
class ForwardBatch:
    """One borrowed columnar view over execution-lane input buffers."""

    forward_mode: ForwardMode | PipelineStage
    row_count: int
    attention: AttentionMetadata
    request_pool_indices: torch.Tensor
    decode_force_finish: torch.Tensor | None = None
    token_row_indices: tuple[int, ...] = ()
    flow_row_indices: tuple[int, ...] = ()
    input_ids: torch.Tensor | None = None
    input_embeddings: torch.Tensor | None = None
    embedding_mask: torch.Tensor | None = None
    positions: torch.Tensor | None = None
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

    def __post_init__(self) -> None:
        """Validate borrowed batch columns against attention mode and row geometry."""

        if self.row_count < 1:
            raise ValueError("forward batch must contain at least one row")
        if int(self.request_pool_indices.numel()) != self.row_count:
            raise ValueError("forward request indices do not align with rows")
        if (
            int(self.attention.prefix_lens.numel()) != self.row_count
            or int(self.attention.query_lens.numel()) != self.row_count
        ):
            raise ValueError("forward KV lengths do not align with rows")
        if self.decode_force_finish is not None and (
            int(self.decode_force_finish.numel()) != self.row_count
            or self.decode_force_finish.dtype is not torch.bool
        ):
            raise ValueError("forward decode finish column does not align with rows")
        row_indices = (*self.token_row_indices, *self.flow_row_indices)
        if row_indices and (
            len(set(row_indices)) != len(row_indices)
            or min(row_indices) < 0
            or max(row_indices) >= self.row_count
        ):
            raise ValueError("forward row indexes are invalid")
        if len(self.token_row_indices) != len(self.token_selections):
            raise ValueError("forward token columns are not aligned")
        if any(
            len(lengths) != self.row_count
            for lengths in (
                self.attention.prefix_lens_cpu,
                self.attention.seq_lens_cpu,
                self.attention.query_lens_cpu,
            )
        ):
            raise ValueError("forward host KV lengths do not align with rows")
        if any(
            prefix < 0 or query < 0 or total != prefix + query
            for prefix, query, total in zip(
                self.attention.prefix_lens_cpu,
                self.attention.query_lens_cpu,
                self.attention.seq_lens_cpu,
                strict=True,
            )
        ):
            raise ValueError("forward sequence lengths must equal cached prefix plus query")
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
class VocabularyPartition:
    """Logical vocabulary layout and non-owning collective identity.

    ``rank`` is the logical vocabulary partition. ``backend_order`` maps each
    physical collective rank to its logical partition, including reordered TP
    groups. Padding belongs to storage and lies outside ``vocab_size``.
    """

    vocab_size: int
    width: int
    rank: int
    backend_order: tuple[int, ...]
    group_name: str | None

    def __post_init__(self) -> None:
        size = len(self.backend_order)
        if (
            self.width < 1
            or not 0 < self.vocab_size <= min(self.width * size, 2**53)
            or not 0 <= self.rank < size
            or sorted(self.backend_order) != list(range(size))
            or (size > 1 and self.group_name is None)
        ):
            raise ValueError("vocabulary partition has invalid geometry or collective membership")


@dataclass(frozen=True, slots=True)
class ForwardOutput:
    """Ordered raw tensors and optional vocabulary sharding aligned with a batch.

    Each result has its own optional vocabulary partition. Hidden states and
    continuous predictions have no vocabulary, including in mixed batches.
    Consumers needing global logits call ``materialize`` collectively;
    distributed selection can consume the local text rows directly.
    """

    values: tuple[torch.Tensor, ...]
    vocabularies: tuple[VocabularyPartition | None, ...] = ()
    request_pool_indices: torch.Tensor | None = None
    output_event: torch.cuda.Event | None = None
    stats: ForwardStats | None = None
    greedy: SamplerOutput | None = None

    def __post_init__(self) -> None:
        if not self.vocabularies:
            object.__setattr__(self, "vocabularies", (None,) * len(self.values))
        if len(self.vocabularies) != len(self.values):
            raise ValueError("vocabulary metadata must align with output rows")
        for value, partition in zip(self.values, self.vocabularies, strict=True):
            if partition is not None and (value.ndim != 2 or value.shape[-1] != partition.width):
                raise ValueError("vocabulary output rows disagree with their partition")

    def materialize(self) -> ForwardOutput:
        """Gather global vocabulary rows, preserving their caller-visible shapes."""

        if self.output_event is not None:
            if not self.values:
                raise RuntimeError("forward output has a fence without a producer tensor")
            torch.cuda.current_stream(self.values[0].device).wait_event(self.output_event)
        if not any(self.vocabularies):
            return self
        from ..nn.logits import gather_vocabulary

        groups: dict[VocabularyPartition, list[int]] = defaultdict(list)
        for index, partition in enumerate(self.vocabularies):
            if partition is not None:
                groups[partition].append(index)
        values = list(self.values)
        for partition, indexes in groups.items():
            sources = tuple(self.values[index] for index in indexes)
            rows = packed_tensor_views(sources)
            if rows is None:
                rows = torch.cat(sources, dim=0)
            gathered = gather_vocabulary(rows.reshape(-1, partition.width), partition)
            results = gathered.split(tuple(value.shape[0] for value in sources))
            for index, value in zip(indexes, results, strict=True):
                values[index] = value
        return replace(self, values=tuple(values), vocabularies=(None,) * len(values))

    def clone(self) -> ForwardOutput:
        """Own detached copies that survive reuse of the producer's storage.

        Outputs on one device with one dtype share a packed allocation. Shapes
        and logical tensor values are preserved independently of source strides;
        storage remains live for as long as any returned tensor is retained.
        """

        if self.output_event is not None:
            if not self.values:
                raise RuntimeError("forward output has a fence without a producer tensor")
            torch.cuda.current_stream(self.values[0].device).wait_event(self.output_event)
        groups: dict[tuple[torch.device, torch.dtype], list[int]] = defaultdict(list)
        for index, value in enumerate(self.values):
            groups[(value.device, value.dtype)].append(index)
        copied = list(self.values)
        for indexes in groups.values():
            if len(indexes) == 1:
                index = indexes[0]
                copied[index] = self.values[index].detach().clone()
                continue
            sources = [self.values[index] for index in indexes]
            packed = torch.cat(tuple(value.detach().reshape(-1) for value in sources))
            views = packed.split(tuple(value.numel() for value in sources))
            for index, view in zip(indexes, views, strict=True):
                copied[index] = view.reshape(self.values[index].shape)
        greedy = self.greedy
        if greedy is not None:
            greedy = greedy.clone()
        return replace(
            self,
            values=tuple(copied),
            output_event=None,
            greedy=greedy,
            request_pool_indices=None
            if self.request_pool_indices is None
            else self.request_pool_indices.clone(),
        )

    def validate_for(self, batch: ForwardBatch) -> None:
        """Require one tensor result for every row in the originating batch."""

        if len(self.values) != batch.row_count:
            raise ValueError("model output count does not match forward rows")
        if any(not isinstance(value, torch.Tensor) for value in self.values):
            raise TypeError("model output values must be tensors")


__all__ = [
    "AttentionMode",
    "ExpertRoute",
    "FlowPatches",
    "ForwardBatch",
    "ForwardOutput",
    "RouteSpan",
    "TokenSelection",
    "VocabularyPartition",
    "packed_tensor_views",
]


def concatenate_views(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Borrow adjacent numerical columns, or concatenate disjoint allocations."""

    tensors = tuple(value.reshape(-1) for value in values)
    packed = packed_tensor_views(tensors)
    return torch.cat(tensors, dim=0) if packed is None else packed
