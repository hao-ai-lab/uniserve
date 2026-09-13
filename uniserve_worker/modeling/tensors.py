"""Borrowed numerical tensors, attention geometry, and vocabulary partitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import torch

TensorViews = Mapping[str, torch.Tensor]


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
    The caller retains backing tensors until all numerical readers finish.
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


def concatenate_views(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Borrow adjacent numerical columns, or concatenate disjoint allocations."""

    tensors = tuple(value.reshape(-1) for value in values)
    packed = packed_tensor_views(tensors)
    return torch.cat(tensors, dim=0) if packed is None else packed


class PositionLayout(StrEnum):
    """Selects temporal-only or temporal-spatial position coordinates."""

    TEMPORAL = "temporal"
    TEMPORAL_SPATIAL = "temporal_spatial"


class ImageRange(StrEnum):
    """Defines whether image values use signed-unit or unit numeric range."""

    SIGNED_UNIT = "signed_unit"
    UNIT = "unit"
