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
from typing import Protocol, runtime_checkable

import torch

from ..nn.mesh import CollectiveAxisTransport, DeviceMesh, PeerAxisTransport


@runtime_checkable
class LatentView(Protocol):
    """Partition-bounded latent scratch access."""

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


class RouteMeshView:
    """Expose only the mesh axes declared by one physical model route."""

    def __init__(self, mesh: DeviceMesh, axes: tuple[str, ...]) -> None:
        self._mesh = mesh
        self._axes = frozenset(axes)
        if any(not axis for axis in self._axes):
            raise ValueError("mesh axis names must not be empty")

    def all_reduce(self, value: torch.Tensor, axis: str) -> torch.Tensor:
        transport = self._transport(axis)
        if transport is None:
            return value
        if not isinstance(transport, CollectiveAxisTransport):
            raise RuntimeError(f"mesh axis {axis!r} does not support all-reduce")
        return transport.all_reduce(value)

    def all_gather(self, value: torch.Tensor, axis: str, dimension: int) -> torch.Tensor:
        transport = self._transport(axis)
        if transport is None:
            return value
        if not isinstance(transport, CollectiveAxisTransport):
            raise RuntimeError(f"mesh axis {axis!r} does not support all-gather")
        return transport.all_gather(value, dimension)

    def dispatch(self, value: torch.Tensor, axis: str, coordinate: int) -> torch.Tensor:
        transport = self._transport(axis)
        if transport is None:
            return value
        if not isinstance(transport, PeerAxisTransport):
            raise RuntimeError(f"mesh axis {axis!r} does not support peer dispatch")
        return transport.copy_to(value, coord=int(coordinate))

    def combine(
        self,
        value: torch.Tensor,
        axis: str,
        coordinate: int,
        target: torch.device,
    ) -> torch.Tensor:
        self._require_axis(axis)
        del coordinate
        return value if value.device == target else value.to(target, non_blocking=True)

    def _transport(self, axis: str):
        self._require_axis(axis)
        if self._mesh.is_trivial(axis):
            return None
        return self._mesh.transport(axis)

    def _require_axis(self, axis: str) -> None:
        if axis not in self._axes:
            raise RuntimeError(f"mesh axis {axis!r} is outside this forward route")


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
    """Bounded execution scratch for large model outputs."""

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


class ForwardMode(StrEnum):
    DENSE = "dense"
    PAGED_DECODE = "paged_decode"
    PAGED_VARLEN = "paged_varlen"
    PACKED = "packed"
    REQUEST_INDEXED_DECODE = "request_indexed_decode"


class ExpertRoute(StrEnum):
    TEXT = "text"
    FLOW = "flow"


@dataclass(frozen=True, slots=True)
class RouteSpan:
    """One contiguous expert span in packed attention token order."""

    route: ExpertRoute
    token_start: int
    token_count: int

    def __post_init__(self) -> None:
        if self.token_start < 0 or self.token_count < 1:
            raise ValueError("packed expert span geometry is invalid")

    @property
    def token_end(self) -> int:
        return self.token_start + self.token_count


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
    forward_mode: ForwardMode
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    query_lens: torch.Tensor
    out_cache_loc: torch.Tensor
    has_cache_writes: bool = True
    block_table: torch.Tensor | None = None
    kv_lens: torch.Tensor | None = None
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
    seq_lens_cpu: tuple[int, ...] = ()
    query_lens_cpu: tuple[int, ...] = ()
    kv_lens_cpu: tuple[int, ...] = ()
    group_id: int = 0
    fully_visible: bool = False
    binding: int = 0
    cuda_graph_capture: bool = False
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
    mesh: MeshView | EmptyMeshView = EmptyMeshView()
    output: OutputView | EmptyOutputView = EmptyOutputView()

    def __post_init__(self) -> None:
        if self.row_count < 1:
            raise ValueError("forward batch must contain at least one row")
        if int(self.req_pool_indices.numel()) != self.row_count:
            raise ValueError("forward request indices do not align with rows")
        if int(self.seq_lens.numel()) != self.row_count or int(self.query_lens.numel()) != self.row_count:
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
        if len(self.seq_lens_cpu) != self.row_count or len(self.query_lens_cpu) != self.row_count:
            raise ValueError("forward host KV lengths do not align with rows")
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

    @property
    def request_pool_indices(self) -> torch.Tensor:
        return self.req_pool_indices


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
    "EmptyLatentView",
    "EmptyMeshView",
    "EmptyOutputView",
    "EncodeKind",
    "ExpertRoute",
    "FlowPatches",
    "ForwardBatch",
    "ForwardMode",
    "ForwardOutput",
    "LatentView",
    "MeshView",
    "OutputView",
    "RouteMeshView",
    "RouteSpan",
    "ModelPhase",
    "TokenSelection",
    "WrittenRange",
    "packed_tensor_views",
]
