"""Worker startup information shared with the scheduler.

A rank answers the engine's ``info`` request (`RequestKind.INFO`, served by
`uniserve_worker.service`) with `WorkerInfo.to_mapping()`: its endpoint,
supported calls, scheduling bounds, KV and latent pool capacity, and the
tensor results of its components. The worker-ipc crate's `WorkerInfo` is the
wire counterpart: the PyO3 extension converts the mapping into it, and the
crate's codec validates the report when it encodes it. The engine derives the
capacity the scheduler plans against from the reports. Values the engine
launched the worker with, such as component placement and queue depth, are
the engine's and are not reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from uniserve_worker.protocol.call import CallKind, MediaCall
from uniserve_worker.protocol.tensor import OutputInfo
from uniserve_worker.protocol.transfer import WorkerEndpoint


class RequestKind(StrEnum):
    """IPC request kinds a worker serves.

    ``info`` asks for the startup description, ``submit`` carries a batch,
    and ``close`` shuts the worker down.
    """

    INFO = "info"
    SUBMIT = "submit"
    CLOSE = "close"


class ResponseKind(StrEnum):
    """IPC response kinds a worker sends.

    ``info`` carries the `WorkerInfo` mapping, ``result`` a batch result,
    ``ok`` the acknowledgement of ``close``, and ``error`` a failure.
    """

    INFO = "info"
    RESULT = "result"
    OK = "ok"
    ERROR = "error"


class KvGroupKind(StrEnum):
    """Distinguishes full-context and sliding-window KV cache groups."""

    FULL = "full"
    SLIDING_WINDOW = "sliding_window"


@dataclass(frozen=True, slots=True)
class KvGroup:
    """Describe one cache group of a worker's KV unit pool.

    A logical page of the group holds ``page_tokens`` tokens of every layer
    in the group and occupies ``units_per_page`` units of the pool.

    Attributes:
        kind: History retention policy of the group.
        window: With `KvGroupKind.SLIDING_WINDOW`, the history tokens any
            reader of the group needs. Encoded only for a sliding window.
        page_tokens: Tokens per logical page; a power of two.
        units_per_page: Units one logical page occupies.
        layer_ids: Global cache-layer ids of this rank's layers in the
            group, in column order.
        num_kv_heads: KV heads this rank stores per layer.
        total_kv_heads: Logical KV heads of each layer.
        kv_head_offset: First logical KV head stored by this rank.
        head_dim: Elements per KV head.
    """

    kind: KvGroupKind
    window: int
    page_tokens: int
    units_per_page: int
    layer_ids: tuple[int, ...]
    num_kv_heads: int
    total_kv_heads: int
    kv_head_offset: int
    head_dim: int

    def to_mapping(self) -> dict[str, object]:
        """Encode the group's policy, page shape and placement."""
        kind: dict[str, object] = {"kind": self.kind.value}
        if self.kind is KvGroupKind.SLIDING_WINDOW:
            kind.update(window=self.window)
        return {
            "kind": kind,
            "page_tokens": self.page_tokens,
            "units_per_page": self.units_per_page,
            "layer_ids": list(self.layer_ids),
            "num_kv_heads": self.num_kv_heads,
            "total_kv_heads": self.total_kv_heads,
            "kv_head_offset": self.kv_head_offset,
            "head_dim": self.head_dim,
        }


@dataclass(frozen=True, slots=True)
class KVCacheInfo:
    """Publish the KV unit pool to the scheduler.

    The pool is `num_units` units on this rank, unit zero being the padding
    sentinel; every group's pages draw from it.
    """

    num_units: int
    dtype: str
    groups: tuple[KvGroup, ...]

    def to_mapping(self) -> dict[str, object]:
        """Encode the unit pool and its groups for scheduler discovery."""
        return {
            "num_units": self.num_units,
            "dtype": self.dtype,
            "groups": [group.to_mapping() for group in self.groups],
        }


@dataclass(frozen=True, slots=True)
class ComponentInfo:
    """A loaded component and its publishable tensor results.

    The order of `outputs` defines product output indices.
    """

    name: str
    outputs: tuple[OutputInfo, ...] = ()

    def to_mapping(self) -> dict[str, object]:
        """Encode the component's results for IPC discovery."""
        return {
            "name": self.name,
            "outputs": [output.to_mapping() for output in self.outputs],
        }


@dataclass(frozen=True, slots=True)
class VideoDenoiserInfo:
    """What the video denoiser a deployment places serves.

    Mirrors the Rust ``VideoDenoiserInfo``. ``schedule_points`` counts the
    fixed schedule's sigma points, the clean endpoint included; ``canvases``
    lists the only canvases the deployment serves, those it prepares.
    """

    tasks: tuple[str, ...]
    schedule_points: int
    video_shift: float
    audio_shift: float
    canvases: tuple[tuple[int, int], ...] = ()

    def to_mapping(self) -> dict[str, object]:
        return {
            "tasks": list(self.tasks),
            "schedule_points": self.schedule_points,
            "video_shift": self.video_shift,
            "audio_shift": self.audio_shift,
            "canvases": [
                {"width": width, "height": height}
                for width, height in self.canvases
            ],
        }


@dataclass(frozen=True, slots=True)
class WorkerInfo:
    """Describes a worker’s capabilities, topology, and resource bounds.

    Attributes:
        max_batch_calls: Maximum calls in one batch.
        max_batch_tokens: Maximum text tokens represented in one batch.
        request_slots: Number of resident request slots.
        latent_page_units: Model-defined units stored in one latent page.
        latent_pages: Physical latent pages, including the reserved sentinel
            page that `latent_capacity_units` excludes.
        buffer_pool_bytes: Persistent buffer-pool capacity in bytes.
        max_unresolved_calls: Maximum unresolved calls per request.
        host_lane_capacity: Concurrent host-lane tasks this rank admits.
        transfer_tickets: Transfer entries this rank's transport holds in
            flight at once; the engine keeps no more transfer calls reserved
            than the narrowest rank holds.
    """

    endpoint: WorkerEndpoint
    supported_calls: tuple[CallKind, ...]
    max_batch_calls: int
    max_batch_tokens: int
    request_slots: int
    kv_cache: KVCacheInfo | None
    latent_page_units: int
    latent_pages: int
    buffer_pool_bytes: int
    max_unresolved_calls: int
    host_lane_capacity: int
    transfer_tickets: int
    encoder_cache_entries: int = 0
    encoder_entry_bytes: int = 0
    components: tuple[ComponentInfo, ...] = ()
    # Whether this rank's device exports a handle another host can import.
    # A descriptor handle reaches only this host, so the engine refuses a
    # cross-host CUDA VMM transfer edge unless both ends report fabric
    # handles.
    fabric_handles: bool = False
    media_components: dict[MediaCall, str] = field(default_factory=dict)
    # What the deployment's video denoiser serves; ``None`` without one.
    video_denoiser: VideoDenoiserInfo | None = None

    @property
    def latent_capacity_units(self) -> int:
        """Report allocatable latent units after reserving the sentinel page."""
        return max(0, self.latent_pages - 1) * self.latent_page_units

    @property
    def uses_kv(self) -> bool:
        """Indicate whether this worker publishes a physical KV cache."""
        return self.kv_cache is not None

    def to_mapping(self) -> dict[str, object]:
        """Encode worker capabilities and resource bounds for IPC discovery."""
        return {
            "media_components": {
                call.value: component
                for call, component in self.media_components.items()
            },
            "video_denoiser": None
            if self.video_denoiser is None
            else self.video_denoiser.to_mapping(),
            "endpoint": self.endpoint.to_mapping(),
            "fabric_handles": self.fabric_handles,
            "components": [
                component.to_mapping() for component in self.components
            ],
            "supported_calls": [value.value for value in self.supported_calls],
            "max_batch_calls": self.max_batch_calls,
            "max_batch_tokens": self.max_batch_tokens,
            "request_slots": self.request_slots,
            "kv_cache": None
            if self.kv_cache is None
            else self.kv_cache.to_mapping(),
            "latent_page_units": self.latent_page_units,
            "latent_pages": self.latent_pages,
            "buffer_pool_bytes": self.buffer_pool_bytes,
            "encoder_cache_entries": self.encoder_cache_entries,
            "encoder_entry_bytes": self.encoder_entry_bytes,
            "max_unresolved_calls": self.max_unresolved_calls,
            "host_lane_capacity": self.host_lane_capacity,
            "transfer_tickets": self.transfer_tickets,
        }


__all__ = [
    "KVCacheInfo",
    "KvGroup",
    "KvGroupKind",
    "RequestKind",
    "ResponseKind",
    "WorkerInfo",
]
