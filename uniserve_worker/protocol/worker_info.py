"""Worker startup information shared with the scheduler.

A rank answers the engine's ``info`` request (`RequestKind.INFO`, served by
`uniserve_worker.service`) with `WorkerInfo.to_mapping()`: its endpoint,
supported calls, scheduling bounds, KV and latent pool capacity, and
components. The worker-ipc crate's `WorkerInfo` is the wire counterpart: the
PyO3 extension converts the mapping into it, and the crate's codec validates
the report when it encodes it. The engine derives the capacity the scheduler
plans against from the reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from uniserve_worker.config.deployment import ComponentConfig
from uniserve_worker.errors import unsupported_setup
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
    """A loaded component's configuration and its publishable tensor results.

    On the wire the `ComponentConfig` fields sit beside ``name`` and
    ``outputs`` in one flat mapping, as the worker-ipc crate's
    `ComponentInfo` flattens them. The order of `outputs` defines product
    output indices.
    """

    name: str
    config: ComponentConfig
    outputs: tuple[OutputInfo, ...] = ()

    def to_mapping(self) -> dict[str, object]:
        """Encode component membership and results for IPC discovery."""
        return {
            "name": self.name,
            **self.config.to_dict(),
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
        queue_depth: Maximum unresolved physical runs, the depth the rank was
            launched with.
        max_batch_calls: Maximum calls in one run.
        max_batch_tokens: Maximum text tokens represented in one run.
        max_prefill_calls: Maximum calls in one prefill run, the rows the
            worker's captured prefill graphs hold; a prefill run no graph
            holds fails. Zero leaves prefill runs bounded by
            ``max_batch_calls`` alone, as when they run eagerly.
        max_decode_calls: Maximum calls in one decode run, the rows the
            worker's largest captured decode graph holds, fewer when the KV
            pool cannot hold a page of every cache group for more rows; a
            decode run no graph holds fails. Zero leaves decode runs bounded
            by ``max_batch_calls`` alone, as when they run eagerly.
        request_slots: Number of resident request slots.
        latent_page_units: Model-defined units stored in one latent page.
        latent_pages: Physical latent pages, including the reserved sentinel
            page that `latent_capacity_units` excludes.
        buffer_pool_bytes: Persistent buffer-pool capacity in bytes.
        max_unresolved_calls: Maximum unresolved calls per request.
        host_lane_capacity: Concurrent host-lane tasks this rank admits.
    """

    endpoint: WorkerEndpoint
    supported_calls: tuple[CallKind, ...]
    queue_depth: int
    max_batch_calls: int
    max_batch_tokens: int
    request_slots: int
    kv_cache: KVCacheInfo | None
    latent_page_units: int
    latent_pages: int
    buffer_pool_bytes: int
    max_unresolved_calls: int
    host_lane_capacity: int
    encoder_cache_entries: int = 0
    encoder_entry_bytes: int = 0
    max_prefill_calls: int = 0
    max_decode_calls: int = 0
    components: tuple[ComponentInfo, ...] = ()
    # Physical ranks of this worker, which `output_rank` needs for a worker
    # without a placed component. The engine knows its own placement, so the
    # report does not carry it.
    world_size: int = 1
    # Whether this rank's device exports a handle another host can import.
    # A descriptor handle reaches only this host, so the engine refuses a
    # cross-host CUDA VMM transfer edge unless both ends report fabric
    # handles.
    fabric_handles: bool = False
    media_components: dict[MediaCall, str] = field(default_factory=dict)
    num_inference_steps: int = 0
    # What the deployment's video denoiser serves; ``None`` without one.
    video_denoiser: VideoDenoiserInfo | None = None

    def output_rank(self, component: str) -> int:
        """Return the rank that publishes a component's host products.

        Host products belong to the component's first member rank, the rank
        whose reports the engine's `WorkerGroup` joins a call's results on
        whenever the call runs there. A call whose units are dealt to other
        ranks publishes no host product, and the engine joins it on the
        first rank it is dealt to. Cooperative numerical outputs may reside
        on other ranks. A single-rank worker without the component resolves
        to rank 0.

        Raises:
            WorkerError: `unsupported_setup` when the component is not
                configured and the worker has more than one rank.
        """
        for candidate in self.components:
            if candidate.name == component:
                return candidate.config.ranks[0]
        if self.world_size == 1:
            return 0
        raise unsupported_setup(
            f"component {component!r} has no publication owner"
        )

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
            "num_inference_steps": self.num_inference_steps,
            "video_denoiser": None
            if self.video_denoiser is None
            else self.video_denoiser.to_mapping(),
            "endpoint": self.endpoint.to_mapping(),
            "fabric_handles": self.fabric_handles,
            "components": [
                component.to_mapping() for component in self.components
            ],
            "supported_calls": [value.value for value in self.supported_calls],
            "queue_depth": self.queue_depth,
            "max_batch_calls": self.max_batch_calls,
            "max_batch_tokens": self.max_batch_tokens,
            "max_prefill_calls": self.max_prefill_calls,
            "max_decode_calls": self.max_decode_calls,
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
        }


__all__ = [
    "KVCacheInfo",
    "KvGroup",
    "KvGroupKind",
    "RequestKind",
    "ResponseKind",
    "WorkerInfo",
]
