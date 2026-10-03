"""Worker startup information shared with the scheduler.

A rank answers the engine's ``info`` request (`RequestKind.INFO`, served by
`uniserve_worker.service`) with `WorkerInfo.to_mapping()`: its endpoint,
supported calls, scheduling bounds, KV and latent pool capacity, components,
and transfer capabilities. The worker-ipc crate's `WorkerInfo` is the wire
counterpart: the PyO3 extension converts the mapping into it, and its
`validate` runs whenever the crate's codec encodes or decodes the response.
The engine also checks the description against its launch configuration and
derives the capacity the scheduler plans against. `WorkerInfo.__post_init__`
checks the record's internal consistency when the worker builds it in
`uniserve_worker.bootstrap.report`.

The decoding helpers at the end of this module are local variants of those in
`uniserve_worker.protocol.validation`, without their exact-type fast paths.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeVar, cast

from uniserve.model import ConditionTiles
from uniserve_worker.config.deployment import ComponentConfig
from uniserve_worker.errors import invalid_descriptor, unsupported_setup
from uniserve_worker.protocol.call import (
    CallKind,
    ForwardMode,
    MediaCall,
    computation,
)
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
            reader of the group needs. Encoded only for a sliding window and
            decoded as zero when absent.
        sink: With `KvGroupKind.SLIDING_WINDOW`, prefix tokens retained
            outside the window; the unit pool supports only zero. Encoded
            only for a sliding window and decoded as zero when absent.
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
    sink: int
    page_tokens: int
    units_per_page: int
    layer_ids: tuple[int, ...]
    num_kv_heads: int
    total_kv_heads: int
    kv_head_offset: int
    head_dim: int

    def __post_init__(self) -> None:
        """Validate the page shape and this rank's layers and heads."""
        if (
            self.page_tokens < 1
            or self.page_tokens & (self.page_tokens - 1)
            or self.units_per_page < 1
            or not self.layer_ids
            or len(set(self.layer_ids)) != len(self.layer_ids)
            or self.num_kv_heads < 1
            or self.head_dim < 1
            or self.kv_head_offset + self.num_kv_heads > self.total_kv_heads
        ):
            raise invalid_descriptor("worker KV group is invalid")
        if self.kind is KvGroupKind.SLIDING_WINDOW and self.sink:
            raise invalid_descriptor(
                "worker KV group retains sliding-window sink tokens"
            )

    @classmethod
    def from_mapping(cls, value: object, where: str) -> KvGroup:
        """Decode and validate one KV group from its scheduler wire mapping.

        The policy is a nested mapping tagged by its own ``kind`` key, with
        ``window`` and ``sink`` beside the tag for a sliding window.
        """
        data = _map(value, where)
        kind_data = _map(data.get("kind"), f"{where}.kind")
        return cls(
            kind=_enum(
                KvGroupKind, kind_data.get("kind"), f"{where}.kind.kind"
            ),
            window=_uint(kind_data.get("window", 0), f"{where}.kind.window"),
            sink=_uint(kind_data.get("sink", 0), f"{where}.kind.sink"),
            page_tokens=_uint(data.get("page_tokens"), f"{where}.page_tokens"),
            units_per_page=_uint(
                data.get("units_per_page"), f"{where}.units_per_page"
            ),
            layer_ids=tuple(
                _uint(item, f"{where}.layer_ids[{index}]")
                for index, item in enumerate(
                    _seq(data.get("layer_ids"), f"{where}.layer_ids")
                )
            ),
            num_kv_heads=_uint(
                data.get("num_kv_heads"), f"{where}.num_kv_heads"
            ),
            total_kv_heads=_uint(
                data.get("total_kv_heads"), f"{where}.total_kv_heads"
            ),
            kv_head_offset=_uint(
                data.get("kv_head_offset"), f"{where}.kv_head_offset"
            ),
            head_dim=_uint(data.get("head_dim"), f"{where}.head_dim"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Encode the group's policy, page shape and placement."""
        kind: dict[str, object] = {"kind": self.kind.value}
        if self.kind is KvGroupKind.SLIDING_WINDOW:
            kind.update(window=self.window, sink=self.sink)
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

    The pool is `num_units` units of `unit_bytes` bytes each on this rank,
    unit zero being the padding sentinel; every group's pages draw from it.
    """

    num_units: int
    unit_bytes: int
    dtype: str
    groups: tuple[KvGroup, ...]

    def __post_init__(self) -> None:
        """Validate the pool size and the groups' combined page shapes.

        Requires an allocatable unit beyond the sentinel, a positive unit
        size, a dtype, and groups whose page sizes divide the largest one and
        whose layers are distinct.
        """
        if (
            self.num_units < 2
            or self.unit_bytes < 1
            or not self.groups
            or not self.dtype
        ):
            raise invalid_descriptor(
                "worker info declares an incomplete KV unit pool"
            )
        largest = max(group.page_tokens for group in self.groups)
        if any(largest % group.page_tokens for group in self.groups) or any(
            group.units_per_page >= self.num_units for group in self.groups
        ):
            raise invalid_descriptor("worker KV group page shape is invalid")
        layers = [layer for group in self.groups for layer in group.layer_ids]
        if len(set(layers)) != len(layers):
            raise invalid_descriptor("worker KV groups must not repeat a layer")

    @classmethod
    def from_mapping(cls, value: object, where: str) -> KVCacheInfo:
        """Decode and validate the KV unit pool from the wire mapping."""
        data = _map(value, where)
        return cls(
            num_units=_uint(data.get("num_units"), f"{where}.num_units"),
            unit_bytes=_uint(data.get("unit_bytes"), f"{where}.unit_bytes"),
            dtype=_str(data.get("dtype"), f"{where}.dtype"),
            groups=tuple(
                KvGroup.from_mapping(item, f"{where}.groups[{index}]")
                for index, item in enumerate(
                    _seq(data.get("groups"), f"{where}.groups")
                )
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        """Encode the unit pool and its groups for scheduler discovery."""
        return {
            "num_units": self.num_units,
            "unit_bytes": self.unit_bytes,
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

    def __post_init__(self) -> None:
        if not self.name:
            raise invalid_descriptor("component must have a name")
        if len({output.name for output in self.outputs}) != len(self.outputs):
            raise invalid_descriptor("component repeats a tensor result name")

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "component"
    ) -> ComponentInfo:
        """Decode one component and its tensor results from the wire mapping."""
        data = _map(value, where)
        return cls(
            name=_str(data.get("name"), f"{where}.name"),
            config=ComponentConfig.from_dict(
                {
                    key: value
                    for key, value in data.items()
                    if key not in {"name", "outputs"}
                }
            ),
            outputs=tuple(
                OutputInfo.from_mapping(output, f"{where}.outputs[{index}]")
                for index, output in enumerate(
                    _seq(data.get("outputs", ()), f"{where}.outputs")
                )
            ),
        )

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
    lists the only canvases the denoiser generates, empty when it follows the
    model's canvas rule; ``max_sequence_rows`` is the checkpoint's packed
    sequence capacity, ``None`` when the checkpoint sets none;
    ``condition_tiles`` is the whole-tile condition packing of a
    multi-region denoiser, ``None`` for dense packing.
    """

    tasks: tuple[str, ...]
    schedule_points: int
    video_shift: float
    audio_shift: float
    canvases: tuple[tuple[int, int], ...] = ()
    max_sequence_rows: int | None = None
    condition_tiles: ConditionTiles | None = None

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
            "max_sequence_rows": self.max_sequence_rows,
            "condition_tiles": None
            if self.condition_tiles is None
            else {
                "rows": self.condition_tiles.rows,
                "video": list(self.condition_tiles.video),
            },
        }

    @classmethod
    def from_mapping(cls, value: object, where: str) -> VideoDenoiserInfo:
        data = _map(value, where)
        rows = data.get("max_sequence_rows")
        tiles = data.get("condition_tiles")
        if tiles is not None:
            tiles = _map(tiles, f"{where}.condition_tiles")
            video = tuple(
                _uint(size, f"{where}.condition_tiles.video")
                for size in _seq(
                    tiles.get("video", ()), f"{where}.condition_tiles.video"
                )
            )
            try:
                tiles = ConditionTiles(
                    _uint(tiles.get("rows"), f"{where}.condition_tiles.rows"),
                    video,  # type: ignore[arg-type]
                )
            except ValueError as error:
                raise invalid_descriptor(
                    f"{where}.condition_tiles: {error}"
                ) from None
        return cls(
            tasks=tuple(
                _str(task, f"{where}.tasks")
                for task in _seq(data.get("tasks", ()), f"{where}.tasks")
            ),
            schedule_points=_uint(
                data.get("schedule_points"), f"{where}.schedule_points"
            ),
            video_shift=float(data.get("video_shift", 0.0)),
            audio_shift=float(data.get("audio_shift", 0.0)),
            canvases=tuple(
                (
                    _uint(canvas.get("width"), f"{where}.canvases"),
                    _uint(canvas.get("height"), f"{where}.canvases"),
                )
                for canvas in (
                    _map(item, f"{where}.canvases")
                    for item in _seq(
                        data.get("canvases", ()), f"{where}.canvases"
                    )
                )
            ),
            max_sequence_rows=None
            if rows is None
            else _uint(rows, f"{where}.max_sequence_rows"),
            condition_tiles=tiles,
        )


@dataclass(frozen=True, slots=True)
class WorkerInfo:
    """Describes a worker’s capabilities, topology, and resource bounds.

    Attributes:
        queue_depth: Maximum unresolved physical runs; the engine requires it
            to equal the depth it launched the rank with.
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

    model_name: str
    endpoint: WorkerEndpoint
    world_size: int
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
    model_dtype: str = ""
    attention_backend: str = ""
    weight_formats: tuple[str, ...] = ()
    activation_formats: tuple[str, ...] = ()
    # Identity of the loaded checkpoint files, distinct from the resolved
    # execution configuration: a lowercase hex SHA-256, or empty for a model
    # loaded without a checkpoint. The worker-ipc crate's
    # `WorkerInfo::validate` accepts an empty identity only from the weightless
    # stub model.
    checkpoint_identity: str = ""
    components: tuple[ComponentInfo, ...] = ()
    device: str = "cpu"
    transfer_backends: tuple[str, ...] = ("local",)
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
        whose reports the engine's `WorkerGroup` joins a call's results on.
        Cooperative numerical outputs may reside on other ranks. A
        single-rank worker without the component resolves to rank 0.

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

    def __post_init__(self) -> None:
        """Validate the advertisement's internal consistency.

        Covers transfer backends, scheduling bounds, the endpoint rank, the
        KV cache that autoregressive calls require, pool sizes, supported and
        media calls, latent pool completeness, the model name, and the
        checkpoint identity format. The KV layout validates itself in
        `KVCacheInfo`. The worker-ipc crate's `WorkerInfo::validate` checks
        most of the same relations; it also checks the components (unique
        names, membership within the world, and parallel degrees) and
        refuses an empty checkpoint identity from any model but the
        weightless stub.
        """
        if (
            not self.device
            or not self.transfer_backends
            or len(set(self.transfer_backends)) != len(self.transfer_backends)
            or any(
                name not in {"local", "shm", "cuda_vmm", "channel"}
                for name in self.transfer_backends
            )
        ):
            raise invalid_descriptor(
                "worker physical transfer capabilities are incomplete"
            )

        for name in (
            "queue_depth",
            "max_batch_calls",
            "max_batch_tokens",
            "request_slots",
            "max_unresolved_calls",
            "host_lane_capacity",
        ):
            if getattr(self, name) < 1:
                raise invalid_descriptor(f"worker info.{name} must be positive")

        if self.world_size < 1 or self.endpoint.rank >= self.world_size:
            raise invalid_descriptor(
                "endpoint rank must satisfy 0 <= rank < world_size"
            )

        # Every token-model forward reads or extends a request's KV cache.
        requires_kv = any(
            isinstance(variant, ForwardMode) for variant in self.supported_calls
        )
        if requires_kv and self.kv_cache is None:
            raise invalid_descriptor(
                "worker advertises token work without a KV cache"
            )

        for name in (
            "latent_page_units",
            "latent_pages",
            "buffer_pool_bytes",
            "encoder_cache_entries",
            "encoder_entry_bytes",
            "max_prefill_calls",
            "max_decode_calls",
        ):
            if getattr(self, name) < 0:
                raise invalid_descriptor(
                    f"worker info.{name} must not be negative"
                )

        if not self.supported_calls and (
            self.components
            or self.kv_cache is not None
            or self.latent_pages
            or self.media_components
        ):
            raise invalid_descriptor(
                "a collective-only worker cannot advertise request resources"
            )
        if len(set(self.supported_calls)) != len(self.supported_calls):
            raise invalid_descriptor("worker info repeats a call kind")

        if self.media_components:
            # A worker reports the component serving each media call it
            # implements. The video call graph may span workers, a model
            # worker decoding and a host worker encoding and muxing, so its
            # completeness and its diffusion step count are the engine's
            # checks over every worker.
            if any(
                not component or call not in self.supported_calls
                for call, component in self.media_components.items()
            ):
                raise invalid_descriptor(
                    "media component uses an unsupported call"
                )

        # A latent pool is either absent or has at least one usable page
        # beyond the reserved sentinel page.
        has_latent_geometry = bool(self.latent_page_units or self.latent_pages)
        if has_latent_geometry:
            if self.latent_page_units < 1 or self.latent_pages < 2:
                raise invalid_descriptor(
                    "worker info declares incomplete latent pool capacity"
                )

        if not self.model_name:
            raise invalid_descriptor("worker model name is empty")
        if self.checkpoint_identity and not _is_sha256_hex(
            self.checkpoint_identity
        ):
            raise invalid_descriptor(
                "worker checkpoint identity must be a lowercase hex SHA-256"
            )

    @classmethod
    def from_mapping(cls, value: object, where: str = "info") -> WorkerInfo:
        """Decode a worker capability advertisement from IPC data.

        The advertisement is validated during decoding.
        """
        data = _map(value, where)
        return cls(
            media_components={
                _enum(MediaCall, call, f"{where}.media_components"): _str(
                    component, f"{where}.media_components"
                )
                for call, component in _map(
                    data.get("media_components", {}),
                    f"{where}.media_components",
                ).items()
            },
            num_inference_steps=_uint(
                data.get("num_inference_steps", 0),
                f"{where}.num_inference_steps",
            ),
            video_denoiser=None
            if data.get("video_denoiser") is None
            else VideoDenoiserInfo.from_mapping(
                data["video_denoiser"], f"{where}.video_denoiser"
            ),
            model_dtype=_str(
                data.get("model_dtype", ""), f"{where}.model_dtype"
            ),
            attention_backend=_str(
                data.get("attention_backend", ""), f"{where}.attention_backend"
            ),
            activation_formats=tuple(
                _str(value, f"{where}.activation_formats")
                for value in _seq(
                    data.get("activation_formats", ()),
                    f"{where}.activation_formats",
                )
            ),
            weight_formats=tuple(
                _str(value, f"{where}.weight_formats")
                for value in _seq(
                    data.get("weight_formats", ()), f"{where}.weight_formats"
                )
            ),
            checkpoint_identity=_str(
                data.get("checkpoint_identity", ""),
                f"{where}.checkpoint_identity",
            ),
            components=tuple(
                ComponentInfo.from_mapping(item, f"{where}.components[{index}]")
                for index, item in enumerate(
                    _seq(data.get("components", ()), f"{where}.components")
                )
            ),
            model_name=_str(data.get("model_name", ""), f"{where}.model_name"),
            endpoint=WorkerEndpoint.from_mapping(
                data.get("endpoint"), f"{where}.endpoint"
            ),
            device=_str(data.get("device"), f"{where}.device"),
            transfer_backends=tuple(
                _str(name, f"{where}.transfer_backends")
                for name in _seq(
                    data.get("transfer_backends"), f"{where}.transfer_backends"
                )
            ),
            fabric_handles=bool(data.get("fabric_handles", False)),
            world_size=_uint(data.get("world_size"), f"{where}.world_size"),
            supported_calls=tuple(
                computation(item, f"{where}.supported_calls[{index}]")
                for index, item in enumerate(
                    _seq(
                        data.get("supported_calls"),
                        f"{where}.supported_calls",
                    )
                )
            ),
            queue_depth=_uint(data.get("queue_depth"), f"{where}.queue_depth"),
            max_batch_calls=_uint(
                data.get("max_batch_calls"), f"{where}.max_batch_calls"
            ),
            max_batch_tokens=_uint(
                data.get("max_batch_tokens"), f"{where}.max_batch_tokens"
            ),
            max_prefill_calls=_uint(
                data.get("max_prefill_calls", 0),
                f"{where}.max_prefill_calls",
            ),
            max_decode_calls=_uint(
                data.get("max_decode_calls", 0),
                f"{where}.max_decode_calls",
            ),
            request_slots=_uint(
                data.get("request_slots"), f"{where}.request_slots"
            ),
            kv_cache=(
                None
                if data.get("kv_cache") is None
                else KVCacheInfo.from_mapping(
                    data.get("kv_cache"), f"{where}.kv_cache"
                )
            ),
            latent_page_units=_uint(
                data.get("latent_page_units"), f"{where}.latent_page_units"
            ),
            latent_pages=_uint(
                data.get("latent_pages"), f"{where}.latent_pages"
            ),
            buffer_pool_bytes=_uint(
                data.get("buffer_pool_bytes"), f"{where}.buffer_pool_bytes"
            ),
            encoder_cache_entries=_uint(
                data.get("encoder_cache_entries"),
                f"{where}.encoder_cache_entries",
            ),
            encoder_entry_bytes=_uint(
                data.get("encoder_entry_bytes"), f"{where}.encoder_entry_bytes"
            ),
            max_unresolved_calls=_uint(
                data.get("max_unresolved_calls"),
                f"{where}.max_unresolved_calls",
            ),
            host_lane_capacity=_uint(
                data.get("host_lane_capacity"), f"{where}.host_lane_capacity"
            ),
        )

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
            "model_name": self.model_name,
            "endpoint": self.endpoint.to_mapping(),
            "device": self.device,
            "transfer_backends": list(self.transfer_backends),
            "fabric_handles": self.fabric_handles,
            "world_size": self.world_size,
            "model_dtype": self.model_dtype,
            "attention_backend": self.attention_backend,
            "weight_formats": list(self.weight_formats),
            "activation_formats": list(self.activation_formats),
            "checkpoint_identity": self.checkpoint_identity,
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


# Local decoding helpers for the startup description; see the module
# docstring for how they relate to `uniserve_worker.protocol.validation`.
_E = TypeVar("_E", bound=StrEnum)


def _enum(kind: type[_E], value: object, where: str) -> _E:
    """Decode one string-backed enum field from scheduler metadata."""
    if not isinstance(value, str):
        raise invalid_descriptor(f"{where} must be a string")
    try:
        return kind(value)
    except ValueError:
        raise invalid_descriptor(
            f"{where} has unknown value {value!r}"
        ) from None


def _map(value: object, where: str) -> Mapping[str, Any]:
    """Require a metadata field to be a mapping."""
    if not isinstance(value, Mapping):
        raise invalid_descriptor(f"{where} must be a map")
    return cast(Mapping[str, Any], value)


def _seq(value: object, where: str) -> Sequence[Any]:
    """Require a metadata field to be a non-string sequence."""
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise invalid_descriptor(f"{where} must be a list")
    return value


def _uint(value: object, where: str) -> int:
    """Decode a non-negative integer metadata field."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise invalid_descriptor(f"{where} must be a non-negative integer")
    return value


def _optional_uint(value: object, where: str) -> int | None:
    """Decode an optional non-negative integer metadata field."""
    return None if value is None else _uint(value, where)


def _bool(value: object, where: str) -> bool:
    """Require a metadata field to contain a boolean."""
    if not isinstance(value, bool):
        raise invalid_descriptor(f"{where} must be a boolean")
    return value


def _str(value: object, where: str) -> str:
    """Require a metadata field to contain text."""
    if not isinstance(value, str):
        raise invalid_descriptor(f"{where} must be a string")
    return value


def _is_sha256_hex(value: str) -> bool:
    """Report whether text is the lowercase hex form of one SHA-256 digest."""
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "KVCacheInfo",
    "KvGroup",
    "KvGroupKind",
    "RequestKind",
    "ResponseKind",
    "WorkerInfo",
]
