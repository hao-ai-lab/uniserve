"""Worker startup information shared with the scheduler."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeVar, cast

from uniserve_worker.protocol.call import (
    VIDEO_STAGES,
    CallKind,
    ForwardMode,
    PipelineStage,
    computation,
)

from ..foundation.errors import invalid_descriptor, unsupported_setup
from ..protocol.tensor import OutputInfo
from ..protocol.transfer import WorkerEndpoint
from .config import ComponentConfig


class RequestKind(StrEnum):
    """Defines IPC request verbs.

    Verbs cover discovery, submission, and shutdown.
    """

    INFO = "info"
    SUBMIT = "submit"
    CLOSE = "close"


class ResponseKind(StrEnum):
    """Defines IPC response categories.

    Categories cover worker information, results, acknowledgements, and
    errors.
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
    """Describe the page count and optional window of one KV cache group."""

    num_blocks: int
    kind: KvGroupKind
    window: int
    sink: int

    @classmethod
    def from_mapping(cls, value: object, where: str) -> KvGroup:
        """Decode and validate one KV group from its scheduler wire mapping."""
        data = _map(value, where)
        kind_data = _map(data.get("kind"), f"{where}.kind")
        return cls(
            num_blocks=_uint(data.get("num_blocks"), f"{where}.num_blocks"),
            kind=_enum(
                KvGroupKind, kind_data.get("kind"), f"{where}.kind.kind"
            ),
            window=_uint(kind_data.get("window", 0), f"{where}.kind.window"),
            sink=_uint(kind_data.get("sink", 0), f"{where}.kind.sink"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Encode KV group settings for the scheduler wire format.

        Covers full-context and sliding-window settings.
        """
        kind: dict[str, object] = {"kind": self.kind.value}
        if self.kind is KvGroupKind.SLIDING_WINDOW:
            kind.update(window=self.window, sink=self.sink)
        return {
            "num_blocks": self.num_blocks,
            "kind": kind,
        }


@dataclass(frozen=True, slots=True)
class KVCacheInfo:
    """Publish KV cache layout to the scheduler.

    Covers block size, dtype, token capacity, and group layout.
    """

    block_size: int
    num_blocks: int
    num_layers: int
    total_layers: int
    layer_offset: int
    num_kv_heads: int
    total_kv_heads: int
    kv_head_offset: int
    head_dim: int
    bytes_per_token: int
    groups: tuple[KvGroup, ...]
    dtype: str

    def __post_init__(self) -> None:
        """Validate published KV dimensions.

        Also validate dtype, token capacity, and group coverage.
        """
        if (
            min(
                self.block_size,
                self.num_blocks,
                self.num_layers,
                self.num_kv_heads,
                self.head_dim,
                self.bytes_per_token,
            )
            < 1
            or not self.groups
            or not self.dtype
        ):
            raise invalid_descriptor(
                "worker info declares incomplete KV dimensions"
            )
        if (
            self.kv_head_offset < 0
            or self.kv_head_offset + self.num_kv_heads > self.total_kv_heads
        ):
            raise invalid_descriptor(
                "worker KV head interval exceeds its logical bounds"
            )
        if (
            self.layer_offset < 0
            or self.layer_offset + self.num_layers > self.total_layers
        ):
            raise invalid_descriptor(
                "worker KV layer interval exceeds its logical bounds"
            )
        if any(group.num_blocks < 1 for group in self.groups):
            raise invalid_descriptor(
                "worker info KV groups must be physical page partitions"
            )
        if sum(group.num_blocks for group in self.groups) != self.num_blocks:
            raise invalid_descriptor(
                "worker info KV groups must cover the physical page pool"
            )

    @classmethod
    def from_mapping(cls, value: object, where: str) -> KVCacheInfo:
        """Decode and validate physical KV dimensions from the wire mapping."""
        data = _map(value, where)
        return cls(
            block_size=_uint(data.get("block_size"), f"{where}.block_size"),
            num_blocks=_uint(data.get("num_blocks"), f"{where}.num_blocks"),
            num_layers=_uint(data.get("num_layers"), f"{where}.num_layers"),
            total_layers=_uint(
                data.get("total_layers"), f"{where}.total_layers"
            ),
            layer_offset=_uint(
                data.get("layer_offset"), f"{where}.layer_offset"
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
            bytes_per_token=_uint(
                data.get("bytes_per_token"), f"{where}.bytes_per_token"
            ),
            groups=tuple(
                KvGroup.from_mapping(item, f"{where}.groups[{index}]")
                for index, item in enumerate(
                    _seq(data.get("groups"), f"{where}.groups")
                )
            ),
            dtype=_str(data.get("dtype"), f"{where}.dtype"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Encode KV dimensions and page groups for scheduler discovery."""
        return {
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "num_layers": self.num_layers,
            "total_layers": self.total_layers,
            "layer_offset": self.layer_offset,
            "num_kv_heads": self.num_kv_heads,
            "total_kv_heads": self.total_kv_heads,
            "kv_head_offset": self.kv_head_offset,
            "head_dim": self.head_dim,
            "bytes_per_token": self.bytes_per_token,
            "groups": [group.to_mapping() for group in self.groups],
            "dtype": self.dtype,
        }


@dataclass(frozen=True, slots=True)
class EntryInfo:
    """Loaded entry membership and publishable tensor results.

    The entry's computation publishes these tensor results.
    """

    name: str
    config: ComponentConfig
    outputs: tuple[OutputInfo, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise invalid_descriptor("entry must have a name")
        if len({output.name for output in self.outputs}) != len(self.outputs):
            raise invalid_descriptor("entry repeats a tensor result name")

    @classmethod
    def from_mapping(cls, value: object, where: str = "entry") -> EntryInfo:
        """Decode one entry and its tensor results from the wire mapping."""
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
        """Encode entry membership and publishable results for IPC discovery."""
        return {
            "name": self.name,
            **self.config.to_dict(),
            "outputs": [output.to_mapping() for output in self.outputs],
        }


@dataclass(frozen=True, slots=True)
class WorkerInfo:
    """Describes a worker’s capabilities, topology, and resource bounds."""

    model_name: str
    endpoint: WorkerEndpoint
    world_size: int
    supported_ops: tuple[CallKind, ...]
    queue_depth: int
    max_batch_ops: int
    max_batch_tokens: int
    request_slots: int
    kv_cache: KVCacheInfo | None
    latent_page_units: int
    latent_pages: int
    buffer_pool_bytes: int
    max_unresolved_ops: int
    host_lane_capacity: int
    encoder_cache_entries: int = 0
    encoder_entry_bytes: int = 0
    configuration_id: str = ""
    components: tuple[EntryInfo, ...] = ()
    device: str = "cpu"
    transfer_backends: tuple[str, ...] = ("local",)
    # Whether this rank's device exports a handle another host can import.
    # A descriptor handle reaches only this host, so the engine refuses a
    # transfer edge that would have to cross one.
    fabric_handles: bool = False
    pipeline_components: dict[PipelineStage, str] = field(default_factory=dict)
    num_inference_steps: int = 0

    def output_rank(self, entry: str) -> int:
        """Resolve the host publication owner from the call's entry.

        Cooperative numerical outputs may reside on different stages. Host
        products belong to the entry's first member, matching the rank-report
        join requirements. An unconfigured local worker has one possible owner.
        """
        for component in self.components:
            if component.name == entry:
                return component.config.ranks[0]
        if self.world_size == 1:
            return 0
        raise unsupported_setup(
            f"computation entry {entry!r} has no publication owner"
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
        """Validate advertised worker info.

        Covers topology, capacities, variants, and cache-group layout.
        """
        if (
            not self.device
            or not self.transfer_backends
            or len(set(self.transfer_backends)) != len(self.transfer_backends)
            or any(
                name not in {"local", "shm", "cuda_vmm"}
                for name in self.transfer_backends
            )
        ):
            raise invalid_descriptor(
                "worker physical transfer capabilities are incomplete"
            )

        for name in (
            "queue_depth",
            "max_batch_ops",
            "max_batch_tokens",
            "request_slots",
            "max_unresolved_ops",
            "host_lane_capacity",
        ):
            if getattr(self, name) < 1:
                raise invalid_descriptor(f"worker info.{name} must be positive")

        if self.world_size < 1 or self.endpoint.rank >= self.world_size:
            raise invalid_descriptor(
                "endpoint rank must satisfy 0 <= rank < world_size"
            )

        requires_kv = any(
            variant
            in {ForwardMode.PREFILL, ForwardMode.DECODE, ForwardMode.VERIFY}
            for variant in self.supported_ops
        )
        if requires_kv and self.kv_cache is None:
            raise invalid_descriptor(
                "worker advertises AR work without a KV cache"
            )

        for name in (
            "latent_page_units",
            "latent_pages",
            "buffer_pool_bytes",
            "encoder_cache_entries",
            "encoder_entry_bytes",
        ):
            if getattr(self, name) < 0:
                raise invalid_descriptor(
                    f"worker info.{name} must not be negative"
                )

        if not self.supported_ops:
            raise invalid_descriptor("worker info must support a work variant")
        if len(set(self.supported_ops)) != len(self.supported_ops):
            raise invalid_descriptor("worker info repeats a work variant")

        if self.pipeline_components:
            if self.num_inference_steps < 1 or set(
                self.pipeline_components
            ) != set(VIDEO_STAGES):
                raise invalid_descriptor(
                    "video components or diffusion step count are incomplete"
                )
            if any(
                not component or stage not in self.supported_ops
                for stage, component in self.pipeline_components.items()
            ):
                raise invalid_descriptor(
                    "pipeline component uses an unsupported call"
                )

        has_latent_geometry = bool(self.latent_page_units or self.latent_pages)
        if has_latent_geometry:
            if self.latent_page_units < 1 or self.latent_pages < 2:
                raise invalid_descriptor(
                    "worker info declares incomplete latent pool capacity"
                )

        if not self.model_name:
            raise invalid_descriptor("worker model name is empty")

    @classmethod
    def from_mapping(cls, value: object, where: str = "info") -> WorkerInfo:
        """Decode a worker capability advertisement from IPC data.

        The advertisement is validated during decoding.
        """
        data = _map(value, where)
        return cls(
            pipeline_components={
                _enum(
                    PipelineStage, stage, f"{where}.pipeline_components"
                ): _str(component, f"{where}.pipeline_components")
                for stage, component in _map(
                    data.get("pipeline_components", {}),
                    f"{where}.pipeline_components",
                ).items()
            },
            num_inference_steps=_uint(
                data.get("num_inference_steps", 0),
                f"{where}.num_inference_steps",
            ),
            configuration_id=str(data.get("configuration_id", "")),
            components=tuple(
                EntryInfo.from_mapping(item, f"{where}.components[{index}]")
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
            supported_ops=tuple(
                computation(item, f"{where}.supported_ops[{index}]")
                for index, item in enumerate(
                    _seq(data.get("supported_ops"), f"{where}.supported_ops")
                )
            ),
            queue_depth=_uint(data.get("queue_depth"), f"{where}.queue_depth"),
            max_batch_ops=_uint(
                data.get("max_batch_ops"), f"{where}.max_batch_ops"
            ),
            max_batch_tokens=_uint(
                data.get("max_batch_tokens"), f"{where}.max_batch_tokens"
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
            max_unresolved_ops=_uint(
                data.get("max_unresolved_ops"), f"{where}.max_unresolved_ops"
            ),
            host_lane_capacity=_uint(
                data.get("host_lane_capacity"), f"{where}.host_lane_capacity"
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        """Encode worker capabilities and resource bounds for IPC discovery."""
        return {
            "pipeline_components": {
                stage.value: component
                for stage, component in self.pipeline_components.items()
            },
            "num_inference_steps": self.num_inference_steps,
            "model_name": self.model_name,
            "endpoint": self.endpoint.to_mapping(),
            "device": self.device,
            "transfer_backends": list(self.transfer_backends),
            "fabric_handles": self.fabric_handles,
            "world_size": self.world_size,
            "configuration_id": self.configuration_id,
            "components": [
                component.to_mapping() for component in self.components
            ],
            "supported_ops": [value.value for value in self.supported_ops],
            "queue_depth": self.queue_depth,
            "max_batch_ops": self.max_batch_ops,
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
            "max_unresolved_ops": self.max_unresolved_ops,
            "host_lane_capacity": self.host_lane_capacity,
        }


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


__all__ = [
    "KVCacheInfo",
    "KvGroup",
    "KvGroupKind",
    "RequestKind",
    "ResponseKind",
    "WorkerInfo",
]
