"""Typed process arguments for one Python worker rank."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import prod

from uniserve.loading import Config as IOConfig
from uniserve_worker.protocol.call import (
    CallKind,
    ForwardMode,
    MediaCall,
    TransferMode,
)

from ..config import WorkerConfig, worker_config_from_namespace

# These launch selectors assign capabilities to a worker pool. A media
# selector includes both tracks; submitted call kinds still identify the
# concrete call.
SUPPORTED_CALL_GROUPS: dict[str, tuple[CallKind, ...]] = {
    "ar_extend": (ForwardMode.PREFILL,),
    "ar_decode": (ForwardMode.DECODE,),
    "ar_verify": (ForwardMode.VERIFY,),
    "encoder_vision": (MediaCall.VISION_ENCODING,),
    "encoder_latent": (MediaCall.LATENT_ENCODING,),
    "encoder_text": (MediaCall.TEXT_ENCODING,),
    "transfer_product": (TransferMode.TENSOR,),
    "transfer_kv_publish": (TransferMode.KV_PUBLISH,),
    "transfer_kv_install": (TransferMode.KV_INSTALL,),
    "diffusion_prepare": (MediaCall.LATENT_PREPARATION,),
    "diffusion_step": (MediaCall.DENOISING,),
    "diffusion_finalize": (MediaCall.IMAGE_DECODING, MediaCall.MUXING),
    "diffusion_decode": (
        MediaCall.VIDEO_DECODING,
        MediaCall.AUDIO_DECODING,
    ),
    "media_append": (
        MediaCall.VIDEO_ENCODING,
        MediaCall.AUDIO_ENCODING,
    ),
}


def _positive_degree(name: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


_SEQUENCE_FIELDS = {
    "local": (),
    "ulysses": ("ulysses_degree",),
    "allgather": ("allgather_degree",),
    "hybrid": ("ulysses_degree", "allgather_degree"),
}


@dataclass(frozen=True, slots=True)
class SequenceConfig:
    """Select the sequence-parallel algorithm and its active degrees."""

    kind: str = "local"
    degrees: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        names = _SEQUENCE_FIELDS.get(self.kind)
        if names is None:
            raise ValueError(
                f"unknown sequence parallel strategy {self.kind!r}"
            )
        if not isinstance(self.degrees, tuple) or len(self.degrees) != len(
            names
        ):
            raise ValueError(
                f"sequence strategy {self.kind!r} requires degrees {names!r}"
            )

        for name, degree in zip(names, self.degrees, strict=True):
            _positive_degree(name, degree)

    @property
    def size(self) -> int:
        return prod(self.degrees)

    @property
    def dimensions(self) -> tuple[tuple[str, int], ...]:
        """Return context axes followed by the Ulysses axis."""
        match self.kind:
            case "local":
                return (("cp", 1), ("ulysses", 1))
            case "ulysses":
                return (("cp", 1), ("ulysses", self.degrees[0]))
            case "allgather":
                return (("cp", self.degrees[0]), ("ulysses", 1))
            case "hybrid":
                return (("cp", self.degrees[1]), ("ulysses", self.degrees[0]))

        raise ValueError(f"unknown sequence parallel strategy {self.kind!r}")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SequenceConfig:
        kind = value.get("kind")
        if not isinstance(kind, str) or kind not in _SEQUENCE_FIELDS:
            raise ValueError(f"unknown sequence parallel strategy {kind!r}")

        names = _SEQUENCE_FIELDS[kind]
        unexpected = value.keys() - {"kind", *names}
        if unexpected:
            raise ValueError(
                f"sequence strategy {kind!r} has unknown fields: "
                f"{sorted(unexpected)}"
            )

        return cls(
            kind,
            tuple(_positive_degree(name, value.get(name, 1)) for name in names),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            **dict(zip(_SEQUENCE_FIELDS[self.kind], self.degrees, strict=True)),
        }


@dataclass(frozen=True, slots=True)
class ParallelConfig:
    """Configure tensor, pipeline and sequence parallelism of one component."""

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    sequence_parallel: SequenceConfig = SequenceConfig()

    def __post_init__(self) -> None:
        _positive_degree("tensor_parallel_size", self.tensor_parallel_size)
        _positive_degree("pipeline_parallel_size", self.pipeline_parallel_size)
        if not isinstance(self.sequence_parallel, SequenceConfig):
            raise ValueError(
                "sequence_parallel must be a sequence configuration"
            )

    @property
    def sequence_parallel_size(self) -> int:
        return self.sequence_parallel.size

    @property
    def world_size(self) -> int:
        return (
            self.tensor_parallel_size
            * self.pipeline_parallel_size
            * self.sequence_parallel_size
        )

    @property
    def dimensions(self) -> tuple[tuple[str, int], ...]:
        """Return mesh axes in rank order, with Ulysses varying fastest."""
        *context, ulysses = self.sequence_parallel.dimensions
        return (
            ("pp", self.pipeline_parallel_size),
            *context,
            ("tp", self.tensor_parallel_size),
            ulysses,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ParallelConfig:
        unexpected = value.keys() - {field.name for field in fields(cls)}
        if unexpected:
            raise ValueError(
                f"unknown parallel_config fields: {sorted(unexpected)}"
            )

        sequence = value.get("sequence_parallel", {"kind": "local"})
        if not isinstance(sequence, Mapping):
            raise ValueError("sequence_parallel must be an object")

        return cls(
            tensor_parallel_size=_positive_degree(
                "tensor_parallel_size", value.get("tensor_parallel_size", 1)
            ),
            pipeline_parallel_size=_positive_degree(
                "pipeline_parallel_size", value.get("pipeline_parallel_size", 1)
            ),
            sequence_parallel=SequenceConfig.from_dict(sequence),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "sequence_parallel": self.sequence_parallel.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ComponentConfig:
    """Configure rank placement and parallelism for one model component."""

    ranks: tuple[int, ...]
    parallel_config: ParallelConfig = ParallelConfig()
    distribution: str | None = None
    units_per_rank: int = 1

    def __post_init__(self) -> None:
        if not self.ranks or len(set(self.ranks)) != len(self.ranks):
            raise ValueError("component ranks must be unique and non-empty")
        if any(type(rank) is not int or rank < 0 for rank in self.ranks):
            raise ValueError("component ranks must be nonnegative integers")
        if self.distribution not in (None, "temporal_units"):
            raise ValueError(
                f"unsupported component distribution {self.distribution!r}"
            )
        if type(self.units_per_rank) is not int or self.units_per_rank < 1:
            raise ValueError("units_per_rank must be a positive integer")
        if (
            self.distribution is None
            and len(self.ranks) != self.parallel_config.world_size
        ):
            raise ValueError(
                "component membership must equal "
                "TP × sequence × pipeline degrees"
            )
        if (
            self.distribution is not None
            and self.parallel_config.world_size != 1
        ):
            raise ValueError(
                "temporal unit distribution requires local decoder parallelism"
            )

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> ComponentConfig:
        unknown = value.keys() - {
            "ranks",
            "parallel_config",
            "distribution",
            "units_per_rank",
        }
        if unknown:
            raise ValueError(f"unknown component fields: {sorted(unknown)}")

        ranks = value.get("ranks")
        parallel = value.get("parallel_config", {})
        if not isinstance(ranks, (list, tuple)) or not isinstance(
            parallel, dict
        ):
            raise ValueError(
                "component requires ranks and a parallel_config object"
            )

        distribution = value.get("distribution")
        units = value.get("units_per_rank", 1)
        if distribution is not None and not isinstance(distribution, str):
            raise ValueError("component distribution must be a string")
        if type(units) is not int:
            raise ValueError("units_per_rank must be a positive integer")

        return cls(
            tuple(ranks),
            ParallelConfig.from_dict(parallel),
            distribution,
            units,
        )

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "ranks": list(self.ranks),
            "parallel_config": self.parallel_config.to_dict(),
        }
        if self.distribution is not None:
            value.update(
                distribution=self.distribution,
                units_per_rank=self.units_per_rank,
            )
        return value


def parse_components(
    value: dict[str, object], world_size: int
) -> tuple[tuple[str, ComponentConfig], ...]:
    """Parse model component placement supplied by the worker launcher."""
    if not value:
        raise ValueError("component configuration must not be empty")

    components = []
    for name, component in sorted(value.items()):
        if not name or not isinstance(component, dict):
            raise ValueError(
                "component configuration requires named component objects"
            )

        resolved = ComponentConfig.from_dict(component)
        if any(rank >= world_size for rank in resolved.ranks):
            raise ValueError(
                f"component {name!r} contains ranks outside the process world"
            )
        components.append((name, resolved))

    return tuple(components)


@dataclass(frozen=True)
class WorkerIpcConfig:
    """Configures IPC launch settings.

    Covers the head's registration address, the payload bound and the queue
    depth, which bounds the batches this rank holds in flight. The rank names
    its own channel endpoint and reports it to the registration address; the
    head binds the channel from that report.
    """

    registration_address: str
    channel_transport: str
    # This rank's word in every chunk header it reads, unique in the instance.
    acknowledgment_slot: int
    # Acknowledgment slots of the ranks on this rank's host, across workers.
    # A host product reaches a consumer among them over shared memory and any
    # other over the rank channel; the head derives this from the placement.
    host_slots: tuple[int, ...]
    # Acknowledgment slots of the ranks that read this rank's device products,
    # which the head derives from the transfer edges. A rank cannot name them
    # itself: it knows its own component, not which component consumes it.
    # Whether any rank that reads this rank's device products is on another
    # host. An interprocess event carries readiness within a host at no cost to
    # the producing stream; only a crossing needs a producer synchronize.
    products_cross_hosts: bool
    max_payload_bytes: int
    queue_depth: int


@dataclass(frozen=True)
class ModelLaunchConfig:
    """Selects checkpoint identity, precision policy."""

    path: str
    quantization_config: dict[str, object]
    # Identity the launching side derived for the checkpoint it names, when
    # it could read that checkpoint locally. A rank whose loaded checkpoint
    # has another identity refuses to start.
    checkpoint_identity: str | None = None


@dataclass(frozen=True)
class DataPlaneConfig:
    """Bind receive and required publication mechanisms for a rank."""

    backends: tuple[str, ...]
    publication_backends: tuple[str, ...]


@dataclass(frozen=True)
class WorkerProcessArgs:
    """Aggregates the validated launch configuration for one worker rank."""

    worker_id: str
    supported_ops: frozenset[CallKind]
    ipc: WorkerIpcConfig
    local_rank: int
    distributed_backend: str | None
    distributed_init_method: str | None
    model: ModelLaunchConfig | None
    data_plane: DataPlaneConfig
    execution: WorkerConfig
    load: IOConfig
    use_stub_model: bool
    components: tuple[tuple[str, ComponentConfig], ...] = ()

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> WorkerProcessArgs:
        """Validate parsed CLI values.

        Values resolve into immutable worker launch configuration.
        """
        supported_ops = _parse_supported_ops(namespace.supported_ops)
        device = _normalize_device(namespace.device)
        generation_device = _parse_mesh(
            str(namespace.mesh or ""),
            device=device,
        )
        backends = _parse_transfer_backends(namespace.transfer_backends)
        publication_backends = _parse_transfer_backends(
            namespace.publish_backends
        )
        model_path = str(namespace.model or "").strip()

        _validate_scalars(namespace)
        use_stub_model = bool(namespace.no_model)
        if use_stub_model and not bool(namespace.allow_stub):
            raise ValueError(
                "--no-model loads synthetic outputs and requires --allow-stub"
            )
        if not use_stub_model and not model_path:
            raise ValueError("--model is required for a model worker")

        if not set(publication_backends).issubset(backends):
            raise ValueError("publication backends must be bound transports")
        if "cuda_vmm" in backends and not device.startswith("cuda:"):
            raise ValueError("CUDA VMM requires a CUDA worker device")

        return cls(
            worker_id=str(namespace.worker_id),
            supported_ops=supported_ops,
            ipc=WorkerIpcConfig(
                registration_address=str(namespace.registration_address),
                channel_transport=str(namespace.channel_transport),
                acknowledgment_slot=int(namespace.acknowledgment_slot),
                host_slots=tuple(int(slot) for slot in namespace.host_slots),
                products_cross_hosts=bool(namespace.products_cross_hosts),
                max_payload_bytes=int(namespace.ipc_payload_cap),
                queue_depth=int(namespace.queue_depth),
            ),
            local_rank=int(namespace.local_rank),
            distributed_backend=_optional_text(namespace.distributed_backend),
            distributed_init_method=_optional_text(
                namespace.distributed_init_method
            ),
            model=(
                ModelLaunchConfig(
                    path=model_path,
                    quantization_config=dict(namespace.quantization_config),
                    # The descriptor carries the key only when the launching
                    # side could derive the identity from a local directory.
                    checkpoint_identity=_optional_text(
                        getattr(namespace, "checkpoint_identity", None)
                    ),
                )
                if model_path
                else None
            ),
            data_plane=DataPlaneConfig(
                backends=backends,
                publication_backends=publication_backends,
            ),
            execution=worker_config_from_namespace(
                namespace, device=device, generation_device=generation_device
            ),
            load=_load_config(namespace),
            use_stub_model=use_stub_model,
            components=parse_components(
                namespace.components, int(namespace.world_size)
            ),
        )


def _validate_scalars(namespace: argparse.Namespace) -> None:
    """Validate positive capacities.

    Also normalize optional scalar process settings.
    """
    positive_fields = {
        "--block-size": namespace.block_size,
        "--max-batch-calls": namespace.max_batch_calls,
        "--max-batch-tokens": namespace.max_batch_tokens,
        "--queue-depth": namespace.queue_depth,
        "--ipc-payload-cap": namespace.ipc_payload_cap,
        "--world-size": namespace.world_size,
        "--max-model-len": namespace.max_model_len,
    }
    for option, value in positive_fields.items():
        if int(value) <= 0:
            raise ValueError(f"{option} must be positive")
    if (
        namespace.kv_token_capacity is not None
        and int(namespace.kv_token_capacity) <= 0
    ):
        raise ValueError("--kv-token-capacity must be positive when provided")

    max_video_seconds = float(namespace.max_video_seconds)
    if not math.isfinite(max_video_seconds) or max_video_seconds <= 0:
        raise ValueError(
            "--max-video-seconds must resolve to a supported frame count"
        )
    max_video_frames = math.floor(max_video_seconds * 24.0 + 0.5)
    if max_video_frames < 6 or max_video_frames > 2**32 - 17:
        raise ValueError(
            "--max-video-seconds must resolve to a supported frame count"
        )

    if int(namespace.rank) < 0 or int(namespace.rank) >= int(
        namespace.world_size
    ):
        raise ValueError("--rank must satisfy 0 <= rank < world-size")


def _load_config(namespace: argparse.Namespace) -> IOConfig:
    """Separate the serialized reader selector into format and loading mode."""
    selected = str(namespace.load_format)
    mode = selected if selected in {"dummy", "layered"} else "eager"
    return IOConfig(
        format="auto" if selected in {"dummy", "layered"} else selected,
        mode=mode,
        download_dir=_optional_text(namespace.download_dir),
        num_threads=namespace.load_threads,
        checksum_manifest=_optional_text(namespace.checksum_manifest),
    )


def _parse_supported_ops(value: object) -> frozenset[CallKind]:
    """Resolve launch capability selectors to concrete call_kinds.

    The capability group names are a worker-side vocabulary that the launching
    side does not model, so it narrows the set only when it has a reason to.
    An absent selector therefore means every group this worker implements.
    """
    if value is None:
        value = ",".join(SUPPORTED_CALL_GROUPS)
    names = tuple(
        part.strip() for part in str(value).split(",") if part.strip()
    )
    if not names:
        raise ValueError("supported calls must list at least one group")
    try:
        calls = tuple(
            call for name in names for call in SUPPORTED_CALL_GROUPS[name]
        )
    except KeyError as error:
        raise ValueError(
            f"unknown call in supported calls {value!r}"
        ) from error
    if len(set(calls)) != len(calls):
        raise ValueError("--supported-ops contains duplicate calls")
    return frozenset(calls)


def _parse_mesh(
    value: str,
    *,
    device: str,
) -> str | None:
    """Parse and validate a named device-mesh declaration."""
    text = value.strip()
    if not text:
        return None

    generation_device: str | None = None
    seen: set[str] = set()
    for raw_entry in text.split(","):
        entry = raw_entry.strip()
        if not entry or "=" not in entry:
            raise ValueError(
                f"invalid --mesh entry {raw_entry!r}; expected key=value"
            )

        key, value = (part.strip() for part in entry.split("=", 1))
        normalized_key = key.replace("_", "-").lower()
        if normalized_key in seen:
            raise ValueError(f"duplicate --mesh key {key!r}")
        seen.add(normalized_key)

        if normalized_key == "tower":
            generation_device = _parse_expert_device(value, device=device)
        else:
            raise ValueError(f"unknown --mesh key {key!r}")

    return generation_device


def _parse_expert_device(value: str, *, device: str) -> str:
    """Resolve the optional flow device.

    Text stays on the Worker's rank device.
    """
    parameters: dict[str, str] = {}
    for raw_part in value.split(";"):
        part = raw_part.strip()
        name, separator, target = part.partition(":")
        normalized_name = name.strip().lower()
        if (
            not separator
            or normalized_name not in {"text", "gen"}
            or not target.strip()
        ):
            raise ValueError("tower params must use text:<device>;gen:<device>")
        if normalized_name in parameters:
            raise ValueError(f"duplicate tower params {normalized_name!r}")
        parameters[normalized_name] = target.strip()
    if "gen" not in parameters:
        raise ValueError("tower params requires gen:<device>")

    understanding_device = _normalize_device(parameters.get("text") or device)
    if understanding_device != device:
        raise ValueError("text expert device must match the Worker rank device")

    generation_device = _normalize_device(parameters["gen"])
    if understanding_device == generation_device:
        raise ValueError("tower text and gen devices must be different")

    return generation_device


def _normalize_device(value: object) -> str:
    """Pin an unindexed CUDA device to the concrete index the rank owns.

    The frontend launches single-GPU (tp=1) workers with ``--device cuda`` while
    the model materializes tensors on ``cuda:0``. Downstream validation compares
    ``torch.device`` objects, and ``torch.device("cuda")`` (index ``None``) does
    not equal ``torch.device("cuda:0")`` — so an unindexed device would reject
    every forward. Each worker process sees its GPU as device 0 under
    ``CUDA_VISIBLE_DEVICES``, so an unindexed CUDA device resolves to ``cuda:0``
    (this mirrors the tower-params normalization above).
    """
    import torch

    device = torch.device(str(value))
    if device.type == "cuda" and device.index is None:
        return "cuda:0"
    return str(value)


def _parse_transfer_backends(value: object) -> tuple[str, ...]:
    """Decode explicit unique physical mechanisms without fallback selection."""
    backends = tuple(part.strip() for part in str(value).split(","))
    if not backends or len(set(backends)) != len(backends):
        raise ValueError("transfer backends must be nonempty and unique")
    if any(
        name not in {"local", "shm", "cuda_vmm", "channel"} for name in backends
    ):
        raise ValueError("transfer backends must name local, shm, or cuda_vmm")
    return backends


def _optional_text(value: object | None) -> str | None:
    """Normalize an optional value to stripped text or ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None
