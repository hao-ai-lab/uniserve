"""Typed process arguments for one Python worker rank.

The engine writes one JSON launch descriptor per rank, and
``uniserve_worker.bootstrap.cli`` reads it into an ``argparse.Namespace``.
``WorkerProcessArgs.from_namespace`` validates that namespace once and resolves
it into immutable values, so later bootstrap stages consume typed fields and
never reparse descriptor text. Values this module rejects raise
``ValueError``, which the launch adapter reports as a usage error before the
worker starts; a value that passes these checks but violates a
``WorkerConfig`` invariant raises ``WorkerError`` from
``WorkerConfig.__post_init__`` instead.

``SequenceConfig``, ``ParallelConfig`` and ``ComponentConfig`` mirror the
component declarations in the ``uniserve-core`` crate's ``parallel`` module:
the same field names, a missing degree defaulting to one, and unknown fields
rejected.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import prod
from pathlib import Path
from typing import Literal

from uniserve.loading import Config as IOConfig
from uniserve.runtime.process_groups import Rendezvous
from uniserve_worker.config.execution import (
    WorkerConfig,
    worker_config_from_namespace,
)
from uniserve_worker.protocol.call import (
    CallKind,
    ForwardMode,
    MediaCall,
    TransferMode,
)

# These launch selectors assign capabilities to a worker pool. The engine
# sends the selected group names as one comma-separated ``supported_calls``
# value. A media selector includes both tracks; submitted call kinds still
# identify the concrete call.
SUPPORTED_CALL_GROUPS: dict[str, tuple[CallKind, ...]] = {
    "ar_extend": (ForwardMode.PREFILL,),
    "ar_decode": (ForwardMode.DECODE,),
    "ar_verify": (ForwardMode.VERIFY,),
    "token_denoising": (ForwardMode.TOKEN_DENOISING,),
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
    "media_read": (MediaCall.MEDIA_READING,),
    "media_append": (
        MediaCall.VIDEO_ENCODING,
        MediaCall.AUDIO_ENCODING,
    ),
}


def _positive_degree(name: str, value: object) -> int:
    # ``type(...) is int`` rather than ``isinstance`` rejects ``True``, which
    # JSON decoding could otherwise let through as a degree of one.
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


# Degree fields each sequence strategy accepts, in the order
# ``SequenceConfig.degrees`` stores them.
_SEQUENCE_FIELDS = {
    "local": (),
    "ulysses": ("ulysses_degree",),
    "allgather": ("allgather_degree",),
    "hybrid": ("ulysses_degree", "allgather_degree"),
}


@dataclass(frozen=True, slots=True)
class SequenceConfig:
    """Select the sequence-parallel algorithm and its active degrees.

    ``degrees`` holds one positive value per field that ``kind`` names in
    ``_SEQUENCE_FIELDS``, in that order; for ``hybrid`` it is
    ``(ulysses_degree, allgather_degree)``.
    """

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
        """Return context axes followed by the Ulysses axis.

        The allgather degree becomes the ``cp`` mesh axis, which model loading
        uses as the context-parallel gather axis. Every strategy reports both
        axes, with size one when inactive, so every component mesh carries the
        same axis names.
        """
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
        """Parse the descriptor object tagged by its ``kind`` field.

        Raises:
            ValueError: If ``kind`` is unknown, a field does not belong to that
                strategy, or a degree is not a positive integer.
        """
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
        """Return mesh axes in rank order, with Ulysses varying fastest.

        Component ranks map onto this shape in row-major order, so the order
        decides which ranks form each pipeline, tensor, context, and Ulysses
        group. ``uniserve_worker.bootstrap.distributed`` and the component
        binding both build their ``DeviceMesh`` from it.
        """
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
    """Configure rank placement and parallelism for one model component.

    Without a ``distribution``, ``ranks`` form one model-parallel group whose
    size equals the parallel world size. With ``temporal_units``, every rank
    runs its own local instance and the component divides independent media
    units across the ranks: each rank takes a contiguous run of
    ``units_per_rank`` units in the order of ``ranks``, and the parallel world
    size must be one. Which components accept a distribution is checked
    against the model's declared calls by ``validate_components`` in
    ``uniserve_worker.bootstrap.components``.
    """

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

        # Type checks happen here; value checks happen in ``__post_init__``.
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
        # The distribution fields appear only with a distribution;
        # ``from_dict`` restores their defaults when they are absent.
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
    value: dict[str, object], world_size: int, *, role: str = "model"
) -> tuple[tuple[str, ComponentConfig], ...]:
    """Parse model component placement supplied by the worker launcher.

    Args:
        value: Mapping from component name to its descriptor object.
        world_size: Size of the process world every rank must fall inside.

    Returns:
        ``(name, config)`` pairs sorted by name.

    Raises:
        ValueError: If the mapping is empty, an entry is malformed, or a rank
            lies outside the process world.
    """
    if role == "experts":
        if value:
            raise ValueError("expert workers have no request components")
        return ()
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
    # A host product reaches a consumer among them over shared storage and any
    # other over the rank channel; the head derives this from the placement.
    host_slots: tuple[int, ...]
    # Whether any rank that reads this rank's device products is on another
    # host. An interprocess event carries readiness within a host at no cost to
    # the producing stream; only a crossing needs a producer synchronize. The
    # head derives this from the transfer edges and the placement, which only
    # it holds.
    products_cross_hosts: bool
    max_payload_bytes: int
    queue_depth: int


@dataclass(frozen=True)
class ModelLaunchConfig:
    """Selects the checkpoint and its quantization policy."""

    path: str
    quantization_config: dict[str, object]
    # Local components supplied by the caller; otherwise use the Hub base.
    base_model: str | None = None


@dataclass(frozen=True)
class DataPlaneConfig:
    """Bind receive and required publication mechanisms for a rank.

    ``WorkerProcessArgs.from_namespace`` requires ``publication_backends`` to
    be a subset of ``backends``: a rank publishes only over mechanisms it also
    binds.
    """

    backends: tuple[str, ...]
    publication_backends: tuple[str, ...]


@dataclass(frozen=True)
class ExpertParallelLaunch:
    """This replica's place in its deployment's expert-parallel world.

    ``rank`` is this physical rank in the union of ``size`` ranks. Leading
    ``attention_ranks`` own attention and request state; remaining ranks
    own experts. Zero selects colocated expert parallelism. The world forms
    at ``rendezvous``, whose store rank 0 serves on its inherited socket.
    """

    rank: int
    size: int
    attention_ranks: int
    rendezvous: Rendezvous


@dataclass(frozen=True)
class WorkerProcessArgs:
    """Aggregates the validated launch configuration for one worker rank."""

    worker_id: str
    supported_calls: frozenset[CallKind]
    ipc: WorkerIpcConfig
    local_rank: int
    distributed_backend: str | None
    # Where a multi-rank group forms its process world; None for one rank.
    rendezvous: Rendezvous | None
    model: ModelLaunchConfig | None
    data_plane: DataPlaneConfig
    execution: WorkerConfig
    load: IOConfig
    use_stub_model: bool
    components: tuple[tuple[str, ComponentConfig], ...] = ()
    expert_parallel: ExpertParallelLaunch | None = None

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> WorkerProcessArgs:
        """Validate parsed launch values into immutable worker configuration.

        Args:
            namespace: Launch descriptor fields, as produced by
                ``uniserve_worker.bootstrap.cli.read_launch_descriptor``.

        Returns:
            The validated launch configuration for this rank.

        Raises:
            ValueError: If a value is malformed, out of range, inconsistent
                with another value, or names an unsupported option.
            WorkerError: If execution settings that pass these checks violate
                a ``WorkerConfig`` invariant.
        """
        role = str(getattr(namespace, "role", "model"))
        supported_calls = (
            frozenset()
            if role == "experts"
            else _parse_supported_calls(namespace.supported_calls)
        )
        expert_parallel = _expert_parallel(namespace)
        if role == "experts" and (
            expert_parallel is None
            or not expert_parallel.attention_ranks
            or expert_parallel.rank < expert_parallel.attention_ranks
        ):
            raise ValueError(
                "expert workers require a disaggregated expert union"
            )
        device = _normalize_device(namespace.device, option="--device")
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

        # ``no_model`` builds the stub model, which serves synthetic outputs;
        # it is accepted only together with ``allow_stub``.
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
        if device.startswith("cuda") or (
            generation_device is not None
            and generation_device.startswith("cuda")
        ):
            _require_native_cuda_representation(namespace)

        return cls(
            worker_id=str(namespace.worker_id),
            supported_calls=supported_calls,
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
            rendezvous=_rendezvous(namespace),
            model=(
                ModelLaunchConfig(
                    path=model_path,
                    quantization_config=dict(namespace.quantization_config),
                    # The descriptor carries the key only when the operator
                    # named a local base checkpoint.
                    base_model=_optional_text(
                        getattr(namespace, "base_model", None)
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
                namespace.components, int(namespace.world_size), role=role
            ),
            expert_parallel=expert_parallel,
        )


def _validate_scalars(namespace: argparse.Namespace) -> None:
    """Validate positive capacities and scalar process settings.

    Raises:
        ValueError: If a value is out of range; the message names the launch
            option.
    """
    positive_fields = {
        "--max-batch-calls": namespace.max_batch_calls,
        "--max-batch-tokens": namespace.max_batch_tokens,
        "--max-request-pool-size": namespace.max_request_pool_size,
        "--queue-depth": namespace.queue_depth,
        "--ipc-payload-cap": namespace.ipc_payload_cap,
        "--world-size": namespace.world_size,
        "--max-model-len": namespace.max_model_len,
        "--flashinfer-workspace-size": namespace.flashinfer_workspace_size,
    }
    for option, value in positive_fields.items():
        if int(value) <= 0:
            raise ValueError(f"{option} must be positive")
    if (
        namespace.kv_token_capacity is not None
        and int(namespace.kv_token_capacity) <= 0
    ):
        raise ValueError("--kv-token-capacity must be positive when provided")

    # The capacity resolves to frames at 24 frames per second, rounded half
    # to even as the server rounds a request's duration. The server admits
    # capacities within its video API range; the worker only requires one
    # that covers at least one frame and fits a u32 once the model's native
    # windows extend it by up to 16 frames.
    max_video_seconds = float(namespace.max_video_seconds)
    if not math.isfinite(max_video_seconds) or max_video_seconds <= 0:
        raise ValueError(
            "--max-video-seconds must resolve to a supported frame count"
        )
    max_video_frames = round(max_video_seconds * 24.0)
    if max_video_frames < 1 or max_video_frames > 2**32 - 17:
        raise ValueError(
            "--max-video-seconds must resolve to a supported frame count"
        )

    if int(namespace.rank) < 0 or int(namespace.rank) >= int(
        namespace.world_size
    ):
        raise ValueError("--rank must satisfy 0 <= rank < world-size")


def _require_native_cuda_representation(namespace: argparse.Namespace) -> None:
    """Reject CUDA launches whose representation no native kernel computes.

    CUDA attention runs only native kernels, which compute in BF16 or FP16
    over BF16 or FP16 caches:

    - an FP32 model (``--dtype float32``) has no native attention kernel;
    - an FP8 KV cache (``--kv-cache-dtype float8_e4m3fn``, or the
      ``kv_cache_dtype`` key of ``--quantization-config``, which overrides
      it) stores one FP8 scale per cache block. The FP8-KV TensorRT-LLM
      kernels read FP8 queries with per-tensor BMM1/BMM2 scales and no
      native kernel reads per-block scales.

    Failing here, before the model loads, names the option instead of the
    first attention layer that cannot be prepared.

    Raises:
        ValueError: The launch selects one of these representations.
    """
    if str(namespace.model_dtype) == "float32":
        raise ValueError(
            "--dtype float32 has no native CUDA attention kernel: CUDA "
            "attention computes in bfloat16 or float16"
        )
    override = dict(namespace.quantization_config).get("kv_cache_dtype")
    storage = override if override is not None else namespace.kv_cache_dtype
    if storage == "float8_e4m3fn":
        option = (
            "--kv-cache-dtype"
            if override is None
            else "--quantization-config kv_cache_dtype"
        )
        raise ValueError(
            f"{option} float8_e4m3fn has no native CUDA attention kernel: "
            "the cache keeps one FP8 scale per block, which no native kernel "
            "reads (the FP8-KV TensorRT-LLM kernels take FP8 queries with "
            "per-tensor BMM1/BMM2 scales); use a bfloat16 or float16 KV cache"
        )


def _load_config(namespace: argparse.Namespace) -> IOConfig:
    """Separate the serialized reader selector into format and loading mode.

    The descriptor carries one ``load_format`` token. A file format implies
    eager loading; ``dummy`` and ``layered`` select a loading mode and leave
    the file format to auto-detection.
    """
    selected = str(namespace.load_format)
    file_format: Literal["auto", "safetensors", "pt"]
    mode: Literal["eager", "layered", "dummy"]
    match selected:
        case "auto" | "safetensors" | "pt":
            file_format, mode = selected, "eager"
        case "dummy" | "layered":
            file_format, mode = "auto", selected
        case _:
            raise ValueError(f"unknown checkpoint load format {selected!r}")

    return IOConfig(
        format=file_format,
        mode=mode,
        download_dir=_optional_path(namespace.download_dir),
        num_threads=namespace.load_threads,
        checksum_manifest=_optional_path(namespace.checksum_manifest),
    )


def _parse_supported_calls(value: object) -> frozenset[CallKind]:
    """Resolve launch capability selectors to concrete call kinds.

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

    # Groups are disjoint, so a duplicate call means a repeated group name.
    if len(set(calls)) != len(calls):
        raise ValueError("supported calls contain duplicate entries")
    return frozenset(calls)


def _parse_mesh(
    value: str,
    *,
    device: str,
) -> str | None:
    """Parse and validate a named device-mesh declaration.

    The declaration is a comma-separated list of ``key=value`` entries; keys
    compare case-insensitively with ``_`` and ``-`` equivalent. ``tower`` is
    the only supported key.

    Returns:
        The generation device selected by a ``tower`` entry, or ``None`` when
        the declaration is empty.
    """
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
    """Resolve the generation device of a ``tower`` mesh entry.

    The value has the form ``text:<device>;gen:<device>``. The text tower
    always runs on the rank device, so ``text`` may be omitted and, when
    given, must name that device. ``gen`` is required and must name a
    different device; model loading places the flow route, denoiser, and image
    decoder modules there.
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

    understanding_device = _normalize_device(
        parameters.get("text") or device, option="--mesh tower text"
    )
    if understanding_device != device:
        raise ValueError("text expert device must match the Worker rank device")

    generation_device = _normalize_device(
        parameters["gen"], option="--mesh tower gen"
    )
    if understanding_device == generation_device:
        raise ValueError("tower text and gen devices must be different")

    return generation_device


def _normalize_device(value: object, *, option: str) -> str:
    """Pin an unindexed CUDA device to the concrete index the rank owns.

    A worker scoped by ``CUDA_VISIBLE_DEVICES`` to one GPU sees it as device 0,
    so the tensors it creates land on ``cuda:0``. Downstream validation
    compares ``torch.device`` objects, and ``torch.device("cuda")`` (index
    ``None``) does not equal ``torch.device("cuda:0")``, so an unindexed
    device would reject every forward; it therefore resolves to ``cuda:0``.
    Every other value, including an indexed CUDA device, is returned
    unchanged. The ``tower`` devices pass through the same normalization.

    Raises:
        ValueError: ``value`` is not a device string torch accepts; the
            message names the launch ``option`` that carried it.
    """
    import torch

    # torch reports a malformed device string as RuntimeError; it is a
    # launch value error like every other rejected setting.
    try:
        device = torch.device(str(value))
    except RuntimeError as error:
        raise ValueError(
            f"{option} must name a torch device, got {value!r}: {error}"
        ) from error
    if device.type == "cuda" and device.index is None:
        return "cuda:0"
    return str(value)


def _parse_transfer_backends(value: object) -> tuple[str, ...]:
    """Decode explicit unique physical mechanisms without fallback selection.

    The order of the declared mechanisms is preserved.
    """
    backends = tuple(part.strip() for part in str(value).split(","))
    if not backends or len(set(backends)) != len(backends):
        raise ValueError("transfer backends must be nonempty and unique")
    if any(
        name not in {"local", "shm", "cuda_vmm", "channel"} for name in backends
    ):
        raise ValueError("transfer backends must name local, shm, or cuda_vmm")
    return backends


def _rendezvous(namespace: argparse.Namespace) -> Rendezvous | None:
    """Resolve where this rank's group forms its process world.

    The process that spawns the group's first rank, the engine or the
    launcher of that rank's host, binds the store's socket before the rank
    exists and the rank inherits it, so the store's port is never free while
    the rank starts. A first rank without the socket would bind the port
    itself and could lose it to another process, so the descriptor must name
    one for that rank and for no other.
    """
    address = _optional_text(namespace.rendezvous_address)
    listen_fd = namespace.rendezvous_listen_fd
    if address is None:
        if listen_fd is not None:
            raise ValueError("a rendezvous socket requires its address")
        return None

    first = int(namespace.rank) == 0
    if first and listen_fd is None:
        raise ValueError("the first rank must inherit its rendezvous socket")
    if not first and listen_fd is not None:
        raise ValueError("only the first rank inherits a rendezvous socket")

    host, _, port = address.rpartition(":")
    return Rendezvous(
        host=host,
        port=int(port),
        listen_fd=None if listen_fd is None else int(listen_fd),
    )


def _expert_parallel(
    namespace: argparse.Namespace,
) -> ExpertParallelLaunch | None:
    """Resolve the replica's expert-parallel world, when the launch names one.

    The descriptor's optional ``expert_parallel`` object carries ``rank``,
    ``size``, leading ``attention_ranks``, the store ``address``, and for
    rank 0 only the inherited ``listen_fd``. ``exchange`` selects the
    transport in ``WorkerConfig``. Worker-local ranks are consecutive in
    the union and keep their independent component meshes.

    Raises:
        ValueError: The object is malformed, the rank lies outside a world of
            at least two replicas, the group has more than one rank, or the
            store socket is missing on rank 0 or present on another rank.
    """
    value = getattr(namespace, "expert_parallel", None)
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "rank",
        "size",
        "attention_ranks",
        "address",
        "listen_fd",
        "exchange",
    }:
        raise ValueError(
            "expert_parallel must carry rank, size, attention_ranks, address, "
            "listen_fd and exchange"
        )
    if value["exchange"] not in ("alltoall", "megamoe", "dwdp", "deepep"):
        raise ValueError(
            f"unknown expert exchange {value['exchange']!r}; expected "
            "alltoall, megamoe, dwdp or deepep"
        )

    rank, size = value["rank"], value["size"]
    if type(rank) is not int or type(size) is not int or size < 2:
        raise ValueError("an expert-parallel world spans at least two ranks")
    if not 0 <= rank < size:
        raise ValueError("expert-parallel rank must satisfy 0 <= rank < size")
    attention = value["attention_ranks"]
    if type(attention) is not int or not 0 <= attention < size:
        raise ValueError("attention ranks must leave at least one expert rank")
    start = rank - int(namespace.rank)
    stop = start + int(namespace.world_size)
    if start < 0 or stop > size or start < attention < stop:
        raise ValueError(
            "worker ranks must fit within one role in the expert union"
        )
    if bool(attention) and (
        value["exchange"] not in {"deepep", "megamoe"}
        or (rank >= attention)
        != (getattr(namespace, "role", "model") == "experts")
    ):
        raise ValueError(
            "disaggregated role or expert exchange disagrees with placement"
        )
    if not attention and (
        int(namespace.world_size) != 1 or value["exchange"] == "deepep"
    ):
        raise ValueError(
            "an expert-parallel replica joins its world with its only rank"
        )

    address, listen_fd = value["address"], value["listen_fd"]
    if not isinstance(address, str) or ":" not in address:
        raise ValueError("expert-parallel store address must be host:port")
    if (rank == 0) != (listen_fd is not None):
        raise ValueError(
            "exactly the expert-parallel world's rank 0 serves its store"
        )
    host, _, port = address.rpartition(":")
    return ExpertParallelLaunch(
        rank=rank,
        size=size,
        attention_ranks=attention,
        rendezvous=Rendezvous(
            host=host,
            port=int(port),
            listen_fd=None if listen_fd is None else int(listen_fd),
        ),
    )


def _optional_text(value: object | None) -> str | None:
    """Normalize an optional value to stripped text or ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_path(value: object | None) -> Path | None:
    """Normalize an optional value to a path or ``None`` when blank."""
    text = _optional_text(value)
    return None if text is None else Path(text)
