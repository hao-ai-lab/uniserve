"""Typed process arguments for one Python worker rank.

The engine writes one JSON launch descriptor per rank, and
``uniserve_worker.bootstrap.cli`` reads it into an ``argparse.Namespace``.
``WorkerProcessArgs.from_namespace`` resolves that namespace once into
immutable values, so later bootstrap stages consume typed fields and never
reparse descriptor text. The command line and the engine validate the launch
values they own before they write the descriptor; this module checks only what
the worker alone knows, such as its device and the transports it can bind.
Values it rejects raise ``ValueError``, which the launch adapter reports as a
usage error before the worker starts; a value that violates a ``WorkerConfig``
invariant raises ``WorkerError`` from ``WorkerConfig.__post_init__`` instead.

``SequenceConfig``, ``ParallelConfig`` and ``ComponentConfig`` mirror the
component declarations in the ``uniserve-core`` crate's ``parallel`` module:
the same field names, a missing degree defaulting to one, and unknown fields
rejected.
"""

from __future__ import annotations

import argparse
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
    ``uniserve_worker.bootstrap.components``. The video decoder and the video
    codec are always ``temporal_units`` with one unit per rank; the engine
    completes that placement from a deployment that omits it, so a launch
    descriptor always carries it.
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
        return ()

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
    queue_depth: int


@dataclass(frozen=True)
class ModelLaunchConfig:
    """Selects the checkpoint and its quantization policy."""

    path: str
    quantization_config: dict[str, object]


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
        role = str(namespace.role)
        expert_parallel = _expert_parallel(namespace)
        device = _normalize_device(namespace.device, option="--device")
        generation_device = _parse_mesh(namespace.mesh, device=device)
        backends = _parse_transfer_backends(namespace.transfer_backends)
        publication_backends = _parse_transfer_backends(
            namespace.publish_backends
        )
        model_path = str(namespace.model or "").strip()

        # ``no_model`` builds the stub model, which serves synthetic outputs.
        use_stub_model = bool(namespace.no_model)
        if not use_stub_model and not model_path:
            raise ValueError("a model worker requires its model path")

        if not set(publication_backends).issubset(backends):
            raise ValueError("publication backends must be bound transports")
        if "cuda_vmm" in backends and not device.startswith("cuda:"):
            raise ValueError("CUDA VMM requires a CUDA worker device")

        return cls(
            worker_id=str(namespace.worker_id),
            ipc=WorkerIpcConfig(
                registration_address=str(namespace.registration_address),
                channel_transport=str(namespace.channel_transport),
                acknowledgment_slot=int(namespace.acknowledgment_slot),
                host_slots=tuple(int(slot) for slot in namespace.host_slots),
                products_cross_hosts=bool(namespace.products_cross_hosts),
                queue_depth=int(namespace.queue_depth),
            ),
            local_rank=int(namespace.local_rank),
            distributed_backend=_optional_text(namespace.distributed_backend),
            rendezvous=_rendezvous(namespace),
            model=(
                ModelLaunchConfig(
                    path=model_path,
                    quantization_config=dict(namespace.quantization_config),
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
    )


def _parse_mesh(value: object | None, *, device: str) -> str | None:
    """Resolve the generation device a ``--worker-mesh`` declaration names.

    The declaration is ``tower=gen:<device>``: the text tower runs on the rank
    device, and model loading places the flow route, denoiser, and image
    decoder modules on ``<device>``, which must differ from it.

    Returns:
        The generation device, or ``None`` when the declaration is empty.

    Raises:
        ValueError: The declaration is not ``tower=gen:<device>`` or names the
            rank device.
    """
    text = _optional_text(value)
    if text is None:
        return None

    key, _, selection = text.partition("=")
    name, _, target = selection.partition(":")
    if (
        key.strip().lower() != "tower"
        or name.strip().lower() != "gen"
        or not target.strip()
    ):
        raise ValueError(
            f"--worker-mesh must be tower=gen:<device>, got {text!r}"
        )

    generation_device = _normalize_device(
        target.strip(), option="--worker-mesh tower gen"
    )
    if generation_device == device:
        raise ValueError(
            "--worker-mesh tower gen must name a device other than the rank's"
        )
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
        name not in {"local", "posix_shm", "cuda_vmm", "channel"}
        for name in backends
    ):
        raise ValueError(
            "transfer backends must name local, posix_shm, cuda_vmm, or channel"
        )
    return backends


def _rendezvous(namespace: argparse.Namespace) -> Rendezvous | None:
    """Resolve where this rank's group forms its process world.

    The process that spawns the group's first rank, the engine or the
    launcher of that rank's host, binds the store's socket before the rank
    exists and the descriptor names it for that rank alone, so the store's
    port is never free while the rank starts.
    """
    address = _optional_text(namespace.rendezvous_address)
    if address is None:
        return None

    host, _, port = address.rpartition(":")
    listen_fd = namespace.rendezvous_listen_fd
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
    the world's rank 0 only the inherited ``listen_fd``. ``exchange`` selects
    the transport in ``WorkerConfig``. The engine places the world and
    validates its shape before it launches a rank.
    """
    value = namespace.expert_parallel
    if value is None:
        return None

    host, _, port = str(value["address"]).rpartition(":")
    listen_fd = value["listen_fd"]
    return ExpertParallelLaunch(
        rank=int(value["rank"]),
        size=int(value["size"]),
        attention_ranks=int(value["attention_ranks"]),
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
