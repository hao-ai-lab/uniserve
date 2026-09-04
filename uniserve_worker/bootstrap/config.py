"""Typed process arguments for one Python worker rank."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from ..execution.batch import RunKind
from ..loader.config import LoadConfig
from .execution_config import ExecutionConfig, execution_config_from_namespace
from .plan import ModelLoadScope, resolve_worker_plan


@dataclass(frozen=True)
class WorkerIpcConfig:
    """Configures the IPC service name, payload bound, inflight limit, and pipeline depth."""

    service_name: str
    max_payload_bytes: int
    max_inflight: int
    pipeline_depth: int


@dataclass(frozen=True)
class WorkerPlacement:
    """Assigns a worker rank to its primary device, tensor-parallel group, and optional tower devices."""

    device: str
    tp_rank: int
    tp_size: int
    tp_backend: str | None
    tp_init_method: str | None
    tower_devices: tuple[str, str] | None

    @property
    def generation_device(self) -> str | None:
        """Expose the flow-tower device when modality towers are split."""

        if self.tower_devices is None:
            return None
        return self.tower_devices[1]


@dataclass(frozen=True)
class WorkerResourceConfig:
    """Caps batch work, model length, video duration, and physical KV allocation."""

    block_size: int
    max_batch_operations: int
    max_batch_tokens: int
    kv_token_capacity: int | None
    max_model_len: int
    max_video_seconds: float


@dataclass(frozen=True)
class ModelLaunchConfig:
    """Selects a checkpoint, attention backend, and quantization settings for model construction."""

    path: str
    attention_backend: str
    quantization_config: dict[str, object]


@dataclass(frozen=True)
class DataPlaneConfig:
    """Selects the transport backend used for cross-operation tensor movement."""

    backend: str


@dataclass(frozen=True)
class WorkerProcessArgs:
    """Aggregates the validated launch configuration for one worker rank."""

    supported_ops: frozenset[RunKind]
    ipc: WorkerIpcConfig
    placement: WorkerPlacement
    resources: WorkerResourceConfig
    model: ModelLaunchConfig | None
    data_plane: DataPlaneConfig
    execution: ExecutionConfig
    load: LoadConfig
    use_stub_model: bool

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> "WorkerProcessArgs":
        """Validate parsed CLI values and resolve them into immutable worker launch configuration."""

        supported_ops = _parse_supported_ops(namespace.supported_ops)
        plan = resolve_worker_plan(supported_ops)
        device = _normalize_device(namespace.device)
        tower_devices = _parse_mesh(
            str(namespace.mesh or ""),
            device=device,
        )
        backend = _normalize_transfer_backend(namespace.transfer_backend)
        use_stub_model = bool(namespace.no_model)
        model_path = str(namespace.model or "").strip()

        _validate_scalars(namespace)
        if use_stub_model and not bool(namespace.allow_stub):
            raise ValueError("--no-model loads synthetic outputs and requires --allow-stub")
        if use_stub_model and plan.model_scope is not ModelLoadScope.WHOLE:
            raise ValueError("--no-model cannot emulate partial model materialization")
        if not use_stub_model and not model_path:
            raise ValueError("--model is required for a model worker")
        _validate_data_plane(backend=backend)

        return cls(
            supported_ops=supported_ops,
            ipc=WorkerIpcConfig(
                service_name=str(namespace.service_name),
                max_payload_bytes=int(namespace.ipc_payload_cap),
                max_inflight=int(namespace.ipc_max_inflight),
                pipeline_depth=int(namespace.pipeline_depth),
            ),
            placement=WorkerPlacement(
                device=device,
                tp_rank=int(namespace.tp_rank),
                tp_size=int(namespace.tp_size),
                tp_backend=_optional_text(namespace.tp_backend),
                tp_init_method=_optional_text(namespace.tp_init_method),
                tower_devices=tower_devices,
            ),
            resources=WorkerResourceConfig(
                block_size=int(namespace.block_size),
                max_batch_operations=int(namespace.max_batch_operations),
                max_batch_tokens=int(namespace.max_batch_tokens),
                kv_token_capacity=(
                    int(namespace.kv_token_capacity)
                    if namespace.kv_token_capacity is not None
                    else None
                ),
                max_model_len=int(namespace.max_model_len),
                max_video_seconds=float(namespace.max_video_seconds),
            ),
            model=(
                ModelLaunchConfig(
                    path=model_path,
                    attention_backend=str(namespace.attention_backend),
                    quantization_config=dict(namespace.quantization_config),
                )
                if model_path
                else None
            ),
            data_plane=DataPlaneConfig(
                backend=backend,
            ),
            execution=execution_config_from_namespace(namespace),
            load=_load_config(namespace),
            use_stub_model=use_stub_model,
        )


def _validate_scalars(namespace: argparse.Namespace) -> None:
    """Validate positive capacities and normalize optional scalar process settings."""

    positive_fields = {
        "--block-size": namespace.block_size,
        "--max-batch-operations": namespace.max_batch_operations,
        "--max-batch-tokens": namespace.max_batch_tokens,
        "--pipeline-depth": namespace.pipeline_depth,
        "--ipc-payload-cap": namespace.ipc_payload_cap,
        "--ipc-max-inflight": namespace.ipc_max_inflight,
        "--tp-size": namespace.tp_size,
        "--max-model-len": namespace.max_model_len,
    }
    for option, value in positive_fields.items():
        if int(value) <= 0:
            raise ValueError(f"{option} must be positive")
    if namespace.kv_token_capacity is not None and int(namespace.kv_token_capacity) <= 0:
        raise ValueError("--kv-token-capacity must be positive when provided")
    max_video_seconds = float(namespace.max_video_seconds)
    if not math.isfinite(max_video_seconds) or max_video_seconds <= 0:
        raise ValueError("--max-video-seconds must resolve to supported media geometry")
    max_video_frames = math.floor(max_video_seconds * 24.0 + 0.5)
    if max_video_frames < 6 or max_video_frames > 2**32 - 17:
        raise ValueError("--max-video-seconds must resolve to supported media geometry")
    if int(namespace.tp_rank) < 0 or int(namespace.tp_rank) >= int(namespace.tp_size):
        raise ValueError("--tp-rank must satisfy 0 <= rank < tp-size")


def _load_config(namespace: argparse.Namespace) -> LoadConfig:
    """Build and validate process configuration from parsed command-line values."""

    threads = getattr(namespace, "load_threads", None)
    load_format = str(getattr(namespace, "load_format", "auto"))
    download_dir = _optional_text(getattr(namespace, "download_dir", None))
    checksum_manifest = _optional_text(getattr(namespace, "checksum_manifest", None))
    if threads is None:
        return LoadConfig(
            load_format=load_format,
            download_dir=download_dir,
            checksum_manifest=checksum_manifest,
        )
    return LoadConfig(
        load_format=load_format,
        download_dir=download_dir,
        num_threads=int(threads),
        checksum_manifest=checksum_manifest,
    )


def _validate_data_plane(*, backend: str) -> None:
    """Validate that the selected transfer backend implements the data plane."""

    if backend not in {"local", "shm", "cuda_ipc"}:
        raise ValueError(f"unknown --transfer-backend {backend!r}")


def _parse_supported_ops(value: object) -> frozenset[RunKind]:
    """Parse unique supported operation kinds from text or an iterable."""

    names = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not names:
        raise ValueError("--supported-ops must list at least one operation")
    try:
        operations = tuple(RunKind(name) for name in names)
    except ValueError as error:
        raise ValueError(f"unknown operation in --supported-ops {value!r}") from error
    if len(set(operations)) != len(operations):
        raise ValueError("--supported-ops contains duplicate operations")
    return frozenset(operations)


def _parse_mesh(
    value: str,
    *,
    device: str,
) -> tuple[str, str] | None:
    """Parse and validate a named device-mesh declaration."""

    text = value.strip()
    if not text:
        return None

    tower_devices: tuple[str, str] | None = None
    seen: set[str] = set()
    for raw_entry in text.split(","):
        entry = raw_entry.strip()
        if not entry or "=" not in entry:
            raise ValueError(f"invalid --mesh entry {raw_entry!r}; expected key=value")
        key, value = (part.strip() for part in entry.split("=", 1))
        normalized_key = key.replace("_", "-").lower()
        if normalized_key in seen:
            raise ValueError(f"duplicate --mesh key {key!r}")
        seen.add(normalized_key)
        if normalized_key == "tower":
            tower_devices = _parse_tower_placement(value, device=device)
        else:
            raise ValueError(f"unknown --mesh key {key!r}")
    return tower_devices


def _parse_tower_placement(value: str, *, device: str) -> tuple[str, str]:
    """Parse tower coordinates and validate their mesh-axis assignments."""

    import torch

    placements: dict[str, str] = {}
    for raw_part in value.split(";"):
        part = raw_part.strip()
        name, separator, target = part.partition(":")
        normalized_name = name.strip().lower()
        if not separator or normalized_name not in {"text", "gen"} or not target.strip():
            raise ValueError("tower placement must use text:<device>;gen:<device>")
        if normalized_name in placements:
            raise ValueError(f"duplicate tower placement {normalized_name!r}")
        placements[normalized_name] = target.strip()
    if "gen" not in placements:
        raise ValueError("tower placement requires gen:<device>")

    understanding_device = placements.get("text") or device
    understanding = torch.device(understanding_device)
    if understanding.type == "cuda" and understanding.index is None:
        understanding_device = "cuda:0"
    generation_device = placements["gen"]
    if torch.device(understanding_device) == torch.device(generation_device):
        raise ValueError("tower text and gen devices must be different")
    return understanding_device, generation_device


def _normalize_device(value: object) -> str:
    """Pin an unindexed CUDA device to the concrete index the rank owns.

    The frontend launches single-GPU (tp=1) workers with ``--device cuda`` while
    the model materializes tensors on ``cuda:0``. Downstream validation compares
    ``torch.device`` objects, and ``torch.device("cuda")`` (index ``None``) does
    not equal ``torch.device("cuda:0")`` — so an unindexed device would reject
    every forward. Each worker process sees its GPU as device 0 under
    ``CUDA_VISIBLE_DEVICES``, so an unindexed CUDA device resolves to ``cuda:0``
    (this mirrors the tower-placement normalization above).
    """

    import torch

    device = torch.device(str(value))
    if device.type == "cuda" and device.index is None:
        return "cuda:0"
    return str(value)


def _normalize_transfer_backend(value: object) -> str:
    """Canonicalize the configured transfer backend name."""

    backend = str(value or "local").strip().lower()
    return "local" if backend == "inproc" else backend


def _optional_text(value: object | None) -> str | None:
    """Normalize an optional value to stripped text or ``None``."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None
