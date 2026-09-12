"""Typed process arguments for one Python worker rank."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

from uniserve_worker.nn.parallel import ComponentConfig, parse_entries
from uniserve_worker.protocol.batch import Computation, ForwardMode, PipelineStage, TransferMode

from ..config import WorkerConfig, worker_config_from_namespace
from ..loader.config import LoadConfig

# These launch selectors assign capabilities to a worker pool. A media selector
# includes both tracks; submitted computations still identify the concrete stage.
SUPPORTED_OP_GROUPS: dict[str, tuple[Computation, ...]] = {
    "ar_extend": (ForwardMode.PREFILL,),
    "ar_decode": (ForwardMode.DECODE,),
    "ar_verify": (ForwardMode.VERIFY,),
    "encoder_vision": (PipelineStage.VISION_ENCODING,),
    "encoder_latent": (PipelineStage.LATENT_ENCODING,),
    "encoder_text": (PipelineStage.TEXT_ENCODING,),
    "transfer_product": (TransferMode.TENSOR,),
    "transfer_kv_publish": (TransferMode.KV_PUBLISH,),
    "transfer_kv_install": (TransferMode.KV_INSTALL,),
    "diffusion_prepare": (PipelineStage.LATENT_PREPARATION,),
    "diffusion_step": (PipelineStage.DENOISING,),
    "diffusion_finalize": (PipelineStage.IMAGE_DECODING, PipelineStage.MUXING),
    "diffusion_decode": (PipelineStage.VIDEO_DECODING, PipelineStage.AUDIO_DECODING),
    "media_append": (PipelineStage.VIDEO_ENCODING, PipelineStage.AUDIO_ENCODING),
}


@dataclass(frozen=True)
class WorkerIpcConfig:
    """Configures the IPC service name, payload bound, inflight limit, and pipeline depth."""

    service_name: str
    max_payload_bytes: int
    max_inflight: int
    pipeline_depth: int


@dataclass(frozen=True)
class ModelLaunchConfig:
    """Selects checkpoint identity, precision policy, and bounded text/video geometry."""

    path: str
    quantization_config: dict[str, object]
    max_text_rows: int
    max_video_seconds: float


@dataclass(frozen=True)
class DataPlaneConfig:
    """Bind receive mechanisms and required publication mechanisms for a rank."""

    backends: tuple[str, ...]
    publication_backends: tuple[str, ...]


@dataclass(frozen=True)
class WorkerProcessArgs:
    """Aggregates the validated launch configuration for one worker rank."""

    worker_id: str
    supported_ops: frozenset[Computation]
    ipc: WorkerIpcConfig
    local_rank: int
    distributed_backend: str | None
    distributed_init_method: str | None
    model: ModelLaunchConfig | None
    data_plane: DataPlaneConfig
    execution: WorkerConfig
    load: LoadConfig
    use_stub_model: bool
    components: tuple[tuple[str, ComponentConfig], ...] = ()

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> "WorkerProcessArgs":
        """Validate parsed CLI values and resolve them into immutable worker launch configuration."""

        supported_ops = _parse_supported_ops(namespace.supported_ops)
        device = _normalize_device(namespace.device)
        generation_device = _parse_mesh(
            str(namespace.mesh or ""),
            device=device,
        )
        backends = _parse_transfer_backends(namespace.transfer_backends)
        publication_backends = _parse_transfer_backends(namespace.publish_backends)
        use_stub_model = bool(namespace.no_model)
        model_path = str(namespace.model or "").strip()

        _validate_scalars(namespace)
        if use_stub_model and not bool(namespace.allow_stub):
            raise ValueError("--no-model loads synthetic outputs and requires --allow-stub")
        if not use_stub_model and not model_path:
            raise ValueError("--model is required for a model worker")
        if not set(publication_backends).issubset(backends):
            raise ValueError("publication backends must be bound transports")
        if "cuda_ipc" in backends and not device.startswith("cuda:"):
            raise ValueError("CUDA IPC requires a CUDA worker device")
        allocator = (
            (os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))
            .replace(" ", "")
            .lower()
        )
        if "cuda_ipc" in publication_backends and (
            "expandable_segments:true" in allocator or "backend:cudamallocasync" in allocator
        ):
            raise ValueError("CUDA IPC publication requires native nonexpandable CUDA allocations")

        return cls(
            worker_id=str(namespace.worker_id),
            supported_ops=supported_ops,
            ipc=WorkerIpcConfig(
                service_name=str(namespace.service_name),
                max_payload_bytes=int(namespace.ipc_payload_cap),
                max_inflight=int(namespace.ipc_max_inflight),
                pipeline_depth=int(namespace.pipeline_depth),
            ),
            local_rank=int(namespace.local_rank),
            distributed_backend=_optional_text(namespace.distributed_backend),
            distributed_init_method=_optional_text(namespace.distributed_init_method),
            model=(
                ModelLaunchConfig(
                    path=model_path,
                    quantization_config=dict(namespace.quantization_config),
                    max_text_rows=int(namespace.max_model_len),
                    max_video_seconds=float(namespace.max_video_seconds),
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
            components=parse_entries(namespace.entries, int(namespace.world_size)),
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
        "--world-size": namespace.world_size,
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
    if int(namespace.rank) < 0 or int(namespace.rank) >= int(namespace.world_size):
        raise ValueError("--rank must satisfy 0 <= rank < world-size")


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


def _parse_supported_ops(value: object) -> frozenset[Computation]:
    """Resolve unique launch capability selectors to concrete computations."""

    names = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not names:
        raise ValueError("--supported-ops must list at least one operation")
    try:
        operations = tuple(operation for name in names for operation in SUPPORTED_OP_GROUPS[name])
    except KeyError as error:
        raise ValueError(f"unknown operation in --supported-ops {value!r}") from error
    if len(set(operations)) != len(operations):
        raise ValueError("--supported-ops contains duplicate operations")
    return frozenset(operations)


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
            raise ValueError(f"invalid --mesh entry {raw_entry!r}; expected key=value")
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
    """Resolve the optional flow device while keeping text on the Worker's rank device."""

    parameters: dict[str, str] = {}
    for raw_part in value.split(";"):
        part = raw_part.strip()
        name, separator, target = part.partition(":")
        normalized_name = name.strip().lower()
        if not separator or normalized_name not in {"text", "gen"} or not target.strip():
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
    if any(name not in {"local", "shm", "cuda_ipc"} for name in backends):
        raise ValueError("transfer backends must name local, shm, or cuda_ipc")
    return backends


def _optional_text(value: object | None) -> str | None:
    """Normalize an optional value to stripped text or ``None``."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None
