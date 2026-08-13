"""Typed worker launch configuration at the CLI boundary."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from ..foundation.runtime_config import (
    ExecutionConfig,
    execution_config_from_namespace,
)
from ..loader.schema import ModelLoadScope
from ..server.worker_kind import WorkerKind
from .plan import WorkerImplementation, resolve_worker_plan


@dataclass(frozen=True)
class WorkerIpcConfig:
    service_name: str
    max_payload_bytes: int
    max_inflight: int
    pipeline_depth: int


@dataclass(frozen=True)
class WorkerPlacement:
    device: str
    tp_rank: int
    tp_size: int
    tp_backend: str | None
    tp_init_method: str | None
    tower_devices: tuple[str, str] | None


@dataclass(frozen=True)
class WorkerResourceConfig:
    block_size: int
    max_batch_tokens: int
    kv_token_capacity: int | None
    generation_kv_capacity_tokens: int | None


@dataclass(frozen=True)
class ModelLaunchConfig:
    path: str
    attention_backend: str


@dataclass(frozen=True)
class DataPlaneConfig:
    backend: str
    defer_sampling: bool


@dataclass(frozen=True)
class WorkerLaunchConfig:
    worker_kind: WorkerKind
    ipc: WorkerIpcConfig
    placement: WorkerPlacement
    resources: WorkerResourceConfig
    model: ModelLaunchConfig | None
    data_plane: DataPlaneConfig
    execution: ExecutionConfig
    use_stub_model: bool
    snapshot_dir: str | None

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> "WorkerLaunchConfig":
        worker_kind = WorkerKind(str(namespace.worker_kind))
        plan = resolve_worker_plan(worker_kind)
        device = _normalize_device(namespace.device)
        tower_devices, generation_kv_capacity_tokens = _parse_mesh(
            str(namespace.mesh or ""),
            device=device,
        )
        backend = _normalize_transfer_backend(namespace.transfer_backend)
        use_stub_model = bool(namespace.no_model)
        model_path = str(namespace.model or "").strip()

        _validate_scalars(namespace, generation_kv_capacity_tokens)
        if use_stub_model and not bool(namespace.allow_stub):
            raise ValueError("--no-model loads synthetic outputs and requires --allow-stub")
        if use_stub_model and plan.implementation is not WorkerImplementation.MODEL:
            raise ValueError(
                f"--no-model is only valid for model workers, not {worker_kind.value!r}"
            )
        if use_stub_model and plan.model_scope is not ModelLoadScope.WHOLE:
            raise ValueError("--no-model cannot emulate partial model materialization")
        if plan.requires_model and not use_stub_model and not model_path:
            raise ValueError(f"--model is required for worker kind {worker_kind.value!r}")
        _validate_data_plane(
            worker_kind,
            backend=backend,
            defer_sampling=bool(namespace.defer_sampling),
        )

        return cls(
            worker_kind=worker_kind,
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
                max_batch_tokens=int(namespace.max_batch_tokens),
                kv_token_capacity=(
                    int(namespace.kv_token_capacity)
                    if namespace.kv_token_capacity is not None
                    else None
                ),
                generation_kv_capacity_tokens=generation_kv_capacity_tokens,
            ),
            model=(
                ModelLaunchConfig(
                    path=model_path,
                    attention_backend=str(namespace.attention_backend),
                )
                if model_path
                else None
            ),
            data_plane=DataPlaneConfig(
                backend=backend,
                defer_sampling=bool(namespace.defer_sampling),
            ),
            execution=execution_config_from_namespace(namespace),
            use_stub_model=use_stub_model,
            snapshot_dir=_optional_text(namespace.snapshot_dir),
        )


def _validate_scalars(
    namespace: argparse.Namespace,
    generation_kv_capacity_tokens: int | None,
) -> None:
    positive_fields = {
        "--block-size": namespace.block_size,
        "--max-batch-tokens": namespace.max_batch_tokens,
        "--pipeline-depth": namespace.pipeline_depth,
        "--ipc-payload-cap": namespace.ipc_payload_cap,
        "--ipc-max-inflight": namespace.ipc_max_inflight,
        "--tp-size": namespace.tp_size,
    }
    for option, value in positive_fields.items():
        if int(value) <= 0:
            raise ValueError(f"{option} must be positive")
    if namespace.kv_token_capacity is not None and int(namespace.kv_token_capacity) <= 0:
        raise ValueError("--kv-token-capacity must be positive when provided")
    if generation_kv_capacity_tokens is not None and generation_kv_capacity_tokens <= 0:
        raise ValueError("--mesh tower-kv-capacity must be positive when provided")
    if int(namespace.tp_rank) < 0 or int(namespace.tp_rank) >= int(namespace.tp_size):
        raise ValueError("--tp-rank must satisfy 0 <= rank < tp-size")


def _validate_data_plane(
    worker_kind: WorkerKind,
    *,
    backend: str,
    defer_sampling: bool,
) -> None:
    if backend not in {"local", "shm", "cuda_ipc"}:
        raise ValueError(f"unknown --transfer-backend {backend!r}")
    if worker_kind in {WorkerKind.UND, WorkerKind.GEN} and backend != "cuda_ipc":
        raise ValueError(f"{worker_kind.value!r} requires same-node CUDA IPC transport")
    if (worker_kind is WorkerKind.SAMPLER or defer_sampling) and backend == "local":
        raise ValueError(f"{worker_kind.value!r} requires a cross-process logits transport")


def _parse_mesh(
    spec: str,
    *,
    device: str,
) -> tuple[tuple[str, str] | None, int | None]:
    text = spec.strip()
    if not text:
        return None, None

    tower_devices: tuple[str, str] | None = None
    generation_kv_capacity_tokens: int | None = None
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
        elif normalized_key == "tower-kv-capacity":
            generation_kv_capacity_tokens = int(value)
        else:
            raise ValueError(f"unknown --mesh key {key!r}")
    return tower_devices, generation_kv_capacity_tokens


def _parse_tower_placement(value: str, *, device: str) -> tuple[str, str]:
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
    backend = str(value or "local").strip().lower()
    return "local" if backend == "inproc" else backend


def _optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
