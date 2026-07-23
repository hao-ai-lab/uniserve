"""Typed process-boundary contract implemented by assembled workers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from ..batch import Batch, ExecutionResult
from ..capabilities import (
    AdapterMode,
    EngineCaps,
    ExecutionConstraints,
    RankInfo,
    RequestKind,
    ResourceClass,
)
from ..foundation.errors import capability_mismatch
from ..runtime.snapshot_store import SnapshotRef
from ..spec import OperationType


@dataclass(frozen=True, slots=True)
class WorkerContract:
    capabilities: EngineCaps

    @classmethod
    def compile(
        cls,
        declared_capabilities: EngineCaps,
        *,
        allowed_operation_types: frozenset[OperationType],
        system_operation_types: frozenset[OperationType] = frozenset(),
        pipeline_depth: int,
        owner: str,
    ) -> WorkerContract:
        if int(pipeline_depth) <= 0:
            raise capability_mismatch("worker pipeline depth must be positive")
        implemented = frozenset(declared_capabilities.supported_operation_types) | frozenset(
            system_operation_types
        )
        effective = tuple(
            operation_type
            for operation_type in OperationType
            if operation_type in allowed_operation_types and operation_type in implemented
        )
        if not effective:
            raise capability_mismatch(
                f"{owner} implements none of the requested operation types "
                f"{sorted(value.value for value in allowed_operation_types)!r}"
            )
        return cls(
            capabilities=replace(
                declared_capabilities,
                supported_operation_types=effective,
                pipeline_depth=int(pipeline_depth),
            ),
        )


@runtime_checkable
class Worker(Protocol):
    @property
    def contract(self) -> WorkerContract: ...

    def execute(self, batch: Batch) -> ExecutionResult: ...

    def drop_session(self, session_id: int) -> None: ...

    def copy_kv(self, copies: tuple[tuple[int, int], ...]) -> None: ...

    def load_adapter(self, adapter_id: int, adapter_path: str) -> None: ...

    def unload_adapter(self, adapter_id: int) -> None: ...

    def release_products(self, handles: tuple[int, ...]) -> None: ...

    def reset_prefix_cache(self) -> None: ...

    def resource_pressure(self) -> list[dict[str, object]]: ...

    def snapshot_session(self, session_id: int) -> SnapshotRef: ...

    def restore_session(self, reference: SnapshotRef) -> None: ...


def model_free_capabilities(
    *,
    block_size: int,
    supported_operation_types: tuple[OperationType, ...],
    supported_controls: tuple[RequestKind, ...] = (),
    num_layers: int = 1,
    num_blocks: int = 1,
    scratch_capacity_tokens: int = 0,
    max_latent_size: int = 0,
    latent_downsample: int = 1,
    encoder_cache_budget: int = 0,
    resource_classes: tuple[ResourceClass, ...] = (ResourceClass.KV_BLOCK,),
    adapter_mode: AdapterMode = AdapterMode.NONE,
    max_batch_operations: int = 1024,
    pipeline_depth: int,
    bytes_per_token: int = 1,
    model_spec_digest: str = "",
    weight_digest: str = "",
) -> EngineCaps:
    return EngineCaps(
        block_size=block_size,
        num_blocks=num_blocks,
        num_layers=num_layers,
        scratch_capacity_tokens=scratch_capacity_tokens,
        supported_operation_types=supported_operation_types,
        max_latent_size=max_latent_size,
        latent_downsample=latent_downsample,
        max_vae_grid_tokens=0,
        max_vit_grid_tokens=0,
        commit_marker_tokens=2,
        gen_rope_advance=2,
        max_cfg_branches=1,
        bytes_per_token=bytes_per_token,
        groups=(),
        kv_dtype="bfloat16",
        model_dtype="bfloat16",
        attention_backend="auto",
        quantization=None,
        rank=RankInfo(),
        pipeline_depth=int(pipeline_depth),
        encoder_cache_budget=encoder_cache_budget,
        supported_controls=supported_controls,
        adapter_mode=adapter_mode,
        execution_constraints=ExecutionConstraints(max_batch_operations),
        resource_classes=resource_classes,
        model_spec_digest=model_spec_digest,
        weight_digest=weight_digest,
    )


__all__ = [
    "Worker",
    "WorkerContract",
    "model_free_capabilities",
]
