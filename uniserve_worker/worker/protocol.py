"""Typed process-boundary contract implemented by assembled workers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from ..batch import Batch, CompletionReport, SamplingOwnership, SnapshotRef, WorkVariant
from ..capabilities import (
    AdapterMode,
    CreditVector,
    EngineCaps,
    ExecutionConstraints,
    RankInfo,
    RequestKind,
    ResourceClass,
    RouteCreditLimits,
    RouteExecutionCapability,
    work_variants_for_operation_types,
)
from ..foundation.errors import capability_mismatch
from ..spec import OperationType


@dataclass(frozen=True, slots=True)
class WorkerContract:
    capabilities: EngineCaps
    effective_operation_types: frozenset[OperationType]

    @classmethod
    def compile(
        cls,
        declared_capabilities: EngineCaps,
        *,
        allowed_operation_types: frozenset[OperationType],
        implemented_operation_types: frozenset[OperationType],
        pipeline_depth: int,
        owner: str,
    ) -> WorkerContract:
        if int(pipeline_depth) <= 0:
            raise capability_mismatch("worker pipeline depth must be positive")
        effective = tuple(
            operation_type
            for operation_type in OperationType
            if operation_type in allowed_operation_types
            and operation_type in implemented_operation_types
        )
        effective_work = work_variants_for_operation_types(effective)
        if not effective_work:
            raise capability_mismatch(
                f"{owner} implements none of the requested operation types "
                f"{sorted(value.value for value in allowed_operation_types)!r}"
            )
        selected_work = set(effective_work)
        declared_work = set(declared_capabilities.supported_work)
        system_work = selected_work - declared_work
        declared_routes = declared_capabilities.execution_constraints.route_capabilities
        if system_work and len(declared_routes) != 1:
            raise capability_mismatch(
                f"{owner} must assign shared system work to one explicit execution route"
            )
        route_capabilities = tuple(
            replace(
                capability,
                supported_work=tuple(
                    variant
                    for variant in WorkVariant
                    if variant in selected_work
                    and (variant in capability.supported_work or variant in system_work)
                ),
            )
            for capability in declared_routes
            if selected_work.intersection(capability.supported_work) or system_work
        )
        return cls(
            capabilities=replace(
                declared_capabilities,
                supported_work=effective_work,
                pipeline_depth=int(pipeline_depth),
                execution_constraints=replace(
                    declared_capabilities.execution_constraints,
                    route_capabilities=route_capabilities,
                ),
            ),
            effective_operation_types=frozenset(effective),
        )


@runtime_checkable
class Worker(Protocol):
    @property
    def contract(self) -> WorkerContract: ...

    def warmup(self) -> None: ...

    def execute(self, batch: Batch) -> CompletionReport: ...

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
    supported_work: tuple[WorkVariant, ...],
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
    completion_payload_bytes: int,
) -> EngineCaps:
    if int(completion_payload_bytes) < 1:
        raise ValueError("model-free completion payload capacity must be positive")
    slots = int(pipeline_depth) * int(max_batch_operations)
    window = min(slots, max(2, int(pipeline_depth)))
    per_operation_staging = int(completion_payload_bytes) + 256
    completion_words = 4 * int(max_batch_operations) + (int(completion_payload_bytes) + 3) // 4
    completion_arena_bytes = int(pipeline_depth) * completion_words * 8
    route_credits = RouteCreditLimits(
        per_request=CreditVector(
            registered_operations=window,
            execution_slots=window,
            completion_slots=window,
            device_products=5 * window,
            kv_pages=int(num_blocks),
            rollback_deltas=17 * window,
            latent_artifact_bytes=(1 << 20) * 5 * window,
            pinned_completion_staging_bytes=per_operation_staging * window,
            transfer_bytes=(1 << 20) * window,
            transfer_tickets=window,
            cpu_tasks=2,
            output_journal_bytes=1 << 30,
        ),
        worker=CreditVector(
            registered_operations=slots,
            execution_slots=slots,
            completion_slots=slots,
            device_products=5 * slots,
            kv_pages=int(num_blocks),
            rollback_deltas=17 * slots,
            latent_artifact_bytes=(1 << 20) * 5 * slots,
            pinned_completion_staging_bytes=max(
                completion_arena_bytes,
                per_operation_staging * window,
            ),
            transfer_bytes=(1 << 20) * slots,
            transfer_tickets=slots,
            cpu_tasks=256,
            output_journal_bytes=(1 << 30) * max_batch_operations,
        ),
    )
    return EngineCaps(
        block_size=block_size,
        num_blocks=num_blocks,
        num_layers=num_layers,
        scratch_capacity_tokens=scratch_capacity_tokens,
        supported_work=supported_work,
        max_latent_size=max_latent_size,
        latent_downsample=latent_downsample,
        max_vae_grid_tokens=0,
        max_vit_grid_tokens=0,
        max_latent_feature_bytes=0,
        max_vision_feature_bytes=0,
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
        execution_constraints=ExecutionConstraints(
            max_batch_operations=max_batch_operations,
            max_speculative_points=17,
            device_sequence_lengths=True,
            device_append_offsets=True,
            incremental_kv_publication=True,
            route_capabilities=(
                RouteExecutionCapability(
                    route=0,
                    supported_work=supported_work,
                    tensorized_mixed=False,
                    sampling_ownership=SamplingOwnership.DESIGNATED_RANK,
                    preemptible=False,
                    credits=route_credits,
                ),
            ),
        ),
        resource_classes=resource_classes,
        model_spec_digest=model_spec_digest,
        weight_digest=weight_digest,
    )


__all__ = [
    "Worker",
    "WorkerContract",
    "model_free_capabilities",
]
