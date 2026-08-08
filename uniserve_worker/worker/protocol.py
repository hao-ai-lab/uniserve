"""Typed process-boundary contract implemented by assembled workers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from ..batch import Batch, CompletionReport, SamplingOwnership, SnapshotRef, WorkVariant
from ..capabilities import (
    EngineCaps,
    ExecutionConstraints,
    RankInfo,
    RequestKind,
    ResourceClass,
    RouteExecutionCapability,
)
from ..foundation.errors import capability_mismatch
from ..runtime.arena_capacity import operation_window


@dataclass(frozen=True, slots=True)
class WorkerContract:
    capabilities: EngineCaps
    effective_work_variants: frozenset[WorkVariant]

    @classmethod
    def compile(
        cls,
        declared_capabilities: EngineCaps,
        *,
        allowed_work_variants: frozenset[WorkVariant],
        implemented_work_variants: frozenset[WorkVariant],
        pipeline_depth: int,
        owner: str,
    ) -> WorkerContract:
        if int(pipeline_depth) <= 0:
            raise capability_mismatch("worker pipeline depth must be positive")
        # The worker executes every implemented leaf it is admitted to, including
        # unadvertised protocol leaves (Verify, Draft); the wire capability only
        # advertises the configured subset, which the declared capabilities have
        # already narrowed. Admission stays unfiltered so those leaves execute.
        effective = allowed_work_variants & implemented_work_variants
        if not effective:
            raise capability_mismatch(
                f"{owner} implements none of the requested work variants "
                f"{sorted(value.value for value in allowed_work_variants)!r}"
            )
        selected_work = set(effective)
        advertised_work = selected_work.intersection(declared_capabilities.supported_work)
        if not advertised_work:
            raise capability_mismatch(f"{owner} advertises no executable work")
        declared_routes = declared_capabilities.execution_constraints.route_capabilities
        route_capabilities = tuple(
            replace(
                capability,
                supported_work=tuple(
                    variant
                    for variant in WorkVariant
                    if variant in advertised_work and variant in capability.supported_work
                ),
            )
            for capability in declared_routes
            if advertised_work.intersection(capability.supported_work)
        )
        return cls(
            capabilities=replace(
                declared_capabilities,
                supported_work=tuple(
                    variant for variant in WorkVariant if variant in advertised_work
                ),
                pipeline_depth=int(pipeline_depth),
                execution_constraints=replace(
                    declared_capabilities.execution_constraints,
                    route_capabilities=route_capabilities,
                ),
            ),
            effective_work_variants=frozenset(effective),
        )


@runtime_checkable
class Worker(Protocol):
    @property
    def contract(self) -> WorkerContract: ...

    def warmup(self) -> None: ...

    def execute(self, batch: Batch) -> CompletionReport: ...

    def drop_session(self, session_id: int) -> None: ...

    def copy_kv(self, copies: tuple[tuple[int, int], ...]) -> None: ...

    def release_products(self, handles: tuple[int, ...]) -> None: ...

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
    max_batch_operations: int = 1024,
    pipeline_depth: int,
    bytes_per_token: int = 1,
    model_spec_digest: str = "",
    weight_digest: str = "",
    completion_payload_bytes: int,
) -> EngineCaps:
    if int(completion_payload_bytes) < 1:
        raise ValueError("model-free completion payload capacity must be positive")
    window = operation_window(
        int(pipeline_depth),
        int(max_batch_operations),
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
        execution_constraints=ExecutionConstraints(
            max_batch_operations=max_batch_operations,
            max_speculative_points=1,
            max_unresolved_window=window,
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
                    max_unresolved_window=window,
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
