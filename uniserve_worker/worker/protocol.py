"""Execution interface hosted by the worker IPC server."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from ..contracts.caps import Caps, ExecutionConstraints
from ..foundation.errors import capability_mismatch
from ..foundation.sizing import DEFAULT_BLOCK_SIZE, DEFAULT_MAX_BATCH_OPS

__all__ = [
    "BaseWorker",
    "ResultPolicy",
    "Worker",
    "WorkerContract",
    "model_free_capabilities",
]


class ResultPolicy(StrEnum):
    """When the serve loop may defer device-to-host result materialization."""

    DEFER_WHEN_AVAILABLE = "defer_when_available"
    SYNCHRONOUS = "synchronous"


@dataclass(frozen=True)
class WorkerContract:
    """Immutable capabilities and delivery semantics of an assembled worker."""

    capabilities: Caps
    result_policy: ResultPolicy

    @classmethod
    def compile(
        cls,
        declared_capabilities: Caps,
        *,
        allowed_ops: frozenset[str],
        pipeline_depth: int,
        result_policy: ResultPolicy,
        owner: str,
    ) -> "WorkerContract":
        effective_ops = tuple(op for op in declared_capabilities.supported_ops if op in allowed_ops)
        if not effective_ops:
            raise capability_mismatch(
                f"{owner} implements none of the requested operations {sorted(allowed_ops)!r}"
            )
        capabilities = replace(
            declared_capabilities,
            supported_ops=effective_ops,
            pipeline_depth=max(1, int(pipeline_depth)),
        )
        return cls(capabilities=capabilities, result_policy=result_policy)


@runtime_checkable
class Worker(Protocol):
    @property
    def contract(self) -> WorkerContract: ...

    def caps(self) -> Caps: ...

    def execute(
        self,
        batch: Mapping[str, Any],
        *,
        defer_text_cpu_results: bool = False,
    ) -> dict[str, Any]: ...

    def drop_request(self, request_id: int) -> None: ...

    def resource_pressure(self) -> list[dict[str, Any]]: ...


class BaseWorker(ABC):
    """Shared contract storage for concrete worker implementations."""

    def __init__(self, *, block_size: int = DEFAULT_BLOCK_SIZE) -> None:
        self.block_size = int(block_size)
        self._contract: WorkerContract | None = None

    @property
    def contract(self) -> WorkerContract:
        if self._contract is None:
            raise RuntimeError(f"{type(self).__name__} contract is not initialized")
        return self._contract

    def caps(self) -> Caps:
        return replace(self.contract.capabilities)

    def resource_pressure(self) -> list[dict[str, Any]]:
        return []

    def _compile_contract(
        self,
        declared_capabilities: Caps,
        *,
        allowed_ops: frozenset[str],
        pipeline_depth: int,
        result_policy: ResultPolicy,
    ) -> None:
        self._contract = WorkerContract.compile(
            declared_capabilities,
            allowed_ops=allowed_ops,
            pipeline_depth=pipeline_depth,
            result_policy=result_policy,
            owner=type(self).__name__,
        )

    @abstractmethod
    def execute(
        self,
        batch: Mapping[str, Any],
        *,
        defer_text_cpu_results: bool = False,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def drop_request(self, request_id: int) -> None: ...


def model_free_capabilities(
    *,
    block_size: int,
    supported_ops: tuple[str, ...],
    supported_controls: tuple[str, ...] = (),
    num_layers: int = 1,
    max_latent_size: int = 0,
    latent_downsample: int = 1,
    encoder_cache_budget: int | None = None,
    resource_classes: tuple[str, ...] = ("kv_block",),
    **overrides: Any,
) -> Caps:
    """Build the minimum wire-compatible capability record for a model-free worker."""

    return Caps(
        block_size=int(block_size),
        num_blocks=int(overrides.pop("num_blocks", 1)),
        num_layers=int(num_layers),
        scratch_capacity_tokens=int(overrides.pop("scratch_capacity_tokens", 0)),
        supported_ops=tuple(supported_ops),
        max_latent_size=int(max_latent_size),
        latent_downsample=int(latent_downsample),
        bytes_per_token=int(overrides.pop("bytes_per_token", 1)),
        supported_controls=tuple(supported_controls),
        adapter_mode=str(overrides.pop("adapter_mode", "none")),
        execution_constraints=ExecutionConstraints(
            max_batch_ops=int(overrides.pop("max_batch_ops", DEFAULT_MAX_BATCH_OPS))
        ),
        resource_classes=tuple(resource_classes),
        encoder_cache_budget=encoder_cache_budget,
        **overrides,
    )
