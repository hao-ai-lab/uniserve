"""Model-owned resources and the public kind-module execution surface."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch

from uniserve_worker.execution.batch import (
    DeviceSelected,
    FixedCheckpoint,
    LogicalLengths,
    Operation,
    ProductRef,
    RequestKey,
    RunKind,
)
from uniserve_worker.execution.forward_batch import AttentionSelection, ModelPhase
from uniserve_worker.execution.graph_bucket import GraphBucket
from uniserve_worker.execution.model_runner import ForwardResult, ModelRunner
from uniserve_worker.execution.output import OutputPool
from uniserve_worker.execution.rows import (
    ForwardRow,
    LaneState,
    LatentExecution,
    OperationIdentity,
)
from uniserve_worker.execution.trace import ExecutionTrace
from uniserve_worker.foundation.errors import invalid_descriptor, unsupported_setup
from uniserve_worker.loader.weight_set import WeightSet
from uniserve_worker.models.generation import GenerationPipeline
from uniserve_worker.models.inputs import ImageProcessor
from uniserve_worker.models.runtime import ExecutionModel, WorkerDeployment
from uniserve_worker.nn.mesh import DeviceMesh
from uniserve_worker.runtime.cache_pool import CachePool
from uniserve_worker.runtime.cpu import CpuPool
from uniserve_worker.runtime.device_events import DeviceEventPool
from uniserve_worker.runtime.device_products import DeviceProductRead, DeviceProducts
from uniserve_worker.runtime.encoder_cache import EncoderCache, EncoderRead
from uniserve_worker.runtime.latent_pool import LatentPool
from uniserve_worker.runtime.req_to_token_pool import ReqToTokenPool
from uniserve_worker.runtime.request import RequestDraft, RequestPool, RequestRuntime
from uniserve_worker.runtime.runtime_states import RuntimeStates
from uniserve_worker.transfer.connector import CachePublications
from uniserve_worker.transfer.tickets import Locator, Transport


@dataclass(slots=True)
class ExecutionResources:
    runner: ModelRunner | None
    model: ExecutionModel
    deployment: WorkerDeployment
    attention: AttentionSelection | None
    requests: RequestPool
    runtime_states: RuntimeStates | None
    cache_pool: CachePool | None
    req_to_token_pool: ReqToTokenPool | None
    cache_publications: CachePublications | None
    latent_pool: LatentPool | None
    _media_mux: Any | None
    _media_output_ring: Any | None
    device_products: DeviceProducts
    encoder_cache: EncoderCache
    _device_events: DeviceEventPool
    _outputs: OutputPool
    _cpu_tasks: CpuPool
    weights: WeightSet
    mesh: DeviceMesh
    transport: Transport | None
    tokenizer: Any | None
    model_name: str
    weight_version: int
    allowed_work_variants: frozenset[RunKind]
    mixed_buckets: frozenset[GraphBucket]
    trace: ExecutionTrace
    _device: torch.device
    _generation_device: torch.device
    _qualified_mixed_buckets: set[GraphBucket] = field(default_factory=set)
    _collective_history: OrderedDict[int, object] = field(default_factory=OrderedDict)
    _transport_publications: dict[OperationIdentity, tuple[Locator, ...]] = field(
        default_factory=dict
    )
    _flow_prefix_slots: dict[RequestKey, set[int]] = field(default_factory=dict)

    @staticmethod
    def operation_identity(operation: Operation) -> OperationIdentity:
        return operation.request_key, int(operation.op_id)

    def request_row(self, scope: LaneState, request_id: int) -> RequestDraft:
        try:
            return scope.request_rows[int(request_id)]
        except KeyError:
            raise invalid_descriptor(
                f"lane has no request row for request {request_id}"
            ) from None

    def generation(self) -> GenerationPipeline:
        value = self.model.generation
        if not isinstance(value, GenerationPipeline):
            raise invalid_descriptor("operation requires model generation behavior")
        return value

    def image_processor(self) -> ImageProcessor:
        value = self.model.image_processor
        if not isinstance(value, ImageProcessor):
            raise invalid_descriptor("operation requires model image processing")
        return value

    def media_mux(self) -> Any:
        if self._media_mux is None:
            raise unsupported_setup("operation requires media mux resources")
        return self._media_mux

    def media_output_ring(self) -> Any:
        if self._media_output_ring is None:
            raise unsupported_setup("operation requires a rank-zero media output ring")
        return self._media_output_ring

    def cache_coordinates(
        self,
        operation: Operation,
        scope: LaneState,
        *,
        group_id: int = 0,
    ) -> tuple[int, int, int, int]:
        request = self.request_row(scope, operation.request_key.request_id)
        slot = int(request.request_pool_idx)
        rows = scope.forward_rows.get(self.operation_identity(operation), ())
        descriptor = next(
            (row for row in rows if int(row.request_pool_index) == slot),
            None,
        )
        parent = self.parent_runtime(operation, request)
        visible = int(parent.kv_visible_len) if descriptor is None else int(descriptor.seq_len)
        pool = self.req_to_token_pool
        if pool is None:
            raise unsupported_setup("operation requires request-to-token storage")
        pool.pages(slot, group_id)
        capacity = pool.allocated_length(slot)
        if visible > capacity:
            raise invalid_descriptor("operation visibility exceeds scheduler block table")
        return slot, int(group_id), visible, capacity

    def latent_row(self, operation: Operation, scope: LaneState) -> LatentExecution:
        row = scope.latent_rows.get(self.operation_identity(operation))
        if row is None:
            raise invalid_descriptor("trajectory operation has no staged latent placement")
        return row

    def require_latent_pool(self) -> LatentPool:
        if self.latent_pool is None:
            raise unsupported_setup("operation requires a physical latent pool")
        return self.latent_pool

    def logical_lengths(
        self,
        operation: Operation,
        request: RequestDraft,
        cache: tuple[int, int, int, int] | None,
        *,
        latent_len: int | None = None,
        computed_len: int | None = None,
    ) -> LogicalLengths:
        parent = self.parent_runtime(operation, request)
        if cache is None:
            visible = parent.kv_visible_len
            computed = parent.kv_computed_len
        else:
            _slot, _group, visible, _capacity = cache
            computed = visible if computed_len is None else int(computed_len)
        return LogicalLengths(
            token_len=request.logical_position,
            kv_visible_len=visible,
            kv_computed_len=computed,
            latent_len=request.flow_step if latent_len is None else int(latent_len),
        )

    @staticmethod
    def output_generations(operation: Operation) -> tuple[int, ...]:
        return tuple(int(reference.generation) for reference in operation.outputs)

    def operation_device(self, operation: Operation) -> torch.device:
        if operation.kind in {
            RunKind.DIFFUSION_PREPARE,
            RunKind.DIFFUSION_STEP,
            RunKind.DIFFUSION_DECODE,
            RunKind.DIFFUSION_FINALIZE,
        }:
            return self._generation_device
        return self._device

    def phase_device(self, phase: ModelPhase) -> torch.device:
        if phase in {ModelPhase.ENCODE_LATENT, ModelPhase.DECODE_LATENT}:
            return torch.device(self.deployment.generation_device or self.deployment.device)
        return torch.device(self.deployment.device)

    def consume_device_product(
        self,
        reference: ProductRef,
        scope: LaneState,
        *,
        consumer_op_id: int,
        device: torch.device | str | None = None,
    ) -> DeviceProductRead:
        candidate = scope.transferred_device_products.get(reference)
        if candidate is not None:
            return self.device_products.consume_candidate(
                candidate,
                consumer_op_id=consumer_op_id,
                device=device,
            )
        return self.device_products.consume(
            reference,
            consumer_op_id=consumer_op_id,
            device=device,
        )

    def consume_encoder_feature(
        self,
        reference: ProductRef,
        scope: LaneState,
        *,
        consumer_op_id: int,
        device: torch.device | str | None = None,
    ) -> EncoderRead:
        candidate = scope.transferred_encoder_features.get(reference)
        if candidate is not None:
            return self.encoder_cache.consume_candidate(
                candidate,
                consumer_op_id=consumer_op_id,
                device=device,
            )
        return self.encoder_cache.consume(
            reference,
            consumer_op_id=consumer_op_id,
            device=device,
        )

    def broadcast_tp_selection(self, value: torch.Tensor) -> torch.Tensor:
        if self.mesh.tp_size <= 1:
            return value
        transport = self.mesh.transport("tp")
        return transport.broadcast(value, src=0)

    def run_observed_forward_group(
        self,
        tasks: tuple[ForwardRow, ...],
        scope: LaneState,
    ) -> ForwardResult:
        from .step import _run_forward_group

        result = _run_forward_group(self, tasks, scope)
        scope.observations.append(result.observation)
        return result

    @staticmethod
    def record_component(scope: LaneState, name: str, started_ns: int) -> None:
        elapsed_us = max(0, (time.perf_counter_ns() - int(started_ns)) // 1000)
        scope.component_us[name] = scope.component_us.get(name, 0) + elapsed_us

    def parent_runtime(self, operation: Operation, request: RequestDraft) -> RequestRuntime:
        parent = operation.parent
        point = parent.point
        runtime = (
            request.execution_runtime_for_operation(parent.op_id, 1)
            if isinstance(point, DeviceSelected)
            else None
        )
        if runtime is None:
            selected = request.resolve_version(parent)
            runtime = None if selected is None else request.runtime_for(selected)
        if runtime is None:
            raise invalid_descriptor("operation parent has no resolved runtime state")
        return runtime

    @staticmethod
    def fixed_parent(operation: Operation) -> FixedCheckpoint:
        point = operation.parent.point
        if not isinstance(point, FixedCheckpoint):
            raise invalid_descriptor("operation names a device parent; depth one commits fixed")
        return point


__all__ = ["ExecutionResources"]
