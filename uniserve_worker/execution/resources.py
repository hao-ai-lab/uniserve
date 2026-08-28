"""Model-owned resources and the public kind-module execution surface."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from uniserve_worker.capabilities import GraphBucket
from uniserve_worker.execution.batch import (
    DevicePoint,
    FixedPoint,
    ForwardMode,
    LogicalLengths,
    Operation,
    ProductRef,
    RequestKey,
)
from uniserve_worker.execution.forward_batch import AttentionSelection, ModelPhase
from uniserve_worker.execution.model_runner import ForwardResult, ModelRunner
from uniserve_worker.execution.rows import (
    ForwardRow,
    LatentExecution,
    OperationIdentity,
    PartitionState,
)
from uniserve_worker.execution.trace import ExecutionTrace
from uniserve_worker.foundation.errors import capability_mismatch, invalid_descriptor
from uniserve_worker.loader.weight_set import WeightSet
from uniserve_worker.models.generation import GenerationPipeline
from uniserve_worker.models.inputs import ImageProcessor
from uniserve_worker.models.minimax_h3 import MiniMaxH3Model
from uniserve_worker.models.minimax_h3.execution import H3MuxCoordinator, H3OutputRing
from uniserve_worker.models.runtime import ExecutionModel, WorkerDeployment
from uniserve_worker.nn.mesh import DeviceMesh
from uniserve_worker.runtime.cache_pool import CachePool
from uniserve_worker.runtime.device_events import DeviceEventPool
from uniserve_worker.runtime.device_products import DeviceProductRead, DeviceProducts
from uniserve_worker.runtime.encoder_cache import EncoderCache, EncoderRead
from uniserve_worker.runtime.latent_pool import LatentPool
from uniserve_worker.runtime.req_to_token_pool import ReqToTokenPool
from uniserve_worker.runtime.runtime_states import RuntimeStates
from uniserve_worker.server.cpu_tasks import BoundedCpuTaskPool
from uniserve_worker.server.request_state import RequestRow, RequestRuntime, RequestTable
from uniserve_worker.transfer.connector import CachePublications
from uniserve_worker.transfer.tickets import Locator, Transport


@dataclass(slots=True)
class ExecutionResources:
    runner: ModelRunner | None
    model: ExecutionModel | MiniMaxH3Model
    deployment: WorkerDeployment
    attention: AttentionSelection | None
    requests: RequestTable
    runtime_states: RuntimeStates | None
    cache_pool: CachePool | None
    req_to_token_pool: ReqToTokenPool | None
    cache_publications: CachePublications | None
    latent_pool: LatentPool | None
    _h3_mux: H3MuxCoordinator | None
    _h3_output_ring: H3OutputRing | None
    _media_spool: Path | None
    device_products: DeviceProducts
    encoder_cache: EncoderCache
    _device_events: DeviceEventPool
    _cpu_tasks: BoundedCpuTaskPool
    weights: WeightSet
    mesh: DeviceMesh
    transport: Transport | None
    tokenizer: Any | None
    architecture_digest: str
    weight_digest: str
    allowed_work_variants: frozenset[ForwardMode]
    mixed_buckets: frozenset[GraphBucket]
    trace: ExecutionTrace
    _device: torch.device
    _generation_device: torch.device
    _qualified_mixed_buckets: set[GraphBucket] = field(default_factory=set)
    _collective_history: OrderedDict[int, str] = field(default_factory=OrderedDict)
    _transport_publications: dict[OperationIdentity, tuple[Locator, ...]] = field(
        default_factory=dict
    )
    _flow_prefix_slots: dict[RequestKey, set[int]] = field(default_factory=dict)

    @staticmethod
    def operation_identity(operation: Operation) -> OperationIdentity:
        return operation.request_key, int(operation.op_id)

    def request_row(self, scope: PartitionState, session_id: int) -> RequestRow:
        try:
            return scope.request_rows[int(session_id)]
        except KeyError:
            raise invalid_descriptor(
                f"partition has no request row for session {session_id}"
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

    def h3_mux(self) -> H3MuxCoordinator:
        if not isinstance(self.model, MiniMaxH3Model) or self._h3_mux is None:
            raise capability_mismatch("operation requires MiniMax H3 mux resources")
        return self._h3_mux

    def h3_output_ring(self) -> H3OutputRing:
        if not isinstance(self.model, MiniMaxH3Model) or self._h3_output_ring is None:
            raise capability_mismatch("operation requires a rank-zero H3 output ring")
        return self._h3_output_ring

    def cache_coordinates(
        self,
        operation: Operation,
        scope: PartitionState,
        *,
        group_id: int = 0,
    ) -> tuple[int, int, int, int]:
        request = self.request_row(scope, operation.request_key.session_id)
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
            raise capability_mismatch("operation requires request-to-token storage")
        pool.pages(slot, group_id)
        capacity = pool.allocated_length(slot)
        if visible > capacity:
            raise invalid_descriptor("operation visibility exceeds scheduler block table")
        return slot, int(group_id), visible, capacity

    def latent_row(self, operation: Operation, scope: PartitionState) -> LatentExecution:
        row = scope.latent_rows.get(self.operation_identity(operation))
        if row is None:
            raise invalid_descriptor("trajectory operation has no staged latent placement")
        return row

    def require_latent_pool(self) -> LatentPool:
        if self.latent_pool is None:
            raise capability_mismatch("operation requires a physical latent pool")
        return self.latent_pool

    def logical_lengths(
        self,
        operation: Operation,
        session: RequestRow,
        cache: tuple[int, int, int, int] | None,
        *,
        latent_len: int | None = None,
        computed_len: int | None = None,
    ) -> LogicalLengths:
        parent = self.parent_runtime(operation, session)
        if cache is None:
            visible = parent.kv_visible_len
            computed = parent.kv_computed_len
        else:
            _slot, _group, visible, _capacity = cache
            computed = visible if computed_len is None else int(computed_len)
        return LogicalLengths(
            token_len=session.logical_position,
            kv_visible_len=visible,
            kv_computed_len=computed,
            latent_len=session.flow_step if latent_len is None else int(latent_len),
        )

    @staticmethod
    def output_generations(operation: Operation) -> tuple[int, ...]:
        return tuple(int(reference.generation) for reference in operation.outputs)

    def operation_device(self, operation: Operation) -> torch.device:
        if operation.work in {
            ForwardMode.GEN_TRANSITION,
            ForwardMode.GEN_FLOW,
            ForwardMode.GEN_DECODE,
            ForwardMode.MATERIALIZE,
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
        scope: PartitionState,
        *,
        consumer_op_id: int,
        producer_plan_digest: str | None = None,
        device: torch.device | str | None = None,
    ) -> DeviceProductRead:
        candidate = scope.transferred_device_products.get(reference)
        if candidate is not None:
            return self.device_products.consume_candidate(
                candidate,
                consumer_op_id=consumer_op_id,
                producer_plan_digest=producer_plan_digest,
                device=device,
            )
        return self.device_products.consume(
            reference,
            consumer_op_id=consumer_op_id,
            producer_plan_digest=producer_plan_digest,
            device=device,
        )

    def consume_encoder_feature(
        self,
        reference: ProductRef,
        scope: PartitionState,
        *,
        consumer_op_id: int,
        producer_plan_digest: str | None = None,
        device: torch.device | str | None = None,
    ) -> EncoderRead:
        candidate = scope.transferred_encoder_features.get(reference)
        if candidate is not None:
            return self.encoder_cache.consume_candidate(
                candidate,
                consumer_op_id=consumer_op_id,
                producer_plan_digest=producer_plan_digest,
                device=device,
            )
        return self.encoder_cache.consume(
            reference,
            consumer_op_id=consumer_op_id,
            producer_plan_digest=producer_plan_digest,
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
        scope: PartitionState,
    ) -> ForwardResult:
        from .step import _run_forward_group

        result = _run_forward_group(self, tasks, scope)
        scope.observations.append(result.observation)
        return result

    @staticmethod
    def record_component(scope: PartitionState, name: str, started_ns: int) -> None:
        elapsed_us = max(0, (time.perf_counter_ns() - int(started_ns)) // 1000)
        scope.component_us[name] = scope.component_us.get(name, 0) + elapsed_us

    def parent_runtime(self, operation: Operation, request: RequestRow) -> RequestRuntime:
        parent = operation.parent
        point = parent.point
        runtime = (
            request.execution_runtime_for_operation(parent.producer_op_id, point.point_index)
            if isinstance(point, DevicePoint) and point.selected_point is None
            else None
        )
        if runtime is None:
            selected = request.resolve_version(parent)
            runtime = None if selected is None else request.runtime_for(selected)
        if runtime is None:
            raise invalid_descriptor("operation parent has no resolved runtime state")
        return runtime

    @staticmethod
    def fixed_parent(operation: Operation) -> FixedPoint:
        point = operation.parent.point
        if not isinstance(point, FixedPoint):
            raise invalid_descriptor("operation names a device parent; depth one commits fixed")
        return point


__all__ = ["ExecutionResources"]
