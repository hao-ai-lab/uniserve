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
    """Owns the model, runtime stores, lanes, and transfer services used for execution."""

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
        """Form the lane-local identity from request generation and operation id."""

        return operation.request_key, int(operation.op_id)

    def request_row(self, scope: LaneState, request_id: int) -> RequestDraft:
        """Return the unique staged request draft for an identifier within the current lane."""

        try:
            return scope.request_rows[int(request_id)]
        except KeyError:
            raise invalid_descriptor(f"lane has no request row for request {request_id}") from None

    def generation(self) -> GenerationPipeline:
        """Require the model's diffusion generation contract for the active operation."""

        value = self.model.generation
        if not isinstance(value, GenerationPipeline):
            raise invalid_descriptor("operation requires model generation behavior")
        return value

    def image_processor(self) -> ImageProcessor:
        """Require the model's image preprocessing contract for the active operation."""

        value = self.model.image_processor
        if not isinstance(value, ImageProcessor):
            raise invalid_descriptor("operation requires model image processing")
        return value

    def media_mux(self) -> Any:
        """Require the rank-local coordinator that finalizes encoded media artifacts."""

        if self._media_mux is None:
            raise unsupported_setup("operation requires media mux resources")
        return self._media_mux

    def media_output_ring(self) -> Any:
        """Require output-owner bounded storage for asynchronous encoded media output."""

        if self._media_output_ring is None:
            raise unsupported_setup("operation requires the media output owner's ring")
        return self._media_output_ring

    def cache_coordinates(
        self,
        operation: Operation,
        scope: LaneState,
        *,
        group_id: int = 0,
    ) -> tuple[int, int, int, int]:
        """Resolve visible, computed, and physical KV coordinates for an operation and cache group."""

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
        """Resolve the staged physical latent placement for an operation in this lane."""

        row = scope.latent_rows.get(self.operation_identity(operation))
        if row is None:
            raise invalid_descriptor("trajectory operation has no staged latent placement")
        return row

    def require_latent_pool(self) -> LatentPool:
        """Require the worker-owned resident latent page pool."""

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
        """Construct post-operation semantic lengths from request state and optional cache coordinates."""

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
        """List logical generations in descriptor output order."""

        return tuple(int(reference.generation) for reference in operation.outputs)

    def operation_device(self, operation: Operation) -> torch.device:
        """Return the model or generation device assigned to an operation kind."""

        if operation.kind in {
            RunKind.DIFFUSION_PREPARE,
            RunKind.DIFFUSION_STEP,
            RunKind.DIFFUSION_DECODE,
            RunKind.DIFFUSION_FINALIZE,
        }:
            return self._generation_device
        return self._device

    def phase_device(self, phase: ModelPhase) -> torch.device:
        """Select the generation device for latent codecs and the model device otherwise."""

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
        """Acquire a device product for one consumer and retain its read lease in the lane."""

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
        """Acquire an encoder feature for one consumer and retain its read lease in the lane."""

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
        """Broadcast device-selected scalar values from tensor-parallel rank zero."""

        if self.mesh.tp_size <= 1:
            return value
        transport = self.mesh.get_group("tp")
        return transport.broadcast(value, src=0)

    def run_observed_forward_group(
        self,
        tasks: tuple[ForwardRow, ...],
        scope: LaneState,
    ) -> ForwardResult:
        """Run a forward group and accumulate its path, token, timing, and attention observations."""

        from .step import _run_forward_group

        result = _run_forward_group(self, tasks, scope)
        scope.observations.append(result.observation)
        return result

    @staticmethod
    def record_component(scope: LaneState, name: str, started_ns: int) -> None:
        """Accumulate elapsed microseconds under a lane-scoped execution component."""

        elapsed_us = max(0, (time.perf_counter_ns() - int(started_ns)) // 1000)
        scope.component_us[name] = scope.component_us.get(name, 0) + elapsed_us

    def parent_runtime(self, operation: Operation, request: RequestDraft) -> RequestRuntime:
        """Resolve the operation’s parent checkpoint to its request runtime state."""

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
        """Require a host-resolved parent checkpoint for a depth-one state transition."""

        point = operation.parent.point
        if not isinstance(point, FixedCheckpoint):
            raise invalid_descriptor("operation names a device parent; depth one commits fixed")
        return point


__all__ = ["ExecutionResources"]
