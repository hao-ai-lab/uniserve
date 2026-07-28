"""Composition root for one canonical model-backed worker."""

from __future__ import annotations

import logging
from dataclasses import replace

from torch import nn

from ..batch import Batch, ExecutionResult
from ..capabilities import RequestKind
from ..execution import ModelExecutor, ModelRunner
from ..forward import AttentionSelection
from ..foundation.errors import capability_mismatch
from ..foundation.runtime_config import ExecutionConfig, graph_memory_budget_bytes
from ..foundation.sizing import device_total_bytes
from ..nn.mesh import DeviceMesh
from ..runtime.adapter_store import AdapterStore
from ..runtime.capabilities import resolve_capabilities
from ..runtime.execution_trace import ExecutionPhase, ExecutionTrace, OperationTrace
from ..runtime.graph_store import GraphStore
from ..runtime.kv_store import KvStore
from ..runtime.latent_store import LatentStore
from ..runtime.mesh_store import MeshStore
from ..runtime.mover import Mover
from ..runtime.product_store import ProductStore
from ..runtime.replay import ReplayStore
from ..runtime.request_session import SessionStore
from ..runtime.residency import ResidencyStore
from ..runtime.snapshot_store import SnapshotProvider, SnapshotRef
from ..spec import DeploymentOverlay, ModelSpec, OperationType, RouteRowKind, resolved_digest
from .protocol import WorkerContract

logger = logging.getLogger(__name__)


class ModelWorker:
    """Own one ready model and all system authorities around its raw forward."""

    def __init__(
        self,
        model: nn.Module,
        *,
        mesh: DeviceMesh,
        model_spec: ModelSpec,
        deployment: DeploymentOverlay,
        attention: AttentionSelection,
        execution: ExecutionConfig,
        tokenizer: object | None,
        allowed_operation_types: frozenset[OperationType],
        defer_sampling: bool = False,
        transfer_backend: str = "local",
        mooncake_device: str,
        mooncake_protocol: str,
        cross_process: bool = False,
        model_spec_digest: str | None = None,
        weight_digest: str | None = None,
        pipeline_depth: int,
        snapshot_dir: str | None = None,
        restore_snapshots: bool = False,
    ) -> None:
        if not isinstance(model, nn.Module) or type(model).forward is nn.Module.forward:
            raise capability_mismatch("model worker requires nn.Module.forward(ForwardBatch)")
        if not isinstance(model_spec, ModelSpec) or not isinstance(deployment, DeploymentOverlay):
            raise capability_mismatch("model worker requires canonical model and deployment specs")
        self.model = model
        self.model_spec = model_spec
        self.deployment = deployment
        self.adapter_store = AdapterStore(
            model,
            weights=model_spec.weights,
            base_digest=weight_digest,
        )
        self.weight_digest = self.adapter_store.base.digest
        self.model_spec_digest = model_spec_digest or resolved_digest(model_spec, deployment)
        if self.model_spec_digest != resolved_digest(model_spec, deployment):
            raise capability_mismatch("loaded model-spec digest does not match its declarations")
        declared = resolve_capabilities(
            model_spec,
            deployment,
            model_spec_digest=self.model_spec_digest,
            weight_digest=self.weight_digest,
        )
        if snapshot_dir is not None:
            declared = replace(
                declared,
                supported_controls=(
                    *declared.supported_controls,
                    RequestKind.SNAPSHOT_SESSION,
                    RequestKind.RESTORE_SESSION,
                ),
            )
        self._contract = WorkerContract.compile(
            declared,
            allowed_operation_types=allowed_operation_types,
            system_operation_types=frozenset(
                {OperationType.SEQUENCE_SAMPLE, OperationType.MATERIALIZE_FRAME}
            ),
            pipeline_depth=pipeline_depth,
            owner=type(self).__name__,
        )
        self.residency = ResidencyStore.from_spec(
            model_spec,
            self._contract.capabilities,
            deployment.resources,
            device=deployment.device,
        )
        self.kv = KvStore(self.residency.kv)
        self.sessions = SessionStore()
        self.latents = LatentStore(
            capacity_tokens=int(self._contract.capabilities.max_latent_size),
            downsample=(1 if model_spec.flow is None else int(model_spec.flow.latent_downsample)),
        )
        self.products = ProductStore(encoder_cache_budget=model_spec.inputs.encoder_cache_budget)
        self.replay = ReplayStore()
        self.mover = Mover(
            transfer_backend=transfer_backend,
            mooncake_device=mooncake_device,
            mooncake_protocol=mooncake_protocol,
            cross_process=bool(cross_process),
        )
        self.graphs = GraphStore(
            enabled=execution.cuda_graph,
            prefill_enabled=execution.prefill_cuda_graph,
            cache=model_spec.cache,
            block_size=deployment.block_size,
            spec_digest=self.model_spec_digest,
            memory_budget_bytes=graph_memory_budget_bytes(device_total_bytes(deployment.device)),
            decode_batch_sizes=execution.cuda_graph_warmup_batches,
            decode_context_blocks=self._decode_context_blocks(),
            prefill_token_sizes=execution.prefill_cuda_graph_warmup_tokens,
        )
        self._execution = execution
        self.trace = ExecutionTrace(self.model_spec_digest)
        self.executor = ModelExecutor(
            spec=model_spec,
            deployment=deployment,
            runner=ModelRunner(model, self.graphs, self.trace),
            attention=attention,
            sessions=self.sessions,
            kv=self.kv,
            latents=self.latents,
            products=self.products,
            replay=self.replay,
            adapters=self.adapter_store,
            mesh=MeshStore(mesh),
            transport=self.mover.transport,
            tokenizer=tokenizer,
            model_spec_digest=self.model_spec_digest,
            weight_digest=self.weight_digest,
            allowed_operation_types=frozenset(
                self._contract.capabilities.supported_operation_types
            ),
            trace=self.trace,
            pipeline_depth=pipeline_depth,
            defer_sampling=defer_sampling,
        )
        self.snapshot_provider: SnapshotProvider | None = None
        if snapshot_dir is not None:
            caps = self._contract.capabilities
            self.snapshot_provider = SnapshotProvider(
                snapshot_dir,
                model_spec_digest=self.model_spec_digest,
                weight_digest=self.weight_digest,
                topology={
                    "rank": caps.rank.to_wire(),
                    "model_scope": deployment.model_scope,
                    "block_size": caps.block_size,
                    "num_blocks": caps.num_blocks,
                    "num_layers": caps.num_layers,
                },
                device=deployment.device,
                sessions=self.sessions,
                kv=self.kv,
                latents=self.latents,
                products=self.products,
                replay=self.replay,
                adapters=self.adapter_store,
                transport=self.mover.transport,
            )
            restored = (
                self.snapshot_provider.restore_latest() if restore_snapshots else ()
            )
            self._contract = replace(
                self._contract,
                capabilities=replace(
                    self._contract.capabilities,
                    restored_sessions=tuple(
                        sorted(reference.session_id for reference in restored)
                    ),
                ),
            )
            logger.info("restored %d durable worker sessions", len(restored))

    @property
    def contract(self) -> WorkerContract:
        return self._contract

    def _decode_context_blocks(self) -> int:
        pool = self.kv.pool
        if pool is None:
            return 0
        max_tokens = max(
            (
                int(route.shape.max_tokens_per_row)
                for route in self.model_spec.routes
                if RouteRowKind.TOKEN in route.row_kinds
            ),
            default=0,
        )
        if max_tokens < 1:
            return 0
        blocks = (max_tokens + int(self.deployment.block_size) - 1) // int(
            self.deployment.block_size
        )
        return min(blocks, int(pool.leasable_num_blocks))

    def execute(self, batch: Batch) -> ExecutionResult:
        result = self.executor.execute(batch)
        if self.snapshot_provider is not None:
            result = self.snapshot_provider.snapshot_execution(
                {operation.session_id for operation in batch.operations},
                result,
            )
        return result

    def warmup(self) -> None:
        """Pay first-use kernel JIT before the worker is reachable.

        The ``fa4_cute`` attention backend JIT-compiles its CUTLASS kernels the
        first time each variant runs (prefill/varlen, paged decode, and the
        flow packed path), costing tens of seconds on the first real request.
        Running one representative prefill + decode (and, for image models, one
        flow step) through the real execution path here moves that compilation
        ahead of readiness, so the first served request is warm. Every failure
        is swallowed: a warmup problem must never block serving.
        """

        import torch

        if torch.device(self.deployment.device).type != "cuda":
            return
        try:
            self._warmup_sequence()
        except Exception:  # noqa: BLE001 - warmup must never block serving.
            logger.warning("sequence warmup failed; first request stays cold", exc_info=True)
        try:
            self._warmup_flow()
        except Exception:  # noqa: BLE001 - warmup must never block serving.
            logger.warning("flow warmup failed; first flow step stays cold", exc_info=True)

    def _warmup_sequence(self) -> None:
        from ..batch import (
            Admission,
            Batch,
            KvAllocation,
            KvLeaseDelta,
            OperationEnvelope,
            SamplingParams,
            SequenceAdmission,
            SequenceMode,
            SequenceOperation,
            TokenInput,
            TokenPolicy,
            TokenSource,
        )

        types = self._contract.capabilities.supported_operation_types
        if OperationType.SEQUENCE_EXTEND not in types:
            return
        pool = self.kv.pool
        if pool is None or self.sessions.session_ids():
            return
        if (
            self._execution.cuda_graph
            and self._execution.prefill_cuda_graph
            and self._execution.prefill_cuda_graph_warmup
        ):
            self._warmup_prefill_graphs()
        configured = (
            self._execution.cuda_graph_warmup_batches
            if (
                self._execution.cuda_graph
                and self._execution.cuda_graph_warmup
                and OperationType.SEQUENCE_DECODE in types
            )
            else (1,)
        )
        batch_sizes = tuple(
            sorted(
                {
                    int(value)
                    for value in configured
                    if 0 < int(value) <= int(pool.leasable_num_blocks)
                },
                reverse=True,
            )
        )
        if not batch_sizes:
            return
        session_ids = tuple(range(1, max(batch_sizes) + 1))
        admissions = tuple(
            Admission.create(
                session_id,
                sequence=SequenceAdmission(
                    sampling=SamplingParams(temperature=0.0, ignore_eos=True),
                    kv=KvAllocation(block_ids=(block_id,), prefix_len=0, group_id=0),
                ),
            )
            for block_id, session_id in enumerate(session_ids)
        )
        by_session = {admission.session_id: admission for admission in admissions}
        extend = SequenceOperation(
            SequenceMode.EXTEND,
            KvLeaseDelta(),
            (0, 1),
            TokenPolicy(),
            TokenInput((0,)),
        )
        step_id = 1
        versions = {session_id: 0 for session_id in session_ids}
        try:
            self.executor.execute(
                Batch(
                    step_id,
                    admissions,
                    (),
                    tuple(
                        OperationEnvelope.create(
                            session_id=session_id,
                            epoch=1,
                            op_id=1,
                            base_version=0,
                            admission_digest=by_session[session_id].digest,
                            model_spec_digest=self.model_spec_digest,
                            weight_digest=self.weight_digest,
                            operation=extend,
                        )
                        for session_id in session_ids
                    ),
                )
            )
            for session_id in session_ids:
                versions[session_id] = 1
            if OperationType.SEQUENCE_DECODE not in types:
                return
            repeats = 2 if self._execution.cuda_graph and self._execution.cuda_graph_warmup else 1
            decode = SequenceOperation(
                SequenceMode.DECODE,
                KvLeaseDelta(),
                (1, 2),
                TokenPolicy(),
                TokenInput((0,), source=TokenSource.LAST_SAMPLED),
            )
            for batch_size in batch_sizes:
                selected = session_ids[:batch_size]
                for _ in range(repeats):
                    step_id += 1
                    operations = []
                    for session_id in selected:
                        version = versions[session_id]
                        position = version
                        operations.append(
                            OperationEnvelope.create(
                                session_id=session_id,
                                epoch=1,
                                op_id=version + 1,
                                base_version=version,
                                admission_digest=by_session[session_id].digest,
                                model_spec_digest=self.model_spec_digest,
                                weight_digest=self.weight_digest,
                                operation=replace(
                                    decode,
                                    position=(position, position + 1),
                                ),
                            )
                        )
                    self.executor.execute(Batch(step_id, (), (), tuple(operations)))
                    for session_id in selected:
                        versions[session_id] += 1
        finally:
            import torch

            device = torch.device(self.deployment.device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            for session_id in session_ids:
                if self.sessions.peek(session_id) is not None:
                    self.drop_session(session_id)

    def _warmup_prefill_graphs(self) -> None:
        import torch

        from ..batch import (
            Admission,
            Batch,
            KvAllocation,
            KvLeaseDelta,
            OperationEnvelope,
            SamplingParams,
            SequenceAdmission,
            SequenceMode,
            SequenceOperation,
            TokenInput,
            TokenPolicy,
        )

        pool = self.kv.pool
        if pool is None or self.sessions.session_ids():
            return
        max_route_tokens = max(
            (
                int(route.shape.max_tokens_per_row)
                for route in self.model_spec.routes
                if RouteRowKind.TOKEN in route.row_kinds
            ),
            default=0,
        )
        capacity = min(
            max_route_tokens,
            int(pool.leasable_num_blocks) * int(pool.block_size),
        )
        token_buckets = tuple(
            sorted(
                {
                    int(value)
                    for value in self._execution.prefill_cuda_graph_warmup_tokens
                    if 0 < int(value) <= capacity
                },
                reverse=True,
            )
        )
        if not token_buckets:
            return
        logger.info("warming %d paged-prefill CUDA graph token buckets", len(token_buckets))
        session_id = 1
        step_id = 0
        device = torch.device(self.deployment.device)
        for token_count in token_buckets:
            block_count = (token_count + int(pool.block_size) - 1) // int(pool.block_size)
            allocation = KvAllocation(
                block_ids=tuple(range(block_count)),
                prefix_len=0,
                group_id=0,
            )
            operation = SequenceOperation(
                SequenceMode.EXTEND,
                KvLeaseDelta(),
                (0, token_count),
                TokenPolicy(),
                TokenInput((0,) * token_count),
            )
            for _ in range(2):
                admission = Admission.create(
                    session_id,
                    sequence=SequenceAdmission(
                        sampling=SamplingParams(temperature=0.0, ignore_eos=True),
                        kv=allocation,
                    ),
                )
                step_id += 1
                try:
                    self.executor.execute(
                        Batch(
                            step_id,
                            (admission,),
                            (),
                            (
                                OperationEnvelope.create(
                                    session_id=session_id,
                                    epoch=1,
                                    op_id=1,
                                    base_version=0,
                                    admission_digest=admission.digest,
                                    model_spec_digest=self.model_spec_digest,
                                    weight_digest=self.weight_digest,
                                    operation=operation,
                                ),
                            ),
                        )
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                finally:
                    if self.sessions.peek(session_id) is not None:
                        self.drop_session(session_id)

    def _warmup_image_geometry(self) -> tuple[int, int]:
        """Largest square image whose latent grid fits the declared capacity.

        Warmup exercises the real denoise path, so its geometry has to obey the
        same latent limits a served request does. Deriving it from the resolved
        capabilities keeps one warmup valid for every image model.
        """

        import math

        caps = self._contract.capabilities
        downsample = max(1, int(caps.latent_downsample))
        capacity = int(caps.max_latent_size)
        if int(caps.max_vae_grid_tokens) > 0:
            capacity = min(capacity, int(caps.max_vae_grid_tokens))
        side = max(1, math.isqrt(max(1, capacity)))
        return side * downsample, side * downsample

    def _warmup_flow(self) -> None:
        from ..batch import (
            Admission,
            Batch,
            FlowAdmission,
            FlowOperation,
            Guidance,
            ImageParams,
            OperationEnvelope,
        )

        if (
            OperationType.FLOW not in self._contract.capabilities.supported_operation_types
            or self.model_spec.flow is None
        ):
            return
        session_id = 2
        height, width = self._warmup_image_geometry()
        admission = Admission.create(
            session_id,
            flow=FlowAdmission(ImageParams(steps=50, height=height, width=width, seed=0)),
        )
        flow = FlowOperation(
            latent_handle=1,
            position=0,
            start_step=0,
            step_count=1,
            conditioning_position=0,
            conditioning=None,
            guidance=Guidance(
                int(self.model_spec.flow.max_cfg_branches), 4.0, 1.0, "global", 0.0, (0.0, 1.0)
            ),
            image_prompt="",
        )
        self.execute(
            Batch(
                3,
                (admission,),
                (),
                (
                    OperationEnvelope.create(
                        session_id=session_id,
                        epoch=1,
                        op_id=1,
                        base_version=0,
                        admission_digest=admission.digest,
                        model_spec_digest=self.model_spec_digest,
                        weight_digest=self.weight_digest,
                        operation=flow,
                    ),
                ),
            )
        )
        self.drop_session(session_id)

    def drop_session(self, session_id: int) -> None:
        session_id = int(session_id)
        session = self.sessions.peek(session_id)
        self._release_records(self.products.session_records(session_id))
        self.products.drop(session_id)
        self.latents.drop_session(session_id)
        self.replay.drop_session(session_id)
        self.kv.drop(session_id)
        self.sessions.drop(session_id)
        if self.snapshot_provider is not None:
            self.snapshot_provider.drop_session(session_id)
        if session is not None:
            self.trace.emit(
                ExecutionPhase.CLEANUP,
                (
                    OperationTrace(
                        session_id=session.session_id,
                        epoch=session.epoch,
                        op_id=0 if session.last_op_id is None else session.last_op_id,
                        version=session.version,
                    ),
                ),
            )

    def copy_kv(self, copies: tuple[tuple[int, int], ...]) -> None:
        self.kv.copy(copies)
        if self.snapshot_provider is not None:
            session_ids = set(self.sessions.session_ids())
            if session_ids:
                self.snapshot_provider.snapshot_sessions(session_ids)

    def load_adapter(self, adapter_id: int, adapter_path: str) -> None:
        if self.deployment.adapter_mode == "none":
            raise capability_mismatch("this worker does not declare adapter controls")
        if self.sessions.session_ids():
            raise capability_mismatch("adapter changes require no live sessions")
        count = self.adapter_store.load(int(adapter_id), str(adapter_path))
        if self.snapshot_provider is not None:
            self.snapshot_provider.snapshot_global()
        logger.info("loaded adapter %s with %d parameter overrides", adapter_id, count)

    def unload_adapter(self, adapter_id: int) -> None:
        if self.deployment.adapter_mode == "none":
            raise capability_mismatch("this worker does not declare adapter controls")
        if self.sessions.session_ids():
            raise capability_mismatch("adapter changes require no live sessions")
        self.adapter_store.unload(int(adapter_id))
        if self.snapshot_provider is not None:
            self.snapshot_provider.snapshot_global()

    def release_products(self, handles: tuple[int, ...]) -> None:
        records = tuple(
            record for handle in handles if (record := self.products.get(int(handle))) is not None
        )
        self._release_records(records)
        self.products.release(tuple(int(handle) for handle in handles))
        affected = self.sessions.discard_product_handles({int(handle) for handle in handles})
        if self.snapshot_provider is not None and affected:
            self.snapshot_provider.snapshot_sessions(affected)

    def reset_prefix_cache(self) -> None:
        return None

    def snapshot_session(self, session_id: int) -> SnapshotRef:
        if self.snapshot_provider is None:
            raise capability_mismatch("this worker has no configured snapshot provider")
        return self.snapshot_provider.snapshot_session(int(session_id))

    def restore_session(self, reference: SnapshotRef) -> None:
        if self.snapshot_provider is None:
            raise capability_mismatch("this worker has no configured snapshot provider")
        self.snapshot_provider.restore(reference)

    def resource_pressure(self) -> list[dict[str, object]]:
        caps = self._contract.capabilities
        counts = {
            "kv_block": self.kv.resident_block_count(),
            "scratch": self.kv.scratch_token_count(),
            "image_latent": self.latents.resident_token_count(),
            "encoder_output": self.products.encoder_output_count(),
            "adapter": self.adapter_store.loaded_count(),
        }
        totals = {
            "kv_block": int(caps.num_blocks),
            "scratch": int(caps.scratch_capacity_tokens),
            "image_latent": int(caps.max_latent_size),
            "encoder_output": int(caps.encoder_cache_budget),
            "adapter": 1,
        }
        return [
            _pressure(value.value, counts[value.value], totals[value.value])
            for value in caps.resource_classes
        ]

    def close(self) -> None:
        self.graphs.close()
        self.mover.close()

    def _release_records(self, records: tuple[object, ...]) -> None:
        from ..runtime.product_store import ProductRecord
        from ..runtime.transfer import Locator

        for record in records:
            if isinstance(record, ProductRecord) and record.locator:
                self.mover.transport.release(Locator.from_wire_json(record.locator))


def _pressure(resource_class: str, used: int, total: int) -> dict[str, object]:
    if used < 0 or total < 0 or used > total:
        raise RuntimeError(
            f"resource pressure invariant failed for {resource_class}: used={used}, total={total}"
        )
    return {
        "class": resource_class,
        "total": total,
        "used": used,
        "evictable": 0,
        "free": total - used,
    }


__all__ = ["ModelWorker"]
