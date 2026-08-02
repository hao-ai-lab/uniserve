"""Composition root for model-free sampling and frame-materialization workers."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from ..batch import Batch, CompletionReport, SnapshotRef
from ..capabilities import RequestKind, ResourceClass, work_variants_for_operation_types
from ..execution import ModelExecutor
from ..foundation.errors import unsupported_control
from ..runtime.execution_trace import ExecutionPhase, ExecutionTrace, OperationTrace
from ..runtime.kv_store import KvStore
from ..runtime.latent_store import LatentStore
from ..runtime.mover import Mover
from ..runtime.product_store import ProductRecord, ProductStore
from ..runtime.replay import ReplayStore
from ..runtime.request_session import SessionStore
from ..runtime.snapshot_store import SnapshotProvider
from ..runtime.transfer import Locator
from ..spec import OperationType
from .protocol import WorkerContract, model_free_capabilities


class SystemWorker:
    """Own reusable execution semantics that require no neural model."""

    def __init__(
        self,
        *,
        allowed_operation_types: frozenset[OperationType],
        block_size: int,
        transfer_backend: str,
        mooncake_device: str,
        mooncake_protocol: str,
        pipeline_depth: int,
        completion_payload_bytes: int,
        device: str,
        snapshot_dir: str | None = None,
        restore_snapshots: bool = False,
    ) -> None:
        supported = frozenset({OperationType.SEQUENCE_SAMPLE, OperationType.MATERIALIZE_FRAME})
        if not allowed_operation_types or not allowed_operation_types <= supported:
            raise ValueError("system worker received a model-backed operation type")
        controls: tuple[RequestKind, ...] = (
            RequestKind.DROP_SESSION,
            RequestKind.RELEASE_PRODUCTS,
        )
        if snapshot_dir is not None:
            controls = (
                *controls,
                RequestKind.SNAPSHOT_SESSION,
                RequestKind.RESTORE_SESSION,
            )
        declared = model_free_capabilities(
            block_size=int(block_size),
            supported_work=work_variants_for_operation_types(
                tuple(value for value in OperationType if value in allowed_operation_types)
            ),
            supported_controls=controls,
            resource_classes=(ResourceClass.ENCODER_OUTPUT,),
            pipeline_depth=pipeline_depth,
            completion_payload_bytes=completion_payload_bytes,
        )
        self._contract = WorkerContract.compile(
            declared,
            allowed_operation_types=allowed_operation_types,
            implemented_operation_types=allowed_operation_types,
            pipeline_depth=pipeline_depth,
            owner=type(self).__name__,
        )
        self.sessions = SessionStore()
        self.kv = KvStore()
        self.latents = LatentStore()
        self.products = ProductStore(
            device_product_byte_capacity=self._contract.capabilities.execution_constraints.route_capabilities[
                0
            ].credits.worker.latent_artifact_bytes,
        )
        self.replay = ReplayStore()
        self.mover = Mover(
            transfer_backend=transfer_backend,
            mooncake_device=mooncake_device,
            mooncake_protocol=mooncake_protocol,
            transfer_byte_capacity=self._contract.capabilities.execution_constraints.route_capabilities[
                0
            ].credits.worker.transfer_bytes,
            cross_process=True,
        )
        self.trace = ExecutionTrace(hashlib.sha256(b"uniserve-system-worker").hexdigest())
        self.executor = ModelExecutor(
            spec=None,
            deployment=None,
            runner=None,
            attention=None,
            sessions=self.sessions,
            kv=self.kv,
            latents=self.latents,
            products=self.products,
            replay=self.replay,
            adapters=None,
            mesh=None,
            transport=self.mover.transport,
            tokenizer=None,
            model_spec_digest=None,
            weight_digest=None,
            allowed_operation_types=allowed_operation_types,
            trace=self.trace,
            pipeline_depth=pipeline_depth,
            completion_payload_bytes=completion_payload_bytes,
            cpu_task_capacity=int(
                self._contract.capabilities.execution_constraints.route_capabilities[
                    0
                ].credits.worker.cpu_tasks
            ),
            pinned_staging_capacity=int(
                self._contract.capabilities.execution_constraints.route_capabilities[
                    0
                ].credits.worker.pinned_completion_staging_bytes
            ),
        )
        self.snapshot_provider: SnapshotProvider | None = None
        if snapshot_dir is not None:
            caps = self._contract.capabilities
            self.snapshot_provider = SnapshotProvider(
                snapshot_dir,
                model_spec_digest=self.trace.candidate_digest,
                weight_digest="",
                topology={
                    "rank": caps.rank.to_wire(),
                    "supported_work": [value.value for value in caps.supported_work],
                    "block_size": caps.block_size,
                },
                device=device,
                sessions=self.sessions,
                kv=self.kv,
                latents=self.latents,
                products=self.products,
                replay=self.replay,
                adapters=None,
                transport=self.mover.transport,
            )
            restored = self.snapshot_provider.restore_latest() if restore_snapshots else ()
            self._contract = replace(
                self._contract,
                capabilities=replace(
                    self._contract.capabilities,
                    restored_snapshots=tuple(
                        sorted(
                            restored, key=lambda reference: reference.version.request_key.session_id
                        )
                    ),
                ),
            )

    @property
    def contract(self) -> WorkerContract:
        return self._contract

    def warmup(self) -> None:
        self.executor.complete_startup()

    def execute(self, batch: Batch) -> CompletionReport:
        return self.executor.execute(batch)

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
        del copies
        raise unsupported_control(RequestKind.COPY_KV.value)

    def load_adapter(self, adapter_id: int, adapter_path: str) -> None:
        del adapter_id, adapter_path
        raise unsupported_control(RequestKind.LOAD_ADAPTER.value)

    def unload_adapter(self, adapter_id: int) -> None:
        del adapter_id
        raise unsupported_control(RequestKind.UNLOAD_ADAPTER.value)

    def release_products(self, handles: tuple[int, ...]) -> None:
        records = tuple(
            record for handle in handles if (record := self.products.get(int(handle))) is not None
        )
        self._release_records(records)
        self.products.release(tuple(int(handle) for handle in handles))
        self.sessions.discard_product_handles({int(handle) for handle in handles})

    def reset_prefix_cache(self) -> None:
        raise unsupported_control(RequestKind.RESET_PREFIX_CACHE.value)

    def snapshot_session(self, session_id: int) -> SnapshotRef:
        if self.snapshot_provider is None:
            raise unsupported_control(RequestKind.SNAPSHOT_SESSION.value)
        return self.snapshot_provider.snapshot_session(int(session_id))

    def restore_session(self, reference: SnapshotRef) -> None:
        if self.snapshot_provider is None:
            raise unsupported_control(RequestKind.RESTORE_SESSION.value)
        self.snapshot_provider.restore(reference)

    def resource_pressure(self) -> list[dict[str, object]]:
        total = int(self.products.encoder_cache_budget)
        used = self.products.encoder_output_count()
        if used > total:
            raise RuntimeError("system product residency exceeds its declared capacity")
        return [
            {
                "class": "encoder_output",
                "total": total,
                "used": used,
                "evictable": 0,
                "free": total - used,
            }
        ]

    def close(self) -> None:
        self.executor.close()
        self.mover.close()

    def _release_records(self, records: tuple[ProductRecord, ...]) -> None:
        for record in records:
            if record.locator:
                self.mover.transport.release(Locator.from_wire_json(record.locator))


__all__ = ["SystemWorker"]
