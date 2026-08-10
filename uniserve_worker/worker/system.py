"""Composition root for model-free sampling and frame-materialization workers."""

from __future__ import annotations

import hashlib

import torch

from ..batch import (
    Batch,
    CacheCopy,
    CompletionReport,
    RecoveryPlacement,
    SamplingOwnership,
    SnapshotRef,
    WorkVariant,
)
from ..capabilities import (
    RankInfo,
    RequestKind,
    ResourceClass,
    WorkerCapabilities,
    configured_work_variants,
)
from ..execution import ModelExecutor
from ..foundation.errors import capability_mismatch, unsupported_control
from ..runtime.arena_capacity import operation_window, system_arena_capacity
from ..runtime.cache_pool import CachePool
from ..runtime.execution_trace import ExecutionPhase, ExecutionTrace, OperationTrace
from ..runtime.latent_store import LatentStore
from ..runtime.mover import Mover
from ..runtime.product_store import ProductRecord, ProductStore
from ..runtime.replay import ReplayStore
from ..runtime.request_session import SessionStore
from ..runtime.snapshot_store import SnapshotProvider
from ..runtime.transfer import Locator

_MAX_OPERATIONS = 1024


class SystemWorker:
    """Own reusable execution semantics that require no neural model."""

    def __init__(
        self,
        *,
        allowed_work_variants: frozenset[WorkVariant],
        block_size: int,
        transfer_backend: str,
        pipeline_depth: int,
        completion_payload_bytes: int,
        device: str,
        snapshot_dir: str | None = None,
    ) -> None:
        supported = frozenset({WorkVariant.MATERIALIZE})
        if not allowed_work_variants <= supported:
            raise ValueError("system worker received a model-backed work variant")
        if not allowed_work_variants:
            raise capability_mismatch("SystemWorker implements none of the requested work variants")
        if int(pipeline_depth) <= 0:
            raise capability_mismatch("worker pipeline depth must be positive")
        if int(completion_payload_bytes) < 1:
            raise ValueError("model-free completion payload capacity must be positive")
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
        advertised_work = configured_work_variants(
            tuple(value for value in WorkVariant if value in allowed_work_variants)
        )
        window = operation_window(int(pipeline_depth), _MAX_OPERATIONS)
        self._capabilities = WorkerCapabilities(
            block_size=int(block_size),
            num_blocks=2,
            num_layers=1,
            num_kv_heads=1,
            head_dim=1,
            scratch_capacity_tokens=0,
            supported_work=advertised_work,
            latent_page_units=0,
            num_latent_pages=0,
            latent_width=0,
            latent_dtype="",
            latent_downsample=1,
            max_vae_grid_tokens=0,
            max_vit_grid_tokens=0,
            max_latent_feature_bytes=0,
            max_vision_feature_bytes=0,
            commit_marker_tokens=2,
            gen_rope_advance=2,
            max_cfg_branches=1,
            bytes_per_token=1,
            groups=(),
            kv_dtype="bfloat16",
            model_dtype="bfloat16",
            attention_backend="auto",
            quantization=None,
            rank=RankInfo(),
            pipeline_depth=int(pipeline_depth),
            encoder_cache_budget=0,
            supported_controls=controls,
            max_batch_operations=_MAX_OPERATIONS,
            max_unresolved_window=window,
            incremental_kv_publication=True,
            tensorized_mixed=False,
            sampling_ownership=SamplingOwnership.DESIGNATED_RANK,
            resource_classes=(ResourceClass.ENCODER_OUTPUT,),
            model_identity="",
            weight_digest="",
        )
        arena = system_arena_capacity(
            pipeline_depth=int(pipeline_depth),
            max_operations=_MAX_OPERATIONS,
            completion_payload_bytes=int(completion_payload_bytes),
        )
        self.sessions = SessionStore()
        self.cache_pool = CachePool(
            num_layers=1,
            request_pages=int(self._capabilities.num_blocks),
            scratch_pages=0,
            page_size=int(self._capabilities.block_size),
            num_kv_heads=1,
            head_dim=1,
            device=device,
            dtype=torch.bfloat16,
        )
        self.latents = LatentStore()
        self.products = ProductStore(
            device_product_capacity=arena.device_products,
            device_product_byte_capacity=arena.device_product_bytes,
        )
        self.replay = ReplayStore()
        self.mover = Mover(
            transfer_backend=transfer_backend,
            transfer_byte_capacity=arena.transfer_bytes,
            transfer_ticket_capacity=arena.transfer_tickets,
            cross_process=True,
        )
        self.trace = ExecutionTrace(hashlib.sha256(b"uniserve-system-worker").hexdigest())
        self.executor = ModelExecutor(
            model=None,
            deployment=None,
            runner=None,
            attention=None,
            sessions=self.sessions,
            cache_pool=self.cache_pool,
            latents=self.latents,
            products=self.products,
            replay=self.replay,
            weights=None,
            mesh=None,
            transport=self.mover.transport,
            tokenizer=None,
            architecture_digest=None,
            weight_digest=None,
            allowed_work_variants=allowed_work_variants,
            trace=self.trace,
            pipeline_depth=pipeline_depth,
            completion_payload_bytes=completion_payload_bytes,
            cpu_task_capacity=arena.cpu_tasks,
            pinned_staging_capacity=arena.pinned_staging_bytes,
        )
        self.snapshot_provider: SnapshotProvider | None = None
        if snapshot_dir is not None:
            caps = self._capabilities
            self.snapshot_provider = SnapshotProvider(
                snapshot_dir,
                model_identity=self.trace.candidate_digest,
                weight_digest="",
                topology={
                    "rank": caps.rank.to_wire(),
                    "supported_work": [value.value for value in caps.supported_work],
                    "block_size": caps.block_size,
                },
                device=device,
                sessions=self.sessions,
                cache_pool=self.cache_pool,
                cache_publications=self.executor.cache_publications,
                latents=self.latents,
                products=self.products,
                replay=self.replay,
                transport=self.mover.transport,
            )

    @property
    def capabilities(self) -> WorkerCapabilities:
        return self._capabilities

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

    def copy_kv(self, copies: tuple[CacheCopy, ...]) -> None:
        del copies
        raise unsupported_control(RequestKind.COPY_KV.value)

    def release_products(self, handles: tuple[int, ...]) -> None:
        records = tuple(
            record for handle in handles if (record := self.products.get(int(handle))) is not None
        )
        self._release_records(records)
        self.products.release(tuple(int(handle) for handle in handles))
        self.sessions.discard_product_handles({int(handle) for handle in handles})

    def snapshot_session(self, placement: RecoveryPlacement) -> SnapshotRef:
        if self.snapshot_provider is None:
            raise unsupported_control(RequestKind.SNAPSHOT_SESSION.value)
        return self.snapshot_provider.snapshot_session(placement)

    def restore_session(
        self,
        reference: SnapshotRef,
        placement: RecoveryPlacement,
    ) -> None:
        if self.snapshot_provider is None:
            raise unsupported_control(RequestKind.RESTORE_SESSION.value)
        self.snapshot_provider.restore(reference, placement)

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
