"""Composition of one configured canonical worker process."""

from __future__ import annotations

import logging
from dataclasses import replace

from torch import nn

from ..spec import ModelSpec
from .config import WorkerLaunchConfig
from .plan import WorkerImplementation, resolve_worker_plan

logger = logging.getLogger(__name__)


def assemble_worker(config: WorkerLaunchConfig):
    """Materialize exactly one worker from a validated launch configuration."""

    plan = resolve_worker_plan(config.worker_kind)
    if plan.implementation is WorkerImplementation.SYSTEM:
        from ..worker.system import SystemWorker

        return SystemWorker(
            allowed_operation_types=plan.allowed_operation_types,
            block_size=config.resources.block_size,
            transfer_backend=config.data_plane.backend,
            pipeline_depth=config.ipc.pipeline_depth,
            completion_payload_bytes=config.ipc.max_payload_bytes,
            device=config.placement.device,
            snapshot_dir=config.snapshot_dir,
            restore_snapshots=config.restore_snapshots,
        )

    from ..backends.attention import resolve_attention_selection
    from ..nn.mesh import TensorParallelSpec
    from ..nn.placement import place_towers
    from ..server.distributed import build_device_mesh

    mesh = build_device_mesh(
        tp_rank=config.placement.tp_rank,
        tp_size=config.placement.tp_size,
        device=config.placement.device,
        tower_devices=config.placement.tower_devices,
        tower_primary=0,
        tp_backend=config.placement.tp_backend,
        tp_init_method=config.placement.tp_init_method,
    )
    parallel = TensorParallelSpec.from_mesh(mesh)
    if plan.model_scope is None:
        raise RuntimeError("model worker plan has no model materialization scope")

    if config.use_stub_model:
        from ..server.stub import StubModel, stub_deployment

        stub = StubModel()
        model: nn.Module = stub
        model_spec: ModelSpec = stub.spec
        deployment = replace(
            stub_deployment(config.resources.block_size),
            device=config.placement.device,
            model_scope=plan.model_scope.value,
            tp_rank=config.placement.tp_rank,
            tp_size=config.placement.tp_size,
            kv_token_capacity=config.resources.kv_token_capacity,
            generation_kv_capacity_tokens=config.resources.generation_kv_capacity_tokens,
            model_dtype=config.execution.model_dtype,
            kv_cache_dtype=config.execution.kv_cache_dtype,
            kv_memory_fraction=config.execution.kv_memory_fraction,
            generation_device=(
                config.placement.tower_devices[1]
                if config.placement.tower_devices is not None
                else None
            ),
        )
        tokenizer = None
        model_spec_digest = None
        weight_digest = None
    else:
        from .model_loader import WorkerModelLoadRequest, load_worker_model

        if config.model is None:
            raise RuntimeError("validated model worker is missing model configuration")
        loaded = load_worker_model(
            WorkerModelLoadRequest(
                model_path=config.model.path,
                device=config.placement.device,
                block_size=config.resources.block_size,
                kv_token_capacity=config.resources.kv_token_capacity,
                attention_backend=config.model.attention_backend,
                execution=config.execution,
                parallel=parallel,
                scope=plan.model_scope,
                generation_kv_capacity_tokens=config.resources.generation_kv_capacity_tokens,
                generation_device=(
                    config.placement.tower_devices[1]
                    if config.placement.tower_devices is not None
                    else None
                ),
            )
        )
        model = loaded.model
        model_spec = loaded.spec
        deployment = loaded.overlay
        tokenizer = loaded.tokenizer
        model_spec_digest = loaded.resolved_digest
        weight_digest = loaded.weight_digest

    place_towers(model, mesh)

    from ..worker.model import ModelWorker

    attention = resolve_attention_selection(
        deployment.attention_backend or "auto",
        tuning=config.execution.flashinfer,
        block_size=deployment.block_size,
    )

    return ModelWorker(
        model,
        mesh=mesh,
        model_spec=model_spec,
        deployment=deployment,
        attention=attention,
        execution=config.execution,
        tokenizer=tokenizer,
        allowed_operation_types=plan.allowed_operation_types,
        defer_sampling=config.data_plane.defer_sampling,
        transfer_backend=config.data_plane.backend,
        cross_process=config.worker_kind.value != "full",
        model_spec_digest=model_spec_digest,
        weight_digest=weight_digest,
        pipeline_depth=config.ipc.pipeline_depth,
        completion_payload_bytes=config.ipc.max_payload_bytes,
        snapshot_dir=config.snapshot_dir,
        restore_snapshots=config.restore_snapshots,
    )


def run_worker(config: WorkerLaunchConfig) -> None:
    """Open the IPC endpoint, assemble the worker, and serve until shutdown."""

    from ..server.app import WorkerServer
    from ..server.ipc import WorkerIpcEndpoint

    endpoint = WorkerIpcEndpoint(
        config.ipc.service_name,
        max_payload=config.ipc.max_payload_bytes,
        max_inflight=config.ipc.max_inflight,
    )
    logger.info(
        "worker IPC endpoint opened",
        extra={
            "service": config.ipc.service_name,
            "max_payload_bytes": config.ipc.max_payload_bytes,
            "max_inflight": config.ipc.max_inflight,
            "worker_kind": config.worker_kind.value,
        },
    )
    worker = assemble_worker(config)
    # Admission begins only after the configured first-use kernel work succeeds
    # and the worker opens a clean serving collective epoch.
    worker.warmup()
    WorkerServer(worker, endpoint).serve()


__all__ = ["assemble_worker", "run_worker"]
