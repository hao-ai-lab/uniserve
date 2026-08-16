"""Composition of one configured canonical worker process."""

from __future__ import annotations

import logging
from dataclasses import replace

from ..loader.weight_set import WeightSet
from ..models.identity import architecture_identity
from ..models.runtime import ExecutionModel
from .config import WorkerLaunchConfig
from .plan import resolve_worker_plan

logger = logging.getLogger(__name__)


def assemble_worker(config: WorkerLaunchConfig):
    """Materialize exactly one worker from a validated launch configuration."""

    plan = resolve_worker_plan(config.worker_kind)
    from ..backends.attention import resolve_attention_selection
    from ..backends.triton import configure_triton_toolchain
    from ..nn.mesh import TensorParallelSpec
    from ..nn.placement import place_towers
    from ..server.distributed import build_device_mesh

    configure_triton_toolchain()

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
    if config.use_stub_model:
        from ..server.stub import StubModel, stub_deployment

        stub = StubModel()
        model: ExecutionModel = stub
        deployment = replace(
            stub_deployment(
                config.resources.block_size,
                max_batch_tokens=config.resources.max_batch_tokens,
            ),
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
        architecture_digest = architecture_identity(
            stub.architecture,
            {"architecture": stub.architecture},
        )
        weight_digest = None
        weights = WeightSet.from_module(stub)
        weight_sidecars: tuple[str, ...] = ("config.json",)
    else:
        from .model_loader import WorkerModelLoadRequest, load_worker_model

        if config.model is None:
            raise RuntimeError("validated model worker is missing model configuration")
        loaded = load_worker_model(
            WorkerModelLoadRequest(
                model_path=config.model.path,
                device=config.placement.device,
                block_size=config.resources.block_size,
                max_batch_tokens=config.resources.max_batch_tokens,
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
                load=config.load,
            )
        )
        model = loaded.model
        deployment = loaded.deployment
        tokenizer = loaded.tokenizer
        architecture_digest = loaded.identity.architecture_digest
        weight_digest = loaded.identity.weight_digest
        weights = loaded.weights
        weight_sidecars = loaded.entry.sidecars

    place_towers(model, mesh)
    weights = WeightSet.from_module(model, digest=weight_digest or weights.digest)

    from ..worker import Worker

    attention = resolve_attention_selection(
        deployment.attention_backend or "auto",
        tuning=config.execution.flashinfer,
        block_size=deployment.block_size,
    )

    return Worker(
        model,
        mesh=mesh,
        deployment=deployment,
        attention=attention,
        execution=config.execution,
        tokenizer=tokenizer,
        allowed_work_variants=plan.allowed_work_variants,
        transfer_backend=config.data_plane.backend,
        cross_process=config.worker_kind.value != "full",
        architecture_digest=architecture_digest,
        weight_digest=weight_digest,
        weights=weights,
        weight_sidecars=weight_sidecars,
        pipeline_depth=config.ipc.pipeline_depth,
        completion_payload_bytes=config.ipc.max_payload_bytes,
        snapshot_dir=config.snapshot_dir,
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
