"""Transactional composition of one configured worker process."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .config import WorkerLaunchConfig
from .plan import WorkerImplementation, resolve_worker_plan

if TYPE_CHECKING:
    from ..contracts.model_family import ModelFamilyDescriptor
    from ..contracts.model_protocols import UniModel
    from ..worker.protocol import Worker

logger = logging.getLogger(__name__)


def assemble_worker(config: WorkerLaunchConfig) -> Worker:
    """Materialize exactly one worker from a validated launch configuration."""

    from ..backends.attention import init_attention_backends
    from ..foundation.runtime_config import set_execution_config
    from ..nn.mesh import set_current_mesh
    from ..server.distributed import build_device_mesh

    set_execution_config(config.execution)
    registered_backends = init_attention_backends()
    logger.debug(
        "attention backends registered: %s",
        registered_backends,
    )
    device_mesh = build_device_mesh(
        tp_rank=config.placement.tp_rank,
        tp_size=config.placement.tp_size,
        device=config.placement.device,
        tower_devices=config.placement.tower_devices,
        tower_primary=0,
        tp_backend=config.placement.tp_backend,
        tp_init_method=config.placement.tp_init_method,
    )
    set_current_mesh(device_mesh)

    plan = resolve_worker_plan(config.worker_kind)
    if plan.implementation is WorkerImplementation.SAMPLER:
        from ..runtime.tensor_store import TensorStore
        from ..runtime.transfer import make_transport
        from ..worker.sampler import SamplerWorker

        return SamplerWorker(
            tensor_store=TensorStore(transport=make_transport(config.data_plane.backend)),
            allowed_ops=plan.allowed_ops,
            pipeline_depth=config.ipc.pipeline_depth,
            result_policy=plan.result_policy,
            block_size=config.resources.block_size,
        )
    if plan.implementation is WorkerImplementation.FRAME_ACCUMULATOR:
        from ..worker.frame_accumulator import FrameAccumulatorWorker

        return FrameAccumulatorWorker(
            allowed_ops=plan.allowed_ops,
            pipeline_depth=config.ipc.pipeline_depth,
            result_policy=plan.result_policy,
            block_size=config.resources.block_size,
        )

    if config.use_stub_model:
        from ..server.stub import StubUniModel

        model: UniModel = StubUniModel()
        descriptor: ModelFamilyDescriptor | None = None
        spec_digest: str | None = None
    else:
        from .model_loader import (
            WorkerModelLoadRequest,
            load_worker_model,
        )

        if config.model is None or plan.model_scope is None:
            raise RuntimeError("validated model worker is missing model configuration")
        loaded_model = load_worker_model(
            WorkerModelLoadRequest(
                model_path=config.model.path,
                device=config.placement.device,
                block_size=config.resources.block_size,
                kv_token_capacity=config.resources.kv_token_capacity,
                attention_backend=config.model.attention_backend,
                scope=plan.model_scope,
                generation_kv_capacity_tokens=(config.resources.generation_kv_capacity_tokens),
                tp_rank=config.placement.tp_rank,
                tp_size=config.placement.tp_size,
            )
        )
        model = loaded_model.model
        descriptor = loaded_model.descriptor
        spec_digest = loaded_model.resolved_digest

    if plan.implementation is WorkerImplementation.ENCODER:
        from ..worker.encoder import EncoderWorker

        return EncoderWorker(
            model,
            allowed_ops=plan.allowed_ops,
            pipeline_depth=config.ipc.pipeline_depth,
            result_policy=plan.result_policy,
            block_size=config.resources.block_size,
        )
    if plan.implementation is WorkerImplementation.MODEL:
        from ..worker.model import ModelWorker

        if plan.model_scope is None:
            raise RuntimeError("model worker plan has no model scope")
        return ModelWorker(
            model,
            allowed_ops=plan.allowed_ops,
            pipeline_depth=config.ipc.pipeline_depth,
            result_policy=plan.result_policy,
            block_size=config.resources.block_size,
            kv_token_capacity=config.resources.kv_token_capacity,
            attention_backend=(
                config.model.attention_backend if config.model is not None else None
            ),
            defer_sampling=config.data_plane.defer_sampling,
            transfer_backend=config.data_plane.backend,
            model_scope=plan.model_scope,
            family_descriptor=descriptor,
            simulation=config.use_stub_model,
            spec_digest=spec_digest,
        )
    raise AssertionError(f"unhandled worker implementation {plan.implementation!r}")


def run_worker(config: WorkerLaunchConfig) -> None:
    """Open the IPC endpoint, assemble the worker, and serve until shutdown."""

    from ..server.app import WorkerServer
    from ..server.ipc import WorkerIpcEndpoint

    ipc_endpoint = WorkerIpcEndpoint(
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
    WorkerServer(worker, ipc_endpoint).serve()
