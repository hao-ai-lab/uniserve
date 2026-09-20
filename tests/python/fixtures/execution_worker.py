"""Construction helpers for worker IPC behavior tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import torch

from tests.python.fixtures.worker_config import stub_worker_config
from uniserve.distributed.mesh import Communicator
from uniserve_models.stub import Model, image_processor
from uniserve_worker.bootstrap.config import ComponentConfig
from uniserve_worker.config import WorkerConfig
from uniserve_worker.execution.component_binding import ComponentBinding
from uniserve_worker.worker import Worker


def execution_worker(
    model: torch.nn.Module | None = None,
    *,
    block_size: int = 16,
    device: str = "cpu",
    queue_depth: int = 1,
    transfer_backends: tuple[str, ...] = ("local",),
    worker_id: str = "worker",
    execution: WorkerConfig | None = None,
    max_batch_tokens: int = 8192,
    max_request_pool_size: int = 128,
    max_batch_calls: int | None = None,
    components: tuple[tuple[str, ComponentConfig], ...] = (),
    bindings: Mapping[str, ComponentBinding] | None = None,
    host_slots: tuple[int, ...] = (0, 1),
) -> Worker:
    ready = Model().to(device) if model is None else model
    worker_config = replace(
        stub_worker_config(block_size, max_batch_tokens=max_batch_tokens),
        device=device,
        max_request_pool_size=max_request_pool_size,
    )
    if max_batch_calls is not None:
        worker_config = replace(
            worker_config,
            max_batch_calls=int(max_batch_calls),
        )
    policy = (
        WorkerConfig(
            graph_policy="off",
            prefill_cuda_graph=False,
            flow_graph_batch_sizes=(1,),
            flow_graph_shapes=((16, 16),),
        )
        if execution is None
        else execution
    )
    if model is None and policy.generation_device is not None:
        # The fixture is the resource owner for its constructed modules.
        ready.latent_encoder.to(policy.generation_device)
        ready.image_decoder.to(policy.generation_device)
    worker_config = replace(
        policy,
        device=worker_config.device,
        block_size=worker_config.block_size,
        kv_token_capacity=worker_config.kv_token_capacity,
        attention_backend=worker_config.attention_backend,
        max_request_pool_size=worker_config.max_request_pool_size,
        encoder_cache_entries=worker_config.encoder_cache_entries,
        max_batch_calls=worker_config.max_batch_calls,
        max_batch_tokens=worker_config.max_batch_tokens,
    )
    worker = Worker(
        ready,
        image_processor=image_processor() if isinstance(ready, Model) else None,
        bindings=bindings,
        sampling_group=Communicator(device=torch.device(device)),
        worker_config=worker_config,
        attention="torch",
        tokenizer=None,
        allowed_work_variants=None,
        transfer_backends=transfer_backends,
        publication_backends=transfer_backends,
        # The fixture's deployment is one host: the worker's own slot and the
        # external consumer slot a test names both read over shared memory.
        host_slots=host_slots,
        worker_id=worker_id,
        queue_depth=queue_depth,
        completion_payload_bytes=1 << 16,
        components=components,
    )
    from .depth_one import configure_physical_pool, submitted_batch

    # The fixture stands in for the engine, which states every call's
    # coordinates from the request state it owns as it submits the batch.
    submit = worker.submit

    def submit_stamped(batch, *arguments, **options):
        return submit(submitted_batch(worker, batch), *arguments, **options)

    worker.submit = submit_stamped

    flow = worker.runner.image_builder
    configure_physical_pool(
        cache_pages=0
        if worker.info.kv_cache is None
        else worker.info.kv_cache.num_blocks,
        request_pool_size=worker.info.request_slots,
        block_size=block_size
        if worker.info.kv_cache is None
        else worker.info.kv_cache.block_size,
        commit_marker_tokens=0 if flow is None else flow.framing,
        max_cfg_branches=1 if flow is None else 3,
        latent_page_units=worker.info.latent_page_units,
        latent_downsample=1 if flow is None else flow.denoiser.downsample,
    )
    return worker


__all__ = ["execution_worker"]
