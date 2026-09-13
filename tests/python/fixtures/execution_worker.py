"""Construction helpers for worker IPC behavior tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import torch

from tests.python.fixtures.worker_config import stub_worker_config
from uniserve_worker.backends.attention import FlashInferTuningConfig, resolve_attention_selection
from uniserve_worker.config import WorkerConfig
from uniserve_worker.execution.model_entry import ModelEntry
from uniserve_worker.modeling.image_diffusion import LatentLayout
from uniserve_worker.modeling.model import Model
from uniserve_worker.models.stub import StubModel
from uniserve_worker.nn.mesh import Communicator
from uniserve_worker.nn.parallel import ComponentConfig
from uniserve_worker.worker import Worker


def execution_worker(
    model: Model | None = None,
    *,
    block_size: int = 16,
    device: str = "cpu",
    pipeline_depth: int = 1,
    transfer_backends: tuple[str, ...] = ("local",),
    worker_id: str = "worker",
    execution: WorkerConfig | None = None,
    max_batch_tokens: int = 8192,
    max_request_pool_size: int = 128,
    max_batch_operations: int | None = None,
    components: tuple[tuple[str, ComponentConfig], ...] = (),
    bindings: Mapping[str, ModelEntry] | None = None,
) -> Worker:
    ready = StubModel() if model is None else model
    worker_config = replace(
        stub_worker_config(block_size, max_batch_tokens=max_batch_tokens),
        device=device,
        max_request_pool_size=max_request_pool_size,
    )
    if max_batch_operations is not None:
        worker_config = replace(
            worker_config,
            max_batch_operations=int(max_batch_operations),
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
    worker_config = replace(
        policy,
        device=worker_config.device,
        block_size=worker_config.block_size,
        kv_token_capacity=worker_config.kv_token_capacity,
        attention_backend=worker_config.attention_backend,
        max_request_pool_size=worker_config.max_request_pool_size,
        encoder_cache_entries=worker_config.encoder_cache_entries,
        max_batch_operations=worker_config.max_batch_operations,
        max_batch_tokens=worker_config.max_batch_tokens,
    )
    worker = Worker(
        ready,
        bindings=bindings,
        sampling_group=Communicator(device=torch.device(device)),
        worker_config=worker_config,
        attention=resolve_attention_selection(
            "torch_sdpa",
            tuning=FlashInferTuningConfig(),
            block_size=block_size,
        ),
        tokenizer=None,
        allowed_work_variants=None,
        transfer_backends=transfer_backends,
        publication_backends=transfer_backends,
        worker_id=worker_id,
        pipeline_depth=pipeline_depth,
        completion_payload_bytes=1 << 16,
        components=components,
    )
    from .depth_one import configure_physical_pool

    flow = worker.model.generation
    configure_physical_pool(
        cache_pages=worker.kv_cache.num_pages,
        request_pool_size=worker.info.request_slots,
        block_size=worker.kv_cache.block_size,
        commit_marker_tokens=(
            int(flow.marker_tokens)
            if flow is not None and flow.latent_layout is LatentLayout.PATCH_TOKENS
            else 0
        ),
        max_cfg_branches=1 if flow is None else flow.max_cfg_branches,
        latent_page_units=worker.info.latent_page_units,
        latent_downsample=1 if flow is None else flow.latent_downsample,
    )
    return worker


__all__ = ["execution_worker"]
