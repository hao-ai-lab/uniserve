"""Construction helpers for canonical worker-boundary conformance tests."""

from __future__ import annotations

from dataclasses import replace

from uniserve_worker.backends.attention import FlashInferTuningConfig, resolve_attention_selection
from uniserve_worker.bootstrap.execution_config import ExecutionConfig
from uniserve_worker.models.runtime import ExecutionModel
from uniserve_worker.nn.mesh import DeviceMesh
from uniserve_worker.server.stub import StubModel, stub_deployment
from uniserve_worker.worker import Worker


def execution_worker(
    model: ExecutionModel | None = None,
    *,
    block_size: int = 16,
    device: str = "cpu",
    pipeline_depth: int = 1,
    snapshot_dir: str | None = None,
    transfer_backend: str = "local",
    execution: ExecutionConfig | None = None,
    max_batch_tokens: int = 8192,
    max_request_pool_size: int = 128,
    max_batch_operations: int | None = None,
) -> Worker:
    ready = StubModel() if model is None else model
    deployment = replace(
        stub_deployment(block_size, max_batch_tokens=max_batch_tokens),
        device=device,
        max_request_pool_size=max_request_pool_size,
    )
    if max_batch_operations is not None:
        deployment = replace(
            deployment,
            max_batch_operations=int(max_batch_operations),
        )
    worker = Worker(
        ready,
        mesh=DeviceMesh.trivial(device),
        deployment=deployment,
        attention=resolve_attention_selection(
            "torch_sdpa",
            tuning=FlashInferTuningConfig(),
            block_size=block_size,
        ),
        execution=(
            ExecutionConfig(
                cuda_graph=False,
                prefill_cuda_graph=False,
                flow_graph_batch_sizes=(1,),
                flow_graph_shapes=((16, 16),),
            )
            if execution is None
            else execution
        ),
        tokenizer=None,
        allowed_work_variants=ready.supported_work,
        transfer_backend=transfer_backend,
        pipeline_depth=pipeline_depth,
        completion_payload_bytes=1 << 16,
        snapshot_dir=snapshot_dir,
    )
    from .depth_one import configure_physical_pool

    configure_physical_pool(
        cache_pages=worker.cache_pool.num_pages,
        request_pool_size=worker.capabilities.max_request_pool_size,
        block_size=worker.cache_pool.block_size,
        commit_marker_tokens=worker.capabilities.commit_marker_tokens,
        max_cfg_branches=worker.capabilities.max_cfg_branches,
        latent_page_units=worker.capabilities.latent_page_units,
        latent_downsample=worker.capabilities.latent_downsample,
    )
    return worker


__all__ = ["execution_worker"]
