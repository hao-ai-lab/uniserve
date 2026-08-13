"""Construction helpers for canonical worker-boundary conformance tests."""

from __future__ import annotations

from dataclasses import replace

from uniserve_worker.backends.attention import resolve_attention_selection
from uniserve_worker.foundation.runtime_config import ExecutionConfig, FlashInferTuningConfig
from uniserve_worker.models.runtime import ExecutionModel
from uniserve_worker.nn.mesh import DeviceMesh
from uniserve_worker.server.stub import StubModel, stub_deployment
from uniserve_worker.worker.model import ModelWorker


def execution_worker(
    model: ExecutionModel | None = None,
    *,
    defer_sampling: bool = False,
    block_size: int = 16,
    device: str = "cpu",
    pipeline_depth: int = 1,
    snapshot_dir: str | None = None,
    transfer_backend: str = "local",
    execution: ExecutionConfig | None = None,
    max_batch_tokens: int = 8192,
    max_request_pool_size: int = 128,
) -> ModelWorker:
    ready = StubModel() if model is None else model
    deployment = replace(
        stub_deployment(block_size, max_batch_tokens=max_batch_tokens),
        device=device,
        max_request_pool_size=max_request_pool_size,
    )
    worker = ModelWorker(
        ready,
        mesh=DeviceMesh.trivial(device),
        deployment=deployment,
        attention=resolve_attention_selection(
            "torch_sdpa",
            tuning=FlashInferTuningConfig(),
            block_size=block_size,
        ),
        execution=(
            ExecutionConfig(cuda_graph=False, prefill_cuda_graph=False)
            if execution is None
            else execution
        ),
        tokenizer=None,
        allowed_work_variants=ready.supported_work,
        defer_sampling=defer_sampling,
        transfer_backend=transfer_backend,
        pipeline_depth=pipeline_depth,
        completion_payload_bytes=1 << 16,
        snapshot_dir=snapshot_dir,
    )
    from .depth_one import configure_physical_pool

    configure_physical_pool(
        request_pages=worker.cache_pool.request_pages,
        scratch_pages=worker.cache_pool.scratch_pages,
        max_cfg_branches=worker.capabilities.max_cfg_branches,
    )
    return worker


__all__ = ["execution_worker"]
