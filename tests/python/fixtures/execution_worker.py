"""Construction helpers for canonical worker-boundary conformance tests."""

from __future__ import annotations

from dataclasses import replace

from torch import nn

from uniserve_worker.backends.attention import resolve_attention_selection
from uniserve_worker.foundation.runtime_config import ExecutionConfig, FlashInferTuningConfig
from uniserve_worker.nn.mesh import DeviceMesh
from uniserve_worker.server.stub import StubModel, stub_deployment
from uniserve_worker.worker.model import ModelWorker


def execution_worker(
    model: nn.Module | None = None,
    *,
    defer_sampling: bool = False,
    block_size: int = 16,
    device: str = "cpu",
    pipeline_depth: int = 1,
    snapshot_dir: str | None = None,
    restore_snapshots: bool = False,
    transfer_backend: str = "local",
) -> ModelWorker:
    ready = StubModel() if model is None else model
    deployment = replace(stub_deployment(block_size), device=device)
    return ModelWorker(
        ready,
        mesh=DeviceMesh.trivial(device),
        model_spec=ready.spec,
        deployment=deployment,
        attention=resolve_attention_selection(
            "torch_sdpa",
            tuning=FlashInferTuningConfig(),
            block_size=block_size,
        ),
        execution=ExecutionConfig(cuda_graph=False, prefill_cuda_graph=False),
        tokenizer=None,
        allowed_work_variants=ready.spec.operation_variants(),
        defer_sampling=defer_sampling,
        transfer_backend=transfer_backend,
        pipeline_depth=pipeline_depth,
        completion_payload_bytes=1 << 16,
        snapshot_dir=snapshot_dir,
        restore_snapshots=restore_snapshots,
    )


__all__ = ["execution_worker"]
