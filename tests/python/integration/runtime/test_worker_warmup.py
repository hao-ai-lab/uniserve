"""Startup warmup drives valid depth-one batches through the real forward path.

Warmup pays first-use attention-kernel JIT before the worker is reachable. Its
value is realized on CUDA, but the synthetic prefill/decode/flow batches it
constructs must stay valid records that actually run a forward. These tests
exercise that construction on the CPU stub so a regression in the batch shapes
is caught without a GPU.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from torch import nn

from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.forward import ForwardBatch, ForwardOutput
from uniserve_worker.server.stub import StubModel
from uniserve_worker.spec import OperationType

pytestmark = pytest.mark.integration


class _Observed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.neural = StubModel()
        self.spec = self.neural.spec
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def forward(self, batch: ForwardBatch) -> ForwardOutput:
        self.calls.append((str(batch.route), tuple(type(row).__name__ for row in batch.rows)))
        return self.neural(batch)


def test_warmup_is_a_safe_noop_off_cuda() -> None:
    worker = execution_worker()
    baseline = (worker.kv.resident_block_count(), worker.kv.scratch_token_count())
    worker.warmup()
    assert (worker.kv.resident_block_count(), worker.kv.scratch_token_count()) == baseline


def test_warmup_image_geometry_fits_the_declared_latent_capacity() -> None:
    worker = execution_worker()
    caps = worker.contract.capabilities
    if caps.max_latent_size <= 0:
        pytest.skip("stub declares no image latent capacity")

    height, width = worker._warmup_image_geometry()

    downsample = caps.latent_downsample
    latent_tokens = (height // downsample) * (width // downsample)
    assert latent_tokens <= caps.max_latent_size
    if caps.max_vae_grid_tokens > 0:
        assert latent_tokens <= caps.max_vae_grid_tokens


def test_warmup_sequence_drives_a_real_forward_and_cleans_up() -> None:
    model = _Observed()
    worker = execution_worker(model)
    if OperationType.SEQUENCE_EXTEND not in worker.contract.capabilities.operation_types:
        pytest.skip("stub does not support sequence extend")

    worker._execution = replace(
        worker._execution,
        cuda_graph=True,
        cuda_graph_warmup=True,
        cuda_graph_warmup_batches=(4, 2),
    )
    worker.products.device_products.capacity = 20
    baseline = (worker.kv.resident_block_count(), worker.kv.scratch_token_count())
    worker._warmup_sequence()

    assert model.calls
    assert all(row_kinds and set(row_kinds) == {"TokenRow"} for _route, row_kinds in model.calls)
    assert {2, 4} <= {len(row_kinds) for _route, row_kinds in model.calls}
    assert (worker.kv.resident_block_count(), worker.kv.scratch_token_count()) == baseline


def test_warmup_flow_drives_a_real_forward_and_cleans_up() -> None:
    model = _Observed()
    worker = execution_worker(model)
    if OperationType.FLOW not in worker.contract.capabilities.operation_types:
        pytest.skip("stub does not support flow")

    baseline = (worker.kv.resident_block_count(), worker.kv.scratch_token_count())
    worker._warmup_flow()

    assert any("FlowRow" in row_kinds for _route, row_kinds in model.calls)
    assert (worker.kv.resident_block_count(), worker.kv.scratch_token_count()) == baseline
