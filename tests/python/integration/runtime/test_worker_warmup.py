"""Startup qualification exercises the production operation topology."""

from __future__ import annotations

from dataclasses import replace

import pytest
from torch import nn

from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.batch import DevicePoint, ProductKind, TokenMode
from uniserve_worker.forward import ForwardBatch, ForwardOutput
from uniserve_worker.server.stub import StubModel
from uniserve_worker.spec import OperationType

pytestmark = pytest.mark.integration


class _Observed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.neural = StubModel()
        self.spec = self.neural.spec
        self.calls: list[tuple[str, tuple[str, ...], str]] = []

    def forward(self, batch: ForwardBatch) -> ForwardOutput:
        self.calls.append(
            (
                str(batch.route),
                tuple(type(row).__name__ for row in batch.rows),
                type(batch.context.attention).__name__,
            )
        )
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


def test_warmup_sequence_qualifies_continuous_device_decode_and_cleans_up() -> None:
    model = _Observed()
    worker = execution_worker(model)
    if OperationType.SEQUENCE_EXTEND not in worker.contract.effective_operation_types:
        pytest.skip("stub does not support sequence extend")

    worker._execution = replace(
        worker._execution,
        cuda_graph=True,
        cuda_graph_warmup=True,
        cuda_graph_warmup_batches=(4, 2),
        prefill_cuda_graph=True,
        prefill_cuda_graph_warmup=True,
        prefill_cuda_graph_warmup_tokens=(4, 2),
    )
    worker.products.device_products.capacity = 20
    baseline = (worker.kv.resident_block_count(), worker.kv.scratch_token_count())
    batches = []
    execute_warmup = worker._execute_warmup

    def observe(batch, *, retain_device_outputs=False):
        batches.append(batch)
        return execute_warmup(batch, retain_device_outputs=retain_device_outputs)

    worker._execute_warmup = observe
    worker._warmup_sequence()

    assert model.calls
    assert all(
        row_kinds and set(row_kinds) == {"TokenRow"}
        for _route, row_kinds, _attention in model.calls
    )
    assert {2, 4} <= {len(row_kinds) for _route, row_kinds, _attention in model.calls}
    assert (worker.kv.resident_block_count(), worker.kv.scratch_token_count()) == baseline

    sequence = next(index for index, batch in enumerate(batches) if len(batch.admissions) == 4)
    serving_batches = batches[sequence:]
    assert [len(batch.operations) for batch in serving_batches] == [4, 4, 4, 2, 2]
    assert all(
        operation.work.mode == TokenMode.EXTEND.value for operation in serving_batches[0].operations
    )
    predecessors = {
        operation.request_key.session_id: operation for operation in serving_batches[0].operations
    }
    generations = {
        output.generation
        for operation in serving_batches[0].operations
        for output in operation.outputs
    }
    assert len(generations) == 4
    for batch in serving_batches[1:]:
        for operation in batch.operations:
            predecessor = predecessors[operation.request_key.session_id]
            token = next(
                output for output in predecessor.outputs if output.kind is ProductKind.TOKEN
            )
            assert operation.work.mode == TokenMode.DECODE.value
            assert operation.parent.producer_op_id == predecessor.op_id
            assert operation.parent.point == DevicePoint(1, None, predecessor.plan_digest)
            assert operation.predicate == token
            token_output = next(
                output for output in operation.outputs if output.kind is ProductKind.TOKEN
            )
            assert token_output.point_range.max_points == 1
            generations.update(output.generation for output in operation.outputs)
            predecessors[operation.request_key.session_id] = operation
    assert len(generations) == sum(
        len(operation.outputs) for batch in serving_batches for operation in batch.operations
    )


def test_warmup_flow_drives_a_real_forward_and_cleans_up() -> None:
    model = _Observed()
    worker = execution_worker(model)
    if OperationType.FLOW not in worker.contract.effective_operation_types:
        pytest.skip("stub does not support flow")

    baseline = (worker.kv.resident_block_count(), worker.kv.scratch_token_count())
    worker._warmup_flow()

    assert any("FlowRow" in row_kinds for _route, row_kinds, _attention in model.calls)
    assert (worker.kv.resident_block_count(), worker.kv.scratch_token_count()) == baseline
