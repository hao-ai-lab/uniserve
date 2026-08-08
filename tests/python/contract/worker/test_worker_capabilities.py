"""Worker capability conformance across the Python wire boundary."""

from __future__ import annotations

import pytest

from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker import batch
from uniserve_worker.batch import SamplingOwnership, WorkVariant
from uniserve_worker.capabilities import WorkerCapabilities
from uniserve_worker.runtime.arena_capacity import operation_window
from uniserve_worker.server.app import dispatch

pytestmark = pytest.mark.contract


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def test_worker_capability_wire_reports_schedulable_work_and_bounds() -> None:
    worker = execution_worker()
    wire = dispatch(worker, {"kind": "get_capabilities"})["capabilities"]

    assert wire["supported_work"] == [
        WorkVariant.TOKEN_EXTEND.value,
        WorkVariant.TOKEN_DECODE.value,
        WorkVariant.ENCODE_VISION.value,
        WorkVariant.ENCODE_LATENT.value,
        WorkVariant.TRANSFER_PRODUCT.value,
        WorkVariant.TRANSFER_KV_PUBLISH.value,
        WorkVariant.TRANSFER_KV_INSTALL.value,
        WorkVariant.GEN_TRANSITION.value,
        WorkVariant.GEN_FLOW.value,
        WorkVariant.MATERIALIZE.value,
    ]
    assert wire["num_layers"] == worker.model_spec.cache.num_layers
    assert wire["num_kv_heads"] == worker.model_spec.cache.num_kv_heads
    assert wire["head_dim"] == worker.model_spec.cache.head_dim
    assert wire["max_batch_operations"] == worker.deployment.max_batch_operations
    assert wire["max_unresolved_window"] == operation_window(
        wire["pipeline_depth"], wire["max_batch_operations"]
    )
    assert wire["incremental_kv_publication"] is True
    assert wire["sampling_ownership"] == SamplingOwnership.DESIGNATED_RANK.value
    assert isinstance(wire["tensorized_mixed"], bool)
    assert _is_digest(wire["protocol_layout_digest"])
    assert wire["protocol_layout_digest"] == batch.protocol_layout_digest()


def test_worker_capability_wire_round_trips_exactly() -> None:
    capabilities = execution_worker().capabilities

    restored = WorkerCapabilities.from_wire(capabilities.to_wire())

    assert restored == capabilities
    assert restored.protocol_layout_digest == batch.protocol_layout_digest()
