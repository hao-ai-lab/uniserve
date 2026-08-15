"""Worker capability conformance across the Python wire boundary."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker import batch
from uniserve_worker.batch import Domain, SamplingOwnership, WorkVariant
from uniserve_worker.bootstrap.capacity import operation_window
from uniserve_worker.capabilities import (
    GraphBucketCapability,
    LaneCapabilities,
    WorkerCapabilities,
)
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
    assert wire["num_layers"] > 0
    assert wire["num_kv_heads"] > 0
    assert wire["head_dim"] > 0
    assert wire["max_batch_operations"] > 0
    assert wire["max_unresolved_window"] == operation_window(
        wire["pipeline_depth"], wire["max_batch_operations"]
    )
    assert wire["incremental_kv_publication"] is True
    assert wire["sampling_ownership"] == SamplingOwnership.DESIGNATED_RANK.value
    assert wire["mixed_buckets"] == [
        {
            "decode_rows": 1,
            "flow_rows": 1,
            "height": 16,
            "width": 16,
            "cfg_branches": cfg_branches,
        }
        for cfg_branches in (1, 2, 3)
    ]
    assert _is_digest(wire["protocol_layout_digest"])
    assert wire["protocol_layout_digest"] == batch.protocol_layout_digest()


def test_worker_capability_wire_round_trips_exactly() -> None:
    base = execution_worker().capabilities
    capabilities = replace(
        base,
        lanes=(
            LaneCapabilities(
                lane_id="decode",
                domains=(Domain.DECODE,),
                resolved_sm_count=64,
                kv_capacity_tokens=65_536,
                latent_capacity_units=None,
                max_batch_operations=128,
                max_batch_tokens=16_384,
                max_inflight=2,
                graph_buckets=(
                    GraphBucketCapability(
                        phase="text_decode",
                        batch_size=32,
                        token_bucket=32,
                        attention_form="paged_decode",
                        height=0,
                        width=0,
                        cfg_branches=1,
                    ),
                ),
                eager_max_batch_operations=128,
                eager_max_batch_tokens=16_384,
            ),
        ),
    )

    restored = WorkerCapabilities.from_wire(capabilities.to_wire())

    assert restored == capabilities
    assert restored.protocol_layout_digest == batch.protocol_layout_digest()
