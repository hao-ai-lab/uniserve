"""Exact worker capability agreement across the Python wire boundary."""

from __future__ import annotations

import pytest

from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker import batch
from uniserve_worker.batch import AdapterMode, WorkVariant
from uniserve_worker.capabilities import EngineCaps
from uniserve_worker.server.app import dispatch

pytestmark = pytest.mark.contract


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _recompute_route_digest(wire: dict) -> str:
    credit_fields = (
        "registered_operations",
        "execution_slots",
        "completion_slots",
        "device_products",
        "kv_pages",
        "rollback_deltas",
        "latent_artifact_bytes",
        "pinned_completion_staging_bytes",
        "transfer_bytes",
        "transfer_tickets",
        "cpu_tasks",
        "output_journal_bytes",
    )
    return batch.route_capability_digest(
        tuple(WorkVariant(value) for value in wire["supported_work"]),
        wire["max_cfg_branches"],
        wire["max_latent_size"],
        wire["max_vae_grid_tokens"],
        wire["max_vit_grid_tokens"],
        wire["max_latent_feature_bytes"],
        wire["max_vision_feature_bytes"],
        AdapterMode(wire["adapter_mode"]),
        wire["execution_constraints"]["max_batch_operations"],
        wire["execution_constraints"]["max_speculative_points"],
        wire["execution_constraints"]["device_sequence_lengths"],
        wire["execution_constraints"]["device_append_offsets"],
        wire["execution_constraints"]["incremental_kv_publication"],
        tuple(
            (
                capability["route"],
                tuple(WorkVariant(value) for value in capability["supported_work"]),
                capability["tensorized_mixed"],
                batch.SamplingOwnership(capability["sampling_ownership"]),
                capability["preemptible"],
                (
                    tuple(capability["credits"]["per_request"][name] for name in credit_fields),
                    tuple(capability["credits"]["worker"][name] for name in credit_fields),
                ),
            )
            for capability in wire["execution_constraints"]["route_capabilities"]
        ),
        wire["kv_dtype"],
        wire["model_dtype"],
        wire["attention_backend"],
    )


def test_capability_wire_carries_the_agreement_digests_and_work_shape():
    worker = execution_worker()
    wire = dispatch(worker, {"kind": "get_capabilities"})["capabilities"]

    assert wire["supported_work"]
    known = {variant.value for variant in WorkVariant}
    assert all(value in known for value in wire["supported_work"])

    assert _is_digest(wire["protocol_layout_digest"])
    assert _is_digest(wire["route_capability_digest"])
    assert wire["protocol_layout_digest"] == batch.protocol_layout_digest()
    assert wire["route_capability_digest"] == _recompute_route_digest(wire)


def test_capability_wire_round_trips_and_recomputes_stable_digests():
    worker = execution_worker()
    caps = worker.contract.capabilities

    restored = EngineCaps.from_wire(caps.to_wire())

    assert restored == caps
    assert restored.protocol_layout_digest == batch.protocol_layout_digest()
    assert restored.route_capability_digest == caps.route_capability_digest
    assert _is_digest(restored.protocol_layout_digest)
    assert _is_digest(restored.route_capability_digest)
