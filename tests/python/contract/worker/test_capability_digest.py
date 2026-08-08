"""Exact worker capability agreement across the Python wire boundary."""

from __future__ import annotations

import pytest

from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker import batch
from uniserve_worker.batch import WorkVariant
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
    return batch.route_capability_digest(
        tuple(WorkVariant(value) for value in wire["supported_work"]),
        wire["max_cfg_branches"],
        wire["max_latent_size"],
        wire["max_vae_grid_tokens"],
        wire["max_vit_grid_tokens"],
        wire["max_latent_feature_bytes"],
        wire["max_vision_feature_bytes"],
        wire["execution_constraints"]["max_batch_operations"],
        wire["execution_constraints"]["max_speculative_points"],
        wire["execution_constraints"]["max_unresolved_window"],
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
                capability["max_unresolved_window"],
                capability["legal_feature_bitset"],
                capability["sampler_processors"],
                capability["processor_order_revision"],
                capability["rng_layouts"],
                capability["graph_eligible"],
                capability["gen_conditioning"],
                capability["max_points_per_operation"],
                tuple(capability["mixed_row_combinations"]),
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
    assert wire["execution_constraints"]["max_speculative_points"] == 1
    known = {variant.value for variant in WorkVariant}
    assert all(value in known for value in wire["supported_work"])

    assert _is_digest(wire["protocol_layout_digest"])
    assert _is_digest(wire["route_capability_digest"])
    assert wire["protocol_layout_digest"] == batch.protocol_layout_digest()
    assert wire["route_capability_digest"] == _recompute_route_digest(wire)


def test_every_route_advertises_preemption_unsupported():
    # Preemption is not a configured serving capability: every advertised route
    # declares a non-preemptible checkpoint scope regardless of its residency
    # classes, matching the unsupported-preemption declaration.
    worker = execution_worker()
    wire = dispatch(worker, {"kind": "get_capabilities"})["capabilities"]
    routes = wire["execution_constraints"]["route_capabilities"]
    assert routes, "worker advertises at least one route capability"
    assert all(capability["preemptible"] is False for capability in routes)


def test_capability_wire_round_trips_and_recomputes_stable_digests():
    worker = execution_worker()
    caps = worker.contract.capabilities

    restored = EngineCaps.from_wire(caps.to_wire())

    assert restored == caps
    assert restored.protocol_layout_digest == batch.protocol_layout_digest()
    assert restored.route_capability_digest == caps.route_capability_digest
    assert _is_digest(restored.protocol_layout_digest)
    assert _is_digest(restored.route_capability_digest)
