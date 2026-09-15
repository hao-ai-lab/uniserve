"""Immutable protocol identities preserve lookup, versioning, and wire semantics."""

import pickle
from dataclasses import replace

from uniserve_worker.protocol.identity import BufferId, ComputationId, RequestKey
from uniserve_worker.protocol.tensor import DType, ShapeBound, TensorRef


def test_buffer_lookup_preserves_identity_across_reconstruction_and_versions():
    reference = TensorRef(RequestKey(2, 7, 3), ComputationId(11, 4), 0, 5, DType.I64, ShapeBound())
    entries = {reference.buffer_id: "value"}
    wire = reference.to_mapping()
    restored = (
        TensorRef.from_mapping(wire),
        TensorRef.from_mapping(dict(wire, producer_op_id=reference.producer_op_id)),
        pickle.loads(pickle.dumps(reference)),
        replace(reference),
    )
    for item in restored:
        assert entries[item.buffer_id] == "value"
        assert item.to_mapping() == wire
        assert BufferId.from_mapping(item.buffer_id.to_mapping()) in entries

    versions = (
        replace(reference, generation=6),
        replace(reference, output_index=1),
        replace(reference, producer_op_id=replace(reference.producer_op_id, batch_id=12)),
        replace(reference, request_key=replace(reference.request_key, request_epoch=4)),
    )
    for item in versions:
        assert item.buffer_id not in entries
        entries[item.buffer_id] = "successor"
    assert len(entries) == 5
    assert entries[reference.buffer_id] == "value"


def test_computation_order_and_lookup_follow_batch_and_request_coordinates():
    first = ComputationId(9, 1)
    second = ComputationId(9, 2)
    third = ComputationId(10, 0)
    entries = {second: "second", first: "first", third: "third"}
    reconstructed = [
        ComputationId.from_mapping(item.to_mapping()) for item in (third, first, second)
    ]
    assert [entries[item] for item in sorted(reconstructed)] == ["first", "second", "third"]
    assert replace(second, request_index=1) in entries
