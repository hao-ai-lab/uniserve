"""Immutable protocol identities preserve lookup and versioning.

They also preserve wire semantics.
"""

import pickle
from dataclasses import replace

import pytest

from uniserve_worker.errors import WorkerError
from uniserve_worker.protocol.identity import BufferId, CallId, RequestKey
from uniserve_worker.protocol.tensor import DType, ShapeBound, TensorRef

pytestmark = pytest.mark.unit


def test_buffer_lookup_preserves_identity_across_reconstruction_and_versions():
    reference = TensorRef(
        RequestKey(2, 7, 3), CallId(11, 4), 0, 5, DType.I64, ShapeBound()
    )
    entries = {reference.buffer_id: "value"}
    wire = reference.to_mapping()
    restored = (
        TensorRef.from_mapping(wire),
        TensorRef.from_mapping(
            dict(wire, producer_call_id=reference.producer_call_id)
        ),
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
        replace(
            reference,
            producer_call_id=CallId(
                12, reference.producer_call_id.request_index
            ),
        ),
        replace(
            reference,
            request_key=RequestKey(2, 7, 4),
        ),
    )
    for item in versions:
        assert item.buffer_id not in entries
        entries[item.buffer_id] = "successor"
    assert len(entries) == 5
    assert entries[reference.buffer_id] == "value"


def test_computation_order_and_lookup_follow_batch_and_request_coordinates():
    first = CallId(9, 1)
    second = CallId(9, 2)
    third = CallId(10, 0)
    entries = {second: "second", first: "first", third: "third"}
    reconstructed = [
        CallId.from_mapping(item.to_mapping())
        for item in (third, first, second)
    ]
    assert [entries[item] for item in sorted(reconstructed)] == [
        "first",
        "second",
        "third",
    ]
    assert CallId(second.batch_id, 1) in entries


@pytest.mark.parametrize(
    ("kind", "mapping"),
    (
        (RequestKey, {"engine_id": 1, "request_id": 7, "request_epoch": -1}),
        (RequestKey, {"engine_id": 1, "request_id": 7, "request_epoch": True}),
        (CallId, {"batch_id": 0, "request_index": 1}),
        (
            BufferId,
            {
                "owner": {"engine_id": 1, "request_id": 7, "request_epoch": 2},
                "producer_call_id": {"batch_id": 3, "request_index": 0},
                "output_index": 0,
                "generation": 0,
            },
        ),
    ),
)
def test_invalid_coordinates_are_rejected_at_wire_decoding(kind, mapping):
    with pytest.raises(WorkerError):
        kind.from_mapping(mapping)
