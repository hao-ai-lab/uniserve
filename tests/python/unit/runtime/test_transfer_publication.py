from __future__ import annotations

import pytest

from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.batch import (
    MAX_TRANSFER_DESCRIPTOR_BYTES,
    TRANSFER_DESCRIPTOR_PREFIX,
)
from uniserve_worker.transfer.tickets import (
    decode_transfer_descriptor,
    encode_transfer_descriptor,
)


def test_transfer_descriptor_round_trips_exact_canonical_provenance() -> None:
    digest = "a" * 64
    for kind in ("encoder", "device_product", "kv", "latent"):
        encoded = encode_transfer_descriptor(kind, {"height": 16, "width": 24}, digest)
        assert decode_transfer_descriptor(encoded) == (
            kind,
            {"height": 16, "width": 24},
            digest,
        )


def test_transfer_descriptor_rejects_noncanonical_or_unbounded_frames() -> None:
    digest = "b" * 64
    canonical = encode_transfer_descriptor("kv", {"snapshot": {}}, digest)
    noncanonical = canonical.replace(b'"kind":"kv"', b'"kind": "kv"')
    oversized = TRANSFER_DESCRIPTOR_PREFIX + b"{" + b" " * MAX_TRANSFER_DESCRIPTOR_BYTES

    with pytest.raises(WorkerError, match="not canonical"):
        decode_transfer_descriptor(noncanonical)
    with pytest.raises(WorkerError, match="descriptor bound"):
        decode_transfer_descriptor(oversized)
    with pytest.raises(WorkerError, match="descriptor bound"):
        encode_transfer_descriptor(
            "device_product", {"payload": "x" * MAX_TRANSFER_DESCRIPTOR_BYTES}, digest
        )
