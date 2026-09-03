from __future__ import annotations

from uniserve_worker.transfer.tickets import decode_transfer_handle, encode_transfer_handle


def _locator() -> dict[str, object]:
    return {
        "transport": "local",
        "endpoint": "worker-a",
        "key": 7,
        "nbytes": 4,
        "dtype": "float32",
        "shape": [1],
        "device": "cpu",
    }


def test_typed_transfer_handles_round_trip() -> None:
    fixed = {"op_id": 3, "point": {"kind": "fixed", "value": 1}}
    cases = {
        "encoder": {
            "generation": 2,
            "height": 16,
            "width": 24,
            "payload_kind": "vision_feature",
            "locator": _locator(),
        },
        "device_product": {
            "generation": 2,
            "height": 0,
            "width": 0,
            "value_range": "",
            "locator": _locator(),
        },
        "kv": {
            "generation": 2,
            "snapshot": {
                "locators": [_locator()],
                "source_version": fixed,
                "destination": "decode",
                "base_version": None,
                "base_extent": 0,
                "published_extent": 1,
                "group_id": 0,
                "scale_identity": "bfloat16",
            },
        },
        "latent": {
            "generation": 2,
            "height": 16,
            "width": 24,
            "latent_units": 1,
            "step": 4,
            "locator": _locator(),
        },
    }
    for kind, value in cases.items():
        assert decode_transfer_handle(encode_transfer_handle(kind, value)) == (kind, value)
