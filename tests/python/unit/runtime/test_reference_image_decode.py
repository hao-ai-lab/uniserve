"""Image decoding contracts using in-memory encoded media."""

import io
import struct
import zlib

import numpy as np
import pytest
from PIL import Image

from uniserve_worker.media.decoded import DecodedMemoryBudget, decode_reference_image


def image_bytes(pixels):
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    return stream.getvalue()


def test_rgb_values_layout_and_bundle_budget():
    pixels = np.array([[[1, 2, 3], [250, 128, 0]]], dtype=np.uint8)
    payload = image_bytes(pixels)
    budget = DecodedMemoryBudget(limit_bytes=28)
    first = decode_reference_image(payload, budget=budget)
    second = decode_reference_image(payload, budget=budget)
    assert first.dtype == np.uint8
    np.testing.assert_array_equal(first, pixels[None])
    np.testing.assert_array_equal(second, pixels[None])
    assert budget.used_bytes == first.nbytes + second.nbytes == 12
    # Each decode needs temporary source/conversion capacity as well as retained
    # output. A third image would exceed the same bundle's remaining capacity.
    with pytest.raises(ValueError, match="aggregate"):
        decode_reference_image(payload, budget=budget)
    assert budget.used_bytes == 12
    np.testing.assert_array_equal(first, pixels[None])


def test_predecode_header_size_check():
    payload = bytearray(image_bytes(np.zeros((1, 1, 3), dtype=np.uint8)))
    payload[16:20] = struct.pack(">I", 4097)
    payload[29:33] = struct.pack(">I", zlib.crc32(payload[12:29]))
    budget = DecodedMemoryBudget()
    with pytest.raises(ValueError, match="dimensions"):
        decode_reference_image(bytes(payload), budget=budget)
    assert budget.used_bytes == 0


def test_decode_failure_restores_bundle_capacity():
    payload = image_bytes(np.zeros((2, 2, 3), dtype=np.uint8))
    budget = DecodedMemoryBudget(limit_bytes=100, used_bytes=5)
    # Preserve the header while truncating the compressed pixel stream.
    with pytest.raises(ValueError):
        decode_reference_image(payload[:45], budget=budget)
    assert budget.used_bytes == 5
    assert decode_reference_image(payload, budget=budget).shape == (1, 2, 2, 3)
    assert budget.used_bytes == 17


def test_animation_is_not_silently_truncated():
    stream = io.BytesIO()
    first = Image.new("RGB", (2, 2), "red")
    second = Image.new("RGB", (2, 2), "blue")
    first.save(stream, format="GIF", save_all=True, append_images=[second])
    budget = DecodedMemoryBudget()
    with pytest.raises(ValueError, match="exactly one frame"):
        decode_reference_image(stream.getvalue(), budget=budget)
    assert budget.used_bytes == 0
