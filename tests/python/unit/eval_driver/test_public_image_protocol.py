from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from uniserve_eval.harness.image_outputs import (
    ImageOutputError,
    decode_openai_image_part,
)

pytestmark = pytest.mark.unit


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 3), (1, 2, 3)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_openai_image_part_decodes_payload_and_metadata() -> None:
    image = decode_openai_image_part({"b64_json": base64.b64encode(_png()).decode("ascii")})
    assert (image.width, image.height, image.mime) == (2, 3, "image/png")
    assert image.data == _png()


def test_openai_image_part_rejects_declared_mime_mismatch() -> None:
    encoded = base64.b64encode(_png()).decode("ascii")
    with pytest.raises(ImageOutputError, match="protocol_image_mime_mismatch"):
        decode_openai_image_part({"image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
