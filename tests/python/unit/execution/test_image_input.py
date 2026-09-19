"""Encoded image inputs preserve the model's pixel values and patch layout."""

import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image

from uniserve.processing import (
    ImageProcessor,
    PatchTransform,
    StrideResize,
    TowerTransform,
)
from uniserve_worker.execution.image_input import prepare_image
from uniserve_worker.protocol.call import MediaCall


@pytest.mark.parametrize("normalization", ("signed_unit", "imagenet"))
@pytest.mark.parametrize("patches", (False, True))
@pytest.mark.parametrize(
    "device,dtype",
    (
        ("cpu", "float32"),
        pytest.param("cuda:0", "bfloat16", marks=pytest.mark.gpu),
    ),
)
def test_encoded_pixels_match_channel_normalization(
    normalization, patches, device, dtype
):
    # Include every byte value in each RGB channel. Geometry already satisfies
    # both tower policies, so the expected values do not depend on a resizer.
    raw = (np.arange(24 * 32 * 3).reshape(24, 32, 3) % 256).astype(np.uint8)
    image = Image.fromarray(raw)
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    transform = (
        PatchTransform(4, 1.0, 24 * 32, 24 * 32, normalization)
        if patches
        else TowerTransform(StrideResize(32, 24, 1, 24 * 32), normalization)
    )
    processor = ImageProcessor(
        vit=transform, staging_dtype=getattr(torch, dtype)
    )

    result = prepare_image(
        processor,
        MediaCall.VISION_ENCODING,
        base64.b64encode(encoded.getvalue()).decode(),
        device=torch.device(device),
    )

    expected = (
        torch.from_numpy(raw).permute(2, 0, 1).contiguous().float() / 255.0
    )
    if normalization == "signed_unit":
        expected = (expected - 0.5) / 0.5
    else:
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        expected = (expected - mean) / std
    if patches:
        # Encoder rows are grid-major; each row contains channel-major pixels.
        expected = torch.stack(
            [
                expected[:, y : y + 4, x : x + 4].reshape(-1)
                for y in range(0, 24, 4)
                for x in range(0, 32, 4)
            ]
        )
        assert result.grid is not None and result.grid.cpu().tolist() == [
            [6, 8]
        ]
        assert result.grid_shape == (6, 8)
    assert (result.height, result.width) == (24, 32)
    torch.testing.assert_close(
        result.pixels.cpu(), expected.to(getattr(torch, dtype)), rtol=0, atol=0
    )
