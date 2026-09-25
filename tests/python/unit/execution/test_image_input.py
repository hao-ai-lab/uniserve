"""Encoded image inputs preserve the model's pixel values and patch layout."""

import base64
import io
import json
from pathlib import Path

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
from uniserve_models import bagel, siglip
from uniserve_models.bagel import vae
from uniserve_worker.model_executor.image_inputs import (
    prepare_image,
    prepare_tensor_image,
)
from uniserve_worker.protocol.call import MediaCall

pytestmark = pytest.mark.unit

# Resize cases shared with the server's Bagel KV-token prediction test.
_BAGEL_RESIZE_CASES = json.loads(
    (
        Path(__file__).parents[2] / "fixtures" / "bagel_image_resize.json"
    ).read_text()
)["cases"]


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


@pytest.mark.parametrize(
    "kind", (MediaCall.LATENT_ENCODING, MediaCall.VISION_ENCODING)
)
def test_image_sources_preserve_the_same_model_canvas(kind):
    raw = np.full((16, 16, 3), 128, dtype=np.uint8)
    encoded = io.BytesIO()
    Image.fromarray(raw).save(encoded, format="PNG")
    processor = ImageProcessor(
        vae=TowerTransform(StrideResize(64, 32, 16, 4096)),
        vit=TowerTransform(StrideResize(64, 48, 16, 4096)),
        staging_dtype=torch.float32,
    )
    arguments = {"device": torch.device("cpu")}
    uploaded = prepare_image(
        processor,
        kind,
        base64.b64encode(encoded.getvalue()).decode(),
        **arguments,
    )
    resident = prepare_tensor_image(
        processor,
        kind,
        torch.from_numpy(raw).permute(2, 0, 1).float() / 255,
        signed_unit=False,
        **arguments,
    )
    assert (uploaded.height, uploaded.width) == (32, 32)
    assert (resident.height, resident.width) == (32, 32)
    expected_size = 32 if kind is MediaCall.LATENT_ENCODING else 48
    assert (
        uploaded.pixels.shape
        == resident.pixels.shape
        == (3, expected_size, expected_size)
    )


def _bagel_processor() -> ImageProcessor:
    """Build BAGEL's image processor at the published vision tower size."""
    config = bagel.Config(
        bagel.TransformerConfig(
            32, 48, 2, 4, 2, 37, 1e-6, 1_000_000.0, 8, True, 64
        ),
        siglip.Config(14, 980, 3, siglip.TransformerConfig(32, 4, 48, 1, 1e-6)),
        vae.Config(8, 3, 2, 32, 3, (1, 1), 1, 2, 0.5, 0.25),
        35,
        36,
        2,
        64,
        1.0,
        "gelu_pytorch_tanh",
    )
    return bagel.image_processor(config)


@pytest.mark.parametrize(
    "case",
    _BAGEL_RESIZE_CASES,
    ids=lambda case: f"{case['width']}x{case['height']}",
)
def test_bagel_resize_matches_the_shared_fixture(case):
    encoded = io.BytesIO()
    Image.new("RGB", (case["width"], case["height"]), (90, 120, 150)).save(
        encoded, format="PNG"
    )
    payload = base64.b64encode(encoded.getvalue()).decode()
    processor = _bagel_processor()

    # Both encoders stage the VAE canvas; the ViT tower resizes it again.
    for kind, tower in (
        (MediaCall.LATENT_ENCODING, "vae"),
        (MediaCall.VISION_ENCODING, "vit"),
    ):
        result = prepare_image(
            processor, kind, payload, device=torch.device("cpu")
        )
        assert (result.height, result.width) == (
            case["vae_height"],
            case["vae_width"],
        )
        assert tuple(result.pixels.shape) == (
            3,
            case[f"{tower}_height"],
            case[f"{tower}_width"],
        )
