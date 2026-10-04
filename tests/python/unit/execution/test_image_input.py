"""Encoded image inputs preserve the model's pixel values and patch layout."""

import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers.models.gemma4.image_processing_gemma4 import (
    Gemma4ImageProcessor,
)

from tests.python.fixtures.model_metadata import (
    diffusion_gemma_metadata,
    read_diffusion_gemma,
)
from uniserve.diffusion import NoiseScale
from uniserve.processing import (
    ImageProcessor,
    PatchTransform,
    PixelBounds,
    StrideResize,
    TowerTransform,
)
from uniserve_models import bagel, diffusion_gemma, siglip
from uniserve_models import sensenova_u1 as u1
from uniserve_models.bagel import vae
from uniserve_models.sensenova_u1 import flow, vision
from uniserve_worker.model_executor.image_inputs import (
    patch_grid_shape,
    prepare_image,
    prepare_tensor_image,
)
from uniserve_worker.protocol.call import MediaCall

pytestmark = pytest.mark.unit

_FIXTURES = Path(__file__).parents[2] / "fixtures"

# Resize cases shared with the server's KV-token prediction tests.
_BAGEL_RESIZE_CASES = json.loads(
    (_FIXTURES / "bagel_image_resize.json").read_text()
)["cases"]
_SENSENOVA_RESIZE_CASES = json.loads(
    (_FIXTURES / "sensenova_image_resize.json").read_text()
)["cases"]
# Soft-token counts shared with the server's prompt planner tests.
_DIFFUSION_GEMMA_TOKEN_CASES = json.loads(
    (_FIXTURES / "diffusion_gemma_image_tokens.json").read_text()
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
        PatchTransform(4, 1, PixelBounds(24 * 32, 24 * 32), normalization)
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
        input_images=1,
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
        input_images=1,
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
            processor,
            kind,
            payload,
            device=torch.device("cpu"),
            input_images=1,
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


def _sensenova_processor() -> ImageProcessor:
    """Build SenseNova U1's image processor at the published patch size."""
    config = u1.Config(
        u1.TransformerConfig(37, 32, 48, 2, 4, 2, 8, ("full_attention",) * 2),
        vision.Config(16, 32, 0.5, 16, 3, 10000.0),
        flow.Config(
            flow.HeadConfig(32, 2, 1.0),
            False,
            True,
            NoiseScale(1.0, "constant", 1, 8),
        ),
        64,
    )
    return u1.image_processor(config)


@pytest.mark.parametrize(
    "case",
    _SENSENOVA_RESIZE_CASES,
    ids=lambda case: (
        f"{case['width']}x{case['height']}-of-{case['input_images']}"
    ),
)
def test_sensenova_input_images_share_a_pixel_budget(case):
    encoded = io.BytesIO()
    Image.new("RGB", (case["width"], case["height"]), (90, 120, 150)).save(
        encoded, format="PNG"
    )
    processor = _sensenova_processor()

    result = prepare_image(
        processor,
        MediaCall.VISION_ENCODING,
        base64.b64encode(encoded.getvalue()).decode(),
        device=torch.device("cpu"),
        input_images=case["input_images"],
    )

    # 16-pixel patches, merged 2x2 into one KV token each by the tower.
    grid = (case["resized_height"] // 16, case["resized_width"] // 16)
    assert result.grid_shape == grid
    assert (
        patch_grid_shape(
            processor.vit,
            case["height"],
            case["width"],
            case["input_images"],
        )
        == grid
    )
    assert grid[0] * grid[1] // 4 == case["kv_tokens"]


def test_sensenova_generated_images_keep_the_single_image_bound():
    # A generated 2048x1152 canvas is not a request input, so it keeps the
    # 2048x2048 bound however many input images the request carries.
    generated = torch.zeros(3, 1152, 2048)

    result = prepare_tensor_image(
        _sensenova_processor(),
        MediaCall.VISION_ENCODING,
        generated,
        device=torch.device("cpu"),
        signed_unit=False,
    )

    assert result.grid_shape == (72, 128)


def _png(image: Image.Image) -> str:
    """Return an image as a base64 PNG payload, which decodes losslessly."""
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    return base64.b64encode(encoded.getvalue()).decode()


def _diffusion_gemma_processor(root) -> ImageProcessor:
    """Build the image processor DiffusionGemma declares for its checkpoint."""
    return diffusion_gemma.image_processor(
        read_diffusion_gemma(root, diffusion_gemma_metadata())
    )


@pytest.mark.parametrize(
    "height,width,mode",
    (
        (480, 640, "RGB"),
        (1080, 1920, "RGB"),
        (224, 224, "RGB"),
        (1, 1, "RGB"),
        (672, 960, "RGB"),
        (1600, 90, "RGB"),
        (10, 8000, "RGB"),
        (8000, 10, "RGB"),
        (300, 200, "RGBA"),
        (333, 777, "L"),
    ),
    ids=lambda value: str(value),
)
def test_gemma4_staging_matches_the_transformers_processor(
    tmp_path, height, width, mode
):
    # Random bytes exercise every filter tap; 672x960 already fills the
    # patch budget exactly, 1x1 upsamples, and the 10x8000 extremes take the
    # one-pooled-row path. RGBA keeps random alpha to cover transparency.
    generator = np.random.default_rng(height * 7919 + width)
    channels = {"RGB": 3, "RGBA": 4, "L": 1}[mode]
    raw = generator.integers(0, 256, (height, width, channels), dtype=np.uint8)
    image = Image.fromarray(raw[..., 0] if mode == "L" else raw, mode=mode)
    payload = _png(image)

    result = prepare_image(
        _diffusion_gemma_processor(tmp_path),
        MediaCall.VISION_ENCODING,
        payload,
        device=torch.device("cpu"),
        input_images=1,
    )

    # The reference receives the decoded file as Transformers callers pass
    # it, and converts it to RGB itself.
    reference = Gemma4ImageProcessor(
        patch_size=16, max_soft_tokens=280, pooling_kernel_size=3
    )
    decoded = Image.open(io.BytesIO(base64.b64decode(payload)))
    expected = reference(images=[decoded], return_tensors="pt")

    count = int(result.pixels.shape[0])
    assert (result.height, result.width) == (height, width)
    torch.testing.assert_close(
        result.pixels, expected["pixel_values"][0, :count], rtol=0, atol=0
    )
    # Rows follow the staged grid in raster order, which is where the
    # reference places its (column, row) patch positions.
    rows, columns = result.grid_shape
    assert result.grid.tolist() == [[rows, columns]]
    y, x = torch.meshgrid(
        torch.arange(rows), torch.arange(columns), indexing="ij"
    )
    assert torch.equal(
        torch.stack((x, y), dim=-1).reshape(-1, 2),
        expected["image_position_ids"][0, :count],
    )
    assert (expected["image_position_ids"][0, count:] == -1).all()
    assert count // 9 == int(expected["num_soft_tokens_per_image"][0])


@pytest.mark.parametrize(
    "case",
    _DIFFUSION_GEMMA_TOKEN_CASES,
    ids=lambda case: f"{case['width']}x{case['height']}",
)
def test_gemma4_staged_soft_tokens_match_the_server_planner(tmp_path, case):
    image = Image.new("RGB", (case["width"], case["height"]), (90, 120, 150))

    result = prepare_image(
        _diffusion_gemma_processor(tmp_path),
        MediaCall.VISION_ENCODING,
        _png(image),
        device=torch.device("cpu"),
        input_images=2,
    )

    # 16-pixel patches, pooled 3x3 into one soft token each by the tower.
    rows, columns = result.grid_shape
    assert result.pixels.shape[0] == rows * columns
    assert rows * columns // 9 == case["soft_tokens"]


@pytest.mark.parametrize("alpha", ("white", "drop"))
def test_alpha_policy_sets_the_color_of_transparent_pixels(alpha):
    # Opaque pixels keep their color under both policies; fully transparent
    # ones become white or keep their stored color.
    raw = np.zeros((24, 32, 4), dtype=np.uint8)
    raw[..., :3] = (10, 20, 30)
    raw[:, :16, 3] = 255
    processor = ImageProcessor(
        vit=TowerTransform(StrideResize(32, 24, 1, 24 * 32), "signed_unit"),
        staging_dtype=torch.float32,
        alpha=alpha,
    )

    result = prepare_image(
        processor,
        MediaCall.VISION_ENCODING,
        _png(Image.fromarray(raw, mode="RGBA")),
        device=torch.device("cpu"),
        input_images=1,
    )

    stored = torch.tensor([10, 20, 30]).float() / 255 * 2 - 1
    transparent = torch.ones(3) if alpha == "white" else stored
    torch.testing.assert_close(result.pixels[:, 0, 0], stored)
    torch.testing.assert_close(result.pixels[:, 0, 31], transparent)
