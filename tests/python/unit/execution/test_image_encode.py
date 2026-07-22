"""Image preprocessing, the system input stage, and sequence-cache ingestion contracts."""

from __future__ import annotations

import base64
import io
import math

import pytest
import torch
from PIL import Image

from uniserve_worker.contracts.forward_batch import EncodeContext
from uniserve_worker.contracts.model_spec import ImageInputSpec, ImagePatchSpec
from uniserve_worker.execution.codec import run_encode_ops
from uniserve_worker.execution.products import ImageEncoder
from uniserve_worker.execution.sequence import SequenceCache
from uniserve_worker.models.bagel import BagelForUnifiedGeneration
from uniserve_worker.nn.vision import build_abs_positions_from_grid_hw
from uniserve_worker.processors import get_processor_for_model
from uniserve_worker.processors.bagel import BagelImageProcessor
from uniserve_worker.processors.image_pipeline import ImageInputPipeline
from uniserve_worker.processors.sensenova import (
    SENSENOVA_IMAGE_GEOMETRY,
    SenseNovaImageProcessor,
    smart_resize,
)
from uniserve_worker.runtime.residency import encoder_handle_from_mm_hash


def _reference_smart_resize(height, width, factor=32, min_pixels=512 * 512, max_pixels=2048 * 2048):
    """Independent resize formula used to validate image geometry."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("aspect")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


@pytest.mark.parametrize(
    "height,width",
    [(700, 1000), (512, 512), (2048, 1152), (3000, 4000), (100, 90), (4096, 4096)],
)
def test_smart_resize_matches_reference(height, width):
    got = smart_resize(
        height,
        width,
        factor=SENSENOVA_IMAGE_GEOMETRY.size_factor,
        min_pixels=SENSENOVA_IMAGE_GEOMETRY.min_pixels,
        max_pixels=SENSENOVA_IMAGE_GEOMETRY.max_pixels,
    )
    assert got == _reference_smart_resize(height, width)


def test_understanding_patch_geometry():
    processor = SenseNovaImageProcessor()
    flattened, grid_hw = processor.understanding_patches(
        Image.new("RGB", (1000, 700), (10, 20, 30))
    )
    grid_h, grid_w = (int(v) for v in grid_hw[0])
    # 700x1000 rounds to 704x992 at factor 32.
    assert (grid_h, grid_w) == (44, 62)
    assert flattened.shape == (grid_h * grid_w, 3 * 16 * 16)
    assert flattened.dtype == torch.float32


def test_rgba_composites_over_white():
    processor = SenseNovaImageProcessor()
    rgba = Image.new("RGBA", (64, 64), (0, 0, 0, 0))  # fully transparent
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    decoded = processor.decode_image_b64(base64.b64encode(buf.getvalue()).decode())
    assert decoded.mode == "RGB"
    assert decoded.getpixel((0, 0)) == (255, 255, 255)


def test_processor_registry_discovers_sensenova_processor():
    from uniserve_worker.models.sensenova.model import SenseNovaU1ForUnifiedGeneration

    processor = get_processor_for_model(SenseNovaU1ForUnifiedGeneration)
    assert isinstance(processor, SenseNovaImageProcessor)


def test_encoder_handle_is_stable_and_nonzero():
    assert encoder_handle_from_mm_hash(0) != 0
    assert encoder_handle_from_mm_hash(1234) == encoder_handle_from_mm_hash(1234)
    assert encoder_handle_from_mm_hash(1234) != encoder_handle_from_mm_hash(1235)


def _deterministic_image_b64(width: int = 96, height: int = 64) -> str:
    image = Image.new("RGB", (width, height))
    image.putdata(
        [((x * 255) // width, (y * 255) // height, (x + y) % 256)
         for y in range(height) for x in range(width)]
    )
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _sensenova_image_spec() -> ImageInputSpec:
    geometry = SENSENOVA_IMAGE_GEOMETRY
    return ImageInputSpec(
        vit=ImagePatchSpec(
            patch_size=geometry.patch_size,
            downsample_ratio=geometry.downsample_ratio,
            min_pixels=geometry.min_pixels,
            max_pixels=geometry.max_pixels,
            multi_image_pixel_budget=geometry.multi_image_pixel_budget,
        ),
    )


def test_input_stage_matches_sensenova_processor_math_bit_for_bit():
    image_b64 = _deterministic_image_b64()
    images = _sensenova_image_spec()
    processor = SenseNovaImageProcessor.from_image_spec(images)
    stage = ImageInputPipeline(images, processor)

    prepared = stage.prepare("vit_encode", image_b64)

    reference = processor.decode_image_b64(image_b64)
    expected_pixels, expected_grid = processor.understanding_patches(reference)
    assert torch.equal(prepared.pixels, expected_pixels)
    assert torch.equal(prepared.grid, expected_grid)
    assert prepared.image_hw == (reference.height, reference.width)


def test_input_stage_matches_bagel_processor_math_bit_for_bit():
    image_b64 = _deterministic_image_b64()
    images = BagelForUnifiedGeneration(config={}).model_spec().inputs.images
    assert images is not None
    processor = BagelImageProcessor.from_image_spec(images)
    stage = ImageInputPipeline(images, processor)

    canvas = processor.prepare_from_b64(image_b64)
    expected_hw = (canvas.size[1], canvas.size[0])

    vit = stage.prepare("vit_encode", image_b64)
    assert torch.equal(vit.pixels, processor.vit_tensor(canvas))
    assert vit.grid is None
    assert vit.image_hw == expected_hw

    vae = stage.prepare("vae_encode", image_b64)
    assert torch.equal(vae.pixels, processor.vae_tensor(canvas))
    assert vae.image_hw == expected_hw


def test_input_stage_stages_declared_dtype():
    image_b64 = _deterministic_image_b64()
    images = _sensenova_image_spec()
    staged_images = ImageInputSpec(vit=images.vit, staging_dtype="bfloat16")
    processor = SenseNovaImageProcessor.from_image_spec(images)

    float_pixels = ImageInputPipeline(images, processor).prepare("vit_encode", image_b64)
    staged = ImageInputPipeline(staged_images, processor).prepare("vit_encode", image_b64)

    assert staged.pixels.dtype == torch.bfloat16
    assert torch.equal(staged.pixels, float_pixels.pixels.to(torch.bfloat16))
    assert torch.equal(staged.grid, float_pixels.grid)


class _TensorOnlyEncodeModel:
    """Encode entry double asserting the system-staged tensor boundary."""

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor | None, torch.Tensor | None, EncodeContext]] = []

    def encode_image(self, pixels, grid=None, *, ctx):
        assert isinstance(ctx, EncodeContext)
        self.calls.append((pixels, grid, ctx))
        return {"req_id": ctx.req_id, "encoder_handle": ctx.handle, "num_tokens": 3}


def test_model_encode_entry_receives_tensors_and_bounded_context():
    images = _sensenova_image_spec()
    stage = ImageInputPipeline(images, SenseNovaImageProcessor.from_image_spec(images))
    model = _TensorOnlyEncodeModel()
    op = {
        "req_id": 5,
        "kind": "vit_encode",
        "image_b64": _deterministic_image_b64(),
        "mm_hash": 77,
        "cond_pos": 9,
        "new_block_ids": [4, 5],
        "pos_range": [10, 12],
    }

    outputs = run_encode_ops(model, (op,), image_stage=stage)

    assert outputs[0]["encoder_handle"] == encoder_handle_from_mm_hash(77)
    pixels, grid, ctx = model.calls[0]
    assert isinstance(pixels, torch.Tensor) and isinstance(grid, torch.Tensor)
    assert ctx.temporal_index == 9
    assert ctx.new_block_ids == (4, 5)
    assert ctx.pos_range == (10, 12)
    assert not hasattr(ctx, "get"), "encode context must be a bounded view, not an op mapping"


def test_cached_encode_row_replays_resident_handle_without_pixels():
    model = _TensorOnlyEncodeModel()

    outputs = run_encode_ops(
        model,
        ({"req_id": 6, "kind": "vit_encode", "image_in": 314, "cond_pos": 2},),
        image_stage=None,
    )

    assert outputs[0]["encoder_handle"] == 314
    pixels, grid, ctx = model.calls[0]
    assert pixels is None and grid is None
    assert ctx.handle == 314 and ctx.temporal_index == 2


class _FakePast:
    def __init__(self) -> None:
        self.appended = 0

    def get_seq_length(self) -> int:
        return 7 + self.appended


class _FakeOutputs:
    def __init__(self, past, logits) -> None:
        self.past_key_values = past
        self.logits = logits


class _FakeOwner:
    """Minimal ImageEncodeAdapter double recording the forward call."""

    device = "cpu"
    hidden = 8

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.feature_calls = 0

    def image_features(self, image_input, *, grid_hw, gen_model=False):
        assert not gen_model
        self.feature_calls += 1
        merge = 2
        tokens = int(grid_hw[0, 0]) * int(grid_hw[0, 1]) // (merge * merge)
        return torch.zeros(tokens, self.hidden)

    def sequence_position_indexes(self, grid_hw, temporal_indexes):
        abs_w, abs_h = build_abs_positions_from_grid_hw(
            grid_hw[:1] // 2,
            device=temporal_indexes.device,
        )
        return torch.stack((temporal_indexes, abs_h, abs_w), dim=0)

    def sequence_forward(self, **kwargs):
        self.calls.append(kwargs)
        past = kwargs["past_key_values"]
        past.appended += kwargs["inputs_embeds"].shape[1]
        return _FakeOutputs(past, torch.zeros(1, kwargs["inputs_embeds"].shape[1], 4))


def test_ingest_appends_patch_block_at_shared_t_index():
    owner = _FakeOwner()
    driver = ImageEncoder(owner)
    cache = SequenceCache()
    cache.past = _FakePast()
    cache.t_index = 11

    grid_hw = torch.tensor([[4, 6]])
    flattened = torch.zeros(24, 3 * 16 * 16)
    num_tokens = driver.ingest_understanding_image(cache, flattened, grid_hw, t_index=12)

    assert num_tokens == 6  # (4//2) * (6//2)
    assert cache.t_index == 12
    call = owner.calls[0]
    indexes = call["indexes"]
    assert indexes.shape == (3, num_tokens)
    assert torch.equal(indexes[0], torch.full((num_tokens,), 12, dtype=torch.long))
    # Spatial rope covers the downsampled 2x3 grid.
    assert indexes[1].max().item() == 1 and indexes[2].max().item() == 2
    mask = call["attention_mask"]["full_attention"]
    assert mask.shape == (1, 1, num_tokens, 7 + num_tokens)
    assert torch.all(mask == 0), "patches attend to the full prefix and each other"


def test_reusable_embeddings_attach_to_each_request_cache_without_reencoding():
    owner = _FakeOwner()
    driver = ImageEncoder(owner)
    grid_hw = torch.tensor([[4, 6]])
    flattened = torch.zeros(24, 3 * 16 * 16)
    embeddings = driver.encode_understanding_image(flattened, grid_hw)

    for t_index in (3, 9):
        cache = SequenceCache()
        cache.past = _FakePast()
        assert (
            driver.ingest_understanding_embeddings(
                cache,
                embeddings,
                grid_hw,
                t_index=t_index,
            )
            == 6
        )
        assert cache.t_index == t_index

    assert owner.feature_calls == 1
    assert len(owner.calls) == 2
