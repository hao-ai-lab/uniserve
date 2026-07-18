"""SenseNova understanding-input contract: preprocessing + patch-block ingest."""
from __future__ import annotations

import base64
import io
import math

import pytest
import torch
from PIL import Image

from uniserve_worker.models.interleaved_text import TextCache
from uniserve_worker.models.sensenova.interleave_runtime import InputImageIngestDriver
from uniserve_worker.processors import get_processor_for_model
from uniserve_worker.processors.sensenova import (
    SENSENOVA_IMAGE_GEOMETRY,
    SenseNovaImageProcessor,
    smart_resize,
)
from uniserve_worker.runtime.residency import encoder_handle_from_mm_hash


def _reference_smart_resize(height, width, factor=32, min_pixels=512 * 512, max_pixels=2048 * 2048):
    """Reference pipeline formula (vLLM-Omni ``_smart_resize``), kept verbatim."""
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
    flattened, grid_hw = processor.understanding_patches(Image.new("RGB", (1000, 700), (10, 20, 30)))
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
    """Minimal InputImageIngestOwner double recording the forward call."""

    device = "cpu"
    hidden = 8

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.feature_calls = 0

    def interleaved_image_features(self, image_input, *, grid_hw, gen_model=False):
        assert not gen_model
        self.feature_calls += 1
        merge = 2
        tokens = int(grid_hw[0, 0]) * int(grid_hw[0, 1]) // (merge * merge)
        return torch.zeros(tokens, self.hidden)

    def interleaved_image_downsample_ratio(self) -> float:
        return 0.5

    def interleaved_text_forward(self, **kwargs):
        self.calls.append(kwargs)
        past = kwargs["past_key_values"]
        past.appended += kwargs["inputs_embeds"].shape[1]
        return _FakeOutputs(past, torch.zeros(1, kwargs["inputs_embeds"].shape[1], 4))


def test_ingest_appends_patch_block_at_shared_t_index():
    owner = _FakeOwner()
    driver = InputImageIngestDriver(owner)
    cache = TextCache()
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
    driver = InputImageIngestDriver(owner)
    grid_hw = torch.tensor([[4, 6]])
    flattened = torch.zeros(24, 3 * 16 * 16)
    embeddings = driver.encode_understanding_image(flattened, grid_hw)

    for t_index in (3, 9):
        cache = TextCache()
        cache.past = _FakePast()
        assert driver.ingest_understanding_embeddings(
            cache,
            embeddings,
            grid_hw,
            t_index=t_index,
        ) == 6
        assert cache.t_index == t_index

    assert owner.feature_calls == 1
    assert len(owner.calls) == 2
