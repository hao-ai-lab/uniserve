"""Resident one-image conditioning at the VAE/transformer boundary (CPU)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from uniserve_worker.execution.bounded_storage import BoundedTensorStorage
from uniserve_worker.models.minimax_h3.image_vae import CausalConv3d
from uniserve_worker.models.minimax_h3.layout import (
    H3ComputeInputs,
    H3Layout,
    bind_request_tensors,
    request_tensor_schema,
)
from uniserve_worker.models.minimax_h3.packing import build_packed_layout, patchify_video
from uniserve_worker.models.minimax_h3.transformer import build_transformer_metadata
from uniserve_worker.models.minimax_h3.video_vae import MiniMaxH3VideoVAE

pytestmark = pytest.mark.unit


class ImageVAE:
    """Explicit deterministic VAE boundary double; no target RNG consumption."""

    def encode(self, image):
        return torch.arange(24 * 30 * 52, dtype=torch.float32).reshape(1, 24, 1, 30, 52)


class RotaryRecorder:
    def forward_into(self, positions, cosine, sine, frequencies):
        # Expose the caller's rotary coordinates at this transformer boundary.
        cosine.zero_()
        cosine[:, :3].copy_(positions)
        sine.zero_()


def resident(tags=None, *, local_start=0, local_end=None):
    packed = build_packed_layout(
        text_rows=448,
        reference_shape=None if tags is None else (480, 832),
        presentation_tags=tags,
    )
    layout = H3Layout(
        packed=packed,
        sp_rank=0,
        sp_size=1,
        tp_size=1,
        ulysses_size=1,
        sequence_kind="local",
        context_col_size=1,
        denoiser_participant=True,
        output_owner=False,
        local_start=local_start,
        local_end=packed.padded_rows if local_end is None else local_end,
        frame_count=124,
        reconstruction_unit_frames=(17,) * 6 + (22,),
        attention="dense",
    )
    storage = BoundedTensorStorage(
        {
            name: torch.zeros(spec.shape, dtype=spec.dtype)
            for name, spec in request_tensor_schema(layout).items()
        }
    )
    slot = bind_request_tensors(storage, layout)
    metadata = build_transformer_metadata(layout, torch.device("cpu"))
    execution = H3ComputeInputs(
        layout=layout,
        scratch=SimpleNamespace(
            rotary_positions=torch.empty_like(metadata.positions),
            rotary_frequencies=torch.empty(0),
        ),
        media=None,
        transformer_metadata=metadata,
        base_tile_valid_sizes=packed.tile_valid_sizes,
        prompt_prefix_indices=torch.arange(packed.prefix_tiles, dtype=torch.int32),
        prompt_dense_indices=torch.arange(
            packed.prefix_tiles + packed.video_tiles, dtype=torch.int32
        ),
        prompt_prefix_counts=torch.tensor(packed.prefix_tiles, dtype=torch.int32),
    )
    transformer = SimpleNamespace(pipeline=SimpleNamespace(first=True), rope=RotaryRecorder())
    return slot, execution, transformer


def test_one_image_rows_positions_and_fixed_clock_reach_transformer():
    # 390 Qwen merged patches plus boundaries and label/prompt text are all
    # text-projection rows. The 390 VAE patches are a distinct representation.
    tags = torch.tensor([1, 1, 0] + [0] * 390 + [0, 1, 1])
    slot, execution, transformer = resident(tags)
    encoded = torch.arange(tags.numel(), dtype=torch.bfloat16)[None, :, None].expand(-1, -1, 5376)
    image = torch.zeros(480, 832, 3, dtype=torch.uint8)
    execution.prepare_tensors(
        slot,
        encoded,
        tags.numel(),
        transformer,
        presentation_tags=tags,
        reference_image=image,
        video_vae=ImageVAE(),
    )
    packed = slot.layout.packed
    assert packed.reference_indices.numel() == 390
    assert packed.audio_indices.numel() == 414
    assert packed.video_indices.numel() == 37 * 24 * 42
    assert packed.reference_segments == ((448, 390),)
    assert packed.audio_indices[0] == 896
    assert packed.video_indices[0] == 1344
    assert torch.equal(slot.text_condition[:, : tags.numel()], encoded)
    assert not slot.text_condition[:, tags.numel() :].count_nonzero()
    assert torch.equal(slot.reference_rows, patchify_video(ImageVAE().encode(image))[0])
    assert not slot.video_rows.count_nonzero()
    assert not slot.audio_rows.count_nonzero()

    # FastVideo's semantic origins ignore resident page padding: the image
    # advances the rotary clock by one, not by its 390 patch rows.
    positions = slot.rotary_cosine[:, :3]
    assert torch.equal(positions[: tags.numel(), 0], torch.arange(tags.numel()).float())
    assert torch.equal(
        positions[packed.reference_indices, 0], torch.full((390,), float(tags.numel()))
    )
    audio_times = torch.arange(207).float() + tags.numel() + 1
    assert torch.equal(positions[packed.audio_indices, 0], audio_times.repeat(2))
    raster_positions = positions[packed.video_untile_indices]
    assert raster_positions[0, 0] == tags.numel() + 1
    assert raster_positions[1008, 0] == torch.tensor(tags.numel() + 1 + 5 / 3)
    # Independent NumPy spatial formula from the release's frame grid.
    area = np.sqrt(30 * 52)
    h = np.linspace((1 - 30 / area) / 2, (1 + 30 / area) / 2, 15, endpoint=False) * 32
    w = np.linspace((1 - 52 / area) / 2, (1 + 52 / area) / 2, 26, endpoint=False) * 32
    spatial = torch.tensor(
        np.stack(np.meshgrid(h, w, indexing="ij"), axis=-1).reshape(-1, 2)
    ).float()
    assert torch.equal(positions[packed.reference_indices, 1:], spatial)
    metadata = execution.transformer_metadata
    clock = torch.tensor([0.25, 0.5, 0.999])[metadata.timestep_indices]
    assert torch.equal(clock[packed.reference_indices], torch.full((390,), 0.999))
    assert torch.equal(clock[: tags.numel()], torch.full((tags.numel(),), 0.25))
    assert torch.equal(packed.token_tags[: tags.numel()], tags)
    assert slot.tile_valid_sizes.sum() == tags.numel() + 390 + 414 + 37 * 1008


@pytest.mark.parametrize("interval", [(0, 512), (512, 1024), (1024, 1536)])
def test_reference_rows_follow_sequence_ownership(interval):
    tags = torch.tensor([1, 0, 1])
    slot, execution, transformer = resident(tags, local_start=interval[0], local_end=interval[1])
    image = torch.zeros(480, 832, 3, dtype=torch.uint8)
    execution.prepare_tensors(
        slot,
        torch.zeros(1, 3, 5376),
        3,
        transformer,
        presentation_tags=tags,
        reference_image=image,
        video_vae=ImageVAE(),
    )
    start, stop = max(interval[0], 448), min(interval[1], 838)
    expected = patchify_video(ImageVAE().encode(image))[0][max(0, start - 448) : max(0, stop - 448)]
    assert torch.equal(slot.reference_rows, expected)
    # Initializing target noise cannot overwrite fixed reference conditioning.
    H3ComputeInputs.initialize_tensors(slot, 123)
    assert torch.equal(slot.reference_rows, expected)


def test_text_only_resident_values_are_byte_identical():
    slot, execution, transformer = resident()
    encoded = torch.randn(1, 395, 5376).bfloat16()
    execution.prepare_tensors(slot, encoded, 395, transformer)
    expected = torch.zeros_like(slot.text_condition)
    expected[:, :395] = encoded
    assert torch.equal(slot.text_condition.view(torch.uint8), expected.view(torch.uint8))
    expected_positions = slot.layout.packed.position_ids.float()
    expected_positions[448:, 0] += 395 - 448
    assert torch.equal(slot.rotary_cosine[:, :3], expected_positions)
    assert not slot.reference_rows.numel()
    assert slot.tile_valid_sizes.sum() == 395 + 414 + 37 * 1008


@pytest.mark.parametrize("fault", ["tags", "raster", "latents"])
def test_reference_contract_rejects_mismatched_products(fault):
    tags = torch.tensor([1, 0, 1])
    slot, execution, transformer = resident(tags)
    vae = ImageVAE()
    if fault == "latents":
        vae.encode = lambda image: torch.zeros(1, 24, 2, 30, 52)
    with pytest.raises(ValueError):
        execution.prepare_tensors(
            slot,
            torch.zeros(1, 3, 5376),
            3,
            transformer,
            presentation_tags=tags if fault != "tags" else torch.ones(3, dtype=torch.long),
            reference_image=torch.zeros(
                480 if fault != "raster" else 448, 832, 3, dtype=torch.uint8
            ),
            video_vae=vae,
        )


class PosteriorVAE(nn.Module):
    """Mock checkpoint VAE boundary with an analytically known posterior."""

    use_tiling = False

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.quant_conv = nn.Identity()

    def encoder(self, pixels):
        h, w = pixels.shape[-2:]
        return pixels.new_zeros(1, 48, 1, h // 16, w // 16)


def test_image_encode_sampling_normalization_and_strict_shape():
    vae = MiniMaxH3VideoVAE(PosteriorVAE(), linear_precision="fp32")
    image = torch.zeros(480, 832, 3, dtype=torch.uint8)
    rng = torch.random.get_rng_state().clone()
    result = vae.encode(image)
    expected = (
        torch.randn((1, 24, 1, 30, 52), generator=torch.Generator().manual_seed(42)).half().float()
    )
    expected = (expected - vae.latents_mean) / vae.latents_std
    assert torch.equal(result, expected)
    assert torch.equal(torch.random.get_rng_state(), rng)
    for bad in (image.float(), image[1:], image[None], image[..., :2]):
        with pytest.raises(ValueError):
            vae.encode(bad)


def test_single_frame_causal_convolution_uses_zero_past():
    conv = CausalConv3d(1, 1)
    with torch.no_grad():
        conv.weight.fill_(1)
        conv.bias.zero_()
    image = torch.ones(1, 1, 1, 4, 4)
    assert torch.equal(conv(image), torch.full_like(image, 9))
