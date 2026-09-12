"""Contract-sized latent handoff and RGB products without checkpoint weights."""

import pytest
import torch

from uniserve_worker.execution.batch import DecodeRange, MediaTrack, RequestKey
from uniserve_worker.execution.bounded_storage import BoundedTensorStorage
from uniserve_worker.models.minimax_h3.layout import (
    H3ComputeInputs,
    H3Layout,
    bind_request_tensors,
    media_tensor_schema,
    request_tensor_schema,
    tensor_output_layout,
)
from uniserve_worker.models.minimax_h3.packing import (
    audio_latent_frames,
    build_packed_layout,
    unpatchify_video,
)
from uniserve_worker.models.minimax_h3.video_vae import (
    H3VideoAssembler,
    MiniMaxH3VideoDecoder,
    MiniMaxH3VideoVAE,
)
from uniserve_worker.nn.layer import LayerConfig
from uniserve_worker.nn.mesh import Communicator, DeviceMesh, EntryBindings
from uniserve_worker.nn.parallel import EntryConfig, ParallelConfig

pytestmark = pytest.mark.unit


def bindings_for(*entries):
    config = ParallelConfig()
    return EntryBindings(
        {name: EntryConfig((0,), config) for name in entries},
        {name: DeviceMesh((0,), 0, config) for name in entries},
        Communicator(),
    )


def cpu_storage(schema):
    # The numerical boundary accepts caller-owned CPU tensors. Pinning is a
    # CUDA transport concern and is not needed for these storage-view tests.
    return BoundedTensorStorage(
        {name: torch.empty(spec.shape, dtype=spec.dtype) for name, spec in schema.items()}
    )


@pytest.mark.parametrize(
    "attention,height,width", [("dense", 768, 1344), ("vsa", 768, 1344), ("dense", 480, 832)]
)
def test_no_reference_target_noise_is_byte_identical_to_checkpoint_rng(attention, height, width):
    bindings = bindings_for("denoiser")
    layout = H3Layout.build(
        bindings,
        frames=39,
        text_rows=64,
        audio_frames=audio_latent_frames(39),
        height=height,
        width=width,
        attention=attention,
    )
    slot = bind_request_tensors(cpu_storage(request_tensor_schema(layout)), layout)
    video, audio = H3ComputeInputs.initialize_tensors(slot, 123)
    raster_rows = video[1][torch.argsort(layout.packed.video_raster_indices)]
    actual_video = unpatchify_video(raster_rows, frames=12, height=height // 16, width=width // 16)
    product = tensor_output_layout(
        bindings,
        "denoiser",
        0,
        None,
        frames=39,
        text_rows=64,
        prompt_tokens=1,
        audio_frames=audio_latent_frames(39),
        height=height,
        width=width,
    )
    assert product.shape == (12 * (height // 32) * (width // 32), 96)
    assert video[1].shape == product.shape
    # Checkpoint protocol: CPU FP32 NCTHW video noise followed by audio noise
    # from the same generator. Physical packing must not change these bytes.
    generator = torch.Generator().manual_seed(123)
    expected_video = torch.randn((1, 24, 12, height // 16, width // 16), generator=generator)
    expected_audio = torch.randn((2 * audio_latent_frames(39), 32), generator=generator)
    assert torch.equal(actual_video.view(torch.uint8), expected_video.view(torch.uint8))
    assert torch.equal(audio[1].view(torch.uint8), expected_audio.view(torch.uint8))


@pytest.mark.parametrize("height,width,frames", [(768, 1344, 39), (480, 832, 124)])
def test_decoder_handoff_and_rgb_products_preserve_target_geometry(height, width, frames):
    bindings = bindings_for("video_decoder", "output")
    layout = H3Layout.build(
        bindings,
        frames=frames,
        text_rows=64,
        audio_frames=audio_latent_frames(frames),
        height=height,
        width=width,
    )
    storage = cpu_storage(media_tensor_schema(layout, bindings))
    execution = H3ComputeInputs.bind(bindings, layout, storage, None, None, torch.device("cpu"))
    slot = bind_request_tensors(cpu_storage(request_tensor_schema(layout)), layout)
    # The actual VAE adapter can pack decoder inputs with meta checkpoint
    # parameters: no neural forward or substituted internal component is used.
    decoder = MiniMaxH3VideoVAE(
        MiniMaxH3VideoDecoder(layer_config=LayerConfig(Communicator(), None)),
        linear_precision="fp32",
    )
    packed = layout.packed
    rows_per_frame = (height // 32) * (width // 32)
    # Each latent frame has a distinct value, exposing temporal stride errors.
    raster = torch.arange(packed.video_frames, dtype=torch.float32).repeat_interleave(
        rows_per_frame
    )
    latents = raster[packed.video_raster_indices, None].expand(-1, 96).contiguous()
    assembler = H3VideoAssembler(torch.device("cpu"))
    total_frames = 0
    previous_value = None
    for unit, count in enumerate(layout.reconstruction_unit_frames):
        decoder_input = decoder.prepare_input(execution, latents, unit, 1, 0)
        assert decoder_input.shape == (1, 24, 7, height // 16, width // 16)
        expected = torch.arange(unit * 5, unit * 5 + 7).view(1, 1, 7, 1, 1).float()
        assert torch.equal(decoder_input, expected.expand_as(decoder_input))
        product = tensor_output_layout(
            bindings,
            "video_decoder",
            0,
            DecodeRange(RequestKey(1, 1, 0), 1, MediaTrack.VIDEO, unit, 1),
            frames=frames,
            text_rows=64,
            prompt_tokens=1,
            audio_frames=packed.audio_frames,
            height=height,
            width=width,
        )
        assert product.shape == (1, 1, 3, 25, height, width)
        value = -0.5 + unit / 8
        segments = torch.full(product.shape, value, dtype=torch.float16)
        rgb = assembler.assemble(slot, execution, segments, unit, 1)
        assert rgb.shape == (count, height, width, 3)
        assert rgb.dtype == torch.uint8
        # Independent checkpoint pixel formula, including half-precision
        # cross-fade arithmetic and FP32 normalization before integer rounding.
        frame_values = torch.full((count,), value, dtype=torch.float16)
        if previous_value is not None:
            mix = torch.arange(5, dtype=torch.float16) / 5
            frame_values[:5] = previous_value * (1 - mix) + value * mix
        colors = (
            (
                (
                    frame_values.float()[:, None] * torch.tensor([0.229, 0.224, 0.225])
                    + torch.tensor([0.485, 0.456, 0.406])
                ).clamp(0, 1)
                * 255
            )
            .round()
            .byte()
        )
        assert torch.equal(rgb, colors[:, None, None, :].expand_as(rgb))
        previous_value = value
        total_frames += count
    assert total_frames == frames


@pytest.mark.parametrize("height,width", [(0, 832), (480, 831), (481, 832)])
def test_target_geometry_rejects_truncated_vae_or_patch_pixels(height, width):
    with pytest.raises(ValueError, match="multiples of 32"):
        build_packed_layout(text_rows=64, height=height, width=width)
