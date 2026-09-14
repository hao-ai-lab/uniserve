"""Loaded H3 window reconstruction through public numerical and native calls."""

import os
from pathlib import Path

import pytest
import torch

from uniserve.distributed.process_groups import initialize_process_groups
from uniserve.loading import load_model
from uniserve.model.batch import DecodeBatch
from uniserve.model.limits import ModelLimits
from uniserve.model.media import VideoSize
from uniserve.nn.layer import LayerConfig
from uniserve.runtime.tensor_buffers import TensorBuffers
from uniserve.runtime.tensors import bind_scratch, bind_state, prepare_constants
from uniserve_models import resolve_model
from uniserve_models.minimax_h3.packing import (
    audio_latent_frames,
    build_packed_layout,
    video_latent_frames,
)
from uniserve_worker.bootstrap.components import bind_components
from uniserve_worker.bootstrap.distributed import initialize_entries
from uniserve_worker.bootstrap.model_loader import loaded_worker_config
from uniserve_worker.config import ComponentConfig, WorkerConfig
from uniserve_worker.execution.diffusion_state import DiffusionState
from uniserve_worker.execution.model_runner import ModelRunner
from uniserve_worker.execution.video import prepare_call

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.model("minimax_h3"),
    pytest.mark.slow,
]


@pytest.mark.parametrize("postprocess", [True, False])
def test_window_decoding_matches_native_reconstruction_and_exact_audio_duration(postprocess):
    checkpoint = os.environ.get("UNISERVE_H3_MODEL", "")
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.fail(
            "UNISERVE_H3_MODEL must name the supported full FastH3 VSA checkpoint directory"
        )
    with initialize_process_groups(
        rank=0, local_rank=0, world_size=1, device="cuda:0"
    ) as environment:
        _decode_requests(checkpoint, environment, postprocess)


@torch.inference_mode()
def _decode_requests(checkpoint, environment, postprocess):
    components = {
        "video_decoder": ComponentConfig((0,), distribution="temporal_units"),
        "audio_decoder": ComponentConfig((0,)),
    }
    if postprocess:
        components["output"] = ComponentConfig((0,))
    bindings = initialize_entries(environment, components)
    worker_config = WorkerConfig(device="cuda:0", model_dtype="bfloat16")
    source = resolve_model(
        checkpoint,
        components=frozenset(name for name, value in bindings.items() if value.owns),
        quantization={"mode": "quality"},
    )
    paths = dict(source.entry.component_paths)
    meshes = {paths[name]: value.mesh for name, value in bindings.items() if value.mesh is not None}
    parallel = {paths[name]: value.config.parallel_config for name, value in bindings.items()}
    layers = source.configure_layers(
        {
            name: LayerConfig(
                mesh.get_group("tp"),
                None,
                pipeline=mesh.get_group("pp"),
                sequence=mesh.get_group("ulysses"),
            )
            for name, mesh in meshes.items()
        }
    )
    loaded = load_model(
        source.model_class,
        source.config,
        sources=source.weights,
        device=environment.local_device,
        dtype=torch.bfloat16,
        parallel=parallel,
        meshes=meshes,
        layers=layers,
        limits=ModelLimits(text_tokens=64, video_frames=39),
    )
    model = loaded.model
    bind_components(model, bindings)
    worker_config = loaded_worker_config(model, worker_config, bindings, 8)
    schedule = None
    runner = ModelRunner(model, worker_config, bindings=bindings, schedule=schedule)
    storage = (
        TensorBuffers.allocate(runner.state_buffers, "cuda:0") if runner.state_buffers else None
    )
    try:
        runner.prepare_fixed_modules()
        assert runner.scratch is not None
        generator = torch.Generator(device="cuda:0").manual_seed(47)
        for frames in (39, 22):
            shape = VideoSize(frames)
            trajectory = DiffusionState(size=shape)
            video_frames = video_latent_frames(frames)
            native = (
                torch.randn((1, 24, video_frames, 48, 84), device="cuda:0", generator=generator)
                * 0.125
            )
            # Independent native NCTHW inputs supply the reconstruction oracle.
            # The public model input follows the denoiser's tiled patch order.
            raster = (
                native.reshape(1, 24, video_frames, 24, 2, 42, 2)
                .permute(0, 2, 3, 5, 1, 4, 6)
                .reshape(-1, 96)
            )
            packed = build_packed_layout(
                text_rows=64, num_frames=frames, audio_frames=audio_latent_frames(frames)
            )
            latents = raster.index_select(0, packed.video_raster_indices.to("cuda:0"))
            original = latents.clone()
            _, constants, scratch = prepare_call(
                model, runner, trajectory, "decode:video", shape, storage
            )
            for window in reversed(model.decode_windows(model.video_info(frames))):
                reference = (
                    runner.run_module(
                        "video_decoder", native[:, :, window.latent_start : window.latent_stop]
                    )
                    .values[0]
                    .clone()
                    .unsqueeze(0)
                )
                result = runner.run_decoder(
                    "video",
                    DecodeBatch((latents,), (shape,), (window,)),
                    constants=constants,
                    scratch=scratch,
                )
                assert result.values[0].dtype == torch.float16
                assert result.values[0].isfinite().all()
                torch.testing.assert_close(result.values[0], reference, rtol=0, atol=0)
            torch.testing.assert_close(latents, original, rtol=0, atol=0)

            audio_frames = audio_latent_frames(frames)
            native_audio = (
                torch.randn((2, 32, audio_frames), device="cuda:0", generator=generator) * 0.05
            )
            audio_rows = native_audio.transpose(1, 2).reshape(-1, 32)
            reference = runner.run_module("audio_decoder", native_audio).values[0].clone()
            _, constants, scratch = prepare_call(
                model, runner, trajectory, "decode:audio", shape, storage
            )
            result = runner.run_decoder(
                "audio",
                DecodeBatch((audio_rows,), (round(frames * 32000 / 24),)),
                constants=constants,
                scratch=scratch,
            )
            samples = round(frames * 32000 / 24)
            assert result.values[0].shape == (samples, 2)
            assert result.values[0].dtype == torch.int16
            torch.testing.assert_close(result.values[0], reference[:samples], rtol=0, atol=0)

            if not postprocess:
                continue
            constants = prepare_constants(
                model.video_output, VideoSize(shape.frames), device="cuda:0"
            )
            state = bind_state(model.video_output.state_buffers(VideoSize(shape.frames)), storage)
            scratch = bind_scratch(
                model.video_output.workspace_buffers(VideoSize(shape.frames)), runner.scratch
            )
            # Zero normalized pixels map to checkpoint channel means. The
            # first window must ignore overlap left by the preceding request.
            state["video_overlap"].fill_(1)
            expected_pixel = torch.tensor([124, 116, 104], dtype=torch.uint8, device="cuda:0")
            for window in model.decode_windows(model.video_info(frames)):
                segment = torch.zeros(
                    (1, 3, window.segment_frames, 768, 1344),
                    dtype=torch.float16,
                    device="cuda:0",
                )
                processed = model.postprocess_video(
                    (segment,), (window,), state=state, constants=constants, scratch=scratch
                )
                pixels = processed.values["video"][0]
                assert pixels.shape == (window.frame_stop - window.frame_start, 768, 1344, 3)
                assert torch.eq(pixels, expected_pixel).all()
                assert torch.count_nonzero(segment) == 0
    finally:
        runner.close()
