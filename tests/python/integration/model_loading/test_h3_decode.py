"""Loaded H3 window reconstruction through public numerical and native calls."""

import os
from pathlib import Path

import pytest
import torch

from uniserve_worker.bootstrap.distributed import initialize_entries, initialize_process_groups
from uniserve_worker.config import WorkerConfig
from uniserve_worker.execution.diffusion_state import DiffusionState
from uniserve_worker.execution.model_runner import ModelRunner
from uniserve_worker.execution.video import prepare_call
from uniserve_worker.loader import LoadRequest, load_model
from uniserve_worker.modeling.batch import DecodeBatch
from uniserve_worker.modeling.components import Call
from uniserve_worker.modeling.geometry import MediaShape
from uniserve_worker.models.minimax_h3.packing import (
    audio_latent_frames,
    build_packed_layout,
    video_latent_frames,
)
from uniserve_worker.nn.parallel import ComponentConfig
from uniserve_worker.runtime.tensor_buffers import TensorBuffers
from uniserve_worker.runtime.tensors import bind_scratch, bind_state, prepare_constants

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
    loaded = load_model(
        LoadRequest(
            model_path=checkpoint,
            execution=WorkerConfig(device="cuda:0", model_dtype="bfloat16"),
            bindings=bindings,
            max_text_rows=64,
            pipeline_depth=8,
            max_video_seconds=39 / 24,
            quantization_config={"mode": "quality"},
        )
    )
    model = loaded.model
    runner = ModelRunner(
        model, loaded.worker_config, bindings=loaded.bindings, schedule=loaded.schedule
    )
    storage = (
        TensorBuffers.allocate(runner.tensor_resources.state, "cuda:0")
        if runner.tensor_resources.state
        else None
    )
    try:
        runner.prepare_fixed_modules()
        assert runner.scratch is not None
        generator = torch.Generator(device="cuda:0").manual_seed(47)
        for frames in (39, 22):
            shape = MediaShape(768, 1344, frames=frames)
            trajectory = DiffusionState(geometry=shape)
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
                model, runner, trajectory, Call.DECODE_VIDEO, shape, storage
            )
            for window in reversed(model.decode_windows(model.output_geometry(frames))):
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
                model, runner, trajectory, Call.DECODE_AUDIO, shape, storage
            )
            result = runner.run_decoder(
                "audio",
                DecodeBatch((audio_rows,), (shape,)),
                constants=constants,
                scratch=scratch,
            )
            samples = round(frames * 32000 / 24)
            assert result.values[0].shape == (samples, 2)
            assert result.values[0].dtype == torch.int16
            torch.testing.assert_close(result.values[0], reference[:samples], rtol=0, atol=0)

            if not postprocess:
                continue
            constants = prepare_constants(model, Call.POSTPROCESS_VIDEO, shape, device="cuda:0")
            state = bind_state(model, Call.POSTPROCESS_VIDEO, shape, storage)
            scratch = bind_scratch(model, Call.POSTPROCESS_VIDEO, shape, runner.scratch)
            # Zero normalized pixels map to checkpoint channel means. The
            # first window must ignore overlap left by the preceding request.
            state["video_overlap"].fill_(1)
            expected_pixel = torch.tensor([124, 116, 104], dtype=torch.uint8, device="cuda:0")
            for window in model.decode_windows(model.output_geometry(frames)):
                segment = torch.zeros(
                    (1, 3, window.segment_frames, 768, 1344),
                    dtype=torch.float16,
                    device="cuda:0",
                )
                processed = model.postprocess_video(
                    (segment,), (window,), state=state, constants=constants, scratch=scratch
                )
                processed.validate(
                    model.tensor_specs(Call.POSTPROCESS_VIDEO, shape), state=state, scratch=scratch
                )
                pixels = processed.values["video"][0]
                assert pixels.shape == (window.frame_stop - window.frame_start, 768, 1344, 3)
                assert torch.eq(pixels, expected_pixel).all()
                assert torch.count_nonzero(segment) == 0
    finally:
        runner.close()
