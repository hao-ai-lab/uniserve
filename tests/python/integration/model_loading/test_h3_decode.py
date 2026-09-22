"""Loaded media capabilities preserve window values.

They also preserve exact sample durations.
"""

import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from uniserve.runtime import (
    ExecutionContext,
    TensorBuffers,
    initialize_process_groups,
)
from uniserve.tensors import TensorOutput
from uniserve_models import loading as models
from uniserve_models.minimax_h3.packing import (
    audio_latent_frames,
    build_packing,
    video_latent_frames,
)
from uniserve_worker.bootstrap.distributed import initialize_components
from uniserve_worker.config.deployment import ComponentConfig
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.execution.model_executor import ModelExecutor

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.model("minimax_h3"),
    pytest.mark.slow,
]


@torch.inference_mode()
@pytest.mark.parametrize("precision", ("quality", "balanced"))
def test_window_decoding_matches_native_reconstruction_and_exact_audio_duration(  # noqa: E501
    precision,
):
    checkpoint = os.environ.get("UNISERVE_H3_MODEL", "")
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.fail(
            "UNISERVE_H3_MODEL must name the supported FastH3 VSA "
            "checkpoint directory"
        )
    modules = {
        "video_decoder": ComponentConfig((0,), distribution="temporal_units"),
        "audio_decoder": ComponentConfig((0,)),
        "muxer": ComponentConfig((0,)),
    }
    with initialize_process_groups(
        rank=0, local_rank=0, world_size=1, device="cuda:0"
    ) as groups:
        bindings = initialize_components(groups, modules)
        source = models.read_config(
            checkpoint,
            modules=frozenset(
                (
                    "video_decoder",
                    "audio_decoder",
                    "video_postprocessor",
                )
            ),
        )
        model = models.load_model(
            source, device="cuda:0", precision=precision
        ).model
        runner = ModelExecutor(
            model,
            WorkerConfig(
                device="cuda:0",
                model_dtype="bfloat16",
                max_sequence_tokens=64,
                max_video_seconds=2,
                max_request_pool_size=1,
            ),
            bindings=bindings,
        )
        storage = TensorBuffers.allocate(runner.state_buffers, device="cuda:0")
        generator = torch.Generator(device="cuda:0").manual_seed(47)
        retained = []
        try:
            # Returning to a previously used duration exercises resource reuse
            # after another request has consumed the same reconstruction owner.
            for frames in (39, 22, 39):
                count = video_latent_frames(frames)
                native = (
                    torch.randn(
                        (1, 24, count, 48, 84),
                        device="cuda:0",
                        generator=generator,
                    )
                    * 0.125
                )
                raster = (
                    native.reshape(1, 24, count, 24, 2, 42, 2)
                    .permute(0, 2, 3, 5, 1, 4, 6)
                    .reshape(-1, 96)
                )
                packed = build_packing(num_text_tokens=64, num_frames=frames)
                latents = raster.index_select(
                    0, packed.video_raster_indices.to("cuda:0")
                )
                original = latents.clone()
                windows = model.video_decoder.frame_slices(frames)
                with ExecutionContext(
                    model.video_decoder.decoder, attention="auto"
                ) as context:
                    context.prepare(None)
                    for index in reversed(range(len(windows))):
                        window = windows[index]
                        with context.activate():
                            expected = (
                                model.video_decoder.decoder(
                                    native[:, :, index * 5 : index * 5 + 7]
                                )
                                .unsqueeze(0)
                                .clone()
                            )
                        for _ in range(2):
                            result = runner.run_module(
                                "video_decoder",
                                (latents,),
                                method="decode",
                                size=frames,
                                frames=(window,),
                                num_frames=(frames,),
                            )
                            actual = result.values[0]
                            assert actual.dtype == torch.float16
                            assert actual.isfinite().all()
                            torch.testing.assert_close(
                                actual, expected, rtol=0, atol=0
                            )
                            assert result.layouts[0].local_slice[0] == slice(
                                index, index + 1
                            )
                        retained.append((actual, expected))
                torch.testing.assert_close(latents, original, rtol=0, atol=0)

                count = audio_latent_frames(frames)
                native_audio = (
                    torch.randn(
                        (2, 32, count), device="cuda:0", generator=generator
                    )
                    * 0.05
                )
                samples = round(frames * 32000 / 24)
                with ExecutionContext(model.audio_decoder.decoder) as context:
                    context.prepare(None)
                    with context.activate():
                        expected_audio = model.audio_decoder.decoder(
                            native_audio
                        )[:samples].clone()
                packed = native_audio.transpose(1, 2).reshape(-1, 32)
                result = runner.run_module(
                    "audio_decoder",
                    (packed,),
                    method="decode",
                    size=count,
                    frames=(slice(0, count),),
                    num_samples=(samples,),
                )
                assert result.values[0].shape == (samples, 2)
                assert result.values[0].dtype == torch.int16
                torch.testing.assert_close(
                    result.values[0], expected_audio, rtol=0, atol=0
                )

                # Section 5.5 distributes audio by media unit, and a unit
                # decoded with the decoder's receptive field of context and
                # trimmed is that unit's share of the whole-track decode.
                # `tests/python/unit/models/test_h3_audio_units.py` holds that
                # to exact equality, where arithmetic does not depend on tensor
                # length. It is not asserted here because it does not hold
                # bitwise on these kernels: decoding one unit's samples with
                # the receptive field of context and with twice or three times
                # it, all sufficient, already disagree by the same few units in
                # the last place. `specs/serving-architecture-progress.md`
                # records that measurement and its control.
                for units in (2, 4):
                    pieces = [
                        runner.run_module(
                            "audio_decoder",
                            (packed,),
                            method="decode",
                            size=count,
                            frames=(window,),
                            num_samples=(samples,),
                        ).values[0]
                        for window in model.audio_decoder.unit_frames(
                            samples, units
                        )
                    ]
                    joined = torch.cat(pieces, dim=0)
                    assert joined.shape == expected_audio.shape
                    assert joined.dtype == torch.int16

                # A new timeline ignores overlap retained from prior requests.
                state = storage.view(
                    model.video_postprocessor.state_buffers(frames)
                )
                state["video_overlap"].fill_(1)
                layout = model.video_decoder.output_layout(frames)["video"]
                pixel = torch.tensor(
                    [124, 116, 104], dtype=torch.uint8, device="cuda:0"
                )
                for index, window in enumerate(windows):
                    segment = torch.zeros(
                        (1, *layout.shape[1:]),
                        dtype=layout.dtype,
                        device="cuda:0",
                    )
                    segment_layout = replace(
                        layout,
                        local_slice=(
                            slice(index, index + 1),
                            *layout.local_slice[1:],
                        ),
                    )
                    result = runner.run_module(
                        "video_decoder",
                        (TensorOutput(segment, segment_layout),),
                        method="forward",
                        size=frames,
                        frames=(window,),
                        num_frames=(frames,),
                        state=state,
                    )
                    pixels = result.values[0]
                    assert pixels.shape == (
                        window.stop - window.start,
                        768,
                        1344,
                        3,
                    )
                    assert torch.eq(pixels, pixel).all()
                    assert torch.count_nonzero(segment) == 0
            for value, expected in retained:
                torch.testing.assert_close(value, expected, rtol=0, atol=0)

            # Independent callers can decode audio while a video input is not
            # ready. Their immutable inputs are initialized before either stream
            # starts; the video fence supplies a device-side progress condition.
            video_stream, audio_stream = (
                torch.cuda.Stream(),
                torch.cuda.Stream(),
            )
            video_done = torch.cuda.Event()
            torch.cuda.synchronize()
            with torch.cuda.stream(video_stream):
                torch.cuda._sleep(2_000_000_000)
                video_result = runner.run_module(
                    "video_decoder",
                    (latents,),
                    method="decode",
                    size=frames,
                    frames=(windows[0],),
                    num_frames=(frames,),
                )
                video_done.record(video_stream)
            with torch.cuda.stream(audio_stream):
                audio_result = runner.run_module(
                    "audio_decoder",
                    (native_audio.transpose(1, 2).reshape(-1, 32),),
                    method="decode",
                    size=count,
                    frames=(slice(0, count),),
                    num_samples=(samples,),
                )
                audio_values = audio_result.values[0].cpu()
            assert not video_done.query(), (
                "independent audio waited for the video input"
            )
            torch.testing.assert_close(
                audio_values, expected_audio.cpu(), rtol=0, atol=0
            )
            video_stream.synchronize()
            torch.testing.assert_close(
                video_result.values[0], retained[-1][1], rtol=0, atol=0
            )
        finally:
            runner.close()
            storage.close()
