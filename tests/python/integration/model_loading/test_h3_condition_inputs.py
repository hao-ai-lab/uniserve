"""A worker encodes MiniMax-H3 conditions as the reference conditioning does.

Each case serves the conditions of a recorded diffusers reference request
through the worker's own path:
the media reader decodes the official media into the conditioner's patch
rows, the video encoder's pixels and the audio encoder's PCM
(``tests/python/fixtures/h3_conditions.py``), and a ``ModelExecutor``
holding the released checkpoint's encoders runs the request's vision, text
and latent encodings (``uniserve_worker.execution.conditions``).

- The presented prompt's ``hidden_states[50]``, with its vision tokens
  spliced in at their M-RoPE positions, follows the H3 text encoder
  contract (``test_h3_text_encoder.py``): measured against the reference
  evaluated in FP32 on the same BF16 weights, the relative L2 error of the
  whole tensor and the median and 90th percentile token errors stay within
  twice the recorded BF16 reference's own.
- The condition latents, every visual unit encoded in its own call as
  separate ranks encode them, follow the H3 latent encoding contract
  (``test_h3_latent_encoding.py``) against the recorded clean condition
  latents: one FP16 ulp of the posterior sample for video, the FP32 VAE
  parity tolerance for audio.

The cases need the released checkpoint (``UNISERVE_H3_MODEL``), the
reference artifacts (``UNISERVE_MINIMAX_H3_REFERENCE``) and the FFmpeg build
the reference decoded with (``UNISERVE_FFMPEG``).
"""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from tests.python.fixtures.h3_conditions import (
    published,
    read,
    recorded_request,
    vision_encoder,
)
from uniserve.distributed import Communicator, DeviceMesh
from uniserve_models import loading as models
from uniserve_worker.config.deployment import ComponentConfig, ParallelConfig
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.execution.conditions import (
    condition_units,
    encode_tracks,
    encode_units,
    vision_features,
    vision_grids,
)
from uniserve_worker.execution.model_executor import ModelExecutor
from uniserve_worker.model_executor.component_binding import ComponentBinding
from uniserve_worker.protocol.video import VideoAdmission, VideoTask

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.model("minimax_h3"),
    pytest.mark.slow,
]

DEVICE = "cuda:0"
# H3 conditions on the output of the 50th decoder layer.
LAYER = 50
# The video encoder rounds each posterior sample to FP16 before normalizing
# it (see ``test_h3_latent_encoding.py``).
FP16_ULP = 2**-10
CASES = ("fl2va_first_8s", "ref2va_image_audio_5s", "ref2va_video_audio_5s")


def _path(variable: str, what: str) -> Path:
    value = os.environ.get(variable, "")
    if not value:
        pytest.fail(f"{variable} must name {what}")
    return Path(value)


@pytest.fixture(scope="module")
def checkpoint() -> Path:
    return _path("UNISERVE_H3_MODEL", "a MiniMax-H3 checkpoint directory")


@pytest.fixture(scope="module")
def root() -> Path:
    return _path(
        "UNISERVE_MINIMAX_H3_REFERENCE", "the MiniMax-H3 reference artifacts"
    )


@pytest.fixture(scope="module")
def ffmpeg() -> str:
    return str(_path("UNISERVE_FFMPEG", "the reference's FFmpeg build"))


def _runner(checkpoint: Path, modules, placed) -> ModelExecutor:
    """Load ``modules`` and run ``placed`` components on one device.

    The deployment also places a video denoiser, whose description sizes
    the text capacities; its weights are not loaded.
    """
    source = models.read_config(checkpoint, modules=frozenset(modules))
    model = models.load_model(source, device=DEVICE).model
    config = WorkerConfig(
        device=DEVICE,
        max_sequence_tokens=8192,
        max_video_seconds=15.0,
        deployment_components=(*(name for name, _ in placed), "denoiser"),
    )
    dimensions = ParallelConfig().dimensions
    group = Communicator((0,), 0, device=torch.device(DEVICE))
    bindings = {
        name: ComponentBinding(
            name,
            ComponentConfig((0,), distribution=distribution),
            group,
            DeviceMesh(
                ranks=(0,),
                rank=0,
                shape=tuple(size for _, size in dimensions),
                axes=tuple(axis for axis, _ in dimensions),
            ),
            group.device,
        )
        for name, distribution in placed
    }
    runner = ModelExecutor(model, config, bindings=bindings)
    # A worker captures the video encoder's still-frame tile before it seals
    # startup and admits requests (``ModelExecutor.warmup``), so keyframes
    # and reference images encode through its replay; the other condition
    # encodings have no graph captured at startup and run eagerly.
    runner.capture_tiles()
    runner.complete_startup()
    return runner


def _request(root: Path, case: str, ffmpeg: str, image_bands: int = 1):
    """Read a recorded request's conditions as the media reader does.

    Returns the run directory, the recorded presentation, the request's
    video admission and its condition products. Reference images are
    admitted in ``image_bands`` bands.
    """
    with published() as publish:
        run, plan, _, conditions = recorded_request(
            root, case, ffmpeg, publish, image_bands
        )
        products = read(conditions, vision_encoder(), ffmpeg)
    presentation = json.loads((run / "presentation.json").read_text())
    video = VideoAdmission(
        VideoTask(plan.task.value),
        text_tags=tuple(presentation["tags"]),
        conditions=conditions,
    )
    return run, presentation, video, products


@pytest.mark.parametrize("case", CASES)
def test_condition_latents_match_the_recorded_encoding(
    checkpoint, root, ffmpeg, case
):
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        patchify_video_latents,
    )

    runner = _runner(
        checkpoint,
        ("video_encoder", "audio_encoder"),
        (("latent_encoder", "temporal_units"),),
    )
    try:
        # A four-rank latent encoder bands a reference image into four units.
        run, _, video, (pixels, samples, _) = _request(
            root, case, ffmpeg, image_bands=4
        )
        recorded = load_file(run / "conditions.safetensors")

        # Every unit in its own call, in reverse, as separate ranks encode
        # them; their rows follow each other in unit order.
        units = len(condition_units(video))
        pixels = pixels.to(DEVICE)
        rows = [
            encode_units(video, range(unit, unit + 1), pixels, runner)[0]
            for unit in reversed(range(units))
        ]
        visual = sum(1 for condition in video.conditions if condition.pixels)
        expected = torch.cat(
            tuple(
                patchify_video_latents(
                    recorded[f"video_condition.{index}"], (1, 2, 2)
                )
                for index in range(visual)
            )
        )
        torch.testing.assert_close(
            torch.cat(rows[::-1]).cpu(), expected, rtol=FP16_ULP, atol=FP16_ULP
        )

        tracks = sum(1 for condition in video.conditions if condition.audio)
        if tracks:
            actual, _ = encode_tracks(video, samples.to(DEVICE), runner)
            expected = torch.cat(
                tuple(
                    recorded[f"audio_condition.{index}"]
                    for index in range(tracks)
                )
            )
            # The audio latent stays FP32 end to end; this is the FP32
            # tolerance of the VAE loading parity tests.
            torch.testing.assert_close(
                actual.cpu(), expected, rtol=1e-4, atol=1e-5
            )
    finally:
        runner.close()
        torch.cuda.empty_cache()


def _errors(value: torch.Tensor, exact: torch.Tensor) -> dict[str, float]:
    """Relative L2 error of the tensor and quantiles of its token errors."""
    value, exact = value.float(), exact.float()
    tokens = (value - exact).norm(dim=-1) / exact.norm(dim=-1)
    return {
        "global": float((value - exact).norm() / exact.norm()),
        "median": float(tokens.quantile(0.5)),
        "p90": float(tokens.quantile(0.9)),
    }


@pytest.fixture(scope="module")
def exact(checkpoint, root):
    """Each case's reference ``hidden_states[50]`` evaluated in FP32.

    The reference conditioner runs on the recorded presentation and
    processor inputs, with the BF16 weights widened to FP32. Layers past the
    50th never reach ``hidden_states[50]``; the 51st keeps that state
    unnormalized, as the full stack's is.
    """
    from diffusers.modular_pipelines.minimax_h3.encoders import (
        get_qwen3vl_prompt_embeds,
    )
    from transformers import (
        Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
    )

    processor = Qwen3VLProcessor.from_pretrained(checkpoint / "processor")
    config = Qwen3VLConfig.from_pretrained(checkpoint / "text_encoder")
    config.text_config.num_hidden_layers = LAYER + 1
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            checkpoint / "text_encoder", config=config, dtype=torch.bfloat16
        )
        .float()
        .to(DEVICE)
        .eval()
    )
    results = {}
    with torch.inference_mode():
        for case in CASES:
            run = root / "reference" / "diffusers" / case / "seed42"
            presentation = json.loads((run / "presentation.json").read_text())
            recorded = load_file(run / "text.safetensors")
            vision = {
                name: recorded[name]
                for name in (
                    "qwen_pixel_values",
                    "qwen_image_grid_thw",
                    "qwen_pixel_values_videos",
                    "qwen_video_grid_thw",
                )
                if name in recorded
            }
            vision = {
                name.removeprefix("qwen_"): value
                for name, value in vision.items()
            }
            results[case] = (
                get_qwen3vl_prompt_embeds(
                    model,
                    processor,
                    presentation["token_ids"],
                    vision or None,
                    text_encoder_layer=LAYER,
                    device=DEVICE,
                    dtype=torch.float32,
                )[0]
                .cpu()
                .clone()
            )
    del model
    torch.cuda.empty_cache()
    return results


@pytest.fixture(scope="module")
def presented(checkpoint, root, ffmpeg, exact):
    """Each case's prompt encoded through the worker's vision and text path.

    Depends on ``exact`` so the reference model is released first.
    """
    runner = _runner(checkpoint, ("text_encoder",), (("text_encoder", None),))
    results = {}
    try:
        for case in CASES:
            _, presentation, video, (_, _, patches) = _request(
                root, case, ffmpeg
            )
            # The planned blocks are the ones the reference presented.
            grids = vision_grids(video)
            assert [list(grid) for grid in grids["image_grids"]] == (
                presentation["image_grid_thw"]
            )
            assert [list(grid) for grid in grids["video_grids"]] == (
                presentation["video_grid_thw"]
            )
            features, _ = vision_features(video, patches.to(DEVICE), runner)
            (hidden,) = runner.encode_text(
                presentation["token_ids"], visual=features, **grids
            ).values
            results[case] = hidden.cpu().clone()
    finally:
        runner.close()
        torch.cuda.empty_cache()
    return results


@pytest.mark.parametrize("case", CASES)
def test_presented_prompt_is_as_accurate_as_the_reference(
    root, exact, presented, case, record_property
):
    run = root / "reference" / "diffusers" / case / "seed42"
    reference = load_file(run / "text.safetensors")["qwen_hidden_states_50"][0]
    actual = presented[case]
    assert actual.shape == reference.shape == (reference.shape[0], 5120)

    ours, theirs = _errors(actual, exact[case]), _errors(reference, exact[case])
    for statistic in ours:
        record_property(f"{case}_{statistic}", ours[statistic])
        record_property(f"{case}_{statistic}_reference", theirs[statistic])
    exceeded = {
        statistic: (ours[statistic], theirs[statistic])
        for statistic in ours
        if ours[statistic] > 2 * theirs[statistic]
    }
    assert not exceeded, (
        f"{case}: errors (ours, reference BF16) exceed twice the "
        f"reference: {exceeded}"
    )
