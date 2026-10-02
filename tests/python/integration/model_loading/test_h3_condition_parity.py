"""A worker's conditioned request computes what the model computes.

Each case serves a recorded diffusers reference request
(``tools/minimax_h3/diffusers_reference.py``) through the worker's request
path on the released checkpoint's denoiser: ``transformer`` for ``fl2va``,
``transformer_ref`` for ``ref2va``. The worker sizes the request from its
admission (``media.video_shape``), evaluates it in a layout of its own that
its runner prepares while serving beside the captured text-only layouts,
draws the request's noise, retains its refined prompt, hands over its
condition latents (``conditions.condition_latents``,
``MediaBuilder.encode_conditions``) and runs the first denoising step
eagerly.

The same request then runs through the model's public calls on the same
loaded denoiser: ``make_size`` and ``layout_size`` at the worker's text
capacity, ``normal_noise``, ``encode_conditions``, and a
``DenoisingRunner``'s ``prepare_latents``, ``prepare_state`` and first
``step``. Both paths take the reference's refined Qwen hidden states and its
clean condition latents as inputs, and the worker path must equal the model
path bit for bit: the size and layout, the seeded draws (each visual
condition's, then the video's and the audio's), the retained conditioning
(prompt, anchored conditions, zero rows past them), the state tables and
initial samples, and the samples after the first step.

How closely the model follows the diffusers reference is the model's own
contract, characterized at the model level; each case prints the first
step's deviation from the reference beside the reference's own deviation
between two valid attention kernels (``envelopes/<workload>.json``) for
information only.

The cases need the released checkpoint (``UNISERVE_H3_MODEL``), the
reference artifacts (``UNISERVE_MINIMAX_H3_REFERENCE``) and the FFmpeg build
the reference decoded with (``UNISERVE_FFMPEG``), which probes the media the
request is planned from.
"""

import gc
import itertools
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from tests.python.fixtures.h3_conditions import published, recorded_request
from uniserve.diffusion import normal_noise
from uniserve.distributed import Communicator, DeviceMesh
from uniserve.execution import DenoisingRunner
from uniserve.media import image, video
from uniserve.model import Condition, ConditionRole, LatentInput
from uniserve.runtime import ExecutionContext, TensorBuffers
from uniserve_models import loading as models
from uniserve_models.minimax_h3 import DenoiserInput, processing, weight_config
from uniserve_worker.config.deployment import ComponentConfig, ParallelConfig
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.execution import media
from uniserve_worker.execution.conditions import condition_latents
from uniserve_worker.execution.model_executor import ModelExecutor
from uniserve_worker.model_executor.component_binding import ComponentBinding
from uniserve_worker.protocol.batch import DiffusionParams, NewRequest
from uniserve_worker.protocol.identity import RequestKey
from uniserve_worker.protocol.video import VideoAdmission, VideoTask
from uniserve_worker.storage.latent_pool import LatentPool
from uniserve_worker.storage.request_slots import RequestSlots

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.model("minimax_h3"),
    pytest.mark.slow,
]

DEVICE = torch.device("cuda", 0)
SEED = 42
# Device storage kept outside graphs for the work areas of eager steps: the
# worker's in the request's own layout and the model path's beside it.
EAGER_MARGIN = 48 << 30

# Each workload's denoiser, duration, text capacity and the condition rows
# its worker provisions, enough for the workload's own conditions.
WORKLOADS = {
    "fl2va_first_8s": ("denoiser", 8.0, 2048, 1024),
    "ref2va_image_audio_5s": ("reference_denoiser", 5.0, 8192, 40_960),
    "ref2va_video_audio_5s": ("reference_denoiser", 5.0, 8192, 40_960),
}


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


def _rel(value: torch.Tensor, reference: torch.Tensor) -> float:
    value, reference = value.double().cpu(), reference.double().cpu()
    return float((value - reference).norm() / reference.norm())


def _equal(value: torch.Tensor, expected: torch.Tensor) -> bool:
    return value.shape == expected.shape and torch.equal(
        value, expected.to(value.device)
    )


def _conditions(plan: processing.RequestPlan) -> tuple[Condition, ...]:
    """Each planned condition as the denoiser sizes it, in request order.

    A keyframe is one frame on the generated canvas, an image reference
    one frame at the size it is resized to, a video reference the frames
    the video encoder encodes with its soundtrack's resampled samples, and
    an audio reference its resampled samples.
    """
    result = []
    for condition in plan.conditions:
        prepared = condition.prepared
        if isinstance(prepared, processing.KeyframeFit):
            role = (
                ConditionRole.FIRST_FRAME
                if prepared.position is processing.FramePosition.FIRST
                else ConditionRole.LAST_FRAME
            )
            result.append(Condition(role, video.Config(1, plan.canvas)))
        elif isinstance(prepared, image.Config):
            result.append(
                Condition(ConditionRole.REFERENCE, video.Config(1, prepared))
            )
        elif isinstance(prepared, processing.VideoClip):
            sound = prepared.soundtrack
            result.append(
                Condition(
                    ConditionRole.REFERENCE,
                    video.Config(prepared.vae_frames, prepared.canvas),
                    0 if sound is None else sound.samples,
                )
            )
        else:
            result.append(
                Condition(ConditionRole.REFERENCE, None, prepared.samples)
            )
    return tuple(result)


def _vision_spans(tags: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """The presentation's maximal runs of video-tagged tokens."""
    spans, start = [], 0
    for tag, run in itertools.groupby(tags):
        stop = start + sum(1 for _ in run)
        if tag == processing.VIDEO_TAG:
            spans.append((start, stop))
        start = stop
    return tuple(spans)


class _Worker:
    """One rank serving a denoiser as the worker does, with two slots."""

    def __init__(self, checkpoint, component, seconds, text, conditions):
        source = models.read_config(checkpoint, modules=frozenset({component}))
        self.model = models.load_model(
            source,
            device=DEVICE,
            weights=weight_config(source.model, preset="quality"),
        ).model
        config = WorkerConfig(
            device=str(DEVICE),
            max_sequence_tokens=text,
            max_video_seconds=seconds,
            min_video_seconds=seconds,
            video_text_capacities=(text,),
            max_condition_rows=conditions,
            max_request_pool_size=2,
            deployment_components=(component,),
        )
        dimensions = ParallelConfig().dimensions
        group = Communicator((0,), 0, device=DEVICE)
        binding = ComponentBinding(
            component,
            ComponentConfig((0,)),
            group,
            DeviceMesh(
                ranks=(0,),
                rank=0,
                shape=tuple(size for _, size in dimensions),
                axes=tuple(axis for axis, _ in dimensions),
            ),
            group.device,
        )
        self.runner = ModelExecutor(
            self.model, config, bindings={component: binding}
        )
        builder = self.runner.media_builder
        pages = builder.sample_pages
        self.slots = RequestSlots(
            2, state_buffers=self.runner.state_buffers, device=DEVICE
        )
        self.pool = LatentPool(
            request_pool_size=2,
            num_pages=2 * pages.pages + 1,
            page_units=pages.page_units,
            latent_width=1,
            dtype=pages.dtype,
            device=DEVICE,
            staging=False,
        )
        self.runner.bind_diffusion_storage(self.slots.bank, self.pool)
        try:
            # As a worker binds its storage grant, graphs may take what the
            # device holds free beyond a margin for eager steps' work areas.
            storage = self.runner.graph_storage
            free, _ = torch.cuda.mem_get_info(DEVICE)
            storage.set_budget(
                DEVICE,
                storage.resident_bytes().get(DEVICE, 0)
                + max(0, free - EAGER_MARGIN),
            )
            # Startup prepares and captures the text-only layouts.
            media.prepare_denoising(self.runner, self.slots.tensor_slots)
        except BaseException:
            self.close()
            raise

    def close(self):
        self.runner.close()
        self.slots.close()
        self.pool.close()


@pytest.fixture(scope="module")
def workers(checkpoint):
    """One worker per denoiser, built on first use and kept for its cases."""
    built = {}

    def worker(case):
        component, seconds, text, conditions = WORKLOADS[case]
        if component not in built:
            # The previous denoiser's weights and graphs leave the device
            # before the next one loads and sizes its graph budget.
            while built:
                built.popitem()[1].close()
            gc.collect()
            torch.cuda.empty_cache()
            built[component] = _Worker(
                checkpoint, component, seconds, text, conditions
            )
        return built[component]

    yield worker
    for value in built.values():
        value.close()


@pytest.mark.parametrize("case", tuple(WORKLOADS))
@torch.inference_mode()
def test_conditioned_request_computes_the_model_first_step(
    workers, root, ffmpeg, case
):
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        patchify_video_latents,
    )

    component, _, text, _ = WORKLOADS[case]
    worker = workers(case)
    runner, builder = worker.runner, worker.runner.media_builder
    denoiser = getattr(worker.model, component)
    run = root / "reference" / "diffusers" / case / "seed42"
    presentation = json.loads((run / "presentation.json").read_text())
    with published() as publish:
        _, plan, _, conditions = recorded_request(root, case, ffmpeg, publish)
    tokens = tuple(presentation["token_ids"])
    tags = tuple(presentation["tags"])
    video_admission = VideoAdmission(
        VideoTask(plan.task.value), text_tags=tags, conditions=conditions
    )
    admission = NewRequest(
        RequestKey(1, 0, 0),
        request_pool_idx=1,
        diffusion=DiffusionParams(
            plan.num_frames,
            len(runner.video_decoder.frame_slices(plan.num_frames)),
            builder.num_steps,
            SEED,
            width=plan.canvas.width,
            height=plan.canvas.height,
        ),
        video=video_admission,
        prompt_token_ids=tokens,
    )

    # The inputs both paths take: the reference's hidden states, refined
    # once, and its clean condition latents, each condition's visual rows
    # before its audio rows.
    hidden = load_file(run / "text.safetensors")["qwen_hidden_states_50"][0]
    refined = runner.encode_conditioning(hidden.to(DEVICE)).values[0]
    recorded = load_file(run / "conditions.safetensors")
    pixels = (
        patchify_video_latents(recorded[f"video_condition.{index}"], (1, 2, 2))
        for index in itertools.count()
    )
    tracks = (
        recorded[f"audio_condition.{index}"] for index in itertools.count()
    )
    visual, audio, latents = [], [], []
    for condition in conditions:
        if condition.pixels is not None:
            visual.append(next(pixels).to(DEVICE))
            latents.append(visual[-1])
        if condition.audio is not None:
            audio.append(next(tracks).to(DEVICE))
            latents.append(audio[-1])

    # The worker path, which evaluates the request in a layout of its own
    # and steps it eagerly beside the captured text-only layouts.
    size = media.video_shape(runner, admission)
    layout = builder.layout(size)
    assert layout not in builder.layouts()
    entry = runner.diffusion_layout(layout)

    views = worker.slots.tensors(1).view(builder.buffers(size))
    builder.stage_request(size, views, seed=SEED)
    initial = builder.sample_views(
        size, worker.pool.bank_view(1, builder.slot_pages(1))
    )
    for target, value in builder.initialize(
        size,
        views,
        initial,
        constants=entry.constants,
        workspace=entry.workspace,
    ):
        target.copy_(value)
    builder.store_conditioning(size, views, refined)

    # The latent encoders publish every visual condition's rows, then every
    # audio track's, in request order.
    reads = tuple(torch.cat(group) for group in (visual, audio) if group)
    builder.encode_conditions(
        size, views, condition_latents(video_admission, reads)
    )

    diffusion = runner.diffusion
    assert diffusion.captures
    schedules = media.open_state(runner, size).schedules
    samples = builder.sample_views(size, diffusion.samples)
    ladder = diffusion.bind(
        layout,
        tuple(
            builder.bind(size, views, samples, schedules, index)
            for index in range(builder.num_steps)
        ),
        schedules,
        state=views,
        slot=1,
        pages=builder.slot_pages(1),
    )
    first = {name: value.clone() for name, value in initial.items()}
    result = runner.run_denoising(ladder, 0, 1)
    assert result.stats.cuda_graph_runtime_mode_counts == {"eager": 1}
    successor = builder.sample_views(
        size, worker.pool.bank_view(0, builder.slot_pages(1))
    )

    # The model path on the same denoiser, in the layout of the worker's
    # text capacity.
    model_conditions = _conditions(plan)
    model_size = denoiser.make_size(
        plan.num_frames,
        len(tokens),
        canvas=plan.canvas,
        conditions=model_conditions,
        vision_spans=_vision_spans(tags),
    )
    model_layout = denoiser.layout_size(
        denoiser.make_size(
            plan.num_frames,
            text,
            canvas=plan.canvas,
            conditions=model_conditions,
        )
    )
    assert size == model_size
    assert layout == model_layout

    draws = tuple(
        torch.empty(shape, dtype=torch.float32)
        for shape in denoiser.condition_noise_shapes(model_size)
    )
    noise = {
        name: torch.empty(
            (1, *denoiser.noise_shape(name, model_layout)), dtype=torch.float32
        )
        for name in denoiser.modalities
    }
    normal_noise((SEED,), out=(*draws, *noise.values()))
    worker_draws = builder.condition_noise(size, views)
    assert len(worker_draws) == len(draws)
    for value, expected in zip(worker_draws, draws, strict=True):
        assert _equal(value, expected)
    for name in denoiser.modalities:
        assert _equal(views[f"{name}_noise"], noise[name][0]), name

    conditioning = torch.zeros(
        denoiser.text_condition_rows(model_layout),
        denoiser.text_condition_width,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    conditioning[: len(tokens)].copy_(refined)
    denoiser.encode_conditions(
        model_size,
        model_layout,
        latents=tuple(latents),
        noise=draws,
        out=conditioning,
    )
    assert _equal(views["text_condition"], conditioning)

    requirements = denoiser.state_buffers(model_layout)
    with (
        ExecutionContext(denoiser) as context,
        TensorBuffers.allocate(requirements, device="cpu") as host,
        TensorBuffers.allocate(requirements, device=DEVICE) as backing,
    ):
        model_runner = DenoisingRunner(denoiser, context=context)
        model_runner.warmup(model_layout)
        request, staged = backing.view(requirements), host.view(requirements)
        model_runner.prepare_latents(
            (model_layout,),
            noise=noise,
            state={
                name: staged[name].unsqueeze(0) for name in denoiser.modalities
            },
        )
        model_runner.prepare_state(
            (model_size,),
            layouts=(model_layout,),
            out={
                name: value
                for name, value in staged.items()
                if name not in denoiser.modalities
            },
        )
        for name, value in request.items():
            value.copy_(staged[name])
        for name in builder.tables(size):
            assert _equal(views[name], request[name]), name
        for name in denoiser.modalities:
            assert _equal(first[name], request[name]), name

        model_schedules = denoiser.make_schedules(
            denoiser.num_steps, shift=None, device=DEVICE
        )
        inputs = DenoiserInput(
            {
                name: (
                    LatentInput(
                        request[name], model_schedules[name].timesteps[0]
                    ),
                )
                for name in denoiser.modalities
            },
            (model_layout,),
            model_schedules["video"].step(0),
            (conditioning,),
        )
        model_runner.step(inputs, model_schedules, state=request)
        for name in denoiser.modalities:
            assert _equal(successor[name], request[name]), name

    floors = json.loads((root / "envelopes" / f"{case}.json").read_text())
    with safe_open(run / "trajectory.safetensors", "pt") as handle:
        for name in denoiser.modalities:
            reference = handle.get_slice(f"{name}_samples")[0]
            print(
                f"{case} {name} step 0 deviation from the reference "
                f"{_rel(successor[name], reference):.3e}, reference kernel "
                f"floor {floors[f'{name}_sample_rel_l2'][0]:.3e}"
            )
