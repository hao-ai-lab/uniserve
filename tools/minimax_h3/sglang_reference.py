"""Generate MiniMax-H3 reference artifacts with SGLang.

Runs one workload of ``workloads.py`` through SGLang's native MiniMax-H3
pipeline in this process on one GPU and records the artifact set of
``diffusers_reference.py``, in its tensor formats, into
``<output>/reference/<impl>/<workload>/seed<seed>/``. SGLang is the numerical
reference for the ref2va layout that combines references with first/last
keyframes (``ref2va_image_keyframe_5s``), which only its hybrid packing
implements; the tool runs any workload SGLang serves.

Deployment. The pipeline is built as ``sglang serve --model-path
/workspace/models/MiniMax-H3 --model-variant <ref2va|fl2va> --num-gpus 1
--performance-mode speed --disable-conditioning-cache`` builds it on its one
rank (SGLang's ``GPUWorker``): native precision as SGLang loads it (DiT BF16
with its FP32 patch, time and output projections, Qwen3-VL BF16, both VAEs
FP32 with the video decode under FP16 autocast, for which SGLang holds the ViT
decoder's block linears in FP16 from load on), every component resident,
eager, no Cache-DiT (``quality="lossless"``), the checkpoint schedule (50 sigma
points, 49 DiT forwards, flow shift 12 for video and 3 for audio). The ref2va
variant loads the checkpoint's native ``Ref2VA/`` partition, whose DiT holds
the tensors of ``transformer_ref`` (Q/K/V fused per head). ``metadata.json``
records the dtypes of the loaded parameters.

DiT attention. ``--attention`` names the PyTorch SDPA kernel of the DiT, its
token refiner included: ``cudnn`` (the canonical reference), ``flash`` or
``efficient``. The DiT is given SGLang's ``torch_sdpa`` backend and the
denoising stage runs inside ``torch.nn.attention.sdpa_kernel`` restricted to
that kernel. Under cuDNN this is the ``scaled_dot_product_attention`` call
that SGLang's SM100 default backend (``dynamic_cudnn_sdpa``) makes in its own
cuDNN-only scope. The text encoder and the VAEs keep SGLang's default
selection. A run fails unless every DiT attention layer resolved
``torch_sdpa`` and the first DiT forward launched exactly the requested
kernel (CUDA profiler); both facts are recorded.

The recording seams sit at the call sites of SGLang's MiniMax-H3 stages:
``encode_ids`` of the Qwen3-VL encoder (presentation, vision inputs and
``hidden_states[50]``), the denoising stage's loop context (clean condition
latents), its packed-layout branch and its denoise loop (refined text,
anchored condition rows, schedules, every DiT velocity before SGLang's
in-place Euler update and every updated sample), and ``torch.randn`` of the
latent-preparation and condition-noise modules (every draw). SGLang's
conventions are converted to the diffusers artifacts' as follows:

* Packed layout: SGLang pads the sequence to a multiple of 64 rows with a
  separate padding document (tag -1). ``position_ids`` and ``token_tags``
  keep the used rows ``[0, cu_seqlens[1])``; ``video_indices``,
  ``audio_indices`` and ``text_indices`` are SGLang's ``img_pos``,
  ``audio_pos`` and ``text_pos``. Rows are in SGLang's order: text, keyframes,
  references in request order (a video reference's soundtrack rows before
  its video rows), generated audio, generated video. Without keyframes this
  is the diffusers order.
* Condition latents: SGLang passes normalized VAE latents as ``[n, 96]``
  patch rows; ``video_condition.<k>`` is their exact unpatchify to ``[1, 24,
  T, H/16, W/16]``, ``audio_condition.<k>`` keeps the channel-major ``[2 Ta,
  32]`` rows, both in request order. ``condition_rows`` holds the anchored
  visual rows in packed order (keyframes first).
* Noise: SGLang reseeds a CPU generator with the request seed for every draw:
  ``video_noise`` ``[1, 24, T, H/16, W/16]``, ``audio_noise`` ``[2 Ta, 32]``,
  then per visual condition in packed order a ``[1, 24, T + n, h, w]`` draw
  (``n`` visual conditions) of which the condition mixes in its first
  ``T_k`` latent frames. ``condition_noise.<k>`` is that slice, ``k`` in
  packed order: keyframes before references, the order of UniServe's
  ``condition_noise_shapes``. ``noise_draw_order`` lists SGLang's draw order.
  ``--replay-noise`` matches draws by name and checks shapes, so the draws of
  a diffusers recording (same names, another order) replay as well.
* Trajectory: per DiT forward, the velocity of the generated rows and the
  generated rows after the Euler update, FP32. Video rows are ``(t, h, w)``
  raster patch rows with channel-major 96 columns and audio rows are
  channel-major, as in the diffusers recordings.
* Schedules: ``sigmas``/``audio_sigmas`` are SGLang's schedules in FP32 and
  ``timesteps``/``audio_timesteps`` are ``float32(1 - sigma)``, the values its
  loop feeds the DiT.
* Decoded media: the video as SGLang delivers it, ``(x * 255).clamp(0,
  255).to(uint8)`` (truncation), and the audio decoder's FP32 ``[2, N]``.

``--teacher <canonical run>`` teacher-forces an alternative kernel: after
each Euler step except the last, the generated rows are overwritten with the
canonical run's recorded samples of that step, so every DiT forward runs on
the canonical run's input (the first one shares the replayed noise, which
defaults to the canonical run's). The per-step relative L2 of the predictions
against the canonical run is printed and recorded in ``metadata.json``.

Prerequisites: run with the interpreter of the SGLang environment
``/workspace/envs/minimax_h3/sglang`` (SGLang ``ef867fa40d`` from
``/workspace/envs/minimax_h3/src/sglang``, the revision of ``refs/sglang``).
The tool puts that environment's ``bin`` directory, which provides the
ffmpeg and ffprobe SGLang's media path requires, first on ``PATH``.

Example, on one GPU: ``CUDA_VISIBLE_DEVICES=3
/workspace/envs/minimax_h3/sglang/bin/python
tools/minimax_h3/sglang_reference.py --workload ref2va_image_keyframe_5s
--seed 42 --attention cudnn --output artifacts/minimax_h3``.
"""

import argparse
import contextlib
import importlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.nn.attention import SDPBackend, sdpa_kernel

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from diffusers_reference import (  # noqa: E402
    CHECKPOINT,
    _checkpoint_revision,
    _git_head,
    write_mp4,
)
from workloads import WORKLOADS, Workload  # noqa: E402

SGLANG_SOURCE = Path("/workspace/envs/minimax_h3/src/sglang")
H3_STAGES = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages."
    "model_specific_stages.minimax_h3"
)

# DiT attention kernels: the PyTorch SDPA backend each name selects and a
# fragment of the CUDA kernel name that backend launches on SM100, by which
# the profiled first DiT forward is checked.
SDPA_KERNELS = {
    "cudnn": (SDPBackend.CUDNN_ATTENTION, "cudnn_generated_fort_native_sdpa"),
    "flash": (SDPBackend.FLASH_ATTENTION, "pytorch_flash::flash_fwd_kernel"),
    "efficient": (SDPBackend.EFFICIENT_ATTENTION, "fmha_cutlassF"),
}

# SGLang attention backend the DiT resolves for every kernel; the kernel is
# chosen by the surrounding sdpa_kernel scope.
DIT_ATTENTION_BACKEND = "torch_sdpa"

# Pipeline components whose loaded precision and attention are recorded.
COMPONENTS = ("transformer", "text_encoder", "video_vae", "audio_vae")


def _h3(name: str):
    """Import a module of SGLang's MiniMax-H3 stage package."""
    return importlib.import_module(f"{H3_STAGES}.{name}")


def rel_l2(value: torch.Tensor, reference: torch.Tensor) -> float:
    """Return ``||value - reference|| / ||reference||`` in float64."""
    value, reference = value.double(), reference.double()
    return float((value - reference).norm() / reference.norm())


def _synchronized_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def _prepare_environment() -> None:
    """Set the process environment of SGLang's serving recipe."""
    # The H3 pipeline requires ffmpeg and ffprobe on PATH; the SGLang
    # environment's bin directory provides the build its media path uses.
    bin_dir = str(Path(sys.executable).parent)
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    os.environ.setdefault("SGLANG_RUST_BUILD_MODE", "never")
    # Importing the Qwen3-VL encoder imports DeepEP, whose import-time check
    # counts the host's NCCL network plugin as a second NCCL runtime; H3 never
    # runs DeepEP.
    os.environ.setdefault("EP_SUPPRESS_NCCL_CHECK", "1")


class _RecordedDraws:
    """Stand-in for a module's ``torch`` that routes ``randn`` to a recorder.

    SGLang draws every noise tensor with ``torch.randn`` on a freshly seeded
    CPU generator inside its latent-preparation and condition-noise modules.
    Installed as such a module's ``torch`` global, this object hands every
    attribute through to ``torch`` except ``randn``, whose calls consume
    ``specs`` in order: ``(name, used_shape)`` pairs where ``used_shape`` is
    the ``[1, C, T, H, W]`` block of a condition draw the caller keeps (its
    first ``T`` latent frames), or ``None`` when the whole draw is used.
    """

    def __init__(
        self, recorder: "Recorder", specs: list[tuple[str, tuple | None]]
    ):
        self._recorder = recorder
        self._specs = list(specs)

    def __getattr__(self, name: str):
        return getattr(torch, name)

    def randn(self, *size, generator=None, **kwargs):
        if not self._specs:
            raise RuntimeError(
                "SGLang made a noise draw the recording does not expect"
            )
        name, used_shape = self._specs.pop(0)
        if len(size) == 1 and not isinstance(size[0], int):
            size = tuple(size[0])
        return self._recorder.draw(name, size, used_shape, generator, kwargs)

    def finish(self) -> None:
        if self._specs:
            raise RuntimeError(
                "SGLang skipped the expected noise draws "
                f"{[name for name, _ in self._specs]}"
            )


class Recorder:
    """Collects the tensors of one SGLang MiniMax-H3 pipeline call.

    ``replay`` maps draw names to stored draws that replace SGLang's fresh
    ones after a shape check. ``teacher`` holds a canonical run's trajectory
    whose samples overwrite the generated rows after every step but the last.
    """

    def __init__(
        self,
        kernel: str,
        replay: dict[str, torch.Tensor] | None = None,
        teacher: dict[str, torch.Tensor] | None = None,
    ):
        self.kernel = kernel
        self.replay = replay
        self.teacher = teacher
        self.draws: dict[str, torch.Tensor] = {}
        self.text: dict | None = None
        self.context = None
        self.clean_conditions: dict | None = None
        self.packed: dict | None = None
        self.token_tags: torch.Tensor | None = None
        self.positive = None
        self.refined_text: torch.Tensor | None = None
        self.condition_rows: torch.Tensor | None = None
        self.sigmas: dict[str, list[float]] | None = None
        self.final_rows: tuple[torch.Tensor, torch.Tensor] | None = None
        self.trajectory: dict[str, list[torch.Tensor]] = {
            f"{modality}_{kind}": []
            for modality in ("video", "audio")
            for kind in ("predictions", "samples")
        }
        self.teacher_deviation: dict[str, list[float]] = {
            "video": [],
            "audio": [],
        }
        self.attention_kernels: list[str] = []
        self.step_seconds: list[float] = []
        self.marks: dict[str, float] = {}

    def draw(self, name, size, used_shape, generator, kwargs) -> torch.Tensor:
        """Make SGLang's draw ``name``; replay and record its used block."""
        value = torch.randn(*size, generator=generator, **kwargs)
        used = value
        if used_shape is not None:
            # Only the temporal extent (dim 2) of a condition draw may exceed
            # the block the condition keeps.
            drawn = tuple(value.shape)
            if (
                len(drawn) != 5
                or drawn[:2] + drawn[3:]
                != tuple(used_shape[:2]) + tuple(used_shape[3:])
                or drawn[2] < used_shape[2]
            ):
                raise ValueError(
                    f"draw {name} of shape {drawn} does not hold the used "
                    f"block {tuple(used_shape)}"
                )
            used = value[:, :, : used_shape[2]]
        if self.replay is not None:
            if name not in self.replay:
                raise KeyError(f"the replayed noise holds no draw {name!r}")
            stored = self.replay[name]
            if tuple(stored.shape) != tuple(used.shape):
                raise ValueError(
                    f"replayed draw {name} has shape {tuple(stored.shape)}, "
                    f"the run uses {tuple(used.shape)}"
                )
            # Writing into the used block keeps SGLang's own slicing and mixing
            # arithmetic on the replayed values.
            used.copy_(stored.to(dtype=used.dtype))
        if name in self.draws:
            raise RuntimeError(f"SGLang drew {name} twice")
        self.draws[name] = used.detach().clone()
        return value

    @contextlib.contextmanager
    def patch(self, pipeline):
        """Install the recording seams for the duration of the context."""
        latent_preparation = _h3("stages.latent_preparation")
        condition_noise = _h3("condition_noise")
        denoise_loop = _h3("denoise_loop")
        denoising = _h3("stages.denoising")
        text_encoding = _h3("stages.text_encoding")
        recorder = self

        # Generated noise: the latent-preparation stage draws the video noise,
        # then the audio noise, each from its own freshly seeded generator.
        latent_draws = _RecordedDraws(
            self, [("video_noise", None), ("audio_noise", None)]
        )

        # Condition noise: one draw per visual condition in packed order; the
        # condition mixes in the first T_k latent frames of its draw.
        original_imgvid = condition_noise.minimax_h3_imgvid_cond_noise_aug_rows

        def imgvid_cond_noise_aug_rows(
            clean_rows, *, condition_shapes, **kwargs
        ):
            specs = [
                (f"condition_noise.{index}", (1, 24, *map(int, shape)))
                for index, shape in enumerate(condition_shapes)
            ]
            draws = _RecordedDraws(recorder, specs)
            condition_noise.torch = draws
            try:
                rows = original_imgvid(
                    clean_rows, condition_shapes=condition_shapes, **kwargs
                )
            finally:
                condition_noise.torch = torch
            draws.finish()
            return rows

        text_stage = next(
            stage
            for stage in pipeline.stages
            if isinstance(stage, text_encoding.MiniMaxH3TextEncodingStage)
        )
        encoder = text_stage.text_encoder
        original_encode = encoder.encode_ids

        def encode_ids(input_ids, **vision):
            started = _synchronized_time()
            hidden = original_encode(input_ids, **vision)
            finished = _synchronized_time()
            if recorder.text is not None:
                raise RuntimeError("SGLang encoded more than one presentation")
            recorder.text = {
                "token_ids": [int(token) for token in input_ids.tolist()],
                "vision": {
                    name: value.detach().cpu().clone()
                    for name, value in vision.items()
                    if value is not None
                },
                "hidden_states": hidden.detach().cpu().clone(),
                "seconds": finished - started,
            }
            return hidden

        original_context = denoising._resolve_full_loop_context

        def resolve_full_loop_context(batch):
            ctx = original_context(batch)
            recorder.context = ctx
            # The clean latents, before the condition noise replaces the
            # loop's condition rows with anchored ones.
            recorder.clean_conditions = _clean_condition_latents(ctx)
            return ctx

        branch_class = denoise_loop.MiniMaxH3DenoiseBranch

        class RecordingBranch(branch_class):
            def __init__(self, *, packed, token_tags, **kwargs):
                recorder.packed = {
                    name: value.detach().cpu().clone()
                    if isinstance(value, torch.Tensor)
                    else value
                    for name, value in packed.items()
                }
                # The text rows already carry the presentation's tags.
                recorder.token_tags = token_tags.detach().cpu().clone()
                super().__init__(packed=packed, token_tags=token_tags, **kwargs)

        original_loop = denoise_loop.minimax_h3_denoise_loop

        def minimax_h3_denoise_loop(**kwargs):
            positive = kwargs["positive"]
            recorder.positive = positive
            recorder.refined_text = (
                positive.static_kwargs["prompt_embeds"].detach().cpu().clone()
            )
            anchors = kwargs.get("keyframe_cond_rows")
            recorder.condition_rows = (
                None
                if anchors is None
                else anchors.detach().float().cpu().clone()
            )
            recorder.sigmas = {
                "video": [float(value) for value in kwargs["sigmas_video"]],
                "audio": [float(value) for value in kwargs["sigmas_audio"]],
            }
            num_steps = len(recorder.sigmas["video"]) - 1
            forward = kwargs["model_forward"]
            step_hook = kwargs["on_step"]
            audio_targets = positive.audio_target_slice
            video_targets = positive.video_target_slice

            def model_forward(model, call_kwargs, step):
                if step == 0:
                    output = recorder._profiled_forward(
                        forward, model, call_kwargs, step
                    )
                else:
                    output = forward(model, call_kwargs, step)
                v_video, v_audio = output
                # The loop updates the samples in place through the velocity
                # tensors, so they are copied before returning.
                recorder.trajectory["video_predictions"].append(
                    v_video.float().cpu().clone()
                )
                recorder.trajectory["audio_predictions"].append(
                    v_audio[audio_targets].float().cpu().clone()
                )
                return output

            def on_step(step, video_rows, audio_rows):
                step_hook(step, video_rows, audio_rows)
                recorder.trajectory["video_samples"].append(
                    video_rows[video_targets].float().cpu().clone()
                )
                recorder.trajectory["audio_samples"].append(
                    audio_rows[audio_targets].float().cpu().clone()
                )
                message = ""
                if recorder.teacher is not None:
                    message = recorder._teacher_step(
                        step,
                        num_steps,
                        video_rows[video_targets],
                        audio_rows[audio_targets],
                    )
                now = _synchronized_time()
                previous = recorder.marks.get(
                    "last_step", recorder.marks["denoise_start"]
                )
                recorder.step_seconds.append(now - previous)
                recorder.marks["last_step"] = now
                sigma = recorder.sigmas["video"][step]
                print(
                    f"step {step + 1}/{num_steps} t={1.0 - sigma:.6f} "
                    f"{now - previous:.2f}s{message}",
                    flush=True,
                )

            kwargs["model_forward"] = model_forward
            kwargs["on_step"] = on_step
            recorder.marks["denoise_start"] = _synchronized_time()
            video_rows, audio_rows = original_loop(**kwargs)
            recorder.final_rows = (
                video_rows[video_targets].float().cpu().clone(),
                audio_rows[audio_targets].float().cpu().clone(),
            )
            return video_rows, audio_rows

        # The DiT runs only inside the denoising stage; the kernel scope ends
        # with it, so the encoders and decoders keep the default dispatch.
        denoise_stage = next(
            stage
            for stage in pipeline.stages
            if isinstance(stage, denoising.MiniMaxH3DenoisingStage)
        )
        original_run = denoise_stage._run_full_loop
        backend = SDPA_KERNELS[self.kernel][0]

        def run_full_loop(batch, server_args):
            with sdpa_kernel([backend]):
                return original_run(batch, server_args)

        latent_preparation.torch = latent_draws
        condition_noise.minimax_h3_imgvid_cond_noise_aug_rows = (
            imgvid_cond_noise_aug_rows
        )
        encoder.encode_ids = encode_ids
        denoising._resolve_full_loop_context = resolve_full_loop_context
        denoise_loop.MiniMaxH3DenoiseBranch = RecordingBranch
        denoise_loop.minimax_h3_denoise_loop = minimax_h3_denoise_loop
        denoise_stage._run_full_loop = run_full_loop
        try:
            yield
        finally:
            latent_preparation.torch = torch
            condition_noise.minimax_h3_imgvid_cond_noise_aug_rows = (
                original_imgvid
            )
            del encoder.encode_ids
            denoising._resolve_full_loop_context = original_context
            denoise_loop.MiniMaxH3DenoiseBranch = branch_class
            denoise_loop.minimax_h3_denoise_loop = original_loop
            del denoise_stage._run_full_loop
        latent_draws.finish()

    def _profiled_forward(self, forward, model, call_kwargs, step):
        """Run the first DiT forward under the CUDA profiler.

        Records the attention kernels it launched and fails unless they are
        exactly the requested SDPA kernel.
        """
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as trace:
            output = forward(model, call_kwargs, step)
            torch.cuda.synchronize()
        names = {
            event.name
            for event in trace.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        }
        launched = {
            kernel
            for kernel, (_, fragment) in SDPA_KERNELS.items()
            if any(fragment in name for name in names)
        }
        self.attention_kernels = sorted(
            name
            for name in names
            if any(fragment in name for _, fragment in SDPA_KERNELS.values())
        )
        if launched != {self.kernel}:
            raise RuntimeError(
                f"the DiT ran SDPA kernels {sorted(launched)}, not "
                f"{self.kernel}: {self.attention_kernels}"
            )
        return output

    def _teacher_step(self, step, num_steps, video_target, audio_target) -> str:
        """Report the step's deviation; continue from the canonical sample."""
        teacher = self.teacher
        for modality in ("video", "audio"):
            ours = self.trajectory[f"{modality}_predictions"][step]
            reference = teacher[f"{modality}_predictions"][step]
            if ours.shape != reference.shape:
                raise ValueError(
                    f"teacher {modality} predictions have shape "
                    f"{tuple(reference.shape)}, the run predicts "
                    f"{tuple(ours.shape)}"
                )
            self.teacher_deviation[modality].append(rel_l2(ours, reference))
        # The last step's sample feeds no forward; it stays the run's own.
        if step + 1 < num_steps:
            video_target.copy_(
                teacher["video_samples"][step].to(video_target.device)
            )
            audio_target.copy_(
                teacher["audio_samples"][step].to(audio_target.device)
            )
        return (
            f" teacher rel L2 video {self.teacher_deviation['video'][-1]:.3e} "
            f"audio {self.teacher_deviation['audio'][-1]:.3e}"
        )


def _clean_condition_latents(ctx) -> dict[str, list[torch.Tensor]]:
    """Return the clean condition latents of a loop context in request order.

    Visual conditions (keyframes, reference images and videos) are unpatchified
    from SGLang's normalized ``[n, 96]`` rows to ``[1, 24, T, H/16, W/16]``;
    audio tracks (audio references and soundtracks of video references) keep
    their channel-major ``[2 Ta, 32]`` rows.
    """
    unpatchify = _h3("packed_tokens").minimax_h3_unpatchify_video_tokens

    def entries(payload, key):
        if payload is None:
            return []
        return list(payload.get(key) or [payload])

    visual = []
    for entry in (
        entries(ctx.keyframe, "keyframes")
        + entries(ctx.ref_image, "images")
        + entries(ctx.ref_video, "videos")
    ):
        frames = int(entry.get("latent_t", 1))
        height, width = int(entry["latent_h"]), int(entry["latent_w"])
        latent = unpatchify(
            entry["rows"].float(),
            latent_shape=[frames, height // 2, width // 2, 24],
            patch_size=[1, 2, 2],
        )
        visual.append((int(entry["condition_index"]), latent.cpu().clone()))
    audio = [
        (int(entry["condition_index"]), entry["rows"].float().cpu().clone())
        for entry in entries(ctx.ref_audio, "audios")
        if int(entry["ref_audio_t"]) > 0
    ]
    return {
        "video": [
            latent for _, latent in sorted(visual, key=lambda item: item[0])
        ],
        "audio": [rows for _, rows in sorted(audio, key=lambda item: item[0])],
    }


def load_pipeline(checkpoint: Path, task: str, scratch: Path):
    """Build SGLang's MiniMax-H3 pipeline on this process's one GPU."""
    from sglang.multimodal_gen.runtime.platforms import (
        initialize_current_platform,
    )

    initialize_current_platform()
    from sglang.multimodal_gen.runtime.platforms.plugins import (
        apply_plugin_hooks,
        load_plugins,
    )

    load_plugins()
    apply_plugin_hooks()
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs,
        set_global_server_args,
    )

    server_args = ServerArgs.from_kwargs(
        model_path=str(checkpoint),
        model_variant="ref2va" if task == "ref2va" else "fl2va",
        num_gpus=1,
        performance_mode="speed",
        disable_conditioning_cache=True,
        component_attention_backends={"transformer": DIT_ATTENTION_BACKEND},
        output_path=str(scratch / "outputs"),
        input_save_path=str(scratch / "inputs"),
    )
    set_global_server_args(server_args)
    if (
        server_args.enable_torch_compile
        or server_args.enable_breakable_cuda_graph
    ):
        raise RuntimeError("the reference runs SGLang's eager recipe")

    from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker

    # The single-rank process group rendezvous on a free loopback port.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    worker = GPUWorker(
        local_rank=0, rank=0, master_port=port, server_args=server_args
    )
    return server_args, worker.pipeline


def build_request(
    server_args,
    workload: Workload,
    inputs: Path,
    seed: int,
    steps: int,
    scratch: Path,
):
    """Turn the workload into the Req SGLang's offline generator would queue."""
    from sglang.multimodal_gen.configs.sample.sampling_params import (
        SamplingParams,
    )
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
    from sglang.multimodal_gen.runtime.pipelines_core.request_utils import (
        expand_request_outputs,
    )

    body = workload.request_body(inputs, seed)
    sampling_params = SamplingParams.from_user_sampling_params_args(
        server_args.model_path,
        server_args=server_args,
        prompt=body["prompt"],
        task=body["task"],
        conditions=body["conditions"],
        target=body["target"],
        seed=seed,
        num_inference_steps=steps,
        quality="lossless",
        save_output=True,
        output_path=str(scratch / "outputs"),
    )
    sampling_params._set_output_file_name()
    req = prepare_request(
        server_args=server_args, sampling_params=sampling_params
    )
    # Probe the media and freeze the canvas, duration and materials.
    sampling_params.prepare_video_request_for_queue(req)
    (req,) = expand_request_outputs(req)
    return req


def _dit_attention_backends(transformer) -> dict[str, int]:
    """Count the SGLang backend and implementation of every DiT attention."""
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
        MiniMaxH3Attention,
    )

    return dict(
        Counter(
            f"{module._attention_backend_enum.name.lower()}/"
            f"{type(module._attention_impl).__name__}"
            for module in transformer.modules()
            if isinstance(module, MiniMaxH3Attention)
        )
    )


def _component_attention_backends(module) -> dict[str, int]:
    """Count a component's SGLang attention implementations by class.

    SGLang's generic attention layer keeps its implementation in
    ``attn_impl``; the Qwen3-VL vision tower and the DiT keep theirs in
    ``_attention_impl``. Implementations bind on first use, so this is read
    after the call.
    """
    counts = Counter()
    for layer in module.modules():
        for attribute in ("attn_impl", "_attention_impl"):
            impl = getattr(layer, attribute, None)
            if impl is not None:
                counts[type(impl).__name__] += 1
    return dict(counts)


def _parameter_dtypes(module) -> dict[str, int]:
    """Element count of a module's parameters and buffers per dtype."""
    counts = Counter()
    for tensor in [*module.parameters(), *module.buffers()]:
        counts[str(tensor.dtype)] += tensor.numel()
    return dict(counts)


def _versions() -> dict:
    import sglang
    import transformers

    ffmpeg = shutil.which("ffmpeg")
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "sglang": sglang.__version__,
        "sglang_source": str(SGLANG_SOURCE),
        "sglang_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=SGLANG_SOURCE,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(),
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tools_commit": _git_head(),
        "ffmpeg": {
            "path": ffmpeg,
            "version": subprocess.run(
                [ffmpeg, "-version"], capture_output=True, text=True
            ).stdout.splitlines()[0],
        },
    }


def _compare_with_source(run_dir: Path, source: Path) -> dict:
    """Compare this run's encoder outputs with the run its noise came from.

    A replayed run shares the source's draws; identical text and condition
    latents show that the two runs differ only in the DiT.
    """
    comparison = {}
    for filename, names in (
        ("text.safetensors", ("qwen_hidden_states_50", "refined_text")),
        ("conditions.safetensors", None),
    ):
        ours_path, theirs_path = run_dir / filename, source / filename
        if not (ours_path.exists() and theirs_path.exists()):
            continue
        ours, theirs = load_file(str(ours_path)), load_file(str(theirs_path))
        for name in names or sorted(ours):
            if name not in theirs:
                continue
            if ours[name].shape != theirs[name].shape:
                comparison[name] = {
                    "equal": False,
                    "shapes": [
                        list(ours[name].shape),
                        list(theirs[name].shape),
                    ],
                }
                continue
            comparison[name] = {
                "equal": bool(torch.equal(ours[name], theirs[name])),
                "rel_l2": rel_l2(ours[name], theirs[name]),
            }
    return comparison


def write_artifacts(
    run_dir: Path,
    args: argparse.Namespace,
    workload: Workload,
    inputs: Path,
    shape: dict,
    output,
    recorder: Recorder,
    timings: dict,
    loaded_dtypes: dict,
    server_args,
    pipeline,
) -> None:
    """Write the diffusers-format artifact set of one recorded SGLang call.

    ``shape`` is the request plan's resolved geometry, ``output`` the
    pipeline's ``OutputBatch`` (decoded media) and ``loaded_dtypes`` the
    components' element counts per dtype right after loading.
    """
    from sglang.multimodal_gen.runtime.entrypoints.utils import (
        _sample_to_uint8_frames,
    )
    from sglang.multimodal_gen.runtime.utils.precision import (
        resolve_decode_precision,
    )

    ctx, packed, positive = recorder.context, recorder.packed, recorder.positive
    if (
        recorder.text is None
        or ctx is None
        or packed is None
        or positive is None
    ):
        raise RuntimeError("the pipeline call bypassed a recording seam")
    latent_t, latent_h, latent_w = ctx.latent_t, ctx.latent_h, ctx.latent_w
    audio_t = ctx.audio_t
    num_steps = len(recorder.sigmas["video"]) - 1
    if len(recorder.trajectory["video_samples"]) != num_steps:
        recorded = len(recorder.trajectory["video_samples"])
        raise RuntimeError(f"recorded {recorded} of {num_steps} steps")

    # Draws in SGLang's draw order.
    draw_order = list(recorder.draws)
    save_file(
        {name: draw.contiguous() for name, draw in recorder.draws.items()},
        str(run_dir / "noise.safetensors"),
    )

    embeddings = ctx.embeddings["positive"]
    text = {
        "qwen_hidden_states_50": recorder.text["hidden_states"][
            None
        ].contiguous(),
        "refined_text": recorder.refined_text[None].contiguous(),
    }
    for name, tensor in recorder.text["vision"].items():
        text[f"qwen_{name}"] = tensor.contiguous()
    save_file(text, str(run_dir / "text.safetensors"))

    token_ids = recorder.text["token_ids"]
    vision = recorder.text["vision"]
    presentation = {
        "token_ids": token_ids,
        "tags": [int(tag) for tag in embeddings["text_token_tags"].tolist()],
        "num_text_rows": len(token_ids),
        "image_grid_thw": vision.get(
            "image_grid_thw", torch.empty(0, 3)
        ).tolist(),
        "video_grid_thw": vision.get(
            "video_grid_thw", torch.empty(0, 3)
        ).tolist(),
    }
    (run_dir / "presentation.json").write_text(json.dumps(presentation) + "\n")

    conditions = {}
    for index, latent in enumerate(recorder.clean_conditions["video"]):
        conditions[f"video_condition.{index}"] = latent.contiguous()
    for index, rows in enumerate(recorder.clean_conditions["audio"]):
        conditions[f"audio_condition.{index}"] = rows.contiguous()
    if recorder.condition_rows is not None:
        conditions["condition_rows"] = recorder.condition_rows.contiguous()
    if conditions:
        save_file(conditions, str(run_dir / "conditions.safetensors"))

    # Used rows only: SGLang's trailing padding document is not part of the
    # layout the diffusers recordings describe.
    used = int(packed["cu_seqlens"][1])
    sigmas, audio_sigmas = recorder.sigmas["video"], recorder.sigmas["audio"]
    save_file(
        {
            "position_ids": packed["img_position_ids"][:used].contiguous(),
            "token_tags": recorder.token_tags[:used]
            .to(torch.long)
            .contiguous(),
            "video_indices": packed["img_pos"].to(torch.long).contiguous(),
            "audio_indices": packed["audio_pos"].to(torch.long).contiguous(),
            "text_indices": packed["text_pos"].to(torch.long).contiguous(),
            "sigmas": torch.tensor(sigmas, dtype=torch.float32),
            "audio_sigmas": torch.tensor(audio_sigmas, dtype=torch.float32),
            # The loop's FP32 timesteps: Python-float 1 - sigma, then FP32.
            "timesteps": torch.tensor(
                [1.0 - sigma for sigma in sigmas[:-1]], dtype=torch.float32
            ),
            "audio_timesteps": torch.tensor(
                [1.0 - sigma for sigma in audio_sigmas[:-1]],
                dtype=torch.float32,
            ),
        },
        str(run_dir / "layout.safetensors"),
    )
    save_file(
        {
            name: torch.stack(values).contiguous()
            for name, values in recorder.trajectory.items()
        },
        str(run_dir / "trajectory.safetensors"),
    )

    # The latents SGLang decodes, unpacked by its own helpers.
    packed_tokens = _h3("packed_tokens")
    video_rows, audio_rows = recorder.final_rows
    save_file(
        {
            "video_latents": packed_tokens.minimax_h3_unpatchify_video_tokens(
                video_rows,
                latent_shape=[latent_t, latent_h // 2, latent_w // 2, 24],
                patch_size=[1, 2, 2],
            ).contiguous(),
            "audio_latents": packed_tokens.minimax_h3_unpack_audio_tokens(
                audio_rows, audio_t=2 * audio_t, audio_channel=2
            ).contiguous(),
        },
        str(run_dir / "final.safetensors"),
    )

    # Decoded media as SGLang delivers them: [1, 3, F, H, W] in [0, 1] to
    # uint8 [F, H, W, 3]; audio [1, 2, N] to [2, N].
    video = torch.from_numpy(
        np.stack(_sample_to_uint8_frames(output.output[0]))
    )
    audio = output.audio[0].float().cpu()
    sample_rate = int(output.audio_sample_rate)
    save_file(
        {"video": video.contiguous(), "audio": audio.contiguous()},
        str(run_dir / "decoded.safetensors"),
    )
    write_mp4(run_dir / "output.mp4", video.numpy(), audio.numpy(), sample_rate)

    (run_dir / "request.json").write_text(
        json.dumps(
            workload.request_body(inputs, args.seed),
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    transformer = pipeline.get_module("transformer")
    transformer = getattr(transformer, "model", transformer)
    dit_backends = _dit_attention_backends(transformer)
    keyframe_anchors = []
    if ctx.keyframe is not None:
        keyframe_anchors = [
            "first" if index == 0 else "last"
            for index in ctx.keyframe["semantic_frame_indices"]
        ]
    partition = "Ref2VA" if workload.task == "ref2va" else "FL2VA"
    config = server_args.pipeline_config
    metadata = {
        "impl": args.impl,
        "workload": workload.name,
        "task": workload.task,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "denoising_steps": num_steps,
        "attention": {
            "dit": {
                "sdpa_kernel": args.attention,
                "sglang_backend": dit_backends,
                "kernels_first_forward": recorder.attention_kernels,
            },
            "text_encoder_and_vaes": {
                name: _component_attention_backends(pipeline.get_module(name))
                for name in COMPONENTS[1:]
            },
            "text_encoder_and_vaes_dispatch": "SGLang default selection",
        },
        "replayed_noise": str(args.replay_noise) if args.replay_noise else None,
        "teacher": (
            {
                "run": str(args.teacher),
                "prediction_rel_l2": recorder.teacher_deviation,
            }
            if args.teacher
            else None
        ),
        "noise_draw_order": draw_order,
        "generator": (
            "SGLang: torch.Generator('cpu').manual_seed(seed) for every draw, "
            "float32"
        ),
        "precision": {
            "configured": {
                "dit": config.dit_precision,
                "text_encoder": list(config.text_encoder_precisions),
                "video_vae": config.vae_precision,
                "video_vae_decode_autocast": str(
                    resolve_decode_precision(server_args, "video_vae")
                ),
                "audio_vae": config.audio_vae_precision,
            },
            # As loaded; the video decode later adds FP16 copies of the ViT
            # decoder weights for its autocast.
            "loaded_elements": loaded_dtypes,
            "dit_fp32_parameters": [
                name
                for name, parameter in transformer.named_parameters()
                if parameter.dtype == torch.float32
            ],
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "partition": partition,
            "revision": _checkpoint_revision(
                args.checkpoint, f"{partition}/transformer"
            ),
        },
        "geometry": {
            "height": int(shape["height"]),
            "width": int(shape["width"]),
            "num_frames": int(shape["frame_count"]),
            "num_latent_frames": latent_t,
            "latent_height": latent_h,
            "latent_width": latent_w,
            "num_audio_latents": audio_t,
            "num_text_rows": len(token_ids),
            "num_condition_video_rows": int(positive.video_target_start),
            "num_condition_audio_rows": int(positive.audio_target_start),
            "sequence_rows": used,
            "keyframe_anchors": keyframe_anchors,
            "audio_samples": int(audio.shape[-1]),
            "audio_sample_rate": sample_rate,
        },
        "sglang": {
            "model_variant": server_args.model_variant,
            "performance_mode": server_args.performance_mode,
            "component_attention_backends": dict(
                server_args.component_attention_backends or {}
            ),
            "packed_rows": int(packed["seq_len"]),
            "cu_seqlens": [
                int(value) for value in packed["cu_seqlens"].tolist()
            ],
        },
        "timings_seconds": {
            **timings,
            "text_encoding": recorder.text["seconds"],
            "denoising": sum(recorder.step_seconds),
            "per_step_median": float(np.median(recorder.step_seconds)),
        },
        "versions": _versions(),
        "tf32": {
            "matmul": torch.backends.cuda.matmul.allow_tf32,
            "cudnn": torch.backends.cudnn.allow_tf32,
        },
        "command": sys.argv,
    }
    if args.replay_noise is not None:
        metadata["replay_source_comparison"] = _compare_with_source(
            run_dir, args.replay_noise.parent
        )
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def run(args: argparse.Namespace) -> None:
    workload = WORKLOADS[args.workload]
    run_dir = (
        args.output
        / "reference"
        / args.impl
        / workload.name
        / f"seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    inputs = (args.output / "inputs").resolve()

    replay = None
    if args.replay_noise is not None:
        replay = load_file(str(args.replay_noise))
    teacher = None
    if args.teacher is not None:
        teacher_metadata = json.loads(
            (args.teacher / "metadata.json").read_text()
        )
        if (teacher_metadata["workload"], teacher_metadata["seed"]) != (
            workload.name,
            args.seed,
        ):
            raise SystemExit(
                f"the teacher run is {teacher_metadata['workload']} seed "
                f"{teacher_metadata['seed']}, not {workload.name} seed "
                f"{args.seed}"
            )
        teacher = load_file(str(args.teacher / "trajectory.safetensors"))

    with tempfile.TemporaryDirectory(prefix="sglang_reference_") as scratch:
        scratch = Path(scratch)
        started = time.perf_counter()
        server_args, pipeline = load_pipeline(
            args.checkpoint, workload.task, scratch
        )
        loaded = time.perf_counter()
        print(f"loaded in {loaded - started:.1f}s", flush=True)
        loaded_dtypes = {
            name: _parameter_dtypes(pipeline.get_module(name))
            for name in COMPONENTS
        }

        req = build_request(
            server_args,
            workload,
            inputs,
            args.seed,
            args.num_inference_steps,
            scratch,
        )
        plan = _h3("resolved_plan").minimax_h3_plan_from_batch(req)
        shape = dict(plan.shape)
        recorder = Recorder(args.attention, replay=replay, teacher=teacher)
        with recorder.patch(pipeline):
            output = pipeline.forward(req, server_args)
        finished = _synchronized_time()
        if getattr(output, "error", None):
            raise RuntimeError(f"SGLang failed the request: {output.error}")

        transformer = pipeline.get_module("transformer")
        backends = _dit_attention_backends(
            getattr(transformer, "model", transformer)
        )
        if set(backends) != {f"{DIT_ATTENTION_BACKEND}/SDPAImpl"}:
            raise RuntimeError(f"the DiT attention resolved {backends}")

        write_artifacts(
            run_dir,
            args,
            workload,
            inputs,
            shape,
            output,
            recorder,
            {"load": loaded - started, "call_total": finished - loaded},
            loaded_dtypes,
            server_args,
            pipeline,
        )
    print(f"wrote {run_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workload", required=True, choices=sorted(WORKLOADS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="artifacts/minimax_h3 root holding inputs/ and reference/",
    )
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument(
        "--attention", choices=sorted(SDPA_KERNELS), default="cudnn"
    )
    parser.add_argument(
        "--impl",
        default=None,
        help="reference/<impl> directory name; defaults to 'sglang' for cuDNN, "
        "'sglang_sdpa_<kernel>' otherwise and 'sglang_sdpa_<kernel>_teacher' "
        "with --teacher",
    )
    parser.add_argument(
        "--replay-noise",
        type=Path,
        default=None,
        help="noise.safetensors of an earlier run whose draws replace SGLang's",
    )
    parser.add_argument(
        "--teacher",
        type=Path,
        default=None,
        help="canonical run directory whose samples every step continues from; "
        "its noise is replayed",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="sigma points including the terminal 0 (50 = 49 DiT forwards)",
    )
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be a non-negative integer")
    if args.teacher is not None:
        teacher_noise = args.teacher / "noise.safetensors"
        if args.replay_noise is None:
            args.replay_noise = teacher_noise
        elif args.replay_noise.resolve() != teacher_noise.resolve():
            parser.error("a teacher-forced run replays the teacher's noise")
    if args.impl is None:
        if args.teacher is not None:
            args.impl = f"sglang_sdpa_{args.attention}_teacher"
        elif args.attention == "cudnn":
            args.impl = "sglang"
        else:
            args.impl = f"sglang_sdpa_{args.attention}"

    _prepare_environment()
    run(args)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
