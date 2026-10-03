"""Generate MiniMax-H3 reference artifacts for the FastVideo checkpoints.

Runs one workload of ``workloads.py`` through FastVideo's MiniMax-H3 pipeline
(``refs/FastVideo`` branch ``feat/fasth3-omniref-pdd-inference``) for either
FastVideo export:

* ``fasth3_v2``: ``FastVideo-FastH3-8-Step-V2`` t2va, the DMD student with
  VSA sparsity 0.8 on 64-token tiles (the sm100a kernel), eight forwards;
* ``omniref``: ``FastH3-OmniRef-DMPDD8-step1900`` ref2va, the PDD student with
  VSA sparsity 0.9 on 128-token tiles and the ``p2_multi_region`` reference
  policy, eight fused-block forwards, other components from the base
  checkpoint.

The run uses the settings of FastVideo's own example scripts
(``examples/inference/basic/basic_fasth3_8step.py`` with ``--profile strict
--vsa-kernel sm100a --no-inference-torch-compile --no-compile-vae`` and
``basic_fasth3_omniref_pdd.py``) on one GPU, and writes the same artifact set
as ``diffusers_reference.py`` into
``<output>/reference/<impl>/<workload>[_<H>x<W>]/seed<seed>/``.

FastVideo executes the pipeline in spawned worker processes, which re-import
this module; when ``MINIMAX_H3_DUMP_DIR`` is set the module installs recording
seams in the worker: every request-generator draw (``randn_tensor`` of the
latent-preparation and packing modules), the presentation and conditioner
output, the token refiner output, the clean condition latents, the packed
layout, the scheduler steps (per-step velocity and sample of the generated
rows) and the decoded media. Reference videos are handed to FastVideo as
frames decoded with the pinned FFmpeg build at 24 fps on their own canvas
(the reference implementation's recipe, see ``diffusers_reference.py``),
so both references condition on identical pixels; images and audio are
decoded by FastVideo itself.

Run with the FastVideo environment's interpreter, e.g. ``CUDA_VISIBLE_DEVICES=3
/workspace/envs/minimax_h3/fastvideo/bin/python
tools/minimax_h3/fastvideo_reference.py --checkpoint omniref --workload
ref2va_video_audio_5s --height 480 --width 832 --seed 42 --output
artifacts/minimax_h3``.
"""

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from workloads import WORKLOADS  # noqa: E402

FASTVIDEO = Path("/workspace/envs/minimax_h3/src/FastVideo")
BASE = Path("/workspace/models/MiniMax-H3")
CHECKPOINTS = {
    "fasth3_v2": Path("/workspace/models/FastVideo-FastH3-8-Step-V2"),
    "omniref": Path("/workspace/models/FastH3-OmniRef-DMPDD8-step1900"),
}
DUMP_ENV = "MINIMAX_H3_DUMP_DIR"
REPLAY_ENV = "MINIMAX_H3_REPLAY_NOISE"
KERNEL_ENV = "MINIMAX_H3_VSA_KERNEL"
REPLAY_TEXT_ENV = "MINIMAX_H3_REPLAY_TEXT"
TEACHER_ENV = "MINIMAX_H3_TEACHER_TRAJECTORY"

# Block-sparse kernels that evaluate FastVideo's VSA-H3 tile-128 forward. The
# sm_100a CUDA kernel is FastVideo's own route; ``triton`` is fastvideo-kernel's
# ``block_sparse_attn_128`` Triton route, which expands the same 128-token
# block map onto its 64-token Triton kernel. Both evaluate the same mask, so
# the pair measures the reference's kernel-to-kernel variation.
VSA_KERNELS = ("sm100a", "triton")


def _save(path: Path, tensors: dict) -> None:
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in tensors.items()
        },
        str(path),
    )


def _route_tile128_to_triton() -> None:
    """Evaluate VSA-H3's tile-128 forward with fastvideo-kernel's Triton route.

    FastVideo calls the sm_100a kernel with the index form of its block map
    (``map_to_index``). The replacement rebuilds the same boolean map from
    those indices and hands it, with the same valid sizes, to
    ``block_sparse_attn_128``, which runs the 64-token Triton kernel on the
    map expanded two by two. Selection, compression and the gate stay
    FastVideo's own code.
    """
    import types

    from fastvideo.attention.backends import video_sparse_attn_h3 as backend
    from fastvideo_kernel import block_sparse_attn_256 as routes

    native = backend._sm100a

    def block_sparse_attn_sm100a(
        q, k, v, q2k_idx, q2k_num, variable_block_sizes, need_lse=False
    ):
        if need_lse:
            raise ValueError("the Triton tile-128 route returns no LSE")
        batch, heads, query_tiles, width = q2k_idx.shape
        key_tiles = variable_block_sizes.numel()
        listed = torch.arange(width, device=q2k_idx.device) < q2k_num[..., None]
        # Unlisted slots scatter into one spare column that is dropped.
        columns = torch.where(listed, q2k_idx.long(), key_tiles)
        mask = torch.zeros(
            (batch, heads, query_tiles, key_tiles + 1),
            dtype=torch.bool,
            device=q.device,
        )
        mask.scatter_(-1, columns, True)
        os.environ["FASTVIDEO_VSA_TRITON"] = "1"
        return routes.block_sparse_attn_128(
            q, k, v, mask[..., :key_tiles], variable_block_sizes
        )

    backend._sm100a = types.SimpleNamespace(
        is_supported=native.is_supported,
        block_sparse_attn_sm100a=block_sparse_attn_sm100a,
    )


def _install_worker_seams(dump_dir: Path) -> None:
    """Patch FastVideo's MiniMax-H3 stages to record the reference tensors.

    Runs in every process that imports this module with the dump variable
    set; only the worker executes the stages, so only it records.
    """
    from fastvideo.models.dits import minimax_h3 as dit
    from fastvideo.models.schedulers import scheduling_minimax_h3 as sched
    from fastvideo.pipelines.basic.minimax_h3 import packing
    from fastvideo.pipelines.basic.minimax_h3.stages import (
        minimax_h3_conditioning as conditioning,
    )
    from fastvideo.pipelines.basic.minimax_h3.stages import (
        minimax_h3_decoding as decoding,
    )
    from fastvideo.pipelines.basic.minimax_h3.stages import (
        minimax_h3_latent_preparation as preparation,
    )

    replay = None
    if os.environ.get(REPLAY_ENV):
        stored = load_file(os.environ[REPLAY_ENV])
        order = json.loads(
            (Path(os.environ[REPLAY_ENV]).parent / "metadata.json").read_text()
        )["noise_draw_order"]
        replay = [stored[name] for name in order]

    state = {
        "draws": [],
        "steps": {"video": [], "audio": []},
        "refined": None,
        "conditions": {},
        "text": None,
        "step_times": [],
    }

    def recorded_randn(original):
        def randn_tensor(
            shape, generator=None, device=None, dtype=None, layout=None
        ):
            index = len(state["draws"])
            if replay is not None:
                value = replay[index].to(device=device, dtype=dtype)
                if tuple(value.shape) != tuple(shape):
                    raise ValueError(f"replayed draw {index} shape mismatch")
            else:
                value = original(
                    shape,
                    generator=generator,
                    device=device,
                    dtype=dtype,
                    layout=layout,
                )
            state["draws"].append(value.detach().cpu().clone())
            return value

        return randn_tensor

    preparation.randn_tensor = recorded_randn(preparation.randn_tensor)
    packing.randn_tensor = recorded_randn(packing.randn_tensor)

    if os.environ.get(KERNEL_ENV, "sm100a") == "triton":
        _route_tile128_to_triton()

    stage = conditioning.MiniMaxH3ConditioningStage
    original_encode = stage._encode_tokens
    original_conditioning = stage.forward

    recorded_text = None
    if os.environ.get(REPLAY_TEXT_ENV):
        text_path = Path(os.environ[REPLAY_TEXT_ENV])
        recorded_text = (
            load_file(str(text_path))["qwen_hidden_states_50"],
            json.loads((text_path.parent / "presentation.json").read_text())[
                "token_ids"
            ],
        )

        def conditioning_forward(self, batch, fastvideo_args):
            # The recorded encoder output replaces the encoder call, so the
            # offloaded encoder never moves to the device: a stand-in without
            # parameters takes its place for the duration of the stage.
            encoder = self.conditioner
            self.conditioner = torch.nn.Module()
            try:
                return original_conditioning(self, batch, fastvideo_args)
            finally:
                self.conditioner = encoder

        stage.forward = conditioning_forward

    def encode_tokens(self, token_ids, token_tags, device, **vision_inputs):
        start = time.perf_counter()
        if recorded_text is None:
            embeds, tags = original_encode(
                self, token_ids, token_tags, device, **vision_inputs
            )
        else:
            embeds, recorded_ids = recorded_text
            if list(token_ids) != recorded_ids:
                raise ValueError(
                    "the replayed text encoding presents other token ids"
                )
            embeds = embeds.to(device=device)
            tags = torch.tensor(token_tags, dtype=torch.long)
        torch.cuda.synchronize()
        state["text"] = {
            "token_ids": list(token_ids),
            "tags": [int(tag) for tag in token_tags],
            "vision": {
                name: value.detach().cpu().clone()
                for name, value in vision_inputs.items()
                if value is not None
            },
            "embeds": embeds.detach().cpu().clone(),
            "seconds": time.perf_counter() - start,
        }
        return embeds, tags

    stage._encode_tokens = encode_tokens

    original_refiner = dit.MiniMaxH3TokenRefiner.forward

    def refiner_forward(self, hidden_states, *args, **kwargs):
        output = original_refiner(self, hidden_states, *args, **kwargs)
        if state["refined"] is None:
            state["refined"] = output.detach().cpu().clone()
        return output

    dit.MiniMaxH3TokenRefiner.forward = refiner_forward

    latents_stage = preparation.MiniMaxH3LatentPreparationStage
    original_keyframe = latents_stage._encode_keyframe_latents
    original_visual = latents_stage._encode_visual_rows
    original_audio = latents_stage._encode_audio_rows
    original_prepare = latents_stage.forward

    def encode_keyframe(self, image, device):
        latents = original_keyframe(self, image, device)
        index = len(
            [k for k in state["conditions"] if k.startswith("video_condition.")]
        )
        state["conditions"][f"video_condition.{index}"] = latents.clone()
        return latents

    def encode_visual(self, references, device, fastvideo_args):
        rows = original_visual(self, references, device, fastvideo_args)
        visual = [r for r in references if r.media_type != "audio"]
        for index, (reference, block) in enumerate(
            zip(visual, rows, strict=True)
        ):
            shape = (
                1,
                24,
                reference.num_latent_frames,
                reference.latent_height,
                reference.latent_width,
            )
            state["conditions"][f"video_condition_rows.{index}"] = (
                block.detach().cpu().clone()
            )
            state["conditions"][f"video_condition_shape.{index}"] = (
                torch.tensor(shape)
            )
        return rows

    def encode_audio(self, references, device):
        rows = original_audio(self, references, device)
        for index, block in enumerate(rows):
            state["conditions"][f"audio_condition.{index}"] = (
                block.detach().cpu().clone()
            )
        return rows

    def prepare(self, batch, fastvideo_args):
        batch = original_prepare(self, batch, fastvideo_args)
        layout = batch.extra[preparation.MINIMAX_H3_LAYOUT_KEY]
        state["layout"] = layout
        if layout.num_condition_video_rows:
            state["conditions"]["condition_rows"] = (
                batch.latents[: layout.num_condition_video_rows]
                .detach()
                .cpu()
                .clone()
            )
        state["prepared_at"] = time.perf_counter()
        return batch

    latents_stage._encode_keyframe_latents = encode_keyframe
    latents_stage._encode_visual_rows = encode_visual
    latents_stage._encode_audio_rows = encode_audio
    latents_stage.forward = prepare

    original_step = sched.MiniMaxH3Scheduler.step
    # Teacher forcing: every step continues from the recorded run's sample of
    # that step, so each prediction is made on the recorded run's inputs.
    teacher = (
        load_file(os.environ[TEACHER_ENV])
        if os.environ.get(TEACHER_ENV)
        else None
    )

    def step(self, model_output, timestep, sample, return_dict=True):
        stream = "video" if model_output.shape[-1] == 96 else "audio"
        if not state["steps"][stream]:
            state.setdefault("initial", {})[stream] = (
                sample.detach().float().cpu().clone()
            )
            state.setdefault("sigmas", {})[stream] = (
                self.sigmas.detach().float().cpu().clone()
            )
        result = original_step(
            self, model_output, timestep, sample, return_dict=return_dict
        )
        updated = result[0] if not return_dict else result.prev_sample
        state["steps"][stream].append(
            {
                "timestep": float(timestep),
                "prediction": model_output.detach().float().cpu().clone(),
                "sample": updated.detach().float().cpu().clone(),
            }
        )
        if teacher is not None:
            # The recorded sample is this run's own update; the next step
            # reads the recorded run's.
            index = len(state["steps"][stream]) - 1
            updated.copy_(teacher[f"{stream}_samples"][index])
        if stream == "audio":
            torch.cuda.synchronize()
            state["step_times"].append(time.perf_counter())
            print(
                f"step {len(state['steps']['audio'])} t={float(timestep):.6f}",
                flush=True,
            )
        return result

    sched.MiniMaxH3Scheduler.step = step

    video_stage = decoding.MiniMaxH3VideoDecodingStage
    audio_stage = decoding.MiniMaxH3AudioDecodingStage
    original_video = video_stage.forward
    original_audio_decode = audio_stage.forward

    def decode_video(self, batch, fastvideo_args):
        layout = batch.extra[preparation.MINIMAX_H3_LAYOUT_KEY]
        rows = batch.latents[layout.num_condition_video_rows :]
        state["final_video_rows"] = rows.detach().float().cpu().clone()
        state["final_audio_rows"] = (
            batch.audio_latents[layout.num_condition_audio_rows :]
            .detach()
            .float()
            .cpu()
            .clone()
        )
        state["denoised_at"] = time.perf_counter()
        batch = original_video(self, batch, fastvideo_args)
        state["pixels"] = batch.output.detach().float().cpu().clone()
        return batch

    def decode_audio(self, batch, fastvideo_args):
        batch = original_audio_decode(self, batch, fastvideo_args)
        state["audio"] = batch.extra["audio"].detach().float().cpu().clone()
        state["audio_sample_rate"] = int(batch.extra["audio_sample_rate"])
        _write_worker_artifacts(dump_dir, state)
        return batch

    video_stage.forward = decode_video
    audio_stage.forward = decode_audio


def _write_worker_artifacts(dump_dir: Path, state: dict) -> None:
    """Write the recorded tensors in the diffusers reference layout."""
    layout = state["layout"]
    patch = 2
    channels = 24
    frames = layout.num_video_latent_frames
    height = layout.latent_height
    width = layout.latent_width

    visual_conditions = sorted(
        k
        for k in state["conditions"]
        if k.startswith("video_condition.")
        or k.startswith("video_condition_rows.")
    )
    draw_names = []
    num_visual = len(
        [k for k in visual_conditions if k.startswith("video_condition_rows.")]
    ) or len([k for k in visual_conditions if k.startswith("video_condition.")])
    draw_names += [f"condition_noise.{k}" for k in range(num_visual)]
    draw_names += ["video_noise", "audio_noise"]
    if len(draw_names) != len(state["draws"]):
        raise RuntimeError(
            f"expected {len(draw_names)} draws, recorded {len(state['draws'])}"
        )
    _save(
        dump_dir / "noise.safetensors",
        dict(zip(draw_names, state["draws"], strict=True)),
    )

    text = state["text"]
    tensors = {
        "qwen_hidden_states_50": text["embeds"],
        "refined_text": state["refined"],
    }
    for name, value in text["vision"].items():
        tensors[f"qwen_{name}"] = value
    _save(dump_dir / "text.safetensors", tensors)
    presentation = {
        "token_ids": text["token_ids"],
        "tags": text["tags"],
        "num_text_rows": len(text["token_ids"]),
        "image_grid_thw": text["vision"]
        .get("image_grid_thw", torch.empty(0, 3))
        .tolist(),
        "video_grid_thw": text["vision"]
        .get("video_grid_thw", torch.empty(0, 3))
        .tolist(),
    }
    (dump_dir / "presentation.json").write_text(json.dumps(presentation) + "\n")

    conditions = dict(state["conditions"])
    for key in [k for k in conditions if k.startswith("video_condition_rows.")]:
        # Unpatchify reference rows to the diffusers [1, 24, T, H, W] layout.
        index = key.rsplit(".", 1)[1]
        _, channels, frames_ref, height_ref, width_ref = conditions[
            f"video_condition_shape.{index}"
        ].tolist()
        block = conditions[key].reshape(
            1,
            frames_ref,
            height_ref // patch,
            width_ref // patch,
            channels,
            1,
            patch,
            patch,
        )
        conditions[f"video_condition.{index}"] = block.permute(
            0, 4, 1, 5, 2, 6, 3, 7
        ).reshape(1, channels, frames_ref, height_ref, width_ref)
    if conditions:
        _save(dump_dir / "conditions.safetensors", conditions)

    _save(
        dump_dir / "layout.safetensors",
        {
            "position_ids": layout.position_ids,
            "token_tags": layout.token_tags,
            "video_indices": layout.video_indices,
            "audio_indices": layout.audio_indices,
            "text_indices": layout.text_indices,
            "sigmas": state["sigmas"]["video"],
            "audio_sigmas": state["sigmas"]["audio"],
            "timesteps": torch.tensor(
                [s["timestep"] for s in state["steps"]["video"]]
            ),
            "audio_timesteps": torch.tensor(
                [s["timestep"] for s in state["steps"]["audio"]]
            ),
        },
    )
    _save(
        dump_dir / "trajectory.safetensors",
        {
            "video_predictions": torch.stack(
                [s["prediction"] for s in state["steps"]["video"]]
            ),
            "audio_predictions": torch.stack(
                [s["prediction"] for s in state["steps"]["audio"]]
            ),
            "video_samples": torch.stack(
                [s["sample"] for s in state["steps"]["video"]]
            ),
            "audio_samples": torch.stack(
                [s["sample"] for s in state["steps"]["audio"]]
            ),
            "video_initial": state["initial"]["video"],
            "audio_initial": state["initial"]["audio"],
        },
    )

    # Unpack the generated rows like the diffusers after-denoise step.
    rows = state["final_video_rows"].reshape(
        1, frames, height // patch, width // patch, channels, 1, patch, patch
    )
    video_latents = rows.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
        1, channels, frames, height, width
    )
    audio_rows = state["final_audio_rows"]
    audio_latents = audio_rows.reshape(
        2, layout.num_audio_latents, audio_rows.shape[-1]
    ).permute(0, 2, 1)
    _save(
        dump_dir / "final.safetensors",
        {
            "video_latents": video_latents,
            "audio_latents": audio_latents,
        },
    )

    pixels = state["pixels"][0].clamp(0, 1)
    video = (pixels * 255.0).round().to(torch.uint8).permute(1, 2, 3, 0)
    audio = state["audio"]
    if audio.ndim == 2 and audio.shape[-1] == 2:
        audio = audio.transpose(0, 1)
    _save(
        dump_dir / "decoded.safetensors",
        {
            "video": video,
            "audio": audio.reshape(2, -1),
        },
    )
    times = state["step_times"]
    (dump_dir / "worker_timings.json").write_text(
        json.dumps(
            {
                "text_encoding": text["seconds"],
                "denoising": times[-1] - state["prepared_at"]
                if times
                else None,
                "per_step": [
                    b - a
                    for a, b in zip(
                        [state["prepared_at"], *times[:-1]], times, strict=True
                    )
                ],
                "num_condition_video_rows": layout.num_condition_video_rows,
                "num_condition_audio_rows": layout.num_condition_audio_rows,
                "sequence_rows": layout.sequence_length,
                "num_latent_frames": frames,
                "latent_height": height,
                "latent_width": width,
                "num_audio_latents": layout.num_audio_latents,
                "audio_sample_rate": state["audio_sample_rate"],
                "draw_order": draw_names,
            }
        )
        + "\n"
    )


if os.environ.get(DUMP_ENV) and __name__ != "__main__":
    _install_worker_seams(Path(os.environ[DUMP_ENV]))


def _ffmpeg_reference_frames(path: Path, num_frames: int) -> np.ndarray:
    from diffusers_reference import decode_reference_video

    return decode_reference_video(path, num_frames)


def _import_example(name: str):
    sys.path.insert(0, str(FASTVIDEO / "examples" / "inference" / "basic"))
    return __import__(name)


def _fasth3_generator(num_frames: int):
    """FastVideo's FastH3-8-Step-V2 configuration (strict profile, one GPU)."""
    example = _import_example("basic_fasth3_8step")
    basic = example.basic_fasth3
    args = example.parse_args(
        [
            "--model-path",
            str(CHECKPOINTS["fasth3_v2"]),
            # The example parser requires a prompt; the request carries the
            # workload's own.
            "--prompt",
            "unused",
            "--num-gpus",
            "1",
            "--vsa-kernel",
            "sm100a",
            "--profile",
            "strict",
            "--no-inference-torch-compile",
            "--no-compile-vae",
            "--no-parallel-vae",
            "--num-frames",
            str(num_frames),
        ]
    )
    environment = basic.configure_environment(args)
    basic.validate_profile_dependencies(args)
    return basic, args, environment


def run(args: argparse.Namespace) -> None:
    workload = WORKLOADS[args.workload]
    inputs = args.output / "inputs"
    suffix = f"_{args.height}x{args.width}" if args.height is not None else ""
    run_dir = (
        args.output
        / "reference"
        / args.impl
        / f"{workload.name}{suffix}"
        / f"seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ[DUMP_ENV] = str(run_dir)
    if args.replay_noise is not None:
        os.environ[REPLAY_ENV] = str(args.replay_noise.resolve())
    os.environ[KERNEL_ENV] = args.vsa_kernel
    if args.replay_text is not None:
        os.environ[REPLAY_TEXT_ENV] = str(args.replay_text.resolve())
    if args.teacher_trajectory is not None:
        os.environ[TEACHER_ENV] = str(args.teacher_trajectory.resolve())

    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
        align_num_frames,
        resolve_canvas_size,
    )
    from fastvideo import VideoGenerator
    from fastvideo.api import (
        GenerationRequest,
        InputConfig,
        OutputConfig,
        SamplingConfig,
    )

    requested_frames = round(workload.duration_seconds * 24)
    num_frames = align_num_frames(requested_frames, 17, 5)
    if args.height is not None:
        height, width = args.height, args.width
    else:
        aspect = (
            (16, 9)
            if workload.aspect_ratio == "auto"
            else tuple(float(v) for v in workload.aspect_ratio.split(":"))
        )
        height, width = resolve_canvas_size(*aspect, 32, 768, 768 * 1344)

    started = time.perf_counter()
    environment = {}
    if args.checkpoint == "fasth3_v2":
        if workload.task != "t2va":
            raise SystemExit("FastH3-8-Step-V2 only serves t2va")
        basic, example_args, environment = _fasth3_generator(num_frames)
        generator = VideoGenerator.from_config(
            basic.build_generator_config(example_args)
        )
        steps = example_args.steps
        inputs_config = None
        contract = json.loads(
            (CHECKPOINTS["fasth3_v2"] / "fastvideo_inference.json").read_text()
        )
    else:
        if workload.task != "ref2va":
            raise SystemExit("the OmniRef export only serves ref2va")
        example = _import_example("basic_fasth3_omniref_pdd")
        from fastvideo.pipelines.basic.minimax_h3 import MiniMaxH3Reference
        from fastvideo.pipelines.basic.minimax_h3.reference import (
            decode_reference_audio,
        )

        example_args = example.parse_args(
            [
                "--model-path",
                str(CHECKPOINTS["omniref"]),
                "--base-model-path",
                str(BASE),
                "--composed-dir",
                str(args.output / "work" / "fasth3_omniref"),
                "--prompt",
                "unused",
                "--image",
                "unused",
            ]
        )
        model_dir, contract = example.resolve_model(example_args)
        example.validate_attention_runtime(contract, 1)
        references = []
        for condition in workload.conditions:
            path = inputs / condition.media
            if condition.role != "reference":
                raise SystemExit("OmniRef reference runs take references only")
            if condition.kind == "video":
                frames = _ffmpeg_reference_frames(path, num_frames)
                try:
                    soundtrack, rate = decode_reference_audio(path)
                except ValueError:
                    soundtrack, rate = None, None
                references.append(
                    MiniMaxH3Reference(
                        source=frames,
                        media_type="video",
                        fps=24.0,
                        soundtrack=soundtrack,
                        sample_rate=rate,
                    )
                )
            else:
                references.append(
                    MiniMaxH3Reference(
                        source=str(path), media_type=condition.kind
                    )
                )
        config = example.build_generator_config(model_dir, contract, 1)
        # Layerwise offload streams each DiT block's unchanged weights to the
        # GPU before it runs; it bounds device memory, not the arithmetic.
        config.engine.offload.dit_layerwise = args.dit_layerwise_offload
        generator = VideoGenerator.from_config(config)
        steps = contract["num_inference_steps"]
        inputs_config = InputConfig(references=references)
    loaded = time.perf_counter()

    try:
        request_kwargs = {}
        if inputs_config is not None:
            request_kwargs["inputs"] = inputs_config
        result = generator.generate(
            GenerationRequest(
                prompt=workload.resolve_prompt(inputs),
                negative_prompt="",
                sampling=SamplingConfig(
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    fps=24,
                    num_inference_steps=steps,
                    guidance_scale=1.0,
                    batch_cfg=False,
                    seed=args.seed,
                ),
                output=OutputConfig(
                    output_path=str(run_dir / "output.mp4"),
                    save_video=True,
                    return_frames=False,
                ),
                **request_kwargs,
            )
        )
    finally:
        generator.shutdown()
    finished = time.perf_counter()

    worker = json.loads((run_dir / "worker_timings.json").read_text())
    (run_dir / "request.json").write_text(
        json.dumps(
            workload.request_body(inputs, args.seed),
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    import fastvideo

    metadata = {
        "impl": args.impl,
        "workload": workload.name,
        "task": workload.task,
        "seed": args.seed,
        "checkpoint": {
            "name": args.checkpoint,
            "path": str(CHECKPOINTS[args.checkpoint]),
            "contract": contract,
        },
        "num_inference_steps": steps,
        "denoising_steps": len(worker["per_step"]),
        "noise_draw_order": worker["draw_order"],
        "replayed_noise": (
            str(args.replay_noise) if args.replay_noise else None
        ),
        "generator": "torch.Generator('cpu'), float32 draws",
        "geometry": {
            "height": height,
            "width": width,
            "num_frames": num_frames,
            **{
                k: worker[k]
                for k in (
                    "num_latent_frames",
                    "latent_height",
                    "latent_width",
                    "num_audio_latents",
                    "num_condition_video_rows",
                    "num_condition_audio_rows",
                    "sequence_rows",
                    "audio_sample_rate",
                )
            },
        },
        "fastvideo_environment": environment,
        "vsa_kernel": args.vsa_kernel,
        "dit_layerwise_offload": args.dit_layerwise_offload,
        "replayed_text": str(args.replay_text) if args.replay_text else None,
        "teacher_trajectory": (
            str(args.teacher_trajectory) if args.teacher_trajectory else None
        ),
        "timings_seconds": {
            "load": loaded - started,
            "text_encoding": worker["text_encoding"],
            "denoising": worker["denoising"],
            "per_step_median": float(np.median(worker["per_step"])),
            "call_total": finished - loaded,
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "fastvideo": getattr(fastvideo, "__version__", "unknown"),
            "fastvideo_source": str(FASTVIDEO),
            "host": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tools_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=TOOLS,
                capture_output=True,
                text=True,
            ).stdout.strip(),
        },
        "result_video_path": getattr(result, "video_path", None),
        "command": sys.argv,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"wrote {run_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", choices=sorted(CHECKPOINTS), required=True
    )
    parser.add_argument("--workload", choices=sorted(WORKLOADS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="canvas override (OmniRef training shape 480)",
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--impl", default=None)
    parser.add_argument("--replay-noise", type=Path, default=None)
    parser.add_argument(
        "--vsa-kernel",
        choices=VSA_KERNELS,
        default="sm100a",
        help="block-sparse kernel of the OmniRef tile-128 forward",
    )
    parser.add_argument(
        "--replay-text",
        type=Path,
        default=None,
        help="text.safetensors whose Qwen hidden states replace the encoder",
    )
    parser.add_argument(
        "--teacher-trajectory",
        type=Path,
        default=None,
        help="trajectory.safetensors whose samples every step continues from",
    )
    parser.add_argument(
        "--dit-layerwise-offload",
        action="store_true",
        help="keep the OmniRef DiT on the host and stream it layer by layer",
    )
    args = parser.parse_args()
    if args.checkpoint != "omniref" and (
        args.vsa_kernel != "sm100a" or args.dit_layerwise_offload
    ):
        parser.error(
            "--vsa-kernel and --dit-layerwise-offload apply to the OmniRef run"
        )
    if (args.height is None) != (args.width is None):
        parser.error("--height and --width go together")
    if args.impl is None:
        args.impl = f"fastvideo_{args.checkpoint}"
    run(args)


if __name__ == "__main__":
    main()
