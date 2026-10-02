"""Generate MiniMax-H3 base-checkpoint reference artifacts with diffusers.

Runs one workload of ``workloads.py`` through the diffusers MiniMax-H3
modular pipeline at the checkpoint's native precision (DiT BF16 with its FP32
patch, time and output projections, Qwen3-VL BF16, both VAEs FP32 with the
video decode under FP16 autocast) on one GPU, and records the tensors the
UniServe parity tests compare against. Every artifact is written to
``<output>/reference/<impl>/<workload>/seed<seed>/``:

* ``request.json``: the serving request body of the run (``file://`` URIs
  point into the inputs directory).
* ``presentation.json``: the Qwen presentation token ids, their AdaLN tags,
  and the Qwen vision grids.
* ``text.safetensors``: ``qwen_hidden_states_50`` (``[1, L, 5120]``),
  ``refined_text`` (the token refiner output, ``[1, L, 5376]``) and the Qwen
  vision inputs (``pixel_values``/``image_grid_thw``,
  ``pixel_values_videos``/``video_grid_thw``).
* ``conditions.safetensors``: ``video_condition.<k>`` (the normalized clean
  VAE latents of keyframe/image/video condition ``k``),
  ``audio_condition.<k>`` (the normalized channel-major audio reference rows)
  and ``condition_rows`` (the noised, patchified anchors).
* ``noise.safetensors``: every draw of the request generator in draw order:
  ``condition_noise.<k>`` (shape of ``video_condition.<k>``), ``video_noise``
  (``[1, 24, T, H/16, W/16]``) and ``audio_noise`` (channel-major rows
  ``[2 Ta, 32]``). Passing the file to ``--replay-noise`` feeds exactly
  these draws to another run.
* ``layout.safetensors``: the packed layout (``position_ids`` FP64,
  ``token_tags``, ``video_indices``, ``audio_indices``, ``text_indices``) and
  the schedules (``sigmas``, ``audio_sigmas``, ``timesteps``,
  ``audio_timesteps``).
* ``trajectory.safetensors``: per denoising step ``i`` (one DiT forward)
  ``video_predictions[i]`` / ``audio_predictions[i]`` (the velocity of the
  generated rows) and ``video_samples[i]`` / ``audio_samples[i]`` (the
  generated rows after the Euler update), all FP32.
* ``final.safetensors``: ``video_latents`` ``[1, 24, T, H/16, W/16]`` and
  ``audio_latents`` ``[2, 32, Ta]``, normalized.
* ``decoded.safetensors``: ``video`` (uint8 ``[F, H, W, 3]``) and ``audio``
  (FP32 ``[2, N]`` at 32 kHz); ``output.mp4`` holds the same media as H.264
  and AAC.
* ``metadata.json``: workload, seed, attention kernel, software versions,
  geometry, row counts and timings.

Reference videos are decoded the way the reference implementation decodes
them, with the pinned FFmpeg build in one pass
(``fps=24,scale=W:H:flags=lanczos,setsar=1`` to RGB24 at the canvas the
video's own aspect ratio resolves to) and handed to diffusers at 24 fps on
that canvas, which its setup step passes through unchanged. Soundtracks and
audio references are decoded by the diffusers reference classes at their
native sample rate and resampled once by the setup step.

Two prerequisites of the diffusers integration are not part of the
repository environment: accelerate (diffusers loads the DiT's FP32 modules
only through its low-memory loading path) and torchaudio (the setup step
resamples reference soundtracks with it). Put them on ``PYTHONPATH`` without
touching the environment, e.g. ``uv pip install --target <dir> --no-deps
accelerate==1.15.0 psutil 'torchaudio==2.11.0+cu130'`` with the PyTorch
cu130 index.

Example, on one GPU: ``CUDA_VISIBLE_DEVICES=2 .venv/bin/python
tools/minimax_h3/diffusers_reference.py --workload t2va_16x9_5s --seed 42
--output artifacts/minimax_h3``.
"""

import argparse
import contextlib
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
from workloads import WORKLOADS, Workload

CHECKPOINT = Path("/workspace/models/MiniMax-H3")
FFMPEG = Path("/workspace/tools/ffmpeg-8.1.2/bin/ffmpeg")
FFPROBE = Path("/workspace/tools/ffmpeg-8.1.2/bin/ffprobe")

# Native precision of every component, as stored in the checkpoint. The DiT
# keeps its FP32 islands through ``_keep_in_fp32_modules``.
NATIVE_DTYPES = {
    "transformer": torch.bfloat16,
    "transformer_ref": torch.bfloat16,
    "text_encoder": torch.bfloat16,
    "vae": torch.float32,
    "audio_vae": torch.float32,
}

# DiT attention kernels. Both are PyTorch SDPA backends; the canonical
# reference uses cuDNN, which is also what PyTorch's default SDPA dispatch
# selects on SM100. The text encoder and the VAEs keep the default dispatch.
ATTENTION_BACKENDS = {"cudnn": "_native_cudnn", "flash": "_native_flash"}

FPS = 24
SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
CANVAS_MULTIPLE = 32


def _parse_aspect(aspect_ratio: str) -> tuple[float, float]:
    width, height = aspect_ratio.split(":")
    return float(width), float(height)


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _checkpoint_revision(checkpoint: Path, component: str) -> str:
    """Return the HF commit the component's index file was downloaded at."""
    meta_dir = checkpoint / ".cache" / "huggingface" / "download" / component
    for meta in sorted(meta_dir.glob("*.metadata")):
        return meta.read_text().splitlines()[0]
    return "unknown"


class Recorder:
    """Collects the intermediate tensors of one pipeline call.

    The recorder patches three module-level seams of the diffusers
    MiniMax-H3 blocks: ``randn_tensor`` in ``before_denoise`` (every draw of
    the request generator), ``get_qwen3vl_prompt_embeds`` in ``encoders``
    (presentation and Qwen output) and the scheduler step of the denoising
    loop (per-step predictions and samples). It also hooks the DiT token
    refiner. Replayed draws are returned instead of fresh ones when
    ``replay`` is given, in draw order, after checking their shapes.
    """

    def __init__(self, replay: list[torch.Tensor] | None = None):
        self.replay = replay
        self.draws: list[torch.Tensor] = []
        self.text: dict = {}
        self.refined_text: torch.Tensor | None = None
        self.video_predictions: list[torch.Tensor] = []
        self.audio_predictions: list[torch.Tensor] = []
        self.video_samples: list[torch.Tensor] = []
        self.audio_samples: list[torch.Tensor] = []
        self.step_seconds: list[float] = []
        self.marks: dict[str, float] = {}

    @contextlib.contextmanager
    def patch(self, transformer: torch.nn.Module, num_steps: int):
        """Install the recording seams for the duration of the context."""
        from diffusers.modular_pipelines.minimax_h3 import (
            before_denoise,
            denoise,
            encoders,
        )

        original_randn = before_denoise.randn_tensor
        original_embeds = encoders.get_qwen3vl_prompt_embeds
        step_class = denoise.MiniMaxH3LoopSchedulerStep
        original_step = step_class.__call__
        recorder = self

        def randn_tensor(
            shape, generator=None, device=None, dtype=None, layout=None
        ):
            if recorder.replay is not None:
                stored = recorder.replay[len(recorder.draws)]
                if tuple(stored.shape) != tuple(shape):
                    raise ValueError(
                        f"replayed draw {len(recorder.draws)} has shape "
                        f"{tuple(stored.shape)}, the run draws {tuple(shape)}"
                    )
                value = stored.to(device=device, dtype=dtype)
            else:
                value = original_randn(
                    shape,
                    generator=generator,
                    device=device,
                    dtype=dtype,
                    layout=layout,
                )
            recorder.draws.append(value.detach().cpu().clone())
            return value

        def get_qwen3vl_prompt_embeds(
            text_encoder, processor, token_ids, vision_inputs=None, **kwargs
        ):
            recorder.marks["text_start"] = time.perf_counter()
            output = original_embeds(
                text_encoder, processor, token_ids, vision_inputs, **kwargs
            )
            torch.cuda.synchronize()
            recorder.marks["text_end"] = time.perf_counter()
            recorder.text = {
                "token_ids": list(token_ids),
                "vision_inputs": {
                    name: value.detach().cpu().clone()
                    for name, value in (vision_inputs or {}).items()
                },
                "hidden_states": output.detach().cpu().clone(),
            }
            return output

        def step(self, components, block_state, i, t):
            video_rows = block_state.num_condition_video_rows
            audio_rows = block_state.num_condition_audio_rows
            recorder.video_predictions.append(
                block_state.noise_pred[0, video_rows:].float().cpu().clone()
            )
            recorder.audio_predictions.append(
                block_state.audio_noise_pred[0, audio_rows:]
                .float()
                .cpu()
                .clone()
            )
            result = original_step(self, components, block_state, i, t)
            recorder.video_samples.append(
                block_state.latents[video_rows:].float().cpu().clone()
            )
            recorder.audio_samples.append(
                block_state.audio_latents[audio_rows:].float().cpu().clone()
            )
            now = time.perf_counter()
            previous = recorder.marks.get(
                "last_step", recorder.marks.get("text_end", now)
            )
            recorder.step_seconds.append(now - previous)
            recorder.marks["last_step"] = now
            print(
                f"step {i + 1}/{num_steps} t={float(t):.6f} "
                f"{now - previous:.2f}s",
                flush=True,
            )
            return result

        def refiner_hook(module, inputs, output):
            if recorder.refined_text is None:
                recorder.refined_text = output.detach().cpu().clone()

        before_denoise.randn_tensor = randn_tensor
        encoders.get_qwen3vl_prompt_embeds = get_qwen3vl_prompt_embeds
        step_class.__call__ = step
        handle = transformer.token_refiner.register_forward_hook(refiner_hook)
        try:
            yield
        finally:
            before_denoise.randn_tensor = original_randn
            encoders.get_qwen3vl_prompt_embeds = original_embeds
            step_class.__call__ = original_step
            handle.remove()


def _display_size(path: Path) -> tuple[int, int]:
    """Return the display width and height of a video's first stream."""
    facts = json.loads(
        subprocess.run(
            [
                str(FFPROBE),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )["streams"][0]
    width, height = int(facts["width"]), int(facts["height"])
    rotation = 0
    for side_data in facts.get("side_data_list", []):
        rotation = int(side_data.get("rotation", rotation))
    if rotation % 180:
        width, height = height, width
    return width, height


def decode_reference_video(path: Path, num_frames: int) -> np.ndarray:
    """Decode a reference video onto 24 fps and its own canvas with FFmpeg.

    Returns ``(frames, height, width, 3)`` uint8 RGB with at most
    ``num_frames`` frames.
    """
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
        resolve_canvas_size,
    )

    width, height = _display_size(path)
    canvas_height, canvas_width = resolve_canvas_size(
        width, height, CANVAS_MULTIPLE, SHORT_EDGE, MAX_PIXELS
    )
    command = [
        str(FFMPEG),
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        f"fps={FPS},scale={canvas_width}:{canvas_height}:flags=lanczos,"
        "setsar=1",
        "-frames:v",
        str(num_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    payload = subprocess.run(command, check=True, capture_output=True).stdout
    frame_bytes = canvas_width * canvas_height * 3
    if not payload or len(payload) % frame_bytes:
        raise ValueError(f"partial or empty RGB24 stream from {path}")
    return np.frombuffer(payload, dtype=np.uint8).reshape(
        -1, canvas_height, canvas_width, 3
    )


def build_inputs(workload: Workload, inputs: Path) -> dict:
    """Translate a workload into diffusers pipeline keyword arguments."""
    from diffusers.modular_pipelines.minimax_h3 import references
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
        align_num_frames,
        resolve_canvas_size,
    )
    from diffusers.utils import load_image

    requested_frames = round(workload.duration_seconds * FPS)
    aligned_frames = align_num_frames(requested_frames, 17, 5)
    kwargs = {
        "prompt": workload.resolve_prompt(inputs),
        "num_frames": requested_frames,
    }
    if workload.aspect_ratio != "auto":
        kwargs["height"], kwargs["width"] = resolve_canvas_size(
            *_parse_aspect(workload.aspect_ratio),
            CANVAS_MULTIPLE,
            SHORT_EDGE,
            MAX_PIXELS,
        )

    if workload.task == "fl2va":
        for condition in workload.conditions:
            image = load_image(str(inputs / condition.media))
            name = "image" if condition.frame_index == 0 else "last_image"
            kwargs[name] = image
    elif workload.task == "ref2va":
        entries = []
        for condition in workload.conditions:
            path = inputs / condition.media
            if condition.role != "reference":
                raise ValueError(
                    f"{workload.name}: the diffusers MiniMax-H3 integration "
                    "has no keyframe input for ref2va"
                )
            if condition.kind == "image":
                entries.append(
                    references.MiniMaxH3ImageReference.from_file(path)
                )
            elif condition.kind == "audio":
                entries.append(
                    references.MiniMaxH3AudioReference.from_file(path)
                )
            else:
                frames = decode_reference_video(path, aligned_frames)
                try:
                    waveform, sample_rate = references._decode_audio_file(path)
                except ValueError:
                    waveform, sample_rate = None, None
                entries.append(
                    references.MiniMaxH3VideoReference(
                        frames=frames,
                        fps=float(FPS),
                        audio=waveform,
                        sample_rate=sample_rate,
                    )
                )
        kwargs["references"] = entries
    return kwargs


def load_pipeline(task: str, checkpoint: Path, attention: str):
    """Load the workflow's components at native precision onto CUDA."""
    from diffusers import ModularPipeline

    pipe = ModularPipeline.from_pretrained(str(checkpoint), workflow=task)
    # The pipeline's blocks are already pruned to the workflow, so this loads
    # only the components the task uses (one of the two DiT partitions).
    pipe.load_components(
        pretrained_model_name_or_path=str(checkpoint),
        dtype=NATIVE_DTYPES,
    )
    for name, component in pipe.components.items():
        if isinstance(component, torch.nn.Module):
            component.to("cuda")
    from diffusers.models.attention_dispatch import (
        AttentionBackendName,
        _AttentionBackendRegistry,
    )

    denoiser = "transformer_ref" if task == "ref2va" else "transformer"
    getattr(pipe, denoiser).set_attention_backend(ATTENTION_BACKENDS[attention])
    # set_attention_backend also makes the kernel the process-wide default,
    # which would force it onto the VAEs (the audio VAE encoder attends in
    # FP32, which cuDNN SDPA cannot run). The DiT processors keep their own
    # pinned backend; every other diffusers attention keeps PyTorch's default
    # SDPA dispatch.
    _AttentionBackendRegistry.set_active_backend(AttentionBackendName.NATIVE)
    return pipe, getattr(pipe, denoiser)


def write_mp4(
    path: Path, video: np.ndarray, audio: np.ndarray, sample_rate: int
) -> None:
    """Mux uint8 RGB frames and a stereo waveform into H.264 + AAC MP4."""
    import av

    with av.open(str(path), "w") as container:
        video_stream = container.add_stream("libx264", rate=FPS)
        video_stream.width = video.shape[2]
        video_stream.height = video.shape[1]
        video_stream.pix_fmt = "yuv420p"
        video_stream.options = {"crf": "12", "preset": "medium"}
        audio_stream = container.add_stream(
            "aac", rate=sample_rate, layout="stereo"
        )
        for frame in video:
            packet = video_stream.encode(
                av.VideoFrame.from_ndarray(frame, format="rgb24")
            )
            container.mux(packet)
        container.mux(video_stream.encode(None))

        chunk = audio_stream.codec_context.frame_size or 1024
        for start in range(0, audio.shape[1], chunk):
            block = np.ascontiguousarray(audio[:, start : start + chunk])
            frame = av.AudioFrame.from_ndarray(
                block, format="fltp", layout="stereo"
            )
            frame.sample_rate = sample_rate
            frame.pts = start
            container.mux(audio_stream.encode(frame))
        container.mux(audio_stream.encode(None))


def _versions() -> dict:
    import diffusers
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(),
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tools_commit": _git_head(),
        "ffmpeg": subprocess.run(
            [str(FFMPEG), "-version"], capture_output=True, text=True
        ).stdout.splitlines()[0],
    }


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
    inputs = args.output / "inputs"
    replay = None
    if args.replay_noise is not None:
        stored = load_file(str(args.replay_noise))
        names = json.loads(
            (args.replay_noise.parent / "metadata.json").read_text()
        )["noise_draw_order"]
        replay = [stored[name] for name in names]

    started = time.perf_counter()
    pipe, transformer = load_pipeline(
        workload.task, args.checkpoint, args.attention
    )
    loaded = time.perf_counter()
    print(f"loaded in {loaded - started:.1f}s", flush=True)

    kwargs = build_inputs(workload, inputs)
    recorder = Recorder(replay)
    generator = torch.Generator("cpu").manual_seed(args.seed)
    num_steps = args.num_inference_steps - 1
    with recorder.patch(transformer, num_steps):
        state = pipe(
            **kwargs,
            generator=generator,
            num_inference_steps=args.num_inference_steps,
            output_type="pt",
        )
    torch.cuda.synchronize()
    finished = time.perf_counter()

    value = state.get
    num_condition_video_rows = int(value("num_condition_video_rows") or 0)
    num_condition_audio_rows = int(value("num_condition_audio_rows") or 0)
    condition_latents = value("condition_latents") or []
    audio_condition_latents = value("audio_condition_latents") or []

    # Name the generator draws: one per visual condition first, then the
    # generated video noise and the generated audio noise.
    draw_names = [f"condition_noise.{k}" for k in range(len(condition_latents))]
    draw_names += ["video_noise", "audio_noise"]
    if len(draw_names) != len(recorder.draws):
        raise RuntimeError(
            f"expected {len(draw_names)} generator draws, recorded "
            f"{len(recorder.draws)}"
        )
    save_file(
        {
            name: draw.contiguous()
            for name, draw in zip(draw_names, recorder.draws)
        },
        str(run_dir / "noise.safetensors"),
    )

    text = {
        "qwen_hidden_states_50": recorder.text["hidden_states"].contiguous(),
        "refined_text": recorder.refined_text.contiguous(),
    }
    for name, tensor in recorder.text["vision_inputs"].items():
        text[f"qwen_{name}"] = tensor.contiguous()
    save_file(text, str(run_dir / "text.safetensors"))

    token_tags = value("text_token_tags")
    presentation = {
        "token_ids": recorder.text["token_ids"],
        "tags": [int(tag) for tag in token_tags.tolist()],
        "num_text_rows": len(recorder.text["token_ids"]),
        "image_grid_thw": recorder.text["vision_inputs"]
        .get("image_grid_thw", torch.empty(0, 3))
        .tolist(),
        "video_grid_thw": recorder.text["vision_inputs"]
        .get("video_grid_thw", torch.empty(0, 3))
        .tolist(),
    }
    (run_dir / "presentation.json").write_text(json.dumps(presentation) + "\n")

    conditions = {}
    for k, latents in enumerate(condition_latents):
        conditions[f"video_condition.{k}"] = latents.contiguous()
    for k, rows in enumerate(audio_condition_latents):
        conditions[f"audio_condition.{k}"] = rows.contiguous()
    if value("condition_rows") is not None:
        conditions["condition_rows"] = value("condition_rows").cpu()
    if conditions:
        save_file(conditions, str(run_dir / "conditions.safetensors"))

    scheduler = pipe.scheduler
    audio_scheduler = pipe.audio_scheduler
    save_file(
        {
            "position_ids": value("position_ids").cpu().contiguous(),
            "token_tags": value("token_tags").cpu().contiguous(),
            "video_indices": value("video_indices").cpu().contiguous(),
            "audio_indices": value("audio_indices").cpu().contiguous(),
            "text_indices": value("text_indices").cpu().contiguous(),
            "sigmas": scheduler.sigmas.cpu().contiguous(),
            "audio_sigmas": audio_scheduler.sigmas.cpu().contiguous(),
            "timesteps": value("timesteps").cpu().contiguous(),
            "audio_timesteps": value("audio_timesteps").cpu().contiguous(),
        },
        str(run_dir / "layout.safetensors"),
    )

    save_file(
        {
            "video_predictions": torch.stack(recorder.video_predictions),
            "audio_predictions": torch.stack(recorder.audio_predictions),
            "video_samples": torch.stack(recorder.video_samples),
            "audio_samples": torch.stack(recorder.audio_samples),
        },
        str(run_dir / "trajectory.safetensors"),
    )
    save_file(
        {
            "video_latents": value("latents").float().cpu().contiguous(),
            "audio_latents": value("audio_latents").float().cpu().contiguous(),
        },
        str(run_dir / "final.safetensors"),
    )

    # Decoded media: [1, F, 3, H, W] in [0, 1] -> uint8 [F, H, W, 3].
    video = value("videos")[0].float().cpu().clamp(0, 1)
    video = (video * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1)
    audio = value("audio")[0].float().cpu()
    sample_rate = int(value("sampling_rate"))
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
    num_text_rows = len(recorder.text["token_ids"])
    metadata = {
        "impl": args.impl,
        "workload": workload.name,
        "task": workload.task,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "denoising_steps": len(recorder.video_samples),
        "attention": {
            "dit": ATTENTION_BACKENDS[args.attention],
            "text_encoder_and_vaes": "torch sdpa default dispatch",
        },
        "replayed_noise": (
            str(args.replay_noise) if args.replay_noise else None
        ),
        "noise_draw_order": draw_names,
        "generator": "torch.Generator('cpu'), float32 draws",
        "precision": {
            name: str(dtype) for name, dtype in NATIVE_DTYPES.items()
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "revision": _checkpoint_revision(
                args.checkpoint,
                "transformer_ref"
                if workload.task == "ref2va"
                else "transformer",
            ),
        },
        "geometry": {
            "height": int(value("height")),
            "width": int(value("width")),
            "num_frames": int(value("num_frames")),
            "num_latent_frames": int(value("num_latent_frames")),
            "latent_height": int(value("latent_height")),
            "latent_width": int(value("latent_width")),
            "num_audio_latents": int(value("num_audio_latents")),
            "num_text_rows": num_text_rows,
            "num_condition_video_rows": num_condition_video_rows,
            "num_condition_audio_rows": num_condition_audio_rows,
            "sequence_rows": int(value("position_ids").shape[0]),
            "keyframe_anchors": list(value("keyframe_anchors") or ()),
            "audio_samples": int(audio.shape[-1]),
            "audio_sample_rate": sample_rate,
        },
        "timings_seconds": {
            "load": loaded - started,
            "text_encoding": recorder.marks["text_end"]
            - recorder.marks["text_start"],
            "denoising": sum(recorder.step_seconds),
            "per_step_median": float(np.median(recorder.step_seconds)),
            "call_total": finished - loaded,
        },
        "versions": _versions(),
        "tf32": {
            "matmul": torch.backends.cuda.matmul.allow_tf32,
            "cudnn": torch.backends.cudnn.allow_tf32,
        },
        "command": sys.argv,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
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
        "--attention", choices=sorted(ATTENTION_BACKENDS), default="cudnn"
    )
    parser.add_argument(
        "--impl",
        default=None,
        help="reference/<impl> directory name; defaults to 'diffusers' for "
        "cuDNN and 'diffusers_sdpa_<kernel>' otherwise",
    )
    parser.add_argument(
        "--replay-noise",
        type=Path,
        default=None,
        help="noise.safetensors of an earlier run to feed instead of draws",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="sigma points including the terminal 0 (50 = 49 DiT forwards)",
    )
    args = parser.parse_args()
    if args.impl is None:
        args.impl = (
            "diffusers"
            if args.attention == "cudnn"
            else f"diffusers_sdpa_{args.attention}"
        )
    if args.seed < 0:
        raise SystemExit("seed must be a non-negative integer")

    from diffusers.utils import is_accelerate_available

    if not is_accelerate_available():
        raise SystemExit(
            "diffusers needs accelerate to load the DiT's FP32 modules; "
            "put accelerate on PYTHONPATH (see the module docstring)"
        )
    try:
        import torchaudio  # noqa: F401
    except ImportError:
        raise SystemExit(
            "the diffusers setup step resamples reference audio with "
            "torchaudio; put it on PYTHONPATH (see the module docstring)"
        ) from None
    run(args)


if __name__ == "__main__":
    main()
