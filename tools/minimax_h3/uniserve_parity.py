"""Run UniServe's MiniMax-H3 base denoiser on a reference run's inputs.

The trajectory acceptance protocol compares UniServe with a recorded
reference run (``diffusers_reference.py``, or ``sglang_reference.py`` for
the layouts only SGLang implements) on identical inputs. This script loads
the run's denoiser (``denoiser`` for t2va and fl2va, ``reference_denoiser``
for ref2va) through the public loader at the ``quality`` preset on one GPU
and injects the run's Qwen ``hidden_states[50]`` (UniServe refines the text
itself), its clean condition latents and every noise draw, then evaluates
the 49 denoising steps eagerly:

* free (default): the trajectory runs on UniServe's own samples; the final
  latents are decoded with UniServe's video and audio decoders. The output
  directory holds ``trajectory.safetensors``, ``final.safetensors``,
  ``decoded.safetensors``, ``text.safetensors`` and ``noise.safetensors``
  in the reference runs' formats, so ``trajectory_metrics.py`` compares it
  with the reference like any other run.
* ``--teacher``: every step starts from the reference's recorded sample of
  the previous step, so each prediction is made on the reference's inputs.

Both modes write ``metrics.json`` with the per-step relative L2 of the
generated rows' predictions and samples against the recording. Outputs go
to ``<output>/<workload>/seed<seed>/uniserve_{free,teacher}/``.

Run from the repository root, e.g. ``CUDA_VISIBLE_DEVICES=0
.venv/bin/python tools/minimax_h3/uniserve_parity.py --run
artifacts/minimax_h3/reference/diffusers/t2va_16x9_5s/seed42 --output
artifacts/minimax_h3/acceptance``.
"""

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from uniserve.diffusion import advance_
from uniserve.execution import DenoisingRunner
from uniserve.media import image, video
from uniserve.model import Condition, ConditionRole, LatentInput, TextSize
from uniserve.runtime import ExecutionContext, TensorBuffers
from uniserve_models import loading as models
from uniserve_models.minimax_h3 import DenoiserInput, weight_config
from uniserve_models.minimax_h3.packing import patchify_video

CHECKPOINT = Path("/workspace/models/MiniMax-H3")

# 32 kHz samples per audio latent frame; an audio track's recorded rows are
# channel-major, two rows per latent frame.
SAMPLES_PER_AUDIO_LATENT = 800


def rel_l2(value: torch.Tensor, reference: torch.Tensor) -> float:
    """Return ``||value - reference|| / ||reference||`` in float64."""
    value, reference = value.double().cpu(), reference.double().cpu()
    return float((value - reference).norm() / reference.norm())


def _vision_spans(tags: list[int]) -> tuple[tuple[int, int], ...]:
    """Spans of vision tokens (tag 0) in the Qwen presentation."""
    spans, start = [], None
    for index, tag in enumerate([*tags, 1]):
        if tag == 0 and start is None:
            start = index
        elif tag != 0 and start is not None:
            spans.append((start, index))
            start = None
    return tuple(spans)


def _latent_frames_to_pixels(latent_frames: int) -> int:
    """Pixel frames of a video condition whose latent has these frames."""
    return 1 if latent_frames == 1 else (latent_frames - 2) // 5 * 17 + 5


def conditions_of(
    run: Path, canvas: image.Config
) -> tuple[tuple[Condition, ...], tuple[torch.Tensor, ...]]:
    """Rebuild the request's conditions and their latents in request order.

    The recording lists the visual conditions' clean latents in request order
    (``video_condition.<k>``, ``[1, 24, T, H/16, W/16]``) and every audio
    track in request order (``audio_condition.<k>``, channel-major rows):
    audio references, and the soundtracks of video references that carry
    one. A keyframe is one frame on the canvas; a reference keeps the raster
    its latent was encoded at.
    """
    request = json.loads((run / "request.json").read_text())
    path = run / "conditions.safetensors"
    tensors = load_file(str(path)) if path.exists() else {}
    visual = [
        tensors[f"video_condition.{index}"]
        for index in range(
            sum(1 for name in tensors if name.startswith("video_condition."))
        )
    ]
    audio = [
        tensors[f"audio_condition.{index}"]
        for index in range(
            sum(1 for name in tensors if name.startswith("audio_condition."))
        )
    ]
    kinds = [condition["type"] for condition in request["conditions"]]
    # Video references take a soundtrack while audio tracks beyond the pure
    # audio references remain.
    spare_tracks = len(audio) - kinds.count("audio")
    visual_iter, audio_iter = iter(visual), iter(audio)

    conditions, latents = [], []
    for entry in request["conditions"]:
        kind = entry["type"]
        if entry["role"] == "keyframe":
            role = (
                ConditionRole.FIRST_FRAME
                if entry["frame_index"] == 0
                else ConditionRole.LAST_FRAME
            )
            latent = next(visual_iter)
            conditions.append(Condition(role, video.Config(1, canvas)))
            latents.append(patchify_video(latent)[0])
            continue

        pixels, samples = None, 0
        if kind in ("image", "video", "video_audio"):
            latent = next(visual_iter)
            _, _, latent_frames, height, width = latent.shape
            pixels = video.Config(
                _latent_frames_to_pixels(latent_frames),
                image.Config(height * 16, width * 16),
            )
            latents.append(patchify_video(latent)[0])
        if kind == "audio" or (
            kind in ("video", "video_audio") and spare_tracks > 0
        ):
            if kind != "audio":
                spare_tracks -= 1
            rows = next(audio_iter)
            samples = rows.shape[0] // 2 * SAMPLES_PER_AUDIO_LATENT
            latents.append(rows)
        conditions.append(Condition(ConditionRole.REFERENCE, pixels, samples))

    if (
        next(visual_iter, None) is not None
        or next(audio_iter, None) is not None
    ):
        raise ValueError("the recorded condition latents exceed the request")
    return tuple(conditions), tuple(latents)


def decode(model, final: dict, num_frames: int, canvas: image.Config, device):
    """Decode final latents with UniServe's decoders, as the server does.

    Returns uint8 ``[F, H, W, 3]`` video and FP32 ``[2, samples]`` audio in
    [-1, 1], the reference runs' decoded format.
    """
    latents = patchify_video(final["video_latents"])[0].to(device)
    output = video.Config(num_frames, canvas)
    decoder, postprocessor = model.video_decoder, model.video_postprocessor
    frames = []
    with (
        ExecutionContext(decoder) as decoding,
        ExecutionContext(postprocessor) as processing,
    ):
        processing.prepare(output)
        # The postprocessor blends consecutive windows through this state.
        overlap = {
            name: torch.zeros(spec.shape, dtype=spec.dtype, device=device)
            for name, spec in postprocessor.state_buffers(output).items()
        }
        prepared = None
        for window in decoder.frame_slices(num_frames):
            # Each window is unpacked from the whole latent and decoded at
            # its segment, which the context is prepared for.
            segment = decoder.segment(output, window)
            if segment != prepared:
                decoding.prepare(segment)
                prepared = segment
            config = decoder.window_input(segment)
            inputs = torch.empty(
                config.shape, dtype=config.dtype, device=device
            )
            decoder.unpack_latents(latents, window, output, out=inputs)
            with decoding.activate():
                (decoded,) = decoder.decode((inputs,), segments=(segment,))
            with processing.activate():
                rgb = postprocessor(
                    (decoder.place(decoded, window, output),),
                    frames=(window,),
                    sizes=(output,),
                    state=overlap,
                    constants=processing.constants,
                    workspace=processing.workspace,
                )
            frames.extend(value.tensor.clone().cpu() for value in rgb)

    audio_decoder = model.audio_decoder
    samples = audio_decoder.track_samples(num_frames, postprocessor.frame_rate)
    count = audio_decoder.latent_frames(samples)
    rows = final["audio_latents"].permute(0, 2, 1).reshape(-1, 32).to(device)
    with ExecutionContext(audio_decoder) as context:
        context.prepare(count)
        with context.activate():
            pcm = audio_decoder.decode(
                (rows,),
                frames=(slice(0, count),),
                num_samples=(samples,),
                workspace=context.workspace,
            )[0]
    return {
        "video": torch.cat(frames).contiguous(),
        # 16-bit PCM to [-1, 1], channel-major like the reference.
        "audio": (pcm.float().cpu() / 32768.0).transpose(0, 1).contiguous(),
    }


@torch.inference_mode()
def run(args) -> Path:
    metadata = json.loads((args.run / "metadata.json").read_text())
    geometry = metadata["geometry"]
    presentation = json.loads((args.run / "presentation.json").read_text())
    text = load_file(str(args.run / "text.safetensors"))
    noise = load_file(str(args.run / "noise.safetensors"))
    trajectory = load_file(str(args.run / "trajectory.safetensors"))
    device = torch.device("cuda", 0)

    component = (
        "reference_denoiser" if metadata["task"] == "ref2va" else "denoiser"
    )
    modules = {component}
    if not args.teacher:
        modules |= {"video_decoder", "video_postprocessor", "audio_decoder"}
    started = time.perf_counter()
    config = models.read_config(args.checkpoint, modules=frozenset(modules))
    model = models.load_model(
        config,
        device=device,
        weights=weight_config(config.model, preset="quality"),
    ).model
    denoiser = getattr(model, component)
    loaded = time.perf_counter()

    canvas = image.Config(geometry["height"], geometry["width"])
    num_frames = geometry["num_frames"]
    tokens = len(presentation["token_ids"])
    conditions, latents = conditions_of(args.run, canvas)
    size = denoiser.make_size(
        num_frames,
        tokens,
        canvas=canvas,
        conditions=conditions,
        vision_spans=_vision_spans(presentation["tags"]),
    )
    layout = denoiser.layout_size(size)

    # The retained conditioning: UniServe's refined text from the injected
    # Qwen output, then the encoded conditions anchored with the recorded
    # condition draws.
    conditioning = torch.zeros(
        denoiser.text_condition_rows(layout),
        denoiser.text_condition_width,
        dtype=torch.bfloat16,
        device=device,
    )
    with ExecutionContext(denoiser.conditioner) as context:
        context.prepare(TextSize(tokens, 1))
        refined = denoiser.conditioner.encode(
            (text["qwen_hidden_states_50"][0].to(device),)
        )[0]
    conditioning[:tokens].copy_(refined)
    draws = tuple(
        noise[f"condition_noise.{index}"]
        for index in range(len(denoiser.condition_noise_shapes(size)))
    )
    denoiser.encode_conditions(
        size, layout, latents=latents, noise=draws, out=conditioning
    )

    requirements = denoiser.state_buffers(layout)
    native = {
        "video": noise["video_noise"].unsqueeze(0),
        "audio": noise["audio_noise"].unsqueeze(0),
    }
    with (
        ExecutionContext(denoiser) as context,
        TensorBuffers.allocate(requirements, device="cpu") as host,
        TensorBuffers.allocate(requirements, device=device) as backing,
    ):
        runner = DenoisingRunner(denoiser, context=context)
        runner.warmup(layout)
        request, staged = backing.view(requirements), host.view(requirements)
        runner.prepare_latents(
            (layout,),
            noise=native,
            state={
                name: staged[name].unsqueeze(0) for name in denoiser.modalities
            },
        )
        runner.prepare_state(
            (size,),
            layouts=(layout,),
            out={
                name: value
                for name, value in staged.items()
                if name not in denoiser.modalities
            },
        )
        for name, value in request.items():
            value.copy_(staged[name])

        # Each local sample row's raster (video) or channel-major (audio)
        # row, the recording's row order.
        order = {
            name: context.constants[f"{name}_indices"]
            for name in denoiser.modalities
        }

        def raster(name, rows):
            out = torch.empty_like(rows)
            out[order[name].to(rows.device)] = rows
            return out

        schedules = denoiser.make_schedules(
            denoiser.num_steps, shift=None, device=device
        )
        samples = {name: request[name] for name in denoiser.modalities}
        recorded = {
            f"{name}_{kind}": []
            for name in denoiser.modalities
            for kind in ("predictions", "samples")
        }
        for step in range(denoiser.num_steps):
            if args.teacher and step:
                for name in denoiser.modalities:
                    reference = trajectory[f"{name}_samples"][step - 1]
                    samples[name].copy_(
                        reference[order[name]].to(samples[name].device)
                    )
            inputs = DenoiserInput(
                {
                    name: (LatentInput(value, schedules[name].timesteps[step]),)
                    for name, value in samples.items()
                },
                (layout,),
                schedules["video"].step(step),
                (conditioning,),
            )
            with context.activate():
                predictions = denoiser(
                    inputs,
                    state=request,
                    constants=context.constants,
                    workspace=context.workspace,
                )
                for name in denoiser.modalities:
                    recorded[f"{name}_predictions"].append(
                        raster(name, predictions[name][0].tensor.float().cpu())
                    )
                advance_(
                    denoiser,
                    inputs.latents,
                    predictions,
                    schedules,
                    inputs.step,
                )
            for name in denoiser.modalities:
                recorded[f"{name}_samples"].append(
                    raster(name, samples[name].float().cpu())
                )
            deviation = rel_l2(
                recorded["video_predictions"][step],
                trajectory["video_predictions"][step],
            )
            print(
                f"step {step}: video prediction rel L2 {deviation:.5f}",
                flush=True,
            )

    mode = "teacher" if args.teacher else "free"
    directory = (
        args.output
        / metadata["workload"]
        / f"seed{metadata['seed']}"
        / f"uniserve_{mode}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    stacked = {
        name: torch.stack(values).contiguous()
        for name, values in recorded.items()
    }
    save_file(stacked, str(directory / "trajectory.safetensors"))

    metrics = {"run": str(args.run), "mode": mode}
    for name in denoiser.modalities:
        for kind in ("prediction", "sample"):
            metrics[f"{name}_{kind}_rel_l2"] = [
                rel_l2(ours, reference)
                for ours, reference in zip(
                    stacked[f"{name}_{kind}s"],
                    trajectory[f"{name}_{kind}s"],
                    strict=True,
                )
            ]

    if not args.teacher:
        # Final latents in the reference's native layouts: video
        # [1, 24, T, H/16, W/16] from patchified 2x2 rows, audio [2, 32, Ta]
        # from channel-major rows.
        height, width = canvas.height // 16, canvas.width // 16
        video_rows = stacked["video_samples"][-1]
        latent_frames = video_rows.shape[0] // ((height // 2) * (width // 2))
        final = {
            "video_latents": video_rows.reshape(
                latent_frames, height // 2, width // 2, 24, 2, 2
            )
            .permute(3, 0, 1, 4, 2, 5)
            .reshape(1, 24, latent_frames, height, width)
            .contiguous(),
            "audio_latents": stacked["audio_samples"][-1]
            .reshape(2, -1, 32)
            .permute(0, 2, 1)
            .contiguous(),
        }
        save_file(final, str(directory / "final.safetensors"))
        save_file(
            decode(model, final, num_frames, canvas, device),
            str(directory / "decoded.safetensors"),
        )
        save_file(
            {
                "qwen_hidden_states_50": text[
                    "qwen_hidden_states_50"
                ].contiguous(),
                "refined_text": refined[None].cpu().contiguous(),
            },
            str(directory / "text.safetensors"),
        )
        save_file(dict(noise), str(directory / "noise.safetensors"))

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                **metadata,
                "impl": "uniserve",
                "mode": mode,
                "reference_run": str(args.run),
                "uniserve_commit": commit,
                "seconds": {"load": loaded - started},
            },
            indent=2,
        )
        + "\n"
    )
    (directory / "metrics.json").write_text(
        json.dumps(
            {
                key: value
                if not isinstance(value, float) or math.isfinite(value)
                else str(value)
                for key, value in metrics.items()
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {directory}", flush=True)
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--teacher", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
