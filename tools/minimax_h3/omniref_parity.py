"""Run the FastH3 OmniRef denoiser eagerly on a FastVideo run's inputs.

Loads the OmniRef component export through the public loader (its base from
``--base``), places only the reference denoiser on one GPU and evaluates the
eight PDD steps of one recorded ``fastvideo_reference.py`` run with that
run's inputs injected: the refined text (FastVideo encodes text with its own
Qwen3-VL path, so the text is not re-encoded), the encoded condition latents
and every noise draw. ``--teacher`` starts every step from the recorded
run's sample of the previous step, so each prediction is made on the
reference's inputs; without it the trajectory runs free.

Compares, per step, the predicted velocities and the updated samples of the
generated rows with the recorded ones (relative L2, raster row order), and
the initial samples bit for bit, and writes the metrics and UniServe's
tensors to ``<output>/parity/omniref/<workload>/<mode>/``.

Run from the repository root with the repository interpreter, e.g.
``.venv/bin/python tools/minimax_h3/omniref_parity.py --run
artifacts/minimax_h3/reference/fastvideo_omniref/<workload>/seed42
--checkpoint /models/FastH3-OmniRef-DMPDD8-step1900 --base /models/MiniMax-H3
--output artifacts/minimax_h3 --teacher``.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from uniserve.diffusion import advance_
from uniserve.execution import DenoisingRunner
from uniserve.media import image, video
from uniserve.model import Condition, ConditionRole, LatentInput
from uniserve.runtime import ExecutionContext, TensorBuffers
from uniserve_models import loading as models
from uniserve_models.minimax_h3 import DenoiserInput

COMPONENT = "reference_denoiser"


def rel_l2(reference: torch.Tensor, other: torch.Tensor) -> float:
    reference, other = reference.double(), other.double()
    return float((other - reference).norm() / reference.norm())


def _conditions(run: Path) -> tuple[Condition, ...]:
    """Rebuild the request's conditions from the recorded geometry.

    The recorded condition tensors list visual references in request order
    (``video_condition.<i>``) and every audio track in request order
    (``audio_condition.<j>``); the request body names each condition's kind.
    """
    request = json.loads((run / "request.json").read_text())
    tensors = load_file(str(run / "conditions.safetensors"))
    visual = iter(
        tensors[f"video_condition.{index}"].shape
        for index in range(
            sum(1 for name in tensors if name.startswith("video_condition."))
        )
    )
    audio = iter(
        tensors[f"audio_condition.{index}"].shape[0] // 2
        for index in range(
            sum(1 for name in tensors if name.startswith("audio_condition."))
        )
    )
    result = []
    for condition in request["conditions"]:
        kind = condition["type"]
        if kind == "audio":
            frames = next(audio)
            result.append(
                Condition(ConditionRole.REFERENCE, None, frames * 800)
            )
            continue
        _, _, latent_frames, height, width = next(visual)
        frames = 1 if latent_frames == 1 else (latent_frames - 2) // 5 * 17 + 5
        sound = next(audio) * 800 if kind == "video" else 0
        result.append(
            Condition(
                ConditionRole.REFERENCE,
                video.Config(frames, image.Config(height * 16, width * 16)),
                sound,
            )
        )
    if next(visual, None) is not None or next(audio, None) is not None:
        raise ValueError("the recorded condition latents exceed the request")
    return tuple(result)


def _condition_latents(run: Path, conditions) -> tuple[torch.Tensor, ...]:
    """Each condition's latents in request order: visual rows, then audio."""
    tensors = load_file(str(run / "conditions.safetensors"))
    visual, audio, result = 0, 0, []
    for condition in conditions:
        if condition.video is not None:
            result.append(tensors[f"video_condition_rows.{visual}"])
            visual += 1
        if condition.audio_samples:
            result.append(tensors[f"audio_condition.{audio}"])
            audio += 1
    return tuple(result)


def _vision_spans(tags: list[int]) -> tuple[tuple[int, int], ...]:
    spans, start = [], None
    for index, tag in enumerate([*tags, 1]):
        if tag == 0 and start is None:
            start = index
        elif tag != 0 and start is not None:
            spans.append((start, index))
            start = None
    return tuple(spans)


@torch.inference_mode()
def run(args) -> dict:
    metadata = json.loads((args.run / "metadata.json").read_text())
    geometry = metadata["geometry"]
    presentation = json.loads((args.run / "presentation.json").read_text())
    text = load_file(str(args.run / "text.safetensors"))
    noise = load_file(str(args.run / "noise.safetensors"))
    trajectory = load_file(str(args.run / "trajectory.safetensors"))
    device = torch.device("cuda", 0)

    started = time.perf_counter()
    config = models.read_config(
        args.checkpoint, modules=frozenset({COMPONENT}), base=args.base
    )
    model = models.load_model(config, device=device).model
    denoiser = getattr(model, COMPONENT)
    loaded = time.perf_counter()

    conditions = _conditions(args.run)
    tokens = len(presentation["token_ids"])
    canvas = image.Config(geometry["height"], geometry["width"])
    size = denoiser.make_size(
        geometry["num_frames"],
        tokens,
        canvas=canvas,
        conditions=conditions,
        vision_spans=_vision_spans(presentation["tags"]),
    )
    layout = denoiser.layout_size(size)

    # The retained conditioning: the reference's refined text, then the
    # encoded conditions anchored with the recorded condition draws.
    conditioning = torch.zeros(
        denoiser.text_condition_rows(layout),
        denoiser.text_condition_width,
        dtype=torch.bfloat16,
        device=device,
    )
    conditioning[:tokens].copy_(text["refined_text"][0])
    draws = tuple(
        noise[f"condition_noise.{index}"]
        for index in range(len(denoiser.condition_noise_shapes(size)))
    )
    denoiser.encode_conditions(
        size,
        layout,
        latents=_condition_latents(args.run, conditions),
        noise=draws,
        out=conditioning,
    )

    requirements = denoiser.state_buffers(layout)
    native = {
        "video": noise["video_noise"].unsqueeze(0),
        "audio": noise["audio_noise"].unsqueeze(0),
    }
    result: dict = {
        "run": str(args.run),
        "teacher": args.teacher,
        "layout": {
            "text_rows": layout.num_text_tokens,
            "condition_rows": layout.condition_rows,
        },
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
                name: staged[name].unsqueeze(0) for name in ("video", "audio")
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

        # Each local sample row's raster (video) or channel-major (audio) row.
        order = {
            name: context.constants[f"{name}_indices"]
            for name in denoiser.modalities
        }

        def raster(name, rows):
            out = torch.empty_like(rows)
            out[order[name].to(rows.device)] = rows
            return out

        result["initial_equal"] = {
            name: torch.equal(
                raster(name, request[name].cpu()),
                trajectory[f"{name}_initial"],
            )
            for name in denoiser.modalities
        }
        schedules = denoiser.make_schedules(
            denoiser.num_steps, shift=None, device=device
        )
        samples = {name: request[name] for name in denoiser.modalities}
        recorded = {
            f"{name}_{kind}": []
            for name in denoiser.modalities
            for kind in ("predictions", "samples")
        }
        step_seconds = []
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
            begin = time.perf_counter()
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
            torch.cuda.synchronize()
            step_seconds.append(time.perf_counter() - begin)
            for name in denoiser.modalities:
                recorded[f"{name}_samples"].append(
                    raster(name, samples[name].float().cpu())
                )
            deviation = rel_l2(
                trajectory["video_predictions"][step],
                recorded["video_predictions"][step],
            )
            print(f"step {step}: video prediction rel L2 {deviation:.4g}")

    for name in ("video", "audio"):
        for kind in ("prediction", "sample"):
            result[f"{name}_{kind}_rel_l2"] = [
                rel_l2(reference, ours)
                for reference, ours in zip(
                    trajectory[f"{name}_{kind}s"],
                    recorded[f"{name}_{kind}s"],
                    strict=True,
                )
            ]
    result["final_video_latent_rel_l2"] = result["video_sample_rel_l2"][-1]
    result["final_audio_latent_rel_l2"] = result["audio_sample_rel_l2"][-1]
    result["seconds"] = {
        "load": loaded - started,
        "steps": step_seconds,
    }
    result["peak_device_bytes"] = torch.cuda.max_memory_allocated(device)

    mode = "teacher" if args.teacher else "free"
    directory = (
        args.output
        / "parity"
        / "omniref"
        / args.run.parent.name
        / args.run.name
        / mode
    )
    directory.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: torch.stack(values) for name, values in recorded.items()},
        str(directory / "trajectory.safetensors"),
    )
    (directory / "metrics.json").write_text(
        json.dumps(
            {
                key: value
                if not isinstance(value, float) or math.isfinite(value)
                else str(value)
                for key, value in result.items()
            },
            indent=2,
        )
        + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", action="store_true")
    args = parser.parse_args()
    result = run(args)
    for key, value in result.items():
        if isinstance(value, list):
            print(key, [f"{item:.4g}" for item in value])
        else:
            print(key, value)


if __name__ == "__main__":
    main()
