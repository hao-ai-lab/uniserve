"""Measure the per-step kernel deviations of a diffusers MiniMax-H3 run.

The per-step level of the trajectory acceptance protocol compares each
implementation's prediction on the canonical reference's own inputs. This
script supplies the valid implementations' side of that comparison: it
loads the denoiser of a recorded canonical run (``diffusers_reference.py``
with SDPA cuDNN), rebuilds the inputs of the requested denoising steps from
the run's recorded tensors (step 0 from the noise, step ``i`` from the
sample after step ``i - 1``), and evaluates the DiT once per step on each
exact SDPA kernel: cuDNN, flash and memory-efficient.

For every step it reports the relative L2 of each kernel's prediction of the
generated rows against the recorded cuDNN prediction. The cuDNN entry
reproduces the recording and must be zero; the flash and efficient entries
are the valid kernels' teacher-forced deviations.

Writes ``<output>/<workload>/seed<seed>/kernel_steps.json``::

    {"run": ..., "steps": [0, 4, ...],
     "video": {"cudnn": [...], "flash": [...], "efficient": [...]},
     "audio": {...}}

Run from the repository root with the diffusers prerequisites on
``PYTHONPATH`` (see ``diffusers_reference.py``), e.g.
``CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<overlay> .venv/bin/python
tools/minimax_h3/kernel_floor.py --run
artifacts/minimax_h3/reference/diffusers/t2va_16x9_5s/seed42 --output
artifacts/minimax_h3/acceptance``.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel

CHECKPOINT = Path("/workspace/models/MiniMax-H3")

# The exact SDPA kernels of the valid implementations; cuDNN is canonical.
KERNELS = {
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
}

# Every fourth of the base schedule's 49 denoising steps.
DEFAULT_STEPS = tuple(range(0, 49, 4))


def rel_l2(value: torch.Tensor, reference: torch.Tensor) -> float:
    """Return ``||value - reference|| / ||reference||`` in float64."""
    value, reference = value.double().cpu(), reference.double().cpu()
    return float((value - reference).norm() / reference.norm())


def load_denoiser(task: str, checkpoint: Path, device: torch.device):
    """Load the run's DiT partition at its stored precision."""
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3RotaryPosEmbed,
        MiniMaxH3Transformer3DModel,
    )

    folder = checkpoint / (
        "transformer_ref" if task == "ref2va" else "transformer"
    )
    settings = {
        key: value
        for key, value in json.loads(
            (folder / "config.json").read_text()
        ).items()
        if not key.startswith("_")
    }
    with torch.device("meta"):
        model = MiniMaxH3Transformer3DModel(**settings)
    state = {}
    for shard in sorted(folder.glob("*.safetensors")):
        state.update(load_file(shard, device=str(device)))
    model.load_state_dict(state, assign=True, strict=True)
    # The rotary table is a non-persistent buffer: build it on the device.
    model.rope = MiniMaxH3RotaryPosEmbed(
        settings["rope_freq_dim"], settings["rope_theta"]
    ).to(device)
    return model


@torch.inference_mode()
def measure(run: Path, steps: tuple[int, ...], checkpoint: Path) -> dict:
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3SetTimestepsStep,
        patchify_video_latents,
    )

    device = torch.device("cuda", 0)
    metadata = json.loads((run / "metadata.json").read_text())
    if metadata["attention"]["dit"] != "_native_cudnn":
        raise SystemExit(f"{run} is not a canonical cuDNN run")
    geometry = metadata["geometry"]
    model = load_denoiser(metadata["task"], checkpoint, device)

    text = load_file(run / "text.safetensors")
    conditions_path = run / "conditions.safetensors"
    conditions = load_file(conditions_path) if conditions_path.exists() else {}
    noise = load_file(run / "noise.safetensors")
    layout = load_file(run / "layout.safetensors")
    trajectory = load_file(run / "trajectory.safetensors")

    # The packed sequence holds the anchored visual condition rows before the
    # generated video rows and the audio condition rows (in request order)
    # before the generated audio rows; predictions are compared on the
    # generated rows only.
    visual_rows = geometry.get("num_condition_video_rows", 0)
    audio_rows = geometry.get("num_condition_audio_rows", 0)
    audio_conditions = [
        conditions[f"audio_condition.{index}"]
        for index in range(
            sum(1 for name in conditions if name.startswith("audio_condition."))
        )
    ]

    result = {
        "run": str(run),
        "workload": metadata["workload"],
        "seed": metadata["seed"],
        "steps": list(steps),
        "video": {name: [] for name in KERNELS},
        "audio": {name: [] for name in KERNELS},
    }
    for step in steps:
        if step == 0:
            video = patchify_video_latents(noise["video_noise"], (1, 2, 2))
            audio = noise["audio_noise"]
        else:
            video = trajectory["video_samples"][step - 1]
            audio = trajectory["audio_samples"][step - 1]
        hidden = (
            torch.cat((conditions["condition_rows"], video))
            if visual_rows
            else video
        )
        audio_hidden = (
            torch.cat((*audio_conditions, audio)) if audio_rows else audio
        )
        timestep, indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
            layout["video_indices"],
            layout["audio_indices"],
            visual_rows,
            audio_rows,
            layout["text_indices"].numel(),
            float(layout["timesteps"][step]),
            float(layout["audio_timesteps"][step]),
            # Condition rows sit at the anchor level of the pipeline.
            max(float(layout["timesteps"][step]), 0.999),
            1.0,
        )
        recorded = (
            trajectory["video_predictions"][step],
            trajectory["audio_predictions"][step],
        )
        line = [f"step {step}"]
        for name, backend in KERNELS.items():
            with sdpa_kernel(backend):
                video_out, audio_out = model(
                    hidden_states=hidden[None].to(device),
                    audio_hidden_states=audio_hidden[None].to(device),
                    encoder_hidden_states=text["qwen_hidden_states_50"].to(
                        device
                    ),
                    timestep=timestep.to(device),
                    timestep_indices=indices.to(device),
                    token_tags=layout["token_tags"].to(device),
                    position_ids=layout["position_ids"].to(device),
                    video_indices=layout["video_indices"].to(device),
                    audio_indices=layout["audio_indices"].to(device),
                    text_indices=layout["text_indices"].to(device),
                    return_dict=False,
                )
            video_deviation = rel_l2(
                video_out[0, visual_rows:].float(), recorded[0]
            )
            audio_deviation = rel_l2(
                audio_out[0, audio_rows:].float(), recorded[1]
            )
            result["video"][name].append(video_deviation)
            result["audio"][name].append(audio_deviation)
            line.append(
                f"{name} video {video_deviation:.5f} "
                f"audio {audio_deviation:.5f}"
            )
        print(" | ".join(line), flush=True)

    # The canonical kernel reproduces its own recording on these inputs.
    for modality in ("video", "audio"):
        if any(result[modality]["cudnn"]):
            raise SystemExit(
                f"cuDNN does not reproduce the recorded {modality} "
                f"predictions: {result[modality]['cudnn']}"
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument(
        "--steps", type=int, nargs="+", default=list(DEFAULT_STEPS)
    )
    args = parser.parse_args()

    result = measure(args.run, tuple(args.steps), args.checkpoint)
    directory = args.output / result["workload"] / f"seed{result['seed']}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kernel_steps.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(f"wrote {directory / 'kernel_steps.json'}", flush=True)


if __name__ == "__main__":
    main()
